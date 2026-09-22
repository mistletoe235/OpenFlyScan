"""Budget-only development trial on frozen inference. Never opens GS/GT.

Config separates scene facts from shared selection policy. Same candidate pool
for internal controls. Output scores describe objectives, not PSNR improvement.
"""
import argparse,json,time,sys,shutil
from dataclasses import asdict,fields
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from openflyscan.planning.area_survey_policy import generate_surveys,connect_surveys,choose_surveys
from openflyscan.planning.region_coverage_geometry import CoverageSettings,altitude_reference,extract_region,scan_reference_center,json_dump
from openflyscan.planning.plan_region_coverage_recapture import sample_points,old_observations,long_axis_yaw,serialize_area,action_score
from openflyscan.planning.risk_extent_tasks import build_risk_tasks
from openflyscan.planning.recapture_budget_core import BudgetPolicy,split_capture_indices,greedy_budget
from openflyscan.planning.recapture_observation_tokens import make_tokens,candidate_utilities
from openflyscan.planning.recapture_overlap_connections import motion_seconds
from openflyscan.planning.reliable_context_preference import build_reliable_context,score_context_action


def resolve_config(path):
    config=json.loads(path.read_text());allowed={'scene','selection','surface_root','manifest','cloud','output','shared_policy','mission','geometry'}
    if set(config)-allowed:raise ValueError(f'Unknown config keys {set(config)-allowed}')
    for key in ['selection','surface_root','manifest','cloud','output','shared_policy']:
        value=Path(config[key]);config[key]=value if value.is_absolute() else (path.parent/value).resolve()
    return config


def validate_geometry(cfg):
    numeric=[v for v in asdict(cfg).values() if type(v) in [int,float]]
    if not np.isfinite(numeric).all():raise ValueError('Nonfinite geometry setting')
    if min(cfg.speed_mps,cfg.interval_s,cfg.min_range_m,cfg.height_step_m,cfg.climb_speed_mps,cfg.descent_speed_mps)<=0:
        raise ValueError('Speeds, range and time/height steps must be positive')
    if not (0<=cfg.forward_overlap<1 and 0<=cfg.side_overlap<1):raise ValueError('Overlap must be in [0,1)')
    if not 0<cfg.min_pair_angle_deg<=cfg.useful_pair_angle_deg<=cfg.max_pair_angle_deg<180:raise ValueError('Invalid triangulation angles')
    if cfg.height_search_max_m<cfg.min_range_m or cfg.clearance_m<0:raise ValueError('Invalid height/clearance range')


def atomic_candidates(candidates,policy,anchors=None,hfov=None,vfov=None,cfg=None):
    result=[];posekeys=set()
    for action in candidates:
        for row in action['rows']:
            sequence=row['waypoints']
            for si,ids in enumerate(split_capture_indices(len(sequence),policy.max_strip_photos,policy.strip_shared_photos)):
                wps=[sequence[i] for i in ids];pos=np.array([w['position_enu_m'] for w in wps])
                key=tuple(np.round(np.array([np.r_[w['position_enu_m'],w['aircraft_yaw_deg'],w['gimbal_pitch_deg']] for w in wps]).ravel(),5))
                if key in posekeys:continue
                posekeys.add(key)
                c={k:v for k,v in action.items() if not k.startswith('_')};c.update(id=f"{action['id']}_row{row['row']}_part{si}",
                    source_action=action['id'],source_task_type=action['task_type'],source_row=row['row'],
                    survey_id=f"{action['id']}_row{row['row']}_part{si}",task_type='partial_continuous_strip',
                    waypoints=wps,rows=[dict(row=0,waypoints=wps)],row_count=1,photo_count=len(wps),
                    length_m=float(np.linalg.norm(np.diff(pos,axis=0),axis=1).sum()),
                    coverage_scope='Partial native support; not a full area/five-direction completion',
                    entry_bridge=[])
                # Whole-row coverage and best old-photo link do not transfer to
                # a cropped strip. Preserve them only as source diagnostics.
                for name in ['coverage','old_view_link','incremental_pair_support','prior_pair_support','expected_pair_support']:
                    if name in c:c[f'source_{name}']=c.pop(name)
                c['reliable_context_link']=None;c['same_height_old_overlap_preference']=0.
                c.pop('context_preview',None)
                if anchors is not None:
                    score_context_action(c,anchors,hfov,vfov,cfg)
                    c.pop('_context_preview',None)
                c['context_scope']='recomputed_for_partial_strip' if anchors is not None else 'unavailable_no_inherited_credit'
                result.append(c)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--export-whole-surveys',action='store_true',help='Export intact local/area surveys for continuous whole-row selection')
    p.add_argument('--target-budget',type=int,help='Explicit equal-region experiment override, saved in protocol')
    p.add_argument('--output',type=Path,help='New run directory; never overwrite earlier trials');a=p.parse_args()
    config=resolve_config(a.config.resolve())
    if a.output is not None:config['output']=a.output.resolve()
    out=config['output'];out.mkdir(parents=True,exist_ok=False);start=time.time()
    snapshot=out/'code_snapshot';snapshot.mkdir()
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename and Path(filename).resolve().parent==Path(__file__).resolve().parent:
            shutil.copy2(filename,snapshot/Path(filename).name)
    shared=json.loads(config['shared_policy'].read_text());policy=BudgetPolicy.from_dict(shared)
    geo=dict(config.get('geometry',{}))
    if a.target_budget is not None:
        if a.target_budget<1:raise ValueError('Positive target budget required')
        geo['target_budget']=a.target_budget;config['geometry']=geo
    unknown=set(geo)-{f.name for f in fields(CoverageSettings)}
    if unknown:raise ValueError(f'Unknown geometry parameters {unknown}')
    cfg=CoverageSettings(**geo)
    validate_geometry(cfg)
    if cfg.height_sampling_policy!='stable_grid' or not cfg.budget_atomic_strips:raise ValueError('New trial needs stable_grid and atomic strips')
    if a.export_whole_surveys:cfg.budget_atomic_strips=False
    mission=config['mission'];allowed={'ceiling_mode','ceiling_enu_z_m','max_relative_altitude_m','historical_reference_acknowledged','start_position_enu_m'}
    if set(mission)-allowed:raise ValueError('Unknown mission parameter')
    manifest=json.loads(config['manifest'].read_text());index={r['stem']:r for r in manifest['records']}
    controlled=manifest.get('experiment_condition')=='registered_pose_control'
    altitude=altitude_reference(manifest)
    if mission['ceiling_mode']=='explicit_enu':
        ceiling=float(mission['ceiling_enu_z_m']);datum=('explicit registered local-Z ceiling; ENU field names are compatibility aliases, not geographic ENU or AGL' if controlled else 'explicit ENU ceiling supplied by task config, not inferred AGL')
    elif mission['ceiling_mode']=='historical_reference_preview':
        if mission.get('historical_reference_acknowledged') is not True:raise ValueError('Historical altitude ambiguity must be explicit')
        ref=altitude.get('takeoff_enu_z')
        if ref is None:ref=altitude.get('conservative_reference_enu_z')
        if ref is None:raise ValueError('No sensor altitude datum; explicit ENU ceiling required')
        ceiling=ref+float(mission['max_relative_altitude_m']);datum='lowest sampled historical takeoff reference, NOT confirmed future takeoff/AGL'
    else:raise ValueError('Unknown altitude coordinate mode')
    if not np.isfinite(ceiling):raise ValueError('Mission ceiling must be finite, in ENU metres')
    initial=np.array(mission.get('start_position_enu_m',manifest['records'][-1]['enu']))
    if initial.shape!=(3,) or not np.isfinite(initial).all():raise ValueError('Initial position must be finite ENU xyz')
    altitude.update(applied_ceiling_enu_z=ceiling,ceiling_policy=datum)
    selection=json.loads(config['selection'].read_text());observation=json.loads((config['surface_root']/'observations/observations.json').read_text())
    selected=selection['selected'][:cfg.target_budget];diags=observation['regions'][:cfg.target_budget]
    if len(selected)!=cfg.target_budget:raise ValueError('Target budget exceeds frozen selection')
    clouds={d['chunk']:dict(np.load(config['surface_root']/f"geometry/chunk_{d['chunk']:04d}.npz")) for d in diags}
    hfov,vfov=np.radians([diags[0]['horizontal_fov_deg'],diags[0]['vertical_fov_deg']])
    regions=[]
    for rank,(s,d) in enumerate(zip(selected,diags),1):
        assert d['rank']==rank and d['chunk']==s['chunk'];np.testing.assert_allclose(d['frozen_xyz'],s['xyz'])
        surface=dict(np.load(config['surface_root']/f'observations/rank_{rank:02d}_surface.npz'))
        regions.append(extract_region(rank,s,d,surface,clouds[d['chunk']],cfg))
    areas,relations=build_risk_tasks(regions,cfg,hfov,vfov,long_axis_yaw);cloud=dict(np.load(config['cloud']));tree=cKDTree(cloud['xyz'])
    json_dump(out/'input_protocol.json',dict(scene=config['scene'],scene_config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()},
        shared_policy=asdict(policy),geometry=asdict(cfg),altitude=altitude,
        selection_method='Frozen top target-budget, unchanged risk scores; no new top-k after seeing GS',
        forbidden_inputs=['GS renders','PSNR','simulator truth','manual region labels']+([] if controlled else ['GS/SfM extrinsics']),
        experiment_condition='registered_pose_control' if controlled else 'sensor_only',
        registered_pose_prior_used=controlled,ordinary_GPS_deployment_claim=False,
        coordinate_frame=manifest.get('coordinate_frame','historical_sensor_frame'),
        development_scene=True,GS_evaluation_in_this_run=False,flight_authorized=False,
        warnings=['No semantic water classifier supplied.','Historical reference is not AGL.',
                  'No verified collision mesh/geofence; observed-cloud proximity is only a proxy.',
                  'Greedy objective is not an independent quality metric.']))
    json_dump(out/'regions_frozen.json',dict(regions=[serialize_area(x) for x in areas],relations=relations))
    candidates=[];generation=[];allrisk=np.vstack([r['raw_points'] for r in regions]);tic=time.time()
    # Old pair score runs only to retain the existing candidate pipeline/context;
    # source-conditioned and control objectives are computed on the same pool.
    for area in areas:
        area['_all_risk_points']=allrisk
        acts,diag,_,_=generate_surveys(area,clouds,index,hfov,vfov,cfg,ceiling,tree,
            sample_points,old_observations,action_score,long_axis_yaw)
        anchors,_=build_reliable_context(area,clouds,index,cfg)
        candidates.extend(acts if a.export_whole_surveys else atomic_candidates(acts,policy,anchors,hfov,vfov,cfg));generation.append(diag)
        print('Candidate task',area['ranks'],'full',len(acts),'pool_so_far',len(candidates),flush=True)
    generation_seconds=time.time()-tic
    if not candidates:raise RuntimeError('No feasible candidates; inspect mission geometry')
    tokens,evidence=make_tokens(regions,clouds,index,policy)
    evidence.update(used_GS_or_SfM=controlled,used_GS_render_or_quality=False,
                    experiment_condition='registered_pose_control' if controlled else 'sensor_only')
    json_dump(out/'observation_witnesses.json',evidence)
    if a.export_whole_surveys:
        chosen,trace=choose_surveys(areas,candidates,cfg)
        clean=lambda c:{k:v for k,v in c.items() if not k.startswith('_')}
        json_dump(out/'candidate_actions.json',[clean(c) for c in candidates])
        json_dump(out/'route_plan.json',dict(actions=[clean(c) for c in chosen],photo_count=sum(c['photo_count'] for c in chosen),selection_trace=trace))
        json_dump(out/'generation_audit.json',generation)
        np.savez_compressed(out/'selection_cache.npz',**{f'token_{k}':v for k,v in tokens.items()})
        print('Whole survey export complete',len(candidates),'candidates',len(tokens['xyz']),'tokens',flush=True)
        return
    json_dump(out/'candidate_actions.json',candidates);json_dump(out/'generation_audit.json',generation)
    tic=time.time();utilities,exploration=candidate_utilities(candidates,tokens,hfov,vfov,cfg,policy,tree);utility_seconds=time.time()-tic
    np.savez_compressed(out/'selection_cache.npz',**utilities,exploration=exploration,**{f'token_{k}':v for k,v in tokens.items()})
    results=[]
    for budget in policy.budgets:
        for method in ['geometry','source_pair','five_sector','repair']:
            weights=tokens['repair_weight'] if method=='repair' else tokens['risk_weight']
            ranks=tokens['rank']
            if method=='five_sector':weights=np.repeat(weights/5,5);ranks=np.repeat(ranks,5)
            ids,audit=greedy_budget(candidates,utilities[method],weights,budget,policy,initial,cfg.speed_mps,cfg.interval_s,ranks,exploration,method,
                uncertain_tokens=(tokens['readiness']<policy.exploration_support_threshold) if method=='repair' else None)
            chosen=[dict(candidates[i]) for i in ids]
            if not chosen:
                actions=[];trajectory=[];warnings=[]
            else:
                actions,trajectory,warnings=connect_surveys(chosen,{'start':{'enu':initial.tolist()}},tree,cfg)
            name=f'{method}_b{budget}';(out/name).mkdir();pos=[initial]+[np.array(w['position_enu_m']) for c in actions for w in c['waypoints']]
            pathlength=float(np.linalg.norm(np.diff(pos,axis=0),axis=1).sum()) if len(pos)>1 else 0
            # Final ordering may differ from greedy cost; report actual proxy time.
            return_seconds=motion_seconds(pos[-1],initial,cfg)
            final_time=(trajectory[-1]['time_lower_bound_s'] if trajectory else 0)+return_seconds
            stats=dict(method=method,photo_budget=budget,photos=audit['used_photos'],
                exploration_photos=audit['exploration_photos'],actions=len(actions),
                uncertain_photo_equivalent=audit['uncertain_photo_equivalent'],
                named_task_ranks=sorted({r for c in actions for r in c['ranks']}),
                path_m_excluding_return=pathlength,return_distance_m=float(np.linalg.norm(pos[-1]-initial)),
                duration_proxy_including_return_s=final_time,clearance_warnings=len(warnings),
                objective=audit['objective'],objective_is_not_quality_metric=True)
            json_dump(out/name/'route_plan.json',dict(scene=config['scene'],method=method,status='preview_only',flight_authorized=False,
                used_GS_or_SfM=controlled,used_GS_render_or_quality=False,
                experiment_condition='registered_pose_control' if controlled else 'sensor_only',
                coordinate_frame=manifest.get('coordinate_frame','historical_sensor_frame'),
                actions=actions,trajectory=trajectory,selection=audit,stats=stats,
                camera=dict(horizontal_fov_deg=float(np.degrees(hfov)),vertical_fov_deg=float(np.degrees(vfov))),
                altitude_reference=altitude,settings=asdict(cfg),transit_clearance_warnings=warnings,
                completion_claim='Partial continuous strips; no complete surface/five-way coverage claim'))
            results.append(stats);print(name,stats,flush=True)
    json_dump(out/'summary.json',dict(results=results,regions=len(regions),tasks=len(areas),candidate_count=len(candidates),witnesses=len(tokens['xyz']),
        generation_seconds=generation_seconds,utility_seconds=utility_seconds,total_seconds=time.time()-start,
        no_GS_used=not controlled,GS_quality_used=False,registered_pose_prior_used=controlled,quality_gain_measured=False,
        baseline_scope='Internal same-pool controls only. source_pair is not full old joint optimizer; five_sector is not original full-grid mission.',
        paper_baselines_pending=['On-the-fly released planner adapter','Mostegel triplet geometry adapter']))


if __name__=='__main__':main()
