"""Frozen Head/Pi3X whole-row cover and continuous preview. No GS selection input."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from openflyscan.planning.directional_surface_cover import whole_rows,select_rows
from openflyscan.planning.continuous_capture_path import order_rows,build_trajectory
from openflyscan.planning.recapture_observation_tokens import candidate_utilities
from openflyscan.planning.recapture_budget_core import BudgetPolicy
from openflyscan.planning.region_coverage_geometry import CoverageSettings,extract_region,json_dump,scan_reference_center
from openflyscan.planning.plan_region_coverage_recapture import long_axis_yaw
from openflyscan.planning.risk_extent_tasks import build_risk_tasks
from openflyscan.planning.reliable_context_preference import build_reliable_context,context_pair


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-run',type=Path,required=True)
    p.add_argument('--full-plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-registered-control',action='store_true',help='Explicit simulation pose-control experiment; not sensor-only deployment')
    p.add_argument('--utility-cache',type=Path,help='Reuse same ordered-row utilities; validated by candidate IDs and frozen tokens')
    a=p.parse_args();out=a.output
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True);start=time.time()
    protocol=json.loads((a.evidence_run/'input_protocol.json').read_text());scene=protocol['scene_config']
    controlled=protocol.get('experiment_condition')=='registered_pose_control'
    if controlled and not a.allow_registered_control:raise ValueError('Registered control requires explicit acknowledgement')
    if protocol.get('experiment_condition') not in [None,'sensor_only','registered_pose_control']:raise ValueError('Unknown input condition')
    cfg=CoverageSettings(**protocol['geometry']);policy=BudgetPolicy.from_dict(protocol['shared_policy'])
    manifest=json.loads(Path(scene['manifest']).read_text());index={r['stem']:r for r in manifest['records']}
    if not controlled and manifest.get('GS_or_SfM_inputs') is not False:raise ValueError('Sensor-only manifest declaration required')
    if controlled and manifest.get('experiment_condition')!='registered_pose_control':raise ValueError('Manifest/protocol condition mismatch')
    root=Path(scene['surface_root']);obs=json.loads((root/'observations/observations.json').read_text())['regions'][:cfg.target_budget]
    hfov,vfov=np.radians([obs[0]['horizontal_fov_deg'],obs[0]['vertical_fov_deg']])
    cloud=dict(np.load(scene['cloud']));tree=cKDTree(cloud['xyz'])
    frozen=json.loads(Path(scene['selection']).read_text())['selected'][:cfg.target_budget]
    clouds={d['chunk']:dict(np.load(root/f"geometry/chunk_{d['chunk']:04d}.npz")) for d in obs}
    regions=[]
    for rank,(s,d) in enumerate(zip(frozen,obs),1):
        if d['rank']!=rank or s['chunk']!=d['chunk']:raise ValueError('Frozen rank/chunk mismatch')
        np.testing.assert_allclose(d['frozen_xyz'],s['xyz'])
        regions.append(extract_region(rank,s,d,dict(np.load(root/f'observations/rank_{rank:02d}_surface.npz')),clouds[d['chunk']],cfg))
    areas,_=build_risk_tasks(regions,cfg,hfov,vfov,long_axis_yaw)
    old=json.loads((a.full_plan/'route_plan.json').read_text())
    raw=json.loads((a.full_plan/'candidate_actions.json').read_text());rows=whole_rows(raw)
    cache=dict(np.load(a.evidence_run/'selection_cache.npz'));tokens={k[6:]:v for k,v in cache.items() if k.startswith('token_')}
    # Never combine an old candidate pool with a different frozen region geometry.
    amap={x['id']:x for x in areas}
    for r in rows:
        if r['area_id'] not in amap or r['ranks']!=amap[r['area_id']]['ranks']:raise ValueError('Candidate area mismatch')
        np.testing.assert_allclose(r['center'],scan_reference_center(amap[r['area_id']]),atol=1e-4)
    print('Whole-row candidates',len(rows),'tokens',len(tokens['xyz']),flush=True)
    pieces=[];tic=time.time();row_ids=np.array([r['id'] for r in rows])
    if a.utility_cache:
        reuse=np.load(a.utility_cache)
        cached_rows=json.loads((a.utility_cache.parent/'candidates.json').read_text())
        np.testing.assert_array_equal(row_ids,[r['id'] for r in cached_rows])
        for new,prior in zip(rows,cached_rows):
            np.testing.assert_array_equal([w['position_enu_m']+[w['aircraft_yaw_deg'],w['gimbal_pitch_deg']] for w in new['waypoints']],
                [w['position_enu_m']+[w['aircraft_yaw_deg'],w['gimbal_pitch_deg']] for w in prior['waypoints']])
        prior_protocol=json.loads((a.utility_cache.parent/'input_protocol.json').read_text())['source_protocol']
        for name in ['geometry','shared_policy','scene_config']:
            if prior_protocol[name]!=protocol[name]:raise ValueError('Utility cache input configuration mismatch')
        for key,value in tokens.items():np.testing.assert_array_equal(value,reuse[f'token_{key}'])
        utility=reuse['repair']
    else:
        for offset in range(0,len(rows),50):
            util,_=candidate_utilities(rows[offset:offset+50],tokens,hfov,vfov,cfg,policy,tree)
            pieces.append(util['repair']);print('Scored',min(offset+50,len(rows)),flush=True)
        utility=np.concatenate(pieces)
    np.savez_compressed(out/'row_cache.npz',repair=utility,row_ids=row_ids,**{f'token_{k}':v for k,v in tokens.items()})
    utility_seconds=time.time()-tic
    chosen,reports=select_rows(rows,utility,tokens)
    selected=[rows[i] for i in chosen];allrisk=np.vstack([r['raw_points'] for r in regions])
    # Endpoint-only context audit: an interior photo must not masquerade as entry overlap.
    for area in areas:
        active=[r for r in selected if r['area_id']==area['id']]
        if not active:continue
        area['_all_risk_points']=allrisk
        anchors,meta=build_reliable_context(area,clouds,index,cfg)
        for r in active:
            r['endpoint_context_links']={};r['endpoint_context_preview']={}
            for j in [0,-1]:
                pairs=[context_pair(anchor,r['waypoints'][j],hfov,vfov,cfg) for anchor in anchors]
                pairs=[x for x in pairs if x is not None and x[0]['score']>0]
                if pairs:
                    report,preview=max(pairs,key=lambda x:(x[0]['score'],x[0]['common_points']))
                    r['endpoint_context_links'][str(j)]=report;r['endpoint_context_preview'][str(j)]=preview
        print('Context checked',area['ranks'],len(active),'rows',flush=True)
    last=manifest['records'][-1];startpos=np.asarray(last['enu']);yaw,pitch=last['camera_ypr_deg'][:2]
    limits=dict(speed_mps=cfg.speed_mps,acceleration_mps2=1.5,climb_mps=cfg.climb_speed_mps,
                descent_mps=cfg.descent_speed_mps,yaw_rate_dps=15.,pitch_rate_dps=10.,interval_s=cfg.interval_s)
    ordered=order_rows(selected,startpos,yaw,pitch,limits)
    trajectory=build_trajectory(ordered,startpos,yaw,pitch,limits,tree,cfg.clearance_m)
    control_rows=order_rows(whole_rows(old['actions']),startpos,yaw,pitch,limits)
    control_trajectory=build_trajectory(control_rows,startpos,yaw,pitch,limits)
    for report in reports:
        rr=[r for r in ordered if r['area_id']==report['area_id']]
        report['directions']=sorted(set(r['survey_direction'] for r in rr))
        report['entry_context_linked_rows']=sum(bool(r.get('entry_context_link')) for r in rr)
        report['new_height_enu_z']=sorted(set(round(r['waypoints'][0]['position_enu_m'][2],3) for r in rr))
    times=np.array([e['t_s'] for e in trajectory['capture_events']])
    assert len(times)==sum(r['photo_count'] for r in ordered)
    if len(times)>1:assert np.diff(times).min()>=cfg.interval_s-1e-6
    summary=dict(scene=scene['scene'],old_photos=old['photo_count'],new_photos=len(times),selected_rows=len(ordered),
        duration_min=trajectory['duration_s']/60,old_duration_same_motion_profile_min=control_trajectory['duration_s']/60,
        utility_seconds=utility_seconds,total_seconds=time.time()-start,
        limits=limits,minimum_photo_interval_s=float(np.diff(times).min()) if len(times)>1 else None,
        entry_context_linked_rows=sum(bool(r.get('entry_context_link')) for r in ordered),
        connector_clearance_warnings=sum(s['clearance_warning'] for s in trajectory['segments'] if s['type']=='noncapture_connector'),
        capture_clearance_warnings=sum(s['clearance_warning'] for s in trajectory['segments'] if s['type']=='capture_row'),
        area_reports=reports,selection_used_GS_render_or_quality=False,registered_pose_prior_used=controlled,
        experiment_condition='registered_pose_control' if controlled else 'sensor_only',flight_authorized=False,hard_photo_budget=None,
        unresolved_notes=['95% of attainable model evidence at 80% of best candidate proxy is NOT 95% true surface coverage.',
          'Missing/uncertain evidence does not prove unobserved directions are good; no whole-building completion claim.',
          'Only observed-point clearance audited; not an obstacle avoidance controller.',
          'Historical altitude reference is inconsistent; mission start/return and legal flight envelope need confirmation.'])
    json_dump(out/'summary.json',summary);json_dump(out/'trajectory.json',trajectory)
    json_dump(out/'route_plan.json',dict(actions=ordered,photo_count=len(times),area_reports=reports,flight_authorized=False,
        altitude_reference=protocol['altitude'],start=dict(stem=last['stem'],position_enu_m=startpos.tolist(),yaw=yaw,pitch=pitch),
        start_assumption='Continue after last recorded camera; no takeoff/return legs included',limits=limits))
    json_dump(out/'input_protocol.json',dict(source_protocol=protocol,full_plan=str(a.full_plan),quality_ratio=.8,
        required_fraction=.95,coverage_floor=1e-4,selection_inputs=['frozen Head risk','Pi3X geometry/confidence/texture','sensor manifest'],
        GS_render_or_quality_used=False,registered_pose_prior_used=controlled,per_area_fixed_height=True,bridge_photos=0))
    json_dump(out/'candidates.json',rows)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['area_reports','unresolved_notes']},indent=2),flush=True)


if __name__=='__main__':main()
