"""Compact patch strips OR atomic five-view area surveys, without entry photos.

Input is frozen image-conditioned risk geometry. No GS/GT inputs. A complete
bundle has nadir and four oblique sweeps at one height above its target plane.
Travel between sweeps is not a photography task and is not certified safe.
"""
from collections import Counter,defaultdict
import numpy as np
from openflyscan.planning.region_coverage_geometry import coverage_action,coverage_measure,scan_reference_center
from openflyscan.planning.recapture_overlap_connections import motion_seconds
from openflyscan.planning.reliable_context_preference import build_reliable_context,score_context_action


def old_link_preference(action):
    link=action.get('old_view_link')
    if not link or link['height_difference_m']>5 or link['direction_delta_deg']>35:return 0.
    return float(link['old_supported_overlap'])


def candidate_heights(area,cfg,ceiling):
    center=scan_reference_center(area)
    heights=[max(cfg.min_range_m,area['reference_range']*s) for s in (.65,.85,1.,1.2)]
    if cfg.height_sampling_policy=='stable_grid':
        if cfg.height_step_m<=0 or cfg.height_search_max_m<cfg.min_range_m:
            raise ValueError('Invalid candidate height grid')
        heights.extend(np.arange(cfg.min_range_m,cfg.height_search_max_m+1e-6,cfg.height_step_m))
        heights=[h for h in heights if h<=cfg.height_search_max_m+1e-6]
    elif cfg.height_sampling_policy=='legacy':
        if ceiling is not None:
            room=ceiling-center[2]-.1;heights.extend([room*.8,room])
    else:raise ValueError('Unknown height sampling policy')
    return sorted(set(round(h,3) for h in heights if h>=cfg.min_range_m and
        (ceiling is None or center[2]+h<=ceiling+1e-6)))


def generate_surveys(area,clouds,manifest,hfov,vfov,cfg,ceiling,tree,
                     sample_points,old_observations,action_score,long_axis_yaw,context_provider=None):
    pts,probes=sample_points(area,cfg)
    history=old_observations(area,pts,clouds,hfov,vfov,manifest)
    heading=long_axis_yaw(area['points']);center=scan_reference_center(area)
    provider=build_reliable_context if context_provider is None else context_provider
    anchors,context_audit=provider(area,clouds,manifest,cfg)
    # Use the same relative height for every direction, not the same slant range.
    # Search height using geometry only, including room below the current ceiling.
    heights=candidate_heights(area,cfg,ceiling)
    rejected=Counter();results=[];bundles=0;local=area.get('task_type')=='local_patch' or cfg.budget_atomic_strips
    if not heights:rejected['no_height_above_target_within_ceiling']+=1
    history_angles=[]
    for view in sorted(history,key=lambda h:-float(h['quality'].sum())):
        if view['quality'].sum()<=0 or view['pitch']>=-30:continue
        yaw,pitch=float(view['yaw']),float(view['pitch'])
        if any(abs((yaw-y+180)%360-180)<10 and abs(pitch-p)<5 for y,p,_ in history_angles):continue
        history_angles.append((yaw,pitch,view['stem']))
        if len(history_angles)==3:break
    stages=[(cfg.nominal_oblique_pitch_deg,'nominal'),(-55.,'geometry_fallback'),(-70.,'geometry_fallback')]
    if cfg.height_sampling_policy=='stable_grid':stages=stages[:1]
    attempts=[]
    for pitch,stage in stages:
        # A fallback is ONLY tried if no complete nominal geometry task exists
        # at ANY height. Higher heuristic score cannot displace valid -45 scans.
        if stage=='geometry_fallback' and results:break
        before=dict(rejected);start=len(results)
        for hi,height in enumerate(heights):
            height_id=f'z{height:.3f}' if cfg.height_sampling_policy=='stable_grid' else f'h{hi}'
            bundle=[];key=f"{area['id']}_{height_id}_{abs(int(pitch))}"
            directions=[(d,heading if d==0 else (heading+(d-1)*90)%360,
                         -90. if d==0 else pitch,'nominal_grid',None) for d in range(5)]
            if local and stage=='nominal':
                for yaw,hp,stem in history_angles:
                    bucket=0 if abs(hp+90)<5 else 1+int(np.round(((yaw-heading)%360)/90))%4
                    directions.append((bucket,yaw,hp,'old_orientation',stem))
            for di,(direction,yaw,actual_pitch,origin,stem) in enumerate(directions):
                distance=height/np.sin(np.radians(-actual_pitch))
                mode='roof_nadir' if actual_pitch==-90 else 'oblique_scan'
                action,error=coverage_action(area,yaw,actual_pitch,distance,hfov,vfov,cfg,ceiling,tree,mode)
                if error:
                    rejected[error]+=1
                    if local:continue
                    break
                measure,_,_=coverage_measure(probes,action['waypoints'],hfov,vfov)
                if measure['one_fraction']<.99 or measure['three_fraction']<.95 or measure['three_with_baseline_fraction']<.9:
                    rejected['incomplete_3d_envelope_coverage']+=1
                    if local:continue
                    break
                action.update(id=f'{key}_d{di}',survey_id=key,survey_direction=direction,
                    height_above_target_m=height,coverage=measure,task_type=area.get('task_type','area_survey'),
                    candidate_origin=origin,orientation_source_stem=stem,angle_stage=stage)
                if local:
                    action['survey_id']=action['id'];action['required_directions']=[direction]
                else:action['required_directions']=list(range(5))
                bundle.append(action)
            if not local and len(bundle)!=5:continue
            for action in bundle:
                combined,prior=action_score(action,area,pts,history,hfov,vfov,cfg,tree)
                action['_utility']=combined;action['_prior']=prior
                score_context_action(action,anchors,hfov,vfov,cfg)
            results.extend(bundle);bundles+=len(bundle) if local else 1
        attempts.append(dict(pitch_deg=pitch,stage=stage,accepted_actions=len(results)-start,
            rejections={k:rejected[k]-before.get(k,0) for k in rejected if rejected[k]>before.get(k,0)}))
    return results,dict(area_id=area['id'],ranks=area['ranks'],candidate_count=len(results),
        complete_task_candidates=bundles,task_type=area.get('task_type','area_survey'),
        rejections=dict(rejected),historical_views=len(history),
        angle_attempts=attempts,context=context_audit,history_orientation_count=len(history_angles),
        tested_heights_above_target_m=heights),pts,probes


def choose_surveys(areas,candidates,cfg):
    groups=defaultdict(list)
    for c in candidates:groups[c['survey_id']].append(c)
    chosen=[];trace=[];total=0
    for area in sorted(areas,key=lambda a:-a['risk']):
        options=[]
        for key,bundle in groups.items():
            expected=set(bundle[0].get('required_directions',range(5)))
            if bundle[0]['area_id']!=area['id'] or {c['survey_direction'] for c in bundle}!=expected:continue
            if cfg.selection_policy=='geometry' and any(c.get('candidate_origin')=='old_orientation' for c in bundle):continue
            count=sum(c['photo_count'] for c in bundle)
            if cfg.photo_budget and total+count>cfg.photo_budget:continue
            utility=np.maximum.reduce([c['_utility'] for c in bundle])
            gain=float(np.maximum(utility-bundle[0]['_prior'],0).mean())
            # Resolution/obliquity proxies counter the tendency to fly arbitrarily
            # high for fewer shots. This score is NOT learned PSNR improvement.
            resolution=min(1.,area['reference_range']/np.mean([c['range_m'] for c in bundle]))**2
            obliques=[c for c in bundle if abs(c['pitch_deg'])<89.]
            obliquity=float(np.mean([np.cos(np.radians(abs(c['pitch_deg']))) for c in obliques])) if obliques else 0.
            old=max(c['same_height_old_overlap_preference'] for c in bundle) if cfg.use_context_preference else 0.
            cost=count*cfg.interval_s+sum(c['length_m'] for c in bundle)/cfg.speed_mps
            # Compact patches: choose one complete short strip by existing
            # observation support, not an automatic five-direction package.
            angular_bonus=1. if area.get('task_type')=='local_patch' else .5+obliquity
            score=(.25+gain)*resolution*angular_bonus*(1+.1*old)/max(cost,1.)
            if cfg.selection_policy=='geometry':score=.25*resolution*angular_bonus/max(cost,1.)
            options.append((score,-count,key,bundle,gain))
        if not options:
            trace.append(dict(area=area['id'],status='no_complete_task_within_constraints'))
            continue
        score,negative_count,key,bundle,gain=max(options,key=lambda v:v[:3])
        chosen.extend(bundle);total-=negative_count
        trace.append(dict(area=area['id'],survey=key,photos=-negative_count,photos_so_far=total,
            incremental_pair_proxy=gain,score=score,entry_photos=0,task_type=area.get('task_type','area_survey'),
            status='complete_local_strip' if area.get('task_type')=='local_patch' else 'complete_five_direction_survey'))
        print('Survey',area['ranks'],'photos',-negative_count,'height above target',bundle[0]['height_above_target_m'],'total',total,flush=True)
    return chosen,trace


def connect_surveys(actions,manifest_index,tree,cfg):
    """Order surveys then their sweeps. Transit gets zero additional photos."""
    bundles=defaultdict(list)
    for action in actions:bundles[action['survey_id']].append(action)
    current=np.array(list(manifest_index.values())[-1]['enu']);ordered=[]
    while bundles:
        key=min(bundles,key=lambda k:min(np.linalg.norm(current-c['waypoints'][0]['position_enu_m']) for c in bundles[k]))
        remaining=bundles.pop(key)
        while remaining:
            idx=min(range(len(remaining)),key=lambda i:np.linalg.norm(current-remaining[i]['waypoints'][0]['position_enu_m']))
            c=dict(remaining.pop(idx));c['entry_bridge']=[]
            if c.get('_context_preview') is not None:c['context_preview']=c['_context_preview']
            c['connection_status']='independent_scan_no_entry_captures'
            c['image_matching_verified']=False
            ordered.append(c);current=np.array(c['waypoints'][-1]['position_enu_m'])
    trajectory=[];warnings=[];current=np.array(list(manifest_index.values())[-1]['enu']);t=0.
    for c in ordered:
        for wi,w in enumerate(c['waypoints']):
            pos=np.array(w['position_enu_m']);distance=np.linalg.norm(pos-current)
            samples=np.linspace(current,pos,max(2,int(np.ceil(distance/2))+1))
            clearance=float(tree.query(samples)[0].min())
            if clearance<cfg.clearance_m:warnings.append(dict(action=c['id'],photo_index=wi,clearance_proxy_m=clearance))
            duration=max(cfg.interval_s,motion_seconds(current,pos,cfg))
            for j,q in enumerate(samples[1:-1],1):
                trajectory.append(dict(position_enu_m=q.tolist(),capture=False,time_lower_bound_s=float(t+j/(len(samples)-1)*duration)))
            t+=duration;trajectory.append(dict(**w,time_lower_bound_s=float(t)));current=pos
    return ordered,trajectory,warnings
