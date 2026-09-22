"""Configurable frozen-risk local coverage planner. No GS, SfM or defect IDs.

Preview only: uncertain geometry cannot certify collision safety / registration.
"""
import argparse
import json
import math
import time
from collections import Counter
from dataclasses import asdict,replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree, ConvexHull

from openflyscan.planning.region_coverage_geometry import (CoverageSettings,altitude_reference,extract_region,merge_regions,
    coverage_action,coverage_measure,unit,sha,json_dump)
from openflyscan.planning.plan_frozen_region_scans import camera_basis,frustum_contains,capture_record,Config
from openflyscan.planning.recapture_overlap_connections import context_for_area,prepare_connected_action,motion_seconds
from openflyscan.planning.area_survey_policy import generate_surveys,choose_surveys,connect_surveys
from openflyscan.planning.risk_extent_tasks import build_risk_tasks


def long_axis_yaw(points):
    """Minimum-area rectangle long side, as Android suggestedRouteHeading.

    Yaw sets the camera-right axis to this long side, not camera facing along it.
    """
    xy=points[:,:2];xy=xy-np.median(xy,axis=0)
    if len(xy)<3 or np.linalg.matrix_rank(xy)<2:return 0.
    hull=xy[ConvexHull(xy).vertices];best=None
    for a,b in zip(hull,np.roll(hull,-1,axis=0)):
        u=unit(b-a);v=np.array([-u[1],u[0]]);q=xy@np.column_stack([u,v]);span=np.ptp(q,axis=0)
        axis=u if span[0]>=span[1] else v;score=float(np.prod(span))
        if best is None or score<best[0]:best=(score,axis)
    axis=best[1]
    return float(np.degrees(np.arctan2(-axis[1],axis[0]))%180)


def sample_points(area,cfg):
    p=area['points'];ids=np.linspace(0,len(p)-1,min(cfg.max_samples_per_region,len(p))).astype(int)
    pts=p[ids]
    # Evaluate localization offsets explicitly, not just line-center samples.
    pad=area['uncertainty_m']+area['cell_m']/2
    u=np.array([1.,0,0]);v=np.array([0.,1,0])
    if area['kind']=='facade':
        u=unit(np.cross(area['normal'],[0,0,1]));v=unit(np.cross(area['normal'],u))
    probes=np.vstack([pts]+[pts+pad*(du*u+dv*v) for du,dv in [(-1,-1),(-1,1),(1,-1),(1,1)]])
    return pts,probes


def q_and_direction(area,points,position,yaw,pitch,hfov,vfov):
    delta=position-points;dist=np.maximum(np.linalg.norm(delta,axis=1),1e-8);direction=delta/dist[:,None]
    incidence=np.maximum(direction@area['normal'],0) if area['trusted_normal'] else np.full(len(points),.5)
    if '_point_normals' in area:
        incidence=np.maximum(np.einsum('ij,ij->i',direction,area['_point_normals']),0)*area['_point_normal_valid']
    q=frustum_contains(points,position,yaw,pitch,hfov,vfov)*incidence*np.minimum(1.,(area['reference_range']/dist)**2)
    return q,direction


def old_observations(area,points,clouds,hfov,vfov,manifest_index):
    seen={}
    for chunk in sorted({r['diag']['chunk'] for r in area['members']}):
        cloud=clouds[chunk];h,w=cloud['target_hw']
        for vi,stem in enumerate(cloud['stems'].astype(str)):
            pose=cloud['camera_poses'][vi];K=cloud['intrinsics'][vi]
            pc=(points-pose[:3,3])@pose[:3,:3];uv=pc@K.T;uv=uv[:,:2]/np.maximum(uv[:,2,None],1e-9)
            inside=(pc[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            yi=np.abs(cloud['dense_y'][:,None]-uv[:,1]).argmin(0);xi=np.abs(cloud['dense_x'][:,None]-uv[:,0]).argmin(0)
            surface=cloud['dense_maps'][vi,yi,xi];raw=cloud['dense_conf'][vi,yi,xi]
            pct=np.searchsorted(np.sort(cloud['dense_conf'][vi].ravel()),raw)/cloud['dense_conf'][vi].size
            distance=np.linalg.norm(points-pose[:3,3],axis=1)
            consistent=inside & (np.linalg.norm(surface-points,axis=1)<=np.maximum(1.,.03*distance))
            rec=manifest_index[stem];yaw,pitch,_=rec['camera_ypr_deg']
            q,d=q_and_direction(area,points,pose[:3,3],yaw,pitch,hfov,vfov)
            q*=pct*consistent
            entry=dict(stem=stem,position=pose[:3,3].tolist(),sensor_position=rec['enu'],yaw=yaw,pitch=pitch,
                quality=q,direction=d,consistent_fraction=float(np.mean(consistent)),
                native_sensor_position_difference_m=float(np.linalg.norm(pose[:3,3]-rec['enu'])))
            if stem not in seen or q.sum()>seen[stem]['quality'].sum():seen[stem]=entry
    return list(seen.values())


def pair_utility(qualities,directions,cfg):
    """Continuous baseline, not directional bins. Best useful pair per sample.

    Extra identical photos cannot raise the score. This is still a geometric
    proxy, not a trained PSNR gain predictor. Three-view count checked separately.
    """
    q=np.array(qualities);d=np.array(directions)
    if len(q)==0:return np.zeros(0)
    value=np.zeros(q.shape[1])
    for i in range(len(q)-1):
        cosine=np.einsum('nk,jnk->jn',d[i],d[i+1:]);angle=np.degrees(np.arccos(np.clip(cosine,-1,1)))
        baseline=np.minimum(1.,np.sin(np.radians(angle))/np.sin(np.radians(cfg.useful_pair_angle_deg)))
        baseline*=((angle>=cfg.min_pair_angle_deg)&(angle<=cfg.max_pair_angle_deg))
        score=np.sqrt(q[i]*q[i+1:])*baseline
        value=np.maximum(value,score.max(0))
    return value


def action_score(action,area,points,history,hfov,vfov,cfg,tree):
    qs=[];ds=[];occluded=[]
    for wp in action['waypoints']:
        pos=np.array(wp['position_enu_m']);q,d=q_and_direction(area,points,pos,wp['aircraft_yaw_deg'],wp['gimbal_pitch_deg'],hfov,vfov)
        active=q>0
        # Same sampled observed-point ray screen as prior prototype, not exact visibility.
        if active.any():
            delta=points[active]-pos
            samples=pos+delta[:,None]*np.linspace(.1,.85,10)[None,:,None]
            close=tree.query(samples.reshape(-1,3))[0].reshape(active.sum(),-1).min(1)<.6
            q[active]*=np.where(close,.3,1.);occluded.append(float(close.mean()))
        qs.append(q);ds.append(d)
    oldq=[h['quality'] for h in history];oldd=[h['direction'] for h in history]
    # Retain exact per-photo geometry support for marginal cross-action pairs.
    # Taking max(action utilities) alone misses pairs formed ACROSS two strips.
    action['_new_q']=np.asarray(qs);action['_new_d']=np.asarray(ds)
    action['_old_q']=np.asarray(oldq).reshape(-1,len(points))
    action['_old_d']=np.asarray(oldd).reshape(-1,len(points),3)
    prior=pair_utility(oldq,oldd,cfg) if history else np.zeros(len(points))
    combined=pair_utility(oldq+qs,oldd+ds,cfg)
    value=float(np.maximum(combined-prior,0).mean())
    wp=action['waypoints'][0];pos=np.array(wp['position_enu_m']);center=np.array(action['center']);links=[]
    visible=frustum_contains(points,pos,wp['aircraft_yaw_deg'],wp['gimbal_pitch_deg'],hfov,vfov)
    for h in history:
        angle=float(np.degrees(np.arccos(np.clip(unit(np.array(h['position'])-center)@unit(pos-center),-1,1))))
        overlap=float(np.mean((h['quality']>.05)&visible))
        links.append(dict(stem=h['stem'],direction_delta_deg=angle,old_supported_overlap=overlap,
                          height_difference_m=float(abs(pos[2]-h['sensor_position'][2])),
                          sensor_position=h['sensor_position'],native_position=h['position'],
                          distance_m=float(np.linalg.norm(pos-np.array(h['sensor_position']))),
                          image_matching_verified=False))
    links.sort(key=lambda h:(-(h['old_supported_overlap'] if h['height_difference_m']<=5 and h['direction_delta_deg']<=35 else 0),h['distance_m']))
    action['old_view_link']=links[0] if links else None
    action['incremental_pair_support']=value;action['prior_pair_support']=float(prior.mean())
    action['expected_pair_support']=float(combined.mean());action['ray_near_point_fraction']=float(np.mean(occluded)) if occluded else 0.
    return combined,prior


def generate_actions(area,clouds,manifest_index,hfov,vfov,cfg,ceiling,tree):
    pts,probes=sample_points(area,cfg);history=old_observations(area,pts,clouds,hfov,vfov,manifest_index)
    heading=long_axis_yaw(area['points']);directions=[]
    if area['kind']!='facade':directions.append((heading,-90.,'roof_nadir'))
    if area['kind']=='facade':
        toward=-area['normal'];yaw=np.degrees(np.arctan2(toward[0],toward[1]))%360
        directions.extend([(yaw,-35.,'facade_scan'),(yaw,-50.,'facade_scan')])
    else:
        for yaw in heading+np.arange(4)*90:
            directions.extend([(float(yaw%360),-55.,'oblique_scan'),(float(yaw%360),-70.,'steep_oblique_scan')])
        # Secondary facade directions are hypotheses, not trusted side constraints.
        for member in area['members']:
            for mode in member['diag']['normal_modes'][:3]:
                n=np.array(mode['normal'])
                if mode['fraction']>=.08 and abs(n[2])<.6:
                    yaw=float(np.degrees(np.arctan2(-n[0],-n[1]))%360)
                    directions.append((yaw,-45.,'secondary_surface_hypothesis'))
    # Reuse well-supported previous image orientations as candidates. Coverage
    # strips need not turn through a fixed global cardinal direction first.
    historical_angles=[]
    for h in sorted(history,key=lambda h:-float(h['quality'].sum())):
        if h['pitch']>=-30:continue
        direction=camera_basis(h['yaw'],h['pitch'])[:,2]
        if any(direction@d>.98 for d in historical_angles):continue
        historical_angles.append(direction)
        directions.append((float(h['yaw']),float(h['pitch']),'history_aligned_scan'))
        if len(historical_angles)>=3:break
    results=[];rejected=Counter();seen=set()
    for yaw,pitch,mode in directions:
        for scale in [.8,1.,1.2]:
            distance=max(cfg.min_range_m,area['reference_range']*scale)
            key=tuple(np.round([yaw,pitch,distance],3))
            if key in seen:continue
            seen.add(key)
            action,reason=coverage_action(area,yaw,pitch,distance,hfov,vfov,cfg,ceiling,tree,mode)
            if reason:rejected[reason]+=1;continue
            measure,_,_=coverage_measure(probes,action['waypoints'],hfov,vfov)
            # Test on nonplanar original samples AND localization padding; no
            # assumption that a planar footprint alone proves actual coverage.
            if measure['one_fraction']<.99 or measure['three_fraction']<.95:
                rejected['incomplete_3d_envelope_coverage']+=1;continue
            if measure['three_with_baseline_fraction']<.9:
                rejected['insufficient_continuous_baseline']+=1;continue
            action['coverage']=measure;action['id']=f"{area['id']}_a{len(results):03d}"
            combined,prior=action_score(action,area,pts,history,hfov,vfov,cfg,tree)
            action['_utility']=combined;action['_prior']=prior;results.append(action)
    return results,dict(area_id=area['id'],ranks=area['ranks'],candidate_count=len(results),rejections=dict(rejected),
        historical_views=len(history),maximum_old_consistent_fraction=max((h['consistent_fraction'] for h in history),default=0.)),pts,probes


def choose_actions(areas,candidates,cfg,contexts,tree,hfov,vfov,ceiling):
    byarea={a['id']:a for a in areas};chosen=[];total=0;chosen_by={a['id']:[] for a in areas};trace=[]
    # Breadth before extra directions: a whole high-risk area should not vanish
    # while another area consumes the budget with duplicate target actions.
    for pass_index in range(cfg.max_actions_per_region):
        for area in sorted(areas,key=lambda x:-x['risk']):
            previous=chosen_by[area['id']];options=[]
            for c in candidates:
                if c['area_id']!=area['id'] or any(x['id']==c['id'] for x in previous):continue
                if total+c['photo_count']>cfg.photo_budget:continue
                if previous:
                    d=-camera_basis(c['yaw_deg'],c['pitch_deg'])[:,2]
                    differences=[np.degrees(np.arccos(np.clip(d@(-camera_basis(p['yaw_deg'],p['pitch_deg'])[:,2]),-1,1))) for p in previous]
                    if min(differences)<25:continue
                    # For mixed/unknown geometry, never call two near-nadir rows
                    # two independent side inspections.
                before=np.maximum.reduce([p['_utility'] for p in previous]) if previous else c['_prior']
                gain=float(np.maximum(c['_utility']-before,0).mean())
                if pass_index>=2 and gain<.02:continue
                link=c['old_view_link'];connection=0 if link is None else link['distance_m']/cfg.speed_mps
                cost=c['photo_count']*cfg.interval_s+c['length_m']/cfg.speed_mps+connection
                # Second complementary pass is a coverage safeguard, not inferred
                # PSNR gain. Its minimum bonus is reported rather than hidden.
                geometric_bonus=.05 if pass_index==1 else 0.
                score=area['risk']*(gain+geometric_bonus)/max(cost,1.)
                if pass_index==0 and c['mode']=='roof_nadir' and area['kind']=='roof':score*=1.1
                options.append((score,-c['photo_count'],c['id'],c,gain))
            if not options:continue
            accepted=None;connected_options=[]
            checked=0
            for _,_,_,c,gain in sorted(options,key=lambda x:x[:3],reverse=True):
                if '_connection_error' in c:continue
                if '_connection' not in c:
                    context,anchors=contexts[area['id']]
                    connection,error=prepare_connected_action(c,area,context,anchors,tree,hfov,vfov,cfg,ceiling)
                    checked+=1
                    if error:c['_connection_error']=error;continue
                    c['_connection']=connection
                if total+c['_connection']['all_photo_count']>cfg.photo_budget:continue
                count=c['_connection']['all_photo_count'];entry=c['_connection']['entry_bridge']
                sequence=entry+c['_connection']['scan_sequence']
                positions=np.array([w['position_enu_m'] for w in sequence])
                path=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum())
                actualcost=count*cfg.interval_s+path/cfg.speed_mps
                bonus=.05 if pass_index==1 else 0.
                connected_options.append((area['risk']*(gain+bonus)/max(actualcost,1),-count,c['id'],c,gain))
                # Bounded candidate beam, NOT exact global optimization.
                if len(connected_options)>=5:break
            if connected_options:
                _,_,_,c,gain=max(connected_options,key=lambda x:x[:3]);accepted=(c,gain)
            if accepted is None:
                print('Connection area',area['ranks'],'pass',pass_index,'none feasible','new checks',checked,flush=True)
                continue
            c,gain=accepted;chosen.append(c);previous.append(c);total+=c['_connection']['all_photo_count']
            print('Connection area',area['ranks'],'pass',pass_index,'action',c['id'],'photos',c['_connection']['all_photo_count'],'total',total,flush=True)
            trace.append(dict(action=c['id'],area=area['id'],pass_index=pass_index,photos_so_far=total,
                incremental_pair_proxy=gain,complementary_coverage_bonus=.05 if pass_index==1 else 0.,
                connection_photo_count=c['_connection']['bridge_photo_count']+c['_connection']['extra_scan_photos']))
    return chosen,trace


def connect_actions(actions,manifest_index,tree,cfg,ceiling,remaining_budget):
    """Execute only already-audited old-to-new photo chains. No free bridges."""
    ordered=[];remaining=list(actions);current=np.array(list(manifest_index.values())[-1]['enu'])
    while remaining:
        costs=[np.linalg.norm(current-np.array(c['_connection']['entry_bridge'][0]['position_enu_m'])) for c in remaining]
        c=dict(remaining.pop(int(np.argmin(costs))));connection=c['_connection']
        c['entry_bridge']=connection['entry_bridge'];c['waypoints']=connection['scan_sequence']
        c['photo_count']=len(c['waypoints']);c['connection_status']=connection['connection_status']
        c['connection_audit']={k:v for k,v in connection.items() if k not in ('entry_bridge','scan_sequence')}
        c['image_matching_verified']=False;ordered.append(c);current=np.array(c['waypoints'][-1]['position_enu_m'])
    trajectory=[];transit_violations=[];current=np.array(list(manifest_index.values())[-1]['enu']);t=0.
    for c in ordered:
        shots=c['entry_bridge']+c['waypoints']
        for wi,w in enumerate(shots):
            p=np.array(w['position_enu_m']);d=np.linalg.norm(p-current)
            samples=np.linspace(current,p,max(2,int(np.ceil(d/2))+1));clearance=float(tree.query(samples)[0].min())
            duration=max(cfg.interval_s,motion_seconds(current,p,cfg))
            if clearance<cfg.clearance_m:transit_violations.append(dict(action=c['id'],photo_index=wi,clearance_proxy_m=clearance))
            for j,q in enumerate(samples[1:-1],1):
                trajectory.append(dict(position_enu_m=q.tolist(),capture=False,time_lower_bound_s=float(t+j/(len(samples)-1)*duration)))
            t+=duration
            trajectory.append(dict(**w,time_lower_bound_s=float(t)));current=p
    return ordered,trajectory,transit_violations


def serialize_area(a):
    pts=a['points']
    return dict(id=a['id'],ranks=a['ranks'],kind=a['kind'],trusted_normal=a['trusted_normal'],normal=a['normal'].tolist(),
        center=a['center'].tolist(),cell_m=a['cell_m'],uncertainty_m=a['uncertainty_m'],occupied_surface_samples=pts.tolist(),
        task_type=a.get('task_type'),risk_extent_audit=a.get('risk_extent_audit'),
        context_surface_samples=a.get('context_surface_points',pts).tolist(),
        risk=a['risk'],reference_range_m=a['reference_range'],
        secondary_normal_modes=[dict(rank=r['rank'],modes=r['diag']['normal_modes']) for r in a['members']],
        members=[dict(rank=r['rank'],frozen=r['frozen'],kind=r['kind'],normal=r['normal'].tolist(),trusted_normal=r['trusted_normal'],
            extent_source=r['extent_source'],context_radius_m=r['context_radius_m'],point_count=len(r['points']),
            raw_point_count=len(r['raw_points']),flags=r['diag']['flags']) for r in a['members']],
        scope='Native predicted risk support; context is NOT a full-building or full-defect target')


def emit_selection_controls(output,plan,areas,candidates,samples,manifest_index,tree,cfg,hfov,vfov):
    """Exact same generated candidates. No repeated Pi3X/context inference."""
    cases=[('no_context_preference',replace(cfg,use_context_preference=False,local_multiview=False)),
           ('geometry_nominal45',replace(cfg,selection_policy='geometry',use_context_preference=False,local_multiview=False))]
    summary=[];mainids={c['id'] for c in plan['actions']}
    clean=lambda c:{k:v for k,v in c.items() if not k.startswith('_')}
    for name,setting in cases:
        chosen,trace=choose_surveys(areas,candidates,setting)
        actions,trajectory,unsafe=connect_surveys(chosen,manifest_index,tree,setting)
        strips=[dict(id=c['id']+'_scan',rank=c['rank'],area_id=c['area_id'],center=c['center'],waypoints=c['waypoints']) for c in actions]
        rows=[]
        for area in areas:
            owned=[c for c in actions if c['area_id']==area['id']];wps=[w for c in owned for w in c['waypoints']]
            cov,_,_=coverage_measure(samples[area['id']][1],wps,hfov,vfov)
            old=next(r for r in plan['regions'] if r['area_id']==area['id'])
            rows.append(dict(old,photos=len(wps),rows=sum(c['row_count'] for c in owned),
                coverage=cov,selected_actions=[c['id'] for c in owned],action_modes=[c['mode'] for c in owned],
                status='planned' if owned else 'UNRESOLVED_no_feasible_action_or_budget',
                task_complete=len(owned)==(1 if area['task_type']=='local_patch' else 5),
                survey_complete=len(owned)==5,scan_directions=[c['survey_direction'] for c in owned],
                connection_status=[c['connection_status'] for c in owned]))
        result=dict(plan,comparison_case=name,settings=asdict(setting),actions=[clean(c) for c in actions],
            groups=strips,segments=strips,regions=rows,trajectory=trajectory,
            photo_count=sum(len(c['waypoints']) for c in actions),scan_photo_count=sum(len(c['waypoints']) for c in actions),
            entry_photo_count=0,selection_history=trace,transit_clearance_warnings=unsafe)
        positions=np.array([w['position_enu_m'] for c in actions for w in c['waypoints']]).reshape(-1,3)
        if len(positions):
            origin=plan['altitude_reference']['enu_absolute_origin_m']
            reference=plan['altitude_reference'].get('takeoff_enu_z')
            if reference is None:reference=plan['altitude_reference'].get('conservative_reference_enu_z')
            result['altitude_exif_range_m']=[float(positions[:,2].min()+origin),float(positions[:,2].max()+origin)]
            result['altitude_takeoff_relative_range_m']=[float(positions[:,2].min()-reference),float(positions[:,2].max()-reference)] if reference is not None else None
        path=output/'controls'/name;path.mkdir(parents=True,exist_ok=True)
        json_dump(path/'route_plan.json',result)
        json_dump(path/'regions_frozen.json',json.loads((output/'regions_frozen.json').read_text()))
        ids={c['id'] for c in actions}
        summary.append(dict(case=name,photo_budget=setting.photo_budget,photos=result['photo_count'],
            completed_tasks=sum(r['task_complete'] for r in rows),selected_action_count=len(ids),
            identical_selected_action_set_to_main=ids==mainids,shared_selected_actions=len(ids&mainids),
            regional_choices=[dict(ranks=r['ranks'],photos=r['photos'],actions=r['selected_actions']) for r in rows],
            quality_improvement_measured=False))
    json_dump(output/'ablation_summary.json',dict(main_photos=plan['photo_count'],cases=summary,
        protocol='Identical frozen targets, geometry, coverage checks and photo cap. Geometry control excludes historical-orientation candidates and removes support/context terms. No-context control removes ONLY context score. Budget is an upper bound, not equal realized photos.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['selection','surface-root','manifest','cloud','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--target-budget',type=int,default=18);p.add_argument('--photo-budget',type=int,default=240,help='0: preview cost of all feasible complete surveys')
    p.add_argument('--max-relative-altitude',type=float,default=80.)
    p.add_argument('--selection-policy',choices=['observation','geometry'],default='observation')
    p.add_argument('--disable-context-preference',action='store_true')
    p.add_argument('--write-controls',action='store_true',help='Replay geometry and no-context selectors on same in-memory candidates')
    p.add_argument('--local-multiview',action='store_true',help='Add local directions only for residual joint geometry gain')
    p.add_argument('--surface-multiview',action='store_true',help='Experimental per-point reliable surface grouping; preserves first strips')
    a=p.parse_args();cfg=CoverageSettings(target_budget=a.target_budget,photo_budget=a.photo_budget,
        max_relative_altitude_m=a.max_relative_altitude,selection_policy=a.selection_policy,
        use_context_preference=not a.disable_context_preference,local_multiview=a.local_multiview or a.surface_multiview,
        surface_multiview=a.surface_multiview)
    a.output.mkdir(parents=True,exist_ok=False);tic=time.time()
    selection=json.loads(a.selection.read_text());observations=json.loads((a.surface_root/'observations/observations.json').read_text())
    assert sha(a.selection)==observations['selection_sha256']
    manifest=json.loads(a.manifest.read_text());manifest_index={r['stem']:r for r in manifest['records']}
    altitude=altitude_reference(manifest);json_dump(a.output/'altitude_reference.json',altitude)
    reference_z=altitude.get('takeoff_enu_z')
    if reference_z is None:reference_z=altitude.get('conservative_reference_enu_z')
    ceiling=reference_z+cfg.max_relative_altitude_m if reference_z is not None else None
    altitude['ceiling_policy']=f'{cfg.max_relative_altitude_m:g}m above lowest sampled sensor takeoff reference; not a known future takeoff datum'
    altitude['assumption']='Configured height is a preview limit above a historical takeoff datum; not user-confirmed AGL or a new flight setting'
    altitude['applied_ceiling_enu_z']=ceiling
    json_dump(a.output/'altitude_reference.json',altitude)
    print('Altitude reference',altitude['status'],'ceiling ENU',ceiling,flush=True)
    selected=selection['selected'][:cfg.target_budget];diags=observations['regions'][:len(selected)]
    clouds={d['chunk']:dict(np.load(a.surface_root/f"geometry/chunk_{d['chunk']:04d}.npz")) for d in diags}
    cloud=dict(np.load(a.cloud));tree=cKDTree(cloud['xyz'])
    hfov,vfov=np.radians([diags[0]['horizontal_fov_deg'],diags[0]['vertical_fov_deg']])
    regions=[]
    for i,(s,d) in enumerate(zip(selected,diags),1):
        assert d['rank']==i and d['chunk']==s['chunk'];np.testing.assert_allclose(d['frozen_xyz'],s['xyz'])
        surface=dict(np.load(a.surface_root/f'observations/rank_{i:02d}_surface.npz'))
        region=extract_region(i,s,d,surface,clouds[d['chunk']],cfg);regions.append(region)
        print('Region',i,region['kind'],'raw',len(surface['xyz']),'surface',len(region['points']),flush=True)
    areas,merge_audit=build_risk_tasks(regions,cfg,hfov,vfov,long_axis_yaw)
    # Persist regions BEFORE any action selection, and before downstream GS access.
    json_dump(a.output/'regions_frozen.json',dict(selection_sha256=sha(a.selection),settings=asdict(cfg),
        regions=[serialize_area(x) for x in areas],merge_audit=merge_audit,used_GS_or_SfM=False))
    candidates=[];diagnostics=[];samples={};contexts={}
    all_risk_points=np.vstack([r['raw_points'] for r in regions])
    for area in areas:
        area['_all_risk_points']=all_risk_points
        acts,diag,pts,probes=generate_surveys(area,clouds,manifest_index,hfov,vfov,cfg,ceiling,tree,
            sample_points,old_observations,action_score,long_axis_yaw)
        candidates.extend(acts);diagnostics.append(diag);samples[area['id']]=(pts,probes)
        print('Area',area['ranks'],'candidates',len(acts),'rejections',diag['rejections'],flush=True)
    chosen,trace=choose_surveys(areas,candidates,cfg)
    if cfg.surface_multiview:
        from openflyscan.planning.local_surface_evidence import rescore_local_candidates
        candidates,chosen,surface_audit=rescore_local_candidates(areas,candidates,chosen,samples,clouds,
            manifest_index,hfov,vfov,cfg,tree,old_observations,action_score)
        json_dump(a.output/'local_surface_evidence.json',surface_audit)
    if cfg.local_multiview:
        from openflyscan.planning.local_multiview_selection import augment_local_directions
        chosen,multi_audit=augment_local_directions(areas,candidates,chosen,cfg,pair_utility)
        json_dump(a.output/'local_multiview_audit.json',multi_audit)
    used=sum(c['photo_count'] for c in chosen)
    actions,trajectory,unsafe=connect_surveys(chosen,manifest_index,tree,cfg)
    strips=[]
    for c in actions:
        if c['entry_bridge']:
            strips.append(dict(id=c['id']+'_entry',rank=c['rank'],area_id=c['area_id'],center=c['center'],waypoints=c['entry_bridge']))
        strips.append(dict(id=c['id']+'_scan',rank=c['rank'],area_id=c['area_id'],center=c['center'],waypoints=c['waypoints']))
    rows=[]
    for area in areas:
        owned=[c for c in actions if c['area_id']==area['id']];wps=[w for c in owned for w in c['waypoints']]
        cov,_,_=coverage_measure(samples[area['id']][1],wps,hfov,vfov)
        rows.append(dict(area_id=area['id'],ranks=area['ranks'],kind=area['kind'],surface_samples=len(area['points']),
            selected_actions=[c['id'] for c in owned],action_modes=[c['mode'] for c in owned],rows=sum(c['row_count'] for c in owned),
            photos=len(wps),entry_photos=sum(len(c['entry_bridge']) for c in owned),coverage=cov,
            status='planned' if owned else 'UNRESOLVED_no_feasible_action_or_budget',
            task_type=area['task_type'],risk_extent_audit=area['risk_extent_audit'],
            task_complete=(1<=len(owned)<=cfg.local_max_directions if area['task_type']=='local_patch' else len(owned)==5),
            survey_complete=len(owned)==5,scan_directions=[c['survey_direction'] for c in owned],
            incomplete_unknown_facade=not area['trusted_normal'],
            connection_status=[c['connection_status'] for c in owned]))
    allshots=[w for s in strips for w in s['waypoints']]
    assert cfg.photo_budget==0 or len(allshots)<=cfg.photo_budget
    positions=np.array([w['position_enu_m'] for w in allshots]).reshape(-1,3)
    axis_error=max((float(np.linalg.norm(camera_basis(w['aircraft_yaw_deg'],w['gimbal_pitch_deg'])[:,2]-unit(np.array(w['target_enu_m'])-w['position_enu_m']))) for w in allshots),default=0)
    assert axis_error<1e-6
    if ceiling is not None and len(positions):assert positions[:,2].max()<=ceiling+1e-6
    def clean(c):return {k:v for k,v in c.items() if not k.startswith('_')}
    plan=dict(status='preview_only',flight_authorized=False,used_GS_or_SfM=False,selection_sha256=sha(a.selection),
        policy='native_risk_extent_local_strip_or_connected_area_survey_no_entry_photos',
        old_overlap_policy='Optional <=5m same-height reliable OUTSIDE-risk context preference; geometry-only, not feature matching',
        angle_policy='-45 nominal; local old orientations allowed; steeper fallback only when nominal geometry has no complete task',
        settings=asdict(cfg),altitude_reference=altitude,camera=dict(horizontal_fov_deg=float(np.degrees(hfov)),vertical_fov_deg=float(np.degrees(vfov))),
        groups=strips,segments=strips,actions=[clean(c) for c in actions],trajectory=trajectory,regions=rows,
        photo_count=len(allshots),scan_photo_count=sum(len(c['waypoints']) for c in actions),entry_photo_count=sum(len(c['entry_bridge']) for c in actions),
        selection_history=trace,transit_clearance_warnings=unsafe,
        max_optical_axis_error=axis_error,source_script_sha256=sha(__file__),seconds=time.time()-tic,
        altitude_exif_range_m=(positions[:,2]+altitude['enu_absolute_origin_m']).min().item() if len(positions) else None,
        limitations=['Observed-cloud clearance is NOT obstacle safety; straight connecting paths may be unsafe',
            'No feature matching verification on future images; overlap and angle are geometry proxies',
            'Connected surface context is not a complete defect polygon; missing facades remain unobserved',
            'Risk task grouping and local-versus-area thresholds are prototype heuristics; GS improvement remains unvalidated',
            '80m interpreted as takeoff-relative for this preview, not user-confirmed mission setting',
            'Three-view coverage includes localization hypotheses and does not prove lack of occlusion'])
    if len(positions):
        plan['altitude_exif_range_m']=[float(v) for v in [positions[:,2].min()+altitude['enu_absolute_origin_m'],positions[:,2].max()+altitude['enu_absolute_origin_m']]]
        plan['altitude_takeoff_relative_range_m']=[float(positions[:,2].min()-reference_z),float(positions[:,2].max()-reference_z)] if ceiling is not None else None
        plan['relative_height_reference']='lowest sampled historical takeoff reference, NOT confirmed future takeoff'
    json_dump(a.output/'route_plan.json',plan);json_dump(a.output/'candidate_actions.json',[clean(c) for c in candidates])
    json_dump(a.output/'generation_audit.json',diagnostics)
    if a.write_controls:emit_selection_controls(a.output,plan,areas,candidates,samples,manifest_index,tree,cfg,hfov,vfov)
    json_dump(a.output/'connection_candidate_audit.json',[dict(id=c['id'],area=c['area_id'],error=c.get('_connection_error'),
        connection={k:v for k,v in c.get('_connection',{}).items() if k not in ('scan_sequence','entry_bridge')}) for c in candidates if '_connection_error' in c or '_connection' in c])
    print(json.dumps(dict(areas=len(areas),actions=len(actions),strips=len(strips),photos=len(allshots),
        entry=sum(len(c['entry_bridge']) for c in actions),
        extra_scan_transition=0,
        altitude=plan.get('altitude_takeoff_relative_range_m'),unsafe_links=len(unsafe),seconds=plan['seconds']),indent=2),flush=True)


if __name__=='__main__':main()
