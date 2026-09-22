"""Sensor/Pi3X-only image-chain checks. Predicted overlap != verified matching.

Avoid small-target saturation: compare common context points, require projected
image spread, and check every new edge. No missing-evidence pass is allowed.
"""
import numpy as np
from scipy.spatial import cKDTree

from openflyscan.planning.plan_frozen_region_scans import camera_basis,frustum_contains
from openflyscan.planning.region_coverage_geometry import unit,unique_voxels


def pose_record(position,yaw,pitch,mode='ENTRY_BRIDGE'):
    p=np.asarray(position,float);forward=camera_basis(yaw,pitch)[:,2]
    return dict(position_enu_m=p.tolist(),target_enu_m=(p+forward*50).tolist(),
        aircraft_yaw_deg=float(yaw%360),gimbal_pitch_deg=float(pitch),capture=True,
        capture_view=mode,minimum_interval_s=2.,speed_mps=4.)


def interpolate(a,b,t):
    p=(1-t)*np.array(a['position_enu_m'])+t*np.array(b['position_enu_m'])
    dy=(b['aircraft_yaw_deg']-a['aircraft_yaw_deg']+180)%360-180
    yaw=a['aircraft_yaw_deg']+t*dy;pitch=(1-t)*a['gimbal_pitch_deg']+t*b['gimbal_pitch_deg']
    return pose_record(p,yaw,pitch)


def context_for_area(area,clouds,manifest_index):
    """Uniform 3D samples from broad old-image context, not only risk patches.

    Points must be supported by >=2 cached images in their SOURCE chunk. This
    is still predicted geometry, not measured SfM tracks or a GS quality label.
    """
    result=[];anchors={};center=area['center'];radius=max(30.,area['reference_range']*2)
    for chunk in sorted({m['diag']['chunk'] for m in area['members']}):
        cloud=clouds[chunk];points=cloud['maps'].reshape(-1,3)
        confidence=cloud['conf'].reshape(-1)
        finite=np.isfinite(points).all(1)&np.isfinite(confidence)
        ids=np.flatnonzero(finite&(confidence>=np.quantile(confidence[finite],.5))&(np.linalg.norm(points-center,axis=1)<=radius))
        ids=ids[np.linspace(0,len(ids)-1,min(3500,len(ids))).astype(int)] if len(ids) else ids
        pts=unique_voxels(points[ids],1.)
        if not len(pts):continue
        support=np.zeros(len(pts),int);viewcounts=[];h,w=cloud['target_hw']
        for vi,stem in enumerate(cloud['stems'].astype(str)):
            pose=cloud['camera_poses'][vi];K=cloud['intrinsics'][vi]
            pc=(pts-pose[:3,3])@pose[:3,:3];uv=pc@K.T;uv=uv[:,:2]/np.maximum(uv[:,2,None],1e-8)
            inside=(pc[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            yi=np.abs(cloud['dense_y'][:,None]-uv[:,1]).argmin(0);xi=np.abs(cloud['dense_x'][:,None]-uv[:,0]).argmin(0)
            err=np.linalg.norm(cloud['dense_maps'][vi,yi,xi]-pts,axis=1)
            good=inside&(err<=np.maximum(1.,.03*np.abs(pc[:,2])))
            support+=good
            viewcounts.append((stem,good))
        reliable=support>=2;result.append(pts[reliable])
        for stem,good in viewcounts:
            count=int((good&reliable).sum())
            if count<30:continue
            rec=manifest_index[stem];yaw,pitch,_=rec['camera_ypr_deg']
            anchor=dict(stem=stem,waypoint=pose_record(rec['enu'],yaw,pitch,'OLD_PHOTO_ANCHOR'),supported_context_points=count,
                evidence='Pi3X multi-image depth-compatible context, not GS/PSNR or matched feature tracks')
            if stem not in anchors or count>anchors[stem]['supported_context_points']:anchors[stem]=anchor
    pts=unique_voxels(np.vstack(result),1.) if result else np.empty((0,3))
    if len(pts)>8000:pts=pts[np.linspace(0,len(pts)-1,8000).astype(int)]
    return pts,list(anchors.values())


def visibility(points,w,hfov,vfov):
    p=np.array(w['position_enu_m']);pc=(points-p)@camera_basis(w['aircraft_yaw_deg'],w['gimbal_pitch_deg'])
    uv=np.column_stack([pc[:,0]/np.maximum(pc[:,2],1e-9)/np.tan(hfov/2),pc[:,1]/np.maximum(pc[:,2],1e-9)/np.tan(vfov/2)])
    visible=(pc[:,2]>0)&(abs(uv[:,0])<1)&(abs(uv[:,1])<1)
    # Nearest predicted depth per image bin, not full mesh ray tracing.
    cells=np.clip(((uv+1)*[32,18]).astype(int),0,[63,35]);key=cells[:,1]*64+cells[:,0]
    minz=np.full(64*36,np.inf);np.minimum.at(minz,key[visible],pc[visible,2])
    visible &= pc[:,2]<=minz[key]+np.maximum(1.,pc[:,2]*.03)
    return visible,uv


def projected_scale_change(points,ua,ub):
    """Compare the projected length of identical local 3-D point pairs.

    This measures perspective scale/foreshortening, not pixel texture or feature
    matchability. No assumption that camera altitude equals surface distance.
    """
    if len(points)<4:return None
    distances,indices=cKDTree(points).query(points,k=4)
    src=np.repeat(np.arange(len(points)),3);dst=indices[:,1:].reshape(-1)
    da=np.linalg.norm(ua[src]-ua[dst],axis=1);db=np.linalg.norm(ub[src]-ub[dst],axis=1)
    good=(distances[:,1:].reshape(-1)>1e-6)&(da>1e-6)&(db>1e-6)
    if good.sum()<20:return None
    ratio=np.maximum(da[good]/db[good],db[good]/da[good])
    return dict(median=float(np.median(ratio)),p90=float(np.quantile(ratio,.9)),point_pairs=int(good.sum()))


def motion_seconds(a,b,cfg):
    delta=np.asarray(b)-a
    vertical_speed=cfg.climb_speed_mps if delta[2]>=0 else cfg.descent_speed_mps
    return float(max(np.linalg.norm(delta)/cfg.speed_mps,abs(delta[2])/vertical_speed))


def overlap_edge(points,a,b,hfov,vfov,cfg):
    if len(points)<30:return dict(passed=False,reason='insufficient_context_points',common_surface_overlap=0.)
    va,ua=visibility(points,a,hfov,vfov);vb,ub=visibility(points,b,hfov,vfov);common=va&vb
    n=int(common.sum());ratio=float(n/max(int(va.sum()),int(vb.sum()),1))
    spans=[];gridcells=[]
    for uv in [ua[common],ub[common]]:
        span=np.ptp(uv,axis=0)/2 if len(uv)>1 else np.zeros(2)
        spans.append(span.tolist());gridcells.append(len(np.unique(np.floor((uv+1)*[6,4]).astype(int),axis=0)))
    aa=camera_basis(a['aircraft_yaw_deg'],a['gimbal_pitch_deg'])[:,2]
    bb=camera_basis(b['aircraft_yaw_deg'],b['gimbal_pitch_deg'])[:,2]
    angle=float(np.degrees(np.arccos(np.clip(aa@bb,-1,1))))
    distance=float(np.linalg.norm(np.array(a['position_enu_m'])-b['position_enu_m']))
    scale=projected_scale_change(points[common],ua[common],ub[common])
    ra=points[common]-a['position_enu_m'];rb=points[common]-b['position_enu_m']
    ra/=np.maximum(np.linalg.norm(ra,axis=1)[:,None],1e-8)
    rb/=np.maximum(np.linalg.norm(rb,axis=1)[:,None],1e-8)
    surface_angle=float(np.quantile(np.degrees(np.arccos(np.clip(np.sum(ra*rb,axis=1),-1,1))),.9)) if n else None
    spread=all(min(x)>=.12 for x in spans) and min(gridcells)>=6
    scale_ok=scale is not None and scale['p90']<=cfg.connection_scale_ratio_p90+1e-6
    parallax_ok=surface_angle is not None and surface_angle<=cfg.connection_surface_angle_p90_deg+1e-6
    passed=(n>=30 and ratio>=cfg.connection_overlap and spread and angle<=cfg.connection_angle_step_deg+1e-6 and scale_ok and parallax_ok)
    return dict(passed=bool(passed),common_surface_overlap=ratio,common_points=n,visible_a=int(va.sum()),visible_b=int(vb.sum()),
        projected_span_xy=spans,occupied_image_grid_cells=gridcells,rotation_step_deg=angle,translation_step_m=distance,
        height_change_m=float(b['position_enu_m'][2]-a['position_enu_m'][2]),
        projected_scale_change=scale,surface_ray_angle_p90_deg=surface_angle,
        reason='pass' if passed else 'context_overlap_spread_angle_or_scale_failed',
        meaning='Common fraction of visible observed context samples, NOT full-image area overlap or feature inlier ratio')


def refine_edge(a,b,points,hfov,vfov,cfg,depth=0):
    measure=overlap_edge(points,a,b,hfov,vfov,cfg)
    duration=motion_seconds(a['position_enu_m'],b['position_enu_m'],cfg)
    if measure['passed'] and duration<=cfg.interval_s+1e-6:return [b],[measure],None
    if depth>=6:return [],[],dict(reason='cannot_establish_local_overlap',edge=measure)
    mid=interpolate(a,b,.5)
    first,m1,error=refine_edge(a,mid,points,hfov,vfov,cfg,depth+1)
    if error:return [],[],error
    second,m2,error=refine_edge(mid,b,points,hfov,vfov,cfg,depth+1)
    if error:return [],[],error
    return first+second,m1+m2,None


def prepare_connected_action(action,area,context,anchors,tree,hfov,vfov,cfg,ceiling):
    first=action['waypoints'][0];last=action['waypoints'][-1]
    choices=[]
    for anchor in anchors:
        w=anchor['waypoint']
        # Old camera is revisited at its sensor position, not optimistically
        # moved to a locally better GS or model-aligned pose.
        if ceiling is not None and w['position_enu_m'][2]>ceiling:continue
        if w['gimbal_pitch_deg']+np.degrees(vfov)/2>=-5:continue
        same=overlap_edge(context,w,w,hfov,vfov,cfg)
        if not same['passed']:continue
        for reverse,end in [(False,first),(True,last)]:
            dist=np.linalg.norm(np.array(w['position_enu_m'])-end['position_enu_m'])
            angle=abs((w['aircraft_yaw_deg']-end['aircraft_yaw_deg']+180)%360-180)
            choices.append((dist+angle*.2,-anchor['supported_context_points'],anchor,reverse))
    for _,_,anchor,reverse in sorted(choices,key=lambda x:x[:2])[:6]:
        scan=list(action['waypoints'])
        if reverse:scan.reverse()
        old=anchor['waypoint'];chain,measures,error=refine_edge(old,scan[0],context,hfov,vfov,cfg)
        if error:continue
        # First new photo repeats a sensor pose with broad old-image support.
        entry=[dict(old)]+chain[:-1]
        scan_chain=[scan[0]];scan_measures=[];extra=[]
        for a,b in zip(scan[:-1],scan[1:]):
            refined,m,error=refine_edge(a,b,context,hfov,vfov,cfg)
            if error:break
            scan_chain.extend(refined);scan_measures.extend(m);extra.extend(refined[:-1])
        if error:continue
        if len(entry)+len(extra)>cfg.max_connection_photos:continue
        allpoints=np.array([w['position_enu_m'] for w in entry+scan_chain])
        if ceiling is not None and allpoints[:,2].max()>ceiling:continue
        if any(w['gimbal_pitch_deg']+np.degrees(vfov)/2>=-5 for w in entry+scan_chain):continue
        path=np.vstack([np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(a-b)/2))+1)) for a,b in zip(allpoints[:-1],allpoints[1:])])
        clearance=float(tree.query(path)[0].min())
        if clearance<cfg.clearance_m:continue
        return dict(entry_bridge=entry,scan_sequence=scan_chain,anchor_stem=anchor['stem'],
            connection_status='every_edge_predicted_context_overlap_pass_features_UNVERIFIED',
            entry_measures=measures,scan_measures=scan_measures,extra_scan_photos=len(extra),
            bridge_photo_count=len(entry),all_photo_count=len(entry)+len(scan_chain),
            minimum_context_overlap=min(m['common_surface_overlap'] for m in measures+scan_measures),
            maximum_scale_ratio_p90=max(m['projected_scale_change']['p90'] for m in measures+scan_measures),
            maximum_height_step_m=max(abs(m['height_change_m']) for m in measures+scan_measures),
            path_clearance_proxy_m=clearance,reverse=reverse,feature_matching_verified=False),None
    return None,dict(reason='no_old_anchor_to_scan_chain_passed',context_points=len(context),anchors_considered=min(6,len(choices)))
