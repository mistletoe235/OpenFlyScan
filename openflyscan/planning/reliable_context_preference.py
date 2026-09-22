"""Optional old/new context preference, never a capture-chain requirement.

Depth-compatible Pi3X observations are not feature tracks or proof of good GS.
No RGB/GS residual, manually picked anchor, or exact camera GT is consumed.
"""
import numpy as np
from scipy.spatial import cKDTree
from openflyscan.planning.region_coverage_geometry import unique_voxels
from openflyscan.planning.recapture_overlap_connections import visibility,projected_scale_change
from openflyscan.planning.plan_frozen_region_scans import camera_basis


def project_old(points,pose,K,hw):
    h,w=hw;pc=(points-pose[:3,3])@pose[:3,:3]
    uv=pc@K.T;uv=uv[:,:2]/np.maximum(uv[:,2,None],1e-8)
    valid=(pc[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    return valid,2*uv/np.array([w,h])-1,pc


def build_reliable_context(area,clouds,manifest,cfg):
    result=[];center=area['center'];radius=max(30.,area['reference_range']*2.)
    risk=area.get('_all_risk_points',area['points']);risk_tree=cKDTree(risk)
    exclusion=area['uncertainty_m']+area['cell_m']/2
    retained=0
    for chunk in sorted({m['diag']['chunk'] for m in area['members']}):
        cloud=clouds[chunk];base=cloud['maps'].reshape(-1,3);conf=cloud['conf'].reshape(-1)
        finite=np.isfinite(base).all(1)&np.isfinite(conf)
        if not finite.any():continue
        ids=np.flatnonzero(finite&(conf>=np.quantile(conf[finite],.5))&(np.linalg.norm(base-center,axis=1)<=radius))
        if len(ids)>2500:ids=ids[np.linspace(0,len(ids)-1,2500).astype(int)]
        points=unique_voxels(base[ids],1.)
        if not len(points):continue
        points=points[risk_tree.query(points)[0]>exclusion]
        if not len(points):continue
        masks=[];uvs=[]
        for vi,stem in enumerate(cloud['stems'].astype(str)):
            pose=cloud['camera_poses'][vi];K=cloud['intrinsics'][vi]
            inside,uv,pc=project_old(points,pose,K,cloud['target_hw'])
            pix=(uv+1)*np.array(cloud['target_hw'][::-1])/2
            yi=np.abs(cloud['dense_y'][:,None]-pix[:,1]).argmin(0)
            xi=np.abs(cloud['dense_x'][:,None]-pix[:,0]).argmin(0)
            depthmap=cloud['dense_maps'][vi];confidence=cloud['dense_conf'][vi]
            error=np.linalg.norm(depthmap[yi,xi]-points,axis=1)
            good=inside&(error<=np.maximum(1.,.03*np.abs(pc[:,2])))
            good &= confidence[yi,xi]>=np.nanquantile(confidence,.5)
            masks.append(good);uvs.append(uv)
        support=np.asarray(masks).sum(0);stable=support>=2;retained+=int(stable.sum())
        if stable.sum()<cfg.context_min_points:continue
        for vi,stem in enumerate(cloud['stems'].astype(str)):
            mask=masks[vi]&stable
            if mask.sum()<cfg.context_min_points:continue
            rec=manifest[stem]
            result.append(dict(stem=stem,chunk=chunk,points=points[mask],old_uv=uvs[vi][mask],
                native_pose=cloud['camera_poses'][vi],sensor_position=np.array(rec['enu']),
                image=rec['image'],hw=cloud['target_hw'],
                sensor_native_position_gap_m=float(np.linalg.norm(cloud['camera_poses'][vi,:3,3]-rec['enu']))))
    # Bounded work, not a claim of globally optimal anchor selection.
    result=sorted(result,key=lambda x:(-len(x['points']),x['stem'],x['chunk']))[:cfg.context_max_anchors]
    return result,dict(anchor_count=len(result),stable_source_points_before_cross_chunk_dedup=retained,
        risk_exclusion_padding_m=exclusion,search_radius_m=radius,
        geometry_only=True,feature_matching_verified=False,GS_quality_verified=False)


def context_pair(anchor,waypoint,hfov,vfov,cfg):
    pos=np.array(waypoint['position_enu_m'])
    dz=float(abs(pos[2]-anchor['sensor_position'][2]))
    if dz>cfg.same_height_tolerance_m:return None
    forward=camera_basis(waypoint['aircraft_yaw_deg'],waypoint['gimbal_pitch_deg'])[:,2]
    angle=float(np.degrees(np.arccos(np.clip(forward@anchor['native_pose'][:3,2],-1,1))))
    if angle>cfg.context_axis_tolerance_deg:return None
    visible,uv=visibility(anchor['points'],waypoint,hfov,vfov)
    count=int(visible.sum());fraction=count/len(visible)
    span_new=float(np.prod(np.ptp(uv[visible],axis=0))/4) if count>1 else 0.
    span_old=float(np.prod(np.ptp(anchor['old_uv'][visible],axis=0))/4) if count>1 else 0.
    scale=projected_scale_change(anchor['points'][visible],anchor['old_uv'][visible],uv[visible]) if count>=cfg.context_min_points else None
    spread=min(1.,span_new/.15,span_old/.15)
    scale_factor=min(1.,1.5/scale['p90']) if scale else 0.
    score=fraction*spread*scale_factor if count>=cfg.context_min_points else 0.
    report=dict(stem=anchor['stem'],chunk=anchor['chunk'],common_points=count,
        old_supported_context_points=len(visible),common_context_fraction=fraction,
        normalized_old_span_area=span_old,normalized_new_span_area=span_new,
        scale_change=scale,score=float(score),height_difference_m=dz,optical_axis_difference_deg=angle,
        old_sensor_native_position_gap_m=anchor['sensor_native_position_gap_m'],
        note='Point visibility/span proxies, NOT image overlap percentage or matched features',
        image_matching_verified=False,GS_used=False)
    ids=np.flatnonzero(visible)
    if len(ids)>200:ids=ids[np.linspace(0,len(ids)-1,200).astype(int)]
    preview=dict(old_image=anchor['image'],old_hw=np.asarray(anchor['hw']).tolist(),
        old_uv=anchor['old_uv'][ids].tolist(),new_uv=uv[ids].tolist(),
        points=anchor['points'][ids].tolist(),old_native_position=anchor['native_pose'][:3,3].tolist(),
        old_sensor_position=anchor['sensor_position'].tolist())
    return report,preview


def score_context_action(action,anchors,hfov,vfov,cfg):
    best=None;best_preview=None;eligible=0
    indices=np.unique(np.linspace(0,len(action['waypoints'])-1,min(5,len(action['waypoints']))).astype(int))
    for index in indices:
        w=action['waypoints'][index]
        for anchor in anchors:
            pair=context_pair(anchor,w,hfov,vfov,cfg)
            if pair is None:continue
            eligible+=1;report,preview=pair;report['photo_index']=int(index)
            if best is None or (report['score'],report['common_points'])>(best['score'],best['common_points']):
                best=report;best_preview=preview
    action['reliable_context_link']=best
    action['context_link_pairs_tested']=eligible
    action['same_height_old_overlap_preference']=best['score'] if best else 0.
    action['_context_preview']=best_preview
