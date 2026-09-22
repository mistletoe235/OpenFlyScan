"""Source-view repair witnesses from frozen inference, not directional GS labels.

Each region has fixed risk mass split over its source image evidence. Multiple
images do not multiply the Head score. Texture is a soft support proxy, never
a water/roof semantic label. Missing semantic evidence remains explicit.
"""
from collections import Counter,defaultdict
import cv2
import numpy as np
from openflyscan.planning.plan_frozen_region_scans import frustum_contains


def patch_texture(image_path,hw):
    rgb=cv2.imread(str(image_path))
    if rgb is None:raise ValueError(f'Cannot load observed RGB: {image_path}')
    h,w=map(int,hw);rgb=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    energy=np.sqrt(gx*gx+gy*gy).reshape(h//14,14,w//14,14).mean((1,3))
    # Image-relative percentile mitigates exposure scale; not a matchability
    # probability, and textured water can still pass.
    pct=np.searchsorted(np.sort(energy.ravel()),energy)/max(energy.size-1,1)
    neighborhood=cv2.blur(pct.astype(np.float32),(3,3),borderType=cv2.BORDER_REFLECT)
    return .5*pct+.5*neighborhood


def share_duplicate_risk(rows,policy):
    """One image patch is not independent evidence in overlapping regions.

    Keep alternative local geometries, but share at most the largest allocated
    patch mass. No first-region-wins deletion; invariant to input order.
    This does not deduplicate distinct images of one physical surface.
    """
    groups=defaultdict(list)
    for r in rows:groups[(r['stem'],*r['pixel_yx'])].append(r)
    for group in groups.values():
        total=sum(r['risk_weight'] for r in group);cap=max(r['risk_weight'] for r in group)
        for r in group:
            r['pre_duplicate_risk_weight']=r['risk_weight']
            r['risk_weight']*=cap/max(total,1e-12)
            r['repair_weight']=r['risk_weight']*(policy.content_floor+(1-policy.content_floor)*r['readiness'])
    return sum(len(group)>1 for group in groups.values())


def make_tokens(regions,clouds,manifest,policy):
    rows=[];textures={};region_records=[]
    for region in regions:
        s=region['surface'];rank=region['rank'];source_counts=Counter(s['stems'].astype(str));region_rows=[]
        for i,(stem,pixel) in enumerate(zip(s['stems'].astype(str),s['pixels'])):
            hw=clouds[region['diag']['chunk']]['target_hw']
            if stem not in textures:textures[stem]=patch_texture(manifest[stem]['image'],hw)
            texture=float(textures[stem][tuple(pixel)])
            point=s['xyz'][i];delta=s['cameras'][i]-point;dist=float(np.linalg.norm(delta));direction=delta/max(dist,1e-8)
            # High risk + unreliable geometry stays present. It has broad
            # direction uncertainty, not a mandatory rejection.
            normal_reliability=float(np.clip(s['normal_quality'][i],0,1)*np.clip(s['conf_percentile'][i],0,1))
            normal=np.array(s['normals'][i]);normal/=max(np.linalg.norm(normal),1e-8)
            loc=float(np.exp(-region['uncertainty_m']/max(region['frozen']['radius'],1.)))
            readiness=loc*texture
            row=dict(rank=rank,xyz=point.tolist(),source_direction=direction.tolist(),source_distance_m=dist,
                normal=normal.tolist(),normal_reliability=normal_reliability,confidence_percentile=float(s['conf_percentile'][i]),
                image_texture_percentile=texture,localization_factor=loc,readiness=readiness,
                uncertainty_m=region['uncertainty_m'],stem=stem,pixel_yx=pixel.tolist(),
                unnormalized_weight=1/max(source_counts[stem],1))
            region_rows.append(row)
        if not region_rows:
            region_records.append(dict(rank=rank,status='no_native_evidence',witnesses=0));continue
        total=sum(r['unnormalized_weight'] for r in region_rows)
        for r in region_rows:
            r['risk_weight']=region['risk']*r.pop('unnormalized_weight')/total
        rows.extend(region_rows);region_records.append(dict(rank=rank,witnesses=len(region_rows),
            risk_mass=sum(r['risk_weight'] for r in region_rows),head_risk=region['risk'],
            mean_readiness=float(np.mean([r['readiness'] for r in region_rows])),
            mean_confidence=float(np.mean([r['confidence_percentile'] for r in region_rows])),
            semantic_class='unknown',status='available'))
    duplicate_groups=share_duplicate_risk(rows,policy)
    for record in region_records:
        record['risk_mass_after_duplicate_sharing']=sum(r['risk_weight'] for r in rows if r['rank']==record['rank'])
    def array(key):return np.array([r[key] for r in rows])
    arrays={key:array(key) for key in ['rank','xyz','source_direction','source_distance_m','normal','normal_reliability',
        'confidence_percentile','image_texture_percentile','localization_factor','readiness','uncertainty_m','risk_weight','repair_weight']}
    return arrays,dict(regions=region_records,tokens=rows,used_GS_or_SfM=False,duplicate_image_patch_groups=duplicate_groups,
        protocol='Image-conditioned witnesses; Head risk mass normalized per region; texture not a semantic class',
        limitations=['These are overlapping image witnesses, not unique physical defect surfaces.',
                    'Textured water/clear roads can still be selected; no semantic model is claimed.',
                    'Source direction is where risk was observed, not an independent direction error prediction.'])


def candidate_utilities(candidates,tokens,hfov,vfov,cfg,policy,tree):
    points=tokens['xyz'];normals=tokens['normal'];reliability=tokens['normal_reliability'];n=len(points)
    geometry=[];repair=[];pair=[];azimuth=[];explore=[]
    for candidate in candidates:
        vis=[];rays=[];qualities=[];matches=[];ranges=[]
        for wp in candidate['waypoints']:
            pos=np.array(wp['position_enu_m']);delta=pos-points;distance=np.maximum(np.linalg.norm(delta,axis=1),1e-8)
            direction=delta/distance[:,None];inside=frustum_contains(points,pos,wp['aircraft_yaw_deg'],wp['gimbal_pitch_deg'],hfov,vfov)
            # Taper uncertain FOV edges with fixed localization hypotheses.
            pad=tokens['uncertainty_m'];fraction=inside.astype(float)
            for xy in [(1,0),(-1,0),(0,1),(0,-1)]:
                shifted=points+np.column_stack([pad*xy[0],pad*xy[1],np.zeros(n)])
                fraction+=frustum_contains(shifted,pos,wp['aircraft_yaw_deg'],wp['gimbal_pitch_deg'],hfov,vfov)
            fraction/=5
            incidence=reliability*np.maximum(np.sum(direction*normals,axis=1),0)+(1-reliability)*.5
            resolution=np.minimum(1.,(tokens['source_distance_m']/distance/policy.desired_linear_resolution_gain)**2)
            q=fraction*incidence*resolution
            # Same observed-cloud ray proxy as existing code, not mesh visibility.
            ids=np.flatnonzero(inside)
            if len(ids):
                raypoints=pos+(points[ids]-pos)[:,None]*np.linspace(.1,.85,8)[None,:,None]
                blocked=tree.query(raypoints.reshape(-1,3))[0].reshape(len(ids),-1).min(1)<.6
                q[ids]*=np.where(blocked,.3,1.)
            angle=np.degrees(np.arccos(np.clip(np.sum(direction*tokens['source_direction'],axis=1),-1,1)))
            bandwidth=policy.source_direction_bandwidth_deg*(2-reliability)
            match=np.exp(-.5*(angle/bandwidth)**2)
            vis.append(inside);rays.append(direction);qualities.append(q);matches.append(match);ranges.append(distance)
        vis=np.asarray(vis);rays=np.asarray(rays);q=np.asarray(qualities);match=np.asarray(matches)
        geometric=np.zeros(n);observed=np.zeros(n);pairscore=np.zeros(n)
        for i in range(len(q)-1):
            angle=np.degrees(np.arccos(np.clip(np.einsum('nk,jnk->jn',rays[i],rays[i+1:]),-1,1)))
            baseline=np.minimum(1.,np.sin(np.radians(angle))/np.sin(np.radians(cfg.useful_pair_angle_deg)))
            baseline*=(angle>=cfg.min_pair_angle_deg)&(angle<=cfg.max_pair_angle_deg)
            quality=np.sqrt(q[i]*q[i+1:])*baseline
            geometric=np.maximum(geometric,quality.max(0))
            directional=quality*np.sqrt(match[i]*match[i+1:])
            observed=np.maximum(observed,directional.max(0))
            # Old one-view witness + proposed strip contribution (not a full
            # replacement for original global joint-pair baseline).
        geometric*=vis.sum(0)>=3;observed*=vis.sum(0)>=3
        for i in range(len(q)):
            angle=np.degrees(np.arccos(np.clip(np.sum(rays[i]*tokens['source_direction'],axis=1),-1,1)))
            baseline=np.minimum(1.,np.sin(np.radians(angle))/np.sin(np.radians(cfg.useful_pair_angle_deg)))
            baseline*=(angle>=cfg.min_pair_angle_deg)&(angle<=cfg.max_pair_angle_deg)
            pairscore=np.maximum(pairscore,np.sqrt(q[i]*tokens['confidence_percentile'])*baseline)
        pairscore=np.maximum(pairscore,geometric)
        # Five viewing sectors for a common-pool diversity control. This is an
        # internal rule baseline, not an implementation of a named paper.
        mean_ray=rays.mean(0);bins=np.where(mean_ray[:,2]>=np.sin(np.radians(67.5)),0,
            1+(np.floor((np.degrees(np.arctan2(mean_ray[:,0],mean_ray[:,1]))+45)%360/90)).astype(int))
        sector=np.zeros((n,5));sector[np.arange(n),bins]=geometric
        geometry.append(geometric);repair.append(observed);pair.append(pairscore);azimuth.append(sector.ravel())
        weight=observed*tokens['risk_weight'];support=float(weight@tokens['readiness']/max(weight.sum(),1e-9))
        explore.append(support<policy.exploration_support_threshold)
    return dict(geometry=np.asarray(geometry),repair=np.asarray(repair),source_pair=np.asarray(pair),
        five_sector=np.asarray(azimuth)),np.asarray(explore)
