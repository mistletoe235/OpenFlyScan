"""Recover surface/observation evidence WITHOUT consulting GS or moving risks.

Confidence is reported raw and as within-view percentile, never a correctness
probability. Surface normals, reprojection tests and seams are diagnostics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def unit(x):return x/max(float(np.linalg.norm(x)),1e-12)


def normal_fit(points,camera):
    points=np.asarray(points);points=points[np.isfinite(points).all(1)]
    if len(points)<6:return None
    center=np.median(points,axis=0);dist=np.linalg.norm(points-center,axis=1)
    points=points[dist<=max(np.quantile(dist,.85)*1.5,.05)]
    if len(points)<6:return None
    values,vectors=np.linalg.eigh(np.cov((points-center).T));values=np.maximum(values,0)
    if values[1]<1e-8:return None
    n=vectors[:,0]
    if n@(camera-center)<0:n=-n
    planarity=float(np.clip(1-values[0]/max(values[1],1e-12),0,1))
    support=float(np.clip(values[1]/max(values[2],1e-12)/.15,0,1))
    return n,planarity*support,float(np.sqrt(values[0]))


def normal_modes(normals,weights):
    """Angular modes keep roof and facade normals separate; no averaging across corners."""
    normals=np.asarray(normals);weights=np.asarray(weights)
    remaining=set(range(len(normals)));groups=[]
    while remaining:
        choices=[]
        for i in sorted(remaining):
            ids=[j for j in sorted(remaining) if normals[i]@normals[j]>=np.cos(np.radians(30))]
            choices.append((float(weights[ids].sum()),i,ids))
        mass,i,ids=max(choices,key=lambda x:(x[0],-x[1]));remaining.difference_update(ids)
        n=unit(np.sum(normals[ids]*weights[ids,None],axis=0))
        groups.append(dict(normal=n.tolist(),weight=mass,members=ids,
            tilt_from_vertical_deg=float(np.degrees(np.arccos(np.clip(n[2],-1,1))))))
    total=max(sum(g['weight'] for g in groups),1e-9)
    for g in groups:g['fraction']=g['weight']/total
    return groups


def dense_samples(cloud,view,pixels):
    """Original Head patch centres (14*p+7) -> retained dense grid."""
    pixels=np.asarray(pixels,int)
    yi=np.abs(cloud['dense_y'][:,None]-(14*pixels[:,0]+7)).argmin(0)
    xi=np.abs(cloud['dense_x'][:,None]-(14*pixels[:,1]+7)).argmin(0)
    return yi,xi


def reproject_consistency(points,cloud):
    rows=[];h,w=map(int,cloud['target_hw'])
    for vi,stem in enumerate(cloud['stems'].astype(str)):
        pose=cloud['camera_poses'][vi];K=cloud['intrinsics'][vi]
        pc=(points-pose[:3,3])@pose[:3,:3];uv=pc@K.T;uv=uv[:,:2]/np.maximum(uv[:,2,None],1e-8)
        inside=(pc[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        ids=np.flatnonzero(inside)
        if not len(ids):continue
        yi=np.abs(cloud['dense_y'][:,None]-uv[ids,1]).argmin(0)
        xi=np.abs(cloud['dense_x'][:,None]-uv[ids,0]).argmin(0)
        xyz=cloud['dense_maps'][vi,yi,xi];obs=(xyz-pose[:3,3])@pose[:3,:3]
        dz=obs[:,2]-pc[ids,2];error=np.linalg.norm(xyz-points[ids],axis=1)
        # Relative-depth and sampled-pixel tolerance, independent of GS labels.
        tol=np.maximum(1.,.03*np.abs(pc[ids,2]))
        consistent=(np.abs(dz)<=tol)&(error<=2*tol)
        rows.append(dict(stem=stem,in_frustum=len(ids),consistent=int(consistent.sum()),
                         consistent_fraction=float(consistent.mean()),
                         occluded_proxy_fraction=float(np.mean(dz < -tol)),
                         free_space_conflict_fraction=float(np.mean(dz>tol)),
                         xyz_residual_median_m=float(np.median(error))))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['selection','geometry','output']:p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);tic=time.time()
    payload=a.selection.read_bytes();selection=json.loads(payload);recovery=json.loads((a.geometry/'manifest.json').read_text())
    assert hashlib.sha256(payload).hexdigest()==recovery['input_selection_sha256']
    clouds={r['chunk']:dict(np.load(r['path'],allow_pickle=False)) for r in recovery['chunks']}
    diagnostics=[];samples={}
    for rank,r in enumerate(selection['selected'],1):
        cloud=clouds[r['chunk']];stems=list(cloud['stems'].astype(str));patches=[];viewrows=[];seams=[]
        raw_values=[];allnormals=[];normal_weights=[]
        for ev in r['image_evidence']:
            stem=ev['stem'];vi=stems.index(stem);pixels=np.asarray(ev['patch_pixels_yx'],int)
            yi,xi=dense_samples(cloud,vi,pixels)
            cf=cloud['dense_conf'][vi];valid=cf[np.isfinite(cf)];cdf=np.sort(valid)
            patch_c=cloud['conf'][vi,pixels[:,0],pixels[:,1]]
            pct=np.searchsorted(cdf,patch_c,side='right')/max(len(cdf),1)
            # Exact 14-stride xyz retains correspondence to original Head risks.
            xyz=cloud['maps'][vi,pixels[:,0],pixels[:,1]];camera=cloud['camera_poses'][vi,:3,3]
            sensor=cloud['sensor_poses'][vi,:3,3];K=cloud['intrinsics'][vi]
            row_normals=[];local_quality=[]
            for j,(y,x) in enumerate(zip(yi,xi)):
                neighborhood=cloud['dense_maps'][vi,max(0,y-2):y+3,max(0,x-2):x+3].reshape(-1,3)
                fit=normal_fit(neighborhood,camera)
                n,quality,noise=(fit if fit is not None else (np.zeros(3),0.,0.))
                rec=dict(xyz=xyz[j].tolist(),normal=n.tolist(),normal_quality=quality,
                         surface_thickness_m=noise,conf_raw=float(patch_c[j]),conf_percentile=float(pct[j]),
                         stem=stem,pixel_yx=pixels[j].tolist(),camera=camera.tolist(),sensor_camera=sensor.tolist(),
                         distance_m=float(np.linalg.norm(xyz[j]-camera)),
                         incidence_cos=float(max(0,n@unit(camera-xyz[j]))) if quality else None)
                patches.append(rec);raw_values.append(float(patch_c[j]));local_quality.append(quality)
                if quality>.5:
                    allnormals.append(n);normal_weights.append(quality*(.2+.8*pct[j]))
                    row_normals.append(n)
            # Same image pixel in other replayed windows; tests alignment, not new independent views.
            for other_id,other in clouds.items():
                if other_id==r['chunk'] or stem not in other['stems']:continue
                oi=list(other['stems'].astype(str)).index(stem)
                other_xyz=other['maps'][oi,pixels[:,0],pixels[:,1]]
                errors=np.linalg.norm(other_xyz-xyz,axis=1)
                seams.append(dict(stem=stem,other_chunk=other_id,median_m=float(np.median(errors)),
                                  p90_m=float(np.quantile(errors,.9)),points=len(errors)))
            viewrows.append(dict(stem=stem,patches=len(pixels),center=np.median(xyz,axis=0).tolist(),
                camera=camera.tolist(),sensor_camera=sensor.tolist(),
                confidence_raw_median=float(np.median(patch_c)),confidence_percentile_median=float(np.median(pct)),
                usable_normal_fraction=float(np.mean(np.array(local_quality)>.5)),
                median_distance_m=float(np.median(np.linalg.norm(xyz-camera,axis=1))),
                approximate_model_pixel_footprint_m=float(np.median(np.linalg.norm(xyz-camera,axis=1))/K[0,0]),
                rgb_patch_std=float(np.mean(cloud['rgb'][vi,pixels[:,0],pixels[:,1]].std(0)))))
        xyz=np.array([p['xyz'] for p in patches]);center=np.median(xyz,axis=0)
        groups=normal_modes(allnormals,normal_weights) if allnormals else []
        if groups:
            n=np.array(groups[0]['normal']);dominant=groups[0]['fraction']
            surface_type='roof_like' if n[2]>.75 else ('facade_like' if abs(n[2])<.45 else 'sloped_or_mixed')
        else:n=np.array([0.,0.,1.]);dominant=0.;surface_type='unresolved'
        # A fitted point average is descriptive only, not a replacement of frozen risk xyz.
        reproj=reproject_consistency(xyz[np.linspace(0,len(xyz)-1,min(len(xyz),40)).astype(int)],cloud)
        stable=[v for v in reproj if v['consistent_fraction']>=.5 and v['consistent']>=3]
        low=float(np.mean(np.array([p['conf_percentile'] for p in patches])<.25))
        view_centers=np.array([v['center'] for v in viewrows]);spread=float(np.quantile(np.linalg.norm(view_centers-center,axis=1),.9))
        h,w=cloud['target_hw'];Ks=cloud['intrinsics'];hfov=float(np.median(2*np.arctan(w/(2*Ks[:,0,0]))));vfov=float(np.median(2*np.arctan(h/(2*Ks[:,1,1]))))
        flags=[]
        if dominant<.65:flags.append('multiple_or_unreliable_surface_normals')
        if len(stable)<2:flags.append('fewer_than_two_geometrically_consistent_views')
        if low>.5:flags.append('most_risk_patches_in_low_confidence_quartile')
        seam=float(np.median([v['median_m'] for v in seams])) if seams else None
        if seam is not None and seam>r['radius']:flags.append('cross_window_location_disagreement_exceeds_region_radius')
        inc=[p['incidence_cos'] for p in patches if p['incidence_cos'] is not None]
        diagnostic=dict(rank=rank,chunk=int(r['chunk']),frozen_xyz=r['xyz'],frozen_score=r['score'],radius_m=r['radius'],
            evidence_xyz_median=center.tolist(),evidence_xyz_quantiles=np.quantile(xyz,[.1,.9],axis=0).tolist(),
            evidence_view_count=len(viewrows),patch_count=len(patches),view_center_spread_p90_m=spread,
            conf_raw_median=float(np.median(raw_values)),conf_percentile_median=float(np.median([p['conf_percentile'] for p in patches])),
            low_confidence_patch_fraction=low,surface_type=surface_type,dominant_normal=n.tolist(),
            dominant_normal_fraction=dominant,normal_modes=groups,views=viewrows,
            reprojection_consistency=reproj,consistent_view_count=len(stable),
            incidence_cos_median=float(np.median(inc)) if inc else None,
            cross_window_disagreement=seams,cross_window_median_m=seam,
            horizontal_fov_deg=float(np.degrees(hfov)),vertical_fov_deg=float(np.degrees(vfov)),
            flags=flags)
        diagnostics.append(diagnostic)
        np.savez_compressed(a.output/f'rank_{rank:02d}_surface.npz',xyz=xyz,
            normals=np.array([p['normal'] for p in patches]),normal_quality=np.array([p['normal_quality'] for p in patches]),
            conf_raw=np.array(raw_values),conf_percentile=np.array([p['conf_percentile'] for p in patches]),
            stems=np.array([p['stem'] for p in patches]),pixels=np.array([p['pixel_yx'] for p in patches]),
            cameras=np.array([p['camera'] for p in patches]),sensor_cameras=np.array([p['sensor_camera'] for p in patches]))
        print(rank,surface_type,'normal fraction',round(dominant,2),'conf percentile',round(diagnostic['conf_percentile_median'],2),
              'consistent views',len(stable),'flags',flags,flush=True)
    result=dict(status='complete',selection_sha256=hashlib.sha256(payload).hexdigest(),selection_unchanged=True,
        geometry_manifest_sha256=hashlib.sha256((a.geometry/'manifest.json').read_bytes()).hexdigest(),
        used_GS_or_SfM=recovery.get('used_GS_or_SfM',False),used_GS_render_or_quality=False,
        confidence_semantics='Raw model confidence plus within-view percentiles, NOT calibrated probability',
        consistency_semantics='Projected point/depth compatibility with tolerance max(1m,3% depth); not independently verified geometry',
        cross_window_scope=f"Only {len(recovery['chunks'])} selected windows; repeated image pixels are not independent observing views",
        regions=diagnostics,seconds=time.time()-tic)
    (a.output/'observations.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
