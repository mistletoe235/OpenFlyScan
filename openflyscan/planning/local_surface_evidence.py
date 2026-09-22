"""Native-risk surface evidence only. Never promote context into extra defects.

Normals must agree in at least two distinct cached image views. This is a
prediction consistency check, not independent ground truth or calibrated trust.
"""
import numpy as np
from scipy.spatial import cKDTree
from openflyscan.planning.analyze_frozen_region_observations import normal_fit, normal_modes


def view_support(points, normals, cloud):
    depth_sets=[set() for _ in points];normal_sets=[set() for _ in points]
    h,w=map(int,cloud['target_hw'])
    for vi,stem in enumerate(cloud['stems'].astype(str)):
        pose=cloud['camera_poses'][vi];pc=(points-pose[:3,3])@pose[:3,:3]
        uv=pc@cloud['intrinsics'][vi].T;uv=uv[:,:2]/np.maximum(uv[:,2,None],1e-8)
        ids=np.flatnonzero((pc[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h))
        if not len(ids):continue
        yi=np.abs(cloud['dense_y'][:,None]-uv[ids,1]).argmin(0)
        xi=np.abs(cloud['dense_x'][:,None]-uv[ids,0]).argmin(0)
        xyz=cloud['dense_maps'][vi,yi,xi];obs=(xyz-pose[:3,3])@pose[:3,:3]
        tol=np.maximum(1.,.03*np.abs(pc[ids,2]))
        ok=(np.abs(obs[:,2]-pc[ids,2])<=tol)&(np.linalg.norm(xyz-points[ids],axis=1)<=2*tol)
        for idx,y,x in zip(ids[ok],yi[ok],xi[ok]):
            depth_sets[idx].add(stem)
            patch=cloud['dense_maps'][vi,max(0,y-2):y+3,max(0,x-2):x+3].reshape(-1,3)
            fit=normal_fit(patch,pose[:3,3])
            if fit is not None and fit[1]>=.5 and fit[0]@normals[idx]>=np.cos(np.radians(30)):
                normal_sets[idx].add(stem)
    return np.array([len(s) for s in depth_sets]),np.array([len(s) for s in normal_sets])


def surface_groups(normals, valid):
    ids=np.flatnonzero(valid);groups=[]
    if not len(ids):return groups
    for mode in normal_modes(normals[ids],np.ones(len(ids))):
        members=ids[mode['members']]
        if len(members)<3:continue
        n=np.asarray(mode['normal']);nz=abs(n[2])
        groups.append(dict(indices=members.tolist(),normal=n.tolist(),count=len(members),
            kind='roof' if nz>=.75 else ('side' if nz<.45 else 'slope')))
    return groups


def prepare_surface_evidence(area,points,clouds):
    raw=[];ns=[];good=[];depth=[];agree=[];source=[]
    for r in area['members']:
        s=r['surface'];p=s['xyz'];n=s['normals'];d,a=view_support(p,n,clouds[r['diag']['chunk']])
        raw.extend(p);ns.extend(n);good.extend((s['normal_quality']>=.5)&(s['conf_percentile']>=.25))
        depth.extend(d);agree.extend(a);source.extend([r['rank']]*len(p))
    distance,ids=cKDTree(np.array(raw)).query(points)
    if np.max(distance)>1e-5:raise ValueError('Scoring samples are not native risk points')
    normals=np.array(ns)[ids];base_good=np.array(good)[ids];depth=np.array(depth)[ids];agree=np.array(agree)[ids]
    valid=base_good&(agree>=2);groups=surface_groups(normals,valid)
    eligible=np.zeros(len(points),bool)
    for g in groups:eligible[g['indices']]=True
    area['_point_normals']=normals;area['_point_normal_valid']=eligible
    # Unqualified points remain unknown (zero incremental utility), NOT good.
    return dict(area_id=area['id'],ranks=area['ranks'],xyz=points.tolist(),normals=normals.tolist(),
        source_rank=np.array(source)[ids].tolist(),normal_quality_pass=base_good.tolist(),
        depth_consistent_views=depth.tolist(),normal_consistent_views=agree.tolist(),
        reliable=valid.tolist(),scored=eligible.tolist(),groups=groups,
        unknown_count=int((~eligible).sum()),total_count=len(points),
        supported_side_points=sum(g['count'] for g in groups if g['kind']=='side'))


def rescore_local_candidates(areas,candidates,chosen,samples,clouds,manifest_index,hfov,vfov,cfg,tree,
                             old_observations,action_score):
    candidates=[dict(c) for c in candidates];records=[]
    for area in areas:
        if area['task_type']!='local_patch':continue
        points=samples[area['id']][0];record=prepare_surface_evidence(area,points,clouds);records.append(record)
        history=old_observations(area,points,clouds,hfov,vfov,manifest_index)
        for c in candidates:
            if c['area_id']!=area['id']:continue
            # First-strip selection/metadata must remain from the frozen baseline.
            probe=dict(c);action_score(probe,area,points,history,hfov,vfov,cfg,tree)
            for key in ['_new_q','_new_d','_old_q','_old_d']:c[key]=probe[key]
            c['_surface_groups']=record['groups']
        print('Surface evidence',area['ranks'],'scored',int(sum(record['scored'])),'/',len(points),
              'groups',[(g['kind'],g['count']) for g in record['groups']],flush=True)
    lookup={c['id']:c for c in candidates}
    return candidates,[lookup[c['id']] for c in chosen],dict(regions=records,
        used_GS_or_SfM=False,protocol='Native risk only; per-point normal+confidence and >=2 distinct agreeing image normals; modes >=3 voxel samples',
        limitations=['Consistent predictions may still be wrong.','Unknown normals do not mean good reconstruction.',
                    'No side defect is inferred from context outside original risk samples.'])
