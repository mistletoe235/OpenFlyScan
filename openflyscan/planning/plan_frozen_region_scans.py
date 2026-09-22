"""Preview-only continuous retake scans from frozen risks and sensor geometry.

No GS, teacher, simulator truth, or scene-specific coordinates are accepted.
This is a geometric planning prototype, NOT a flight-authorized mission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from openflyscan.planning.generate_candidate_routes import yaw_pitch_to_target, path_length, make_transit_leg


@dataclass
class Config:
    image_aspect: float = 4 / 3
    forward_overlap: float = .80
    side_overlap: float = .70
    interval_s: float = 2.0
    speed_mps: float = 4.0
    oblique_pitch_deg: float = -55.0
    min_surface_gap_m: float = 20.0
    point_clearance_proxy_m: float = 12.0
    margin_rmse_multiplier: float = 2.0
    max_local_climb_m: float = 60.0
    min_cell_points: int = 8


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def camera_basis(yaw, pitch):
    """ENU, image right/down/forward, DJI yaw clockwise from north."""
    y, p = np.radians([yaw, pitch])
    f = np.array([np.sin(y)*np.cos(p), np.cos(y)*np.cos(p), np.sin(p)])
    r = np.array([np.cos(y), -np.sin(y), 0.])
    return np.column_stack([r, np.cross(f, r), f])


def frustum_contains(points, position, yaw, pitch, hfov, vfov):
    q = (np.asarray(points)-position) @ camera_basis(yaw, pitch)
    return ((q[:, 2] > 0) & (np.abs(q[:, 0]) <= q[:, 2]*np.tan(hfov/2))
            & (np.abs(q[:, 1]) <= q[:, 2]*np.tan(vfov/2)))


def footprint(position, yaw, pitch, z, hfov, vfov):
    rays = np.array([[x*np.tan(hfov/2), y*np.tan(vfov/2), 1]
                     for x, y in [(-1,-1),(1,-1),(1,1),(-1,1)]]) @ camera_basis(yaw,pitch).T
    if np.any(rays[:,2] >= -1e-7):
        raise ValueError('Camera frustum crosses horizon; cannot use a planar footprint')
    return position + rays * ((z-position[2])/rays[:,2])[:,None]


class Surface:
    def __init__(self, xyz, cell, min_points):
        self.xyz = xyz
        self.tree = cKDTree(xyz[:,:2])
        self.cell = cell
        self.origin = np.min(xyz[:,:2], axis=0)
        keys = np.floor((xyz[:,:2]-self.origin)/cell).astype(int)
        unique, inv = np.unique(keys, axis=0, return_inverse=True)
        order = np.argsort(inv); groups = np.split(order, np.cumsum(np.bincount(inv))[:-1])
        self.cells = {}
        for key, ids in zip(unique, groups):
            if len(ids) >= min_points:
                z = np.quantile(xyz[ids,2], [.2,.5,.8,.95])
                self.cells[tuple(key)] = z

    def nearby(self, xy, radius):
        return self.xyz[self.tree.query_ball_point(xy,radius)]

    def component(self, xy, surface_z, tolerance, max_radius):
        """Grow a similar-height upper surface; bounded by sensor-derived footprint.

        This is a scan envelope, not a semantic roof segmentation or surface ID.
        """
        origin_key = np.floor((xy-self.origin)/self.cell).astype(int)
        candidates = [k for k in self.cells if np.linalg.norm(
            self.origin+(np.array(k)+.5)*self.cell-xy) <= max_radius]
        allowed = {k for k in candidates if abs(self.cells[k][2]-surface_z) <= tolerance}
        if not allowed:
            return np.empty((0,2))
        start = min(allowed, key=lambda k: np.linalg.norm(np.array(k)-origin_key))
        if np.linalg.norm(self.origin+(np.array(start)+.5)*self.cell-xy) > 3*self.cell:
            return np.empty((0,2))
        seen={start}; stack=[start]
        while stack:
            x,y=stack.pop()
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                k=(x+dx,y+dy)
                if k in allowed and k not in seen:
                    seen.add(k);stack.append(k)
        return self.origin+(np.array(sorted(seen))+.5)*self.cell


def region_envelopes(selected, records, surface, hfov, vfov, cfg):
    by_stem={r['stem']:r for r in records}
    cams=np.array([r['enu'] for r in records])
    result=[]
    for rank,r in enumerate(selected,1):
        seed=np.array(r['xyz'],float); fit=float(r['GPS_fit_rmse']); radius=float(r['radius'])
        margin=max(radius,cfg.margin_rmse_multiplier*fit)
        near=surface.nearby(seed[:2],max(2*margin,4*surface.cell))
        if len(near)<20:
            result.append(dict(ranks=[rank],status='review_only',reason='insufficient_local_geometry',
                               seed=seed.tolist(),score=r['score']))
            continue
        zq=np.quantile(near[:,2],[.2,.5,.8,.95]); top=float(zq[2])
        # The original risk xyz is preserved; the aiming surface is separately inferred.
        evidence=[by_stem[e['stem']] for e in r['image_evidence'] if e['stem'] in by_stem]
        old_z=float(np.median([e['enu'][2] for e in evidence])) if evidence else float(np.median(cams[:,2]))
        wider=surface.nearby(seed[:2],max(60.,6*margin))
        reference_z=float(np.quantile(wider[:,2],.35))
        gap=max(cfg.min_surface_gap_m,old_z-reference_z)
        nominal_footprint=2*gap*np.tan(min(hfov,vfov)/2)
        support=surface.component(seed[:2],top,max(2*fit,3.),max(2*margin,.75*nominal_footprint))
        lo=seed[:2]-margin;hi=seed[:2]+margin
        if len(support):
            lo=np.minimum(lo,support.min(0)-surface.cell/2)
            hi=np.maximum(hi,support.max(0)+surface.cell/2)
        visibility=[]
        for e in evidence:
            yaw,pitch,_=e['camera_ypr_deg']
            visibility.append(bool(frustum_contains(seed[None],np.array(e['enu']),yaw,pitch,hfov,vfov)[0]))
        result.append(dict(ranks=[rank],status='proposed',seed=seed.tolist(),score=float(r['score']),
                           bbox_xy=[lo.tolist(),hi.tolist()],aim_z=top,
                           original_risk_z=float(seed[2]),local_z_quantiles=zq.tolist(),
                           uncertainty_padding_m=margin,fit_rmse_m=fit,reference_gap_m=gap,
                           previous_camera_z=old_z,evidence_stems=[e['stem'] for e in evidence],
                           existing_evidence_frustum_hits=sum(visibility),
                           surface_extension_cells=len(support),
                           geometry_conflict_m=float(max(zq[0]-seed[2],seed[2]-zq[3],0)),
                           water_semantics='unknown; no semantic confidence in this cache'))
    return result


def merge_envelopes(regions):
    """Merge overlapping planning envelopes, not point identities or risk scores."""
    groups=[]
    for region in regions:
        if region['status']!='proposed':
            continue
        matching=[]
        for i,g in enumerate(groups):
            if any(np.all(np.minimum(region['bbox_xy'][1],r['bbox_xy'][1]) >=
                          np.maximum(region['bbox_xy'][0],r['bbox_xy'][0])) and
                   abs(region['aim_z']-r['aim_z']) <= max(region['fit_rmse_m'],r['fit_rmse_m'])*3
                   for r in g):
                matching.append(i)
        combined=[region]
        for i in reversed(matching):combined.extend(groups.pop(i))
        groups.append(combined)
    return sorted(groups,key=lambda g:min(r['ranks'][0] for r in g))


def capture_record(position,target,yaw_hint,mode,cfg):
    yaw,pitch=yaw_pitch_to_target(position.tolist(),target.tolist())
    if np.linalg.norm(position[:2]-target[:2])<1e-6:yaw=yaw_hint % 360
    return dict(position_enu_m=position.tolist(),target_enu_m=target.tolist(),
                aircraft_yaw_deg=yaw,gimbal_pitch_deg=pitch,capture_view=mode,capture=True,
                speed_mps=cfg.speed_mps,minimum_interval_s=cfg.interval_s)


def make_scan(bounds,z,flight_z,hfov,vfov,cfg):
    lo,hi=np.array(bounds);center=(lo+hi)/2
    segments=[]
    for mode,azimuth in [('NADIR',0),('NORTH_OBLIQUE',0),('EAST_OBLIQUE',90),
                         ('SOUTH_OBLIQUE',180),('WEST_OBLIQUE',270)]:
        pitch=-90 if mode=='NADIR' else cfg.oblique_pitch_deg
        # azimuth describes the camera-facing direction, not aircraft position.
        yaw=float(azimuth);basis=camera_basis(yaw,pitch);along=basis[:2,0];cross=np.array([-along[1],along[0]])
        corners=np.array([[lo[0],lo[1]],[lo[0],hi[1]],[hi[0],lo[1]],[hi[0],hi[1]]])-center
        amin,amax=np.min(corners@along),np.max(corners@along)
        cmin,cmax=np.min(corners@cross),np.max(corners@cross)
        dz=flight_z-z
        heading=np.array([math.sin(math.radians(yaw)),math.cos(math.radians(yaw))])
        offset=np.zeros(2) if mode=='NADIR' else -heading*dz/math.tan(math.radians(-pitch))
        pos=np.r_[center+offset,flight_z]
        fp=footprint(pos,yaw,pitch,z,hfov,vfov)[:,:2]-center
        # Use the inscribed symmetric footprint around optical-axis target.
        widths=np.array([2*min(abs(np.min(fp@v)),abs(np.max(fp@v))) for v in [along,cross]])
        along_spacing=min(widths[0]*(1-cfg.forward_overlap),cfg.speed_mps*cfg.interval_s)
        cross_spacing=widths[1]*(1-cfg.side_overlap)
        if min(along_spacing,cross_spacing)<=0:raise ValueError('Invalid footprint spacing')
        # Edge footprint margin covers the envelope; retain nonzero parallax travel.
        aextent=max((amax-amin)-.6*widths[0],2*along_spacing)
        cextent=max((cmax-cmin)-.6*widths[1],0.)
        xs=np.linspace(-aextent/2,aextent/2,max(3,math.ceil(aextent/along_spacing)+1))
        ys=np.linspace(-cextent/2,cextent/2,max(1,math.ceil(cextent/cross_spacing)+1))
        for row,y in enumerate(ys):
            targets=[np.r_[center+along*x+cross*y,z] for x in (xs if row%2==0 else xs[::-1])]
            shots=[capture_record(np.r_[t[:2]+offset,flight_z],t,yaw,mode,cfg) for t in targets]
            # Actual spacing may be smaller after linspace: lower speed, not cadence.
            speed=min(cfg.speed_mps,min(np.linalg.norm(np.array(b['position_enu_m'])-
                       a['position_enu_m']) for a,b in zip(shots,shots[1:]))/cfg.interval_s)
            for shot in shots:shot['speed_mps']=speed
            segments.append(dict(mode=mode,row=row,waypoints=shots,
                                 footprint_widths_m=widths.tolist(),
                                 designed_forward_overlap=1-(aextent/(len(xs)-1))/widths[0],
                                 designed_side_overlap=1-(cextent/(len(ys)-1))/widths[1] if len(ys)>1 else None))
    return segments


def cloud_clearance(segments,surface,cfg):
    """Observed upper-surface proxy only, never treats empty cells as safe."""
    samples=[]
    for s in segments:
        for a,b in zip(s['waypoints'],s['waypoints'][1:]):
            p=np.array(a['position_enu_m']);q=np.array(b['position_enu_m'])
            samples.extend(np.linspace(p,q,max(2,math.ceil(np.linalg.norm(q-p)/4)+1)))
    required=-np.inf;unknown=0;min_gap=np.inf
    for p in samples:
        pts=surface.nearby(p[:2],cfg.point_clearance_proxy_m)
        if len(pts)<cfg.min_cell_points:unknown+=1;continue
        top=float(np.quantile(pts[:,2],.95));required=max(required,top+cfg.point_clearance_proxy_m)
        min_gap=min(min_gap,p[2]-top)
    return required,dict(sample_count=len(samples),unknown_samples=unknown,
                         minimum_observed_vertical_gap_m=None if not np.isfinite(min_gap) else min_gap,
                         collision_free_verified=False)


def union_grid(group,cell=5.):
    """Unique samples of selected scan envelopes, never fill the outer rectangle."""
    lo=np.min([r['bbox_xy'][0] for r in group],axis=0)
    hi=np.max([r['bbox_xy'][1] for r in group],axis=0)
    xy=np.array([[x,y] for x in np.linspace(lo[0],hi[0],max(2,math.ceil((hi[0]-lo[0])/cell)+1))
                for y in np.linspace(lo[1],hi[1],max(2,math.ceil((hi[1]-lo[1])/cell)+1))])
    points=[]
    for p in xy:
        members=[r for r in group if np.all(p>=r['bbox_xy'][0]) and np.all(p<=r['bbox_xy'][1])]
        if members:
            r=min(members,key=lambda r:np.linalg.norm(p-np.array(r['seed'][:2])))
            points.append([*p,r['aim_z']])
    return np.asarray(points)


def choose_scan_segments(segments,grid,records,stems,hfov,vfov):
    """Greedy set cover over continuous strips, not independent camera points.

    Three new view directions incl. nadir and >=3 photo observations per surface
    sample. Previous-view direction count breaks ties only; it is not a learned
    expected-GS-gain model. All original target cells stay in the objective.
    """
    names=list(dict.fromkeys(s['mode'] for s in segments));mode_ids=[names.index(s['mode']) for s in segments]
    support=np.stack([np.sum([frustum_contains(grid,np.array(w['position_enu_m']),w['aircraft_yaw_deg'],
                         w['gimbal_pitch_deg'],hfov,vfov) for w in s['waypoints']],axis=0) for s in segments])
    old={mode:0 for mode in names}
    for r in records:
        if r['stem'] not in stems:continue
        yaw,pitch,_=r['camera_ypr_deg']
        if pitch < -75:mode='NADIR'
        else:mode=['NORTH_OBLIQUE','EAST_OBLIQUE','SOUTH_OBLIQUE','WEST_OBLIQUE'][int((yaw+45)%360//90)]
        old[mode]+=1
    total=np.zeros(len(grid),int);dirs=np.zeros((len(names),len(grid)),bool);chosen=[];remaining=set(range(len(segments)))
    def deficit(t,d):return np.maximum(3-t,0)+np.maximum(3-d.sum(0),0)+(~d[names.index('NADIR')])
    while remaining:
        before=deficit(total,dirs);candidates=[]
        if not before.any():break
        for i in sorted(remaining):
            d=dirs.copy();d[mode_ids[i]]|=support[i]>0
            gain=int(np.sum(before-deficit(total+support[i],d)))
            cost=len(segments[i]['waypoints'])
            candidates.append(((gain/cost,-old[segments[i]['mode']],-cost,-i),i))
        best,i=max(candidates)
        if best[0]<=0:break
        chosen.append(i);remaining.remove(i);total+=support[i];dirs[mode_ids[i]]|=support[i]>0
    if np.any(deficit(total,dirs)):
        raise ValueError('Candidate strips cannot cover all target-envelope samples; do not silently discard targets')
    return [segments[i] for i in chosen],dict(candidate_strips=len(segments),selected_strips=len(chosen),
            objective='>=3 photos, >=3 new directions including nadir per union sample',
            original_direction_histogram=old,target_samples=len(grid),
            minimum_photos=int(total.min()),minimum_new_directions=int(dirs.sum(0).min()))


def connect_segments(segments,records,evidence_stems,center,hfov,vfov,cfg):
    """Order scan lines; attach an explicit, counted visual-overlap approach."""
    evidence=[r for r in records if r['stem'] in evidence_stems]
    if not evidence:raise ValueError('No acquired image evidence for route')
    choices=[]
    for i,s in enumerate(segments):
        for reverse in [False,True]:
            w=s['waypoints'][-1 if reverse else 0];pos=np.array(w['position_enu_m']);target=np.array(w['target_enu_m'])
            newdir=(pos-target)/np.linalg.norm(pos-target)
            for e in evidence:
                old=np.array(e['enu']);d=old-target
                angle=float(np.degrees(np.arccos(np.clip(d@newdir/max(np.linalg.norm(d),1e-8),-1,1))))
                visible=bool(frustum_contains(target[None],old,*e['camera_ypr_deg'][:2],hfov,vfov)[0])
                # Visibility first, then compatible direction, then length (not risk reranking).
                choices.append(((not visible,angle,np.linalg.norm(old-pos)),i,reverse,e))
    cost,index,rev,anchor=min(choices,key=lambda x:x[0]);remaining=list(segments)
    first=remaining.pop(index)
    if rev:first={**first,'waypoints':list(reversed(first['waypoints']))}
    ordered=[first]
    while remaining:
        last=np.array(ordered[-1]['waypoints'][-1]['position_enu_m'])
        _,i,rev=min((np.linalg.norm(last-s['waypoints'][-1 if rev else 0]['position_enu_m']),i,rev)
                   for i,s in enumerate(remaining) for rev in [False,True])
        s=remaining.pop(i)
        if rev:s={**s,'waypoints':list(reversed(s['waypoints']))}
        ordered.append(s)
    first_shot=ordered[0]['waypoints'][0];start=np.array(first_shot['position_enu_m'])
    old=np.array(anchor['enu']);target=np.array(first_shot['target_enu_m'])
    # Remain at scan altitude; do not descend to an old possibly unsafe pose.
    vector=old[:2]-start[:2];length=np.linalg.norm(vector)
    approach=start.copy()
    if length>1:approach[:2]+=vector/length*min(length,2*cfg.speed_mps*cfg.interval_s)
    bridge=[]
    if np.linalg.norm(approach-start)>1:
        n=max(2,math.ceil(np.linalg.norm(approach-start)/(cfg.speed_mps*cfg.interval_s)))
        for p in np.linspace(approach,start,n+1)[:-1]:
            w=capture_record(p,target,first_shot['aircraft_yaw_deg'],'OVERLAP_APPROACH',cfg)
            # account for shorter final spacing
            w['speed_mps']=min(cfg.speed_mps,np.linalg.norm(approach-start)/n/cfg.interval_s)
            bridge.append(w)
    link=dict(anchor_stem=anchor['stem'],old_position_enu_m=old.tolist(),
              direction_delta_deg=cost[1],target_in_original_frustum=not cost[0],
              matching_features_verified=False,bridge_photo_count=len(bridge))
    return ordered,bridge,link


def explicit_trajectory(segments,bridge,cfg):
    """Reuse the established interpolator, but never inherit its safety claim."""
    trajectory=list(bridge)
    for s in segments:
        if trajectory:
            a=trajectory[-1];b=s['waypoints'][0]
            transition=make_transit_leg(a['position_enu_m'],b['position_enu_m'],
                    a['aircraft_yaw_deg'],b['aircraft_yaw_deg'],a['gimbal_pitch_deg'],
                    b['gimbal_pitch_deg'],cfg.speed_mps,cfg.speed_mps*cfg.interval_s)
            for w in transition:
                w['is_safe_transit']=False;w['safety_status']='unverified_preview';w['capture_view']='TRANSIT'
            trajectory.extend(transition[1:-1])
        trajectory.extend(s['waypoints'])
    return trajectory


def audit_views(segments,bridge,bounds,z,hfov,vfov,cfg,grid=None):
    lo,hi=np.array(bounds)
    if grid is None:grid=np.array([[x,y,z] for x in np.linspace(lo[0],hi[0],11) for y in np.linspace(lo[1],hi[1],11)])
    modes={};top_ray=-90.;target_error=0
    for s in segments:
        masks=[]
        for w in s['waypoints']:
            p=np.array(w['position_enu_m']);t=np.array(w['target_enu_m'])
            masks.append(frustum_contains(grid,p,w['aircraft_yaw_deg'],w['gimbal_pitch_deg'],hfov,vfov))
            f=camera_basis(w['aircraft_yaw_deg'],w['gimbal_pitch_deg'])[:,2]
            target_error=max(target_error,float(np.degrees(np.arccos(np.clip(f@(t-p)/np.linalg.norm(t-p),-1,1)))))
            top_ray=max(top_ray,w['gimbal_pitch_deg']+np.degrees(vfov)/2)
        modes[s['mode']]=modes.get(s['mode'],np.zeros(len(grid),int))+np.array(masks).sum(0)
    for w in bridge:top_ray=max(top_ray,w['gimbal_pitch_deg']+np.degrees(vfov)/2)
    counts=np.stack(list(modes.values()))
    return dict(planar_grid_points=len(grid),planar_coverage_fraction=float(np.mean(counts.sum(0)>0)),
                planar_three_view_fraction=float(np.mean(counts.sum(0)>=3)),
                planar_five_direction_fraction=float(np.mean((counts>0).sum(0)==5)),
                minimum_direction_count=int((counts>0).sum(0).min()),
                max_optical_axis_target_error_deg=target_error,highest_image_ray_pitch_deg=top_ray,
                occlusion_tested=False,real_feature_overlap_verified=False,
                warning='Coverage is a horizontal-envelope FOV proxy; not actual facade visibility or reconstruction gain.')


def build_plan(selection,manifest,xyz,cfg):
    if manifest.get('GS_or_SfM_inputs') is not False:raise ValueError('Requires sensor-only manifest')
    if selection['status']!='ranking_frozen':raise ValueError('Selection not frozen')
    records=manifest['records'];f35=float(np.median([r['f35'] for r in records]))
    hfov=2*np.arctan(36/(2*f35));vfov=2*np.arctan(36/(2*f35*cfg.image_aspect))
    selected=selection['selected'];cell=max(2.,float(np.median([r['radius'] for r in selected]))/2)
    surface=Surface(xyz,cell,cfg.min_cell_points)
    regions=region_envelopes(selected,records,surface,hfov,vfov,cfg)
    groups=merge_envelopes(regions);areas=[]
    for gi,group in enumerate(groups,1):
        bounds=[np.min([r['bbox_xy'][0] for r in group],axis=0),np.max([r['bbox_xy'][1] for r in group],axis=0)]
        center=(bounds[0]+bounds[1])/2;z=max(r['aim_z'] for r in group)
        fit=max(r['fit_rmse_m'] for r in group);oldz=float(np.median([r['previous_camera_z'] for r in group]))
        flightz=max(oldz,z+np.median([r['reference_gap_m'] for r in group]))
        climb_cap=max(r['enu'][2] for r in records)+cfg.max_local_climb_m
        segments=[]
        for _ in range(5):
            segments=make_scan(bounds,z,flightz,hfov,vfov,cfg)
            required,clearance=cloud_clearance(segments,surface,cfg)
            if required<=flightz+.01:break
            flightz=required
        else:
            raise ValueError('Observed-cloud height iteration did not converge; do not export stale waypoints')
        if flightz>climb_cap:raise ValueError('Required preview altitude exceeds configured local climb cap')
        stems=sorted({s for r in group for s in r['evidence_stems']})
        grid=union_grid(group)
        all_segments=segments
        segments,selection_audit=choose_scan_segments(segments,grid,records,stems,hfov,vfov)
        segments,bridge,link=connect_segments(segments,records,stems,np.r_[center,z],hfov,vfov,cfg)
        selection_audit['extra_old_view_link_strips']=0
        if link['direction_delta_deg']>35 or not link['target_in_original_frustum']:
            compatible,_,best_link=connect_segments(all_segments,records,stems,np.r_[center,z],hfov,vfov,cfg)
            extra=compatible[0]
            if best_link['target_in_original_frustum'] and best_link['direction_delta_deg']<link['direction_delta_deg']:
                identity=lambda s:(s['mode'],s['row'])
                if identity(extra) not in {identity(s) for s in segments}:
                    segments,bridge,link=connect_segments(segments+[extra],records,stems,np.r_[center,z],hfov,vfov,cfg)
                    selection_audit['extra_old_view_link_strips']=1
        # Audit approach and inter-strip transits as well, not just the scan lines.
        scan_shots=bridge+[w for s in segments for w in s['waypoints']]
        required,clearance=cloud_clearance([dict(waypoints=scan_shots)],surface,cfg)
        clearance['meets_observed_proxy']=bool(required<=flightz+.01)
        audit=audit_views(segments,bridge,bounds,z,hfov,vfov,cfg,grid=grid)
        # Every point is retained even if uncertain. Flag it instead of turning low confidence into zero risk.
        flags=[]
        if not link['target_in_original_frustum']:flags.append('old_view_link_not_confirmed_by_FOV')
        if link['direction_delta_deg']>35:flags.append('large_old_new_view_direction_change')
        if max(r['geometry_conflict_m'] for r in group)>max(2*fit,cell):flags.append('risk_z_disagrees_with_bootstrap_surface')
        if sum(r['surface_extension_cells'] for r in group)<4:flags.append('weak_surface_extent')
        if not clearance['meets_observed_proxy']:flags.append('observed_cloud_clearance_proxy_failed')
        if clearance['unknown_samples']:flags.append('unknown_geometry_on_path')
        if audit['highest_image_ray_pitch_deg']>=-5:flags.append('near_horizon_view')
        positions=[w['position_enu_m'] for w in scan_shots]
        travel=path_length(positions)
        # Segment turns have explicit no-capture transit; estimate includes cruise + capture cadence.
        scan_length=sum(path_length([w['position_enu_m'] for w in s['waypoints']]) for s in segments)
        motion_seconds=sum(path_length([w['position_enu_m'] for w in s['waypoints']])/s['waypoints'][0]['speed_mps'] for s in segments)
        turns=len(segments)-1
        seconds=motion_seconds+(travel-scan_length)/cfg.speed_mps+len(bridge)*cfg.interval_s+turns*cfg.interval_s
        areas.append(dict(area_id=f'A{gi}',candidate_ranks=sorted(r['ranks'][0] for r in group),
                          bbox_xy=np.array(bounds).tolist(),aim_z=z,flight_z=flightz,
                          previous_camera_z_median=oldz,local_climb_from_evidence_m=flightz-oldz,
                          uncertainty_padding_m=max(r['uncertainty_padding_m'] for r in group),
                          score=max(r['score'] for r in group),segments=segments,bridge=bridge,old_view_link=link,
                          envelope_boxes=[r['bbox_xy'] for r in group],strip_selection=selection_audit,
                          trajectory=explicit_trajectory(segments,bridge,cfg),
                          audit=audit,clearance_proxy=clearance,review_flags=flags,
                          photo_count=len(scan_shots),bridge_photo_count=len(bridge),
                          intra_area_path_m=travel,estimated_seconds=seconds))
    return dict(schema='frozen_region_scans_preview_v1',scene=manifest['scene'],config=asdict(cfg),
                reference_wgs84=manifest['reference_wgs84'],coordinate_frame='ENU metres; z relative to EXIF reference, NOT AGL',
                camera=dict(horizontal_fov_deg=float(np.degrees(hfov)),vertical_fov_deg=float(np.degrees(vfov)),
                            source='EXIF 35mm focal; 4:3 aspect assumption'),
                source_candidates=regions,areas=areas,photo_count=sum(a['photo_count'] for a in areas),
                bridge_photo_count=sum(a['bridge_photo_count'] for a in areas),
                intra_area_path_m=sum(a['intra_area_path_m'] for a in areas),
                estimated_scan_seconds=sum(a['estimated_seconds'] for a in areas),
                excluded_costs=['launch/landing','inter-area safe transit','battery changes','wind'],
                ground_height_known=False,flight_authorized=False,
                limitations=['GPS fit RMSE is an uncertainty proxy, not a calibrated confidence interval',
                             'No saved Pi3X confidence in bootstrap cache: do not invent confidence weighting',
                             'No semantic water mask available; all candidates retained for review',
                             'No obstacle mesh, surveyed ground datum or real feature matching verification',
                             'Upper-surface envelope scans; facade occlusion and vertical facade coverage not certified',
                             'Frozen 12 targets only; not proof all scene defects have been found'],
                GS_or_teacher_used_for_planning=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',required=True,type=Path);p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--cloud',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    selection=json.loads(a.selection.read_text());manifest=json.loads(a.manifest.read_text())
    xyz=np.load(a.cloud)['xyz'];xyz=xyz[np.isfinite(xyz).all(1)]
    result=build_plan(selection,manifest,xyz,Config())
    result['inputs']={k:dict(path=str(v.resolve()),sha256=sha(v)) for k,v in
                      [('selection',a.selection),('manifest',a.manifest),('cloud',a.cloud)]}
    result['planner_sha256']=sha(Path(__file__))
    (a.output/'route_plan.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ['scene','photo_count','bridge_photo_count','intra_area_path_m','estimated_scan_seconds']},indent=2))
    for area in result['areas']:
        print(area['area_id'],area['candidate_ranks'],'photos',area['photo_count'],'climb',round(area['local_climb_from_evidence_m'],1),area['review_flags'])


if __name__=='__main__':main()
