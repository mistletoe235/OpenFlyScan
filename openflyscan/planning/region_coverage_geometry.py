"""Image-bound surface envelopes and plane coverage primitives. No GS inputs."""
from dataclasses import dataclass, asdict
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import re

import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.ndimage import binary_dilation

from openflyscan.planning.plan_frozen_region_scans import camera_basis, frustum_contains, capture_record, Config


@dataclass
class CoverageSettings:
    target_budget: int = 18
    photo_budget: int = 240
    forward_overlap: float = .8
    side_overlap: float = .7
    speed_mps: float = 4.
    interval_s: float = 2.
    min_range_m: float = 20.
    clearance_m: float = 12.
    max_relative_altitude_m: float = 80.
    min_pair_angle_deg: float = 8.
    useful_pair_angle_deg: float = 20.
    max_pair_angle_deg: float = 60.
    support_context_radius_factor: float = 4.
    max_actions_per_region: int = 3
    max_samples_per_region: int = 144
    risk_cluster_step_factor: float = 1.
    survey_patch_equivalents: float = 2.
    nominal_oblique_pitch_deg: float = -45.
    height_sampling_policy: str = 'legacy'
    height_step_m: float = 10.
    height_search_max_m: float = 120.
    budget_atomic_strips: bool = False
    same_height_tolerance_m: float = 5.
    context_axis_tolerance_deg: float = 35.
    context_min_points: int = 30
    context_max_anchors: int = 24
    selection_policy: str = 'observation'
    use_context_preference: bool = True
    local_multiview: bool = False
    surface_multiview: bool = False
    local_max_directions: int = 3
    local_min_mean_gain: float = .02
    local_min_improved_fraction: float = .1
    local_point_gain_threshold: float = .05
    local_min_direction_separation_deg: float = 25.
    connection_overlap: float = .8
    connection_angle_step_deg: float = 15.
    max_connection_photos: int = 32
    connection_scale_ratio_p90: float = 1.25
    connection_surface_angle_p90_deg: float = 20.
    climb_speed_mps: float = 2.
    descent_speed_mps: float = 1.5


def unit(v):
    v=np.asarray(v,float)
    return v/max(np.linalg.norm(v),1e-9)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def altitude_reference(manifest):
    """Read sensor-only EXIF/XMP; never infer takeoff from reconstructed terrain."""
    if manifest.get('experiment_condition')=='registered_pose_control':
        if manifest.get('coordinate_frame')!='registered_local_zup_meters' or manifest.get('GS_or_SfM_inputs') is not True:
            raise ValueError('Controlled registered pose requires explicit provenance and frame')
        return dict(status='registered_local_zup_not_AGL',coordinate_frame=manifest['coordinate_frame'],
                    experiment_condition='registered_pose_control',geographic_ENU_verified=False)
    if manifest.get('altitude_reference',{}).get('status')=='explicit_simulator_ENU_datum_not_AGL':
        if manifest.get('coordinate_frame')!='ENU_meters':raise ValueError('Declared ENU altitude requires ENU coordinates')
        return dict(manifest['altitude_reference'])
    records=manifest['records'];offsets=[];sources=[];enu_offsets=[]
    # Stratified samples cover all recorded sections without assuming one file folder is one flight.
    for index in np.unique(np.linspace(0,len(records)-1,min(60,len(records))).astype(int)):
        r=records[index]
        with Path(r.get('raw_xmp_source',r['image'])).open('rb') as f:text=f.read(256*1024).decode('latin1')
        absolute=re.search(r'drone-dji:AbsoluteAltitude="([+\-0-9.]+)"',text)
        relative=re.search(r'drone-dji:RelativeAltitude="([+\-0-9.]+)"',text)
        if absolute and relative:
            aa=float(absolute.group(1));rr=float(relative.group(1));offsets.append(aa-rr)
            sources.append(dict(stem=r['stem'],absolute_altitude_m=aa,relative_altitude_m=rr,takeoff_absolute_m=aa-rr))
        enu_offsets.append(r['alt']-r['enu'][2])
    origin=float(np.median(enu_offsets))
    if len(offsets)<3:
        return dict(status='unknown',enu_absolute_origin_m=origin,takeoff_absolute_m=None,samples=sources)
    values=np.array(offsets);spread=float(np.ptp(values))
    # Different sortie references / drift are not silently averaged into one safe altitude.
    stable=spread<=2.
    takeoff=float(np.median(values))
    return dict(status='sensor_reference_consistent' if stable else 'inconsistent_references',
        enu_absolute_origin_m=origin,enu_origin_spread_m=float(np.ptp(enu_offsets)),
        takeoff_absolute_m=takeoff if stable else None,takeoff_candidate_median_m=takeoff,
        takeoff_reference_spread_m=spread,samples=sources,
        conservative_reference_absolute_m=float(values.min()),
        conservative_reference_enu_z=float(values.min()-origin),
        reference_absolute_range_m=[float(values.min()),float(values.max())],
        assumption='80m ceiling interpreted as above takeoff for prototype; user has not confirmed flight setting',
        takeoff_enu_z=takeoff-origin if stable else None)


def unique_voxels(points,cell):
    keys=np.floor(np.asarray(points)/cell).astype(int)
    _,index=np.unique(keys,axis=0,return_index=True)
    return np.asarray(points)[index]


def largest_seed_component(points,seeds,cell):
    """2-D connected occupied cells, retaining only components touching evidence."""
    keys=np.floor(points[:,:2]/cell).astype(int);groups=defaultdict(list)
    for k,p in zip(keys,points):groups[tuple(k)].append(p)
    seedkeys=set(map(tuple,np.floor(seeds[:,:2]/cell).astype(int)))
    start=set(groups)&seedkeys
    if not start and groups:
        centers=np.array(list(groups))*cell+cell/2
        d,i=cKDTree(centers).query(np.median(seeds[:,:2],axis=0))
        if d<2*cell:start={tuple(list(groups)[i])}
    seen=set(start);stack=list(start)
    while stack:
        k=stack.pop()
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]:
            q=(k[0]+dx,k[1]+dy)
            if q in groups and q not in seen:seen.add(q);stack.append(q)
    return np.array([[*(np.array(k)*cell+cell/2),np.median(np.array(groups[k])[:,2])] for k in sorted(seen)]).reshape(-1,3)


def extract_region(rank,selected,diag,surface,cloud,cfg):
    """Grow same-height roof support only in a bounded, image-observed context.

    Not a semantic building mask. All raw risk points remain separate from this
    conservative scan envelope. Low-confidence unknown sides are never invented.
    """
    original=surface['xyz'];center=np.median(original,axis=0);radius=float(selected['radius'])
    cell=float(np.clip(radius/3,.75,2.));normal=np.array(diag['dominant_normal'])
    structural_roof=normal[2]>.75 and diag['dominant_normal_fraction']>=.65
    trusted=(diag['dominant_normal_fraction']>=.65 and diag['consistent_view_count']>=2 and
             diag['conf_percentile_median']>=.25 and not any('cross_window' in f for f in diag['flags']))
    kind='roof' if structural_roof else ('facade' if trusted and abs(normal[2])<.45 else 'uncertain')
    context_radius=cfg.support_context_radius_factor*radius
    evidence=set(surface['stems'].astype(str));context=[]
    for i,stem in enumerate(cloud['stems'].astype(str)):
        if stem not in evidence:continue
        pts=cloud['dense_maps'][i].reshape(-1,3)
        finite=np.isfinite(pts).all(1)
        nearby=np.linalg.norm(pts-center,axis=1)<=context_radius
        context.append(pts[finite&nearby])
    context=np.vstack(context) if context else original
    if kind=='roof':
        # The limit guards crossing the roof edge down onto surrounding podiums.
        height_tolerance=max(1.,.15*radius)
        compatible=context[np.abs((context-center)@normal)<=height_tolerance]
        expanded=largest_seed_component(compatible,original,cell) if len(compatible) else np.empty((0,3))
    else:expanded=np.empty((0,3))
    extent_source='bounded_connected_same_height_observed_surface' if len(expanded)>=4 else 'native_risk_patch_only'
    if len(expanded)<4:expanded=unique_voxels(original,cell)
    uncertainty=max(cell,min(radius,float(diag['view_center_spread_p90_m'])))
    # Keep location uncertainty distinct from observed occupied surface.
    if diag['cross_window_median_m'] is not None:
        uncertainty=max(uncertainty,min(2*radius,diag['cross_window_median_m']))
    return dict(rank=rank,frozen=selected,diag=diag,points=expanded,raw_points=original,
        center=center,normal=normal,kind=kind,trusted_normal=trusted,cell_m=cell,
        uncertainty_m=uncertainty,context_radius_m=context_radius,context_points=context,
        extent_source=extent_source,reference_range=float(np.median(np.linalg.norm(surface['cameras']-original,axis=1))),
        risk=float(selected['score']),surface=surface)


def merge_regions(regions):
    """No radius-only merge or transitive merge across incompatible surfaces."""
    relations=[];allowed={}
    for i,a in enumerate(regions):
        for j in range(i+1,len(regions)):
            b=regions[j];gap=float(cKDTree(a['points']).query(b['points'])[0].min())
            touch=gap<=1.75*max(a['cell_m'],b['cell_m'])
            normal_ok=a['normal']@b['normal']>=np.cos(np.radians(25))
            plane_gap=abs(float((a['center']-b['center'])@unit(a['normal']+b['normal'])))
            compatible=a['kind']==b['kind']=='roof' and normal_ok and plane_gap<=max(2.,.25*max(a['frozen']['radius'],b['frozen']['radius']))
            # Uncertain targets merge only with DIRECT shared original image patches.
            def pixels(r):return {(e['stem'],*p) for e in r['frozen']['image_evidence'] for p in e['patch_pixels_yx']}
            shared=bool(pixels(a)&pixels(b))
            if a['kind']==b['kind']=='uncertain' and shared:compatible=True
            allow=bool(touch and compatible)
            allowed[i,j]=allow
            if touch or gap<10:relations.append(dict(a=a['rank'],b=b['rank'],surface_gap_m=gap,plane_gap_m=plane_gap,
                normal_compatible=bool(normal_ok),shared_image_patch=shared,merge_allowed=allow))
    groups=[]
    for i in range(len(regions)):
        eligible=[g for g in groups if all(allowed.get((min(i,j),max(i,j)),False) for j in g)]
        if eligible:eligible[0].append(i)
        else:groups.append([i])
    result=[]
    for gi,indices in enumerate(groups,1):
        members=[regions[i] for i in indices];points=unique_voxels(np.vstack([r['points'] for r in members]),min(r['cell_m'] for r in members))
        normal=unit(sum(r['normal']*len(r['points']) for r in members));kind=members[0]['kind']
        result.append(dict(id=f'area_{gi:02d}',ranks=[r['rank'] for r in members],members=members,points=points,
            center=np.median(points,axis=0),normal=normal,kind=kind,
            trusted_normal=all(r['trusted_normal'] for r in members),
            uncertainty_m=max(r['uncertainty_m'] for r in members),cell_m=min(r['cell_m'] for r in members),
            reference_range=float(np.median([r['reference_range'] for r in members])),risk=max(r['risk'] for r in members)))
    return result,relations


def plane_axes(normal,yaw):
    right=camera_basis(yaw,-55)[:,0];u=unit(right-normal*(right@normal))
    if np.linalg.norm(u)<.5:u=unit(np.cross(normal,[0,0,1]))
    return u,unit(np.cross(normal,u))


def plane_footprint(position,target,yaw,pitch,normal,u,v,hfov,vfov):
    rays=np.array([[x*np.tan(hfov/2),y*np.tan(vfov/2),1.] for x,y in [(-1,-1),(-1,1),(1,1),(1,-1)]])@camera_basis(yaw,pitch).T
    denominator=rays@normal;numerator=float((target-position)@normal)
    if np.any(np.abs(denominator)<1e-8) or np.any(numerator/denominator<=0):return None
    world=position+rays*(numerator/denominator)[:,None]
    uv=np.column_stack([(world-target)@u,(world-target)@v])
    hull=ConvexHull(uv);eq=hull.equations
    hw=min(abs(uv[:,0].min()),abs(uv[:,0].max()));hh=min(abs(uv[:,1].min()),abs(uv[:,1].max()))
    if min(hw,hh)<=1e-6:return None
    # A centered rectangle completely inside the perspective footprint.
    scale=min(1.,float(np.min(-eq[:,2]/np.maximum(abs(eq[:,0])*hw+abs(eq[:,1])*hh,1e-12))))
    if scale<=0:return None
    return dict(width=2*hw*scale*.95,height=2*hh*scale*.95,world_corners=world)


def scan_reference_center(area):
    """Horizontal surveys use the upper native support plane, not mean terrain.

    This is a flight grid reference, NOT a flattening of reconstructed surfaces.
    All original point heights remain in coverage/visibility verification.
    """
    center=np.median(area['points'],axis=0).copy()
    if area['kind']!='facade':center[2]=np.max(area['points'][:,2])
    return center


def coverage_action(area,yaw,pitch,distance,hfov,vfov,cfg,altitude_max_enu,tree,mode):
    """Full rectangle scan covering the observed envelope; not just one line.

    Axis follows plane/camera geometry. For roofs all rows are horizontal;
    facades use stacked height rows. Unknown geometry uses a horizontal scan
    proxy but all coverage validation uses the original 3-D support points.
    """
    pts=area['points'];center=scan_reference_center(area)
    normal=area['normal'] if area['kind'] in ('roof','facade') else np.array([0.,0.,1.])
    if area['kind']=='roof':normal=np.array([0.,0.,1.]) # horizontal vehicle rows; check nonplanar points afterwards
    u,v=plane_axes(normal,yaw);forward=camera_basis(yaw,pitch)[:,2]
    position=center-distance*forward
    if pitch+np.degrees(vfov)/2>=-5:return None,'horizon'
    if normal@(position-center)<=0:return None,'backside_plane'
    footprint=plane_footprint(position,center,yaw,pitch,normal,u,v,hfov,vfov)
    if footprint is None:return None,'no_finite_plane_footprint'
    # Project every height along the fixed viewing direction onto the scan
    # reference plane. Merely dropping z loses oblique coverage at roof steps.
    denominator=float(forward@normal)
    if abs(denominator)<1e-8:return None,'view_parallel_to_reference_plane'
    projected=pts+(((center-pts)@normal)/denominator)[:,None]*forward
    uv=np.column_stack([(projected-center)@u,(projected-center)@v]);margin=area['uncertainty_m']+area['cell_m']/2
    lo=uv.min(0)-margin;hi=uv.max(0)+margin
    width,height=footprint['width'],footprint['height']
    spacing=min(width*(1-cfg.forward_overlap),cfg.speed_mps*cfg.interval_s)
    if spacing<.5:return None,'footprint_too_narrow'
    # Three images must also supply nonzero baseline even if the area is tiny.
    baseline=2*distance*np.tan(np.radians(cfg.min_pair_angle_deg)/2)
    extent=max(hi[0]-lo[0],baseline)
    mid=(lo[0]+hi[0])/2;xs=np.linspace(mid-extent/2,mid+extent/2,max(3,int(np.ceil(extent/spacing))+1))
    across=max(0.,hi[1]-lo[1]-height)
    count=max(1,int(np.ceil(across/(height*(1-cfg.side_overlap))))+1)
    ys=np.linspace((lo[1]+hi[1])/2-across/2,(lo[1]+hi[1])/2+across/2,count)
    rows=[];all_positions=[]
    for ri,y in enumerate(ys):
        shots=[]
        for x in (xs if ri%2==0 else xs[::-1]):
            target=center+x*u+y*v;pos=target-distance*forward
            if altitude_max_enu is not None and pos[2]>altitude_max_enu+1e-6:return None,'altitude_ceiling'
            shots.append(capture_record(pos,target,yaw,mode,Config()))
            all_positions.append(pos)
        speed=min(cfg.speed_mps,abs(xs[1]-xs[0])/cfg.interval_s)
        for w in shots:w['speed_mps']=speed
        rows.append(dict(row=ri,waypoints=shots))
    all_positions=np.array(all_positions)
    path=[]
    for a,b in zip(all_positions[:-1],all_positions[1:]):
        path.append(np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(b-a)/2))+1)))
    clearance=float(tree.query(np.vstack(path))[0].min())
    if clearance<cfg.clearance_m:return None,'observed_cloud_clearance'
    waypoints=[w for r in rows for w in r['waypoints']]
    bounds=np.array([center+x*u+y*v for x,y in [(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])]])
    return dict(id='',area_id=area['id'],ranks=area['ranks'],rank=min(area['ranks']),mode=mode,
        center=center.tolist(),normal=normal.tolist(),waypoints=waypoints,rows=rows,
        row_count=len(rows),photo_count=len(waypoints),range_m=distance,
        pitch_deg=pitch,yaw_deg=yaw,plane_bounds_enu=bounds.tolist(),
        scan_reference_height_m=float(center[2]),native_height_range_m=[float(pts[:,2].min()),float(pts[:,2].max())],
        projected_footprint_m=[width,height],target_envelope_m=(hi-lo).tolist(),
        forward_overlap=1-abs(xs[1]-xs[0])/width,
        side_overlap=1-abs(ys[1]-ys[0])/height if len(ys)>1 else None,
        photo_spacing_m=abs(xs[1]-xs[0]),clearance_proxy_m=clearance,
        length_m=float(np.linalg.norm(np.diff(all_positions,axis=0),axis=1).sum()),
        small_target_single_row=len(rows)==1,
        boundary_kind='conservative plane bounding rectangle; occupied support shown separately'),None


def coverage_measure(points,waypoints,hfov,vfov,min_pair_angle_deg=8.):
    if not waypoints:return dict(one_fraction=0.,three_fraction=0.,three_with_baseline_fraction=0.,min_count=0),np.zeros(len(points)),np.zeros(len(points))
    hits=[];directions=[]
    for w in waypoints:
        p=np.array(w['position_enu_m']);hits.append(frustum_contains(points,p,w['aircraft_yaw_deg'],w['gimbal_pitch_deg'],hfov,vfov))
        d=p-points;directions.append(d/np.maximum(np.linalg.norm(d,axis=1)[:,None],1e-9))
    hits=np.array(hits);directions=np.array(directions);maxangle=np.zeros(len(points))
    for i in range(len(hits)-1):
        dot=np.einsum('nk,jnk->jn',directions[i],directions[i+1:])
        angle=np.degrees(np.arccos(np.clip(dot,-1,1)))*(hits[i+1:]&hits[i])
        maxangle=np.maximum(maxangle,angle.max(0))
    counts=hits.sum(0)
    return dict(one_fraction=float(np.mean(counts>=1)),three_fraction=float(np.mean(counts>=3)),
        three_with_baseline_fraction=float(np.mean((counts>=3)&(maxangle>=min_pair_angle_deg))),min_count=int(counts.min()),
        meaning='FOV plus continuous pair angle on occupied evidence samples, NOT true occlusion or GS quality'),counts,maxangle
