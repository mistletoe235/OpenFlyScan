"""Separate frozen risk support from neighboring geometric context.

Task grouping is not a claim that points lie on one plane. No identifiers, GS
images, or manually labelled defects participate in grouping/classification.
"""
import numpy as np
from scipy.spatial import cKDTree
from openflyscan.planning.region_coverage_geometry import unique_voxels,unit,plane_axes,plane_footprint
from openflyscan.planning.plan_frozen_region_scans import camera_basis


def disk_support_area(members,cell):
    """Deduplicated XY area of frozen risk disks; not point-count density."""
    keys=set();sizes=[]
    for member in members:
        center=np.asarray(member['frozen']['xyz'])[:2];radius=float(member['frozen']['radius'])
        lo=np.floor((center-radius)/cell).astype(int);hi=np.ceil((center+radius)/cell).astype(int)
        x,y=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
        ij=np.column_stack([x.ravel(),y.ravel()]);inside=np.linalg.norm((ij+.5)*cell-center,axis=1)<=radius
        own=set(map(tuple,ij[inside]));keys.update(own);sizes.append(len(own)*cell**2)
    return len(keys)*cell**2,max(sizes,default=cell**2)


def build_risk_tasks(regions,cfg,hfov,vfov,long_axis_yaw):
    """Connect native risk patches, not expanded roof surfaces.

    One nominal photo travel step is the maximum gap between native supports.
    Large uncertainty does NOT make two patches connected. A component spanning
    >=2 largest patch-equivalents or >one reference camera footprint gets an
    area survey. The thresholds are explicit unvalidated prototype choices.
    """
    gap_limit=cfg.risk_cluster_step_factor*cfg.speed_mps*cfg.interval_s
    neighbors={i:set() for i in range(len(regions))};relations=[]
    for i,a in enumerate(regions):
        for j in range(i+1,len(regions)):
            b=regions[j]
            # Group photography tasks in XY, not surfaces into a fictitious
            # common plane. Preserve ALL heights for later 3-D coverage checks.
            gap=float(cKDTree(a['raw_points'][:,:2]).query(b['raw_points'][:,:2])[0].min())
            connected=gap<=gap_limit
            if connected:neighbors[i].add(j);neighbors[j].add(i)
            relations.append(dict(a=a['rank'],b=b['rank'],native_support_xy_gap_m=gap,
                median_height_gap_m=float(abs(np.median(a['raw_points'][:,2])-np.median(b['raw_points'][:,2]))),
                threshold_m=gap_limit,task_connected=connected))
    pending=set(neighbors);components=[]
    while pending:
        seed=min(pending,key=lambda i:regions[i]['rank']);stack=[seed];pending.remove(seed);group=[]
        while stack:
            i=stack.pop();group.append(i)
            for j in sorted(neighbors[i]&pending):pending.remove(j);stack.append(j)
        components.append(sorted(group,key=lambda i:regions[i]['rank']))
    areas=[]
    for indices in components:
        members=[regions[i] for i in indices];cell=min(r['cell_m'] for r in members)
        points=unique_voxels(np.vstack([r['raw_points'] for r in members]),cell)
        context=unique_voxels(np.vstack([r['points'] for r in members]),cell)
        normal=unit(sum(r['normal']*len(r['raw_points']) for r in members))
        kind=members[0]['kind'] if len({r['kind'] for r in members})==1 else 'uncertain'
        area=dict(id=f'area_{len(areas)+1:02d}',ranks=[r['rank'] for r in members],members=members,
            points=points,context_surface_points=context,center=np.median(points,axis=0),normal=normal,
            kind=kind,trusted_normal=all(r['trusted_normal'] for r in members),cell_m=cell,
            uncertainty_m=max(r['uncertainty_m'] for r in members),
            reference_range=float(np.median([r['reference_range'] for r in members])),risk=max(r['risk'] for r in members))
        support,largest=disk_support_area(members,cell)
        # Without uncertainty inflation: localization doubt alone is not evidence
        # of a large dense defect. Route coverage still includes uncertainty later.
        yaw=long_axis_yaw(points);n=np.array([0.,0.,1.]) if kind!='facade' else normal
        u,v=plane_axes(n,yaw);pitch=-90 if kind!='facade' else -45
        distance=max(cfg.min_range_m,area['reference_range']);center=area['center']
        pos=center-distance*camera_basis(yaw,pitch)[:,2]
        fp=plane_footprint(pos,center,yaw,pitch,n,u,v,hfov,vfov)
        span=np.ptp(np.column_stack([(points-center)@u,(points-center)@v]),axis=0)
        exceeds=fp is not None and bool(np.any(span>np.array([fp['width'],fp['height']])))
        ratio=support/max(largest,cell**2)
        dense=ratio>=cfg.survey_patch_equivalents or exceeds
        area['task_type']='area_survey' if dense else 'local_patch'
        area['risk_extent_audit']=dict(native_point_count=len(points),context_point_count=len(context),
            risk_disk_union_area_m2=support,largest_risk_disk_area_m2=largest,effective_patch_area_ratio=ratio,
            minimum_area_ratio_for_survey=cfg.survey_patch_equivalents,exceeds_reference_footprint=exceeds,native_span_m=span.tolist(),
            reference_footprint_m=[fp['width'],fp['height']] if fp is not None else None,
            scope='Native predicted risk points only; neighboring roof geometry is context, not extra defect',
            classification='Connected multi-patch extent or large native footprint' if dense else 'Compact native risk extent',
            validated_defect_polygon=False)
        areas.append(area)
    return areas,relations
