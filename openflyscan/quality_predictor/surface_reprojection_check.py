"""Current-view projection/depth diagnostics. No removal metadata or GS inputs."""
import numpy as np

def check_surface_observations(point, maps, poses, intrinsics, patch_size=14):
    point=np.asarray(point,float);rows=[]
    for i,(world,pose,K) in enumerate(zip(maps,poses,intrinsics)):
        h,w=world.shape[:2];local=pose[:3,:3].T@(point-pose[:3,3])
        if local[2]<=1e-6:
            rows.append(dict(view=i,status='behind'));continue
        uv=K@local;uv=uv[:2]/uv[2];pixel=uv/patch_size-.5
        if not (0<=pixel[0]<w-1 and 0<=pixel[1]<h-1):
            rows.append(dict(view=i,status='outside'));continue
        x,y=np.floor(pixel).astype(int);dx,dy=pixel-[x,y]
        neighbors=world[y:y+2,x:x+2].reshape(4,3)
        if not np.isfinite(neighbors).all():
            rows.append(dict(view=i,status='invalid'));continue
        camera_neighbors=(neighbors-pose[:3,3])@pose[:3,:3]
        z=camera_neighbors[:,2];zrange=float(np.ptp(z)/local[2])
        weights=np.array([(1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy])
        predicted_depth=float(weights@z);relative_error=float((predicted_depth-local[2])/local[2])
        rows.append(dict(view=i,status='depth_boundary' if zrange>.1 else 'compared',
            uv_fraction=(uv/np.array([w*patch_size,h*patch_size])).tolist(),
            expected_depth=float(local[2]),predicted_depth=predicted_depth,relative_depth_error=relative_error,
            neighborhood_relative_depth_range=zrange))
    compared=[r for r in rows if r['status']=='compared']
    return dict(views=len(rows),in_frame=sum(r['status'] in ('compared','depth_boundary') for r in rows),
        depth_boundary=sum(r['status']=='depth_boundary' for r in rows),
        depth_support={str(t):sum(abs(r['relative_depth_error'])<=t for r in compared) for t in (.01,.02,.05)},
        occluded_at_5pct=sum(r['relative_depth_error']<-.05 for r in compared),observations=rows)
