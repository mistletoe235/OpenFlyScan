"""Target-free spatial queries from the current native Pi3X point maps."""
import numpy as np


def propose_current_regions(maps, confidence, max_regions=128, confidence_quantile=.25):
    valid = np.isfinite(maps).all(-1) & np.isfinite(confidence)
    if not 0<=confidence_quantile<1:raise ValueError('invalid confidence quantile')
    if confidence_quantile>0:valid &= confidence >= np.quantile(confidence[valid], confidence_quantile)
    points = maps[valid]
    origin = np.median(points, axis=0)
    distances = np.linalg.norm(np.diff(maps, axis=2), axis=-1)
    distances = distances[np.isfinite(distances) & (distances > 1e-6)]
    size = max(2 * float(np.median(distances)), 1e-5)
    keys = np.floor((points-origin)/size).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    eligible = np.flatnonzero(counts >= 3)
    centers = np.asarray([np.median(points[inverse==index],axis=0) for index in eligible]).reshape(-1, 3)
    if len(centers)>max_regions:
        start = int(np.argmin(np.linalg.norm(centers-origin,axis=1)))
        chosen=[start];d=np.linalg.norm(centers-centers[start],axis=1)
        while len(chosen)<max_regions:
            d[chosen]=-1;i=int(np.argmax(d));chosen.append(i)
            d=np.minimum(d,np.linalg.norm(centers-centers[i],axis=1))
        centers=centers[chosen]
        eligible=eligible[chosen]
    source_indices = np.flatnonzero(valid)
    members = [source_indices[inverse == index] for index in eligible]
    return dict(centers=centers,origin=origin,cell_size=size,valid_points=points,
                source_patch_indices=members,source_patch_shape=maps.shape[:3])
