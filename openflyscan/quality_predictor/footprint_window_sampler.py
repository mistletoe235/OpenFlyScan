"""Deterministic sliding spatial windows over Pi3X-estimated image footprints."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FootprintWindow:
    anchor_xy: tuple[float, float]
    stems_indices: tuple[int, ...]
    core_indices: tuple[int, ...]
    seam_indices: tuple[int, ...]
    scores: tuple[float, ...]


def point_to_bboxes(anchor: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    anchor=np.asarray(anchor,float).reshape(1,2)
    gap=np.maximum(np.maximum(mins-anchor,anchor-maxs),0.0)
    return np.linalg.norm(gap,axis=1)


def sample_footprint_window(
    anchor_xy: np.ndarray,
    centers: np.ndarray,
    bbox_mins: np.ndarray,
    bbox_maxs: np.ndarray,
    *,
    available_indices: np.ndarray | None = None,
    core_count: int = 21,
    seam_count: int = 9,
) -> FootprintWindow:
    anchor=np.asarray(anchor_xy,float).reshape(2)
    centers=np.asarray(centers,float);mins=np.asarray(bbox_mins,float);maxs=np.asarray(bbox_maxs,float)
    available=np.arange(len(centers),dtype=int) if available_indices is None else np.asarray(available_indices,int)
    if len(available)<core_count+seam_count:raise ValueError("fewer than 30 available spatial footprints")
    bbox_gap=point_to_bboxes(anchor,mins,maxs)
    center_distance=np.linalg.norm(centers-anchor[None,:],axis=1)
    positive=bbox_gap[available][bbox_gap[available]>1e-9]
    scale=float(np.median(positive)) if len(positive) else max(float(np.median(center_distance[available])),1.0)
    # Footprints that actually cover the anchor always rank first.  Center
    # distance breaks ties and contributes only weakly outside the footprint.
    score=bbox_gap/max(scale,1e-6)+0.05*center_distance/max(float(np.median(center_distance[available])),1e-6)
    order=available[np.lexsort((available,center_distance[available],score[available]))]
    core=order[:core_count];seam=order[core_count:core_count+seam_count];selected=np.r_[core,seam]
    return FootprintWindow(tuple(map(float,anchor)),tuple(map(int,selected)),tuple(map(int,core)),
                           tuple(map(int,seam)),tuple(map(float,score[selected])))


def shifted_anchors(base_core_indices: list[int], centers: np.ndarray, fraction: float = .18) -> list[tuple[str,np.ndarray]]:
    points=np.asarray(centers)[np.asarray(base_core_indices,int)]
    anchor=np.median(points,axis=0)
    extent=np.quantile(points,.9,axis=0)-np.quantile(points,.1,axis=0)
    global_scale=np.median(np.linalg.norm(points-anchor[None,:],axis=1))
    extent=np.maximum(extent,max(float(global_scale)*.5,1e-3))
    return [("center",anchor),("x_minus",anchor+[-fraction*extent[0],0]),
            ("x_plus",anchor+[fraction*extent[0],0]),("y_minus",anchor+[0,-fraction*extent[1]]),
            ("y_plus",anchor+[0,fraction*extent[1]])]
