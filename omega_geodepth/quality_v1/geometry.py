"""Calibrated pinhole geometry and visibility-aware cross-view validation.

Depth is camera Z-depth, not Euclidean ray distance. Camera poses are W2C.
Coordinate units are arbitrary unless anchored to independent metric measurements.
"""
from __future__ import annotations
import numpy as np
from .prediction import Prediction


def backproject_pixels(pred: Prediction, view: int, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    ys = np.asarray(ys)
    xs = np.asarray(xs)
    if ys.shape != xs.shape:
        raise ValueError('Pixel coordinate shapes differ')
    h, w = pred.depth.shape[1:]
    if np.any(ys < 0) or np.any(ys >= h) or np.any(xs < 0) or np.any(xs >= w):
        raise ValueError('Pixel coordinates outside image')
    z = pred.depth[view,ys,xs]
    pixels = np.stack((xs,ys,np.ones_like(xs)),axis=-1).astype(np.float64)
    rays = pixels @ np.linalg.inv(pred.intrinsics[view].astype(np.float64)).T
    camera = rays * z[...,None]
    rot = pred.extrinsics[view,:,:3].astype(np.float64)
    trans = pred.extrinsics[view,:,3].astype(np.float64)
    world = (camera-trans) @ rot
    world = world.astype(np.float32)
    world[~np.isfinite(z) | (z<=0)] = np.nan
    return world


def backproject_world(pred: Prediction, view: int) -> np.ndarray:
    h,w = pred.depth.shape[1:]
    ys,xs = np.indices((h,w))
    return backproject_pixels(pred,view,ys,xs)


def project_world(points: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray):
    points = np.asarray(points,dtype=np.float64)
    rot = np.asarray(extrinsics[:,:3],dtype=np.float64)
    trans = np.asarray(extrinsics[:,3],dtype=np.float64)
    camera = points @ rot.T + trans
    uvw = camera @ np.asarray(intrinsics,dtype=np.float64).T
    z = camera[...,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        u = uvw[...,0] / uvw[...,2]
        v = uvw[...,1] / uvw[...,2]
    return u,v,z


def _targets(source: int, views: int, budget: int) -> list[int]:
    if budget<1:
        raise ValueError('max_targets_per_view must be >= 1')
    nearby=[source+d for d in (-1,1,-2,2) if 0<=source+d<views]
    global_samples=np.linspace(0,views-1,min(views,budget*2),dtype=int).tolist()
    candidates=nearby+global_samples+list(range(views))
    result=[]
    for target in candidates:
        if target!=source and target not in result:
            result.append(target)
            if len(result)==budget:
                break
    return result


def multiview_consistency(pred: Prediction, relative_tolerance: float = 0.05,
                          max_targets_per_view: int = 8) -> dict[str,np.ndarray]:
    if max_targets_per_view<1 or max_targets_per_view>64:
        raise ValueError('max_targets_per_view must be between 1 and 64')
    if not 0 < relative_tolerance < 1:
        raise ValueError('relative_tolerance must be in (0,1)')
    pred.validate()
    n,h,w=pred.depth.shape
    valid_pixels=pred.valid
    support=np.zeros((n,h,w),dtype=np.uint16)
    tested=np.zeros_like(support)
    for source in range(n):
        mask=valid_pixels[source]
        if not mask.any():
            continue
        sy,sx=np.nonzero(mask)
        world=backproject_pixels(pred,source,sy,sx)
        for target in _targets(source,n,max_targets_per_view):
            u,v,z=project_world(world,pred.extrinsics[target],pred.intrinsics[target])
            in_bounds=(np.isfinite(u)&np.isfinite(v)&np.isfinite(z)&(z>0)&
                       (u>=0)&(u<w)&(v>=0)&(v<h))
            if not in_bounds.any():
                continue
            idx=np.flatnonzero(in_bounds)
            tx=np.floor(u[idx]+.5).astype(np.int64)
            ty=np.floor(v[idx]+.5).astype(np.int64)
            in_round=(tx>=0)&(tx<w)&(ty>=0)&(ty<h)
            idx=idx[in_round];tx=tx[in_round];ty=ty[in_round]
            if len(idx)==0:
                continue
            target_z=pred.depth[target,ty,tx]
            valid_target=valid_pixels[target,ty,tx]
            # A source point behind an observed target surface is occluded, not contradicted.
            occluded=z[idx] > target_z*(1+relative_tolerance)
            check=valid_target & ~occluded
            if not check.any():
                continue
            idx_checked=idx[check]
            tested[source,sy[idx_checked],sx[idx_checked]]+=1
            diff=np.abs(z[idx_checked]-target_z[check])
            threshold=relative_tolerance*np.maximum(z[idx_checked],target_z[check])
            agreed=diff<=threshold
            support[source,sy[idx_checked[agreed]],sx[idx_checked[agreed]]]+=1
    score=np.full((n,h,w),np.nan,dtype=np.float32)
    np.divide(support,tested,out=score,where=tested>0)
    return {'support':support,'tested':tested,'score':score}


def camera_baseline_ratio(pred: Prediction) -> float:
    """Largest camera-center separation divided by median valid depth.

    This is a dimensionless parallax proxy, NOT a triangulation quality proof.
    """
    n=pred.depth.shape[0]
    if n<2:
        return 0.
    rotation=pred.extrinsics[:,:,:3].astype(np.float64)
    translation=pred.extrinsics[:,:,3].astype(np.float64)
    centers=-np.einsum('nji,nj->ni',rotation,translation)
    maximum=0.
    for i in range(n):
        distances=np.linalg.norm(centers[i+1:]-centers[i],axis=1)
        if len(distances):
            maximum=max(maximum,float(distances.max()))
    depth=pred.depth[pred.valid]
    if not len(depth):
        return 0.
    median=float(np.median(depth))
    return maximum/median if median>0 else 0.
