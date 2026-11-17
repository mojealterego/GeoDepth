"""Deterministic confidence-filtered point selection and voxel fusion."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .prediction import Prediction
from .geometry import backproject_pixels


def confidence_mask(pred: Prediction, quantile: float = 0.1) -> np.ndarray:
    if not 0<=quantile<1:
        raise ValueError('confidence quantile must be in [0,1)')
    valid=pred.valid
    result=np.zeros_like(valid)
    for i in range(len(valid)):
        if valid[i].any():
            cutoff=np.quantile(pred.conf[i][valid[i]],quantile)
            result[i]=valid[i] & (pred.conf[i]>=cutoff)
    return result


def collect_points(pred: Prediction, *, max_points: int = 1_000_000,
                   confidence_quantile: float = 0.1,
                   consistency: dict | None = None,
                   min_support: int = 0,
                   min_score: float = 0.) -> tuple[np.ndarray,dict]:
    if max_points < 1 or max_points > 10_000_000:
        raise ValueError('max_points must be between 1 and 10 million')
    if min_support < 0 or min_score < 0 or min_score > 1:
        raise ValueError('invalid geometric filter')
    mask=confidence_mask(pred,confidence_quantile)
    if consistency is not None:
        if min_support:
            mask &= consistency['support']>=min_support
        if min_score:
            mask &= consistency['score']>=min_score
    counts=mask.sum(axis=(1,2)).astype(np.int64)
    total=int(counts.sum())
    if total==0:
        return np.empty((0,3),dtype=np.float32),{'eligible':0,'selected':0}
    take=min(max_points,total)
    selected=np.arange(take,dtype=np.int64) * total // take
    cumulative=np.cumsum(counts)
    h,w=pred.depth.shape[1:]
    chunks=[]
    for i in range(len(counts)):
        if counts[i]==0:
            continue
        left=int(cumulative[i]-counts[i]);right=int(cumulative[i])
        choice=selected[(selected>=left)&(selected<right)]-left
        if not len(choice):
            continue
        flat=np.flatnonzero(mask[i].ravel())[choice]
        y=flat//w;x=flat%w
        chunks.append(backproject_pixels(pred,i,y,x))
    points=np.concatenate(chunks,axis=0) if chunks else np.empty((0,3),dtype=np.float32)
    return points,{'eligible':total,'selected':len(points),'sampling':'deterministic_even_indices'}


def voxel_fuse(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if not np.isfinite(voxel_size) or voxel_size<=0:
        raise ValueError('voxel_size must be positive and finite')
    pts=np.asarray(points,dtype=np.float32)
    if pts.ndim!=2 or pts.shape[1]!=3 or not np.isfinite(pts).all():
        raise ValueError('points must be finite XYZ coordinates')
    if not len(pts):
        return pts
    with np.errstate(over='ignore',invalid='ignore'):
        coords=np.floor(pts.astype(np.float64)/voxel_size)
    if not np.isfinite(coords).all() or np.any(coords < -2**63) or np.any(coords >= 2**63):
        raise ValueError('voxel index overflow')
    grid=coords.astype(np.int64)
    _,inverse=np.unique(grid,axis=0,return_inverse=True)
    size=int(inverse.max())+1
    count=np.bincount(inverse,minlength=size)
    result=np.column_stack([np.bincount(inverse,weights=pts[:,k],minlength=size)/count
                            for k in range(3)])
    return result.astype(np.float32)


def write_binary_ply(path: str | Path, points: np.ndarray) -> None:
    path=Path(path)
    pts=np.asarray(points,dtype='<f4')
    if pts.ndim!=2 or pts.shape[1]!=3 or not np.isfinite(pts).all():
        raise ValueError('invalid point cloud')
    header=('ply\nformat binary_little_endian 1.0\n'
            'comment omega_geodepth: coordinate scale described in report.json\n'
            f'element vertex {len(pts)}\n'
            'property float x\nproperty float y\nproperty float z\nend_header\n')
    with path.open('wb') as f:
        f.write(header.encode('ascii'))
        f.write(pts.tobytes(order='C'))
