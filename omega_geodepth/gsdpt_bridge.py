"""Bridge normalized DA3 output to a bounded RGB/ray Gaussian head.

This is post-inference geometry, not a DA3 backbone feature extractor. Relative
units are preserved; invalid depth is never filled with invented geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .gsdpt import camera

if TYPE_CHECKING:
    from .da3_adapter import DA3Output


@dataclass(frozen=True)
class GaussianInput:
    features: torch.Tensor
    rays: torch.Tensor
    depth: torch.Tensor
    pose: torch.Tensor
    K: torch.Tensor
    units: str


def prepare_gaussians(prediction: DA3Output, index: int, *, max_points: int = 512) -> GaussianInput:
    if not 1 <= max_points <= 512 or prediction.units != 'arbitrary_scale':
        raise ValueError('invalid limit or units')
    z, rgb = prediction.depth_z, prediction.rgb
    if z.ndim != 3 or not 0 <= index < z.shape[0] <= 8 or not 0 < z.shape[1]*z.shape[2] <= 4_000_000:
        raise ValueError('invalid depth shape or index')
    if not np.issubdtype(z.dtype, np.floating) or rgb.shape != (*z.shape, 3) or rgb.dtype != np.uint8:
        raise ValueError('invalid depth or RGB contract')
    if prediction.intrinsics_k.shape != (len(z),3,3) or prediction.pose_c2w.shape != (len(z),4,4):
        raise ValueError('invalid camera batch')
    k = np.asarray(prediction.intrinsics_k[index], dtype=np.float64)
    pose = torch.tensor(prediction.pose_c2w[index], dtype=torch.float32)
    camera(pose)
    if not np.isfinite(k).all() or min(k[0,0],k[1,1]) <= 0 or k[1,0] != 0 or not np.allclose(k[2],[0,0,1]):
        raise ValueError('invalid intrinsics')
    valid = np.flatnonzero(np.isfinite(z[index]) & (z[index]>0))
    if not len(valid):
        raise ValueError('no observed depth')
    count = min(len(valid), max_points)
    ids = valid[np.arange(count, dtype=np.int64)*len(valid)//count]
    y,x = np.divmod(ids,z.shape[2])
    ry = (y-k[1,2])/k[1,1]
    rx = (x-k[0,2]-k[0,1]*ry)/k[0,0]
    rays = torch.tensor(np.stack((rx,ry,np.ones_like(rx)),1),dtype=torch.float32)
    depth = torch.tensor(z[index,y,x],dtype=torch.float32)
    colors = torch.tensor(rgb[index,y,x],dtype=torch.float32)/255
    K = torch.tensor(k,dtype=torch.float32)
    if not all(bool(torch.isfinite(t).all()) for t in (rays,depth,K)) or not bool((depth>0).all()) or not bool((K.diag()[:2]>0).all()):
        raise ValueError('camera or depth exceeds float32 range')
    return GaussianInput(torch.cat((colors,rays[:,:2]),1),rays,depth,pose,K,prediction.units)
