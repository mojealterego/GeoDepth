"""Opt-in native DA3 auxiliary features for the experimental Gaussian head.

Layer 11 exports normalized patch features (not the concatenated DPT tokens).
Use only source/training views during feature extraction; DA3 is multi-view.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from .gsdpt_bridge import GaussianInput, prepare_gaussians

if TYPE_CHECKING:
    from .da3_adapter import DA3Output

CHANNELS = {'DA3-SMALL': 384, 'DA3-BASE': 768}
PATCH = 14
LAYER = 11
MAX_SIDE = 504


def forward_with_backbone(net, x: torch.Tensor, ext, intr, model_id: str, ray: bool):
    """Run one native forward and copy the bounded feature grid to CPU."""
    if model_id not in CHANNELS or x.ndim != 5:
        raise ValueError('invalid feature model or input rank')
    b,n,c,h,w = x.shape
    if b != 1 or not 1 <= n <= 8 or c != 3 or not all(PATCH <= d <= MAX_SIDE and d % PATCH == 0 for d in (h,w)):
        raise ValueError('invalid feature input shape or resource limit')
    raw = net(x, ext, intr, [LAYER], False, ray, 'middle')
    aux = raw.get('aux') if isinstance(raw, Mapping) else None
    key = f'feat_layer_{LAYER}'
    if not isinstance(aux, Mapping) or key not in aux:
        raise ValueError('missing native auxiliary features')
    t = aux[key]
    if not isinstance(t, torch.Tensor) or not t.is_floating_point() or t.shape != (b,n,h//PATCH,w//PATCH,CHANNELS[model_id]):
        raise ValueError('invalid native feature shape')
    grid = t[0].detach().to(device='cpu',dtype=torch.float32).contiguous().numpy().copy()
    if not np.isfinite(grid).all():
        raise ValueError('nonfinite native features')
    grid.setflags(write=False)
    return raw, grid


def prepare_backbone_gaussians(prediction: DA3Output, index: int, *, max_points: int = 512) -> GaussianInput:
    """Append the corresponding native patch descriptor to each sampled ray."""
    model_id = prediction.model_id
    z = prediction.depth_z
    grid = prediction.backbone_features
    if model_id not in CHANNELS or z.ndim != 3 or not isinstance(grid,np.ndarray):
        raise ValueError('missing or invalid backbone features')
    n,h,w = z.shape
    if not 1 <= n <= 8 or not all(PATCH <= d <= MAX_SIDE and d % PATCH == 0 for d in (h,w)) or grid.dtype != np.float32 or grid.shape != (n,h//PATCH,w//PATCH,CHANNELS[model_id]):
        raise ValueError('feature alignment mismatch')
    base = prepare_gaussians(prediction,index,max_points=max_points)
    valid = np.flatnonzero(np.isfinite(z[index]) & (z[index]>0))
    count = len(base.depth)
    ids = valid[np.arange(count,dtype=np.int64)*len(valid)//count]
    y,x = np.divmod(ids,w)
    descriptors = torch.from_numpy(grid[index,y//PATCH,x//PATCH].copy())
    if not bool(torch.isfinite(descriptors).all()):
        raise ValueError('nonfinite sampled features')
    return replace(base, features=torch.cat((base.features,descriptors),dim=1))
