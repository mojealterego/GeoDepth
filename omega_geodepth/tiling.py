"""Bounded-tile RGB inference with global conditioning and smooth residual fusion.

A simple, auditable approximation to coarse-to-fine tiling, NOT PatchFusion itself.
Only synthesizes relative depth; no real-world metric claims. Inputs are float32 RGB
already decoded into memory, so this is NOT a 400MP out-of-core image decoder.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _starts(length: int, tile: int, step: int) -> list[int]:
    if length <= tile:
        return [0]
    stops=list(range(0,length-tile+1,step))
    if stops[-1]!=length-tile:
        stops.append(length-tile)
    return stops


def _pad8(x: np.ndarray) -> tuple[np.ndarray,int,int]:
    h,w=x.shape[:2]
    ph=(-h)%8; pw=(-w)%8
    if ph or pw:
        x=np.pad(x,((0,ph),(0,pw),(0,0)),mode='edge')
    return x,h,w


def _forward(model:torch.nn.Module,image:np.ndarray,device:str='cpu')->tuple[np.ndarray,np.ndarray]:
    rgb,h,w=_pad8(image)
    tensor=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1)[None])).to(device)
    with torch.inference_mode():
        logz,logvar=model(tensor)
    outz=logz[0,0,:h,:w].detach().to('cpu',dtype=torch.float32).numpy()
    outv=logvar[0,0,:h,:w].detach().to('cpu',dtype=torch.float32).numpy()
    if not np.isfinite(outz).all() or not np.isfinite(outv).all():
        raise ValueError('NONFINITE_TILE_OUTPUT')
    return outz,outv


def _resize(array:np.ndarray,size:tuple[int,int],mode:str='bilinear')->np.ndarray:
    h,w=size
    tensor=torch.from_numpy(np.ascontiguousarray(array))[None,None].float()
    return F.interpolate(tensor,size=(h,w),mode=mode,align_corners=False)[0,0].numpy()


def tiled_predict(model:torch.nn.Module,image:np.ndarray,*,tile_size:int=256,
                  overlap:int=64,global_side:int=256,device:str='cpu',
                  correction_cap:float=.65,correction_weight:float=.35)->tuple[np.ndarray,np.ndarray]:
    """Tile predictions aligned to a global guide in log-depth space.

    Relative scale is fixed *only* inside the image. Log-variances are fused as
    a conservative mixture of total tile variance and correction disagreement.
    Numbers are empirical and not confidence intervals without calibration.
    """
    image=np.asarray(image)
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.float32 or not np.isfinite(image).all():
        raise ValueError('INVALID_RGB_ARRAY')
    h,w=image.shape[:2]
    if min(h,w)<8:
        raise ValueError('IMAGE_TOO_SMALL')
    if (tile_size<32 or tile_size>1024 or tile_size%8!=0 or
        overlap<8 or overlap>=tile_size or overlap%8!=0 or
        global_side<32 or global_side>1024 or global_side%8!=0 or
        not 0<=correction_weight<=1 or correction_cap<=0):
        raise ValueError('INVALID_TILE_CONFIGURATION')
    if tile_size*tile_size>1_048_576:
        raise ValueError('INVALID_TILE_RESOURCE_BUDGET')
    # Use memory-bounded network inputs: global thumbnail max dimension <= global_side.
    ratio=min(1.0,global_side/max(h,w))
    gh=max(8,round(h*ratio/8)*8)
    gw=max(8,round(w*ratio/8)*8)
    if (gh,gw)==(h,w):
        thumb=image
    else:
        small=F.interpolate(torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1)[None])),
                            size=(gh,gw),mode='area')
        thumb=small[0].permute(1,2,0).numpy().astype(np.float32)
    global_log,global_var=_forward(model,thumb,device)
    guide=_resize(global_log,(h,w))
    guide_var=_resize(global_var,(h,w))
    if h<=tile_size and w<=tile_size:
        # A direct native-size inference is never worse than interpolating a thumbnail.
        direct_log,direct_var=_forward(model,image,device)
        return direct_log,direct_var
    total=np.zeros((h,w),dtype=np.float32)
    weights=np.zeros((h,w),dtype=np.float32)
    var_total=np.zeros((h,w),dtype=np.float32)
    # A positive window, including at image borders, avoids zero-weight corners.
    step=tile_size-overlap
    for y in _starts(h,tile_size,step):
        for x in _starts(w,tile_size,step):
            tile=image[y:y+tile_size,x:x+tile_size]
            hh,ww=tile.shape[:2]
            local,logvar=_forward(model,tile,device)
            reference=guide[y:y+hh,x:x+ww]
            delta=local-reference
            # Remove the arbitrary per-crop scale gauge. Robust for depth discontinuities.
            delta=delta-np.median(delta)
            correction=correction_weight*np.clip(delta,-correction_cap,correction_cap)
            yy=np.arange(hh,dtype=np.float32)
            xx=np.arange(ww,dtype=np.float32)
            # cosine ramps clipped to a positive floor at the external border.
            blend_h=np.maximum(.03,np.sin(np.pi*(yy+.5)/hh)**2)
            blend_w=np.maximum(.03,np.sin(np.pi*(xx+.5)/ww)**2)
            weight=blend_h[:,None]*blend_w[None,:]
            total[y:y+hh,x:x+ww]+=weight*correction
            # Within-tile uncertainty plus disagreement with global guide.
            var_total[y:y+hh,x:x+ww]+=weight*(np.exp(np.clip(logvar,-14,7))+correction**2)
            weights[y:y+hh,x:x+ww]+=weight
    if (weights<=0).any():
        raise RuntimeError('TILING_UNCOVERED_PIXEL')
    result=guide+total/weights
    # Do not let blending cause calibrated uncertainty to shrink artificially.
    final_var=np.log(np.maximum(np.exp(np.clip(guide_var,-14,7)),var_total/weights))
    # The network emits centered logarithmic depth on each input frame.
    result-=result.mean()
    return result.astype(np.float32),final_var.astype(np.float32)
