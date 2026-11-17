"""Observed-only, Z-buffered rendering of 5D spatiotemporal point fields.

View synthesis is deliberately sparse: it does not hallucinate occluded pixels.
Scene flow is nearest-neighbor and thus warping is explicitly approximate.
"""
from __future__ import annotations
import numpy as np
from .geometry import validate_pose_c2w
from .reconstruction5d import Scene5D


def select_scene_time(scene:Scene5D,target_s:float)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    times=np.unique(scene.time_s)
    if (not np.isfinite(target_s) or len(times)==0 or target_s<times[0]-1e-9
        or target_s>times[-1]+1e-9):
        raise ValueError('TIME_OUTSIDE_OBSERVED_INTERVAL')
    # Select the nearest observation, avoiding extrapolation across multiple intervals.
    hi=min(int(np.searchsorted(times,target_s,side='left')),len(times)-1)
    lo=max(hi-1,0)
    idx=lo if abs(times[lo]-target_s)<abs(times[hi]-target_s) else hi
    selected=scene.time_s==times[idx]
    xyz=scene.xyz[selected].astype(np.float64).copy()
    moved=scene.matches_valid[selected]
    velocity=scene.velocity[selected]
    if moved.any():
        xyz[moved]+=(float(target_s)-times[idx])*velocity[moved]
    return xyz.astype(np.float32),scene.rgb[selected],scene.confidence_proxy[selected]


def render_scene5d(scene:Scene5D,target_s:float,K:np.ndarray,pose_c2w:np.ndarray,
                  width:int,height:int)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    if not (1<=width<=8192 and 1<=height<=8192 and width*height<=24_000_000):
        raise ValueError('RENDER_RESOURCE_LIMIT')
    k=np.asarray(K,dtype=np.float64)
    if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:
        raise ValueError('RENDER_INVALID_INTRINSICS')
    try:
        pose=validate_pose_c2w(pose_c2w)
    except ValueError as e:
        raise ValueError('RENDER_POSE_INVALID') from e
    xyz,color,_=select_scene_time(scene,target_s)
    w2c=np.linalg.inv(pose)
    c=xyz.astype(np.float64)@w2c[:3,:3].T+w2c[:3,3]
    z=c[:,2]
    valid=np.isfinite(c).all(axis=1)&(z>0)
    c=c[valid];z=z[valid];color=color[valid]
    if not len(z):
        return np.zeros((height,width,3),dtype=np.uint8),np.full((height,width),np.nan,np.float32),np.zeros((height,width),bool)
    yn=c[:,1]/z;xn=c[:,0]/z
    uu=np.rint(k[0,0]*xn+k[0,1]*yn+k[0,2]).astype(np.int64)
    vv=np.rint(k[1,1]*yn+k[1,2]).astype(np.int64)
    inside=(uu>=0)&(uu<width)&(vv>=0)&(vv<height)
    uu,vv,z,color=uu[inside],vv[inside],z[inside],color[inside]
    px=vv*width+uu
    order=np.lexsort((z,px)) # sort primarily by pixel, then by depth
    sorted_pixels=px[order]
    take=np.concatenate(([True],sorted_pixels[1:]!=sorted_pixels[:-1])) if len(sorted_pixels) else np.zeros(0,bool)
    chosen=order[take]
    image=np.zeros((height*width,3),dtype=np.uint8)
    depth=np.full((height*width,),np.nan,dtype=np.float32)
    image[px[chosen]]=color[chosen]
    depth[px[chosen]]=z[chosen].astype(np.float32)
    return image.reshape(height,width,3),depth.reshape(height,width),np.isfinite(depth.reshape(height,width))
