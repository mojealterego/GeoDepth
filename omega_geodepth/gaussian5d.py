"""Finite, observed-only 5D Gaussian surfel renderer (CPU).

Not a pretrained 4DGS/5D radiance model. Each observed surfel stores XYZ,
observation timestamp, approximate sRGB appearance and nearest-neighbor motion.
No occluded geometry is synthesized; output alpha expresses raster coverage only.
"""
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import numpy as np
from .geometry import validate_pose_c2w
from .reconstruction5d import Scene5D
from .infer import save_atomic

@dataclass(frozen=True)
class GaussianField5D:
    means_xyz:np.ndarray
    times:np.ndarray
    rgb:np.ndarray
    sigma_world:float
    confidence:np.ndarray
    velocity:np.ndarray
    velocity_valid:np.ndarray
    units:str
    provenance:str


def gaussian_from_scene(scene:Scene5D,*,sigma_world:float|None=None,
                        max_points:int=50000)->GaussianField5D:
    if not (1<=max_points<=200000):raise ValueError('GAUSSIAN_RESOURCE_LIMIT')
    n=len(scene.xyz)
    if n==0 or n>max_points:raise ValueError('GAUSSIAN_RESOURCE_LIMIT')
    width=float(scene.voxel_size if sigma_world is None else sigma_world)
    if not np.isfinite(width) or not .00001<=width<=1000:raise ValueError('GAUSSIAN_BAD_SIGMA')
    if np.asarray(scene.xyz).shape!=(n,3) or np.asarray(scene.rgb).shape!=(n,3):
        raise ValueError('GAUSSIAN_INVALID_FIELD')
    if not np.isfinite(scene.xyz).all() or not np.isfinite(scene.time_s).all():
        raise ValueError('GAUSSIAN_NONFINITE_FIELD')
    return GaussianField5D(
        np.asarray(scene.xyz,np.float32),np.asarray(scene.time_s,np.float64),
        np.asarray(scene.rgb,np.uint8),width,
        np.clip(np.nan_to_num(scene.confidence_proxy.astype(np.float32),nan=0.),0,1),
        np.asarray(scene.velocity,np.float32),np.asarray(scene.matches_valid,bool),
        scene.units,scene.provenance)


def _at_time(g:GaussianField5D,t:float):
    unique=np.unique(g.times)
    if not np.isfinite(t) or t<unique[0]-1e-9 or t>unique[-1]+1e-9:
        raise ValueError('TIME_OUTSIDE_OBSERVED_INTERVAL')
    closest=int(np.argmin(np.abs(unique-t)))
    idx=np.flatnonzero(g.times==unique[closest])
    pos=g.means_xyz[idx].astype(np.float64).copy()
    matched=g.velocity_valid[idx] & np.isfinite(g.velocity[idx]).all(axis=1)
    pos[matched]+=(t-unique[closest])*g.velocity[idx[matched]]
    return pos,g.rgb[idx],g.confidence[idx]


def render_gaussian5d(g:GaussianField5D,target_s:float,K:np.ndarray,
                      pose_c2w:np.ndarray,width:int,height:int,
                      *,base_opacity:float=.85,max_radius_px:int=7):
    """Project finite Gaussian surfels and alpha-composite them front to back.

    Return (uint8 RGB, float32 alpha, float32 nearest *optical Z*). Pixels with
    no observation remain alpha=0 and NaN Z (not inferred geometry).
    """
    if not 1<=width<=2048 or not 1<=height<=2048 or width*height>1_000_000:
        raise ValueError('GAUSSIAN_RENDER_RESOURCE_LIMIT')
    if not .01<=base_opacity<=1 or not 1<=max_radius_px<=12:
        raise ValueError('GAUSSIAN_BAD_RENDER_PARAMETERS')
    k=np.asarray(K,dtype=np.float64)
    if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0 or not np.allclose(k[2],[0,0,1],atol=1e-6):
        raise ValueError('GAUSSIAN_INVALID_INTRINSICS')
    pose=validate_pose_c2w(pose_c2w)
    xyz,color,conf=_at_time(g,target_s)
    w2c=np.linalg.inv(pose)
    camera=xyz@w2c[:3,:3].T+w2c[:3,3]
    zz=camera[:,2]
    usable=np.isfinite(camera).all(axis=1)&(zz>1e-6)&(conf>0)
    camera,zz,color,conf=camera[usable],zz[usable],color[usable],conf[usable]
    image=np.zeros((height,width,3),np.float64)
    alpha=np.zeros((height,width),np.float64)
    depth=np.full((height,width),np.nan,np.float32)
    if not len(zz):return image.astype(np.uint8),alpha.astype(np.float32),depth
    x=camera[:,0]/zz;y=camera[:,1]/zz
    u=k[0,0]*x+k[0,1]*y+k[0,2]
    v=k[1,1]*y+k[1,2]
    sigma=np.clip(g.sigma_world*np.sqrt(abs(k[0,0]*k[1,1]))/zz,.55,max_radius_px/2)
    # Point splats are depth sorted; the first visible sample contributes first.
    for i in np.argsort(zz,kind='stable'):
        radius=min(max_radius_px,int(np.ceil(2.6*sigma[i])))
        cx=int(np.rint(u[i]));cy=int(np.rint(v[i]))
        x0=max(0,cx-radius);x1=min(width,cx+radius+1)
        y0=max(0,cy-radius);y1=min(height,cy+radius+1)
        if x0>=x1 or y0>=y1:continue
        yy,xx=np.ogrid[y0:y1,x0:x1]
        a=np.clip(base_opacity*conf[i]*np.exp(-((xx-u[i])**2+(yy-v[i])**2)/(2*sigma[i]**2)),0,.999)
        trans=1-alpha[y0:y1,x0:x1]
        weight=trans*a
        patch=image[y0:y1,x0:x1]
        patch+=weight[...,None]*color[i].astype(float)
        alpha[y0:y1,x0:x1]+=weight
        missing=np.isnan(depth[y0:y1,x0:x1])&(weight>.01)
        depth[y0:y1,x0:x1][missing]=float(zz[i])
    return np.rint(image).clip(0,255).astype(np.uint8),alpha.astype(np.float32),depth


def export_gaussian5d(g:GaussianField5D,output:Path)->None:
    bio=BytesIO()
    np.savez_compressed(bio,means_xyz=g.means_xyz,observation_time_s=g.times,
                        rgb_srgb=g.rgb,sigma_world=np.float64(g.sigma_world),
                        confidence_proxy=g.confidence,velocity_nearest_neighbor=g.velocity,
                        velocity_valid=g.velocity_valid,units=np.array(g.units),
                        provenance=np.array(g.provenance),
                        representation=np.array('observed_gaussian_surfels_not_trained_4DGS'))
    save_atomic(Path(output),bio.getvalue())
