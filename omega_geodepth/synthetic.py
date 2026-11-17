"""Deterministic synthetic ray-cast dataset; no pretrained weights, no web data."""
from __future__ import annotations
import numpy as np


def generate_scene(seed:int,size:tuple[int,int]=(64,64)) -> tuple[np.ndarray,np.ndarray]:
    rng=np.random.default_rng(seed)
    h,w=size
    yy,xx=np.mgrid[0:h,0:w].astype(np.float32)
    x=(xx-(w-1)*.5)/(w*.85)
    y=(yy-(h-1)*.5)/(h*.85)
    # Background tilted plane, depth increases with screen y and x.
    bz=float(rng.uniform(5,9))
    sx=float(rng.uniform(-.4,.4)); sy=float(rng.uniform(-.4,.4))
    depth=(bz/(1+sx*x+sy*y)).astype(np.float32)
    # Perspective texture and lit wall; background is not a depth-channel encoding.
    tile=np.sin(x*depth*rng.uniform(3,7))*np.sin(y*depth*rng.uniform(3,7))
    base=rng.uniform(.17,.65,size=3).astype(np.float32)
    rgb=np.ones((h,w,3),dtype=np.float32)*base
    rgb=np.clip(rgb*(0.78+.15*tile[...,None]+.12*(1-y[...,None])) ,0,1)
    # Independent foreground primitive materials; shape and shadows cue depth.
    for _ in range(int(rng.integers(2,6))):
        u=float(rng.uniform(-.4,.4)); v=float(rng.uniform(-.35,.35))
        z=float(rng.uniform(2.,5.7)); radius=float(rng.uniform(.35,1.25))
        a=1+x*x+y*y
        b=-2*(x*u+y*v+z)
        c=u*u+v*v+z*z-radius*radius
        disc=b*b-4*a*c
        t=np.where(disc>0,(-b-np.sqrt(np.maximum(disc,0)))/(2*a),np.inf)
        mask=(t>0)&(t<depth)
        safe_t=np.where(np.isfinite(t),t,0.0)
        px=x*safe_t-u; py=y*safe_t-v; pz=safe_t-z
        nx=px/radius;ny=py/radius;nz=pz/radius
        diffuse=np.maximum(0,.35*nx-.45*ny-.82*nz)
        highlight=np.maximum(0, .4*nx-.3*ny-.85*nz)**15
        color=rng.uniform(.1,.9,size=3).astype(np.float32)
        shaded=np.clip((.27+.74*diffuse)[...,None]*color+.2*highlight[...,None],0,1)
        rgb=np.where(mask[...,None],shaded,rgb)
        depth=np.where(mask,t,depth)
    # One to three fronto-parallel flat panels as occluders.
    for _ in range(int(rng.integers(1,3))):
        left=int(rng.integers(0,max(1,w-12)))
        top=int(rng.integers(0,max(1,h-12)))
        rw=int(rng.integers(10,max(11,w//2)))
        rh=int(rng.integers(10,max(11,h//2)))
        z=float(rng.uniform(2.5,6.))
        panel=(xx>=left)&(xx<left+rw)&(yy>=top)&(yy<top+rh)&(z<depth)
        color=rng.uniform(.15,.88,size=3).astype(np.float32)
        shading=.75+.14*np.clip((xx-left)/max(1,rw),0,1)
        rgb=np.where(panel[...,None],np.clip(shading[...,None]*color,0,1),rgb)
        depth=np.where(panel,z,depth)
    rgb=np.clip(rgb+rng.normal(0,.012,size=rgb.shape),0,1).astype(np.float32)
    return rgb.astype(np.float32), depth.astype(np.float32)


def log_relative_depth(z:np.ndarray)->np.ndarray:
    d=np.log(z.astype(np.float32))
    return d-np.mean(d,axis=(-2,-1),keepdims=True)
