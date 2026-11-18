"""Bounded deterministic robust direction-to-camera fitting, separate from native DA3.

Ray order is [world direction, world origin] (pinned DA3 implementation).
Only directions enter the projective fit; do not reinterpret depth as ray length.
"""
from __future__ import annotations
import numpy as np


def _angles(d, expected):
    v=expected/np.linalg.norm(expected,axis=1,keepdims=True)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(d,v),axis=1),np.sum(d*v,axis=1)))


def fit_ray_cameras(rays, confidence, *, image_shape, max_samples=2048, angular_limit_deg=8.):
    """IRLS on a 9x9 normal matrix; returns C2W, pixel K, and diagnostics.

    Bounded sample count, clipped confidence, robust origin median and angular
    support. This is an alternative estimator, not upstream RANSAC or calibration.
    """
    a=np.asarray(rays);c=np.asarray(confidence)
    if a.ndim!=4 or a.shape[-1]!=6 or not 1<=len(a)<=8 or min(a.shape[1:3])<4 or a.shape[1]*a.shape[2]>1_000_000 or c.shape!=a.shape[:-1] or a.dtype.kind!='f' or c.dtype.kind not in 'fiu':raise ValueError('FAST_RAY_SHAPE')
    if len(image_shape)!=2 or any(not isinstance(v,(int,np.integer)) or not 4<=v<=8192 for v in image_shape) or not isinstance(max_samples,int) or not 64<=max_samples<=8192 or not np.isfinite(angular_limit_deg) or not 0<angular_limit_deg<=20:raise ValueError('FAST_RAY_LIMIT')
    h,w=map(int,image_shape);rh,rw=a.shape[1:3]
    # Stratified 2D grid avoids aliasing when a flat stride is a multiple of width.
    ny=min(rh,max(4,int(np.sqrt(max_samples*rh/rw))));nx=min(rw,max(4,max_samples//ny));ny=min(ny,max_samples//nx)
    yy,xx=np.meshgrid(np.linspace(0,rh-1,ny,dtype=int),np.linspace(0,rw-1,nx,dtype=int),indexing='ij');yy=yy.ravel();xx=xx.ravel()
    pixel=np.stack([2*(xx+.5)/rw-1,2*(yy+.5)/rh-1,np.ones(len(xx))],axis=-1)
    pix_transform=np.array([[w/2,0,w/2],[0,h/2,h/2],[0,0,1.]])
    poses=[];intrinsics=[];rows=[]
    for i in range(len(a)):
        sample=a[i,yy,xx].astype(float);conf=c[i,yy,xx].astype(float);d=sample[:,:3];o=sample[:,3:];norm=np.linalg.norm(d,axis=1)
        ok=np.isfinite(sample).all(1)&(np.abs(sample).max(1)<1e12)&np.isfinite(conf)&(conf>0)&(norm>1e-9)
        if ok.sum()<max(32,int(np.ceil(.8*len(ok)))):raise ValueError('FAST_RAY_SUPPORT')
        p=pixel[ok];d=d[ok]/norm[ok,None];o=o[ok];conf=conf[ok]
        if np.ptp(p[:,0])<1.5 or np.ptp(p[:,1])<1.5:raise ValueError('FAST_RAY_SPATIAL_SUPPORT')
        base=np.clip(conf/np.median(conf),.1,3.);weights=base.copy()
        # Each ray constrains (I-dd^T) H p = 0. H is 3x3 in row-major order.
        proj=np.eye(3)[None]-d[:,:,None]*d[:,None,:]
        design=(proj[:,:,:,None]*p[:,None,None,:]).reshape(-1,3,9)
        for _ in range(7):
            gram=np.einsum('nri,nrj,n->ij',design,design,weights,optimize=True)
            eig,vec=np.linalg.eigh(gram)
            if eig[-1]<=0 or eig[1]<eig[-1]*1e-8:raise ValueError('FAST_RAY_DEGENERATE')
            H=vec[:,0].reshape(3,3)
            if np.linalg.det(H)<0:H=-H
            if np.linalg.cond(H)>1e4:raise ValueError('FAST_RAY_ILL_CONDITIONED')
            angular=_angles(d,p@H.T);cut=max(.5,2.5*float(np.median(angular)))
            weights=base/(1+(angular/cut)**2)**2
        R,U=np.linalg.qr(H);sign=np.where(np.diag(U)<0,-1.,1.);R=R*sign;U=U*sign[:,None]
        if np.linalg.det(R)<0:raise ValueError('FAST_RAY_IMPROPER_ROTATION')
        K=pix_transform@np.linalg.inv(U);K/=K[2,2]
        expected=(p@np.linalg.inv(np.linalg.inv(pix_transform)@K).T)@R.T
        angular=_angles(d,expected);p90=float(np.quantile(angular,.9))
        if p90>angular_limit_deg or not np.isfinite(K).all() or not .05*w<=K[0,0]<=20*w or not .05*h<=K[1,1]<=20*h or not -w<=K[0,2]<=2*w or not -h<=K[1,2]<=2*h:raise ValueError('FAST_RAY_QUALITY_GATE')
        center=np.median(o,axis=0);pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=center
        poses.append(pose);intrinsics.append(K);rows.append({'view':i,'samples':len(ok),'valid':int(ok.sum()),'angular_p90_deg':p90,'origin_p90':float(np.quantile(np.linalg.norm(o-center,axis=1),.9))})
    return np.stack(poses),np.stack(intrinsics),{'per_view':rows,'worst_angular_p90_deg':max(r['angular_p90_deg'] for r in rows),'scope':'internal ray fit only; relative gauge; not physical accuracy'}
