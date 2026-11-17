"""Bounded, trainable time-conditioned 3D Gaussian field (4DGS research baseline).

Each observed 3D Gaussian has a learned position, linear world-space velocity,
scale, color and opacity. Time is a conditioning variable (x, y, z, t), not an
extra spatial dimension. Render uses differentiable front-to-back Gaussian
alpha compositing with *spherical* 3D splats projected through a calibrated
pinhole camera. This is NOT the full deformable 4DGS paper architecture or a
photorealistic NeRF; no unobserved geometry is invented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file, load_file

from .geometry import validate_pose_c2w
from .reconstruction5d import Frame5D, _validate_frame, load_frame_manifest
from .infer import save_atomic

MAX_SPLATS = 512
MAX_RENDER_PIXELS = 16384
MAX_SPLAT_PIXELS = 2_000_000
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024


def _intrinsics(K: np.ndarray) -> np.ndarray:
    k=np.asarray(K,dtype=np.float64)
    if (k.shape!=(3,3) or not np.isfinite(k).all() or
            min(k[0,0],k[1,1])<=0 or
            not np.allclose(k[1,0],0,atol=1e-8) or
            not np.allclose(k[2,:],[0.,0.,1.],atol=1e-8)):
        raise ValueError('GAUSSIAN_INVALID_INTRINSICS')
    return k


def _inverse_logit(x:torch.Tensor)->torch.Tensor:
    return torch.logit(x.clamp(min=1e-5,max=1-1e-5))


class TrainableGaussian4D(nn.Module):
    """Time-dependent observed Gaussian field with differentiable renderer."""

    def __init__(self,means_xyz:torch.Tensor,velocity:torch.Tensor,rgb:torch.Tensor,
                 sigma_world:torch.Tensor,opacity:torch.Tensor,reference_time:float):
        super().__init__()
        n=int(means_xyz.shape[0])
        if not (1<=n<=MAX_SPLATS):raise ValueError('GAUSSIAN_SPLAT_RESOURCE_LIMIT')
        data=(means_xyz,velocity,rgb,sigma_world,opacity)
        shapes=((n,3),(n,3),(n,3),(n,1),(n,1))
        if any(tuple(x.shape)!=shape for x,shape in zip(data,shapes)):
            raise ValueError('GAUSSIAN_INVALID_PARAMETER_SHAPE')
        if any(not bool(torch.isfinite(x).all()) for x in data):
            raise ValueError('GAUSSIAN_NONFINITE_PARAMETER')
        if (not bool((sigma_world>0).all()) or
            not bool(((rgb>=0)&(rgb<=1)).all()) or
            not bool(((opacity>0)&(opacity<1)).all()) or
            not math.isfinite(reference_time)):
            raise ValueError('GAUSSIAN_INVALID_PARAMETER_RANGE')
        self.reference_time=float(reference_time)
        self.means_xyz=nn.Parameter(means_xyz.detach().float().clone())
        self.velocity=nn.Parameter(velocity.detach().float().clone())
        self.log_sigma=nn.Parameter(sigma_world.detach().float().log().clone())
        self.rgb_logits=nn.Parameter(_inverse_logit(rgb.detach().float().clone()))
        self.opacity_logits=nn.Parameter(_inverse_logit(opacity.detach().float().clone()))

    @property
    def n_splats(self)->int:
        return self.means_xyz.shape[0]

    def world_positions(self, time_s: float) -> torch.Tensor:
        """Positions in scene coordinates at time t; subclasses may learn smooth motion."""
        return self.means_xyz + (float(time_s) - self.reference_time) * self.velocity

    def motion_parameters(self) -> list[nn.Parameter]:
        return [self.velocity]

    def temporal_regularizer(self) -> torch.Tensor:
        return self.means_xyz.new_zeros(())

    def render(self,time_s:float,K:np.ndarray,pose_c2w:np.ndarray,width:int,height:int):
        """Return differentiable RGB [H,W,3], alpha [H,W], optical depth [H,W].

        Unobserved depth has NaN; gradients are only applied where training data
        carries a valid depth observation. Occlusion order is explicitly z-sorted.
        """
        if not (1<=height and 1<=width and width*height<=MAX_RENDER_PIXELS
                and width*height*self.n_splats<=MAX_SPLAT_PIXELS):
            raise ValueError('GAUSSIAN_RENDER_RESOURCE_LIMIT')
        if not math.isfinite(float(time_s)):
            raise ValueError('GAUSSIAN_BAD_TIMESTAMP')
        k=_intrinsics(K)
        pose=validate_pose_c2w(pose_c2w)
        device=self.means_xyz.device
        dtype=self.means_xyz.dtype
        kk=torch.as_tensor(k,dtype=dtype,device=device)
        p=torch.as_tensor(pose,dtype=dtype,device=device)
        w2c=torch.linalg.inv(p)
        xyz=self.world_positions(time_s)
        cam=xyz@w2c[:3,:3].T+w2c[:3,3]
        z=cam[:,2]
        usable=(z>1e-4).to(dtype)
        safe_z=z.clamp(min=1e-4)
        v=kk[1,1]*cam[:,1]/safe_z+kk[1,2]
        u=kk[0,0]*cam[:,0]/safe_z+kk[0,1]*cam[:,1]/safe_z+kk[0,2]
        ys=torch.arange(height,dtype=dtype,device=device)
        xs=torch.arange(width,dtype=dtype,device=device)
        yy,xx=torch.meshgrid(ys,xs,indexing='ij')
        sigma=(self.log_sigma.exp().squeeze(1)*torch.sqrt(kk[0,0]*kk[1,1])/safe_z).clamp(.3,12.)
        r2=(xx.unsqueeze(0)-u[:,None,None]).square()+(yy.unsqueeze(0)-v[:,None,None]).square()
        opacity=torch.sigmoid(self.opacity_logits).flatten()*usable
        alpha=(opacity[:,None,None]*torch.exp(-r2/(2*sigma[:,None,None].square()))).clamp(0,.995)
        order=torch.argsort(safe_z)
        alpha=alpha[order]
        zz=safe_z[order,None,None]
        colors=torch.sigmoid(self.rgb_logits)[order]
        trans=torch.cumprod(torch.cat((torch.ones_like(alpha[:1]),
                     (1-alpha[:-1]).clamp(min=1e-6)),dim=0),dim=0)
        contribution=trans*alpha
        total=contribution.sum(dim=0)
        rgb=(contribution.unsqueeze(-1)*colors[:,None,None,:]).sum(dim=0)
        depth=(contribution*zz).sum(dim=0)/total.clamp(min=1e-5)
        depth=torch.where(total>1e-5,depth,torch.full_like(depth,float('nan')))
        return rgb,total.clamp(0,1),depth


@dataclass(frozen=True)
class GaussianTrainConfig:
    steps:int=100
    lr:float=.03
    depth_weight:float=.10
    silhouette_weight:float=.025
    velocity_regularization:float=.00005
    seed:int=42
    optimize_velocity_only:bool=False
    timeout_s:float=240.
    deformation_regularization:float=.0001

    def validate(self):
        if not (1<=self.steps<=2000 and math.isfinite(self.lr) and 1e-5<=self.lr<=.3
            and all(math.isfinite(x) and x>=0 for x in
                    (self.depth_weight,self.silhouette_weight,self.velocity_regularization,self.deformation_regularization))
            and 1<=self.timeout_s<=7200):
            raise ValueError('GAUSSIAN_INVALID_TRAINING_CONFIG')


def init_gaussians_from_depth(frame:Frame5D,*,num_splats:int=64,seed:int=42)->TrainableGaussian4D:
    _validate_frame(frame)
    if not 1<=num_splats<=MAX_SPLATS:raise ValueError('GAUSSIAN_SPLAT_RESOURCE_LIMIT')
    depth=np.asarray(frame.depth_z,dtype=np.float64)
    valid=np.asarray(frame.valid,bool)&np.isfinite(depth)&(depth>0)
    y,x=np.nonzero(valid)
    if not len(x):raise ValueError('GAUSSIAN_NO_VALID_DEPTH')
    rng=np.random.default_rng(seed)
    idx=rng.choice(len(x),size=min(num_splats,len(x)),replace=False)
    x=x[idx].astype(np.float64);y=y[idx].astype(np.float64);z=depth[y.astype(int),x.astype(int)]
    k=_intrinsics(frame.intrinsics)
    yn=(y-k[1,2])/k[1,1]
    xn=(x-k[0,2]-k[0,1]*yn)/k[0,0]
    cam=np.stack((xn*z,yn*z,z),axis=-1)
    pose=validate_pose_c2w(frame.pose_c2w)
    world=cam@pose[:3,:3].T+pose[:3,3]
    rgb=np.asarray(frame.rgb)[y.astype(int),x.astype(int)].astype(np.float32)/255
    pixel_spacing=math.sqrt(len(np.nonzero(valid)[0])/len(idx))
    widths=(z*pixel_spacing/max(math.sqrt(k[0,0]*k[1,1]),1e-5)*.8).clip(1e-4,1e3)
    n=len(idx)
    return TrainableGaussian4D(torch.from_numpy(world.astype(np.float32)),
              torch.zeros((n,3)),torch.from_numpy(rgb),
              torch.from_numpy(widths.astype(np.float32)[:,None]),
              torch.full((n,1),.75),float(frame.time_s))


def _frame_target(f:Frame5D,device:torch.device):
    _validate_frame(f)
    h,w=f.depth_z.shape
    mask=np.asarray(f.valid,dtype=bool)&np.isfinite(f.depth_z)&(np.asarray(f.depth_z)>0)
    if not np.any(mask):raise ValueError('GAUSSIAN_NO_VALID_DEPTH')
    return (torch.from_numpy(np.asarray(f.rgb).astype(np.float32)/255).to(device),
            torch.from_numpy(np.where(mask,np.asarray(f.depth_z),0).astype(np.float32)).to(device),
            torch.from_numpy(mask).to(device))


def _loss(model:TrainableGaussian4D, frames:list[Frame5D], targets:list, cfg:GaussianTrainConfig)->torch.Tensor:
    losses=[]
    for f,(rgb_target,depth_target,mask) in zip(frames,targets):
        h,w=f.depth_z.shape
        rgb,alpha,depth=model.render(f.time_s,f.intrinsics,f.pose_c2w,w,h)
        rgb_mae=(rgb[mask]-rgb_target[mask]).abs().mean()
        depth_err=(torch.nan_to_num(depth,nan=0.)[mask]-depth_target[mask]).abs()
        # Depth is supervised where splats cover a physical observation.
        overlap=alpha[mask]>.01
        depth_mae=depth_err[overlap].mean() if bool(overlap.any()) else rgb_mae.new_tensor(0.)
        silhouette=(1-alpha[mask]).abs().mean()
        losses.append(rgb_mae+cfg.depth_weight*depth_mae+cfg.silhouette_weight*silhouette)
    return (torch.stack(losses).mean() + cfg.velocity_regularization*model.velocity.square().mean()
            + cfg.deformation_regularization*model.temporal_regularizer())


def train_gaussians(model:TrainableGaussian4D,frames:list[Frame5D],cfg:GaussianTrainConfig)->dict:
    cfg.validate()
    if not 2<=len(frames)<=24:raise ValueError('GAUSSIAN_TRAINING_REQUIRES_2_TO_24_FRAMES')
    times=[float(f.time_s) for f in frames]
    if not np.all(np.diff(times)>0):raise ValueError('GAUSSIAN_TIMESTAMPS_NOT_INCREASING')
    if any((np.asarray(f.depth_z).size*model.n_splats>MAX_SPLAT_PIXELS or
            np.asarray(f.depth_z).size>MAX_RENDER_PIXELS) for f in frames):
        raise ValueError('GAUSSIAN_TRAINING_RESOURCE_LIMIT')
    device=model.means_xyz.device
    targets=[_frame_target(f,device) for f in frames]
    params=model.motion_parameters() if cfg.optimize_velocity_only else list(model.parameters())
    opt=torch.optim.Adam(params,lr=cfg.lr)
    deadline=time.monotonic()+cfg.timeout_s
    with torch.no_grad():
        start=float(_loss(model,frames,targets,cfg).item())
    if not math.isfinite(start):raise ValueError('GAUSSIAN_NONFINITE_INITIAL_LOSS')
    history=[]
    for step in range(cfg.steps):
        if time.monotonic()>deadline:raise TimeoutError(f'GAUSSIAN_TRAIN_TIMEOUT_AFTER_{step}_STEPS')
        opt.zero_grad(set_to_none=True)
        loss=_loss(model,frames,targets,cfg)
        if not bool(torch.isfinite(loss)):raise ValueError('GAUSSIAN_NONFINITE_TRAINING_LOSS')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.)
        opt.step()
        if any(not bool(torch.isfinite(x).all()) for x in params):
            raise ValueError('GAUSSIAN_NONFINITE_TRAINING_PARAMETER')
        if step==0 or (step+1)%max(1,cfg.steps//10)==0:
            history.append([step+1,float(loss.detach().item())])
    with torch.no_grad(): final=float(_loss(model,frames,targets,cfg).item())
    return {'schema_version':'omega-gaussian-training-v1',
            'representation':'learned_spatiotemporal_gaussian_field_not_full_4dgs_paper',
            'loss_initial':start,'loss_final':final,'steps_completed':cfg.steps,
            'frames':len(frames),'splats':model.n_splats,'seed':cfg.seed,
            'loss_curve':history,'relative_depth_is_not_metric':True,
            'color_space':'srgb_observation_approximation'}


def _file_sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as src:
        for chunk in iter(lambda:src.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def save_gaussian_checkpoint(model:TrainableGaussian4D,root:Path,*,units:str,
                             provenance:str,training_report:dict)->dict:
    if units not in ('arbitrary_scale','meters') or provenance not in (
        'relative_estimated','scale_aligned','metric_estimated','metric_measured'):
        raise ValueError('GAUSSIAN_INVALID_PROVENANCE')
    if (units=='meters') != (provenance!='relative_estimated'):
        raise ValueError('GAUSSIAN_UNITS_PROVENANCE_MISMATCH')
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    if root.is_symlink():raise ValueError('GAUSSIAN_UNSAFE_CHECKPOINT_PATH')
    tensors={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    with tempfile.NamedTemporaryFile(dir=root,prefix='.gaussian-',suffix='.safetensors',delete=False) as f:
        temp=Path(f.name)
    try:
        save_file(tensors,str(temp))
        os.replace(temp,root/'gaussians.safetensors')
    finally:
        temp.unlink(missing_ok=True)
    manifest={'schema_version':'omega-gaussian-checkpoint-v1',
        'representation':'trainable_4d_gaussian_scene_v1',
        'reference_time_s':model.reference_time,'splats':model.n_splats,
        'units':units,'provenance':provenance,
        'weights_sha256':_file_sha256(root/'gaussians.safetensors'),
        'training_report':training_report,
        'limitations':['local bounded CPU/GPU optimization, not pretrained',
                       'linear scene motion only','spherical projected gaussians',
                       'no occluded scene geometry or verified metric scale']}
    save_atomic(root/'checkpoint.json',json.dumps(manifest,indent=2,ensure_ascii=False,
                                     sort_keys=True,allow_nan=False).encode('utf8'))
    return manifest


def load_gaussian_checkpoint(root:Path,*,device:str='cpu')->TrainableGaussian4D:
    if device not in ('cpu','cuda'):raise ValueError('GAUSSIAN_INVALID_DEVICE')
    root=Path(root)
    if root.is_symlink():raise ValueError('GAUSSIAN_UNSAFE_CHECKPOINT_PATH')
    meta=root/'checkpoint.json'
    weights=root/'gaussians.safetensors'
    if any(p.is_symlink() for p in (meta,weights)):
        raise ValueError('GAUSSIAN_UNSAFE_CHECKPOINT_PATH')
    if not meta.is_file() or not weights.is_file():raise FileNotFoundError('GAUSSIAN_CHECKPOINT_MISSING')
    if meta.stat().st_size>200000 or weights.stat().st_size>MAX_CHECKPOINT_BYTES:
        raise ValueError('GAUSSIAN_CHECKPOINT_RESOURCE_LIMIT')
    doc=json.loads(meta.read_text(encoding='utf8'))
    if doc.get('schema_version')!='omega-gaussian-checkpoint-v1':
        raise ValueError('GAUSSIAN_CHECKPOINT_SCHEMA_MISMATCH')
    if _file_sha256(weights)!=doc.get('weights_sha256'):
        raise ValueError('GAUSSIAN_CHECKPOINT_SHA256_MISMATCH')
    weights_data=load_file(str(weights),device='cpu')
    required={'means_xyz','velocity','rgb_logits','opacity_logits','log_sigma'}
    if set(weights_data)!=required:raise ValueError('GAUSSIAN_CHECKPOINT_TENSORS_INVALID')
    if not 1<=int(doc['splats'])<=MAX_SPLATS:raise ValueError('GAUSSIAN_SPLAT_RESOURCE_LIMIT')
    means=weights_data['means_xyz'];vel=weights_data['velocity']
    rgb=torch.sigmoid(weights_data['rgb_logits']);opacity=torch.sigmoid(weights_data['opacity_logits'])
    sigma=weights_data['log_sigma'].exp()
    model=TrainableGaussian4D(means,vel,rgb,sigma,opacity,float(doc['reference_time_s']))
    if model.n_splats!=int(doc['splats']):raise ValueError('GAUSSIAN_CHECKPOINT_SPLAT_COUNT_MISMATCH')
    with torch.no_grad():
        model.rgb_logits.copy_(weights_data['rgb_logits'])
        model.opacity_logits.copy_(weights_data['opacity_logits'])
        model.log_sigma.copy_(weights_data['log_sigma'])
    return model.to(device)


def resize_frame_for_training(f:Frame5D,max_side:int)->Frame5D:
    if not 8<=max_side<=128:raise ValueError('GAUSSIAN_RESIZE_RESOURCE_LIMIT')
    _validate_frame(f)
    h,w=f.depth_z.shape
    scale=min(1.,max_side/max(h,w))
    if scale==1.:return f
    hh=max(1,int(round(h*scale)));ww=max(1,int(round(w*scale)))
    from PIL import Image
    rgb=np.asarray(Image.fromarray(f.rgb).resize((ww,hh),Image.Resampling.BILINEAR))
    xx=np.clip(np.rint((np.arange(ww)+.5)*w/ww-.5).astype(int),0,w-1)
    yy=np.clip(np.rint((np.arange(hh)+.5)*h/hh-.5).astype(int),0,h-1)
    depth=np.asarray(f.depth_z)[np.ix_(yy,xx)].copy()
    valid=np.asarray(f.valid)[np.ix_(yy,xx)].copy()
    intr=_intrinsics(f.intrinsics).copy()
    sx=ww/w;sy=hh/h
    intr[0,0]*=sx;intr[0,1]*=sx;intr[1,1]*=sy
    intr[0,2]=sx*(intr[0,2]+.5)-.5
    intr[1,2]=sy*(intr[1,2]+.5)-.5
    sig=None if f.sigma_log_z is None else np.asarray(f.sigma_log_z)[np.ix_(yy,xx)].copy()
    return Frame5D(f.time_s,depth,rgb,intr,f.pose_c2w,valid,sig)


def main(argv:list[str]|None=None):
    parser=argparse.ArgumentParser(description='OMEGA experimental trainable dynamic 4D Gaussians')
    sub=parser.add_subparsers(dest='mode',required=True)
    train=sub.add_parser('train')
    train.add_argument('--manifest',required=True,type=Path)
    train.add_argument('--output',required=True,type=Path)
    train.add_argument('--splats',type=int,default=64)
    train.add_argument('--steps',type=int,default=100)
    train.add_argument('--lr',type=float,default=.03)
    train.add_argument('--seed',type=int,default=42)
    train.add_argument('--max-side',type=int,default=32)
    train.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    train.add_argument('--timeout-s',type=float,default=240.)
    render=sub.add_parser('render')
    render.add_argument('--checkpoint',type=Path,required=True)
    render.add_argument('--manifest',type=Path,required=True)
    render.add_argument('--time',type=float,required=True)
    render.add_argument('--output',type=Path,required=True)
    render.add_argument('--max-side',type=int,default=32)
    render.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args(argv)
    frames,meta=load_frame_manifest(args.manifest)
    frames=[resize_frame_for_training(f,args.max_side) for f in frames]
    if args.mode=='train':
        torch.manual_seed(args.seed)
        model=init_gaussians_from_depth(frames[0],num_splats=args.splats,seed=args.seed).to(args.device)
        report=train_gaussians(model,frames,GaussianTrainConfig(steps=args.steps,lr=args.lr,seed=args.seed,
                                                               timeout_s=args.timeout_s))
        manifest=save_gaussian_checkpoint(model,args.output,
            units=meta.get('units','arbitrary_scale'),provenance=meta.get('provenance','relative_estimated'),
            training_report=report)
        print(json.dumps({'training':report,'checkpoint_sha256':manifest['weights_sha256']},indent=2))
    else:
        if not frames[0].time_s<=args.time<=frames[-1].time_s:
            raise ValueError('GAUSSIAN_TIME_OUTSIDE_TRAINED_INTERVAL')
        model=load_gaussian_checkpoint(args.checkpoint,device=args.device)
        frame=frames[0]
        with torch.no_grad():
            rgb,alpha,depth=model.render(args.time,frame.intrinsics,frame.pose_c2w,
                                         frame.rgb.shape[1],frame.rgb.shape[0])
        from io import BytesIO
        from PIL import Image
        image=Image.fromarray(np.rint(rgb.detach().cpu().numpy()*255).clip(0,255).astype('uint8'))
        out=BytesIO();image.save(out,format='PNG');save_atomic(args.output,out.getvalue())
        np.savez_compressed(args.output.with_suffix('.npz'),
                alpha=alpha.detach().cpu().numpy(),depth=depth.detach().cpu().numpy(),time_s=args.time)
        print(json.dumps({'image':str(args.output),'covered_pixels':int((alpha.detach().cpu()>.01).sum())}))


if __name__=='__main__':main()
