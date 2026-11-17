"""OMEGA v0.6: bounded, learned nonlinear temporal deformation of Gaussian splats.

Each observed Gaussian moves through a differentiable, smooth, time-conditioned
radial basis function field. No extrapolation; no synthetic geometry in occluded
regions. Unlike a full 4DGS reference implementation this is a research baseline
with spherical splats, not anisotropic SH splatting or a general NeRF.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .trainable_4dgs import (
    TrainableGaussian4D, GaussianTrainConfig, train_gaussians,
    resize_frame_for_training, init_gaussians_from_depth, MAX_SPLATS,
    MAX_CHECKPOINT_BYTES,
)
from .reconstruction5d import Frame5D, load_frame_manifest
from .infer import save_atomic

MAX_KNOTS = 12
MAX_MANIFEST_BYTES = 256_000


class DeformableGaussian4D(TrainableGaussian4D):
    """Smooth per-splat time-dependent translation with anchored temporal RBFs.

    x(t)=x0 + v(t-t0) + sum_k [b_k(t)-b_k(t0)] * a_k,
    b=softmax(-(t-k)^2/(2*h^2)). Anchor gives x(t0)=x0; all parameters
    receive gradients through the existing alpha-compositing renderer.
    """
    def __init__(self, means_xyz:torch.Tensor, velocity:torch.Tensor, rgb:torch.Tensor,
                 sigma_world:torch.Tensor, opacity:torch.Tensor, reference_time:float,
                 knot_times:Sequence[float]):
        t=np.asarray(knot_times,dtype=np.float64)
        if (t.ndim!=1 or not 3<=len(t)<=MAX_KNOTS or
                not np.isfinite(t).all() or np.max(np.abs(t))>1e6 or not np.all(np.diff(t)>1e-4) or
                not math.isfinite(reference_time) or not t[0]<=reference_time<=t[-1]):
            raise ValueError('GAUSSIAN_INVALID_KNOTS')
        super().__init__(means_xyz,velocity,rgb,sigma_world,opacity,reference_time)
        if not np.all(np.diff(t.astype('float32'))>0):
            raise ValueError('GAUSSIAN_INVALID_KNOTS')
        self.register_buffer('knot_times',torch.from_numpy(t.astype('float32')))
        self.deformation_offsets=torch.nn.Parameter(
            torch.zeros((len(t),self.n_splats,3),dtype=torch.float32,device=self.means_xyz.device))

    def _rbf(self,t:float)->torch.Tensor:
        knots=self.knot_times.to(device=self.means_xyz.device,dtype=self.means_xyz.dtype)
        spacing=(knots[1:]-knots[:-1]).median().clamp(min=1e-4)
        width=spacing*.8
        return torch.softmax(-.5*((knots-float(t))/width).square(),dim=0)

    def world_positions(self, time_s:float)->torch.Tensor:
        t=float(time_s)
        if not math.isfinite(t):raise ValueError('GAUSSIAN_BAD_TIMESTAMP')
        lower=float(self.knot_times[0]);upper=float(self.knot_times[-1])
        if t<lower-1e-7 or t>upper+1e-7:
            raise ValueError('GAUSSIAN_TIME_OUTSIDE_TRAINED_INTERVAL')
        weights=self._rbf(t)-self._rbf(self.reference_time)
        offsets=torch.einsum('k,knc->nc',weights,self.deformation_offsets)
        return super().world_positions(t)+offsets

    def motion_parameters(self)->list[torch.nn.Parameter]:
        return [self.velocity,self.deformation_offsets]

    def temporal_regularizer(self)->torch.Tensor:
        amplitude=self.deformation_offsets.square().mean()
        curvature=self.deformation_offsets[2:]-2*self.deformation_offsets[1:-1]+self.deformation_offsets[:-2]
        return amplitude+curvature.square().mean()


def upgrade_linear_gaussians(linear:TrainableGaussian4D, knot_times:Sequence[float])->DeformableGaussian4D:
    """Convert a verified v0.5 scene without destroying the existing parameters."""
    with torch.no_grad():
        output=DeformableGaussian4D(linear.means_xyz,linear.velocity,
            torch.sigmoid(linear.rgb_logits),torch.exp(linear.log_sigma),
            torch.sigmoid(linear.opacity_logits),linear.reference_time,knot_times)
        output.rgb_logits.copy_(linear.rgb_logits)
        output.opacity_logits.copy_(linear.opacity_logits)
        output.log_sigma.copy_(linear.log_sigma)
    return output.to(linear.means_xyz.device)


def _sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _secure_root(root:Path)->Path:
    root=Path(root)
    if root.is_symlink() or any((root/name).is_symlink() for name in ('checkpoint.json','gaussians.safetensors')):
        raise ValueError('GAUSSIAN_UNSAFE_CHECKPOINT_PATH')
    return root


def _validate_provenance(units:str,provenance:str)->None:
    if units not in ('arbitrary_scale','meters') or provenance not in (
        'relative_estimated','scale_aligned','metric_estimated','metric_measured') or (
        (units=='meters') != (provenance!='relative_estimated')):
        raise ValueError('GAUSSIAN_UNITS_PROVENANCE_MISMATCH')


def save_deformable_checkpoint(model:DeformableGaussian4D,root:Path,*,units:str,
                               provenance:str,training_report:dict)->dict:
    _validate_provenance(units,provenance)
    root=_secure_root(root)
    root.mkdir(parents=True,exist_ok=True)
    state={name:value.detach().cpu().contiguous() for name,value in model.state_dict().items()}
    if any(not bool(torch.isfinite(t).all()) for t in state.values()):
        raise ValueError('GAUSSIAN_NONFINITE_CHECKPOINT_TENSOR')
    with tempfile.NamedTemporaryFile(dir=root,prefix='.gaussian-',suffix='.safetensors',delete=False) as f:
        temp=Path(f.name)
    try:
        save_file(state,str(temp))
        if temp.stat().st_size>MAX_CHECKPOINT_BYTES:raise ValueError('GAUSSIAN_CHECKPOINT_RESOURCE_LIMIT')
        os.replace(temp,root/'gaussians.safetensors')
    finally:temp.unlink(missing_ok=True)
    meta={'schema_version':'omega-gaussian-deformable-v2',
          'representation':'smooth_temporal_rbf_gaussians_experimental',
          'reference_time_s':model.reference_time,'splats':model.n_splats,
          'knot_times_s':model.knot_times.tolist(), 'units':units,'provenance':provenance,
          'weights_sha256':_sha(root/'gaussians.safetensors'),
          'training_report':training_report,
          'limitations':['interpolation only inside observed time interval',
                         'spherical splats rather than full anisotropic 4DGS',
                         'input geometry and depth provenance are unchanged']}
    payload=json.dumps(meta,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode('utf8')
    if len(payload)>MAX_MANIFEST_BYTES:raise ValueError('GAUSSIAN_CHECKPOINT_RESOURCE_LIMIT')
    save_atomic(root/'checkpoint.json',payload)
    return meta


def load_deformable_checkpoint(root:Path,*,device:str='cpu')->DeformableGaussian4D:
    if device not in ('cpu','cuda'):raise ValueError('GAUSSIAN_INVALID_DEVICE')
    root=_secure_root(root)
    meta=root/'checkpoint.json';path=root/'gaussians.safetensors'
    if not meta.is_file() or not path.is_file():raise FileNotFoundError('GAUSSIAN_CHECKPOINT_MISSING')
    if meta.stat().st_size>MAX_MANIFEST_BYTES or path.stat().st_size>MAX_CHECKPOINT_BYTES:
        raise ValueError('GAUSSIAN_CHECKPOINT_RESOURCE_LIMIT')
    doc=json.loads(meta.read_text(encoding='utf8'))
    if doc.get('schema_version')!='omega-gaussian-deformable-v2':
        raise ValueError('GAUSSIAN_CHECKPOINT_SCHEMA_MISMATCH')
    _validate_provenance(doc.get('units'),doc.get('provenance'))
    if _sha(path)!=doc.get('weights_sha256'):
        raise ValueError('GAUSSIAN_CHECKPOINT_SHA256_MISMATCH')
    weights=load_file(str(path),device='cpu')
    required={'means_xyz','velocity','log_sigma','rgb_logits','opacity_logits','knot_times','deformation_offsets'}
    if set(weights)!=required:raise ValueError('GAUSSIAN_CHECKPOINT_TENSORS_INVALID')
    if any(t.dtype!=torch.float32 for t in weights.values()):
        raise ValueError('GAUSSIAN_CHECKPOINT_TENSOR_DTYPE_INVALID')
    if any(not bool(torch.isfinite(t).all()) for t in weights.values()):
        raise ValueError('GAUSSIAN_NONFINITE_CHECKPOINT_TENSOR')
    n=int(doc['splats'])
    if not 1<=n<=MAX_SPLATS:raise ValueError('GAUSSIAN_SPLAT_RESOURCE_LIMIT')
    time=torch.as_tensor(weights['knot_times'],dtype=torch.float32)
    if time.ndim!=1 or len(time)>MAX_KNOTS:raise ValueError('GAUSSIAN_INVALID_KNOTS')
    if weights['deformation_offsets'].shape!=(len(time),n,3):
        raise ValueError('GAUSSIAN_CHECKPOINT_TENSORS_INVALID')
    if len(doc.get('knot_times_s',[]))!=len(time) or not np.allclose(doc['knot_times_s'],time.numpy(),rtol=0,atol=1e-6):
        raise ValueError('GAUSSIAN_CHECKPOINT_KNOT_MISMATCH')
    ref=float(doc['reference_time_s'])
    model=DeformableGaussian4D(weights['means_xyz'],weights['velocity'],
        torch.sigmoid(weights['rgb_logits']),torch.exp(weights['log_sigma']),
        torch.sigmoid(weights['opacity_logits']),ref,time.tolist())
    model.load_state_dict(weights,strict=True)
    return model.to(device)


def heldout_metrics(model:TrainableGaussian4D,frames:Sequence[Frame5D])->dict:
    """Evaluate *only* supplied holdout frames; all metrics retain observed-mask scope."""
    if not frames:raise ValueError('GAUSSIAN_EMPTY_HOLDOUT')
    total_err=0.;n=0;covered=0;valid=0;depth_err=0.;depth_n=0
    with torch.no_grad():
        for f in frames:
            from .reconstruction5d import _validate_frame
            _validate_frame(f)
            h,w=f.depth_z.shape
            rgb,alpha,depth=model.render(f.time_s,f.intrinsics,f.pose_c2w,w,h)
            mask=np.asarray(f.valid,bool)&np.isfinite(f.depth_z)&(f.depth_z>0)
            if not mask.any():raise ValueError('GAUSSIAN_NO_VALID_DEPTH')
            m=torch.from_numpy(mask).to(device=rgb.device)
            target=torch.from_numpy(f.rgb.astype('float32')/255).to(device=rgb.device)
            total_err+=float((rgb[m]-target[m]).abs().sum().item());n+=int(m.sum())*3
            overlap=m & (alpha>.01)
            valid+=int(m.sum());covered+=int(overlap.sum())
            if bool(overlap.any()):
                tgt_depth=torch.from_numpy(np.where(mask,f.depth_z,0).astype('float32')).to(device=rgb.device)
                depth_err+=float((depth[overlap]-tgt_depth[overlap]).abs().sum().item())
                depth_n+=int(overlap.sum())
    return {'frames':len(frames),'sample_times_s':[float(f.time_s) for f in frames],
            'rgb_mae_valid':total_err/max(1,n),
            'depth_mae_covered':(depth_err/depth_n) if depth_n else None,
            'fraction_predicted_covered':covered/max(valid,1),
            'rgb_evaluation_scope':'observed valid depth pixels only',
            'depth_evaluation_scope':'predicted alpha > .01 and valid observed depth only'}


def main(argv:list[str]|None=None)->None:
    p=argparse.ArgumentParser(description='OMEGA v0.6 bounded, nonlinear temporal Gaussian field')
    commands=p.add_subparsers(dest='mode',required=True)
    t=commands.add_parser('train')
    t.add_argument('--manifest',type=Path,required=True)
    t.add_argument('--output',type=Path,required=True)
    t.add_argument('--knots',type=int,default=5)
    t.add_argument('--splats',type=int,default=48)
    t.add_argument('--steps',type=int,default=150)
    t.add_argument('--lr',type=float,default=.03)
    t.add_argument('--seed',type=int,default=42)
    t.add_argument('--depth-weight',type=float,default=.6)
    t.add_argument('--deformation-regularization',type=float,default=.05)
    t.add_argument('--max-side',type=int,default=32)
    t.add_argument('--timeout-s',type=float,default=240.)
    t.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    t.add_argument('--motion-only',action='store_true')
    t.add_argument('--upgrade-from',type=Path,default=None,
                   help='Previously trained v0.5 checkpoint (retains existing parameters)')
    r=commands.add_parser('render')
    r.add_argument('--checkpoint',type=Path,required=True)
    r.add_argument('--manifest',type=Path,required=True)
    r.add_argument('--time',type=float,required=True)
    r.add_argument('--output',type=Path,required=True)
    r.add_argument('--max-side',type=int,default=32)
    args=p.parse_args(argv)
    frames,meta=load_frame_manifest(args.manifest)
    frames=[resize_frame_for_training(f,args.max_side) for f in frames]
    if args.mode=='train':
        if not 3<=args.knots<=MAX_KNOTS:raise ValueError('GAUSSIAN_INVALID_KNOTS')
        torch.manual_seed(args.seed)
        knots=np.linspace(frames[0].time_s,frames[-1].time_s,args.knots).tolist()
        if args.upgrade_from:
            from .trainable_4dgs import load_gaussian_checkpoint
            model=upgrade_linear_gaussians(load_gaussian_checkpoint(args.upgrade_from),knots)
        else:
            init=init_gaussians_from_depth(frames[0],num_splats=args.splats,seed=args.seed)
            model=upgrade_linear_gaussians(init,knots)
        model=model.to(args.device)
        rep=train_gaussians(model,frames,GaussianTrainConfig(steps=args.steps,lr=args.lr,
            optimize_velocity_only=args.motion_only,seed=args.seed,timeout_s=args.timeout_s,
            depth_weight=args.depth_weight,
            deformation_regularization=args.deformation_regularization))
        saved=save_deformable_checkpoint(model,args.output,units=meta.get('units','arbitrary_scale'),
            provenance=meta.get('provenance','relative_estimated'),training_report=rep)
        print(json.dumps({'training':rep,'sha256':saved['weights_sha256']},indent=2))
    else:
        model=load_deformable_checkpoint(args.checkpoint)
        f=frames[0]
        with torch.no_grad():
            rgb,alpha,depth=model.render(args.time,f.intrinsics,f.pose_c2w,
                f.rgb.shape[1],f.rgb.shape[0])
        from io import BytesIO
        from PIL import Image
        out=BytesIO()
        Image.fromarray(np.rint(rgb.numpy()*255).clip(0,255).astype('uint8')).save(out,format='PNG')
        save_atomic(args.output,out.getvalue())
        data={'alpha':alpha.numpy(),'depth':depth.numpy(),'time_s':args.time}
        buffer=BytesIO();np.savez_compressed(buffer,**data)
        save_atomic(args.output.with_suffix('.npz'),buffer.getvalue())
        print(json.dumps({'output':str(args.output),'alpha_covered':int((alpha>.01).sum())}))


if __name__=='__main__':
    main()
