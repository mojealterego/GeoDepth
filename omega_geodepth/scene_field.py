"""Scene-conditioned RGB fitting in XYZ, real time (or static), and log footprint.

KPlanes is used as a mathematical basis on a dimensionless unit cube. Its
native metric serialization is NOT used for relative scenes. No radiance,
occupancy, motion correspondence, or unseen geometry is inferred here.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
from .reconstruction5d import Frame5D, _validate_frame

SCHEMA = 'omega.geodepth.scene_field/1'

@dataclass(frozen=True)
class SceneSamples:
    coordinates: np.ndarray
    rgb: np.ndarray
    view_index: np.ndarray
    units: str
    provenance: str
    reference_radius: float
    time_basis: str


def _contract(units, provenance, reference_radius, time_basis):
    allowed = {'meters': {'scale_aligned', 'metric_estimated', 'metric_measured'},
               'arbitrary_scale': {'relative_estimated'}}
    if units not in allowed or provenance not in allowed[units]:
        raise ValueError('INCONSISTENT_GEOMETRY_PROVENANCE')
    if not isinstance(reference_radius, (int, float)) or isinstance(reference_radius, bool) or not math.isfinite(reference_radius) or reference_radius <= 0:
        raise ValueError('INVALID_RADIUS')
    if time_basis not in ('capture_seconds', 'static_assumption'):
        raise ValueError('INVALID_TIME_BASIS')


def collect_samples(frames, *, units, provenance, reference_radius, time_basis,
                    stride=1, max_samples=100_000):
    """Sample observed pixels; permit simultaneous views without invented times."""
    _contract(units, provenance, reference_radius, time_basis)
    if not 2 <= len(frames) <= 128 or type(stride) is not int or not 1 <= stride <= 1000000 or type(max_samples) is not int or not 1 <= max_samples <= 100000:
        raise ValueError('INVALID_SAMPLE_BUDGET')
    coords, rgbs, views = [], [], []
    total = 0
    prev = -math.inf
    for i, f in enumerate(frames):
        _validate_frame(f)
        t = float(f.time_s)
        if t < prev or (time_basis == 'static_assumption' and t != 0):
            raise ValueError('INVALID_CAPTURE_TIME')
        prev = t
        d = np.asarray(f.depth_z)
        mask = np.asarray(f.valid)
        if mask.dtype != bool or d.dtype.kind not in 'fiu':
            raise ValueError('INVALID_DEPTH_MASK')
        ds = d[::stride, ::stride]
        good = mask[::stride, ::stride] & np.isfinite(ds) & (ds > 0)
        yy, xx = np.nonzero(good)
        n = len(xx)
        total += n
        if n == 0:
            raise ValueError('NO_VALID_DEPTH_IN_VIEW')
        if total > max_samples:
            raise MemoryError('SAMPLE_BUDGET_EXCEEDED')
        k = np.asarray(f.intrinsics, dtype=np.float64)
        if abs(k[1,0]) > 1e-10:
            raise ValueError('UNSUPPORTED_INTRINSICS')
        uv = np.column_stack((xx * stride, yy * stride, np.ones(n)))
        z = ds[good].astype(np.float64)
        xyz = np.linalg.solve(k, uv.T).T * z[:,None]
        p = np.asarray(f.pose_c2w, dtype=np.float64)
        world = xyz @ p[:3,:3].T + p[:3,3]
        # footprint is a nominal projected pixel scale, not an uncertainty radius.
        scale = np.log2(z) - .5*(np.log2(k[0,0])+np.log2(k[1,1])) - np.log2(reference_radius)
        points = np.column_stack((world, np.full(n,t), scale))
        if not np.isfinite(points).all():
            raise ValueError('NONFINITE_SCENE_COORDINATES')
        coords.append(points); rgbs.append(f.rgb[::stride,::stride][good]/255.); views.append(np.full(n,i,dtype=np.int32))
    return SceneSamples(np.concatenate(coords), np.concatenate(rgbs), np.concatenate(views), units, provenance, reference_radius, time_basis)


def align_baseline(frames, baselines, *, evidence_id, max_relative_residual=.1):
    """Estimate a global similarity scale from caller-supplied camera distances in meters.

    The caller must substantiate baseline measurements and same-scene gauge.
    A residual gate checks consistency, not authenticity or depth accuracy.
    """
    if not isinstance(evidence_id,str) or not evidence_id.strip() or len(evidence_id)>512 or not 2<=len(frames)<=128 or not 2<=len(baselines)<=512 or not math.isfinite(max_relative_residual) or not 0<max_relative_residual<=.2:
        raise ValueError('INVALID_BASELINE_EVIDENCE')
    for f in frames: _validate_frame(f)
    predicted=[]; measured=[]; seen=set()
    for i,j,b in baselines:
        if type(i) is not int or type(j) is not int or not 0<=i<len(frames) or not 0<=j<len(frames) or i==j or tuple(sorted((i,j))) in seen or not isinstance(b,(float,int)) or isinstance(b,bool) or not math.isfinite(b) or b<=0:
            raise ValueError('INVALID_BASELINE')
        seen.add(tuple(sorted((i,j))))
        a=float(np.linalg.norm(np.asarray(frames[i].pose_c2w)[:3,3]-np.asarray(frames[j].pose_c2w)[:3,3]))
        if not math.isfinite(a) or a<=1e-10: raise ValueError('DEGENERATE_BASELINE')
        predicted.append(a); measured.append(b)
    a=np.asarray(predicted); b=np.asarray(measured)
    scale=float(np.median(b/a)); residual=np.abs(a*scale/b-1.)
    if not math.isfinite(scale) or scale<=0 or not np.isfinite(residual).all() or np.max(residual)>max_relative_residual:
        raise ValueError('INCONSISTENT_BASELINE_SCALE')
    out=[]
    for f in frames:
        pose=np.array(f.pose_c2w,dtype=np.float64,copy=True);pose[:3,3]*=scale
        d=np.asarray(f.depth_z,dtype=np.float64)*scale
        if not np.isfinite(d[np.asarray(f.valid,dtype=bool)]).all() or not np.isfinite(pose).all():raise ValueError('SCALE_OVERFLOW')
        out.append(replace(f,depth_z=d,pose_c2w=pose))
    return out, {'units':'meters','provenance':'scale_aligned','evidence_id':evidence_id,'scale':scale,'baseline_count':len(a),'max_relative_residual':float(np.max(residual)),'note':'Caller-declared baselines; not independent proof of metric accuracy.'}


class SceneField:
    def __init__(self, network, lower, upper):
        self.network=network
        self.lower=np.asarray(lower,dtype=np.float64);self.upper=np.asarray(upper,dtype=np.float64)
        if self.lower.shape!=(5,) or self.upper.shape!=(5,) or not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all() or not np.all(self.upper>self.lower) or not np.isfinite(self.upper-self.lower).all():
            raise ValueError('INVALID_NORMALIZATION')

    def normalized(self, coords):
        a=np.asarray(coords)
        if a.ndim!=2 or a.shape[1]!=5 or a.dtype.kind not in 'fiu' or not np.isfinite(a).all() or len(a)>100000:
            raise ValueError('INVALID_QUERY')
        x=(a-self.lower)/(self.upper-self.lower)
        if not np.isfinite(x).all() or np.any(x < -1e-8) or np.any(x>1+1e-8):raise ValueError('OUTSIDE_SCENE_DOMAIN')
        return np.clip(x,0,1).astype(np.float32)

    def predict(self, coords):
        """Predicted RGB at supplied coordinates, not proof of observation/occupancy."""
        import torch
        x=self.normalized(coords); self.network.eval()
        with torch.no_grad():
            rows=[self.network(torch.from_numpy(chunk)).clamp(0,1).cpu().numpy() for chunk in np.array_split(x,max(1,math.ceil(len(x)/2048)))]
        return np.concatenate(rows)


def _new_network(seed=0):
    import torch
    from representations4d.fived.field import Axis5D, GridSpec5D
    from representations4d.fived.neural import KPlanes5D
    # Internal unit cube; the upstream class's metric naming is not exported.
    grid=GridSpec5D(tuple(Axis5D(0,1/(n-1),n) for n in (24,24,24,8,12)),1.)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return KPlanes5D(grid,feature_dim=8,hidden_dim=48,output_dim=3)


def fit_scene_field(samples, *, holdout_views, steps=200, seed=0, split='view'):
    """Train only on non-held-out view colors; domain bounds use all input XYZts.

    Inference target coordinates are supplied (conditional color regression).
    For split='time', all views sharing held-out times must be withheld.
    """
    import torch
    if not isinstance(samples,SceneSamples) or type(steps) is not int or not 1<=steps<=10000 or type(seed) is not int or not 0<=seed<2**32 or split not in ('view','time'):
        raise ValueError('INVALID_TRAINING_CONFIGURATION')
    _contract(samples.units,samples.provenance,samples.reference_radius,samples.time_basis)
    c=np.asarray(samples.coordinates);y=np.asarray(samples.rgb);v=np.asarray(samples.view_index)
    if c.ndim!=2 or c.shape[1]!=5 or not 8<=len(c)<=100000 or y.shape!=(len(c),3) or v.shape!=(len(c),) or v.dtype.kind not in 'iu' or not np.isfinite(c).all() or not np.isfinite(y).all() or np.any(y<0) or np.any(y>1):raise ValueError('INVALID_TRAINING_SAMPLES')
    if not holdout_views or any(type(i) is not int for i in holdout_views) or len(set(holdout_views))!=len(holdout_views) or not set(holdout_views)<=set(v.tolist()):raise ValueError('INVALID_HOLDOUT_VIEWS')
    mask=np.isin(v,holdout_views);train=np.flatnonzero(~mask);test=np.flatnonzero(mask)
    if len(train)<8 or len(test)<1:raise ValueError('INSUFFICIENT_TRAIN_TEST_DATA')
    if split=='time' and (samples.time_basis!='capture_seconds' or np.intersect1d(c[train,3],c[test,3]).size):raise ValueError('TIME_GROUP_HOLDOUT_REQUIRED')
    lower=c.min(0);upper=c.max(0);padding=np.maximum((upper-lower)*.001,1e-5)
    model=SceneField(_new_network(seed),lower-padding,upper+padding)
    x=torch.from_numpy(model.normalized(c));target=torch.from_numpy(y.astype(np.float32));tr=torch.from_numpy(train)
    g=torch.Generator(device='cpu').manual_seed(seed);opt=torch.optim.Adam(model.network.parameters(),lr=.01)
    def mse(indices):
        out=model.predict(c[indices]);return float(np.mean((out-y[indices])**2))
    report={'schema':SCHEMA,'split':split,'holdout_views':sorted(holdout_views),'train_samples':len(train),'test_samples':len(test),'steps':steps,'seed':seed,'train_mse_initial':mse(train),'test_mse_initial':mse(test),'evaluation_scope':'held-out RGB, supplied geometry; domain bounds include holdout coordinates'}
    for _ in range(steps):
        model.network.train(); idx=tr[torch.randint(len(tr),(min(256,len(tr)),),generator=g)]
        opt.zero_grad(set_to_none=True);loss=(model.network(x[idx])-target[idx]).square().mean()
        if not torch.isfinite(loss):raise ValueError('NONFINITE_TRAINING_LOSS')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.network.parameters(),1.,error_if_nonfinite=True);opt.step()
    report.update(train_mse_final=mse(train),test_mse_final=mse(test),parameters=sum(p.numel() for p in model.network.parameters())); model.network.eval()
    return model,report


def save_scene_field(path,model,samples,report):
    from safetensors.torch import save_file
    path=Path(path)
    if path.suffix!='.safetensors' or not path.parent.is_dir() or not isinstance(model,SceneField):raise ValueError('INVALID_CHECKPOINT_PATH')
    _contract(samples.units,samples.provenance,samples.reference_radius,samples.time_basis)
    md={'schema':SCHEMA,'units':samples.units,'provenance':samples.provenance,'reference_radius':samples.reference_radius,'time_basis':samples.time_basis,'lower':model.lower.tolist(),'upper':model.upper.tolist(),'report':report}
    payload=json.dumps(md,allow_nan=False,sort_keys=True)
    if len(payload)>64000:raise ValueError('CHECKPOINT_METADATA_LIMIT')
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as handle: temp=Path(handle.name)
    try:
        save_file({k:v.detach().cpu().contiguous() for k,v in model.network.state_dict().items()},str(temp),metadata={'scene_field':payload})
        with temp.open('rb') as handle:os.fsync(handle.fileno())
        os.link(temp,path)
    finally:temp.unlink(missing_ok=True)


def load_scene_field(path):
    from safetensors import safe_open
    import torch
    path=Path(path)
    if path.suffix!='.safetensors' or not path.is_file() or path.is_symlink() or path.stat().st_size>2_000_000:raise ValueError('INVALID_CHECKPOINT')
    net=_new_network();expected=net.state_dict()
    with safe_open(str(path),framework='pt',device='cpu') as f:
        raw=(f.metadata() or {}).get('scene_field','')
        if not raw or len(raw)>64000:raise ValueError('INVALID_CHECKPOINT_METADATA')
        md=json.loads(raw)
        if not isinstance(md,dict) or md.get('schema')!=SCHEMA or set(md)!={'schema','units','provenance','reference_radius','time_basis','lower','upper','report'}:raise ValueError('INVALID_CHECKPOINT_SCHEMA')
        _contract(md['units'],md['provenance'],md['reference_radius'],md['time_basis'])
        if set(f.keys())!=set(expected):raise ValueError('INVALID_CHECKPOINT_KEYS')
        state={}
        for key,val in expected.items():
            t=f.get_tensor(key)
            if t.shape!=val.shape or t.dtype!=val.dtype or not torch.isfinite(t).all() or (key in ('_origins','_spans') and not torch.equal(t,val)):raise ValueError('INVALID_CHECKPOINT_TENSOR')
            state[key]=t
    net.load_state_dict(state,strict=True);net.eval()
    return SceneField(net,md['lower'],md['upper']),md
