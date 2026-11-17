"""Synthetic-domain conformal calibration; never represent it as real-world coverage."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from .train import dataset
from .infer import load_verified_model, save_atomic

WEIGHTS_FILENAME='omega_geodepth_synthetic_v0.1.safetensors' # compatibility fallback

def conformal_quantile(scores:np.ndarray,alpha:float=.1)->float:
    arr=np.asarray(scores,dtype=np.float64).ravel()
    if arr.size<3 or not np.isfinite(arr).all() or (arr<0).any() or not 0<alpha<1:
        raise ValueError('INVALID_CONFORMAL_SCORES')
    # Pixel-level empirical quantile; spatial correlation means no distribution-free
    # per-image coverage guarantee. Honest synthetic-domain calibration only.
    qprob=min(1.,math.ceil((len(arr)+1)*(1-alpha))/len(arr))
    return float(np.quantile(arr,qprob,method='higher'))



def scene_coverage_quantile(scores:np.ndarray,*,alpha_scene:float=.1,
                            min_pixel_coverage:float=.90)->float:
    """Nested image-level conformal quantile for synthetic scene coverage.

    At a given q, at least min_pixel_coverage pixels qualify on a new exchangeable
    scene with probability ~1-alpha_scene, subject to the calibration assumptions.
    """
    s=np.asarray(scores,dtype=np.float64)
    if s.ndim!=3 or s.shape[0]<3 or min(s.shape[1:])<2:
        raise ValueError('SCENE_SCORES_SHAPE')
    if not np.isfinite(s).all() or (s<0).any() or not 0<min_pixel_coverage<=1:
        raise ValueError('INVALID_SCENE_SCORES')
    q_image=np.quantile(s,min_pixel_coverage,axis=(1,2),method='higher')
    return conformal_quantile(q_image,alpha_scene)


def load_verified_calibration(folder:Path,model_manifest:dict)->dict|None:
    path=folder/'calibration.json'
    if not path.is_file():
        return None
    raw=path.read_bytes()
    calibration_hash=model_manifest.get('calibration_sha256')
    if calibration_hash is not None and hashlib.sha256(raw).hexdigest()!=calibration_hash:
        raise ValueError('CALIBRATION_SHA256_MISMATCH')
    calibration=json.loads(raw)
    expected=model_manifest['weights_sha256'][model_manifest.get('weights_file',WEIGHTS_FILENAME)]
    if calibration.get('model_sha256')!=expected:
        raise ValueError('CALIBRATION_MODEL_MISMATCH')
    q=calibration.get('q');alpha=calibration.get('alpha')
    if (not isinstance(q,(int,float)) or not math.isfinite(q) or q<=0
        or not isinstance(alpha,(int,float)) or not 0<alpha<1):
        raise ValueError('INVALID_CALIBRATION_PARAMETERS')
    if 'q_scene_coverage' in calibration:
        q_scene=calibration['q_scene_coverage']
        if not isinstance(q_scene,(int,float)) or not math.isfinite(q_scene) or q_scene<=0:
            raise ValueError('INVALID_SCENE_CALIBRATION_PARAMETERS')
    return calibration

@torch.inference_mode()
def collect_errors(model:torch.nn.Module,count:int,seed:int,batch:int=16,size:int=64) -> tuple[np.ndarray,np.ndarray]:
    xs,ys=dataset(count,seed,size)
    vals=[];rels=[]
    for i in range(0,count,batch):
        logd,logvar=model(xs[i:i+batch])
        target=ys[i:i+batch]
        sigma=torch.exp(0.5*logvar).clamp(min=.01)
        vals.append(((logd-target).abs()/sigma).numpy().ravel())
        rels.append((logd-target).abs().numpy().ravel())
    return np.concatenate(vals),np.concatenate(rels)


def calibrate(model_dir:Path, calibration_count:int=96, audit_count:int=96,alpha:float=.1)->dict:
    torch.set_num_threads(3)
    model,manifest=load_verified_model(model_dir)
    scores,_=collect_errors(model,calibration_count,seed=60_000)
    q=conformal_quantile(scores,alpha)
    scene_scores=scores.reshape(calibration_count,-1,64) if scores.size == calibration_count*64*64 else None
    if scene_scores is None:
        raise ValueError('SCENE_SCORES_SHAPE: calibration assumes 64x64 geometry')
    q_scene=scene_coverage_quantile(scene_scores,alpha_scene=alpha,min_pixel_coverage=.90)
    audit_scores,audit_abs=collect_errors(model,audit_count,seed=70_000)
    coverage=float((audit_scores<=q).mean())
    audit_scene_coverage=float(((audit_scores.reshape(audit_count,-1)<=q_scene).mean(axis=1)>=.90).mean())
    item={
        'status':'synthetic_only_calibrated','alpha':alpha,'q':q,
        'q_scene_coverage':q_scene,'min_scene_pixel_coverage':.90,
        'empirical_scene_level_coverage_synthetic_audit':audit_scene_coverage,
        'nominal_coverage':1-alpha,
        'empirical_coverage_separate_synthetic_audit':coverage,
        'audit_mae_log_depth':float(audit_abs.mean()),
        'calibration_samples':calibration_count,'audit_samples':audit_count,
        'calibration_seed_start':60_000,'audit_seed_start':70_000,
        'score':'absolute_log_depth_error / predicted_sigma_log',
        'coverage_scope':'pixel-marginal on procedurally generated scenes only, not real-world images',
        'model_sha256':manifest['weights_sha256'][manifest.get('weights_file',WEIGHTS_FILENAME)],
        'content_filter':'none',
    }
    content=json.dumps(item,ensure_ascii=False,indent=2).encode()
    save_atomic(model_dir/'calibration.json',content)
    # Integrity binding, NOT a cryptographic signature.
    manifest['calibration_sha256']=hashlib.sha256(content).hexdigest()
    save_atomic(model_dir/'manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2).encode())
    return item


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',type=Path,default=Path('models'))
    p.add_argument('--calibration-count',type=int,default=96)
    p.add_argument('--audit-count',type=int,default=96)
    p.add_argument('--alpha',type=float,default=.1)
    a=p.parse_args()
    if a.calibration_count<4 or a.audit_count<4: p.error('Need >=4 distinct scenes in each partition')
    print(json.dumps(calibrate(a.model_dir,a.calibration_count,a.audit_count,a.alpha),indent=2))

if __name__=='__main__':main()
