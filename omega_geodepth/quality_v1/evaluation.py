"""Ground-truth-only evaluation, with explicit optional scale alignment."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from .metrics import depth_metrics
from .pipeline import sha256_file


def _load_npy(path: Path) -> np.ndarray:
    if not path.is_file() or path.is_symlink() or path.stat().st_size>1_000_000_000:
        raise ValueError(f'Input NPY missing, symlinked or exceeds limit: {path}')
    arr=np.load(path,allow_pickle=False,mmap_mode='r')
    if arr.dtype.kind!='f' or arr.ndim not in (2,3) or arr.size>20_000_000:
        raise ValueError('Expected bounded 2D/3D floating point NPY')
    return arr


def evaluate(prediction_path: Path, truth_path: Path, *, alignment: str='none') -> dict:
    if alignment not in ('none','median-scale'):
        raise ValueError('alignment must be none or median-scale')
    p=_load_npy(prediction_path)
    gt=_load_npy(truth_path)
    if p.shape!=gt.shape:
        raise ValueError('Predicted and ground truth shapes differ')
    valid=np.isfinite(p)&np.isfinite(gt)&(p>0)&(gt>0)
    if not valid.any():
        raise ValueError('No valid ground truth samples')
    p=p.astype(np.float64)
    gt=gt.astype(np.float64)
    scale=1.
    if alignment=='median-scale':
        scale=float(np.median(gt[valid]/p[valid]))
        if not np.isfinite(scale) or scale<=0:
            raise ValueError('Invalid fitted scale')
    result=depth_metrics(p*scale,gt)
    return {'alignment':alignment,'scale_factor':scale,'metrics':result,
            'prediction_sha256':sha256_file(prediction_path),
            'ground_truth_sha256':sha256_file(truth_path),
            'caution':'Scores describe supplied ground truth only. Median alignment uses evaluation labels and cannot be reported as calibrated metric inference.'}


def save_evaluation(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.omega-eval-',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(report,f,indent=2,sort_keys=True)
            f.write('\n');f.flush();os.fsync(f.fileno())
        os.link(temp,path)  # atomic, no overwrite
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
