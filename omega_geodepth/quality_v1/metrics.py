"""Ground-truth-only depth quality metrics; no synthetic benchmark claims."""
from __future__ import annotations
import numpy as np


def depth_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> dict:
    p=np.asarray(predicted,dtype=np.float64)
    g=np.asarray(ground_truth,dtype=np.float64)
    if p.shape!=g.shape:
        raise ValueError('ground-truth and predicted shapes differ')
    truth_valid=np.isfinite(g)&(g>0)
    mask=np.isfinite(p)&(p>0)&truth_valid
    n_truth=int(truth_valid.sum())
    if not mask.any():
        raise ValueError('no valid ground-truth samples')
    p=p[mask];g=g[mask]
    ratio=np.maximum(p/g,g/p)
    log_error=np.log(p)-np.log(g)
    return {'valid_samples':int(mask.sum()),
            'ground_truth_valid_samples':n_truth,
            'prediction_coverage':float(mask.sum()/n_truth),
            'abs_rel':float(np.mean(np.abs(p-g)/g)),
            'rmse':float(np.sqrt(np.mean((p-g)**2))),
            'delta_1':float(np.mean(ratio<1.25)),
            'silog':float(np.sqrt(max(0.,np.mean(log_error**2)-np.mean(log_error)**2)))}
