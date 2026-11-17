"""Robust scale-only calibration from independent metric Z-depth anchors."""
from __future__ import annotations
import math
import numpy as np
from .prediction import Prediction


def calibrate_scale(pred: Prediction, anchors: list[dict], *,
                    max_relative_error: float = 0.15) -> dict:
    if pred.scale_status!='relative':
        raise ValueError('calibration requires relative prediction')
    pred.validate()
    if not 0 < max_relative_error < 1:
        raise ValueError('max_relative_error must be in (0,1)')
    if not isinstance(anchors,list) or not 4<=len(anchors)<=100000:
        raise ValueError('at least 4 metric anchors are required')
    n,h,w=pred.depth.shape
    pairs=[]
    cells=[]
    unique=set()
    for a in anchors:
        try:
            vi,x,y=a['view'],a['x'],a['y']
            if any(type(v) is not int for v in (vi,x,y)):
                raise ValueError('anchor indices must be integers')
            depth=float(a['depth_m'])
        except (KeyError,ValueError,TypeError) as exc:
            raise ValueError('invalid anchor format') from exc
        if not (0<=vi<n and 0<=x<w and 0<=y<h):
            raise ValueError('anchor outside predicted image')
        if (vi,x,y) in unique:
            raise ValueError('duplicate anchor positions are not independent')
        unique.add((vi,x,y))
        if not math.isfinite(depth) or depth<=0:
            raise ValueError('anchor depth_m must be positive and finite')
        predicted=float(pred.depth[vi,y,x])
        if not math.isfinite(predicted) or predicted<=0:
            raise ValueError('anchor corresponds to invalid predicted depth')
        pairs.append((predicted,depth))
        cells.append((vi,y*4//h,x*4//w))
    p=np.array([a for a,b in pairs]); m=np.array([b for a,b in pairs])
    ratios=m/p
    scale=float(np.median(ratios))
    relative_errors=np.abs(scale*p-m)/m
    inliers=relative_errors<=max_relative_error
    if int(inliers.sum())<4 or float(inliers.mean())<0.75:
        raise ValueError('inconsistent metric anchors; scale calibration rejected')
    scale=float(np.median(ratios[inliers]))
    relative_errors=np.abs(scale*p-m)/m
    inliers=relative_errors<=max_relative_error
    if int(inliers.sum())<4 or float(inliers.mean())<0.75:
        raise ValueError('inconsistent metric anchors after robust fit')
    occupied={cell for cell,ok in zip(cells,inliers) if ok}
    if len(occupied)<4:
        raise ValueError('anchor spatial coverage insufficient')
    return {'coverage_cells':len(occupied),'scale':scale,'unit':'m','anchor_count':len(p),
            'inliers':int(inliers.sum()),
            'median_relative_error':float(np.median(relative_errors[inliers])),
            'rmse_m':float(np.sqrt(np.mean(((scale*p[inliers])-m[inliers])**2))),
            'max_relative_error_allowed':max_relative_error,
            'warning':'Metric scale is validated only at supplied anchors; no global accuracy guarantee.'}


def apply_calibration(pred: Prediction, report: dict) -> Prediction:
    if pred.scale_status!='relative':
        raise ValueError('calibration requires relative prediction')
    scale=float(report['scale'])
    if not math.isfinite(scale) or scale<=0:
        raise ValueError('invalid calibration scale')
    extrinsics=pred.extrinsics.copy()
    extrinsics[:,:,3]*=scale
    return Prediction(pred.depth*scale,pred.conf.copy(),extrinsics,pred.intrinsics.copy(),
                      scale_status='metric_calibrated')
