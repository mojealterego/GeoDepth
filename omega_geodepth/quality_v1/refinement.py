"""Conservative, optional geometric consensus refinement (not learned retraining).

Uses only reprojection-compatible pixels and never claims better metric accuracy.
No-parallax scenes are skipped because identical camera centers provide no
independent multiview constraint.
"""
from __future__ import annotations
import numpy as np
from .prediction import Prediction
from .geometry import (backproject_pixels,project_world,camera_baseline_ratio,_targets)


def refine_depth_multiview(pred: Prediction, *, relative_tolerance: float = 0.05,
                           max_targets_per_view: int = 8,
                           min_baseline_ratio: float = 0.005,
                           blend: float = 0.5) -> tuple[Prediction,dict]:
    if not np.isfinite(blend) or not 0<=blend<=1:
        raise ValueError('blend must be in [0,1]')
    if not 0<relative_tolerance<1:
        raise ValueError('relative_tolerance must be in (0,1)')
    if max_targets_per_view<1 or max_targets_per_view>64:
        raise ValueError('max_targets_per_view must be between 1 and 64')
    if not np.isfinite(min_baseline_ratio) or min_baseline_ratio<0:
        raise ValueError('min_baseline_ratio must be finite and non-negative')
    pred.validate()
    n,h,w=pred.depth.shape
    valid_pixels=pred.valid
    if n<2 or camera_baseline_ratio(pred)<min_baseline_ratio:
        return pred,{'status':'skipped_no_parallax','changed_pixels':0,'mean_relative_change':0.}
    refined=pred.depth.copy()
    for source in range(n):
        sy,sx=np.nonzero(valid_pixels[source])
        if not len(sy):
            continue
        src_depth=pred.depth[source,sy,sx].astype(np.float64)
        sum_depth=src_depth.copy()
        count=np.ones(len(src_depth),dtype=np.int32)
        source_world=backproject_pixels(pred,source,sy,sx)
        for target in _targets(source,n,max_targets_per_view):
            u,v,z=project_world(source_world,pred.extrinsics[target],pred.intrinsics[target])
            valid=np.isfinite(u)&np.isfinite(v)&np.isfinite(z)&(z>0)&(u>=0)&(u<w)&(v>=0)&(v<h)
            if not valid.any():
                continue
            idx=np.flatnonzero(valid)
            tx=np.floor(u[idx]+.5).astype(np.int64)
            ty=np.floor(v[idx]+.5).astype(np.int64)
            within=(tx>=0)&(tx<w)&(ty>=0)&(ty<h)
            idx=idx[within];tx=tx[within];ty=ty[within]
            if not len(idx):
                continue
            target_depth=pred.depth[target,ty,tx]
            good=valid_pixels[target,ty,tx]
            error=np.abs(z[idx]-target_depth)
            good &= error<=relative_tolerance*np.maximum(z[idx],target_depth)
            idx=idx[good];tx=tx[good];ty=ty[good]
            if not len(idx):
                continue
            target_world=backproject_pixels(pred,target,ty,tx)
            _,_,backprojected_z=project_world(target_world,pred.extrinsics[source],pred.intrinsics[source])
            close=(np.isfinite(backprojected_z)&(backprojected_z>0)&
                   (np.abs(backprojected_z-src_depth[idx])<=
                    relative_tolerance*np.maximum(backprojected_z,src_depth[idx])))
            idx=idx[close]
            if not len(idx):
                continue
            sum_depth[idx]+=backprojected_z[close]
            count[idx]+=1
        supported=count>1
        if supported.any():
            consensus=sum_depth[supported]/count[supported]
            refined[source,sy[supported],sx[supported]]=(
                (1-blend)*src_depth[supported]+blend*consensus).astype(np.float32)
    changed=(np.isfinite(pred.depth)&np.isfinite(refined)&
             (np.abs(pred.depth-refined)>1e-7*np.maximum(1,np.abs(pred.depth))))
    relative=np.abs(refined[changed]-pred.depth[changed])/pred.depth[changed]
    stats={'status':'applied','changed_pixels':int(changed.sum()),
           'mean_relative_change':float(relative.mean()) if len(relative) else 0.,
           'blend':blend,'relative_tolerance':relative_tolerance,
           'note':'Geometric consensus is not independently validated accuracy improvement'}
    return Prediction(refined,pred.conf.copy(),pred.extrinsics.copy(),pred.intrinsics.copy(),
                      scale_status=pred.scale_status),stats
