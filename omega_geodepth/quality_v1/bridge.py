"""Static multiview quality gate -> existing 5D scene field; no invented motion."""
from __future__ import annotations
import numpy as np
from .prediction import Prediction
from .geometry import multiview_consistency,camera_baseline_ratio
from .refinement import refine_depth_multiview
from .fusion import confidence_mask


def prepare_static_field(raw, *, assume_static: bool, reference_radius: float,
                         min_view_coverage: float=.1, min_mean_consistency: float=.8,
                         min_baseline_ratio: float=.005, relative_tolerance: float=.1,
                         min_support: int=1, confidence_quantile: float=.1,
                         refine: bool=False, stride: int=2, max_samples: int=100_000):
    """Return observed XYZ/time/scale RGB samples, only for an explicit static group.

    Camera pose + scale stay relative. Confidence is not a probability. Gate
    checks estimated agreement, not physical truth. No timestamp is inferred.
    """
    from ..geometry import validate_pose_c2w
    from ..reconstruction5d import Frame5D
    from ..scene_field import collect_samples
    if assume_static is not True:
        raise ValueError('explicit static scene assumption required')
    if raw.units!='arbitrary_scale' or raw.provenance!='relative_estimated':
        raise ValueError('relative DA3 geometry required')
    if (not 0<min_view_coverage<=1 or not 0<min_mean_consistency<=1
        or not np.isfinite(min_baseline_ratio) or min_baseline_ratio<=0
        or type(min_support) is not int or min_support<1):
        raise ValueError('invalid multiview quality gate')
    poses=np.asarray(raw.pose_c2w)
    if len(poses)<2:raise ValueError('at least two views required')
    p=Prediction(raw.depth_z,raw.confidence_raw,np.stack([np.linalg.inv(validate_pose_c2w(v))[:3] for v in poses]),raw.intrinsics_k)
    if camera_baseline_ratio(p)<min_baseline_ratio:raise ValueError('insufficient camera parallax')
    if refine:
        p,refinement=refine_depth_multiview(p,relative_tolerance=relative_tolerance,min_baseline_ratio=min_baseline_ratio)
    else:refinement={'status':'disabled'}
    q=multiview_consistency(p,relative_tolerance=relative_tolerance)
    tested=q['tested']>0
    counts=p.valid.sum(axis=(1,2))
    coverage=tested.sum(axis=(1,2))/np.maximum(counts,1)
    if np.any(coverage<min_view_coverage):raise ValueError('insufficient view coverage')
    means=[float(np.mean(q['score'][i][tested[i]])) for i in range(len(poses))]
    if min(means)<min_mean_consistency:raise ValueError('insufficient per-view consistency')
    valid=confidence_mask(p,confidence_quantile)&(q['support']>=min_support)&(q['score']>=min_mean_consistency)
    rgb=np.asarray(raw.rgb)
    if rgb.shape!=(*p.depth.shape,3) or rgb.dtype!=np.uint8:raise ValueError('invalid RGB')
    frames=[Frame5D(0.,p.depth[i],rgb[i],raw.intrinsics_k[i],poses[i],valid[i],confidence_raw=p.conf[i]) for i in range(len(poses))]
    samples=collect_samples(frames,units='arbitrary_scale',provenance='relative_estimated',reference_radius=reference_radius,time_basis='static_assumption',stride=stride,max_samples=max_samples)
    return samples,{'view_count':len(poses),'model_id':raw.model_id,'weights_sha256':raw.weights_sha256,
                    'time_basis':'static_assumption','units':'arbitrary_scale','qualified_samples':len(samples.coordinates),
                    'view_coverage':coverage.tolist(),'view_mean_consistency':means,'refinement':refinement,
                    'interpretation':'Estimated agreement only, no independent metric or temporal accuracy evidence'}
