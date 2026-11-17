"""Metric-only bridge to Representations5D. No relative-to-metric promotion."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np
from .reconstruction5d import Frame5D, _validate_frame

@dataclass(frozen=True)
class MetricScaleSamples:
    coordinates: np.ndarray
    values: np.ndarray
    confidence: np.ndarray
    reference_radius_m: float
    provenance: str
    note: str = 'Caller-declared metric geometry; uniform observation weights, not calibrated confidence.'

def lift_metric_sequence(frames: Sequence[Frame5D], *, units: str, provenance: str,
                         reference_radius_m: float, stride: int=1,
                         max_samples: int=100_000) -> MetricScaleSamples:
    """Preserve caller-provided metric camera poses and optical Z; RGB is a feature.

    DA3 Small output must be externally scale-aligned before calling. This
    function does not validate the truth of the caller's calibration evidence.
    Nonzero skew must be rectified upstream: Representations5D has no skew field.
    """
    from representations4d.geometry import CameraIntrinsics, DepthFrame, DepthKind, RigidTransform
    from representations4d.fived.lift import lift_metric_rgbd
    if units != 'meters' or provenance not in ('scale_aligned','metric_measured','metric_estimated'):
        raise ValueError('EXPLICIT_METRIC_GEOMETRY_REQUIRED')
    if isinstance(reference_radius_m,(bool,np.bool_)) or not math.isfinite(float(reference_radius_m)) or reference_radius_m<=0:
        raise ValueError('INVALID_REFERENCE_RADIUS')
    if type(stride) is not int or not 1<=stride<=1_000_000 or type(max_samples) is not int or not 1<=max_samples<=100_000:
        raise ValueError('INVALID_BUDGET')
    if not 1<=len(frames)<=1024: raise ValueError('INVALID_FRAME_COUNT')
    batches=[]; prev=-math.inf; seen=0
    for f in frames:
        _validate_frame(f)
        t=float(f.time_s)
        if t<=prev: raise ValueError('TIMESTAMPS_MUST_BE_STRICTLY_INCREASING')
        prev=t
        k=np.asarray(f.intrinsics,dtype=float)
        if not np.allclose([k[0,1],k[1,0],k[2,0],k[2,1]],0,rtol=0,atol=1e-10):
            raise ValueError('RECTIFIED_ZERO_SKEW_INTRINSICS_REQUIRED')
        d=np.asarray(f.depth_z); mask=np.asarray(f.valid)
        if mask.dtype!=np.bool_: raise ValueError('BOOLEAN_VALIDITY_REQUIRED')
        if d.dtype.kind not in 'fiu': raise ValueError('REAL_DEPTH_REQUIRED')
        good=mask[::stride,::stride]&np.isfinite(d[::stride,::stride])&(d[::stride,::stride]>0)
        n=int(good.sum()); seen+=n
        if n==0: raise ValueError('NO_VALID_DEPTH_IN_FRAME')
        if seen>max_samples: raise MemoryError('SAMPLE_BUDGET_EXCEEDED')
        h,w=d.shape; pose=np.asarray(f.pose_c2w,dtype=float)
        batch=lift_metric_rgbd(DepthFrame(d,DepthKind.METRIC,mask),
            CameraIntrinsics(w,h,k[0,0],k[1,1],k[0,2],k[1,2]),
            RigidTransform(pose[:3,:3],pose[:3,3]),t,float(reference_radius_m),
            features=np.asarray(f.rgb,dtype=np.float32)/255.,stride=stride,max_points=n)
        batches.append(batch)
    return MetricScaleSamples(*(np.concatenate([getattr(b,key) for b in batches]) for key in ('coordinates','values','confidence')),float(reference_radius_m),provenance)

def fit_metric_scale_field(samples: MetricScaleSamples, grid, *, reference_radius_m: float,
                           max_active_cells: int=250_000):
    from representations4d.fived.field import GridSpec5D, SparseField5D
    if not isinstance(samples,MetricScaleSamples) or not isinstance(grid,GridSpec5D):
        raise ValueError('INVALID_SCALE_FIELD_CONTRACT')
    if not (samples.reference_radius_m==grid.reference_radius_m==reference_radius_m):
        raise ValueError('REFERENCE_RADIUS_MISMATCH')
    return SparseField5D.fit(grid,samples.coordinates,samples.values,samples.confidence,max_active_cells=max_active_cells)
