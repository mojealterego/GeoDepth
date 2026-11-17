"""Optional geometric audit of DA3's *relative-scale* multi-view predictions.

This is a depth reprojection consistency check, not metric accuracy certification.
Missing overlap causes abstention. Occluded samples are not counted as agreement.
"""
from __future__ import annotations

import math
import numpy as np
from .da3_adapter import DA3Output
from .geometry import validate_pose_c2w


def verify_multiview_reprojection(
    pred: DA3Output,
    *,
    max_relative_error: float = .10,
    min_consistency_fraction: float = .8,
    min_coverage: float = .1,
    min_samples: int = 20,
    max_samples_per_pair: int = 12000,
) -> dict:
    if (not math.isfinite(max_relative_error) or max_relative_error <= 0
        or not 0 < min_consistency_fraction <= 1 or not 0 < min_coverage <= 1
        or min_samples < 1 or max_samples_per_pair < min_samples):
        raise ValueError('INVALID_REPROJECTION_THRESHOLDS')
    depth = np.asarray(pred.depth_z,dtype=np.float64)
    poses = np.asarray(pred.pose_c2w,dtype=np.float64)
    cameras = np.asarray(pred.intrinsics_k,dtype=np.float64)
    if (depth.ndim != 3 or depth.shape[0] < 2 or poses.shape!=(depth.shape[0],4,4)
        or cameras.shape!=(depth.shape[0],3,3)):
        raise ValueError('INVALID_REPROJECTION_INPUT')
    for pose in poses:validate_pose_c2w(pose)
    for k in cameras:
        if (not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0
            or not np.allclose(k[2],[0,0,1],atol=1e-6)):
            raise ValueError('INVALID_REPROJECTION_INTRINSICS')
    n,h,w=depth.shape
    records=[]
    for i in range(n-1):
        j=i+1
        valid=np.isfinite(depth[i])&(depth[i]>0)
        ys,xs=np.nonzero(valid)
        source_pixels=len(xs)
        if source_pixels > max_samples_per_pair:
            stride=int(np.ceil(source_pixels/max_samples_per_pair))
            xs=xs[::stride];ys=ys[::stride]
        sampled=len(xs)
        if not sampled:
            records.append({'src':i,'dst':j,'coverage':0.,'observed':0,
                            'median_relative_residual':None,'agreement_fraction':None,
                            'reason':'INSUFFICIENT_VALID_OVERLAP','passed':False})
            continue
        z=depth[i,ys,xs]
        k0=cameras[i];k1=cameras[j]
        ny=(ys-k0[1,2])/k0[1,1]
        nx=(xs-k0[0,2]-k0[0,1]*ny)/k0[0,0]
        xyz_c=np.column_stack([nx*z,ny*z,z])
        r0=poses[i,:3,:3];t0=poses[i,:3,3]
        xyz_world=xyz_c@r0.T+t0
        r1=poses[j,:3,:3];t1=poses[j,:3,3]
        xyz_target=(xyz_world-t1)@r1
        z_pred=xyz_target[:,2]
        with np.errstate(divide='ignore',invalid='ignore'):
            uy=k1[1,1]*xyz_target[:,1]/z_pred+k1[1,2]
            ux=k1[0,0]*xyz_target[:,0]/z_pred+k1[0,1]*xyz_target[:,1]/z_pred+k1[0,2]
        in_view=np.isfinite(ux)&np.isfinite(uy)&np.isfinite(z_pred)&(z_pred>0)
        in_view &= (ux>=0)&(ux<w)&(uy>=0)&(uy<h)
        ix=np.zeros(sampled,dtype=np.int64);iy=np.zeros(sampled,dtype=np.int64)
        ix[in_view]=np.rint(ux[in_view]).astype(np.int64).clip(0,w-1)
        iy[in_view]=np.rint(uy[in_view]).astype(np.int64).clip(0,h-1)
        observed=np.full(sampled,np.nan,dtype=np.float64)
        observed[in_view]=depth[j,iy[in_view],ix[in_view]]
        observed_valid=in_view&np.isfinite(observed)&(observed>0)
        # A nearer surface in the target view hides a more distant projected point.
        # Such points are not counted as support for the camera geometry.
        occluded=observed_valid&(z_pred>observed*(1+max_relative_error))
        visible=observed_valid&(~occluded)
        count=int(visible.sum())
        coverage=count/max(sampled,1)
        if count:
            residual=np.abs(z_pred[visible]-observed[visible])/np.maximum(np.maximum(z_pred[visible],observed[visible]),1e-12)
            median=float(np.median(residual))
            agreement=float(np.mean(residual<=max_relative_error))
        else:median=None;agreement=None
        enough=count>=min_samples and coverage>=min_coverage
        passed=bool(enough and agreement is not None and agreement>=min_consistency_fraction)
        reason=('PASS' if passed else 'INSUFFICIENT_VALID_OVERLAP' if not enough else 'DEPTH_REPROJECTION_MISMATCH')
        records.append({'src':i,'dst':j,'sampled':sampled,'observed':count,
                        'coverage':float(coverage),'median_relative_residual':median,
                        'agreement_fraction':agreement,'occluded':int(occluded.sum()),
                        'reason':reason,'passed':passed})
    return {'passed':bool(records and all(x['passed'] for x in records)),
            'units':pred.units,'interpretation':'relative multi-view consistency, not metric measurement',
            'thresholds':{'max_relative_error':max_relative_error,'min_coverage':min_coverage,
                          'min_consistency_fraction':min_consistency_fraction,'min_samples':min_samples},
            'pairs':records}
