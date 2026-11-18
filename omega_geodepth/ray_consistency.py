"""DA3 depth-ray diagnostics and explicit reference-camera gauge reconciliation.

The pinned upstream implementation stores directions first, centres last,
although the paper uses the reverse notation. Neither task measures truth.
"""
from __future__ import annotations
import numpy as np
from .visual_benchmark import fit_similarity, _poses, _indices, _rotation_angle


def audit_rays(prediction, rays, confidence, *, max_samples_per_view=4096):
    """Compare dense predicted rays against *predicted* pinhole cameras.

    Use the normalized bin-centre convention of upstream ray_utils, not a
    silently unit-normalized depth ray. Ray magnitude error is separate.
    Confidence only selects finite positive support, not probabilities.
    """
    ray=np.asarray(rays);conf=np.asarray(confidence)
    z=np.asarray(prediction.depth_z);p=_poses(prediction.pose_c2w) if len(z)>1 else np.asarray(prediction.pose_c2w,dtype=float)
    from .geometry import validate_pose_c2w
    for pose in p: validate_pose_c2w(pose)
    k=np.asarray(prediction.intrinsics_k,dtype=float)
    if z.ndim!=3 or ray.ndim!=4 or ray.shape[0]!=len(z) or ray.shape[-1]!=6 or min(ray.shape[1:3])<2 or conf.shape!=ray.shape[:-1] or ray.dtype.kind!='f' or conf.dtype.kind not in 'fiu' or k.shape!=(len(z),3,3) or not np.isfinite(k).all() or np.any(k[:,0,0]<=0) or np.any(k[:,1,1]<=0):
        raise ValueError('RAY_AUDIT_INVALID_SHAPE')
    if not isinstance(max_samples_per_view,int) or not 64<=max_samples_per_view<=65536: raise ValueError('RAY_AUDIT_SAMPLE_LIMIT')
    n,h,w=z.shape;rh,rw=ray.shape[1:3]
    if n>8 or rh*rw>1_000_000:raise ValueError('RAY_AUDIT_RESOURCE_LIMIT')
    # Select a deterministic uniform subset across the entire ray grid.
    ids=np.linspace(0,rh*rw-1,min(rh*rw,max_samples_per_view),dtype=np.int64)
    yy,xx=np.divmod(ids,rw);pixel=np.stack([(xx+.5)*w/rw,(yy+.5)*h/rh,np.ones(len(ids))],-1)
    rows=[]
    for i in range(n):
        a=ray[i,yy,xx].astype(float);c=conf[i,yy,xx];d=a[:,:3];o=a[:,3:];dn=np.linalg.norm(d,axis=1)
        valid=np.isfinite(a).all(1)&np.isfinite(c)&(c>0)&(dn>1e-9)
        if valid.sum()<max(4,int(np.ceil(.8*len(ids)))):raise ValueError('RAY_AUDIT_INSUFFICIENT_SUPPORT')
        d=d[valid];o=o[valid];dn=dn[valid]
        expected=(pixel[valid]@np.linalg.inv(k[i]).T)@p[i,:3,:3].T
        en=np.linalg.norm(expected,axis=1);u=d/dn[:,None];v=expected/en[:,None]
        angular=np.degrees(np.arctan2(np.linalg.norm(np.cross(u,v),axis=1),np.sum(u*v,axis=1)))
        depth=z[i][np.isfinite(z[i])&(z[i]>0)]
        if not depth.size:raise ValueError('RAY_AUDIT_INVALID_DEPTH')
        origin=np.linalg.norm(o-p[i,:3,3],axis=1)/np.median(depth)
        # Z-depth requires R^T d to have third component 1, not ||d||=1.
        zerr=np.abs(d@p[i,:3,2]-1.)
        if not np.isfinite(angular).all() or not np.isfinite(origin).all() or not np.isfinite(zerr).all():raise ValueError('RAY_AUDIT_NONFINITE')
        rows.append({'view':i,'samples':len(ids),'valid':int(valid.sum()),'direction_p90_deg':float(np.quantile(angular,.9)),'origin_p90_over_depth':float(np.quantile(origin,.9)),'z_normalization_p90':float(np.quantile(zerr,.9))})
    return {'views':n,'ray_grid':[rh,rw],'per_view':rows,'worst_direction_p90_deg':max(r['direction_p90_deg'] for r in rows),'worst_origin_p90_over_depth':max(r['origin_p90_over_depth'] for r in rows),'worst_z_normalization_p90':max(r['z_normalization_p90'] for r in rows),'scope':'internal consistency; not calibrated confidence or physical accuracy'}


def reconcile_camera_gauge(prediction, reference_poses, *, anchor_indices, validation_indices, center_tolerance, rotation_tolerance_deg):
    """Return a separate reference-scaled estimate, only after independent camera checks.

    Fit exclusively on anchor camera centres; validate all other declared
    cameras, including orientation. Do NOT substitute measured cameras or K,
    change raw predictions, or assert that depth is a measured quantity.
    """
    p=_poses(prediction.pose_c2w);q=_poses(reference_poses)
    if p.shape!=q.shape or prediction.units!='arbitrary_scale' or not np.isfinite(center_tolerance) or center_tolerance<=0 or not np.isfinite(rotation_tolerance_deg) or not 0<rotation_tolerance_deg<=180:raise ValueError('GAUGE_INVALID_CONTRACT')
    a=_indices(anchor_indices,len(p),3);v=_indices(validation_indices,len(p),1)
    if set(a)&set(v) or set(a)|set(v)!=set(range(len(p))):raise ValueError('GAUGE_REQUIRE_DISJOINT_COMPLETE_SPLIT')
    sim=fit_similarity(p[a,:3,3],q[a,:3,3],inlier_threshold=center_tolerance)
    aligned=sim.apply_poses(p);errors=np.linalg.norm(aligned[:,:3,3]-q[:,:3,3],axis=1)
    angular=_rotation_angle(aligned[:,:3,:3]@q[:,:3,:3].transpose(0,2,1))
    if np.any(errors>center_tolerance) or np.any(angular>rotation_tolerance_deg):raise ValueError('GAUGE_CAMERA_VALIDATION_FAILED')
    z=np.asarray(prediction.depth_z,dtype=np.float64)*sim.scale
    if not np.isfinite(z[np.isfinite(prediction.depth_z)]).all():raise ValueError('GAUGE_NONFINITE_DEPTH')
    return {'scale':sim.scale,'rotation':sim.rotation,'translation':sim.translation,'aligned_poses':aligned,'aligned_depth_z':z,'units':'reference_units_estimated','anchor_indices':a.tolist(),'validation_indices':v.tolist(),'max_center_error':float(errors.max()),'max_rotation_error_deg':float(angular.max()),'scope':'reference-camera aligned estimate; not measured depth or independent validation of surface geometry'}
