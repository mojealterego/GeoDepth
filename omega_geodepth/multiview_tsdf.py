"""Conservative multiview projective TSDF fusion.

Input frames share an explicitly documented coordinate/scale convention. TSDF
cells have a count of independent camera observations. Unknown cells stay
unobserved; unobserved space is NEVER a measured free space surface.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .reconstruction5d import Frame5D, _validate_frame, _points_from_frame
from .tsdf5d import TSDFGrid
from .geometry import validate_pose_c2w


def integrate_multiview_tsdf(
    frames: list[Frame5D], *, voxel_size: float = .05,
    truncation: float = .15, max_voxels: int = 262144,
    reject_disagreement: float = 1.0,
) -> TSDFGrid:
    """Fuse calibrated depth from multiple *static* views in world coordinates.

    static_mask=True means a pixel is eligible for the static reconstruction.
    Missing mask means static is assumed, not proven. Returned weights count
    independent frame observations, not sensor precision or probabilities.
    """
    if len(frames) < 2 or len(frames)>128:
        raise ValueError('TSDF_REQUIRES_TWO_TO_128_FRAMES')
    if not (np.isfinite(voxel_size) and voxel_size>0 and
            np.isfinite(truncation) and truncation>voxel_size and
            2<=max_voxels<=8_000_000 and 0<reject_disagreement<=2):
        raise ValueError('INVALID_TSDF_RESOURCE_OR_GEOMETRY_PARAMETERS')
    for f in frames:
        _validate_frame(f)
    times=np.array([f.time_s for f in frames])
    if not np.all(np.diff(times)>0):
        raise ValueError('TIMESTAMPS_MUST_BE_STRICTLY_INCREASING')
    # Bounds derive from observed *static* samples only.
    all_points=[]
    for f in frames:
        effective=f
        if f.static_mask is not None:
            from dataclasses import replace
            effective=replace(f, valid=np.asarray(f.valid,dtype=bool)&np.asarray(f.static_mask,dtype=bool))
        try:
            points,*_=_points_from_frame(effective,max_points=120_000)
            all_points.append(points)
        except ValueError as e:
            if 'NO_VALID_DEPTH' not in str(e):raise
    if not all_points: raise ValueError('NO_STATIC_GEOMETRY_OBSERVED')
    lower=np.min([p.min(axis=0) for p in all_points],axis=0)-1.05*truncation
    upper=np.max([p.max(axis=0) for p in all_points],axis=0)+1.05*truncation
    dims=np.ceil((upper-lower)/voxel_size).astype(np.int64)+1
    count=int(np.prod(dims))
    if np.any(dims<2) or count>max_voxels:
        raise ValueError(f'TSDF_RESOURCE_LIMIT: {count} voxels requested, max {max_voxels}')
    sum_sdf=np.zeros(count,dtype=np.float64)
    sum_sq=np.zeros(count,dtype=np.float64)
    weights=np.zeros(count,dtype=np.uint16)
    chunksize=24_000
    # Raster-to-voxel projection in batches, respecting affine pixel intrinsics.
    for f in frames:
        pose=validate_pose_c2w(f.pose_c2w)
        w2c=np.linalg.inv(pose)
        k=np.asarray(f.intrinsics,dtype=np.float64)
        dep=np.asarray(f.depth_z,dtype=np.float64)
        mask=np.asarray(f.valid,dtype=bool).copy()
        if f.static_mask is not None:mask &= np.asarray(f.static_mask,dtype=bool)
        h,w=dep.shape
        for start in range(0,count,chunksize):
            inds=np.arange(start,min(start+chunksize,count))
            coords=np.column_stack(np.unravel_index(inds,tuple(dims)))
            pos=lower+coords*voxel_size
            cam=pos@w2c[:3,:3].T+w2c[:3,3]
            z=cam[:,2]
            safe=(z>1e-7)&np.isfinite(cam).all(axis=1)
            if not safe.any():continue
            ids=np.flatnonzero(safe)
            yn=cam[ids,1]/z[ids]
            xn=cam[ids,0]/z[ids]
            u=np.rint(k[0,0]*xn+k[0,1]*yn+k[0,2]).astype(np.int64)
            v=np.rint(k[1,1]*yn+k[1,2]).astype(np.int64)
            inside=(u>=0)&(u<w)&(v>=0)&(v<h)
            ids,u,v=ids[inside],u[inside],v[inside]
            if not len(ids):continue
            ok=mask[v,u]&np.isfinite(dep[v,u])&(dep[v,u]>0)
            ids,u,v=ids[ok],u[ok],v[ok]
            if not len(ids):continue
            sdf=dep[v,u]-z[ids]
            # Behind the observed surface beyond truncation is unknown, not free space.
            good=sdf>=-truncation
            ids,sdf=ids[good],sdf[good]
            if not len(ids):continue
            vals=np.clip(sdf/truncation,-1.0,1.0)
            index=inds[ids]
            sum_sdf[index]+=vals
            sum_sq[index]+=vals**2
            weights[index]+=1
    observed=weights>0
    result=np.ones(count,dtype=np.float32)
    result[observed]=(sum_sdf[observed]/weights[observed]).astype(np.float32)
    # Two contradictory surfaces must not produce a fabricated averaged zero crossing.
    n=weights>=2
    var=np.zeros(count,dtype=np.float64)
    var[n]=np.maximum(0,sum_sq[n]/weights[n]-result[n].astype(float)**2)
    conflicts=n & (np.sqrt(var)>reject_disagreement/2)
    weights[conflicts]=0
    result[conflicts]=1.
    return TSDFGrid(result.reshape(tuple(dims)),weights.reshape(tuple(dims)),
                    lower.astype(np.float64),float(voxel_size),float(truncation),float(times[0]))
