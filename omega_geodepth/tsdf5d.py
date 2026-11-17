"""Bounded TSDF reconstruction of each observed time slice.

Unknown space is stored as weight zero and must never be treated as measured.
Each frame is kept separate: combining moving subjects across time would
otherwise smear geometry and falsely imply static surfaces.
"""
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import numpy as np
from .geometry import validate_pose_c2w
from .infer import save_atomic
from .reconstruction5d import Frame5D,_validate_frame,_points_from_frame

@dataclass
class TSDFGrid:
    tsdf:np.ndarray
    weights:np.ndarray
    origin:np.ndarray
    voxel_size:float
    truncation:float
    time_s:float


def integrate_tsdf_frame(f:Frame5D,*,voxel_size:float=.04,truncation:float=.12,
                         max_voxels:int=262144)->TSDFGrid:
    _validate_frame(f)
    if not np.isfinite(voxel_size) or voxel_size<=0 or not np.isfinite(truncation) or truncation<=voxel_size or max_voxels<1:
        raise ValueError('INVALID_TSDF_PARAMETERS')
    pts,_,_,_=_points_from_frame(f,max_points=150000)
    lower=pts.min(axis=0)-truncation*1.05
    upper=pts.max(axis=0)+truncation*1.05
    dims=np.ceil((upper-lower)/voxel_size).astype(np.int64)+1
    if np.any(dims<2) or int(np.prod(dims))>max_voxels:
        raise ValueError('TSDF_RESOURCE_LIMIT')
    tsdf=np.ones(tuple(map(int,dims)),dtype=np.float32)
    weights=np.zeros_like(tsdf,dtype=np.uint8)
    grid_idx=np.indices(tuple(map(int,dims)),dtype=np.int32).reshape(3,-1).T
    # Chunked evaluation (bounded temporary allocations), strict calibrated SE(3).
    pose=validate_pose_c2w(f.pose_c2w)
    w2c=np.linalg.inv(pose)
    k=np.asarray(f.intrinsics,dtype=np.float64)
    depth=np.asarray(f.depth_z,dtype=np.float64)
    valid=np.asarray(f.valid,dtype=bool)
    h,w=depth.shape
    flat=tsdf.reshape(-1)
    flat_weights=weights.reshape(-1)
    for start in range(0,len(grid_idx),50000):
        ix=grid_idx[start:start+50000]
        xworld=lower+ix*voxel_size
        camera=xworld@w2c[:3,:3].T+w2c[:3,3]
        z=camera[:,2]
        safe=z>1e-6
        u=np.zeros(len(ix),dtype=np.int32);v=np.zeros(len(ix),dtype=np.int32)
        y=np.zeros(len(ix));x=np.zeros(len(ix))
        y[safe]=camera[safe,1]/z[safe]
        x[safe]=camera[safe,0]/z[safe]
        u[safe]=np.rint(k[0,0]*x[safe]+k[0,1]*y[safe]+k[0,2]).astype(np.int32)
        v[safe]=np.rint(k[1,1]*y[safe]+k[1,2]).astype(np.int32)
        inside=safe&(u>=0)&(u<w)&(v>=0)&(v<h)
        idx=np.flatnonzero(inside)
        if not len(idx):continue
        actual=valid[v[idx],u[idx]]&np.isfinite(depth[v[idx],u[idx]])&(depth[v[idx],u[idx]]>0)
        idx=idx[actual]
        if not len(idx):continue
        signed=depth[v[idx],u[idx]]-z[idx]
        # Standard projective TSDF excludes deep space behind the visible surface.
        ok=signed>=-truncation
        idx=idx[ok]
        sdf=signed[ok]
        if len(idx):
            flat[start+idx]=np.clip(sdf/truncation,-1,1)
            flat_weights[start+idx]=1
    return TSDFGrid(tsdf,weights,lower.astype(np.float64),float(voxel_size),float(truncation),float(f.time_s))


def extract_surface_mesh(grid:TSDFGrid)->tuple[np.ndarray,np.ndarray]:
    try:
        from skimage.measure import marching_cubes
    except ImportError as ex:
        raise RuntimeError('TSDF_MESH_REQUIRES_SCIKIT_IMAGE') from ex
    values=grid.tsdf.copy()
    # Do not fabricate surfaces in never-observed voxels. Neighbors at zero weight
    # are set to positive side; resulting boundaries remain provisional.
    values[grid.weights==0]=1
    if values.min()>=0 or values.max()<=0:
        return np.empty((0,3),np.float32),np.empty((0,3),np.int32)
    vertices,faces,_,_=marching_cubes(values,level=0,spacing=(grid.voxel_size,)*3)
    # Reject zero crossings adjacent to unknown space; unknown==+1 is only a
    # marching-cubes sentinel, not actual free-space evidence.
    cell=np.floor(vertices/grid.voxel_size).astype(np.int64)
    cell=np.minimum(np.maximum(cell,0),np.asarray(values.shape)-2)
    covered=np.ones(len(vertices),dtype=bool)
    for dx in (0,1):
        for dy in (0,1):
            for dz in (0,1):
                covered&=grid.weights[cell[:,0]+dx,cell[:,1]+dy,cell[:,2]+dz]>0
    faces=faces[covered[faces].all(axis=1)]
    used=np.unique(faces.reshape(-1))
    if not len(used):return np.empty((0,3),np.float32),np.empty((0,3),np.int32)
    map_back=np.full(len(vertices),-1,dtype=np.int32)
    map_back[used]=np.arange(len(used),dtype=np.int32)
    return (vertices[used]+grid.origin).astype(np.float32),map_back[faces].astype(np.int32)


def write_tsdf_npz(grid:TSDFGrid,path:Path)->None:
    data=BytesIO()
    np.savez_compressed(data,tsdf=grid.tsdf,weights=grid.weights,origin=grid.origin,
                        voxel_size=np.array(grid.voxel_size),truncation=np.array(grid.truncation),
                        time_s=np.array(grid.time_s))
    save_atomic(path,data.getvalue())


def write_obj_mesh(vertices:np.ndarray,faces:np.ndarray,path:Path)->None:
    data=BytesIO()
    for v in vertices:data.write(('v '+' '.join(f'{x:.7f}' for x in v)+'\n').encode())
    for f in faces:data.write(('f '+' '.join(str(int(x)+1) for x in f)+'\n').encode())
    save_atomic(path,data.getvalue())
