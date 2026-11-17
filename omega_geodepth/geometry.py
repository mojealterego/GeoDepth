"""Typed geometry operations: estimation != measurement; fail closed on ambiguous units."""
from __future__ import annotations
from dataclasses import dataclass
import io
import numpy as np

@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    skew: float = 0.0

    def __post_init__(self):
        if not np.all(np.isfinite([self.fx, self.fy, self.cx, self.cy, self.skew])) or min(self.fx, self.fy) <= 0:
            raise ValueError('INVALID_INTRINSICS: positive finite focal lengths required')

    def as_matrix(self) -> np.ndarray:
        return np.array([[self.fx,self.skew,self.cx],[0,self.fy,self.cy],[0,0,1]],dtype=np.float64)


def scaled_intrinsics(k: CameraIntrinsics, offset_x: float, offset_y: float, scale_x: float, scale_y: float) -> CameraIntrinsics:
    """Resize/crop in pixel-center convention (0.5,0.5); scale skew too."""
    if not all(np.isfinite(x) for x in [offset_x,offset_y,scale_x,scale_y]) or scale_x <= 0 or scale_y <= 0:
        raise ValueError('INVALID_CROP_SCALE')
    return CameraIntrinsics(
        k.fx * scale_x, k.fy * scale_y,
        scale_x*(k.cx-offset_x+0.5)-0.5,
        scale_y*(k.cy-offset_y+0.5)-0.5,
        k.skew * scale_x,
    )


def _ray_factors(shape:tuple[int,int],k:CameraIntrinsics)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    h,w=shape
    u,v=np.meshgrid(np.arange(w,dtype=np.float64),np.arange(h,dtype=np.float64))
    y=(v-k.cy)/k.fy
    x=(u-k.cx-k.skew*y)/k.fx
    factor=np.sqrt(x*x+y*y+1)
    return x,y,factor


def optical_z_to_range(optical_z:np.ndarray,k:CameraIntrinsics)->np.ndarray:
    if optical_z.ndim != 2:
        raise ValueError('INVALID_DEPTH_SHAPE')
    return optical_z*_ray_factors(optical_z.shape,k)[2]


def range_to_optical_z(ray_range:np.ndarray,k:CameraIntrinsics)->np.ndarray:
    if ray_range.ndim != 2:
        raise ValueError('INVALID_DEPTH_SHAPE')
    return ray_range/_ray_factors(ray_range.shape,k)[2]


def anchored_scale(prediction:np.ndarray, anchors:np.ndarray, valid:np.ndarray, max_iterations:int=20)->float:
    """Robust positive multiplicative scale in log-depth (Huber IRLS)."""
    if prediction.shape != anchors.shape or prediction.shape != valid.shape or prediction.ndim!=2:
        raise ValueError('ANCHOR_SHAPE_MISMATCH')
    good=np.asarray(valid,dtype=bool) & np.isfinite(prediction) & np.isfinite(anchors) & (prediction>0) & (anchors>0)
    if good.sum()<3:
        raise ValueError('ANCHOR_INSUFFICIENT_VALID_POINTS')
    data=np.log(anchors[good].astype(np.float64))-np.log(prediction[good].astype(np.float64))
    est=float(np.median(data))
    # fixed Huber threshold in log units avoids arbitrary outlier weighting.
    for _ in range(max_iterations):
        residual=data-est
        delta=0.15
        weights=np.minimum(1.0,delta/np.maximum(np.abs(residual),1e-12))
        update=float(np.sum(weights*data)/np.sum(weights))
        if abs(update-est)<1e-10: break
        est=update
    scale=float(np.exp(est))
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('ANCHOR_INVALID_SCALE')
    return scale



@dataclass(frozen=True)
class AnchorAlignment:
    scale: float
    valid_points: int
    covered_grid_cells: int
    inlier_fraction: float
    p90_abs_log_residual: float
    robust_log_scale_sigma: float


def checked_anchor_alignment(prediction:np.ndarray, anchors:np.ndarray, valid:np.ndarray,
                              *, min_points:int=8, min_coverage_cells:int=4,
                              max_residual_log:float=.30)->AnchorAlignment:
    """Gate metric-scale alignment on spatial diversity and robust residuals.

    Does not certify that anchors came from a real sensor: acquisition provenance
    must be independently verified. Failing alignment never claims meter units.
    """
    if prediction.ndim!=2 or prediction.shape!=anchors.shape or prediction.shape!=valid.shape:
        raise ValueError('ANCHOR_SHAPE_MISMATCH')
    good=(np.asarray(valid,dtype=bool) & np.isfinite(prediction) &
          np.isfinite(anchors) & (prediction>0) & (anchors>0))
    n=int(good.sum())
    if n<min_points:
        raise ValueError('ANCHOR_INSUFFICIENT_VALID_POINTS')
    yy,xx=np.nonzero(good)
    h,w=prediction.shape
    # Sixteen spatial cells are enough to detect one-region degeneracy.
    occupied=set(zip((yy*4//h).tolist(),(xx*4//w).tolist()))
    if len(occupied)<min_coverage_cells:
        raise ValueError('ANCHOR_SPATIAL_COVERAGE')
    scale=anchored_scale(prediction,anchors,good)
    residual=np.log(anchors[good].astype(np.float64))-np.log(prediction[good].astype(np.float64))-np.log(scale)
    absres=np.abs(residual)
    p90=float(np.quantile(absres,.90))
    inlier=float(np.mean(absres<=max_residual_log))
    if p90>max_residual_log or inlier<.90:
        raise ValueError('ANCHOR_RESIDUAL_TOO_LARGE')
    # Median absolute deviation, scaled; conservative floor from p90.
    mad=1.4826*float(np.median(np.abs(residual-np.median(residual))))
    scatter=max(mad,p90/1.645,0.0)
    return AnchorAlignment(scale,n,len(occupied),inlier,p90,scatter)


def validate_pose_c2w(pose:np.ndarray) -> np.ndarray:
    p=np.asarray(pose,dtype=np.float64)
    if p.shape != (4,4) or not np.isfinite(p).all() or not np.allclose(p[3],[0,0,0,1],atol=1e-6):
        raise ValueError('INVALID_POSE_C2W')
    r=p[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(r),1,atol=1e-4):
        raise ValueError('INVALID_POSE_C2W_NOT_SE3')
    return p


def export_ply(depth:np.ndarray, rgb:np.ndarray, k:CameraIntrinsics, *, units:str,
               provenance:str, metric_anchor_validated:bool=False, pose_c2w:np.ndarray|None=None,
               max_points:int=1_000_000) -> bytes:
    """Export a color point cloud with units and provenance in header; never fabricate metric depth."""
    if units not in {'meters','arbitrary_scale'}:
        raise ValueError('INVALID_UNITS')
    if units=='meters' and not metric_anchor_validated:
        raise ValueError('METRIC_ANCHOR_REQUIRED')
    if units=='meters' and provenance not in {'scale_aligned','metric_estimated','metric_measured','reconstructed_3d'}:
        raise ValueError('INVALID_METRIC_PROVENANCE')
    if provenance not in {'relative_estimated','scale_aligned','metric_estimated','metric_measured','reconstructed_3d','generated_synthetic'}:
        raise ValueError('INVALID_PROVENANCE')
    if depth.ndim!=2 or rgb.shape!=(depth.shape[0],depth.shape[1],3) or depth.size>max_points:
        raise ValueError('INVALID_SHAPE_OR_RESOURCE_LIMIT')
    if np.asarray(rgb).dtype!=np.uint8:
        raise ValueError('EXPECTED_UINT8_RGB')
    pose=np.eye(4) if pose_c2w is None else validate_pose_c2w(pose_c2w)
    x,y,_=_ray_factors(depth.shape,k)
    mask=np.isfinite(depth)&(depth>0)
    pts=np.column_stack((x[mask]*depth[mask],y[mask]*depth[mask],depth[mask])).astype(np.float64)
    pts=pts@pose[:3,:3].T+pose[:3,3]
    colors=rgb[mask]
    out=io.BytesIO()
    out.write(('ply\nformat binary_little_endian 1.0\n'
               'comment OCAF_GEODEPTH_OMEGA_UNITS='+units+'\n'
               'comment OCAF_GEODEPTH_OMEGA_PROVENANCE='+provenance+'\n'
               'comment COORDINATE_FRAME=opencv_world_right_handed\n'
               f'element vertex {len(pts)}\n'
               'property float x\nproperty float y\nproperty float z\n'
               'property uchar red\nproperty uchar green\nproperty uchar blue\n'
               'end_header\n').encode('ascii'))
    record=np.empty(len(pts),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for idx,key in enumerate(('x','y','z')): record[key]=pts[:,idx].astype(np.float32)
    for idx,key in enumerate(('red','green','blue')): record[key]=colors[:,idx]
    out.write(record.tobytes())
    return out.getvalue()
