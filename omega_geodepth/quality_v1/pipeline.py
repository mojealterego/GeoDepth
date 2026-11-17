"""Atomic local processing and verifiable provenance, without hidden I/O."""
from __future__ import annotations
import hashlib
import ctypes
import errno
import sys
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from . import __version__
from .calibration import calibrate_scale,apply_calibration
from .fusion import collect_points,voxel_fuse,write_binary_ply
from .geometry import multiview_consistency,camera_baseline_ratio
from .prediction import Prediction
from .refinement import refine_depth_multiview


def sha256_file(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(8*1024*1024):
            digest.update(block)
    return digest.hexdigest()


def _publish_noreplace(stage: Path, output: Path) -> None:
    """Atomic no-replace publication on Linux; exclusive directory fallback."""
    if sys.platform.startswith('linux'):
        libc=ctypes.CDLL(None,use_errno=True)
        renameat2=getattr(libc,'renameat2',None)
        if renameat2 is not None:
            renameat2.argtypes=(ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint)
            renameat2.restype=ctypes.c_int
            result=renameat2(-100,os.fsencode(stage),-100,os.fsencode(output),1)
            if result==0:
                return
            error=ctypes.get_errno()
            if error==errno.EEXIST:
                raise FileExistsError(f'output exists: {output}')
            if error not in (errno.ENOSYS,errno.EINVAL,errno.EOPNOTSUPP):
                raise OSError(error,os.strerror(error),str(output))
    # Fallback: exclusive reservation prevents overwrite, completion marker
    # signals to consumers that the directory is ready.
    output.mkdir(exist_ok=False)
    try:
        for file in stage.iterdir():
            if file.name!='.complete':
                os.replace(file,output/file.name)
        os.replace(stage/'.complete',output/'.complete')
    except BaseException:
        shutil.rmtree(output)
        raise


def run_pipeline(pred: Prediction, output: str | Path, *,
                 source: dict | None = None, anchors: list[dict] | None = None,
                 model_id: str = 'unspecified_prediction',
                 min_view_coverage: float = 0.,
                 voxel_size: float = 0.025, max_points: int = 1_000_000,
                 confidence_quantile: float = 0.1, relative_tolerance: float = 0.05,
                 min_support: int = 0, min_score: float = 0.,
                 min_valid_ratio: float = 0., min_mean_consistency: float = 0.,
                 min_baseline_ratio: float = 0., save_quality_maps: bool = False,
                 max_targets_per_view: int = 8, refine_depth: bool = False,
                 refine_blend: float = 0.5) -> dict:
    output=Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'output exists: {output}')
    pred.validate()
    if pred.scale_status!='relative':
        raise ValueError('pipeline accepts relative inputs only; apply anchors here')
    if not 0<=min_view_coverage<=1:
        raise ValueError('invalid coverage threshold')
    if not isinstance(model_id,str) or not 1<=len(model_id)<=128:
        raise ValueError('invalid model label')
    if not pred.valid.any():
        raise ValueError('no valid depth')
    for name,value in [('min_valid_ratio',min_valid_ratio),
                       ('min_mean_consistency',min_mean_consistency)]:
        if not 0<=value<=1:
            raise ValueError(f'{name} must be in [0,1]')
    if not np.isfinite(min_baseline_ratio) or min_baseline_ratio<0:
        raise ValueError('min_baseline_ratio must be finite and non-negative')
    valid_ratio=float(np.mean(pred.valid))
    if valid_ratio<min_valid_ratio:
        raise ValueError(f'valid-depth ratio {valid_ratio:.3f} is below required {min_valid_ratio:.3f}')
    calibration=None
    if anchors is not None:
        calibration=calibrate_scale(pred,anchors)
        pred=apply_calibration(pred,calibration)
    # Relative tolerance is invariant to uniform scale.
    check=multiview_consistency(pred,relative_tolerance=relative_tolerance,
                                max_targets_per_view=max_targets_per_view)
    def summarize(candidate, result):
        tested=result['tested']>0
        coverage=(tested.sum(axis=(1,2))/np.maximum(candidate.valid.sum(axis=(1,2)),1)).tolist()
        mean=float(np.mean(result['score'][tested])) if tested.any() else None
        return tested,mean,coverage
    tested,mean_consistency,coverage=summarize(pred,check)
    before={'mean_consistency':mean_consistency,'view_coverage':coverage}
    baseline=camera_baseline_ratio(pred)
    if baseline<min_baseline_ratio:
        raise ValueError('camera parallax below required dimensionless baseline ratio')
    warnings=[]
    if pred.depth.shape[0]>1 and baseline<1e-6:
        warnings.append('No independent parallax: duplicate camera centers cannot establish independent 3D geometry')
    refined_stats={'status':'disabled','changed_pixels':0}
    geometry_pred=pred
    if refine_depth:
        geometry_pred,refined_stats=refine_depth_multiview(
            pred,relative_tolerance=relative_tolerance,
            max_targets_per_view=max_targets_per_view,blend=refine_blend)
        # Final geometry must be audited again, never reuse stale support maps.
        check=multiview_consistency(geometry_pred,relative_tolerance=relative_tolerance,
                                     max_targets_per_view=max_targets_per_view)
        tested,mean_consistency,coverage=summarize(geometry_pred,check)
    if min_mean_consistency>0 and (mean_consistency is None or mean_consistency<min_mean_consistency):
        raise ValueError('mean multiview consistency below required threshold or no consistency evidence')
    if min_view_coverage>0 and min(coverage)<min_view_coverage:
        raise ValueError('view coverage below required threshold')
    if not tested.any():
        warnings.append('No cross-view comparisons were testable')
    points,point_stats=collect_points(geometry_pred,max_points=max_points,
        confidence_quantile=confidence_quantile,consistency=check,
        min_support=min_support,min_score=min_score)
    fused=voxel_fuse(points,voxel_size=voxel_size)
    if not len(fused):
        raise ValueError('empty geometry after quality filtering')
    report={
        'engine':'OMEGA GeoDepth', 'version':__version__,
        'model':model_id, 'scale_status':pred.scale_status,
        'coordinate_system':'camera: X right, Y down, Z forward; extrinsics world-to-camera',
        'unit':'m' if calibration else 'arbitrary_relative_unit',
        'source':source or {}, 'calibration':calibration,'warnings':warnings,
        'refinement':refined_stats,
        'quality_gates':{'min_valid_ratio':min_valid_ratio,
                         'min_mean_consistency':min_mean_consistency,
                         'min_baseline_ratio':min_baseline_ratio,'min_view_coverage':min_view_coverage,'status':'passed'},
        'quality':{'views':int(pred.depth.shape[0]),'height':int(pred.depth.shape[1]),
                   'width':int(pred.depth.shape[2]),'valid_depth_pixels':int(pred.valid.sum()),
                   'invalid_depth_pixels':int(pred.valid.size-pred.valid.sum()),
                   'valid_depth_ratio':valid_ratio,
                   'confidence_semantics':'raw model score; quantile filter, not calibrated probability'},
        'multiview':{'audit_stage':'final_geometry','pre_refinement':before,'view_coverage':coverage,'tested_pixels':int(tested.sum()),
                     'supported_comparisons':int(check['support'].sum()),
                     'tested_comparisons':int(check['tested'].sum()),
                     'mean_consistency':mean_consistency,
                     'max_baseline_over_median_depth':baseline,
                     'relative_tolerance':relative_tolerance,
                     'max_targets_per_view':max_targets_per_view},
        'geometry':{**point_stats,'fused_points':len(fused),
                    'voxel_size':voxel_size,'max_points':max_points,
                    'min_support':min_support,'min_score':min_score,
                    'confidence_quantile':confidence_quantile},
        'limitations':['Uncalibrated depth is relative, not metric',
                       'Model confidence is not a probability',
                       'Consistency measures reprojection agreement, not absolute truth',
                       'No physical accuracy or performance gain is claimed without benchmarks']}
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.omega-geodepth-',dir=output.parent))
    try:
        write_binary_ply(stage/'geometry.ply',fused)
        report['artifacts']={'geometry.ply':sha256_file(stage/'geometry.ply')}
        if save_quality_maps:
            np.savez_compressed(stage/'quality.npz',support=check['support'],
                                tested=check['tested'],score=check['score'])
            report['artifacts']['quality.npz']=sha256_file(stage/'quality.npz')
        (stage/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
        (stage/'.complete').write_text('OMEGA-GEODEPTH-'+__version__+'\n',encoding='ascii')
        _publish_noreplace(stage,output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return report
