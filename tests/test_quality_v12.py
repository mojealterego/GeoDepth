import io
import zipfile
from pathlib import Path
import numpy as np
import pytest


def scene(n=2,h=12,w=16):
    from omega_geodepth.quality_v1.prediction import Prediction
    d=np.full((n,h,w),2.,dtype=np.float32)
    k=np.repeat(np.array([[[20.,0.,7.5],[0.,20.,5.5],[0.,0.,1.]]]),n,axis=0)
    e=np.repeat(np.eye(4)[None,:3],n,axis=0)
    e[:,0,3]=np.arange(n)*.1
    return Prediction(d,np.ones_like(d),e,k)


def test_native_npz_import(tmp_path):
    from omega_geodepth.quality_v1.prediction import load_prediction_npz
    p=scene(); pose=np.linalg.inv(np.concatenate([p.extrinsics,np.repeat(np.array([[[0.,0.,0.,1.]]]),2,axis=0)],axis=1))
    path=tmp_path/'native.npz'; np.savez(path,depth_z=p.depth,confidence_raw=p.conf,intrinsics_k=p.intrinsics,pose_c2w=pose,units='arbitrary_scale',provenance='relative_estimated')
    result=load_prediction_npz(path)
    assert np.allclose(p.extrinsics,result.extrinsics) and result.scale_status=='relative'


def test_reject_forged_metric(tmp_path):
    from omega_geodepth.quality_v1.prediction import load_prediction_npz
    p=scene(); path=tmp_path/'metric.npz'; np.savez(path,depth=p.depth,conf=p.conf,intrinsics=p.intrinsics,extrinsics=p.extrinsics,units='meters')
    with pytest.raises(ValueError,match='metric|relative'):
        load_prediction_npz(path)


def test_reject_huge_header_before_load(tmp_path):
    from omega_geodepth.quality_v1.prediction import load_prediction_npz
    path=tmp_path/'huge.npz'; b=io.BytesIO(); np.lib.format.write_array_header_1_0(b,{'shape':(2,100000,100000),'fortran_order':False,'descr':'<f4'})
    with zipfile.ZipFile(path,'w') as z:
        for name in ['depth','conf','extrinsics','intrinsics']:z.writestr(name+'.npy',b.getvalue())
    with pytest.raises(ValueError,match='limit|payload'):
        load_prediction_npz(path)


def test_coverage_gate(tmp_path):
    from omega_geodepth.quality_v1.pipeline import run_pipeline
    p=scene();p.extrinsics[1,0,3]=100.
    with pytest.raises(ValueError,match='coverage'):
        run_pipeline(p,tmp_path/'out',min_view_coverage=.1)
    assert not (tmp_path/'out').exists()


def test_empty_output_gate(tmp_path):
    from omega_geodepth.quality_v1.pipeline import run_pipeline
    p=scene();p.depth[:]=np.nan
    with pytest.raises(ValueError,match='empty|no valid'):
        run_pipeline(p,tmp_path/'out')


def test_refinement_audits_final_geometry(tmp_path):
    from omega_geodepth.quality_v1.pipeline import run_pipeline
    p=scene();p.depth[1]*=1.03
    out=tmp_path/'out';r=run_pipeline(p,out,refine_depth=True,min_view_coverage=.1,save_quality_maps=True)
    assert r['refinement']['changed_pixels']>0
    assert 'pre_refinement' in r['multiview'] and r['multiview']['audit_stage']=='final_geometry'
    assert r['model']=='unspecified_prediction' and (out/'.complete').exists()


def test_voxel_overflow():
    from omega_geodepth.quality_v1.fusion import voxel_fuse
    with pytest.raises(ValueError,match='overflow'):
        voxel_fuse(np.array([[1e30,0,0]],dtype=np.float32),1e-12)


def test_no_fractional_anchor():
    from omega_geodepth.quality_v1.calibration import calibrate_scale
    with pytest.raises(ValueError,match='anchor'):
        calibrate_scale(scene(),[{'view':0,'x':x+.2,'y':y,'depth_m':4.} for x,y in [(0,0),(4,4),(8,8),(12,11)]])


def test_no_double_calibration():
    from omega_geodepth.quality_v1.calibration import calibrate_scale,apply_calibration
    p=scene(); a=[{'view':0,'x':x,'y':y,'depth_m':4.} for x,y in [(0,0),(4,4),(8,8),(12,11)]]
    aligned=apply_calibration(p,calibrate_scale(p,a))
    with pytest.raises(ValueError,match='relative'):
        calibrate_scale(aligned,a)


def test_base_only_accepts_fp32(tmp_path):
    from omega_geodepth.da3_real import DA3RealCPU
    with pytest.raises(ValueError,match='BASE.*FP32'):
        DA3RealCPU(tmp_path,source_dir=tmp_path,model_id='DA3-BASE',profile='fp16')


def test_static_bridge_preserves_simultaneous_time_and_units():
    from omega_geodepth.quality_v1.bridge import prepare_static_field
    from omega_geodepth.da3_adapter import DA3Output
    p=scene(); hom=np.concatenate([p.extrinsics,np.repeat(np.array([[[0.,0.,0.,1.]]]),2,axis=0)],axis=1)
    raw=DA3Output(p.depth,p.conf,p.intrinsics,np.linalg.inv(hom),np.zeros((*p.depth.shape,3),np.uint8),'arbitrary_scale','relative_estimated','DA3-SMALL','a'*64)
    samples,report=prepare_static_field(raw,assume_static=True,reference_radius=1.,min_view_coverage=.1)
    assert samples.units=='arbitrary_scale' and np.all(samples.coordinates[:,3]==0)
    assert samples.time_basis=='static_assumption' and report['view_count']==2 and report['qualified_samples']>0
    with pytest.raises(ValueError,match='static'):
        prepare_static_field(raw,assume_static=False,reference_radius=1.)


def test_evaluation_reports_missing_depth():
    from omega_geodepth.quality_v1.metrics import depth_metrics
    report=depth_metrics(np.array([1.,np.nan]),np.array([1.,3.]))
    assert report['ground_truth_valid_samples']==2 and report['prediction_coverage']==.5


def test_base_missing_checkpoint_fail_closed(tmp_path):
    from omega_geodepth.da3_real import DA3RealCPU
    root=Path(__file__).resolve().parents[1]
    from shutil import copyfile
    copyfile(root/'models/da3_base/config.json',tmp_path/'config.json')
    with pytest.raises(FileNotFoundError,match='CHECKPOINT_MANIFEST_MISSING'):
        DA3RealCPU(tmp_path,source_dir=root/'vendor/da3_src',model_id='DA3-BASE')
