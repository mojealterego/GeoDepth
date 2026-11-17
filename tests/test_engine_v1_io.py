import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pytest
from omega_geodepth.quality_v1.prediction import load_prediction_npz, Prediction
from omega_geodepth.quality_v1.fusion import voxel_fuse, write_binary_ply, collect_points, confidence_mask
from omega_geodepth.quality_v1.metrics import depth_metrics
from test_engine_v1_geometry import make_prediction


def test_npz_loading_with_strict_array_validation(tmp_path):
    p=make_prediction()
    f=tmp_path/'good.npz'
    np.savez_compressed(f, depth=p.depth, conf=p.conf, extrinsics=p.extrinsics, intrinsics=p.intrinsics)
    actual=load_prediction_npz(f)
    assert actual.depth.shape == (2,3,4)


def test_npz_disallows_unexpected_members(tmp_path):
    f=tmp_path/'bad.npz'
    with zipfile.ZipFile(f,'w') as z:
        z.writestr('../escape.npy', b'garbage')
    with pytest.raises(ValueError,match='Unexpected'):
        load_prediction_npz(f)


def test_npz_rejects_size_limit(tmp_path):
    p=make_prediction()
    f=tmp_path/'too_big.npz'
    np.savez_compressed(f,depth=p.depth,conf=p.conf,extrinsics=p.extrinsics,intrinsics=p.intrinsics)
    with pytest.raises(ValueError,match='limit'):
        load_prediction_npz(f,max_uncompressed_bytes=10)


def test_voxel_fusion_deterministic_and_weighted():
    pts=np.array([[0.,0.,0.],[.1,0.,0.],[3.,0.,0.]],dtype=np.float32)
    a=voxel_fuse(pts,voxel_size=1.)
    b=voxel_fuse(pts[::-1],voxel_size=1.)
    assert a.shape==(2,3)
    assert np.allclose(a,b)
    assert np.allclose(a[0], [.05,0.,0.])


def test_confidence_quantile_does_not_claim_probability():
    p=make_prediction(n=1,h=2,w=2)
    p.conf[0]=np.array([[-10., 1.],[2., 100.]])
    mask=confidence_mask(p,quantile=.5)
    assert mask.sum()==2


def test_ply_export_has_valid_binary_header_and_points(tmp_path):
    out=tmp_path/'out.ply'
    points=np.array([[1,2,3],[4,5,6]],dtype=np.float32)
    write_binary_ply(out,points)
    data=out.read_bytes()
    assert b'element vertex 2\n' in data
    assert data.endswith(points.astype('<f4').tobytes())


def test_metrics_are_zero_on_exact_ground_truth():
    gt=np.array([[1.,2.],[3.,4.]])
    result=depth_metrics(gt,gt)
    assert result['abs_rel']==0.
    assert result['rmse']==0.
    assert result['delta_1']==1.


def test_metrics_refuse_empty_ground_truth():
    with pytest.raises(ValueError,match='valid'):
        depth_metrics(np.ones((2,2)),np.zeros((2,2)))


def test_collect_points_enforces_budget():
    p=make_prediction(n=2,h=4,w=4)
    points, stats=collect_points(p, max_points=8)
    assert len(points)==8
    assert stats['selected']==8


def test_npz_allows_safe_additional_da3_arrays_without_loading_them(tmp_path):
    p=make_prediction()
    f=tmp_path/'extended.npz'
    np.savez_compressed(f,depth=p.depth,conf=p.conf,extrinsics=p.extrinsics,
                        intrinsics=p.intrinsics,auxiliary=np.ones((1,2),dtype=np.float32))
    loaded=load_prediction_npz(f)
    assert loaded.depth.shape==p.depth.shape
