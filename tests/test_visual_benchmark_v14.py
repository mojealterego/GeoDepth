import numpy as np
import pytest
from omega_geodepth.visual_benchmark import fit_similarity, pose_metrics, cloud_metrics, evaluate_scene, select_context_views


def poses(centers):
    p=np.broadcast_to(np.eye(4),(len(centers),4,4)).copy()
    p[:,:3,3]=centers
    return p


def test_known_similarity_and_outlier():
    rng=np.random.default_rng(11); x=rng.normal(size=(12,3)); r=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    y=2.3*x@r.T+[1,2,3]; y[3]+=25
    a=fit_similarity(x,y,inlier_threshold=.01,seed=8)
    np.testing.assert_allclose(a.apply(x[np.arange(12)!=3]),y[np.arange(12)!=3],atol=1e-10)
    assert a.scale==pytest.approx(2.3) and sum(a.inliers)==11

@pytest.mark.parametrize('x',[np.zeros((5,3)),np.array([[i,0,0] for i in range(5)])])
def test_degenerate_alignment(x):
    with pytest.raises(ValueError): fit_similarity(x,x,inlier_threshold=.1)


def test_pose_invariance_and_zero_baselines():
    p=poses([[0,0,0],[1,0,0],[0,1,0],[0,0,0]])
    q=p.copy();q[:,:3,3]=3*p[:,:3,3]+[5,4,3]
    report=pose_metrics(p,q)
    assert report['auc_3']==pytest.approx(1) and report['auc_30']==pytest.approx(1)
    assert report['translation_pairs']==5 and report['total_pairs']==6
    with pytest.raises(ValueError):pose_metrics(poses([[0,0,0],[0,0,0]]),poses([[0,0,0],[0,0,0]]))


def test_pose_translation_error():
    r=pose_metrics(poses([[0,0,0],[1,0,0]]),poses([[0,0,0],[0,1,0]]))
    assert r['auc_30']==0 and r['translation_mean_deg']==pytest.approx(90)


def test_asymmetric_cloud_and_f1():
    a=np.array([[0.,0,0],[1,0,0]]);b=np.array([[0.,0,0],[1,0,0],[2,0,0],[3,0,0]])
    r=cloud_metrics(a,b,threshold=.1)
    assert r['accuracy']==0 and r['completeness']==.75 and r['chamfer']==.375
    assert r['precision']==1 and r['recall']==.5 and r['f1']==pytest.approx(2/3)


def test_holdout_alignment_cannot_use_evaluation():
    p=poses([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[2,1,0],[2,2,0]])
    q=p.copy();q[4:,:3,3]+=20
    x=np.array([[0.,0,2],[1,0,2],[0,1,2]])
    r=evaluate_scene(p,q,x,x,calibration_indices=[0,1,2,3],evaluation_indices=[4,5],inlier_threshold=.001,distance_threshold=.1,reference_units='m')
    assert r['alignment']['scale']==pytest.approx(1) and r['heldout_center_rmse']>20
    with pytest.raises(ValueError):evaluate_scene(p,q,x,x,calibration_indices=[0,1,2,3],evaluation_indices=[3,5],inlier_threshold=.001,distance_threshold=.1,reference_units='m')


def test_context_split():
    p=poses([[i,0,0] for i in range(10)])
    a=select_context_views(p,context_count=3,target_stride=4)
    assert a['targets']==[0,4,8] and len(a['context'])==3 and len(set(a['context']))==3
    assert not set(a['context'])&set(a['targets'])
    assert a==select_context_views(p,context_count=3,target_stride=4)


def test_invalid_cloud_and_pose():
    with pytest.raises(ValueError):cloud_metrics(np.zeros((0,3)),np.zeros((1,3)),threshold=.1)
    with pytest.raises(ValueError):cloud_metrics(np.array([[np.nan,0,0]]),np.zeros((1,3)),threshold=.1)
    p=poses([[0,0,0],[1,0,0]]);p[0,0,0]=-1
    with pytest.raises(ValueError):pose_metrics(p,p)


def test_pose_conditioning_contract():
    from omega_geodepth.da3_real import validate_camera_conditions, normalize_camera_conditions
    import torch
    p=poses([[0,0,0],[1,0,0],[0,2,0]])
    k=np.broadcast_to(np.array([[200.,0,70],[0,200,70],[0,0,1.]]),(3,3,3)).copy()
    e,i=validate_camera_conditions(p,k,3)
    np.testing.assert_allclose(e,np.linalg.inv(p));np.testing.assert_allclose(i,k)
    t=normalize_camera_conditions(torch.from_numpy(e)[None]);np.testing.assert_allclose(t[0,0],np.eye(4));assert t[0,1,0,3]==pytest.approx(-1.)
    with pytest.raises(ValueError):validate_camera_conditions(p,None,3)
    with pytest.raises(ValueError):validate_camera_conditions(p,k,2)
    k[0,0,0]=-1
    with pytest.raises(ValueError):validate_camera_conditions(p,k,3)


def test_real_pose_conditioned_small():
    from pathlib import Path
    from omega_geodepth.da3_real import DA3RealCPU
    root=Path(__file__).resolve().parents[1]
    e=DA3RealCPU(root/'models/da3',source_dir=root/'vendor/da3_src')
    p=poses([[0,0,0],[.25,0,0]])
    k=np.repeat(np.array([[[120.,0,69.5],[0,120,69.5],[0,0,1.]]]),2,axis=0)
    im=np.random.default_rng(5).integers(0,256,(140,140,3),dtype=np.uint8)
    result=e.inference([im,np.roll(im,3,axis=1)],process_res=140,camera_poses_c2w=p,camera_intrinsics=k)
    assert result.pose_conditioned and result.units=='arbitrary_scale' and result.provenance=='relative_estimated'
    assert np.isfinite(result.depth_z).any() and result.depth_z.shape==(2,140,140)
