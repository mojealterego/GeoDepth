import numpy as np
import pytest
from omega_geodepth.ray_consistency import audit_rays, reconcile_camera_gauge
from omega_geodepth.da3_adapter import DA3Output
from omega_geodepth.da3_real import DA3RealCPU


def fixture():
    n,h,w=4,8,12
    poses=np.repeat(np.eye(4)[None],n,axis=0)
    poses[:,:3,3]=[[0,0,0],[1,0,0],[0,1,0],[1,1,0]]
    k=np.repeat(np.array([[[10.,0,6.],[0,8.,4.],[0,0,1.]]]),n,axis=0)
    y,x=np.mgrid[:h,:w];p=np.stack([x+.5,y+.5,np.ones_like(x)],-1)
    d=np.stack([p@np.linalg.inv(a).T for a in k]);o=np.broadcast_to(poses[:,None,None,:3,3],d.shape)
    ray=np.concatenate([d,o],axis=-1);z=np.full((n,h,w),4.,np.float32)
    pred=DA3Output(z,np.ones_like(z),k,poses,np.zeros((n,h,w,3),np.uint8),'arbitrary_scale','relative_estimated','DA3-SMALL','f'*64,True)
    return pred,ray


def test_ideal_rays():
    pred,ray=fixture();r=audit_rays(pred,ray,np.ones(ray.shape[:-1]))
    assert r['worst_direction_p90_deg']<1e-6 and r['worst_origin_p90_over_depth']<1e-10
    assert r['worst_z_normalization_p90']<1e-10


def test_wrong_order_or_magnitude_not_hidden():
    pred,ray=fixture();ray[...,0:3]*=2
    r=audit_rays(pred,ray,np.ones(ray.shape[:-1]))
    assert r['worst_direction_p90_deg']<1e-6 and r['worst_z_normalization_p90']==pytest.approx(1.)
    ray[...,3:]+=1
    assert audit_rays(pred,ray,np.ones(ray.shape[:-1]))['worst_origin_p90_over_depth']>.4


def test_bad_rays_fail_closed():
    pred,ray=fixture();ray[0]=np.nan
    with pytest.raises(ValueError):audit_rays(pred,ray,np.ones(ray.shape[:-1]))
    with pytest.raises(ValueError):audit_rays(pred,ray,np.zeros(ray.shape[:-1]))


def test_known_camera_scale_preserves_geometry_and_metadata():
    pred,_=fixture();ref=pred.pose_c2w.copy();ref[:,:3,3]=2.5*ref[:,:3,3]+[3,4,5]
    r=reconcile_camera_gauge(pred,ref,anchor_indices=[0,1,2],validation_indices=[3],center_tolerance=.01,rotation_tolerance_deg=1.)
    assert r['scale']==pytest.approx(2.5)
    np.testing.assert_allclose(r['aligned_poses'],ref,atol=1e-8)
    np.testing.assert_allclose(r['aligned_depth_z'],pred.depth_z*2.5)
    assert pred.units=='arbitrary_scale' and r['units']=='reference_units_estimated'


def test_reject_bad_reference_holdout():
    pred,_=fixture();ref=pred.pose_c2w.copy();ref[3,:3,3]+=2
    with pytest.raises(ValueError):reconcile_camera_gauge(pred,ref,anchor_indices=[0,1,2],validation_indices=[3],center_tolerance=.01,rotation_tolerance_deg=1.)
    with pytest.raises(ValueError):reconcile_camera_gauge(pred,ref,anchor_indices=[0,1,2],validation_indices=[2,3],center_tolerance=.01,rotation_tolerance_deg=1.)


def test_reject_orientation_mismatch():
    pred,_=fixture();ref=pred.pose_c2w.copy();ref[3,:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    with pytest.raises(ValueError):reconcile_camera_gauge(pred,ref,anchor_indices=[0,1,2],validation_indices=[3],center_tolerance=.01,rotation_tolerance_deg=1.)


def test_actual_ray_decoder():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    e=DA3RealCPU(root/'models/da3',source_dir=root/'vendor/da3_src')
    im=np.random.default_rng(5).integers(0,256,(140,140,3),dtype=np.uint8)
    r=e.inference([im,np.roll(im,3,axis=1)],process_res=140,pose_decoder='ray',audit_depth_rays=True)
    assert r.pose_decoder=='ray' and r.ray_audit['views']==2 and r.units=='arbitrary_scale'
    assert np.isfinite(r.depth_z).any()
    assert not e.net.head._forward_hooks
    with pytest.raises(ValueError):e.inference([im],pose_decoder='unknown')
