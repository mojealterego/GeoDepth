from types import SimpleNamespace
import numpy as np
import pytest
import torch
from omega_geodepth.gsdpt import GaussianHead, render
from omega_geodepth.gsdpt_bridge import prepare_gaussians


def prediction():
    return SimpleNamespace(depth_z=np.full((1,4,5),2.,dtype=np.float32),
        rgb=np.full((1,4,5,3),128,dtype=np.uint8),
        intrinsics_k=np.array([[[4.,.5,2.],[0.,5.,1.5],[0.,0.,1.]]]),
        pose_c2w=np.eye(4)[None],units='arbitrary_scale')


def test_projection_with_skew():
    p=prediction(); p.pose_c2w[0,0,3]=3
    prepared=prepare_gaussians(p,0,max_points=20)
    xyz,*_=GaussianHead(5)(prepared.features,prepared.rays,prepared.depth,prepared.pose)
    c=xyz.detach().numpy()-p.pose_c2w[0,:3,3]
    uv=c@p.intrinsics_k[0].T; uv=uv[:,:2]/uv[:,2:]
    yy,xx=np.indices((4,5))
    np.testing.assert_allclose(uv,np.stack((xx.ravel(),yy.ravel()),1),atol=1e-6)
    assert prepared.units=='arbitrary_scale'
    assert render(GaussianHead(5)(prepared.features,prepared.rays,prepared.depth,prepared.pose),prepared.K,prepared.pose,4,5)[0].shape==(4,5,3)


def test_sampling_and_invalid_depth():
    p=prediction(); p.depth_z[0,0,:]=np.nan
    a=prepare_gaussians(p,0,max_points=7); b=prepare_gaussians(p,0,max_points=7)
    assert a.depth.shape==(7,)
    assert torch.equal(a.rays,b.rays)
    assert torch.isfinite(a.depth).all()
    assert torch.unique(a.rays,dim=0).shape[0]==7


@pytest.mark.parametrize('field,value', [('units','metres'),('rgb',np.zeros((1,4,5,3))),('intrinsics_k',np.zeros((1,3,3))),('pose_c2w',np.zeros((1,4,4))),('depth_z',np.zeros((1,4,5)))])
def test_invalid_prediction(field,value):
    p=prediction(); setattr(p,field,value)
    with pytest.raises(ValueError): prepare_gaussians(p,0)


@pytest.mark.parametrize('index,limit', [(-1,5),(1,5),(0,513),(0,0)])
def test_limits(index,limit):
    with pytest.raises(ValueError): prepare_gaussians(prediction(),index,max_points=limit)
