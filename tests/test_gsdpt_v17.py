import pytest
import torch
from omega_geodepth.gsdpt import GaussianHead, render, fit, View

def scene():
    torch.manual_seed(17)
    k=torch.tensor([[8.,0,3.5],[0,8.,3.5],[0,0,1.]])
    p=torch.eye(4)
    q=p.clone(); q[0,3]=.15
    rgb=torch.rand(64,3)*.8+.1
    yy,xx=torch.meshgrid(torch.arange(8),torch.arange(8),indexing='ij')
    rays=torch.stack(((xx.flatten()-3.5)/8,(yy.flatten()-3.5)/8,torch.ones(64)),1)
    features=torch.cat((rgb,rays[:,:2]),1)
    z=torch.full((64,),2.)
    head=GaussianHead(5)
    # Independent target generated from known geometry, not the trainable head.
    truth=(rays*2.2,rgb,torch.full((64,),.15),torch.full((64,),.8))
    targets=[]
    for pose in (p,q):
        im,a,d=render(truth,k,pose,8,8)
        targets.append(View(k,pose,im,d,a>.1))
    return head,features,rays,z,targets

def test_fit_and_distinct_holdout():
    h,f,r,z,views=scene()
    result=fit(h,f,r,z,torch.eye(4),views[:1],views[1:],steps=40)
    assert result['train_after'] < result['train_before']
    assert result['holdout_after'] < result['holdout_before']

def test_gradient():
    h,f,r,z,views=scene()
    out=render(h(f,r,z,torch.eye(4)),views[0].K,views[0].pose,8,8)
    out[0].mean().backward()
    assert h.proj.weight.grad.abs().sum()>0

def test_reject_holdout_leak():
    h,f,r,z,v=scene()
    with pytest.raises(ValueError,match='overlap'):
        fit(h,f,r,z,torch.eye(4),v[:1],v[:1])

def test_reject_invalid_depth():
    h,f,r,z,v=scene(); z[0]=float('nan')
    with pytest.raises(ValueError): h(f,r,z,torch.eye(4))

def test_reject_resource_limit():
    h,f,r,z,v=scene()
    with pytest.raises(ValueError): render(h(f,r,z,torch.eye(4)),v[0].K,v[0].pose,1024,1024)

def test_reject_source_holdout_leak():
    h,f,r,z,v=scene()
    with pytest.raises(ValueError,match='overlap'):
        fit(h,f,r,z,v[1].pose,v[:1],v[1:])
