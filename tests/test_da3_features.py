from types import SimpleNamespace
import numpy as np
import pytest
import torch
from omega_geodepth.da3_features import forward_with_backbone, prepare_backbone_gaussians
from omega_geodepth.gsdpt import GaussianHead


class FeatureNet(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.calls = 0

    def forward(self, x, ext, intr, layers, gs, ray, strategy):
        self.calls += 1
        assert layers == [11] and not gs and strategy == 'middle'
        b,n,_,h,w = x.shape
        grid = torch.arange(h//14*w//14, dtype=torch.float32).reshape(1,1,h//14,w//14,1)
        return {'aux': {'feat_layer_11': grid.expand(b,n,-1,-1,self.channels)}}


@pytest.mark.parametrize('model,channels', [('DA3-SMALL',384),('DA3-BASE',768)])
def test_feature_export_and_head(model,channels):
    net = FeatureNet(channels)
    _, grid = forward_with_backbone(net, torch.zeros(1,1,3,28,28), None, None, model, False)
    assert grid.shape == (1,2,2,channels) and net.calls == 1
    p = SimpleNamespace(depth_z=np.full((1,28,28),2.,np.float32), rgb=np.full((1,28,28,3),128,np.uint8),
        intrinsics_k=np.eye(3)[None],pose_c2w=np.eye(4)[None],units='arbitrary_scale',model_id=model,backbone_features=grid)
    p.depth_z[0,:14] = np.nan
    prepared = prepare_backbone_gaussians(p,0,max_points=392)
    assert prepared.features.shape == (392,5+channels)
    torch.testing.assert_close(prepared.features[:14,5:],torch.full((14,channels),2.))
    torch.testing.assert_close(prepared.features[14:28,5:],torch.full((14,channels),3.))
    out = GaussianHead(5+channels)(prepared.features,prepared.rays,prepared.depth,prepared.pose)
    assert out[0].shape == (392,3)


def test_invalid_grid():
    with pytest.raises(ValueError,match='feature shape'):
        forward_with_backbone(FeatureNet(12),torch.zeros(1,1,3,28,28),None,None,'DA3-SMALL',False)


def test_missing_features():
    net=lambda *args: {'aux': {}}
    with pytest.raises(ValueError,match='missing'):
        forward_with_backbone(net,torch.zeros(1,1,3,28,28),None,None,'DA3-SMALL',False)


def test_nonfinite_features():
    net=lambda *args: {'aux': {'feat_layer_11':torch.full((1,1,2,2,384),float('nan'))}}
    with pytest.raises(ValueError,match='nonfinite'):
        forward_with_backbone(net,torch.zeros(1,1,3,28,28),None,None,'DA3-SMALL',False)


@pytest.mark.parametrize('shape', [(1,1,3,29,28),(2,1,3,28,28),(1,9,3,28,28),(1,1,3,518,518)])
def test_invalid_input(shape):
    with pytest.raises(ValueError):
        forward_with_backbone(FeatureNet(384),torch.zeros(shape),None,None,'DA3-SMALL',False)


def test_alignment_failure():
    p=SimpleNamespace(model_id='DA3-SMALL',depth_z=np.ones((1,28,28),np.float32),backbone_features=np.zeros((1,3,2,384),np.float32))
    with pytest.raises(ValueError,match='alignment'):
        prepare_backbone_gaussians(p,0)
