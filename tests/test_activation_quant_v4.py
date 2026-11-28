import pytest
import torch
from omega_geodepth.da3_quant.activation_v4 import quantize_linear


def test_activation_sensitive_quantization():
    w=torch.tensor([[10.,.12,.24,.36,.48,.6,.72,.84]])
    x=torch.eye(8); x[0,0]=.001
    q=quantize_linear(w,x,group_size=8)
    baseline=(w/(w.abs().max()/7)).round().clamp(-7,7)*(w.abs().max()/7)
    assert ((x@q.restore().T-x@w.T)**2).mean() < ((x@baseline.T-x@w.T)**2).mean()
    assert q.codes.min()>=-7 and q.codes.max()<=7


def test_padding_and_zero():
    w=torch.zeros(3,5); q=quantize_linear(w,torch.ones(2,5),group_size=4)
    assert torch.equal(q.restore(),w)
    assert q.codes.shape==(3,2,4)

@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite(bad):
    with pytest.raises(ValueError): quantize_linear(torch.tensor([[bad]]),torch.ones(1,1))


def test_shape_validation():
    with pytest.raises(ValueError): quantize_linear(torch.ones(2,3),torch.ones(1,2))
