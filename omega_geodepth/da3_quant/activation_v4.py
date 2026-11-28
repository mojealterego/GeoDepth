"""Calibration-weighted symmetric group quantization for linear layers.
Reference dequantizer, not a packed checkpoint or native low-bit kernel.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class QuantizedLinear:
    codes: torch.Tensor
    scales: torch.Tensor
    in_features: int

    def restore(self):
        return (self.codes.float()*self.scales).flatten(1)[:,:self.in_features]


@torch.no_grad()
def quantize_linear(weight, activations, *, bits=4, group_size=64):
    if bits not in (4,6,8) or not 1<=group_size<=256:
        raise ValueError('invalid quantization configuration')
    if weight.ndim!=2 or activations.ndim!=2 or not weight.numel() or not activations.numel() or weight.shape[1]!=activations.shape[1]:
        raise ValueError('invalid linear/calibration shape')
    if weight.numel()>16_777_216 or activations.numel()>16_777_216 or weight.device!=activations.device:
        raise ValueError('resource or device mismatch')
    w=weight.detach().float(); x=activations.detach().float()
    if not bool(torch.isfinite(w).all()) or not bool(torch.isfinite(x).all()):
        raise ValueError('nonfinite calibration or weights')
    n=w.shape[1]; pad=(-n)%group_size; limit=(1<<(bits-1))-1
    # Normalize before squaring to avoid overflow. Diagonal input covariance proxy.
    x=x/x.abs().max().clamp_min(torch.finfo(x.dtype).tiny)
    importance=x.square().mean(0).clamp_min(1e-8)
    importance=F.pad(importance,(0,pad)).reshape(1,-1,group_size)
    g=F.pad(w,(0,pad)).reshape(w.shape[0],-1,group_size)
    peak=g.abs().amax(-1,keepdim=True)
    normalized=g/torch.where(peak>0,peak,torch.ones_like(peak))
    best=torch.full_like(peak,float('inf')); codes=torch.zeros_like(g,dtype=torch.int8); scales=torch.ones_like(peak)
    for ratio in (1.,.95,.9,.8,.7,.6,.5,.4,.3,.2,.1,.075,.05):
        step=ratio/limit
        q=(normalized/step).round().clamp(-limit,limit)
        error=((q*step-normalized).square()*importance).sum(-1,keepdim=True)
        improve=error<best
        best=torch.where(improve,error,best)
        codes=torch.where(improve,q.to(torch.int8),codes)
        scales=torch.where(improve,(peak*step).clamp_min(torch.finfo(w.dtype).tiny),scales)
    return QuantizedLinear(codes,scales,n)
