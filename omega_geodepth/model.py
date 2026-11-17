"""A small trainable U-Net geometry backbone with a log-depth and log-variance head.

Weights in this distribution are trained from scratch on procedural raycast scenes.
This is NOT Depth Anything 3, nor a general-domain foundation model.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, cin:int, cout:int):
        super().__init__()
        groups=min(4,cout)
        self.net=nn.Sequential(
            nn.Conv2d(cin,cout,3,padding=1),nn.GroupNorm(groups,cout),nn.SiLU(),
            nn.Conv2d(cout,cout,3,padding=1),nn.GroupNorm(groups,cout),
        )
        self.skip=nn.Identity() if cin==cout else nn.Conv2d(cin,cout,1)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return F.silu(self.net(x)+self.skip(x))

class OmegaDepthNet(nn.Module):
    def __init__(self,width:int=16):
        super().__init__()
        if width < 8 or width % 4:
            raise ValueError('width must be divisible by 4 and >=8')
        self.width=width
        self.stem=ResidualBlock(3,width)
        self.e1=ResidualBlock(width,width*2)
        self.e2=ResidualBlock(width*2,width*4)
        self.e3=ResidualBlock(width*4,width*6)
        self.mid=ResidualBlock(width*6,width*6)
        self.d2=ResidualBlock(width*10,width*4)
        self.d1=ResidualBlock(width*6,width*2)
        self.d0=ResidualBlock(width*3,width)
        self.head=nn.Conv2d(width,2,1)

    def forward(self,rgb:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        if rgb.ndim != 4 or rgb.size(1) != 3:
            raise ValueError('RGB_INPUT_MUST_BE_BCHW')
        x0=self.stem(rgb)
        x1=self.e1(F.avg_pool2d(x0,2))
        x2=self.e2(F.avg_pool2d(x1,2))
        x3=self.mid(self.e3(F.avg_pool2d(x2,2)))
        a=self.d2(torch.cat((F.interpolate(x3,size=x2.shape[-2:],mode='bilinear',align_corners=False),x2),1))
        b=self.d1(torch.cat((F.interpolate(a,size=x1.shape[-2:],mode='bilinear',align_corners=False),x1),1))
        c=self.d0(torch.cat((F.interpolate(b,size=x0.shape[-2:],mode='bilinear',align_corners=False),x0),1))
        out=self.head(c)
        log_relative=out[:,0:1]
        log_relative=log_relative-log_relative.mean(dim=(-1,-2),keepdim=True)
        log_variance=out[:,1:2].clamp(min=-7,max=3)
        return log_relative,log_variance
