"""Experimental feature-to-Gaussian head; no claim of pretrained DA3 integration."""
from dataclasses import dataclass
import torch
from torch import nn


def finite(*xs):
    if any(not bool(torch.isfinite(x).all()) for x in xs):
        raise ValueError('nonfinite input')


def camera(p):
    finite(p)
    if p.shape != (4, 4) or not torch.allclose(p[3], p.new_tensor([0,0,0,1])):
        raise ValueError('invalid camera pose')
    r=p[:3,:3]
    if not torch.allclose(r.T@r, torch.eye(3,device=p.device,dtype=p.dtype),atol=1e-5) or not torch.allclose(torch.linalg.det(r),p.new_tensor(1.),atol=1e-5):
        raise ValueError('invalid camera rotation')


class GaussianHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        if not 3 <= channels <= 4096:
            raise ValueError('invalid feature count')
        self.proj=nn.Linear(channels,6)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, features, rays, depth, pose):
        n=depth.numel()
        if not 1<=n<=512 or depth.shape!=(n,) or rays.shape!=(n,3) or features.shape!=(n,self.proj.in_features):
            raise ValueError('invalid input shape or resource limit')
        finite(features,rays,depth); camera(pose)
        if not bool((depth>0).all()) or not torch.allclose(rays[:,2],torch.ones_like(depth)):
            raise ValueError('invalid optical depth or rays')
        a=self.proj(features)
        z=depth*torch.exp(.5*torch.tanh(a[:,0]))
        xyz=(rays*z[:,None])@pose[:3,:3].T+pose[:3,3]
        rgb=torch.sigmoid(torch.logit(features[:,:3].clamp(.001,.999))+a[:,1:4])
        return xyz,rgb,z*.05*torch.exp(a[:,4].clamp(-2,2)),torch.sigmoid(a[:,5])


def render(g,K,pose,h,w):
    xyz,rgb,sigma,opacity=g; n=len(xyz)
    if not (1<=n<=512 and 1<=h and 1<=w and h*w<=16384 and n*h*w<=2_000_000):
        raise ValueError('render resource limit')
    finite(*g,K); camera(pose)
    if K.shape!=(3,3) or not bool((K.diag()[:2]>0).all()) or not torch.allclose(K[2],K.new_tensor([0,0,1])) or K[1,0]!=0:
        raise ValueError('invalid intrinsics')
    if xyz.shape!=(n,3) or rgb.shape!=(n,3) or sigma.shape!=(n,) or opacity.shape!=(n,) or not bool(((sigma>0)&(opacity>=0)&(opacity<=1)).all()) or not bool(((rgb>=0)&(rgb<=1)).all()):
        raise ValueError('invalid Gaussian parameters')
    c=(xyz-pose[:3,3])@pose[:3,:3]; z=c[:,2]; safe=z.clamp_min(1e-4)
    uv=c@K.T; u,v=(uv[:,:2]/safe[:,None]).unbind(1)
    yy,xx=torch.meshgrid(torch.arange(h,device=z.device,dtype=z.dtype),torch.arange(w,device=z.device,dtype=z.dtype),indexing='ij')
    s=(sigma*(K[0,0]*K[1,1]).sqrt()/safe).clamp(.3,12)
    dist=(xx-u[:,None,None]).square()+(yy-v[:,None,None]).square()
    alpha=(opacity[:,None,None]*(z>1e-4)[:,None,None]*torch.exp(-dist/(2*s[:,None,None].square()))).clamp(0,.995)
    order=torch.argsort(safe,stable=True); alpha=alpha[order]
    trans=torch.cumprod(torch.cat((torch.ones_like(alpha[:1]),1-alpha[:-1]),0),0)
    weights=trans*alpha; total=weights.sum(0)
    return (weights[...,None]*rgb[order,None,None,:]).sum(0),total,(weights*safe[order,None,None]).sum(0)/total.clamp_min(1e-6)


@dataclass(frozen=True)
class View:
    K: torch.Tensor
    pose: torch.Tensor
    rgb: torch.Tensor
    depth: torch.Tensor
    valid: torch.Tensor


def fit(head,features,rays,depth,pose,train,holdout,*,steps=40,lr=.02):
    if not train or not holdout or len(train)+len(holdout)>16 or not 1<=steps<=1000 or not 0<lr<=.1:
        raise ValueError('invalid fit configuration')
    if any(torch.allclose(a,b.pose) for a in [pose,*(v.pose for v in train)] for b in holdout):
        raise ValueError('train/holdout camera overlap')
    for v in (*train,*holdout):
        camera(v.pose); finite(v.K,v.rgb)
        if v.valid.dtype!=torch.bool or v.depth.ndim!=2 or v.valid.shape!=v.depth.shape or v.rgb.shape!=(*v.depth.shape,3) or not bool(v.valid.any()):
            raise ValueError('invalid view')
        finite(v.depth[v.valid])
        if not bool((v.depth[v.valid]>0).all()): raise ValueError('invalid target depth')
    def loss(views):
        g=head(features,rays,depth,pose); terms=[]
        for v in views:
            rgb,a,z=render(g,v.K,v.pose,*v.depth.shape)
            m=v.valid
            terms.append((rgb-v.rgb).square().mean()+(z[m]-v.depth[m]).square().mean()*.25+(1-a[m]).square().mean()*.05)
        return torch.stack(terms).mean()
    with torch.no_grad(): before=float(loss(train)); hb=float(loss(holdout))
    optimizer=torch.optim.Adam(head.parameters(),lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True); objective=loss(train); finite(objective); objective.backward()
        nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True); optimizer.step()
    with torch.no_grad(): return {'train_before':before,'train_after':float(loss(train)),'holdout_before':hb,'holdout_after':float(loss(holdout))}
