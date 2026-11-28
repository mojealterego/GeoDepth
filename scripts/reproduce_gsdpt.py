"""Reproduce a trained experimental head on a controlled synthetic scene."""
import argparse
import json
from pathlib import Path

import torch
from omega_geodepth.gsdpt import GaussianHead, View, fit, render
from omega_geodepth.gsdpt_checkpoint import load_head, save_head


def reproduce(output: Path) -> dict:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        k=torch.tensor([[8.,0,3.5],[0,8.,3.5],[0,0,1.]])
        p=torch.eye(4)
        q=p.clone(); q[0,3]=.15
        rgb=torch.rand(64,3)*.8+.1
        yy,xx=torch.meshgrid(torch.arange(8),torch.arange(8),indexing='ij')
        rays=torch.stack(((xx.flatten()-3.5)/8,(yy.flatten()-3.5)/8,torch.ones(64)),1)
        features=torch.cat((rgb,rays[:,:2]),1)
        z=torch.full((64,),2.)
        head=GaussianHead(5)
        truth=(rays*2.2,rgb,torch.full((64,),.15),torch.full((64,),.8))
        views=[]
        for pose in (p,q):
            im,a,d=render(truth,k,pose,8,8)
            views.append(View(k,pose,im,d,a>.1))
        report=fit(head,features,rays,z,p,views[:1],views[1:],steps=40)
        if not (report['train_after']<report['train_before'] and report['holdout_after']<report['holdout_before']):
            raise RuntimeError('synthetic fit failed quality gate')
        digest=save_head(head,output)
        restored=load_head(output,digest)
        with torch.no_grad():
            a=render(head(features,rays,z,p),k,q,8,8)
            b=render(restored(features,rays,z,p),k,q,8,8)
            exact=all(torch.equal(x,y) for x,y in zip(a,b))
        if not exact:
            raise RuntimeError('checkpoint changed rendering')
        return dict(report,sha256=digest,reload_exact=exact,seed=17,steps=40,
                    scope='synthetic RGB/ray head; same renderer for targets; no DA3 weights')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(reproduce(args.output),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
