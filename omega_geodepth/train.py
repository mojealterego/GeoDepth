"""Reproducible CPU training of independent baseline on procedural raycast scenes."""
from __future__ import annotations
import argparse, json, hashlib, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from .model import OmegaDepthNet
from .synthetic import generate_scene, log_relative_depth


def dataset(count:int, start_seed:int, size:int) -> tuple[torch.Tensor,torch.Tensor]:
    xs=[];ys=[]
    for seed in range(start_seed,start_seed+count):
        rgb,depth=generate_scene(seed,(size,size))
        xs.append(torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))))
        ys.append(torch.from_numpy(np.ascontiguousarray(log_relative_depth(depth)[None])))
    return torch.stack(xs),torch.stack(ys)

@torch.no_grad()
def evaluate(model:OmegaDepthNet, xs:torch.Tensor,ys:torch.Tensor,batch:int=16) -> dict:
    model.eval()
    absolute_sum=0.0; samples=0; nll_sum=0.0
    for start in range(0,len(xs),batch):
        logz,logvar=model(xs[start:start+batch])
        residual=logz-ys[start:start+batch]
        absolute_sum+=residual.abs().sum().item()
        nll_sum+=(residual.square()*torch.exp(-logvar)+logvar).sum().item()/2
        samples+=residual.numel()
    return {'log_depth_mae': absolute_sum/samples, 'heteroskedastic_nll': nll_sum/samples,
            'constant_log_depth_baseline_mae': ys.abs().mean().item()}


def train(args:argparse.Namespace)->dict:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    x_train,y_train=dataset(args.samples,0,args.size)
    x_val,y_val=dataset(args.val_samples,50_000,args.size)
    model=OmegaDepthNet(args.width)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    history=[];tic=time.monotonic()
    model.train()
    generator=torch.Generator().manual_seed(args.seed)
    for step in range(1,args.steps+1):
        idx=torch.randint(0,args.samples,(args.batch,),generator=generator)
        xs,ys=x_train[idx],y_train[idx]
        logz,logvar=model(xs)
        delta=logz-ys
        # Geometry supervison + lightweight log-depth uncertainty head (synthetic only)
        point=F.smooth_l1_loss(logz,ys,beta=.12)
        dx=F.l1_loss(logz[:,:,:,1:]-logz[:,:,:,:-1],ys[:,:,:,1:]-ys[:,:,:,:-1])
        dy=F.l1_loss(logz[:,:,1:,:]-logz[:,:,:-1,:],ys[:,:,1:,:]-ys[:,:,:-1,:])
        nll=.5*(delta.square()*torch.exp(-logvar)+logvar).mean()
        loss=point+.10*(dx+dy)+.04*nll
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optim.step()
        if step==1 or step%max(1,args.steps//10)==0 or step==args.steps:
            history.append({'step':step,'train_loss':round(loss.item(),6)})
            print(f'{step}/{args.steps} loss={loss.item():.6f}',flush=True)
    final=evaluate(model,x_val,y_val)
    duration=round(time.monotonic()-tic,2)
    out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    version=getattr(args,'release_version','0.2.0')
    base=f'omega_geodepth_synthetic_v{version.rsplit(".",1)[0]}'
    weights=out/f'{base}.safetensors'
    save_file({k:v.detach().contiguous().cpu() for k,v in model.state_dict().items()},str(weights))
    model.eval()
    # Scripting validates control flow and supports variable H/W (multiples of 8).
    scripted=torch.jit.script(model)
    scripted=torch.jit.freeze(scripted)
    torch.jit.save(scripted,str(out/f'{base}.torchscript.pt'))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [weights,out/f'{base}.torchscript.pt']}
    summary={
        'model':'OMEGA GeoDepth Ω synthetic-trained relative depth model',
        'version':version,'weights_file':weights.name,'torchscript_file':f'{base}.torchscript.pt','architecture':'small residual encoder-decoder with log-depth / log-variance head',
        'provenance':'trained_from_scratch_synthetic_raycast_v0.2','weight_origin':'fully local random initialization plus supervised training',
        'not_depth_anything_3':True,'not_foundation_model':True,'not_validated_on_real_photographs':True,
        'not_abliterated_language_model':True,'content_classifier_present':False,'nsfw_blocklist_present':False,
        'units':'arbitrary_scale','depth_definition':'optical_z_relative','uncertainty_status':'trained on synthetic data; NOT calibrated on real data',
        'width':args.width,'parameters':sum(p.numel() for p in model.parameters()),
        'training':{'steps':args.steps,'train_samples':args.samples,'validation_samples':args.val_samples,
                    'resolution':[args.size,args.size],'seed':args.seed,'batch':args.batch,'threads':args.threads,
                    'duration_seconds':duration,'optimizer':'AdamW','last_loss':history[-1]['train_loss']},
        'validation':final,'loss_history':history,'weights_sha256':hashes,
        'limits':['procedural training only','no real-world accuracy guarantees','no measured metric depth',
                  'uncertainty is not conformally calibrated','single-image depth only, no 4D or camera pose'],
    }
    (out/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description='Train reproducible geometry baseline')
    p.add_argument('--output',default='models');p.add_argument('--steps',type=int,default=400)
    p.add_argument('--samples',type=int,default=512);p.add_argument('--val-samples',type=int,default=96)
    p.add_argument('--size',type=int,default=64);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--width',type=int,default=16);p.add_argument('--threads',type=int,default=3)
    p.add_argument('--lr',type=float,default=.0007);p.add_argument('--seed',type=int,default=20261115)
    p.add_argument('--release-version',default='0.2.0')
    args=p.parse_args()
    if any(v<=0 for v in (args.steps,args.samples,args.val_samples,args.size,args.batch,args.threads)) or args.size%8:
        p.error('All counts >0; size divisible by 8')
    print(json.dumps(train(args),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
