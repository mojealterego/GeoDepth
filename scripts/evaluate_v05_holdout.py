"""Independent synthetic moving-sphere RGB-D holdout at untrained times.

This raycasting ground truth was never used in train_gaussians and is evaluated
on 2 unseen time instants. It does not represent real-world generalization.
"""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw
from omega_geodepth.trainable_4dgs import (
    load_gaussian_checkpoint,init_gaussians_from_depth,resize_frame_for_training)
from omega_geodepth.reconstruction5d import load_frame_manifest


def raycast(t:float,shape:int):
    h=w=shape
    u,v=np.meshgrid(np.arange(w),np.arange(h))
    focal=1.05*w
    k=np.array([[focal,0,(w-1)/2],[0,focal,(h-1)/2],[0,0,1]],dtype=np.float64)
    x=(u-k[0,2])/focal;y=(v-k[1,2])/focal
    rays=np.stack((x,y,np.ones_like(x)),axis=-1)
    a=(rays*rays).sum(axis=-1)
    center=np.array([-.45+1.125*t,0.,2.25]);radius=.52
    b=-2*(rays*center).sum(axis=-1)
    c=float(center@center-radius*radius)
    disc=b*b-4*a*c
    hit=disc>0
    near=(-b-np.sqrt(np.maximum(disc,0)))/(2*a)
    good=hit&(near>0)
    z=np.full((h,w),4.,np.float32)
    z[good]=near[good].astype(np.float32)
    rgb=np.zeros((h,w,3),np.uint8)
    rgb[...,0]=np.clip(55+30*u/w+15*np.sin(3*x),0,255).astype('uint8')
    rgb[...,1]=np.clip(90+45*v/h,0,255).astype('uint8')
    rgb[...,2]=155
    p=rays*z[...,None]
    normal=np.clip((p-center)/radius,-1,1)
    diffuse=np.clip(.45+.55*(normal[...,2]*(-.6)+normal[...,1]*(-.8)),0,1)
    color=np.stack((210*diffuse+20,75*diffuse+20,62*diffuse+12),axis=-1)
    rgb[good]=np.clip(color,0,255)[good].astype(np.uint8)
    return rgb,z,k,good


def evaluate(weights:Path,source_manifest:Path,output:Path,max_side:int=32):
    original,_=load_frame_manifest(source_manifest)
    first=resize_frame_for_training(original[0],max_side)
    baseline=init_gaussians_from_depth(first,num_splats=72,seed=42)
    model=load_gaussian_checkpoint(weights)
    times=[.2,.6]
    result=[]; previews=[]
    with torch.no_grad():
        for t in times:
            target,depth,k,sphere=raycast(t,max_side)
            for label,m in [('initial',baseline),('trained',model)]:
                rgb,alpha,z=m.render(t,k,np.eye(4),max_side,max_side)
                im=np.clip(np.rint(rgb.detach().numpy()*255),0,255).astype(np.uint8)
                mae=float(np.abs(im.astype(float)-target.astype(float)).mean()/255)
                mse=float(np.mean(((im.astype(float)-target.astype(float))/255)**2))
                zhat=z.detach().numpy()
                mask=(alpha.detach().numpy()>.3)&np.isfinite(zhat)
                depth_mae=float(np.abs(zhat[mask]-depth[mask]).mean()) if mask.any() else None
                result.append({'time_s':t,'model':label,'rgb_mae_normalized':mae,
                               'psnr_db':float(-10*math.log10(max(mse,1e-12))),
                               'depth_mae_visible_units':depth_mae,
                               'coverage_ratio':float((alpha.detach().numpy()>.3).mean())})
                if label=='trained':previews.append(Image.fromarray(im).resize((256,256),Image.Resampling.NEAREST))
            previews.insert(-1 if previews else 0,Image.fromarray(target).resize((256,256),Image.Resampling.NEAREST))
    output.mkdir(parents=True,exist_ok=True)
    panel=Image.new('RGB',(256*4,256),(10,12,16))
    for i,img in enumerate(previews[:4]):panel.paste(img,(256*i,0))
    panel.save(output/'holdout_comparison.png')
    totals={label:np.mean([r['rgb_mae_normalized'] for r in result if r['model']==label]) for label in ('initial','trained')}
    report={'dataset':'analytical_moving_sphere_holdout_not_real_photographs',
            'training_times_s':[0.,.4,.8], 'heldout_times_s':times,
            'rgb_mae_mean':{label:float(v) for label,v in totals.items()},
            'relative_improvement_pct':float(100*(totals['initial']-totals['trained'])/totals['initial']),
            'individual':result}
    (output/'holdout_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    evaluate(a.checkpoint,a.manifest,a.output)
