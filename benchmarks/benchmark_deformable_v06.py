"""Reproducible synthetic validation of learned nonlinear motion vs v0.5 linear model.

The test is not real-world RGB-D: it renders independent analytic trajectories of
one Gaussian whose true center is prescribed by a known motion equation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import math
import numpy as np
import torch
from PIL import Image
from omega_geodepth.reconstruction5d import Frame5D
from omega_geodepth.trainable_4dgs import TrainableGaussian4D,GaussianTrainConfig,train_gaussians
from omega_geodepth.deformable_4dgs import (
    DeformableGaussian4D,heldout_metrics,save_deformable_checkpoint,
)


def camera()->np.ndarray:
    return np.array([[24.,0.,11.5],[0.,24.,11.5],[0.,0.,1.]])


def initial_params():
    return dict(means_xyz=torch.tensor([[0.,0.,2.2]],dtype=torch.float32),
        velocity=torch.zeros(1,3),rgb=torch.tensor([[.95,.15,.04]]),
        sigma_world=torch.tensor([[.15]]),opacity=torch.tensor([[.97]]),
        reference_time=0.)


def analytic_x(t:float,trajectory:str)->float:
    if trajectory=='reversal': return .30*math.sin(math.pi*t)
    if trajectory=='acceleration':return .19*t*t-.18*t
    raise ValueError('unknown trajectory')


def frames_for(times:tuple[float,...],trajectory:str)->list[Frame5D]:
    generator=TrainableGaussian4D(**initial_params())
    result=[]
    with torch.no_grad():
        for time_s in times:
            generator.means_xyz[0,0]=analytic_x(time_s,trajectory)
            rgb,alpha,depth=generator.render(0.,camera(),np.eye(4),24,24)
            a=alpha.cpu().numpy()
            valid=a>.03
            if not np.any(valid):raise RuntimeError('insufficient synthetic coverage')
            result.append(Frame5D(time_s,np.where(valid,depth.numpy(),np.nan).astype('float32'),
                np.rint(rgb.numpy()*255).clip(0,255).astype('uint8'),camera(),np.eye(4),valid))
    return result


def report(output:Path,steps:int=160,seed:int=42,depth_weight:float=.6,deformation_regularization:float=.05)->dict:
    output=Path(output)
    output.mkdir(exist_ok=True,parents=True)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    times_train=(0.,.5,1.,1.5,2.)
    times_holdout=(.25,.75,1.25,1.75)
    times_audit=(.125,.375,.625,.875,1.125,1.375,1.625,1.875)
    trajectories=['reversal','acceleration']
    result={"schema":"omega-deformable-v06-benchmark-v1",
            "dataset":"analytical synthetic moving Gaussian, 24x24, 1 splat, 2 distinct trajectories",
            "train_times_s":times_train,"validation_times_s":times_holdout,
            "additional_audit_times_s":times_audit,
            "training_steps_each":steps,"optimization":"Adam, lr=0.055, motion parameters only",
            "seed":seed,"depth_weight":depth_weight,"deformation_regularization":deformation_regularization,"results":[]}
    panels=[]
    for name in trajectories:
        train=frames_for(times_train,name)
        held=frames_for(times_holdout,name)
        audit=frames_for(times_audit,name)
        baseline=TrainableGaussian4D(**initial_params())
        deform=DeformableGaussian4D(**initial_params(),knot_times=times_train)
        cfg=GaussianTrainConfig(steps=steps,lr=.055,optimize_velocity_only=True,
            timeout_s=200.,velocity_regularization=1e-5,deformation_regularization=deformation_regularization,depth_weight=depth_weight)
        base_train=train_gaussians(baseline,train,cfg)
        deform_train=train_gaussians(deform,train,cfg)
        base_metrics=heldout_metrics(baseline,held)
        deform_metrics=heldout_metrics(deform,held)
        audit_base=heldout_metrics(baseline,audit)
        audit_deform=heldout_metrics(deform,audit)
        base=base_metrics['rgb_mae_valid'];new=deform_metrics['rgb_mae_valid']
        if not (math.isfinite(base) and math.isfinite(new)):
            raise RuntimeError('nonfinite benchmark results')
        row={"trajectory":name,"linear_v05":base_metrics,
             "deformable_v06":deform_metrics,
             "additional_audit_linear":audit_base,
             "additional_audit_deformable":audit_deform,
             "improvement_pct_on_heldout_rgb_mae":100*(base-new)/base if base>0 else None,
             "train_loss_linear":base_train['loss_final'],
             "train_loss_deformable":deform_train['loss_final']}
        result['results'].append(row)
        if name=='reversal':
            save_deformable_checkpoint(deform,output/'checkpoint',units='arbitrary_scale',
                provenance='relative_estimated',training_report={"steps_completed":steps,
                    "trajectory":name,"source":"synthetic teacher"})
        time=.75
        gt=held[1]
        for m in (baseline,deform):
            with torch.no_grad():
                rgb,alpha,d=m.render(time,camera(),np.eye(4),24,24)
            panels.append(np.rint(rgb.detach().numpy()*255).clip(0,255).astype('uint8'))
        panels.append(gt.rgb)
    grid=np.concatenate((np.concatenate(panels[:3],axis=1),np.concatenate(panels[3:],axis=1)),axis=0)
    Image.fromarray(grid).resize((72*4,48*4),Image.Resampling.NEAREST).save(output/'comparison.png')
    result['comparison_image_columns']=['linear_v05','deformable_v06','analytic_ground_truth']
    result['comparison_image_rows']=trajectories
    result['warnings']=['Synthetic-only; no claim of generalization to real images',
        'Reference teacher shares Gaussian rasterization family; validation tests temporal interpolation',
        'RGB MAE is evaluated only on pixels with valid observed depth',
        'The model does not infer depth outside observed regions',
        'Hyperparameters were evaluated on validation times; additional audit times are separately reported']
    (output/'benchmark.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=160)
    parser.add_argument("--depth-weight",type=float,default=.6)
    parser.add_argument("--deform-reg",type=float,default=.05)
    args=parser.parse_args()
    print(json.dumps(report(args.output,args.steps,depth_weight=args.depth_weight,deformation_regularization=args.deform_reg),indent=2))
