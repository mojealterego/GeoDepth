"""Small reproducible benchmark: static photographed DA3 pair + known plane."""
from pathlib import Path
import json
import numpy as np
import torch
from omega_geodepth.da3_real import DA3RealCPU
from omega_geodepth.quality_v1.prediction import Prediction
from omega_geodepth.quality_v1.pipeline import run_pipeline
from omega_geodepth.quality_v1.refinement import refine_depth_multiview
from omega_geodepth.quality_v1.metrics import depth_metrics
from omega_geodepth.quality_v1.bridge import prepare_static_field


def main(root: Path, out: Path):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    engine=DA3RealCPU(root/'models/da3',source_dir=root/'vendor/da3_src')
    pred=engine.inference(sorted((root/'examples/v08_real_da3/images').glob('*.png')),process_res=280)
    p=Prediction(pred.depth_z,pred.confidence_raw,np.linalg.inv(pred.pose_c2w)[:,:3],pred.intrinsics_k)
    report=run_pipeline(p,out/'photo_geometry',source={'weights_sha256':pred.weights_sha256,'static_assumption':True},model_id=pred.model_id,min_view_coverage=.1,min_baseline_ratio=.005,min_mean_consistency=.8,min_support=1,relative_tolerance=.1,refine_depth=True,save_quality_maps=True,voxel_size=.025,max_points=100_000)
    samples,bridge=prepare_static_field(pred,assume_static=True,reference_radius=1.,refine=True,stride=4)
    d=np.stack([np.full((48,64),1.95),np.full((48,64),2.05)]).astype(np.float32)
    k=np.repeat(np.array([[[75.,0.,31.5],[0.,75.,23.5],[0.,0.,1.]]]),2,axis=0)
    e=np.repeat(np.eye(4)[None,:3],2,axis=0);e[1,0,3]=.1
    initial=Prediction(d,np.ones_like(d),e,k)
    after,stats=refine_depth_multiview(initial,blend=.5,relative_tolerance=.1)
    gt=np.full_like(d,2.)
    summary={'photo_pair':{'views':2,'processed_shape':list(pred.depth_z.shape),'gates':report['quality_gates'],'reprojection':report['multiview'],'point_cloud':report['geometry'],'refinement':report['refinement'],'bridge':bridge,'metric_ground_truth':False},'synthetic_plane':{'truth_z':2.,'input_z':[1.95,2.05],'pre':depth_metrics(d,gt),'post':depth_metrics(after.depth,gt),'refinement':stats,'scope':'two artificially biased depth maps; not an independent generalization benchmark'}}
    (out/'benchmark.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2,allow_nan=False))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();main(Path(__file__).resolve().parents[1],a.output)
