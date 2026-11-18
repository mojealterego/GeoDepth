"""Paired audit of camera token and native ray decoders on the v14 scene family."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from benchmark_reference_v14 import scene,cloud
from omega_geodepth.da3_real import DA3RealCPU
from omega_geodepth.visual_benchmark import evaluate_scene
from omega_geodepth.ray_consistency import reconcile_camera_gauge
from omega_geodepth.quality_v1.metrics import depth_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',nargs='+',type=int,default=[11,29,53]);p.add_argument('--models',nargs='+',choices=['DA3-SMALL','DA3-BASE'],default=['DA3-SMALL','DA3-BASE']);p.add_argument('--resolution',type=int,choices=[140,280],default=280);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
    report={'protocol':'v15_paired_native_camera_decoders','scope':'synthetic v14 analytic scene family; all 8 views are input; calibration cameras 0-3, evaluation 4-7; no true novel view or real-world metric claim','seeds':a.seeds,'resolution':a.resolution,'center_tolerance_m':.25,'rotation_tolerance_deg':5,'surface_threshold_m':.1,'rows':[]}
    for model in a.models:
        engine=DA3RealCPU(root/'models'/('da3_base' if model.endswith('BASE') else 'da3'),source_dir=root/'vendor/da3_src',model_id=model)
        for seed in a.seeds:
            images,z,poses,k=scene(seed,size=a.resolution);ks=np.repeat(k[None],8,0);cal=[0,1,2,3];ev=[4,5,6,7]
            for decoder in ['token','ray']:
                for conditioned in [False,True]:
                    row={'model':model,'seed':seed,'decoder':decoder,'conditioned':conditioned}
                    t=time.monotonic()
                    try:
                        pred=engine.inference(images,process_res=a.resolution,pose_decoder=decoder,audit_depth_rays=True,**({'camera_poses_c2w':poses,'camera_intrinsics':ks} if conditioned else {}))
                        row['inference_seconds']=time.monotonic()-t;row['rays']=pred.ray_audit
                        if pred.depth_z.shape!=z.shape:raise ValueError('unexpected pixel grid')
                        audit=evaluate_scene(pred.pose_c2w,poses,cloud(pred.depth_z,pred.pose_c2w,pred.intrinsics_k,ev),cloud(z,poses,ks,ev),calibration_indices=cal,evaluation_indices=ev,inlier_threshold=.25,distance_threshold=.1,reference_units='m')
                        row.update(status='aligned',audit=audit,depth=depth_metrics(pred.depth_z[ev]*audit['alignment']['scale'],z[ev]))
                        try:
                            alignment=reconcile_camera_gauge(pred,poses,anchor_indices=cal,validation_indices=ev,center_tolerance=.25,rotation_tolerance_deg=5)
                            row['strict_camera_gate']={'status':'accepted','scale':alignment['scale'],'max_center_error':alignment['max_center_error'],'max_rotation_error_deg':alignment['max_rotation_error_deg']}
                        except ValueError as exc:row['strict_camera_gate']={'status':'rejected','reason':str(exc)}
                    except (ValueError,RuntimeError) as exc:row.update(status='rejected',reason=str(exc))
                    report['rows'].append(row);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(model,seed,decoder,conditioned,row['status'],row.get('audit',{}).get('cloud',{}).get('f1'),row.get('strict_camera_gate',{}).get('status'),flush=True)
        del engine
if __name__=='__main__':main()
