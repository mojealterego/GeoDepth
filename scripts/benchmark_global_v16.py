"""Sequential CPU test: native/fast camera fitting and overlapping DA3 chunk graph."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from benchmark_reference_v14 import scene,cloud
from omega_geodepth.da3_real import DA3RealCPU
from omega_geodepth.chunk_graph import align_da3_chunks
from omega_geodepth.visual_benchmark import evaluate_scene
from omega_geodepth.infer import save_atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--models',nargs='+',choices=['DA3-SMALL','DA3-BASE'],default=['DA3-SMALL','DA3-BASE']);p.add_argument('--resolution',type=int,choices=[140,280],default=140);p.add_argument('--repeats',type=int,choices=[1,2,3],default=2);a=p.parse_args();torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1];images,z,poses,k=scene(11,size=a.resolution);ks=np.repeat(k[None],8,0)
    report={'scope':'Single analytic seed 11 (previous v15 family), 8 context views; reference cameras 0..3 alignment, 4..7 evaluation. Timings sequential CPU wall clock incl preprocessing, native/fast decoder and ray audit, not decoder-only. Warm-up excluded. Not external or real-world metrics.','resolution':a.resolution,'repeats':a.repeats,'models':{}}
    def save():a.output.parent.mkdir(parents=True,exist_ok=True);save_atomic(a.output,(json.dumps(report,indent=2,allow_nan=False)+'\n').encode())
    for model in a.models:
        engine=DA3RealCPU(root/'models'/('da3_base' if model.endswith('BASE') else 'da3'),source_dir=root/'vendor/da3_src',model_id=model)
        engine.inference(images[:2],process_res=a.resolution)
        rows=[];section={'decoder_comparison':rows};report['models'][model]=section
        for conditioned in (False,True):
            for decoder in ('ray','fast_ray'):
                row={'conditioned':conditioned,'decoder':decoder,'seconds':[]};rows.append(row)
                for i in range(a.repeats):
                    start=time.monotonic()
                    try:r=engine.inference(images,process_res=a.resolution,pose_decoder=decoder,audit_depth_rays=True,**({'camera_poses_c2w':poses,'camera_intrinsics':ks} if conditioned else {}))
                    except (ValueError,RuntimeError) as exc:row.update(status='rejected',reason=str(exc));break
                    row['seconds'].append(time.monotonic()-start)
                    row['rays']=r.ray_audit;row['status']='inferred'
                    if i==0:
                        try:row['reference']=evaluate_scene(r.pose_c2w,poses,cloud(r.depth_z,r.pose_c2w,r.intrinsics_k,[4,5,6,7]),cloud(z,poses,ks,[4,5,6,7]),calibration_indices=[0,1,2,3],evaluation_indices=[4,5,6,7],inlier_threshold=.25,distance_threshold=.1,reference_units='m')
                        except ValueError as exc:row['reference_rejected']=str(exc)
                if row['seconds']:row['median_seconds']=float(np.median(row['seconds']))
                print(model,conditioned,decoder,row['status'],row.get('median_seconds'),row.get('reference',{}).get('cloud',{}).get('f1'),flush=True);save()
        groups=[(0,1,2,3,4,5),(2,3,4,5,6,7),(0,1,4,5,6,7)]
        try:
            chunks=[(ids,engine.inference([images[i] for i in ids],process_res=a.resolution,pose_decoder='fast_ray')) for ids in groups]
            g,merged,audit=align_da3_chunks(chunks,center_tolerance=.15,rotation_tolerance_deg=5,depth_log_p90_limit=.20,intrinsics_relative_limit=.05,min_depth_overlap=.70)
            section['chunk_graph']={'status':'accepted' if merged is not None else 'rejected','groups':groups,'thresholds':{'center_first_chunk_units':.15,'rotation_deg':5,'depth_log_p90':.2,'intrinsics_relative':.05,'min_depth_overlap':.70},'scales':g.scales.tolist(),'audit':audit}
        except (ValueError,RuntimeError) as exc:section['chunk_graph']={'status':'rejected','reason':str(exc),'groups':groups}
        print(model,'chunk_graph',section['chunk_graph']['status'],flush=True);save();del engine
if __name__=='__main__':main()
