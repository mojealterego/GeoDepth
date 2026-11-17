"""Run pinned Base/Small at two resolutions; keep rejected results solely for audit."""
from pathlib import Path
import argparse,json,time,resource
import numpy as np
import torch
from omega_geodepth.da3_real import DA3RealCPU
from omega_geodepth.da3_selection import audit_bidirectional
from omega_geodepth.da3_validation import verify_multiview_reprojection
from omega_geodepth.quality_v1.prediction import Prediction
from omega_geodepth.quality_v1.pipeline import run_pipeline
from omega_geodepth.quality_v1.bridge import prepare_static_field

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['DA3-SMALL','DA3-BASE'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    t=time.monotonic();engine=DA3RealCPU(root/'models'/('da3_base' if a.model=='DA3-BASE' else 'da3'),source_dir=root/'vendor/da3_src',model_id=a.model)
    start=time.monotonic()-t
    images=sorted((root/'examples/v08_real_da3/images').glob('*.png'))
    report={'model':a.model,'checkpoint':engine.digest,'parameters':sum(x.numel() for x in engine.net.parameters()),'load_seconds':start,'device':'cpu','threads':1,'units':'arbitrary_scale','scope':'same two photographs; not independent metric truth','runs':[]}
    for resolution in (140,280,392,504):
        t=time.monotonic();raw=engine.inference(images,process_res=resolution);elapsed=time.monotonic()-t
        audit=audit_bidirectional(raw); legacy=verify_multiview_reprojection(raw)
        r={'resolution':resolution,'shape':list(raw.depth_z.shape),'seconds':elapsed,'audit':audit,'legacy':legacy,'rss_peak_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'geometry_export':'rejected'}
        if audit['passed']:
            scene=Prediction(raw.depth_z,raw.confidence_raw,np.linalg.inv(raw.pose_c2w)[:,:3],raw.intrinsics_k)
            try:
                result=run_pipeline(scene,a.output/f'geometry_{resolution}',source={'weights_sha256':raw.weights_sha256,'static_assumption':True},model_id=raw.model_id,min_view_coverage=.1,min_baseline_ratio=.005,min_mean_consistency=.8,min_support=1,relative_tolerance=.1,refine_depth=True,save_quality_maps=True,voxel_size=.025,max_points=100_000)
                samples,bridge=prepare_static_field(raw,assume_static=True,reference_radius=1.,refine=True,stride=4)
                r.update(geometry_export='passed',geometry=result['geometry'],bridge=bridge)
            except ValueError as e:r['geometry_error']=str(e)
        report['runs'].append(r)
        (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'model':a.model,'res':resolution,'legacy':legacy['passed'],'bidirectional':audit['passed'],'worst_agreement':audit['worst_agreement'],'coverage':audit['worst_coverage'],'export':r['geometry_export'],'sec':elapsed}),flush=True)
if __name__=='__main__':main()
