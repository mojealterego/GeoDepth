"""Profile comparison on supplied DA3 photographs; consistency, not ground truth."""
from pathlib import Path
import json, time, math
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from omega_geodepth.reconstruction5d import build_from_da3_sequence,build_scene5d,save_scene5d
from omega_geodepth.da3_profiles import PINS

ROOT=Path(__file__).resolve().parents[1]

def canonical(points,pose):
    return (points-pose[:3,3])@pose[:3,:3]

def run():
    report={'method':'two supplied photographs; nominal times only; shared median scale across both views; first camera gauge; no physical ground truth','process_res':280,'profiles':{}}
    baseline=None
    for profile in ('fp32','fp16','aq6','aq4'):
        path=ROOT/'models/da3' if profile=='fp32' else ROOT/'models/da3_profiles'/profile
        start=time.perf_counter()
        frames,meta=build_from_da3_sequence(ROOT/'examples/v08_real_da3/manifest.json',path,'DA3-SMALL',source_dir=ROOT/'vendor/da3_src',runtime='native_network',process_res=280,profile=profile,allow_experimental=profile.startswith('aq'),verify_reprojection=True)
        elapsed=time.perf_counter()-start
        scene=build_scene5d(frames,voxel_size=.08,neighbor_radius=.12,max_points_per_frame=24000)
        output=ROOT/'examples/v09_profiles'/profile
        result=save_scene5d(scene,output,frames,mesh=True)
        result['input_metadata']=meta
        (output/'report.json').write_text(json.dumps(result,indent=2))
        current=np.stack([f.depth_z for f in frames]); intr=np.stack([f.intrinsics for f in frames]); poses=np.stack([f.pose_c2w for f in frames]); xyz=canonical(scene.xyz,poses[0]);
        if baseline is None: baseline=(current,intr,poses,xyz,cKDTree(xyz))
        ref,kref,pref,xref,tree=baseline
        valid=np.isfinite(ref)&np.isfinite(current)&(ref>0)&(current>0)
        scale=float(np.median(ref[valid]/current[valid])); aligned=xyz*scale
        depth_mae=float(np.mean(np.abs(current[valid]*scale-ref[valid])/ref[valid]))
        dist_a=tree.query(aligned)[0]; dist_b=cKDTree(aligned).query(xref)[0]
        # normalize distances by median reference Z, not metres
        norm=float(np.median(ref[valid]))
        rot_ref=pref[0,:3,:3].T@pref[1,:3,:3]; rot=poses[0,:3,:3].T@poses[1,:3,:3]
        angle=float(np.degrees((Rotation.from_matrix(rot_ref).inv()*Rotation.from_matrix(rot)).magnitude()))
        t_ref=canonical(pref[1,:3,3],pref[0]); t=canonical(poses[1,:3,3],poses[0])*scale
        report['profiles'][profile]={'profile':meta['weight_profile'],'seconds_load_and_infer':elapsed,'point_count':len(xyz),'common_scale_to_fp32':scale,'relative_depth_mae_percent':depth_mae*100,'max_k_delta_px':float(np.max(np.abs(intr-kref))),'relative_rotation_delta_deg':angle,'relative_translation_error_normalized':float(np.linalg.norm(t-t_ref)/norm),'symmetric_chamfer_normalized':float((dist_a.mean()+dist_b.mean())/(2*norm)),'reprojection_passed':meta['multiview_reprojection_audit']['passed']}
        print(profile,report['profiles'][profile],flush=True)
    # Engineering compatibility gate (not an assertion of metric accuracy).
    fp=report['profiles']['fp16']
    report['fp16_regression_gate']={'passed':bool(fp['relative_depth_mae_percent']<.5 and fp['max_k_delta_px']<.5 and fp['symmetric_chamfer_normalized']<.02 and fp['reprojection_passed']), 'limits':{'depth_mae_percent':.5,'k_delta_px':.5,'chamfer_normalized':.02}}
    (ROOT/'reports/v09_profile_geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['fp16_regression_gate']['passed']: raise RuntimeError('FP16 geometry regression gate failed')
if __name__=='__main__': run()
