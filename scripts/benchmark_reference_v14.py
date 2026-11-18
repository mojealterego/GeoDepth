"""Analytic multicamera audit; all images are input to DA3, not NVS."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from omega_geodepth.da3_real import DA3RealCPU
from omega_geodepth.visual_benchmark import evaluate_scene,pose_metrics
from omega_geodepth.quality_v1.metrics import depth_metrics

def scene(seed,size=280):
    rng=np.random.default_rng(seed)
    centers=np.array([[-.65,-.28,0],[.65,-.28,.03],[-.60,.35,.08],[.58,.38,-.05],[-.32,.06,-.1],[.32,.13,.1],[.05,-.47,.06],[.03,.44,-.06]])
    k=np.array([[size*.85,0,(size-1)/2],[0,size*.85,(size-1)/2],[0,0,1.]])
    yy,xx=np.indices((size,size));ray=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(k).T
    spheres=[(np.array([-.65,.05,3.4]),.67,np.array([.75,.27,.12])),(np.array([.72,-.14,4.1]),.85,np.array([.13,.56,.79])),(np.array([.15,.65,3.7]),.43,np.array([.28,.8,.24]))]
    spheres=[(c+rng.normal(0,.13,3),r*rng.uniform(.85,1.15),color) for c,r,color in spheres]
    poses=[];images=[];depths=[]
    for c in centers:
        f=np.array([0,0,4.5])-c;f/=np.linalg.norm(f);right=np.cross([0,1,0],f);right/=np.linalg.norm(right);down=np.cross(f,right)
        rot=np.stack([right,down,f],axis=1);pose=np.eye(4);pose[:3,:3]=rot;pose[:3,3]=c;poses.append(pose)
        d=ray@rot.T;z=(6.-c[2])/d[...,2];p=c+z[...,None]*d
        pattern=(np.sin(7*p[...,0]+seed)*np.sin(7*p[...,1])+1)/2;grid=(np.cos(19*p[...,0])+np.cos(21*p[...,1]))*.04
        rgb=(.36+.24*pattern+grid)[...,None]*np.array([.93,1.,.84])
        for center,r,color in spheres:
            v=c-center;a=np.sum(d*d,axis=-1);b=np.sum(d*v,axis=-1);disc=b*b-a*(v@v-r*r);t=(-b-np.sqrt(np.maximum(disc,0)))/a
            mask=(disc>=0)&(t>0)&(t<z);normal=(c+t[...,None]*d-center)/r
            lambert=np.clip(normal@np.array([-.35,-.42,-.837]),0,1);texture=.86+.14*np.sin(22*normal[...,0])*np.cos(17*normal[...,1]);shade=(.22+.73*lambert)*texture
            rgb=np.where(mask[...,None],shade[...,None]*color,rgb);z=np.where(mask,t,z)
        images.append(np.round(np.clip(rgb,0,1)*255).astype(np.uint8));depths.append(z)
    return images,np.array(depths),np.array(poses),k

def cloud(depth,poses,ks,idx):
    h,w=depth.shape[1:];yy,xx=np.mgrid[0:h:4,0:w:4];pix=np.stack([xx,yy,np.ones_like(xx)],-1);clouds=[]
    for i in idx:
        z=depth[i,yy,xx];mask=np.isfinite(z)&(z>0);p=(pix@np.linalg.inv(ks[i]).T)*z[...,None];p=p@poses[i,:3,:3].T+poses[i,:3,3];clouds.append(p[mask])
    return np.concatenate(clouds)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--conditioned',action='store_true');args=p.parse_args();root=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
    result={'protocol':'analytic_reference_v14','scope':'3 procedural variants of the same scene family, 8 views each, all input to DA3. Calibration indices 0..3 only; geometry/pose audit 4..7. Not real-world data or NVS.','inlier_threshold_m':.25,'f1_threshold_m':.10,'seeds':[11,29,53],'models':{},'pose_conditioned':args.conditioned,'camera_metadata_available_for_all_views':args.conditioned}
    for model in ['DA3-SMALL','DA3-BASE']:
        engine=DA3RealCPU(root/'models'/('da3_base' if model.endswith('BASE') else 'da3'),source_dir=root/'vendor/da3_src',model_id=model)
        rows=[];result['models'][model]=rows
        for seed in result['seeds']:
            images,z,poses,k=scene(seed);start=time.monotonic();pred=engine.inference(images,process_res=280,**({'camera_poses_c2w':poses,'camera_intrinsics':np.repeat(k[None],8,axis=0)} if args.conditioned else {}));elapsed=time.monotonic()-start
            idx=[4,5,6,7];cal=[0,1,2,3];row={'seed':seed,'seconds':elapsed,'input_shape':list(images[0].shape),'predicted_shape':list(pred.depth_z.shape),'pose_without_alignment':pose_metrics(pred.pose_c2w[idx],poses[idx])}
            if pred.depth_z.shape!=z.shape:raise RuntimeError('benchmark requires identical pixel grids')
            try:
                report=evaluate_scene(pred.pose_c2w,poses,cloud(pred.depth_z,pred.pose_c2w,pred.intrinsics_k,idx),cloud(z,poses,np.repeat(k[None],8,axis=0),idx),calibration_indices=cal,evaluation_indices=idx,inlier_threshold=.25,distance_threshold=.1,reference_units='m')
                row.update(status='aligned',audit=report,depth=depth_metrics(pred.depth_z[idx]*report['alignment']['scale'],z[idx]))
            except ValueError as exc:row.update(status='rejected_alignment',reason=str(exc))
            rows.append(row);print(model,seed,row['status'],row.get('audit',{}).get('cloud',{}).get('f1'),flush=True)
            args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        del engine
if __name__=='__main__':main()
