"""Reproducible view/time-blocked conditional RGB-field fitting, no geometry hallucination."""
from pathlib import Path
import sys, argparse, json, hashlib
from dataclasses import replace
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from omega_geodepth.reconstruction5d import load_frame_manifest, build_from_da3_sequence
from omega_geodepth.scene_field import collect_samples, fit_scene_field, save_scene_field, load_scene_field, align_baseline


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',choices=['npz','da3'],required=True)
    p.add_argument('--model-dir',type=Path);p.add_argument('--da3-source',type=Path)
    p.add_argument('--process-res',type=int,default=280);p.add_argument('--profile',choices=['fp32','fp16'],default='fp32')
    p.add_argument('--time-basis',choices=['static_assumption','capture_seconds'],required=True)
    p.add_argument('--holdout',type=int,nargs='+',required=True);p.add_argument('--split',choices=['view','time'],default='view')
    p.add_argument('--reference-radius',type=float,required=True);p.add_argument('--stride',type=int,default=4)
    p.add_argument('--steps',type=int,default=300);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--baseline-evidence',type=Path,help='JSON with evidence_id and baselines [[i,j,meters],...] for externally calibrated DA3')
    a=p.parse_args();torch.set_num_threads(1)
    if a.source=='da3':
        if a.model_dir is None or a.da3_source is None:p.error('DA3 needs both model and source directories')
        frames,meta=build_from_da3_sequence(a.manifest,a.model_dir,'DA3-SMALL',source_dir=a.da3_source,runtime='native_network',process_res=a.process_res,profile=a.profile,verify_reprojection=True)
    else: frames,meta=load_frame_manifest(a.manifest)
    if a.baseline_evidence:
        if meta['units']!='arbitrary_scale':p.error('Baseline scaling accepts relative input only')
        if a.baseline_evidence.stat().st_size>65536:p.error('Baseline evidence too large')
        b=json.loads(a.baseline_evidence.read_text());frames,alignment=align_baseline(frames,b['baselines'],evidence_id=b['evidence_id']);meta.update(alignment,scale_alignment=alignment)
    if a.time_basis=='static_assumption':frames=[replace(f,time_s=0.) for f in frames]
    samples=collect_samples(frames,units=meta['units'],provenance=meta['provenance'],reference_radius=a.reference_radius,time_basis=a.time_basis,stride=a.stride)
    model,report=fit_scene_field(samples,holdout_views=a.holdout,steps=a.steps,seed=a.seed,split=a.split)
    mask=np.isin(samples.view_index,a.holdout)
    mean=samples.rgb[~mask].mean(0)
    from scipy.spatial import cKDTree
    x=model.normalized(samples.coordinates)
    _,idx=cKDTree(x[~mask]).query(x[mask],k=1)
    pred=model.predict(samples.coordinates[mask]);actual=samples.rgb[mask]
    report.update(geometry_units=samples.units,geometry_provenance=samples.provenance,time_basis=samples.time_basis,reference_radius=samples.reference_radius,source=a.source,source_metadata=meta,mean_color_baseline_mse=float(np.mean((mean-actual)**2)),nearest_observation_baseline_mse=float(np.mean((samples.rgb[~mask][idx]-actual)**2)),test_rgb_mae=float(np.mean(np.abs(pred-actual))),manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest())
    a.out.mkdir(parents=True,exist_ok=False)
    checkpoint=a.out/'scene.safetensors';save_scene_field(checkpoint,model,samples,report)
    loaded,_=load_scene_field(checkpoint);np.testing.assert_allclose(loaded.predict(samples.coordinates[mask]),pred,atol=1e-6)
    report['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report['checkpoint_roundtrip']=True
    for index in a.holdout:
        f=frames[index]; h,w=f.depth_z.shape
        good=f.valid[::a.stride,::a.stride]&np.isfinite(f.depth_z[::a.stride,::a.stride])&(f.depth_z[::a.stride,::a.stride]>0)
        sel=samples.view_index[mask]==index
        preview=np.zeros((*good.shape,3),dtype=np.uint8);preview[good]=np.round(pred[sel]*255).astype(np.uint8)
        original=f.rgb[::a.stride,::a.stride].copy();original[~good]=0
        img=Image.fromarray(np.concatenate([original,preview],axis=1));img.save(a.out/f'view_{index:03d}_target_prediction.png')
    (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
