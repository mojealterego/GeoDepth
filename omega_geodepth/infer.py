"""Offline photo-to-depth inference, SHA-256-verified weights, provenance and PLY export."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file

from .geometry import CameraIntrinsics, checked_anchor_alignment, export_ply, scaled_intrinsics
from .tiling import tiled_predict
from .model import OmegaDepthNet
from .synthetic import generate_scene

MAX_SOURCE_PIXELS=24_000_000  # no fictional 400MP out-of-core support


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def load_verified_model(model_dir:Path,device:str='cpu')->tuple[OmegaDepthNet,dict]:
    manifest=json.loads((model_dir/'manifest.json').read_text(encoding='utf-8'))
    filename=manifest.get('weights_file','omega_geodepth_synthetic_v0.1.safetensors')
    model_path=model_dir/filename
    expected=manifest['weights_sha256'][filename]
    if sha256(model_path)!=expected:
        raise ValueError('MODEL_SHA256_MISMATCH: refusing weights')
    if manifest.get('provenance') not in ('trained_from_scratch_synthetic_raycast_v0.1','trained_from_scratch_synthetic_raycast_v0.2'):
        raise ValueError('MODEL_PROVENANCE_INVALID')
    model=OmegaDepthNet(width=int(manifest['width'])).to(device)
    state=load_file(str(model_path),device=device)
    model.load_state_dict(state,strict=True)
    model.eval()
    return model,manifest


def save_atomic(path:Path,data:bytes)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    name=None
    try:
        with tempfile.NamedTemporaryFile(dir=str(path.parent),prefix='.omega-',delete=False) as f:
            name=f.name
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if name and os.path.exists(name):os.unlink(name)


def load_image(path:Path,max_side:int,*,tiled:bool=False)->tuple[np.ndarray,np.ndarray,tuple[int,int],CameraIntrinsics]:
    if not 64<=max_side<=(4096 if tiled else 1024) or max_side%8:
        raise ValueError('INVALID_MAX_SIDE')
    with Image.open(path) as im:
        # Reject oversized headers BEFORE allocating EXIF-transposed / decoded pixels.
        width,height=im.size
        if width*height>MAX_SOURCE_PIXELS:
            raise ValueError('INPUT_RESOURCE_LIMIT: source exceeds 24MP; out-of-core not yet implemented')
        # EXIF orientation is applied to pixels. No EXIF metadata is passed into ML.
        im=ImageOps.exif_transpose(im)
        width,height=im.size
        if width*height>MAX_SOURCE_PIXELS:
            raise ValueError('INPUT_RESOURCE_LIMIT: source exceeds 24MP; out-of-core not yet implemented')
        if min(width,height)<8:
            raise ValueError('INPUT_TOO_SMALL')
        original=im.convert('RGB')
        ratio=min(1.,max_side/max(width,height))
        w=max(8,round(width*ratio/8)*8)
        h=max(8,round(height*ratio/8)*8)
        resized=original.resize((w,h),Image.Resampling.BICUBIC)
        rgb=np.array(resized,dtype=np.uint8)
    # Assume default pinhole K for visualization only. This is NOT camera calibration.
    intrinsics=CameraIntrinsics(1.2*width,1.2*height,(width-1)/2,(height-1)/2)
    k=scaled_intrinsics(intrinsics,0,0,w/width,h/height)
    x=np.ascontiguousarray(rgb.astype(np.float32)/255)
    return x,rgb,(width,height),k


def infer_image(image_path:Path,out_dir:Path,model_dir:Path,max_side:int=256,
                anchors_path:Path|None=None,ply:bool=True,device:str='cpu',*,
                tiled:bool=False,tile_size:int=256,overlap:int=64,global_side:int=256)->dict[str,Any]:
    model,manifest=load_verified_model(model_dir,device)
    rgb,rgb_u8,source_dims,k=load_image(image_path,max_side,tiled=tiled)
    if tiled:
        logd_np,logvar_np=tiled_predict(model,rgb,tile_size=tile_size,overlap=overlap,
                                        global_side=global_side,device=device)
    else:
        inp=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1)[None])).to(device)
        with torch.inference_mode(): logd,logvar=model(inp)
        logd_np=logd.squeeze().cpu().numpy().astype(np.float32)
        logvar_np=logvar.squeeze().cpu().numpy().astype(np.float32)
    depth=np.exp(np.clip(logd_np,-15,15))
    sigma_log=np.sqrt(np.exp(np.clip(logvar_np,-14,7)))
    # Calibration is optional and pinned to the exact model checkpoint.
    from .calibration import load_verified_calibration
    cal=load_verified_calibration(model_dir,manifest)
    # Conformal fit is from DIRECT 64x64 synthetic data; other resolutions and
    # tiled inference are distribution shifts and must not inherit its coverage.
    if tiled or rgb.shape[:2]!=(64,64):
        cal=None
    units='arbitrary_scale';provenance='relative_estimated';scale=None;anchor_diagnostics=None
    if anchors_path is not None:
        if anchors_path.stat().st_size>64_000_000:
            raise ValueError('ANCHOR_RESOURCE_LIMIT')
        with np.load(anchors_path,allow_pickle=False) as a:
            if 'depth_m' not in a.files or 'valid_mask' not in a.files:
                raise ValueError('ANCHOR_FORMAT: expected depth_m and valid_mask')
            alignment=checked_anchor_alignment(depth,a['depth_m'],a['valid_mask'])
        scale=alignment.scale
        anchor_diagnostics={'valid_points':alignment.valid_points,
            'covered_grid_cells':alignment.covered_grid_cells,
            'inlier_fraction':alignment.inlier_fraction,
            'p90_abs_log_residual':alignment.p90_abs_log_residual,
            'robust_log_scale_sigma':alignment.robust_log_scale_sigma,
            'sensor_provenance_verified':False}
        # Scale uncertainty is additional; synthetic conformal coverage no longer applies.
        sigma_log=np.sqrt(sigma_log**2+alignment.robust_log_scale_sigma**2)
        cal=None
        depth*=scale
        units='meters';provenance='scale_aligned'
    if not np.isfinite(depth).all() or (depth<=0).any():
        raise ValueError('INVALID_DEPTH_OUTPUT')
    out_dir.mkdir(parents=True,exist_ok=True)
    # Visualization deliberately separate from numeric science data.
    lower=float(np.percentile(depth,2)); upper=float(np.percentile(depth,98))
    visual=(np.clip((upper-depth)/max(upper-lower,1e-6),0,1)*255).astype(np.uint8)
    from io import BytesIO
    preview_buffer=BytesIO()
    Image.fromarray(visual,mode='L').save(preview_buffer,format='PNG')
    save_atomic(out_dir/'depth_preview.png',preview_buffer.getvalue())
    npz_buffer=BytesIO()
    # Numerically stable log-depth interval for the calibrated synthetic distribution.
    q=cal.get('q_scene_coverage',cal['q']) if cal is not None else None
    lower_depth=np.exp(np.clip(np.log(depth)-q*sigma_log,-20,20)) if q is not None else np.empty(0,dtype=np.float32)
    upper_depth=np.exp(np.clip(np.log(depth)+q*sigma_log,-20,20)) if q is not None else np.empty(0,dtype=np.float32)
    np.savez_compressed(npz_buffer,depth_z=depth.astype(np.float32),sigma_log_z=sigma_log,
                        depth_lower_calibrated_synthetic=lower_depth.astype(np.float32),
                        depth_upper_calibrated_synthetic=upper_depth.astype(np.float32),
                        valid_mask=np.isfinite(depth),intrinsics_k=k.as_matrix(),source_rgb=rgb_u8)
    save_atomic(out_dir/'depth_geometry.npz',npz_buffer.getvalue())
    pointcloud_status='disabled'
    if ply and depth.size<=1_100_000:
        ply_data=export_ply(depth,rgb_u8,k,units=units,provenance=provenance,
                            metric_anchor_validated=anchors_path is not None,max_points=1_100_000)
        save_atomic(out_dir/'pointcloud.ply',ply_data)
        pointcloud_status='exported'
    elif ply:
        pointcloud_status='skipped: point limit 1100000'
    # Avoid retaining an obsolete PLY from a previous run in the same output directory.
    if pointcloud_status!='exported':
        (out_dir/'pointcloud.ply').unlink(missing_ok=True)
    report={
        'model':manifest['model'],'model_provenance':manifest['provenance'],
        'input_sha256':sha256(image_path),
        'image_original_size':[source_dims[0],source_dims[1]],
        'inference_size':[rgb.shape[1],rgb.shape[0]],
        'geometry_definition':'optical_z','units':units,'provenance':provenance,
        'metric_scale_factor':scale,'anchor_diagnostics':anchor_diagnostics,
        'processing_mode':'tiled_global_local' if tiled else 'direct',
        'processing_settings':({'tile_size':tile_size,'overlap':overlap,'global_side':global_side}
                               if tiled else {}),
        'pointcloud_status':pointcloud_status,
        'uncertainty':'sigma(log depth); synthetic-trained, calibration only for direct 64x64 synthetic scenes',
        'conformal_calibration':cal if cal is not None else 'not_available',
        'content_classification':'not_performed','nsfw_filter':'none',
        'depth_range':[float(depth.min()),float(depth.max())],
        'camera_intrinsics_status':'assumed_for_viz_only','k':k.as_matrix().tolist(),
        'model_sha256':manifest['weights_sha256'][manifest.get('weights_file','omega_geodepth_synthetic_v0.1.safetensors')],
        'output_sha256':{n:sha256(out_dir/n) for n in
                         (['depth_preview.png','depth_geometry.npz']+
                         (['pointcloud.ply'] if pointcloud_status=='exported' else []))},
        'limitations':manifest['limits'],
    }
    save_atomic(out_dir/'result.json',json.dumps(report,indent=2,ensure_ascii=False).encode('utf-8'))
    return report


def main()->None:
    p=argparse.ArgumentParser(prog='omega_geodepth',description='OCAF GeoDepth Ω: offline relative depth estimator, any content')
    sub=p.add_subparsers(dest='command',required=True)
    i=sub.add_parser('infer',help='Run content-agnostic depth inference on a local image')
    i.add_argument('--image',type=Path,required=True)
    i.add_argument('--output',type=Path,required=True)
    i.add_argument('--model-dir',type=Path,default=Path('models'))
    i.add_argument('--max-side',type=int,default=256)
    i.add_argument('--anchors',type=Path,help='npz with aligned depth_m and valid_mask')
    i.add_argument('--no-ply',action='store_true')
    i.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    i.add_argument('--tiled',action='store_true',help='Use overlapping global/local tiles')
    i.add_argument('--tile-size',type=int,default=256)
    i.add_argument('--overlap',type=int,default=64)
    i.add_argument('--global-side',type=int,default=256)
    d=sub.add_parser('make-demo',help='Create deterministic raycast scene and its true depth')
    d.add_argument('--output',type=Path,required=True)
    d.add_argument('--seed',type=int,default=2026)
    d.add_argument('--size',type=int,default=128)
    v=sub.add_parser('verify',help='Verify weights checksum and run a real forward pass')
    v.add_argument('--model-dir',type=Path,default=Path('models'))
    args=p.parse_args()
    if args.command=='infer':
        print(json.dumps(infer_image(args.image,args.output,args.model_dir,args.max_side,args.anchors,not args.no_ply,args.device,
                                         tiled=args.tiled,tile_size=args.tile_size,overlap=args.overlap,global_side=args.global_side),ensure_ascii=False,indent=2))
    elif args.command=='make-demo':
        rgb,depth=generate_scene(args.seed,(args.size,args.size))
        args.output.mkdir(parents=True,exist_ok=True)
        save_atomic(args.output/'synthetic_scene.png',_png(rgb))
        from io import BytesIO
        b=BytesIO();np.savez_compressed(b,depth_z_true_synthetic=depth)
        save_atomic(args.output/'synthetic_ground_truth.npz',b.getvalue())
        print(f'created {args.output}')
    elif args.command=='verify':
        m,manifest=load_verified_model(args.model_dir)
        with torch.inference_mode():logd,logvar=m(torch.rand(1,3,64,64))
        assert torch.isfinite(logd).all() and torch.isfinite(logvar).all()
        print(json.dumps({'verified':True,'sha256':manifest['weights_sha256'],'output_shape':list(logd.shape)},indent=2))


def _png(rgb:np.ndarray)->bytes:
    from io import BytesIO
    b=BytesIO();Image.fromarray((np.clip(rgb,0,1)*255).astype(np.uint8)).save(b,format='PNG')
    return b.getvalue()

if __name__=='__main__': main()
