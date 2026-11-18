"""Genuine DA3 Small / optional Base neural inference using the upstream network without export extras.

Loads only a SHA-256 pinned, user-provided checkpoint. Heavy optional upstream
exports (Open3D, gsplat, moviepy, COLMAP) are not imported. This is *not* a
reimplementation of DA3's network; the original network code is executed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
from dataclasses import replace
from contextlib import nullcontext
import sys
import hashlib
import json
import zipfile
import threading
import numpy as np

from .da3_adapter import (
    DA3Output,
    OFFICIAL_WEIGHT_SHA256,
    validate_da3_checkpoint,
    normalize_da3_prediction,
    validate_official_config,
)

def verify_da3_runtime_source(source_dir: Path, archive: Path) -> None:
    """Validate unchanged native DA3 source against the pinned original ZIP.

    Checks actual bytes of all importable source files, not an editable local
    manifest. Does not accept symlinks or newly added Python executables.
    """
    from .da3_source import inspect_da3_source_archive
    source_dir = Path(source_dir)
    archive = Path(archive)
    if source_dir.is_symlink() or archive.is_symlink():
        raise ValueError('DA3_SOURCE_UNSAFE_PATH')
    inspect_da3_source_archive(archive)
    verified=set()
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            prefix='Depth-Anything-3-main/src/'
            if not member.startswith(prefix) or not member.endswith(('.py','.yaml')):
                continue
            rel=member.removeprefix('Depth-Anything-3-main/')
            target=source_dir/rel
            if any(p.is_symlink() for p in (target, *target.parents) if p!=source_dir.parent):
                raise ValueError('DA3_SOURCE_UNSAFE_PATH')
            if not target.is_file():
                raise ValueError('DA3_SOURCE_MISSING')
            if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(z.read(member)).digest():
                raise ValueError('DA3_SOURCE_TAMPERED')
            verified.add(rel)
    for path in (source_dir/'src'/'depth_anything_3').rglob('*'):
        if path.is_symlink():
            raise ValueError('DA3_SOURCE_UNSAFE_PATH')
        if path.is_file() and path.suffix.lower() in ('.py','.yaml','.pth','.so','.pyd','.dll'):
            if path.relative_to(source_dir).as_posix() not in verified:
                raise ValueError('DA3_SOURCE_UNDECLARED_PYTHON')
    if len(verified)<30:
        raise ValueError('DA3_SOURCE_INCOMPLETE')

def validate_camera_conditions(camera_poses_c2w, camera_intrinsics, count):
    """Validate original-pixel camera K and C2W, returning upstream W2C and K.

    Conditions are hints to DA3, not independent depth measurements.
    """
    if camera_poses_c2w is None and camera_intrinsics is None:
        return None, None
    if camera_poses_c2w is None or camera_intrinsics is None:
        raise ValueError('DA3_CAMERA_CONDITIONS_REQUIRE_POSES_AND_INTRINSICS')
    from .geometry import validate_pose_c2w
    poses=np.asarray(camera_poses_c2w,dtype=np.float64)
    k=np.asarray(camera_intrinsics,dtype=np.float64)
    if poses.shape!=(count,4,4) or k.shape!=(count,3,3):
        raise ValueError('DA3_CAMERA_CONDITION_SHAPE')
    if not np.isfinite(k).all() or np.any(k[:,0,0]<=0) or np.any(k[:,1,1]<=0) or not np.allclose(k[:,2,:],[0,0,1],atol=1e-7,rtol=0) or not np.allclose(k[:,1,0],0,atol=1e-7,rtol=0) or not np.allclose(k[:,0,1],0,atol=1e-7,rtol=0):
        raise ValueError('DA3_CAMERA_CONDITION_INVALID_K')
    for pose in poses:validate_pose_c2w(pose)
    if np.max(np.abs(poses))>1e8 or np.max(np.abs(k))>1e8:
        raise ValueError('DA3_CAMERA_CONDITION_RANGE')
    return np.linalg.inv(poses),k.copy()


def normalize_camera_conditions(w2c):
    """Match upstream API first-camera gauge and median baseline normalization."""
    import torch
    norm=w2c@torch.linalg.inv(w2c[:,:1])
    cameras=torch.linalg.inv(norm)
    median_dist=torch.median(cameras[...,:3,3].norm(dim=-1)).clamp(min=.1)
    norm=norm.clone();norm[...,:3,3]/=median_dist
    return norm


class DA3RealCPU:
    """Run the pinned official DA3 network (Small or Base), CPU or CUDA."""
    def __init__(self, model_dir: Path, *, source_dir: Path, device: str = 'cpu',
                 profile: str = 'fp32', allow_experimental: bool = False,
                 model_id: str = 'DA3-SMALL'):
        if model_id not in OFFICIAL_WEIGHT_SHA256:
            raise ValueError('DA3_MODEL_NOT_ALLOWED')
        if model_id=='DA3-BASE' and profile!='fp32':
            raise ValueError('DA3-BASE requires FP32; derived profiles are Small-only')
        self.base_model_id=model_id
        import torch
        from safetensors import safe_open
        if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('DA3_INVALID_DEVICE')
        if Path(model_dir).is_symlink() or Path(source_dir).is_symlink():
            raise ValueError('DA3_UNSAFE_MODEL_OR_SOURCE_PATH')
        model_dir = Path(model_dir).resolve(strict=True)
        source_dir = Path(source_dir).resolve(strict=True)
        model_file = model_dir/'model.safetensors'
        config = validate_official_config(model_dir/'config.json',model_id)
        if profile=='int8' and not allow_experimental:
            raise ValueError('DA3_EXPERIMENTAL_OPT_IN_REQUIRED')
        if profile=='int8' and device!='cpu':
            raise ValueError('INT8_REQUIRES_CPU_FP32')
        if profile in ('fp32','int8'):
            validate_da3_checkpoint(model_dir,model_id)
            digest=OFFICIAL_WEIGHT_SHA256[model_id]
            config_pin={'DA3-SMALL':'a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2',
                        'DA3-BASE':'5e34115ebc17bd2d8d43033c5f72e9446ac8833fd61d3fa160b7e67e0bb5b7b5'}[model_id]
            if hashlib.sha256((model_dir/'config.json').read_bytes()).hexdigest() != config_pin:
                raise ValueError('DA3_CONFIG_PIN_MISMATCH')
            weight_context=nullcontext((model_file,{'model_id':model_id,'profile':'fp32',
                'artifact_sha256':digest,'runtime_weights_sha256':digest,'parent_sha256':digest,
                'experimental':False,'execution_precision':'float32','units':'arbitrary_scale'}))
        else:
            from .da3_profiles import runtime_weights
            weight_context=runtime_weights(model_dir,profile,allow_experimental=allow_experimental)
        verify_da3_runtime_source(source_dir,source_dir.parent/'Depth-Anything-3-main.zip')
        if not (source_dir/'LICENSE').is_file():
            raise ValueError('DA3_SOURCE_MISSING')
        # Guard against Python modules imported earlier from a different DA3 tree.
        # Otherwise sys.modules could silently override the byte-pinned sources.
        source_root=(source_dir/'src').resolve()
        for name,module in tuple(sys.modules.items()):
            if name=='depth_anything_3' or name.startswith('depth_anything_3.'):
                filename=getattr(module,'__file__',None)
                paths=([Path(filename)] if filename else
                       [Path(x) for x in getattr(module,'__path__',())])
                if not paths or any(not x.resolve().is_relative_to(source_root) for x in paths):
                    raise RuntimeError('DA3_SOURCE_IMPORT_COLLISION')
        # Locate the runtime compat library next to the verified source archive,
        # not next to the installed wheel (which carries no model or vendor).
        compat_dir=source_dir.parent/'da3_runtime_compat'
        compat_file=compat_dir/'addict.py'
        if compat_file.is_symlink() or not compat_file.is_file():
            raise RuntimeError('DA3_COMPAT_NOT_AVAILABLE')
        if hashlib.sha256(compat_file.read_bytes()).hexdigest() != '4e53bc2fd6703c9c30ce2bacaa3f2c41e205b715c17bd930591e52796b4b62a0':
            raise RuntimeError('DA3_COMPAT_HASH_MISMATCH')
        addon=sys.modules.get('addict')
        if addon is not None and Path(getattr(addon,'__file__','/') or '/').resolve()!=compat_file.resolve():
            raise RuntimeError('DA3_COMPAT_IMPORT_COLLISION')
        compat=str(compat_dir)
        vendor=str(source_root)
        for item in (vendor,compat):
            if item not in sys.path: sys.path.insert(0,item)
        from depth_anything_3.cfg import create_object
        from depth_anything_3.utils.io.input_processor import InputProcessor
        from depth_anything_3.utils.io.output_processor import OutputProcessor
        from omegaconf import OmegaConf
        with weight_context as (runtime_file,profile_info):
            self.net = create_object(OmegaConf.create(config['config']))
            weights={}
            with safe_open(runtime_file,framework='pt',device='cpu') as f:
                md=f.metadata() or {}
                for k in f.keys():
                    weights[k.removeprefix('model.')]=f.get_tensor(k)
                for alias,referent in md.items():
                    if not alias.startswith('model.') or not referent.startswith('model.'):
                        raise ValueError('DA3_INVALID_WEIGHT_ALIAS')
                    target=referent.removeprefix('model.')
                    if target not in weights:
                        raise ValueError('DA3_INVALID_WEIGHT_ALIAS')
                    weights[alias.removeprefix('model.')]=weights[target]
            self.net.load_state_dict(weights,strict=True)
            del weights
        self.profile_info=profile_info
        self.net.eval().to(device)
        if profile=='int8':
            from .int8_runtime import quantize_backbone
            stats=quantize_backbone(self.net)
            profile_info.update(stats, model_id='DA3-SMALL-INT8-EXPERIMENTAL',
                                profile='int8',experimental=True,runtime_weights_sha256=None,
                                provenance_verification='runtime_quantization_from_pinned_fp32')
        self.processor=InputProcessor()
        self.output_processor=OutputProcessor()
        self.device=device
        self._forward_lock=threading.Lock()
        self.digest=profile_info['artifact_sha256']
        self.model_id=profile_info['model_id']

    def inference(self, images: Sequence[np.ndarray], *, process_res:int=140,
                  camera_poses_c2w:np.ndarray|None=None, camera_intrinsics:np.ndarray|None=None,
                  pose_decoder:str="token", audit_depth_rays:bool=False) -> DA3Output:
        import torch
        if pose_decoder not in ("token","ray") or not isinstance(audit_depth_rays,bool):
            raise ValueError("DA3_INVALID_POSE_DECODER")
        if not 1<=len(images)<=8 or not 112<=process_res<=504:
            raise ValueError('DA3_RESOURCE_LIMIT')
        exts, intr=validate_camera_conditions(camera_poses_c2w,camera_intrinsics,len(images))
        validated=[]
        from PIL import Image
        for image in images:
            if isinstance(image,(Path,str)):
                with Image.open(image) as img:
                    if img.width*img.height > 4_000_000:
                        raise ValueError('DA3_INPUT_TOO_LARGE')
                    image=np.asarray(img.convert('RGB'))
            if not isinstance(image,np.ndarray) or image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
                raise ValueError('DA3_INVALID_INPUT_IMAGE')
            if image.shape[0]*image.shape[1]>4_000_000:
                raise ValueError('DA3_INPUT_TOO_LARGE')
            validated.append(image)
        with torch.inference_mode():
            tensors,ext_tensor,intr_tensor=self.processor(validated,extrinsics=exts,intrinsics=intr,process_res=process_res,num_workers=1)
            tensors=tensors.to(self.device)[None].float()
            ext_tensor=normalize_camera_conditions(ext_tensor.to(self.device)[None].float()) if ext_tensor is not None else None
            intr_tensor=intr_tensor.to(self.device)[None].float() if intr_tensor is not None else None
            with self._forward_lock:
                captured={}
                def capture(_module,_inputs,raw):
                    # Copy before the native decoder removes rays/mutates confidence.
                    captured['ray']=raw['ray'][0].detach().cpu().clone().numpy()
                    captured['conf']=raw['ray_conf'][0].detach().cpu().clone().numpy()
                hook=self.net.head.register_forward_hook(capture) if audit_depth_rays else None
                try:
                    # Upstream ray RANSAC draws from global RNG; preserve caller state.
                    devices=[torch.cuda.current_device()] if self.device=='cuda' else []
                    with torch.random.fork_rng(devices=devices,enabled=pose_decoder=='ray'):
                        if pose_decoder=='ray':
                            torch.random.default_generator.manual_seed(42)
                            if devices:torch.cuda.manual_seed(42)
                        output=self.net(tensors,ext_tensor,intr_tensor,[],False,pose_decoder=='ray','middle')
                finally:
                    if hook is not None:hook.remove()
            pred=self.output_processor(output)
            mean=torch.tensor([.485,.456,.406],device=self.device).view(1,3,1,1)
            std=torch.tensor([.229,.224,.225],device=self.device).view(1,3,1,1)
            rgb=(tensors[0]*std+mean).clamp(0,1).permute(0,2,3,1)
            pred.processed_images=(rgb*255).round().to(torch.uint8).cpu().numpy()
        result=replace(normalize_da3_prediction(pred,model_id=self.base_model_id,weights_sha256=self.digest),model_id=self.model_id,pose_conditioned=exts is not None,pose_decoder=pose_decoder)
        if audit_depth_rays:
            from .ray_consistency import audit_rays
            conf=captured['conf']
            if conf.ndim==4 and conf.shape[-1]==1:conf=conf[...,0]
            result=replace(result,ray_audit=audit_rays(result,captured['ray'],conf))
        return result


def main() -> None:
    """Offline original pretrained DA3 inference. Writes relative depth only."""
    import argparse
    from io import BytesIO
    from PIL import Image
    from .infer import save_atomic

    parser=argparse.ArgumentParser(description='OCAF GeoDepth Ω - authentic DA3, offline')
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--model-id',choices=('DA3-SMALL','DA3-BASE'),default='DA3-SMALL')
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--input',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--process-res',type=int,default=280)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    gate=parser.add_mutually_exclusive_group()
    gate.add_argument('--verify-reprojection',action='store_true')
    gate.add_argument('--verify-all-pairs',action='store_true',help='Require every ordered camera pair to pass (static views only)')
    parser.add_argument('--fallback-resolutions',type=int,nargs='+',default=[],help='Up to three alternate resolutions after --process-res; requires --verify-all-pairs')
    parser.add_argument('--profile',choices=('fp32','fp16','aq6','aq4','int8'),default='fp32')
    parser.add_argument('--allow-experimental',action='store_true')
    parser.add_argument('--pose-decoder',choices=('token','ray'),default='token')
    parser.add_argument('--audit-depth-rays',action='store_true')
    args=parser.parse_args()
    if args.fallback_resolutions and not args.verify_all_pairs:
        parser.error('--fallback-resolutions requires --verify-all-pairs')
    engine=DA3RealCPU(args.model_dir,source_dir=args.source_dir,device=args.device,profile=args.profile,allow_experimental=args.allow_experimental,model_id=args.model_id)
    audit=None; selection=None
    if args.verify_all_pairs:
        from .da3_selection import infer_first_consistent
        result,selection=infer_first_consistent(engine,args.input,[args.process_res,*args.fallback_resolutions],inference_options={"pose_decoder":args.pose_decoder,"audit_depth_rays":args.audit_depth_rays})
        if result is None:
            print(json.dumps(selection,ensure_ascii=False,allow_nan=False))
            raise ValueError('DA3_ALL_PAIRS_REPROJECTION_UNVERIFIABLE')
        audit=selection['attempts'][-1]['audit']
    else:
        result=engine.inference(args.input,process_res=args.process_res,pose_decoder=args.pose_decoder,audit_depth_rays=args.audit_depth_rays)
    if args.verify_reprojection:
        from .da3_validation import verify_multiview_reprojection
        audit=verify_multiview_reprojection(result)
        if not audit['passed']:
            raise ValueError('DA3_MULTIVIEW_REPROJECTION_UNVERIFIABLE')
    args.output.mkdir(parents=True,exist_ok=True)
    buff=BytesIO()
    np.savez_compressed(buff,depth_z=result.depth_z,confidence_raw=result.confidence_raw,
        intrinsics_k=result.intrinsics_k,pose_c2w=result.pose_c2w,rgb=result.rgb,
        units=np.array(result.units),provenance=np.array(result.provenance))
    save_atomic(args.output/'prediction.npz',buff.getvalue())
    for i,depth in enumerate(result.depth_z):
        valid=np.isfinite(depth)&(depth>0)
        dmin,dmax=np.percentile(depth[valid],[2,98])
        image=np.nan_to_num(np.clip((depth-dmin)/max(float(dmax-dmin),1e-8),0,1))
        b=BytesIO()
        Image.fromarray(np.round(image*255).astype(np.uint8)).save(b,format='PNG')
        save_atomic(args.output/f'depth_relative_{i:02d}.png',b.getvalue())
    meta={'model':result.model_id,'weight_profile':engine.profile_info,'weights_sha256':result.weights_sha256,
          'source':'official DA3 network from pinned user-provided source ZIP',
          'view_count':len(result.depth_z),'depth_shape':list(result.depth_z.shape),
          'units':result.units,'provenance':result.provenance,
          'confidence':'DA3 raw confidence, not calibrated probability',
          'reprojection_audit':audit,'resolution_selection':selection,
          'pose_decoder':result.pose_decoder,'ray_audit':result.ray_audit,
          'warning':'Depth is relative; do not report output in metres without a metric anchor.'}
    save_atomic(args.output/'metadata.json',json.dumps(meta,indent=2,ensure_ascii=False).encode())
    print(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
