"""Strict local-only adapter for original Depth Anything 3 checkpoint files.

DA3-SMALL and DA3-BASE are ANY-VIEW RELATIVE geometry estimators.
The actual weights must exist on disk and match a SHA-256 manifest.
No downloading or unpickling occurs during inference.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import hashlib
import json
import os
import numpy as np

from .geometry import validate_pose_c2w

ALLOWED_DA3={
    'DA3-SMALL': 'depth-anything/DA3-SMALL',
    'DA3-BASE': 'depth-anything/DA3-BASE',
}

# SHA-256 of official upstream model.safetensors (Hugging Face LFS/Xet pointers,
# DA3-SMALL commit e08cab65 and DA3-BASE commit 6544606).
# A self-signed local manifest is NOT proof of upstream provenance.
OFFICIAL_WEIGHT_SHA256 = {
    'DA3-SMALL': '364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf',
    'DA3-BASE': 'e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5',
}


def _check_checkpoint_paths(root:Path)->None:
    if root.is_symlink():
        raise ValueError('DA3_UNSAFE_CHECKPOINT_PATH')
    for filename in ('model.safetensors','config.json','README.md','LICENSE','omega_da3_manifest.json'):
        candidate=root/filename
        if candidate.is_symlink():
            raise ValueError('DA3_UNSAFE_CHECKPOINT_PATH')


def _assert_official_weights(path:Path, model_id:str)->str:
    # Never trust a digest provided by a locally authored manifest alone.
    digest=file_sha256(path)
    if digest!=OFFICIAL_WEIGHT_SHA256[model_id]:
        raise ValueError('DA3_OFFICIAL_HASH_MISMATCH')
    return digest

DA3_ALLOWED_OBJECTS={
    ('depth_anything_3.model.da3','DepthAnything3Net'),
    ('depth_anything_3.model.dinov2.dinov2','DinoV2'),
    ('depth_anything_3.model.dualdpt','DualDPT'),
    ('depth_anything_3.model.cam_enc','CameraEnc'),
    ('depth_anything_3.model.cam_dec','CameraDec'),
}


def validate_official_config(path:Path,model_id:str)->dict:
    # DA3 official configs describe dynamic object constructors. An untrusted
    # checkpoint must never be allowed to import arbitrary Python modules.
    if path.stat().st_size>1024*1024:raise ValueError('DA3_CONFIG_TOO_LARGE')
    config=json.loads(path.read_text(encoding='utf-8'))
    if config.get('model_name')!=model_id.lower():
        raise ValueError('DA3_CONFIG_MODEL_MISMATCH')
    objects=[]
    def inspect(node):
        if isinstance(node,dict):
            if '__object__' in node:
                spec=node['__object__']
                obj=(spec.get('path'),spec.get('name')) if isinstance(spec,dict) else (None,None)
                if obj not in DA3_ALLOWED_OBJECTS:
                    raise ValueError('DA3_CONFIG_UNAPPROVED_OBJECT')
                objects.append(obj)
            for val in node.values():inspect(val)
        elif isinstance(node,list):
            for val in node:inspect(val)
    inspect(config)
    if not objects or ('depth_anything_3.model.da3','DepthAnything3Net') not in objects:
        raise ValueError('DA3_CONFIG_MISSING_MODEL')
    return config


def file_sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def validate_da3_checkpoint(model_dir:Path,model_id:str)->dict:
    if model_id not in ALLOWED_DA3:
        raise ValueError('DA3_MODEL_NOT_ALLOWED: only official Apache-2.0 ANY-VIEW Small/Base')
    root=Path(model_dir)
    _check_checkpoint_paths(root)
    root=root.resolve(strict=False)
    manifest_path=root/'omega_da3_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('DA3_CHECKPOINT_MANIFEST_MISSING: run register-da3 after downloading')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('repo_id')!=ALLOWED_DA3[model_id] or manifest.get('license')!='apache-2.0':
        raise ValueError('DA3_MANIFEST_ID_OR_LICENSE_MISMATCH')
    files=manifest.get('files')
    if not isinstance(files,dict) or 'model.safetensors' not in files or 'config.json' not in files:
        raise ValueError('DA3_CHECKPOINT_FILES_MISSING')
    validate_official_config(root/'config.json',model_id)
    for filename,expected in files.items():
        if filename not in ('model.safetensors','config.json','README.md','LICENSE'):
            raise ValueError('DA3_UNAPPROVED_CHECKPOINT_FILE')
        path=root/filename
        if not path.is_file() or not isinstance(expected,str) or len(expected)!=64:
            raise ValueError('DA3_CHECKPOINT_FILES_MISSING')
        if file_sha256(path)!=expected:
            raise ValueError('DA3_SHA256_MISMATCH')
    _assert_official_weights(root/'model.safetensors',model_id)
    return manifest


def register_da3_checkpoint(model_dir:Path,model_id:str,revision:str)->dict:
    """Register user-provided ORIGINAL files, never create fake DA3 weights."""
    if model_id not in ALLOWED_DA3: raise ValueError('DA3_MODEL_NOT_ALLOWED')
    root=Path(model_dir)
    _check_checkpoint_paths(root)
    root=root.resolve(strict=True)
    if not (root/'model.safetensors').is_file() or not (root/'config.json').is_file():
        raise FileNotFoundError('DA3_CHECKPOINT_FILES_MISSING')
    validate_official_config(root/'config.json',model_id)
    # Validate safetensors header without reading model data or running untrusted code.
    from safetensors import safe_open
    with safe_open(root/'model.safetensors',framework='pt',device='cpu') as reader:
        if not reader.keys(): raise ValueError('DA3_EMPTY_SAFETENSORS')
    upstream_hash=_assert_official_weights(root/'model.safetensors',model_id)
    files={x:file_sha256(root/x) for x in ('config.json','model.safetensors')}
    for optional in ('README.md','LICENSE'):
        if (root/optional).is_file(): files[optional]=file_sha256(root/optional)
    manifest={'schema_version':'omega-da3-local-v1','model_id':model_id,
              'repo_id':ALLOWED_DA3[model_id], 'revision':revision,
              'license':'apache-2.0','verified_with':'official_upstream_sha256_and_local_sha256',
              'official_weights_sha256':upstream_hash,
              'files':files,'provenance':'user_supplied_checkpoint_origin_not_verified'}
    out=root/'omega_da3_manifest.json'
    from .infer import save_atomic
    save_atomic(out,json.dumps(manifest,indent=2,sort_keys=True).encode('utf-8'))
    return manifest


@dataclass(frozen=True)
class DA3Output:
    depth_z:np.ndarray            # N,H,W, relative optical-axis Z
    confidence_raw:np.ndarray     # DA3 confidence is not a calibrated probability
    intrinsics_k:np.ndarray       # N,3,3 at processed resolution
    pose_c2w:np.ndarray           # N,4,4, SE3
    rgb:np.ndarray                # N,H,W,3, uint8 processed images
    units:str
    provenance:str
    model_id:str
    weights_sha256:str
    pose_conditioned:bool=False
    pose_decoder:str="token"
    ray_audit:dict|None=None
    backbone_features:np.ndarray|None=None  # N,H/14,W/14,C; opt-in layer 11


def normalize_da3_prediction(prediction:Any,*,model_id:str,weights_sha256:str)->DA3Output:
    if model_id not in ALLOWED_DA3:raise ValueError('DA3_MODEL_NOT_ALLOWED')
    depth=np.asarray(getattr(prediction,'depth',None))
    if depth.ndim!=3 or not np.issubdtype(depth.dtype,np.number):
        raise ValueError('INVALID_DA3_DEPTH')
    # In real images, sky/occlusion pixels may be invalid. Preserve them as
    # missing measurements rather than rejecting a whole frame or inventing Z.
    valid=np.isfinite(depth)&(depth>0)
    if not np.all(valid.reshape(depth.shape[0],-1).any(axis=1)):
        raise ValueError('INVALID_DA3_DEPTH')
    depth=np.where(valid,depth,np.nan)
    n,h,w=depth.shape
    conf_source=getattr(prediction,'conf',None)
    if conf_source is None:raise ValueError('INVALID_DA3_CONFIDENCE')
    conf=np.asarray(conf_source)
    if conf.shape!=depth.shape or not np.isfinite(conf).all():raise ValueError('INVALID_DA3_CONFIDENCE')
    exts_source=getattr(prediction,'extrinsics',None)
    intr_source=getattr(prediction,'intrinsics',None)
    rgb_source=getattr(prediction,'processed_images',None)
    if exts_source is None:raise ValueError('INVALID_DA3_POSE')
    exts=np.asarray(exts_source,dtype=np.float64)
    if exts.shape not in ((n,3,4),(n,4,4)):raise ValueError('INVALID_DA3_POSE')
    if exts.shape[1:]==(3,4):
        w2c=np.broadcast_to(np.eye(4),(n,4,4)).copy()
        w2c[:,:3,:]=exts
    else:w2c=exts
    poses=[]
    for e in w2c:
        try:
            validate_pose_c2w(e) # validates SE(3), regardless of direction
            poses.append(validate_pose_c2w(np.linalg.inv(e)))
        except (ValueError,np.linalg.LinAlgError) as error:
            raise ValueError('INVALID_DA3_POSE') from error
    if intr_source is None:raise ValueError('INVALID_DA3_INTRINSICS')
    intr=np.asarray(intr_source,dtype=np.float64)
    if (intr.shape!=(n,3,3) or not np.isfinite(intr).all() or
        not np.all(intr[:,0,0]>0) or not np.all(intr[:,1,1]>0) or
        not np.allclose(intr[:,2,:],[0,0,1],atol=1e-5)):
        raise ValueError('INVALID_DA3_INTRINSICS')
    rgb=np.asarray(rgb_source)
    if rgb.shape!=(n,h,w,3) or rgb.dtype!=np.uint8: raise ValueError('INVALID_DA3_RGB')
    if len(weights_sha256)!=64:raise ValueError('DA3_WEIGHT_HASH_INVALID')
    return DA3Output(depth.astype(np.float32),conf.astype(np.float32),intr,
                     np.stack(poses,axis=0),rgb,'arbitrary_scale','relative_estimated',model_id,weights_sha256)


def resolve_verified_da3_source(source_dir:Path)->Path:
    """Use the byte-for-byte extracted, checked user-provided source tree.

    Return the source import root without executing any third-party code.
    """
    from .da3_source import verify_extracted_da3_source
    root=Path(source_dir).absolute()
    verify_extracted_da3_source(root)
    source=root/'src'
    if not (source/'depth_anything_3'/'api.py').is_file():
        raise ValueError('DA3_SOURCE_API_MISSING')
    return source


def _load_official_da3_class(source_dir:Path|None):
    import importlib
    import importlib.util
    import sys
    if source_dir is not None:
        source_root=resolve_verified_da3_source(source_dir).resolve()
        # Imported namespace packages are not allowed to silently shadow the
        # explicit source archive. The source tree contains no package __init__.
        root_pkg=sys.modules.get('depth_anything_3.api')
        if root_pkg is not None:
            existing=Path(getattr(root_pkg,'__file__','') or '/not-found').resolve()
            if not existing.is_relative_to(source_root):
                raise RuntimeError('DA3_SOURCE_IMPORT_COLLISION')
        src_str=str(source_root)
        if src_str not in sys.path:sys.path.insert(0,src_str)
        importlib.invalidate_caches()
        spec=importlib.util.find_spec('depth_anything_3.api')
        if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(source_root):
            raise RuntimeError('DA3_SOURCE_IMPORT_COLLISION')
    else:
        source_root=None
    try:
        from depth_anything_3.api import DepthAnything3
    except (ImportError,ModuleNotFoundError) as ex:
        raise RuntimeError('DA3_OFFICIAL_PACKAGE_OR_DEPENDENCIES_MISSING') from ex
    if source_root is not None:
        module_path=Path(sys.modules[DepthAnything3.__module__].__file__).resolve()
        if not module_path.is_relative_to(source_root):
            raise RuntimeError('DA3_SOURCE_IMPORT_COLLISION')
    return DepthAnything3


class DA3OfflineBackend:
    """Load only verified local checkpoints. Official DA3 source is an extra dependency."""
    def __init__(self,model_dir:Path,model_id:str='DA3-SMALL',device:str='cpu',
                 source_dir:Path|None=None):
        self.model_dir=Path(model_dir)
        self.manifest=validate_da3_checkpoint(self.model_dir,model_id)
        self.model_id=model_id
        if device not in ('cpu','cuda'):raise ValueError('DA3_INVALID_DEVICE')
        # Disable model hub network access in Hugging Face; workers should also be isolated by OS.
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['TRANSFORMERS_OFFLINE']='1'
        os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
        DepthAnything3=_load_official_da3_class(source_dir)
        self.model=DepthAnything3.from_pretrained(str(self.model_dir)).to(device)
        self.device=device

    def inference(self,images:Sequence[str|Path|np.ndarray],*,process_res:int=504,
                  use_ray_pose:bool=False,ref_view_strategy:str='middle')->DA3Output:
        if not (1<=len(images)<=128):raise ValueError('DA3_INVALID_VIEW_COUNT')
        if not (128<=process_res<=1536):raise ValueError('DA3_INVALID_PROCESS_RES')
        if len(images)*(process_res**2)>8_000_000:raise ValueError('DA3_RESOURCE_LIMIT')
        if ref_view_strategy not in ('first','middle','saddle_balanced','saddle_sim_range'):
            raise ValueError('DA3_INVALID_REF_VIEW_STRATEGY')
        args={'image':[str(x) if isinstance(x,Path) else x for x in images],
              'process_res':process_res,'use_ray_pose':use_ray_pose,
              'ref_view_strategy':ref_view_strategy}
        out=self.model.inference(**args)
        return normalize_da3_prediction(out,model_id=self.model_id,
                                        weights_sha256=self.manifest['files']['model.safetensors'])
