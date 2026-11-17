"""A working, offline-only ComfyUI V1 adapter for OMEGA GeoDepth relative depth.

Outputs are relative (arbitrary scale), irrespective of subject content. This is
not the 18-node production catalog from the 80-page OCAF specification.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from omega_geodepth.infer import load_verified_model
from omega_geodepth.tiling import _forward, tiled_predict

class OmegaGeoDepth:
    CATEGORY='OCAF/GeoDepth Ω'
    FUNCTION='execute'
    RETURN_TYPES=('IMAGE','MASK','STRING')
    RETURN_NAMES=('depth_preview_rgb','relative_depth_preview','provenance_json')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'image':('IMAGE',),
            'model_directory':('STRING',{'default':'models'}),
            'processing':(['direct','tiled'],),
            'tile_size':('INT',{'default':256,'min':32,'max':1024,'step':8}),
            'overlap':('INT',{'default':64,'min':8,'max':512,'step':8}),
            'global_side':('INT',{'default':256,'min':32,'max':1024,'step':8}),
        }}

    def execute(self,image:torch.Tensor,model_directory:str,processing:str,
                tile_size:int,overlap:int,global_side:int):
        if processing not in ('direct','tiled'):
            raise ValueError('INVALID_PROCESSING_MODE')
        if image.ndim!=4 or image.shape[-1]!=3 or not (1<=image.shape[0]<=8):
            raise ValueError('EXPECTED_COMFY_IMAGE_BHWC_RGB')
        if image.shape[1]*image.shape[2]>24_000_000:
            raise ValueError('IMAGE_RESOURCE_LIMIT')
        if not torch.isfinite(image).all().item():
            raise ValueError('INVALID_INPUT_IMAGE')
        model_dir=Path(model_directory).expanduser()
        if not model_dir.is_dir():
            raise FileNotFoundError('MODEL_DIR_MISSING')
        model,manifest=load_verified_model(model_dir,'cpu')
        previews=[]
        for tensor in image:
            rgb=np.ascontiguousarray(tensor.detach().to('cpu',dtype=torch.float32).numpy())
            if processing=='tiled':
                logd,_=tiled_predict(model,rgb,tile_size=tile_size,overlap=overlap,
                                     global_side=global_side,device='cpu')
            else:
                logd,_=_forward(model,rgb,device='cpu')
            near=-logd
            low,high=np.percentile(near,[2,98])
            preview=np.clip((near-low)/max(float(high-low),1e-6),0,1).astype(np.float32)
            previews.append(torch.from_numpy(preview))
        mask=torch.stack(previews)
        image_out=mask.unsqueeze(-1).expand(-1,-1,-1,3).contiguous()
        meta={'model':manifest['model'],'model_version':manifest['version'],
              'model_sha256':manifest['weights_sha256'][manifest.get('weights_file','omega_geodepth_synthetic_v0.1.safetensors')],
              'units':'arbitrary_scale','provenance':'relative_estimated',
              'preview_only':True,'processing':processing,'content_filter':'none',
              'quality_limit':'synthetic-only training; no measured metric distance'}
        return (image_out,mask,json.dumps(meta,ensure_ascii=False))

NODE_CLASS_MAPPINGS={'OCAF_Omega_GeoDepth':OmegaGeoDepth}
NODE_DISPLAY_NAME_MAPPINGS={'OCAF_Omega_GeoDepth':'OCAF GeoDepth Ω — Relative Depth (Offline)'}
