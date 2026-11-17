"""Byte-pinned DA3 Small derivatives. AQ artifacts are transport formats only."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import json
import tempfile
from types import MappingProxyType
from .da3_adapter import file_sha256, validate_official_config, OFFICIAL_WEIGHT_SHA256

PARENT=OFFICIAL_WEIGHT_SHA256['DA3-SMALL']
CONFIG_SHA='39888df953f0cb75ae8486e1d9168bdcbf3c74cb1234b93a0cf3b98386d79578'
PINS=MappingProxyType({
    'fp16':('DA3-SMALL-FP16','388b86c06bae986d0166e35c72813993574efdaa33b0d9ec469e52caae9f98d5',None,68649566),
    'aq6':('DA3-SMALL-AQ6-EXPERIMENTAL','9d0a64860243d7a142669bc96c37d41eb2507e2c5d50e1650c622aeed88c9a14','c35654fd789f6243ef3eb066bb04815b2b1101a4fcd68de3153351be48e4e082',33882700),
    'aq4':('DA3-SMALL-AQ4-EXPERIMENTAL','cf905633897e8899bc1e318c0f01ac88578c421b4f23d50035011d865c723bbe','882508c2d6b49a8cc3697417e7f268e124e1229fcc1ca660e71defb83bb7da2f',29142924),
})

def validate_profile(folder:Path, profile:str, *, allow_experimental:bool=False)->dict:
    if profile not in PINS: raise ValueError('DA3_UNKNOWN_PROFILE')
    identifier,digest,descriptor,size=PINS[profile]
    if descriptor and not allow_experimental: raise ValueError('DA3_EXPERIMENTAL_OPT_IN_REQUIRED')
    folder=Path(folder)
    filenames=['model.safetensors','config.json']+(['quantization.json'] if descriptor else [])
    if folder.is_symlink() or any((folder/name).is_symlink() for name in filenames):
        raise ValueError('DA3_PROFILE_UNSAFE_PATH')
    for name,expected in zip(filenames,[digest,CONFIG_SHA,descriptor]):
        p=folder/name
        if not p.is_file(): raise FileNotFoundError(f'DA3_PROFILE_FILE_MISSING: {p}')
        if p.stat().st_size>150_000_000 or (name=='model.safetensors' and p.stat().st_size!=size) or file_sha256(p)!=expected:
            raise ValueError('DA3_PROFILE_HASH_MISMATCH: '+name)
    validate_official_config(folder/'config.json','DA3-SMALL')
    if descriptor:
        from .da3_quant.omega_adaptive import validate_adaptive
        doc=json.loads((folder/'quantization.json').read_text())
        if doc.get('source_sha256')!=PARENT or doc.get('profile')!=profile:
            raise ValueError('DA3_PROFILE_PARENT_MISMATCH')
        verified=validate_adaptive(folder)
        if not verified['ok']: raise ValueError('DA3_PROFILE_INVALID: '+verified['reason'])
    return {'model_id':identifier,'profile':profile,'parent_sha256':PARENT,
            'artifact_sha256':digest,'config_sha256':CONFIG_SHA,'descriptor_sha256':descriptor,
            'license':'Apache-2.0','experimental':bool(descriptor),'units':'arbitrary_scale',
            'execution_precision':'float32','storage_format':'packed_aq' if descriptor else 'safetensors_fp16',
            'provenance_verification':'pinned_user_supplied_derivative_not_official_checkpoint'}

@contextmanager
def runtime_weights(folder:Path,profile:str,*,allow_experimental:bool=False):
    """Yield only verified standard SafeTensors; never retain restored AQ files."""
    folder=Path(folder)
    info=validate_profile(folder,profile,allow_experimental=allow_experimental)
    if not info['experimental']:
        yield folder/'model.safetensors',dict(info,runtime_weights_sha256=info['artifact_sha256'])
        return
    from .da3_quant.omega_adaptive import restore_adaptive
    with tempfile.TemporaryDirectory(prefix='geodepth-aq-') as temp:
        out=Path(temp)/'model.safetensors'
        result=restore_adaptive(folder,out)
        yield out,dict(info,runtime_weights_sha256=result['sha256'])
