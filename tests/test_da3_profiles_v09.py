from pathlib import Path
import shutil
import pytest
import numpy as np
from omega_geodepth.da3_profiles import validate_profile, runtime_weights
from omega_geodepth.da3_real import DA3RealCPU

ROOT=Path(__file__).resolve().parents[1]
PROFILES=ROOT/'models/da3_profiles'
SRC=ROOT/'vendor/da3_src'

def test_fp16_pin_and_runtime():
    info=validate_profile(PROFILES/'fp16','fp16')
    assert info['parent_sha256'].startswith('364492e3')
    with runtime_weights(PROFILES/'fp16','fp16') as (path,meta):
        assert path==PROFILES/'fp16'/'model.safetensors'
        assert meta['model_id']=='DA3-SMALL-FP16'
        assert meta['runtime_weights_sha256']==meta['artifact_sha256']

def test_fp16_tamper_rejected(tmp_path):
    shutil.copytree(PROFILES/'fp16',tmp_path/'p')
    with (tmp_path/'p'/'model.safetensors').open('r+b') as f:
        f.seek(-1,2); original=f.read(1); f.seek(-1,2); f.write(bytes([original[0]^1]))
    with pytest.raises(ValueError,match='DA3_PROFILE_HASH_MISMATCH'):
        validate_profile(tmp_path/'p','fp16')

@pytest.mark.parametrize('profile',['aq6','aq4'])
def test_experimental_requires_opt_in(profile):
    with pytest.raises(ValueError,match='DA3_EXPERIMENTAL_OPT_IN_REQUIRED'):
        validate_profile(PROFILES/profile,profile)

@pytest.mark.parametrize('profile',['aq6','aq4'])
def test_aq_restoration(profile):
    if not (PROFILES/profile).is_dir(): pytest.skip('optional AQ add-on not installed')
    from safetensors import safe_open
    with runtime_weights(PROFILES/profile,profile,allow_experimental=True) as (path,meta):
        assert meta['experimental'] and meta['runtime_weights_sha256']!=meta['artifact_sha256']
        with safe_open(path,framework='pt') as f: assert len(f.keys())==437 and len(f.metadata())==6
    assert not path.exists()

def test_fp16_real_inference():
    engine=DA3RealCPU(PROFILES/'fp16',source_dir=SRC,profile='fp16')
    result=engine.inference([np.zeros((126,168,3),dtype=np.uint8)],process_res=140)
    assert result.model_id=='DA3-SMALL-FP16' and result.units=='arbitrary_scale'
    assert result.weights_sha256==engine.profile_info['artifact_sha256']

def test_config_tamper_rejected(tmp_path):
    shutil.copytree(PROFILES/'fp16',tmp_path/'p')
    with (tmp_path/'p'/'config.json').open('a') as f: f.write('\n')
    with pytest.raises(ValueError,match='DA3_PROFILE_HASH_MISMATCH'):
        validate_profile(tmp_path/'p','fp16')

def test_fp16_exact_rounding_lineage():
    import torch
    from safetensors import safe_open
    with safe_open(ROOT/'models/da3/model.safetensors',framework='pt') as original, safe_open(PROFILES/'fp16/model.safetensors',framework='pt') as derived:
        assert original.keys()==derived.keys() and original.metadata()==derived.metadata()
        for name in original.keys():
            t=derived.get_tensor(name)
            assert t.dtype==torch.float16 and torch.equal(original.get_tensor(name).half(),t),name

def test_aq_5d_must_gate_reprojection():
    from omega_geodepth.reconstruction5d import build_from_da3_sequence
    with pytest.raises(ValueError,match='DA3_EXPERIMENTAL_REPROJECTION_REQUIRED'):
        build_from_da3_sequence(ROOT/'examples/v08_real_da3/manifest.json',PROFILES/'aq6','DA3-SMALL',runtime='native_network',source_dir=SRC,profile='aq6',allow_experimental=True)
