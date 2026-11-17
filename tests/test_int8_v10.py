import torch
import pytest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def test_quantize_backbone_only():
    from omega_geodepth.int8_runtime import quantize_backbone
    from torch.ao.nn.quantized.dynamic import Linear
    model=torch.nn.Module()
    model.backbone=torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.ReLU(),torch.nn.Linear(16,8))
    model.head=torch.nn.Linear(8,1)
    model.eval()
    stats=quantize_backbone(model)
    assert isinstance(model.backbone[0],Linear) and isinstance(model.head,torch.nn.Linear)
    assert stats['linear_modules']==2 and stats['packed_weight_bytes']<stats['original_weight_bytes']
    assert torch.isfinite(model.backbone(torch.ones(2,8))).all()

def test_reject_training():
    from omega_geodepth.int8_runtime import quantize_backbone
    model=torch.nn.Module(); model.backbone=torch.nn.Sequential(torch.nn.Linear(8,8))
    with pytest.raises(ValueError,match='INT8_REQUIRES_EVAL'): quantize_backbone(model)

def test_real_da3_int8():
    from omega_geodepth.da3_real import DA3RealCPU
    engine=DA3RealCPU(ROOT/'models/da3',source_dir=ROOT/'vendor/da3_src',profile='int8',allow_experimental=True)
    assert engine.profile_info['execution_precision']=='dynamic_int8_backbone_fp32_heads'
    yy,xx=np.mgrid[:126,:168]; rgb=np.stack([(xx+yy)%256,xx%256,yy%256],axis=-1).astype(np.uint8)
    out=engine.inference([rgb,np.roll(rgb,3,1)],process_res=140)
    assert out.depth_z.shape[0]==2 and np.isfinite(out.depth_z).all()
    assert len(engine.profile_info['quantized_linear_payload_sha256'])==64
    assert engine.profile_info['runtime_weights_sha256'] is None
    assert out.model_id=='DA3-SMALL-INT8-EXPERIMENTAL' and out.units=='arbitrary_scale'

def test_int8_requires_explicit_consent():
    from omega_geodepth.da3_real import DA3RealCPU
    with pytest.raises(ValueError,match='DA3_EXPERIMENTAL_OPT_IN_REQUIRED'):
        DA3RealCPU(ROOT/'models/da3',source_dir=ROOT/'vendor/da3_src',profile='int8')


def test_int8_5d_requires_reprojection(tmp_path):
    import json
    from omega_geodepth.reconstruction5d import build_from_da3_sequence
    for name in ('a.png','b.png'):
        (tmp_path/name).write_bytes(b'dummy')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'frames':[{'image':'a.png','time_s':0},{'image':'b.png','time_s':1}]}))
    with pytest.raises(ValueError,match='DA3_EXPERIMENTAL_REPROJECTION_REQUIRED'):
        build_from_da3_sequence(manifest,ROOT/'models/da3','DA3-SMALL',runtime='native_network',profile='int8',allow_experimental=True)
