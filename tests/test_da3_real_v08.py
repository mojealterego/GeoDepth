"""DA3-Small with original official weights; should not require internet."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
WEIGHTS = HERE / 'models' / 'da3'
SRC = HERE / 'vendor' / 'da3_runtime_compat'
OFFICIAL_DIGEST = '364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf'


def test_official_weights_present_and_config_verified():
    from omega_geodepth.da3_adapter import validate_official_config
    assert WEIGHTS.joinpath('model.safetensors').is_file()
    assert hashlib.sha256(WEIGHTS.joinpath('model.safetensors').read_bytes()).hexdigest() == OFFICIAL_DIGEST
    config = validate_official_config(WEIGHTS / 'config.json', 'DA3-SMALL')
    assert config['model_name'] == 'da3-small'


def test_official_inference_cpu_two_views():
    from omega_geodepth.da3_real import DA3RealCPU
    backend = DA3RealCPU(WEIGHTS, source_dir=HERE / 'vendor' / 'da3_src', device='cpu')
    yy, xx = np.mgrid[:126, :168]
    rgb = np.stack([(xx + yy) % 256, xx % 256, yy % 256], axis=-1).astype(np.uint8)
    predicted = backend.inference([rgb, np.roll(rgb, 3, 1)], process_res=140)
    assert predicted.model_id == 'DA3-SMALL'
    assert predicted.depth_z.shape[0] == 2
    assert predicted.depth_z.ndim == 3
    assert np.isfinite(predicted.depth_z).any()
    assert predicted.units == 'arbitrary_scale'
    assert predicted.provenance == 'relative_estimated'
    assert predicted.pose_c2w.shape == (2, 4, 4)
    assert predicted.intrinsics_k.shape == (2, 3, 3)


def test_source_code_integrity_rejects_runtime_tampering(tmp_path):
    from omega_geodepth.da3_real import verify_da3_runtime_source
    from shutil import copytree
    source = HERE/'vendor'/'da3_src'
    copied = tmp_path/'src'
    copytree(source, copied)
    verify_da3_runtime_source(copied, HERE/'vendor'/'Depth-Anything-3-main.zip')
    (copied/'src'/'depth_anything_3'/'cfg.py').write_text('raise RuntimeError("tampered")')
    with pytest.raises(ValueError,match='DA3_SOURCE_TAMPERED'):
        verify_da3_runtime_source(copied, HERE/'vendor'/'Depth-Anything-3-main.zip')


def test_real_da3_reconstruction5d_two_frames(tmp_path):
    from omega_geodepth.reconstruction5d import build_from_da3_sequence, build_scene5d, save_scene5d
    from PIL import Image
    yy,xx=np.mgrid[:126,:168]
    rgb=np.stack([(xx+yy)%256,xx%256,yy%256],axis=-1).astype(np.uint8)
    for i in range(2):
        Image.fromarray(np.roll(rgb,i*3,axis=1)).save(tmp_path/f'f{i}.png')
    (tmp_path/'manifest.json').write_text(json.dumps({'frames':[{'image':'f0.png','time_s':0},{'image':'f1.png','time_s':0.5}]}))
    frames,meta=build_from_da3_sequence(tmp_path/'manifest.json',WEIGHTS,'DA3-SMALL',
        source_dir=HERE/'vendor'/'da3_src',process_res=140,runtime='native_network')
    assert len(frames)==2 and meta['units']=='arbitrary_scale'
    scene=build_scene5d(frames,voxel_size=.05,max_points_per_frame=1500)
    report=save_scene5d(scene,tmp_path/'output',frames,mesh=True)
    assert report['total_points']>100
    assert report['units']=='arbitrary_scale'
    assert (tmp_path/'output'/'scene5d.ply').stat().st_size>100


def test_rejects_bad_input_before_network():
    from omega_geodepth.da3_real import DA3RealCPU
    backend=DA3RealCPU(WEIGHTS,source_dir=HERE/'vendor'/'da3_src')
    with pytest.raises(ValueError,match='DA3_INVALID_INPUT_IMAGE'):
        backend.inference([np.zeros((8,8,3),dtype=np.float32)],process_res=140)


def test_official_da3_cli_outputs(tmp_path):
    import subprocess,sys
    from PIL import Image
    yy,xx=np.mgrid[:126,:168]
    im=np.stack([(xx+yy)%256,xx%256,yy%256],axis=-1).astype(np.uint8)
    for i in range(2):
        Image.fromarray(np.roll(im,i*3,axis=1)).save(tmp_path/f'image{i}.png')
    r=subprocess.run([sys.executable,'-m','omega_geodepth.da3_real','--model-dir',str(WEIGHTS),
        '--source-dir',str(HERE/'vendor'/'da3_src'),'--input',str(tmp_path/'image0.png'),str(tmp_path/'image1.png'),
        '--process-res','140','--output',str(tmp_path/'pred')],capture_output=True,text=True,timeout=50,
        cwd=str(HERE), env={**__import__('os').environ,'PYTHONPATH':str(HERE)})
    assert r.returncode==0, r.stdout+r.stderr
    result=np.load(tmp_path/'pred'/'prediction.npz')
    assert result['depth_z'].shape[0]==2
    assert (tmp_path/'pred'/'metadata.json').is_file()
    assert (tmp_path/'pred'/'depth_relative_00.png').is_file()


def test_da3_native_rejects_imported_foreign_module(monkeypatch):
    import sys, types
    from omega_geodepth.da3_real import DA3RealCPU
    fake=types.ModuleType('depth_anything_3.cfg')
    fake.__file__='/tmp/untrusted_da3_cfg.py'
    monkeypatch.setitem(sys.modules,'depth_anything_3.cfg',fake)
    with pytest.raises(RuntimeError,match='DA3_SOURCE_IMPORT_COLLISION'):
        DA3RealCPU(WEIGHTS,source_dir=HERE/'vendor'/'da3_src')
