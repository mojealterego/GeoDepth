"""TDD: nonlinear temporal deformation, persistence, holdout and fail-closed contracts."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from omega_geodepth.trainable_4dgs import GaussianTrainConfig, train_gaussians, TrainableGaussian4D
from omega_geodepth.deformable_4dgs import (
    DeformableGaussian4D, save_deformable_checkpoint, load_deformable_checkpoint,
    heldout_metrics,
)
from omega_geodepth.reconstruction5d import Frame5D


def K():
    return np.array([[24., 0., 11.5], [0., 24., 11.5], [0., 0., 1.]])


def scene(knots=(0., .5, 1., 1.5, 2.)):
    return DeformableGaussian4D(
        torch.tensor([[0., 0., 2.2]], dtype=torch.float32),
        torch.zeros(1, 3), torch.tensor([[.95, .15, .04]]),
        torch.tensor([[.15]]), torch.tensor([[.97]]),
        reference_time=0., knot_times=knots)


def observations(times):
    # Independent analytic trajectory. Nonlinear motion reverses direction.
    gt=scene()
    frames=[]
    with torch.no_grad():
        for t in times:
            # Generated from an external analytic function, not fitted knot parameters.
            x=.30*np.sin(np.pi*t)
            gt.means_xyz[0,0]=float(x)
            rgb,alpha,depth=gt.render(0.,K(),np.eye(4),24,24)
            mask=(alpha.numpy()>.03)
            frames.append(Frame5D(float(t),np.where(mask,depth.numpy(),np.nan).astype('float32'),
                    np.rint(rgb.numpy()*255).clip(0,255).astype('uint8'),K(),np.eye(4),mask))
    return frames


def test_non_linear_deformation_and_reference_anchor():
    m=scene()
    with torch.no_grad():
        m.deformation_offsets[2,0,0]=.24
    assert torch.allclose(m.world_positions(0.),m.means_xyz)
    assert abs(float(m.world_positions(1.)[0,0].detach()))>.05
    assert not torch.allclose(m.world_positions(.5),m.world_positions(1.))
    with pytest.raises(ValueError,match='GAUSSIAN_TIME_OUTSIDE_TRAINED_INTERVAL'):
        m.render(2.1,K(),np.eye(4),24,24)


def test_deformation_gradients_train_on_a_non_linear_motion():
    torch.manual_seed(4)
    m=scene()
    train=observations((0., .5, 1., 1.5, 2.))
    report=train_gaussians(m,train,GaussianTrainConfig(steps=100,lr=.055,
               optimize_velocity_only=True,timeout_s=180.))
    assert report['loss_final'] < report['loss_initial']*.8
    assert m.deformation_offsets.grad is not None
    assert float(m.deformation_offsets.detach().abs().sum())>.02


def test_save_load_and_tamper_detection():
    m=scene()
    with torch.no_grad():m.deformation_offsets[2,0,0]=.12
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        save_deformable_checkpoint(m,root,units='arbitrary_scale',provenance='relative_estimated',training_report={})
        loaded=load_deformable_checkpoint(root)
        torch.testing.assert_close(loaded.world_positions(.8),m.world_positions(.8))
        assert loaded.reference_time==m.reference_time
        assert loaded.knot_times.tolist()==m.knot_times.tolist()
        (root/'gaussians.safetensors').write_bytes((root/'gaussians.safetensors').read_bytes()+b'bad')
        with pytest.raises(ValueError,match='SHA256'):
            load_deformable_checkpoint(root)


def test_invalid_knots_or_fake_metric_units_fail_closed():
    for knots in ((0.,0.,1.),(0.,1.),(0.,float('nan'),1.),(2.,3.,4.)):
        with pytest.raises(ValueError,match='GAUSSIAN_INVALID_KNOTS'):
            scene(knots)
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError,match='PROVENANCE'):
            save_deformable_checkpoint(scene(),Path(td),units='meters',provenance='relative_estimated',training_report={})


def test_holdout_does_not_use_training_times_and_reports_coverage():
    m=scene()
    metrics=heldout_metrics(m,observations((.25,.75,1.25,1.75)))
    assert metrics['frames']==4
    assert metrics['rgb_mae_valid']>=0.
    assert 0<=metrics['fraction_predicted_covered']<=1
    assert metrics['sample_times_s']==[.25,.75,1.25,1.75]


def test_linear_checkpoint_backward_compatible():
    from omega_geodepth.trainable_4dgs import save_gaussian_checkpoint,load_gaussian_checkpoint
    old=TrainableGaussian4D(torch.tensor([[0.,0.,2.]]),torch.zeros(1,3),torch.tensor([[.3,.6,.2]]),
        torch.tensor([[.1]]),torch.tensor([[.9]]),0.)
    with tempfile.TemporaryDirectory() as td:
        save_gaussian_checkpoint(old,Path(td),units='arbitrary_scale',provenance='relative_estimated',training_report={})
        loaded=load_gaussian_checkpoint(Path(td))
        torch.testing.assert_close(old.means_xyz,loaded.means_xyz)


def test_forged_units_in_local_manifest_rejected():
    import json
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        save_deformable_checkpoint(scene(),root,units='arbitrary_scale',provenance='relative_estimated',training_report={})
        path=root/'checkpoint.json'
        doc=json.loads(path.read_text())
        doc['units']='meters'
        path.write_text(json.dumps(doc))
        with pytest.raises(ValueError,match='PROVENANCE'):
            load_deformable_checkpoint(root)


def test_nonfinite_deformation_weights_fail_closed_after_manifest_recomputed():
    from safetensors.torch import load_file,save_file
    import json,hashlib
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        save_deformable_checkpoint(scene(),root,units='arbitrary_scale',provenance='relative_estimated',training_report={})
        path=root/'gaussians.safetensors'
        data=load_file(str(path));data['deformation_offsets'][0,0,0]=float('nan')
        save_file(data,str(path))
        m=root/'checkpoint.json';doc=json.loads(m.read_text())
        doc['weights_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        m.write_text(json.dumps(doc))
        with pytest.raises(ValueError,match='NONFINITE'):
            load_deformable_checkpoint(root)


def test_invalid_large_times_rejected_before_float32_precision_collapse():
    with pytest.raises(ValueError,match='GAUSSIAN_INVALID_KNOTS'):
        DeformableGaussian4D(**{**dict(means_xyz=torch.tensor([[0.,0.,2.2]]),
            velocity=torch.zeros(1,3),rgb=torch.tensor([[.95,.15,.04]]),
            sigma_world=torch.tensor([[.15]]),opacity=torch.tensor([[.97]]),
            reference_time=1e10+.25), 'knot_times': (1e10,1e10+.25,1e10+.5)})


def test_end_to_end_new_cli_train_and_render(tmp_path):
    import json
    from omega_geodepth.deformable_4dgs import main
    frames=observations((0., .5, 1., 1.5, 2.))
    records=[]
    for i,f in enumerate(frames):
        np.savez_compressed(tmp_path/f'{i}.npz',depth_z=f.depth_z,source_rgb=f.rgb,
             intrinsics_k=f.intrinsics,valid_mask=f.valid)
        records.append({'time_s':f.time_s,'npz':f'{i}.npz','pose_c2w':f.pose_c2w.tolist()})
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'units':'arbitrary_scale','provenance':'relative_estimated',
         'frames':records}))
    target=tmp_path/'model'
    main(['train','--manifest',str(manifest),'--output',str(target),
        '--steps','6','--knots','5','--splats','4','--max-side','24',
        '--motion-only'])
    assert (target/'checkpoint.json').is_file()
    main(['render','--checkpoint',str(target),'--manifest',str(manifest),
          '--time','.75','--max-side','24','--output',str(tmp_path/'mid.png')])
    assert (tmp_path/'mid.png').is_file()
    with np.load(tmp_path/'mid.npz') as data:
        assert data['alpha'].shape==(24,24)
        assert data['depth'].shape==(24,24)


def test_upgrade_preserves_legacy_scene_parameters():
    from omega_geodepth.deformable_4dgs import upgrade_linear_gaussians
    linear=TrainableGaussian4D(torch.tensor([[.05,.01,2.]]),torch.tensor([[.1,0,0]]),
        torch.tensor([[.3,.6,.2]]),torch.tensor([[.1]]),torch.tensor([[.9]]),0.)
    with torch.no_grad():
        a=linear.render(.75,K(),np.eye(4),24,24)
        b=upgrade_linear_gaussians(linear,(0., .5, 1., 1.5, 2.)).render(.75,K(),np.eye(4),24,24)
    for left,right in zip(a,b):
        torch.testing.assert_close(left,right,equal_nan=True)


def test_checkpoint_files_are_not_allowed_to_be_symlinks(tmp_path):
    model=scene()
    save_deformable_checkpoint(model,tmp_path,units='arbitrary_scale',
        provenance='relative_estimated',training_report={})
    (tmp_path/'checkpoint.json').rename(tmp_path/'real.json')
    (tmp_path/'checkpoint.json').symlink_to(tmp_path/'real.json')
    with pytest.raises(ValueError,match='UNSAFE'):
        load_deformable_checkpoint(tmp_path)
