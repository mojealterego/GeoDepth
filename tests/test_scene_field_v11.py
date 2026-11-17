import numpy as np
import pytest
from omega_geodepth.reconstruction5d import Frame5D
from omega_geodepth.scene_field import collect_samples, fit_scene_field, save_scene_field, load_scene_field, align_baseline


def frame(t=0., tx=0.):
    y,x=np.mgrid[:8,:8]; p=np.eye(4); p[0,3]=tx
    rgb=np.stack([x*20,y*20,np.full_like(x,100)],-1).astype('uint8')
    return Frame5D(t,np.full((8,8),2.),rgb,np.array([[10.,0.,3.5],[0.,10.,3.5],[0.,0.,1.]]),p,np.ones((8,8),bool))


def test_static_group_and_relative():
    s=collect_samples([frame(),frame(0,.2)],units='arbitrary_scale',provenance='relative_estimated',reference_radius=.1,time_basis='static_assumption')
    assert s.coordinates.shape==(128,5)
    assert np.all(s.coordinates[:,3]==0)
    assert s.coordinates[0,4]==pytest.approx(1)
    assert s.view_index[64]==1
    assert s.units=='arbitrary_scale'


def test_invalid_contracts():
    for kw in [dict(units='meters',provenance='relative_estimated'),dict(units='arbitrary_scale',provenance='metric_measured'),dict(time_basis='static_assumption')]:
        args=dict(units='arbitrary_scale',provenance='relative_estimated',reference_radius=.1,time_basis='capture_seconds');args.update(kw)
        with pytest.raises(ValueError):collect_samples([frame(),frame(1)],**args)


def test_budget():
    with pytest.raises(MemoryError):collect_samples([frame(),frame()],units='arbitrary_scale',provenance='relative_estimated',reference_radius=.1,time_basis='static_assumption',max_samples=63)


def test_alignment():
    fs=[frame(),frame(0,1),frame(0,2)]
    aligned,meta=align_baseline(fs,[(0,1,.5),(0,2,1.)],evidence_id='synthetic-rig')
    assert meta['scale']==pytest.approx(.5)
    assert aligned[0].depth_z[0,0]==pytest.approx(1.)
    assert aligned[2].pose_c2w[0,3]==pytest.approx(1.)
    assert fs[2].pose_c2w[0,3]==2
    with pytest.raises(ValueError):align_baseline(fs,[(0,1,.5),(0,2,2.)],evidence_id='bad-rig')


def test_fit_holdout_and_checkpoint(tmp_path):
    fs=[frame(),frame(.5,.1),frame(1,.2)]
    s=collect_samples(fs,units='meters',provenance='metric_measured',reference_radius=.1,time_basis='capture_seconds')
    model,report=fit_scene_field(s,holdout_views=[1],steps=20,seed=2)
    assert report['train_samples']==128 and report['test_samples']==64
    assert report['holdout_views']==[1]
    assert report['train_mse_final']<report['train_mse_initial']
    path=tmp_path/'test.safetensors'; save_scene_field(path,model,s,report)
    restored,meta=load_scene_field(path)
    np.testing.assert_allclose(model.predict(s.coordinates[:8]),restored.predict(s.coordinates[:8]),atol=1e-6)
    assert meta['units']=='meters'
    with pytest.raises(ValueError):restored.predict(np.array([[999,0,0,0,0.]]))
    with pytest.raises(FileExistsError):save_scene_field(path,model,s,report)


def test_time_holdout_group():
    s=collect_samples([frame(),frame(),frame(1),frame(1)],units='meters',provenance='metric_estimated',reference_radius=.1,time_basis='capture_seconds')
    with pytest.raises(ValueError,match='TIME_GROUP'):fit_scene_field(s,holdout_views=[0],steps=1,split='time')
