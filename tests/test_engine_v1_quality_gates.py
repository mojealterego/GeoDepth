import json
import numpy as np
import pytest
from omega_geodepth.quality_v1.pipeline import run_pipeline
from omega_geodepth.quality_v1.geometry import camera_baseline_ratio
from test_engine_v1_geometry import make_prediction


def test_identical_camera_positions_do_not_count_as_parallax():
    assert camera_baseline_ratio(make_prediction(n=2))==0.


def test_known_translation_produces_dimensionless_parallax():
    p=make_prediction(n=2)
    p.extrinsics[1,0,3]=1.
    assert camera_baseline_ratio(p)==pytest.approx(.5)


def test_minimum_valid_ratio_fails_closed_before_writing(tmp_path):
    p=make_prediction(n=1)
    p.depth[0,0,:]=np.nan
    dest=tmp_path/'out'
    with pytest.raises(ValueError,match='valid-depth ratio'):
        run_pipeline(p,dest,min_valid_ratio=.9)
    assert not dest.exists()


def test_minimum_consistency_requires_evidence(tmp_path):
    with pytest.raises(ValueError,match='consistency'):
        run_pipeline(make_prediction(n=1),tmp_path/'out',min_mean_consistency=.8)


def test_minimum_parallax_rejects_duplicate_camera(tmp_path):
    with pytest.raises(ValueError,match='parallax'):
        run_pipeline(make_prediction(n=2),tmp_path/'out',min_baseline_ratio=.01)


def test_quality_maps_are_written_only_when_requested(tmp_path):
    p=make_prediction(n=2)
    dest=tmp_path/'out'
    report=run_pipeline(p,dest,save_quality_maps=True)
    with np.load(dest/'quality.npz',allow_pickle=False) as data:
        assert data['support'].shape==(2,3,4)
        assert data['tested'].sum()==24
    assert report['artifacts']['quality.npz']
    assert 'no independent parallax' in ' '.join(report['warnings']).lower()


def test_output_has_completion_marker_and_no_staging_directory(tmp_path):
    output=tmp_path/'ready'
    run_pipeline(make_prediction(n=1),output)
    assert (output/'.complete').read_text().strip()=='OMEGA-GEODEPTH-0.12.0'
    assert not list(tmp_path.glob('.omega-geodepth-*'))
