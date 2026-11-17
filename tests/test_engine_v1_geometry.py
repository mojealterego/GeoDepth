import numpy as np
import pytest

from omega_geodepth.quality_v1.geometry import backproject_world, project_world, multiview_consistency
from omega_geodepth.quality_v1.prediction import Prediction


def make_prediction(n=2, h=3, w=4, depth=2.0):
    d = np.full((n, h, w), depth, dtype=np.float32)
    c = np.full_like(d, 0.75)
    k = np.repeat(np.array([[[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]]], dtype=np.float32), n, axis=0)
    e = np.repeat(np.array([[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]], dtype=np.float32), n, axis=0)
    return Prediction(d, c, e, k)


def test_backprojection_center_and_projection_roundtrip():
    p = make_prediction()
    world = backproject_world(p, 0)
    assert np.allclose(world[1, 1], [0., 0., 2.])
    u, v, z = project_world(world.reshape(-1, 3), p.extrinsics[0], p.intrinsics[0])
    y, x = np.indices((3, 4))
    assert np.allclose(u.reshape(3, 4), x)
    assert np.allclose(v.reshape(3, 4), y)
    assert np.allclose(z, 2.)


def test_extrinsics_are_world_to_camera_and_translation_is_respected():
    p = make_prediction(n=1)
    p.extrinsics[0, 0, 3] = 1.
    world = backproject_world(p, 0)
    assert np.allclose(world[1, 1], [-1., 0., 2.])


def test_identical_views_are_consistent():
    p = make_prediction()
    out = multiview_consistency(p, relative_tolerance=0.01)
    assert np.all(out['support'] == 1)
    assert np.all(out['tested'] == 1)
    assert np.allclose(out['score'], 1.)


def test_mismatched_depth_is_not_consistent():
    p = make_prediction()
    p.depth[1] *= 1.5
    out = multiview_consistency(p, relative_tolerance=0.05)
    assert np.all(out['support'][0] == 0)
    assert np.all(out['tested'][0] == 1)


def test_occluded_source_is_not_falsely_penalized():
    p = make_prediction()
    p.depth[1] = 1.0
    out = multiview_consistency(p, relative_tolerance=0.05)
    assert np.all(out['tested'][0] == 0)
    assert np.isnan(out['score'][0]).all()


def test_single_view_consistency_is_unknown_not_zero():
    out = multiview_consistency(make_prediction(n=1))
    assert not out['tested'].any()
    assert np.isnan(out['score']).all()


def test_invalid_intrinsics_are_rejected():
    p = make_prediction()
    p.intrinsics[0, 0, 0] = 0
    with pytest.raises(ValueError, match='intrinsics'):
        p.validate()


def test_invalid_rotation_is_rejected():
    p = make_prediction()
    p.extrinsics[0, 0, 0] = 10
    with pytest.raises(ValueError, match='rotation'):
        p.validate()


def test_depth_nan_is_masked_not_projected():
    p = make_prediction()
    p.depth[0, 0, 0] = np.nan
    xyz = backproject_world(p, 0)
    assert np.isnan(xyz[0, 0]).all()
