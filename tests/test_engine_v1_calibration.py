import numpy as np
import pytest
from omega_geodepth.quality_v1.calibration import calibrate_scale, apply_calibration
from test_engine_v1_geometry import make_prediction


def test_calibration_with_metric_anchors_scales_depth_and_camera_translation():
    p = make_prediction(n=1, h=4, w=4)
    p.extrinsics[0, 0, 3] = 3.
    anchors = [{'view': 0, 'x': x, 'y': y, 'depth_m': 8.} for y, x in [(0,0),(1,1),(2,2),(3,3)]]
    report = calibrate_scale(p, anchors)
    assert report['scale'] == pytest.approx(4.)
    new = apply_calibration(p, report)
    assert np.allclose(new.depth, 8.)
    assert new.extrinsics[0, 0, 3] == pytest.approx(12.)


def test_bad_metric_anchors_fail_closed():
    p = make_prediction(n=1, h=4, w=4)
    anchors = [{'view': 0, 'x': x, 'y': y, 'depth_m': v} for (y,x),v in zip([(0,0),(1,1),(2,2),(3,3)],[2,5,10,20])]
    with pytest.raises(ValueError, match='inconsistent'):
        calibrate_scale(p, anchors)


def test_fewer_than_four_anchors_rejected():
    with pytest.raises(ValueError, match='at least 4'):
        calibrate_scale(make_prediction(n=1), [{'view':0,'x':1,'y':1,'depth_m':2.}])


def test_out_of_bounds_anchors_rejected():
    with pytest.raises(ValueError, match='outside'):
        calibrate_scale(make_prediction(n=1), [{'view':0,'x':100,'y':1,'depth_m':2.}]*4)
