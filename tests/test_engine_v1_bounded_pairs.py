import numpy as np
import pytest
from omega_geodepth.quality_v1.geometry import multiview_consistency
from test_engine_v1_geometry import make_prediction


def test_large_sequence_uses_bounded_deterministic_target_selection():
    p=make_prediction(n=12,h=2,w=2)
    a=multiview_consistency(p,max_targets_per_view=4)
    b=multiview_consistency(p,max_targets_per_view=4)
    assert np.array_equal(a['tested'],b['tested'])
    assert np.all(a['tested']==4)
    assert np.all(a['support']==4)


def test_target_budget_rejects_zero():
    with pytest.raises(ValueError,match='max_targets'):
        multiview_consistency(make_prediction(),max_targets_per_view=0)
