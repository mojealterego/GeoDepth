import numpy as np
import pytest
from omega_geodepth.reconstruction5d import Frame5D
from omega_geodepth.scale5d_bridge import lift_metric_sequence, fit_metric_scale_field
from representations4d.fived.field import Axis5D, GridSpec5D

def frame(t=0., trans=0.):
    p=np.eye(4); p[0,3]=trans
    return Frame5D(t,np.full((2,2),2.),np.full((2,2,3),128,dtype=np.uint8),np.array([[100.,0,0],[0,100.,0],[0,0,1]]),p,np.ones((2,2),bool))

def test_metric_lift():
    b=lift_metric_sequence([frame(),frame(1,1)],units='meters',provenance='scale_aligned',reference_radius_m=.01)
    np.testing.assert_allclose(b.coordinates[0],[0,0,2,0,1]); np.testing.assert_allclose(b.coordinates[4],[1,0,2,1,1]); np.testing.assert_allclose(b.values,128/255.)

def test_relative_rejected():
    with pytest.raises(ValueError): lift_metric_sequence([frame()],units='arbitrary_scale',provenance='relative_estimated',reference_radius_m=.01)

def test_skew_rejected():
    f=frame(); f.intrinsics[0,1]=1
    with pytest.raises(ValueError): lift_metric_sequence([f],units='meters',provenance='scale_aligned',reference_radius_m=.01)

def test_budget_and_time():
    for frames,budget in [([frame(),frame()],10),([frame(),frame(1)],3)]:
        with pytest.raises((ValueError,MemoryError)): lift_metric_sequence(frames,units='meters',provenance='metric_measured',reference_radius_m=.01,max_samples=budget)

def test_fit():
    grid=GridSpec5D(tuple(Axis5D(a,s,c) for a,s,c in [(-1,1,4),(-1,1,4),(1,1,3),(0,1,2),(0,1,3)]),.01)
    b=lift_metric_sequence([frame(),frame(1)],units='meters',provenance='metric_measured',reference_radius_m=.01)
    f=fit_metric_scale_field(b,grid,reference_radius_m=.01)
    out=f.query(b.coordinates)
    assert out.supported.all(); np.testing.assert_allclose(out.values,b.values)

def test_radius_mismatch():
    grid=GridSpec5D(tuple(Axis5D(0,1,4) for _ in range(5)),.02)
    b=lift_metric_sequence([frame()],units='meters',provenance='metric_measured',reference_radius_m=.01)
    with pytest.raises(ValueError,match='REFERENCE_RADIUS_MISMATCH'):
        fit_metric_scale_field(b,grid,reference_radius_m=.01)

def test_invalid_mask_rejected():
    f=frame(); f.valid=f.valid.astype(np.uint8)
    with pytest.raises(ValueError,match='BOOLEAN_VALIDITY_REQUIRED'):
        lift_metric_sequence([f],units='meters',provenance='metric_measured',reference_radius_m=.01)
