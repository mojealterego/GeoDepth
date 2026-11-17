import numpy as np
import pytest
from omega_geodepth.quality_v1.refinement import refine_depth_multiview
from test_engine_v1_geometry import make_prediction


def test_refinement_moves_agreeing_views_toward_consensus():
    p=make_prediction(n=2,h=3,w=4)
    p.extrinsics[1,0,3]=0.1
    p.depth[1]*=1.05
    out,stats=refine_depth_multiview(p,relative_tolerance=.1,blend=1.)
    assert out.depth[0,1,1]>p.depth[0,1,1]
    assert out.depth[1,1,1]<p.depth[1,1,1]
    assert stats['changed_pixels']>0
    assert np.allclose(p.depth[0],2.)


def test_refinement_rejects_no_parallax():
    p=make_prediction(n=2)
    out,stats=refine_depth_multiview(p)
    assert stats['status']=='skipped_no_parallax'
    assert np.array_equal(out.depth,p.depth)


def test_refinement_does_not_fuse_occluded_depth():
    p=make_prediction(n=2)
    p.extrinsics[1,0,3]=0.1
    p.depth[1]*=.5
    out,stats=refine_depth_multiview(p,relative_tolerance=.05,blend=1.)
    assert np.array_equal(out.depth,p.depth)
    assert stats['changed_pixels']==0


def test_refinement_blend_must_be_valid():
    with pytest.raises(ValueError,match='blend'):
        refine_depth_multiview(make_prediction(),blend=2.)
