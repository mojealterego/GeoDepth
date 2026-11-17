import unittest
import numpy as np
from omega_geodepth.da3_adapter import DA3Output
from omega_geodepth.da3_validation import verify_multiview_reprojection


def make_scene(*,bad_depth=False,missing=False):
    n,h,w=2,48,64
    depth=np.full((n,h,w),2.,dtype=np.float32)
    if bad_depth: depth[1]=3.5
    if missing: depth[1]=np.nan
    k=np.repeat(np.array([[[50,2,31.5],[0,52,23.5],[0,0,1]]],dtype=float),n,axis=0)
    p=np.repeat(np.eye(4)[None],n,axis=0)
    p[1,0,3]=.05
    return DA3Output(depth,np.ones_like(depth),k,p,np.zeros((n,h,w,3),dtype=np.uint8),
                     'arbitrary_scale','relative_estimated','DA3-SMALL','a'*64)

class DA3MultiviewTests(unittest.TestCase):
    def test_perfect_consistency_across_camera_baseline_with_skew(self):
        result=verify_multiview_reprojection(make_scene(),min_samples=50)
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['pairs']),1)
        self.assertLess(result['pairs'][0]['median_relative_residual'],1e-6)
        self.assertGreater(result['pairs'][0]['coverage'],0.80)

    def test_geometric_disagreement_fails(self):
        result=verify_multiview_reprojection(make_scene(bad_depth=True),min_samples=50)
        self.assertFalse(result['passed'])
        self.assertEqual(result['pairs'][0]['reason'],'DEPTH_REPROJECTION_MISMATCH')

    def test_no_correspondences_abstains(self):
        result=verify_multiview_reprojection(make_scene(missing=True),min_samples=50)
        self.assertFalse(result['passed'])
        self.assertEqual(result['pairs'][0]['reason'],'INSUFFICIENT_VALID_OVERLAP')

    def test_invalid_thresholds_rejected(self):
        with self.assertRaisesRegex(ValueError,'INVALID_REPROJECTION_THRESHOLDS'):
            verify_multiview_reprojection(make_scene(),max_relative_error=-1)

    def test_skips_out_of_view_points_without_false_positive(self):
        out=make_scene()
        out.pose_c2w[1,0,3]=30.
        result=verify_multiview_reprojection(out,min_samples=50)
        self.assertFalse(result['passed'])
        self.assertEqual(result['pairs'][0]['reason'],'INSUFFICIENT_VALID_OVERLAP')

if __name__=='__main__':unittest.main()
