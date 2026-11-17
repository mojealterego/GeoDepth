import unittest
import numpy as np

from omega_geodepth.geometry import (
    CameraIntrinsics, optical_z_to_range, range_to_optical_z,
    scaled_intrinsics, anchored_scale, export_ply,
)
from omega_geodepth.model import OmegaDepthNet
from omega_geodepth.synthetic import generate_scene

class GeometryTests(unittest.TestCase):
    def test_pixel_centers_after_roi_scale(self):
        k = CameraIntrinsics(1200, 1180, 600, 400, 2)
        s = scaled_intrinsics(k, offset_x=30, offset_y=20, scale_x=.5, scale_y=.75)
        self.assertAlmostEqual(s.fx, 600)
        self.assertAlmostEqual(s.fy, 885)
        self.assertAlmostEqual(s.cx, .5*(600-30+.5)-.5)
        self.assertAlmostEqual(s.cy, .75*(400-20+.5)-.5)
        self.assertAlmostEqual(s.skew, 1)

    def test_range_conversion_preserves_geometry(self):
        k = CameraIntrinsics(500, 500, 500, 500)
        z = np.ones((1000, 1000), dtype=np.float32) * 7
        r = optical_z_to_range(z, k)
        self.assertAlmostEqual(float(r[0, 0]), 7*np.sqrt(3), places=4)
        np.testing.assert_allclose(range_to_optical_z(r, k), z, atol=2e-6)

    def test_anchor_rejects_invalid(self):
        with self.assertRaisesRegex(ValueError, 'ANCHOR'):
            anchored_scale(np.ones((5,5)), np.zeros((5,5)), np.ones((5,5), dtype=bool))

    def test_anchor_robust_against_outlier(self):
        pred=np.ones((20,20),np.float32)*2
        obs=np.ones((20,20),np.float32)*6
        obs[0,0] = 1000
        result = anchored_scale(pred, obs, np.ones((20,20),bool))
        self.assertLess(abs(result - 3.0), .08)

    def test_ply_refuses_fake_metric_units(self):
        img=np.zeros((3,4,3), dtype=np.uint8)
        depth=np.ones((3,4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'METRIC_ANCHOR_REQUIRED'):
            export_ply(depth,img,CameraIntrinsics(4,4,2,2),units='meters', provenance='relative_estimated', metric_anchor_validated=False)

    def test_ply_contains_provenance(self):
        img=np.zeros((3,4,3), dtype=np.uint8)
        depth=np.ones((3,4), dtype=np.float32)
        p=export_ply(depth,img,CameraIntrinsics(4,4,2,2),units='arbitrary_scale',provenance='relative_estimated')
        self.assertIn(b'OCAF_GEODEPTH_OMEGA_UNITS=arbitrary_scale',p)
        self.assertIn(b'element vertex 12',p)

class ModelTests(unittest.TestCase):
    def test_network_real_forward(self):
        import torch
        m=OmegaDepthNet(width=12).eval()
        with torch.no_grad():
            z,v=m(torch.rand(2,3,64,80))
        self.assertEqual(tuple(z.shape),(2,1,64,80))
        self.assertEqual(tuple(v.shape),(2,1,64,80))
        self.assertTrue(torch.isfinite(z).all().item())
        self.assertTrue(torch.isfinite(v).all().item())
        self.assertLess(float(z.mean(dim=(2,3)).abs().max()),1e-4)

    def test_synthetic_ground_truth_raycast(self):
        rgb,depth=generate_scene(17,size=(64,64))
        self.assertEqual(rgb.shape,(64,64,3))
        self.assertEqual(depth.shape,(64,64))
        self.assertTrue(np.isfinite(depth).all())
        self.assertGreater(float(depth.min()),0)
        self.assertGreater(float(depth.max()-depth.min()),0.2)

if __name__=='__main__': unittest.main()
