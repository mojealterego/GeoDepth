"""CPU Gaussian scene-field rasterizer contracts (red before implementation)."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from omega_geodepth.gaussian5d import GaussianField5D, gaussian_from_scene, render_gaussian5d, export_gaussian5d
from omega_geodepth.reconstruction5d import Scene5D


def miniature():
    xyz=np.array([[0,0,3],[0.1,0,3],[0,0,5]],dtype=np.float32)
    return Scene5D(xyz,np.array([0.,0.,0.]),np.array([1.,0.5,.1],np.float32),
        np.array([[255,0,0],[0,255,0],[0,0,255]],np.uint8),
        np.ones(3,np.float32),np.full((3,3),np.nan,np.float32),np.zeros(3,bool),
        np.zeros(3,np.int32),np.ones(3,np.int32),'arbitrary_scale','relative_estimated',.05,.2)

class Gaussian5DTests(unittest.TestCase):
    def test_raster_support_and_depth_occlusion(self):
        gs=gaussian_from_scene(miniature(),sigma_world=.1)
        K=np.array([[35.,0,8.],[0,35.,8.],[0,0,1]])
        rgb,alpha,depth=render_gaussian5d(gs,0.,K,np.eye(4),17,17)
        self.assertEqual(rgb.shape,(17,17,3))
        self.assertGreater(alpha[8,8],.3)
        self.assertAlmostEqual(float(depth[8,8]),3.,delta=.01)
        self.assertGreater(int(rgb[8,8,0]),int(rgb[8,8,2]))
        self.assertTrue(np.isnan(depth[0,0]))

    def test_export_keeps_units_times_and_field_hash(self):
        gs=gaussian_from_scene(miniature())
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'gaussians.npz'
            export_gaussian5d(gs,path)
            with np.load(path,allow_pickle=False) as f:
                self.assertEqual(f['means_xyz'].shape,(3,3))
                self.assertEqual(str(f['units']),'arbitrary_scale')

    def test_no_intervals_outside_actual_observations(self):
        gs=gaussian_from_scene(miniature())
        with self.assertRaisesRegex(ValueError,'TIME_OUTSIDE'):
            render_gaussian5d(gs,2,np.eye(3),np.eye(4),17,17)

    def test_points_resource_limit_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'GAUSSIAN_RESOURCE'):
            gaussian_from_scene(miniature(),max_points=1)

if __name__=='__main__':unittest.main()
