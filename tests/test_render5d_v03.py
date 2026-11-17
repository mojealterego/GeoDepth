"""Deterministic observed-only rendering of time-indexed point fields."""
import unittest
import numpy as np
from omega_geodepth.render5d import select_scene_time,render_scene5d
from omega_geodepth.reconstruction5d import Scene5D

def scene():
    xyz=np.array([[0,0,2],[0,0,4],[.2,0,2],[.2,0,4]],np.float32)
    return Scene5D(xyz,np.array([0.,0.,1.,1.]),
                   np.array([.4,.6,.4,.6],np.float32),
                   np.array([[255,0,0],[0,0,255],[255,0,0],[0,0,255]],np.uint8),
                   np.ones(4,np.float32),
                   np.array([[np.nan]*3,[np.nan]*3,[.1,0,0],[np.nan]*3],np.float32),
                   np.array([False,False,True,False]),
                   np.array([0,0,1,1],np.int32),
                   np.ones(4,np.int32),'arbitrary_scale','relative_estimated',.05,.2)

class RenderingTests(unittest.TestCase):
    def test_depth_zbuffer_prefers_nearest(self):
        s=scene()
        K=np.array([[10,0,5],[0,10,5],[0,0,1]],float)
        image,depth,mask=render_scene5d(s,0.,K,np.eye(4),11,11)
        self.assertTrue(mask[5,5])
        self.assertAlmostEqual(float(depth[5,5]),2)
        self.assertEqual(image[5,5,0],255)
        self.assertFalse(mask[0,0])
        self.assertTrue(np.isnan(depth[0,0]))

    def test_interpolation_only_moves_matched_points(self):
        s=scene()
        pts,col,conf=select_scene_time(s,.5)
        self.assertEqual(pts.shape,(2,3))
        self.assertAlmostEqual(float(pts[0,0]),.15,places=4)
        self.assertAlmostEqual(float(pts[1,0]),.2,places=4)

    def test_invalid_time_or_pose_rejected(self):
        s=scene();K=np.eye(3);K[0,0]=10;K[1,1]=10
        with self.assertRaisesRegex(ValueError,'TIME_OUTSIDE'):
            select_scene_time(s,20.)
        with self.assertRaisesRegex(ValueError,'RENDER_POSE'):
            render_scene5d(s,0.,K,np.zeros((4,4)),10,10)

if __name__=='__main__':unittest.main()

class RenderCLIIntegrationTests(unittest.TestCase):
    def test_saved_intermediate_png_and_mask(self):
        import tempfile
        from pathlib import Path
        from omega_geodepth.reconstruction5d import Frame5D,build_scene5d,save_scene5d
        k=np.array([[10,0,3.5],[0,10,3.5],[0,0,1]],float)
        im=np.ones((8,8,3),dtype=np.uint8)*125
        frames=[Frame5D(t,np.ones((8,8),np.float32)*2,im,k,np.eye(4),np.ones((8,8),bool))
                for t in (0.,1.)]
        scene_out=build_scene5d(frames,voxel_size=.1,neighbor_radius=.2)
        with tempfile.TemporaryDirectory() as td:
            save_scene5d(scene_out,Path(td),frames,render_intervals=2)
            self.assertTrue((Path(td)/'renders'/'between_0000_0001.png').is_file())
            self.assertTrue((Path(td)/'renders'/'between_0000_0002.png').is_file())
            self.assertTrue((Path(td)/'renders'/'between_0000_0001.npz').is_file())
