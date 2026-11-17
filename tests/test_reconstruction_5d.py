"""5D field is (x,y,z,time,appearance proxy); NOT five spatial dimensions."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from omega_geodepth.reconstruction5d import Frame5D, build_scene5d, save_scene5d, load_frame_manifest


def frame(t, *, camera_x=0.0, depth=2., size=(8,8)):
    h,w=size
    rgb=np.full((h,w,3),128,dtype=np.uint8)
    k=np.array([[10,0,(w-1)/2],[0,10,(h-1)/2],[0,0,1]],dtype=np.float64)
    pose=np.eye(4,dtype=np.float64); pose[0,3]=camera_x
    return Frame5D(t,np.full((h,w),depth,np.float32),rgb,k,pose,
                   np.ones((h,w),dtype=bool),None)

class Reconstruct5DTests(unittest.TestCase):
    def test_real_nonempty_5d_fields_and_velocity(self):
        scene=build_scene5d([frame(0),frame(0.2)],voxel_size=0.05,depth_jump_rel=.5,neighbor_radius=.16)
        self.assertGreater(len(scene.xyz),0)
        self.assertEqual(scene.xyz.shape[1],3)
        self.assertEqual(scene.luma.shape,(len(scene.xyz),))
        self.assertTrue(np.isin(scene.time_s,[0.,.2]).all())
        self.assertTrue(np.all(np.isfinite(scene.xyz)))
        self.assertGreater(scene.matches_valid.sum(),0)
        self.assertLess(np.median(np.linalg.norm(scene.velocity[scene.matches_valid],axis=1)),1e-3)
        self.assertEqual(scene.units,'arbitrary_scale')
        self.assertEqual(scene.provenance,'relative_estimated')

    def test_moving_camera_pose_is_applied(self):
        scene=build_scene5d([frame(0),frame(.5,camera_x=1.)],voxel_size=.01,
                            neighbor_radius=.5)
        x0=scene.xyz[scene.frame_index==0,0].mean()
        x1=scene.xyz[scene.frame_index==1,0].mean()
        self.assertAlmostEqual(float(x1-x0),1,delta=.03)

    def test_timestamp_order_is_enforced(self):
        with self.assertRaisesRegex(ValueError,'TIMESTAMP'):
            build_scene5d([frame(.5),frame(.2)])

    def test_incompatible_metric_provenance_denied(self):
        with self.assertRaisesRegex(ValueError,'METRIC'):
            build_scene5d([frame(0),frame(.1)],units='meters',provenance='relative_estimated')

    def test_bad_calibration_denied(self):
        f=frame(0);f.intrinsics[0,0]=0
        with self.assertRaisesRegex(ValueError,'INTRINSICS'):
            build_scene5d([f,frame(.1)])

    def test_meshes_are_real_triangle_faces_no_spanning_depth_jumps(self):
        from omega_geodepth.reconstruction5d import depth_surface_mesh
        f=frame(0,size=(4,4));f.depth_z[:,2:]=20
        verts,tri,cols=depth_surface_mesh(f,max_rel_discontinuity=.2)
        self.assertGreater(len(tri),0)
        # Two independent surfaces: faces never span left-depth=2/right-depth=20.
        # The pose-correct 3D X may cross x=1; only z labels identify the seam.
        for face in tri:
            z=verts[face,2]
            self.assertTrue(np.all(z==2.) or np.all(z==20.))

    def test_manifest_paths_restricted_to_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            (p/'input.json').write_text(json.dumps({'units':'arbitrary_scale','frames':[
                {'time_s':0,'npz':'../private.npz','pose_c2w':np.eye(4).tolist()},
                {'time_s':1,'npz':'../private.npz','pose_c2w':np.eye(4).tolist()}]}))
            with self.assertRaisesRegex(ValueError,'UNSAFE_INPUT_PATH'):
                load_frame_manifest(p/'input.json')

    def test_end_to_end_npz_ply_mesh_and_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            scene=build_scene5d([frame(0),frame(.1)],voxel_size=.1,neighbor_radius=.3)
            manifest=save_scene5d(scene,p,[frame(0),frame(.1)],mesh=True)
            self.assertGreater(manifest['total_points'],0)
            self.assertTrue((p/'scene5d.npz').is_file())
            self.assertTrue((p/'scene5d.ply').is_file())
            self.assertTrue((p/'meshes'/'frame_0000.obj').is_file())
            self.assertTrue((p/'report.json').is_file())
            with np.load(p/'scene5d.npz',allow_pickle=False) as z:
                self.assertEqual(z['features_5d'].shape[1],5)
                self.assertEqual(z['xyz'].shape[0],manifest['total_points'])

if __name__=='__main__':unittest.main()
