"""Contract tests written before v0.4 multiview TSDF and dynamic rendering."""
import unittest
import numpy as np
from omega_geodepth.reconstruction5d import Frame5D, build_scene5d
from omega_geodepth.render5d import select_scene_time
from omega_geodepth.multiview_tsdf import integrate_multiview_tsdf


def flat(t, shift=0, d=2, h=32, w=32, invalid=False):
    k=np.array([[30.,0,(w-1)/2],[0,30.,(h-1)/2],[0,0,1]])
    pose=np.eye(4);pose[0,3]=shift
    rgb=np.ones((h,w,3),dtype=np.uint8)*150
    valid=np.ones((h,w),dtype=bool)
    if invalid:valid[:]=False
    return Frame5D(float(t),np.full((h,w),d,dtype=np.float32),rgb,k,pose,valid)

class FusionTest(unittest.TestCase):
    def test_multiview_observations_accumulate_in_world_frame(self):
        grid=integrate_multiview_tsdf([flat(0),flat(1,shift=.05)],voxel_size=.1,
                                      truncation=.3,max_voxels=200000)
        self.assertGreater(int((grid.weights>=2).sum()),100)
        self.assertGreater(int((grid.weights>0).sum()),int((grid.weights>=2).sum()))
        self.assertTrue(np.all(grid.tsdf[grid.weights>0]<=1.))
        self.assertTrue(np.all(grid.tsdf[grid.weights>0]>=-1.))

    def test_invalid_measurements_not_observations(self):
        grid=integrate_multiview_tsdf([flat(0),flat(1,invalid=True)],voxel_size=.15,
                                      truncation=.3,max_voxels=200000)
        self.assertEqual(int((grid.weights>=2).sum()),0)
        self.assertGreater(int((grid.weights==1).sum()),100)

    def test_no_unsafe_fusion_without_calibration(self):
        f=flat(0);f.pose_c2w[:3,:3]=2*np.eye(3)
        with self.assertRaisesRegex(ValueError,'POSE|SE3'):
            integrate_multiview_tsdf([f,flat(1)])

    def test_static_mask_excludes_dynamic_object(self):
        a,b=flat(0),flat(1)
        a.static_mask=np.ones((32,32),bool)
        b.static_mask=np.zeros((32,32),bool)
        grid=integrate_multiview_tsdf([a,b],voxel_size=.15,truncation=.3)
        self.assertEqual(int((grid.weights>=2).sum()),0)

    def test_fusion_does_not_mutate_input_frame_valid_mask(self):
        a,b=flat(0),flat(1)
        b.static_mask=np.zeros((32,32),dtype=bool)
        expected=b.valid.copy()
        integrate_multiview_tsdf([a,b],voxel_size=.15,truncation=.3)
        np.testing.assert_array_equal(b.valid,expected)

    def test_timestamp_strict(self):
        with self.assertRaisesRegex(ValueError,'TIMESTAMP'):
            integrate_multiview_tsdf([flat(1),flat(1)])

    def test_resource_budget_no_allocation(self):
        with self.assertRaisesRegex(ValueError,'RESOURCE'):
            integrate_multiview_tsdf([flat(0),flat(1)],voxel_size=.0001,max_voxels=100)

class SceneFlowTest(unittest.TestCase):
    def test_static_scene_interpolation_temporal_symmetric(self):
        scene=build_scene5d([flat(0),flat(1)],voxel_size=.08,neighbor_radius=.2)
        xyz0,_,_=select_scene_time(scene,0)
        xyz1,_,_=select_scene_time(scene,1)
        xyzhalf,_,_=select_scene_time(scene,.5)
        # Scene positions should interpolate rather than pop to next frame.
        self.assertGreater(len(xyzhalf),0)
        self.assertLess(abs(np.median(xyzhalf[:,2])-2),.01)

    def test_mutual_matching_prevents_many_to_one_collapse(self):
        a=flat(0,h=6,w=6)
        b=flat(1,h=6,w=6)
        b.depth_z[1:,:]=0
        b.valid[1:,:]=False
        scene=build_scene5d([a,b],voxel_size=.01,neighbor_radius=.5)
        self.assertLessEqual(int((scene.matches_valid & (scene.frame_index==1)).sum()),int(b.valid.sum()))
        self.assertEqual(int((scene.matches_valid & (scene.frame_index==0)).sum()),
                         int((scene.matches_valid & (scene.frame_index==1)).sum()))

if __name__=='__main__':unittest.main()

class PipelineV04Tests(unittest.TestCase):
    def test_static_three_view_fusion_has_correct_observed_surface(self):
        from omega_geodepth.tsdf5d import extract_surface_mesh
        frames=[flat(0),flat(.1,shift=.05),flat(.2,shift=-.05)]
        grid=integrate_multiview_tsdf(frames,voxel_size=.1,truncation=.3)
        vertices,faces=extract_surface_mesh(grid)
        self.assertGreater(len(faces),500)
        self.assertLess(float(np.mean(np.abs(vertices[:,2]-2.))),.03)
        self.assertGreater(int((grid.weights>=3).sum()),500)

    def test_atomic_export_end_to_end_fusion_and_gaussian_renders(self):
        import tempfile,hashlib
        from pathlib import Path
        from omega_geodepth.reconstruction5d import save_scene5d
        frames=[flat(0,h=16,w=16),flat(.1,shift=.05,h=16,w=16),flat(.2,shift=-.05,h=16,w=16)]
        scene=build_scene5d(frames,voxel_size=.08,neighbor_radius=.2)
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)
            report=save_scene5d(scene,dest,frames,mesh=True,tsdf=True,
                                tsdf_voxel_size=.12,tsdf_truncation=.35,
                                fused_static_tsdf=True,gaussian_splats=True,render_intervals=1)
            self.assertGreater(report['fused_static_tsdf']['multiview_voxels'],0)
            self.assertEqual(report['gaussian5d']['primitive_count'],len(scene.xyz))
            self.assertTrue((dest/'gaussian5d.npz').is_file())
            self.assertTrue((dest/'static_fused_tsdf.obj').is_file())
            for rel,expected in report['artifacts_sha256'].items():
                self.assertEqual(hashlib.sha256((dest/rel).read_bytes()).hexdigest(),expected)
