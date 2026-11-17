import tempfile
import unittest
from pathlib import Path
import numpy as np
from omega_geodepth.tsdf5d import integrate_tsdf_frame, extract_surface_mesh, write_tsdf_npz
from omega_geodepth.reconstruction5d import Frame5D

class TSDFTests(unittest.TestCase):
    def setUp(self):
        h=w=24
        self.k=np.array([[30.,0.,11.5],[0.,30.,11.5],[0.,0.,1.]])
        self.f=Frame5D(0.,np.full((h,w),2.,dtype=np.float32),
                np.ones((h,w,3),dtype=np.uint8)*135,self.k,np.eye(4),
                np.ones((h,w),dtype=bool))

    def test_signed_distance_plane(self):
        grid=integrate_tsdf_frame(self.f,voxel_size=.10,truncation=.25,max_voxels=200000)
        self.assertGreater(int(grid.weights.sum()),10)
        self.assertGreater(grid.tsdf[grid.weights>0].max(),0)
        self.assertLess(grid.tsdf[grid.weights>0].min(),0)
        verts,faces=extract_surface_mesh(grid)
        self.assertGreater(len(verts),0)
        self.assertGreater(len(faces),0)
        self.assertLess(float(np.median(np.abs(verts[:,2]-2))),.16)

    def test_reject_excessive_grid(self):
        with self.assertRaisesRegex(ValueError,'TSDF_RESOURCE_LIMIT'):
            integrate_tsdf_frame(self.f,voxel_size=.001,truncation=.25,max_voxels=20000)

    def test_save_real_npz(self):
        with tempfile.TemporaryDirectory() as td:
            grid=integrate_tsdf_frame(self.f,voxel_size=.10,truncation=.25,max_voxels=200000)
            path=Path(td)/'tsdf.npz';write_tsdf_npz(grid,path)
            with np.load(path,allow_pickle=False) as d:
                self.assertEqual(d['tsdf'].ndim,3)
                self.assertEqual(d['weights'].shape,d['tsdf'].shape)

if __name__=='__main__':unittest.main()
