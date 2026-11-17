import json
import unittest
import tempfile
from pathlib import Path
import numpy as np
from omega_geodepth.geometry import CameraIntrinsics, validate_pose_c2w, scaled_intrinsics
from omega_geodepth.infer import load_image, MAX_SOURCE_PIXELS

class IntegrityTests(unittest.TestCase):
    def test_invalid_camera_pose_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'INVALID_POSE_C2W_NOT_SE3'):
            pose=np.eye(4);pose[:3,:3]*=2
            validate_pose_c2w(pose)

    def test_negative_focal_rejected(self):
        with self.assertRaisesRegex(ValueError,'INVALID_INTRINSICS'):
            CameraIntrinsics(-5,100,50,50)

    def test_invalid_scaling_rejected(self):
        with self.assertRaisesRegex(ValueError,'INVALID_CROP_SCALE'):
            scaled_intrinsics(CameraIntrinsics(2,2,0,0),0,0,0.0,1.0)

    def test_source_image_24mp_ceiling(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'small.png'
            Image.new('RGB',(70,70)).save(p)
            _,rgb,dims,k=load_image(p,64)
            self.assertEqual(dims,(70,70))
            self.assertEqual(rgb.shape[0]%8,0)
            self.assertEqual(rgb.shape[1]%8,0)

if __name__=='__main__':unittest.main()
