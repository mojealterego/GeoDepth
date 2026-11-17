import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from omega_geodepth.geometry import checked_anchor_alignment
from omega_geodepth.tiling import tiled_predict
from omega_geodepth.infer import load_image
from omega_geodepth.calibration import conformal_quantile, scene_coverage_quantile


class TilingTests(unittest.TestCase):
    def test_constant_model_blending_has_no_seams(self):
        class Constant(torch.nn.Module):
            def forward(self, x):
                shp=(x.shape[0],1,x.shape[2],x.shape[3])
                return x.new_zeros(shp), x.new_zeros(shp)
        rgb=np.random.default_rng(43).random((128,176,3)).astype(np.float32)
        l,v=tiled_predict(Constant(),rgb,tile_size=64,overlap=24,global_side=64)
        self.assertEqual(l.shape,(128,176))
        self.assertTrue(np.isfinite(l).all())
        self.assertTrue(np.isfinite(v).all())
        np.testing.assert_allclose(l,0,atol=1e-6)

    def test_tiling_rejects_invalid_window(self):
        class Constant(torch.nn.Module):
            def forward(self,x):
                return x[:,0:1]*0,x[:,0:1]*0
        img=np.ones((64,64,3),np.float32)
        with self.assertRaisesRegex(ValueError,'INVALID_TILE'):
            tiled_predict(Constant(),img,tile_size=64,overlap=64,global_side=64)

    def test_tiling_runs_real_checkpoint(self):
        from omega_geodepth.infer import load_verified_model
        root=Path(__file__).resolve().parents[1]
        model,_=load_verified_model(root/'models')
        img=np.random.default_rng(8).random((96,136,3)).astype(np.float32)
        l,v=tiled_predict(model,img,tile_size=64,overlap=24,global_side=64)
        self.assertEqual(l.shape,(96,136))
        self.assertTrue(np.isfinite(l).all())
        self.assertTrue(np.isfinite(v).all())

class AnchorQualityTests(unittest.TestCase):
    def test_clustered_anchors_rejected(self):
        p=np.ones((64,64),dtype=np.float32)
        a=p*3
        mask=np.zeros_like(p,dtype=bool);mask[:3,:3]=True
        with self.assertRaisesRegex(ValueError,'ANCHOR_SPATIAL_COVERAGE'):
            checked_anchor_alignment(p,a,mask)

    def test_divergent_anchor_observations_rejected(self):
        p=np.ones((64,64),dtype=np.float32)
        rng=np.random.default_rng(9)
        a=(2.0*np.exp(rng.normal(0,1.2,size=p.shape))).astype(np.float32)
        with self.assertRaisesRegex(ValueError,'ANCHOR_RESIDUAL_TOO_LARGE'):
            checked_anchor_alignment(p,a,np.ones_like(p,dtype=bool))

    def test_quality_gate_robust_to_singular_outlier(self):
        p=np.ones((64,64),dtype=np.float32)
        a=p*3;a[0,0]=1000
        result=checked_anchor_alignment(p,a,np.ones_like(p,dtype=bool))
        self.assertAlmostEqual(result.scale,3,places=2)
        self.assertGreater(result.valid_points,4000)
        self.assertGreaterEqual(result.inlier_fraction,.95)

class InputSecurityTests(unittest.TestCase):
    def test_oversize_image_rejected_before_exif_processing(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            img=Path(td)/'large.png'
            Image.new('RGB',(5000,5000)).save(img)
            with patch('omega_geodepth.infer.ImageOps.exif_transpose',side_effect=AssertionError('decoded before validation')):
                with self.assertRaisesRegex(ValueError,'INPUT_RESOURCE_LIMIT'):
                    load_image(img,64)

class SceneCalibrationTests(unittest.TestCase):
    def test_nested_quantile_for_scene_coverage(self):
        scores=np.ones((20,4,4),dtype=np.float64)
        scores[-1,:,:]=20
        result=scene_coverage_quantile(scores,alpha_scene=.1,min_pixel_coverage=.90)
        self.assertGreaterEqual(result,1.)
        self.assertLessEqual(result,20.)

    def test_scene_calibration_rejects_flattened_array(self):
        with self.assertRaisesRegex(ValueError,'SCENE_SCORES_SHAPE'):
            scene_coverage_quantile(np.arange(50),alpha_scene=.1,min_pixel_coverage=.90)

if __name__=='__main__': unittest.main()

class PipelineV02Tests(unittest.TestCase):
    def test_tiled_export_has_explicit_processing_mode(self):
        from omega_geodepth.infer import infer_image
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'sample.png'
            Image.new('RGB',(136,88),'#986754').save(source)
            out=Path(td)/'output'
            result=infer_image(source,out,Path(__file__).resolve().parents[1]/'models',
                              max_side=128,tiled=True,tile_size=64,overlap=24,global_side=64)
            self.assertEqual(result['processing_mode'],'tiled_global_local')
            self.assertEqual(result['units'],'arbitrary_scale')
            self.assertEqual(result['pointcloud_status'],'exported')

class CalibrationIntegrityTests(unittest.TestCase):
    def test_modified_calibration_rejected_if_manifest_pins_hash(self):
        from omega_geodepth.calibration import load_verified_calibration
        import json
        with tempfile.TemporaryDirectory() as td:
            model_dir=Path(td)
            (model_dir/'calibration.json').write_text(json.dumps({'model_sha256':'abc','q':1.0,'alpha':.1}))
            with self.assertRaisesRegex(ValueError,'CALIBRATION_SHA256_MISMATCH'):
                load_verified_calibration(model_dir,{
                    'weights_sha256':{'omega_geodepth_synthetic_v0.1.safetensors':'abc'},
                    'calibration_sha256':'0000'})

class CalibrationScopeTests(unittest.TestCase):
    def test_tiled_output_does_not_claim_direct_64px_calibration(self):
        from omega_geodepth.infer import infer_image
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            src=td/'scene.png';Image.new('RGB',(144,144),'#777777').save(src)
            out=td/'out'
            report=infer_image(src,out,Path(__file__).resolve().parents[1]/'models',max_side=128,
                               tiled=True,tile_size=64,overlap=24,global_side=64,ply=False)
            self.assertEqual(report['conformal_calibration'],'not_available')
            with np.load(out/'depth_geometry.npz',allow_pickle=False) as data:
                self.assertEqual(data['depth_lower_calibrated_synthetic'].size,0)

    def test_larger_direct_output_does_not_claim_64px_calibration(self):
        from omega_geodepth.infer import infer_image
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            src=td/'scene.png';Image.new('RGB',(144,144),'#777777').save(src)
            out=td/'out'
            report=infer_image(src,out,Path(__file__).resolve().parents[1]/'models',max_side=128,ply=False)
            self.assertEqual(report['conformal_calibration'],'not_available')
