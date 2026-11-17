"""Integration tests against actual trained weights; should fail if distribution incomplete."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from omega_geodepth.infer import infer_image, load_verified_model
from omega_geodepth.train import dataset, evaluate

ROOT=Path(__file__).resolve().parents[1]

class TrainedModelTests(unittest.TestCase):
    def test_real_weights_improve_over_constant_baseline(self):
        net,meta=load_verified_model(ROOT/'models')
        x,y=dataset(16,80_000,64)
        torch.set_num_threads(2)
        metrics=evaluate(net,x,y)
        self.assertLess(metrics['log_depth_mae'], metrics['constant_log_depth_baseline_mae']*.6)

    def test_torchscript_matches_safetensors(self):
        net,_=load_verified_model(ROOT/'models')
        scripted=torch.jit.load(str(ROOT/'models'/_.get('torchscript_file','omega_geodepth_synthetic_v0.1.torchscript.pt'))).eval()
        torch.manual_seed(9)
        x=torch.rand(1,3,64,64)
        with torch.no_grad():
            py=net(x)
            sc=scripted(x)
        for expected,actual in zip(py,sc):
            self.assertTrue(torch.allclose(expected,actual,atol=1e-6))

    def test_end_to_end_safeguards_and_files(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            image=td/'input.png'
            Image.new('RGB',(80,80),'#aa5555').save(image)
            dest=td/'output'
            report=infer_image(image,dest,ROOT/'models',max_side=64)
            self.assertEqual(report['units'],'arbitrary_scale')
            self.assertEqual(report['provenance'],'relative_estimated')
            self.assertEqual(report['content_classification'],'not_performed')
            self.assertIn('conformal_calibration',report)
            self.assertTrue((dest/'pointcloud.ply').is_file())
            self.assertTrue((dest/'depth_preview.png').is_file())
            with np.load(dest/'depth_geometry.npz',allow_pickle=False) as archive:
                self.assertEqual(archive['depth_z'].shape,(64,64))
                self.assertEqual(archive['depth_lower_calibrated_synthetic'].shape,(64,64))
                self.assertTrue((archive['depth_upper_calibrated_synthetic']>=archive['depth_lower_calibrated_synthetic']).all())
            with (dest/'pointcloud.ply').open('rb') as f:
                head=f.read(200)
                self.assertIn(b'OCAF_GEODEPTH_OMEGA_UNITS=arbitrary_scale',head)

    def test_sha256_tampering_blocks_inference(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)
            shutil.copy(ROOT/'models'/'manifest.json',target/'manifest.json')
            m=json.loads((ROOT/'models'/'manifest.json').read_text())
            source=ROOT/'models'/m.get('weights_file','omega_geodepth_synthetic_v0.1.safetensors')
            corrupted=bytearray(source.read_bytes()); corrupted[-1]^=0x01
            (target/source.name).write_bytes(corrupted)
            with self.assertRaisesRegex(ValueError,'MODEL_SHA256_MISMATCH'):
                load_verified_model(target)

    def test_metric_anchor_requires_real_input(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            image=td/'scene.png'
            Image.new('RGB',(80,80),'#338888').save(image)
            dest=td/'unanchored'
            infer_image(image,dest,ROOT/'models',max_side=64,ply=False)
            with np.load(dest/'depth_geometry.npz',allow_pickle=False) as arr:
                predicted=arr['depth_z']
            anchors=td/'anchors.npz'
            mask=np.zeros_like(predicted,dtype=bool)
            mask[::5,::5]=True
            np.savez_compressed(anchors,depth_m=predicted*3,valid_mask=mask)
            report=infer_image(image,td/'anchored',ROOT/'models',max_side=64,anchors_path=anchors)
            self.assertEqual(report['units'],'meters')
            self.assertEqual(report['provenance'],'scale_aligned')
            self.assertAlmostEqual(report['metric_scale_factor'],3,places=5)

if __name__=='__main__':unittest.main()
