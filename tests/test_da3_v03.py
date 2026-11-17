"""First tests for DA3 offline contract; no downloaded checkpoint is assumed."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from omega_geodepth.da3_adapter import normalize_da3_prediction, DA3OfflineBackend, validate_da3_checkpoint


class MockPrediction:
    def __init__(self):
        self.depth=np.ones((2, 4, 6),dtype=np.float32)*2
        self.conf=np.ones((2, 4, 6),dtype=np.float32)*0.75
        self.extrinsics=np.stack([np.eye(4)[:3],np.eye(4)[:3]]).astype(np.float32)
        self.intrinsics=np.stack([np.array([[9.,0,2.],[0,9.,2.],[0,0,1.]])]*2)
        self.processed_images=np.ones((2,4,6,3),dtype=np.uint8)*128


class DA3ContractTests(unittest.TestCase):
    def test_normalize_official_world_to_camera(self):
        p=MockPrediction();p.extrinsics[1,0,3]=-1
        result=normalize_da3_prediction(p,model_id='DA3-SMALL',weights_sha256='a'*64)
        self.assertEqual(result.depth_z.shape,(2,4,6))
        self.assertEqual(result.pose_c2w.shape,(2,4,4))
        self.assertAlmostEqual(result.pose_c2w[1,0,3],1)
        self.assertEqual(result.units,'arbitrary_scale')
        self.assertEqual(result.provenance,'relative_estimated')
        self.assertEqual(result.rgb.dtype,np.uint8)

    def test_missing_or_bad_pose_fails_closed(self):
        p=MockPrediction();p.extrinsics[0,0,0]=3
        with self.assertRaisesRegex(ValueError,'INVALID_DA3_POSE'):
            normalize_da3_prediction(p,model_id='DA3-SMALL',weights_sha256='a'*64)

    def test_missing_confidence_rejected_not_fabricated(self):
        p=MockPrediction();p.conf=None
        with self.assertRaisesRegex(ValueError,'INVALID_DA3_CONFIDENCE'):
            normalize_da3_prediction(p,model_id='DA3-SMALL',weights_sha256='a'*64)

    def test_checkpoint_requires_real_weights_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'config.json').write_text('{}')
            with self.assertRaisesRegex((FileNotFoundError,ValueError),'DA3_'):
                validate_da3_checkpoint(p,'DA3-SMALL')

    def test_offline_backend_does_not_auto_download(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex((FileNotFoundError,ValueError),'DA3_'):
                DA3OfflineBackend(Path(td),'DA3-SMALL')

    def test_unknown_model_denied(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'DA3_MODEL_NOT_ALLOWED'):
                DA3OfflineBackend(Path(td),'DA3-GIANT-1.1')

if __name__ == '__main__':unittest.main()

class DA3SupplyChainTests(unittest.TestCase):
    def test_unapproved_object_import_is_rejected(self):
        from omega_geodepth.da3_adapter import validate_official_config
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'config.json'
            p.write_text(json.dumps({'model_name':'da3-small','config':{
                '__object__':{'path':'subprocess','name':'Popen','args':'as_params'}}}))
            with self.assertRaisesRegex(ValueError,'DA3_CONFIG_UNAPPROVED_OBJECT'):
                validate_official_config(p,'DA3-SMALL')

    def test_backend_checksum_pinned_to_local_files(self):
        from omega_geodepth.da3_adapter import validate_da3_checkpoint,file_sha256
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            (p/'config.json').write_text(json.dumps({'model_name':'da3-small','config':{
                '__object__':{'path':'depth_anything_3.model.da3','name':'DepthAnything3Net','args':'as_params'}}}))
            (p/'model.safetensors').write_bytes(b'corrupted')
            m={'repo_id':'depth-anything/DA3-SMALL','license':'apache-2.0',
               'files':{n:file_sha256(p/n) for n in ('config.json','model.safetensors')}}
            (p/'omega_da3_manifest.json').write_text(json.dumps(m))
            (p/'model.safetensors').write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError,'DA3_SHA256_MISMATCH'):
                validate_da3_checkpoint(p,'DA3-SMALL')

class DA3MaskTests(unittest.TestCase):
    def test_nonfinite_depth_is_marked_unknown_not_misrepresented_as_zero(self):
        p=MockPrediction();p.depth[0,0,0]=np.inf;p.depth[0,0,1]=0
        result=normalize_da3_prediction(p,model_id='DA3-SMALL',weights_sha256='a'*64)
        self.assertTrue(np.isnan(result.depth_z[0,0,0]))
        self.assertTrue(np.isnan(result.depth_z[0,0,1]))
        self.assertEqual(result.depth_z[0,0,2],2)

class DA3BackendSmokeWithFakeUpstream(unittest.TestCase):
    def test_registered_checkpoint_loads_official_api_contract_in_offline_mode(self):
        """Tests the integration boundary only; NEVER claims real DA3 accuracy."""
        import sys,types
        from safetensors.torch import save_file
        from omega_geodepth.da3_adapter import register_da3_checkpoint
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            (p/'config.json').write_text(json.dumps({'model_name':'da3-small','config':{
                '__object__':{'path':'depth_anything_3.model.da3','name':'DepthAnything3Net','args':'as_params'}}}))
            import torch
            save_file({'net.fake.weight':torch.zeros(2,2)},str(p/'model.safetensors'))
            from omega_geodepth.da3_adapter import OFFICIAL_WEIGHT_SHA256,file_sha256
            with patch.dict(OFFICIAL_WEIGHT_SHA256,{'DA3-SMALL':file_sha256(p/'model.safetensors')}):
                register_da3_checkpoint(p,'DA3-SMALL','fake-test-not-authentic')
            class FakeModel:
                @classmethod
                def from_pretrained(cls,where):
                    assert str(p)==str(where)
                    return cls()
                def to(self,device):
                    assert device=='cpu';return self
                def inference(self,**kwargs):
                    assert kwargs['process_res']==256
                    assert kwargs['ref_view_strategy']=='middle'
                    return MockPrediction()
            pkg=types.ModuleType('depth_anything_3')
            api=types.ModuleType('depth_anything_3.api')
            api.DepthAnything3=FakeModel
            with patch.dict(sys.modules,{'depth_anything_3':pkg,'depth_anything_3.api':api}):
                with patch.dict(OFFICIAL_WEIGHT_SHA256,{'DA3-SMALL':file_sha256(p/'model.safetensors')}):
                    backend=DA3OfflineBackend(p,'DA3-SMALL')
                out=backend.inference(['a.png','b.png'],process_res=256)
                self.assertEqual(out.depth_z.shape,(2,4,6))
                self.assertEqual(out.units,'arbitrary_scale')
                import os
                self.assertEqual(os.environ.get('HF_HUB_OFFLINE'),'1')
