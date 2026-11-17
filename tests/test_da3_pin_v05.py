"""TDD for official HF checkpoint digest pinning and no symlink escapes."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
import torch
from safetensors.torch import save_file

class DA3PinnedTests(unittest.TestCase):
    def test_official_digest_rejects_self_attested_fake_weights(self):
        from omega_geodepth.da3_adapter import register_da3_checkpoint
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            (p/'config.json').write_text(json.dumps({'model_name':'da3-small',
                'config':{'__object__':{'path':'depth_anything_3.model.da3','name':'DepthAnything3Net'}}}))
            save_file({'fake':torch.ones(1)},str(p/'model.safetensors'))
            with self.assertRaisesRegex(ValueError,'DA3_OFFICIAL_HASH_MISMATCH'):
                register_da3_checkpoint(p,'DA3-SMALL','test')
            self.assertFalse((p/'omega_da3_manifest.json').exists())

    def test_official_hash_pinning_in_final_validation(self):
        from omega_geodepth.da3_adapter import validate_da3_checkpoint,file_sha256
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            (p/'config.json').write_text(json.dumps({'model_name':'da3-small',
                'config':{'__object__':{'path':'depth_anything_3.model.da3','name':'DepthAnything3Net'}}}))
            save_file({'fake':torch.ones(1)},str(p/'model.safetensors'))
            (p/'omega_da3_manifest.json').write_text(json.dumps({'repo_id':'depth-anything/DA3-SMALL',
                'license':'apache-2.0','files':{'model.safetensors':file_sha256(p/'model.safetensors'),
                'config.json':file_sha256(p/'config.json')}}))
            with self.assertRaisesRegex(ValueError,'DA3_OFFICIAL_HASH_MISMATCH'):
                validate_da3_checkpoint(p,'DA3-SMALL')

    def test_reject_checkpoint_symlink_outside_root(self):
        from omega_geodepth.da3_adapter import register_da3_checkpoint
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);p=d/'model';p.mkdir()
            (p/'config.json').write_text(json.dumps({'model_name':'da3-small',
                'config':{'__object__':{'path':'depth_anything_3.model.da3','name':'DepthAnything3Net'}}}))
            save_file({'fake':torch.ones(1)},str(d/'outside.safetensors'))
            (p/'model.safetensors').symlink_to(d/'outside.safetensors')
            with self.assertRaisesRegex(ValueError,'DA3_UNSAFE_CHECKPOINT_PATH'):
                register_da3_checkpoint(p,'DA3-SMALL','test')

if __name__=='__main__':unittest.main()
