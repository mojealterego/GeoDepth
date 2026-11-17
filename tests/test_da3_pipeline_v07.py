"""Contract test: realistic multiview geometry through DA3 boundary to 5D.

DA3 model itself is replaced with a labelled synthetic contract fixture. This
is NOT a pretrained DA3 inference test and is not reported as such.
"""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
import numpy as np
from omega_geodepth.da3_adapter import DA3Output
from omega_geodepth.reconstruction5d import build_from_da3_sequence,build_scene5d,save_scene5d

class DA3PipelineContractTests(unittest.TestCase):
    @staticmethod
    def output(bad_depth=False):
        n=3;h=24;w=32
        depth=np.full((n,h,w),2.,dtype=np.float32)
        if bad_depth:depth[1]=3.5
        k=np.repeat(np.array([[[40.,0,15.5],[0,40,11.5],[0,0,1]]]),n,axis=0)
        poses=np.repeat(np.eye(4)[None],n,axis=0)
        poses[1,0,3]=.03;poses[2,0,3]=.06
        rgb=np.zeros((n,h,w,3),dtype=np.uint8)
        rgb[:,:,:,0]=150
        return DA3Output(depth,np.ones_like(depth),k,poses,rgb,
            'arbitrary_scale','relative_estimated','DA3-SMALL','a'*64)

    @staticmethod
    def setup_scene(td):
        root=Path(td)
        frames=[]
        for i in range(3):
            name=f'frame{i}.png';(root/name).write_bytes(b'fixture')
            frames.append({'image':name,'time_s':i*.3})
        manifest=root/'inputs.json';manifest.write_text(json.dumps({'frames':frames}))
        return manifest

    def test_from_da3_contract_to_5d_artifacts_with_reprojection_gate(self):
        with tempfile.TemporaryDirectory() as td:
            inp=self.setup_scene(td)
            class FakeDA3:
                def __init__(self,*args,**kwargs):pass
                def inference(self,images,*,process_res):
                    self_tests=DA3PipelineContractTests
                    assert len(images)==3
                    return self_tests.output()
            with patch('omega_geodepth.da3_adapter.DA3OfflineBackend',FakeDA3):
                frames,meta=build_from_da3_sequence(inp,Path(td),'DA3-SMALL',
                    process_res=256,verify_reprojection=True)
            self.assertEqual(meta['multiview_reprojection_audit']['passed'],True)
            self.assertEqual(meta['units'],'arbitrary_scale')
            scene=build_scene5d(frames,voxel_size=.05)
            output=Path(td)/'output'
            report=save_scene5d(scene,output,frames,mesh=True)
            self.assertGreater(report['total_points'],10)
            self.assertEqual(report['units'],'arbitrary_scale')
            self.assertTrue((output/'scene5d.ply').is_file())

    def test_bad_depth_causes_gate_to_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            inp=self.setup_scene(td)
            class FakeDA3:
                def __init__(self,*args,**kwargs):pass
                def inference(self,images,*,process_res):
                    return DA3PipelineContractTests.output(bad_depth=True)
            with patch('omega_geodepth.da3_adapter.DA3OfflineBackend',FakeDA3):
                with self.assertRaisesRegex(ValueError,'DA3_MULTIVIEW_REPROJECTION_UNVERIFIABLE'):
                    build_from_da3_sequence(inp,Path(td),'DA3-SMALL',
                        process_res=256,verify_reprojection=True)

    def test_user_source_provenance_verification(self):
        from omega_geodepth.da3_source import extract_da3_source
        from omega_geodepth.da3_adapter import resolve_verified_da3_source
        src=Path(__file__).resolve().parents[1]/'vendor/Depth-Anything-3-main.zip'
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'official';extract_da3_source(src,out)
            self.assertTrue((resolve_verified_da3_source(out)/'depth_anything_3/api.py').is_file())
            (out/'src/depth_anything_3/api.py').write_text('HACK')
            with self.assertRaisesRegex(ValueError,'DA3_SOURCE_TAMPERED'):
                resolve_verified_da3_source(out)

if __name__=='__main__':unittest.main()

class DA3ImportCollisionTests(unittest.TestCase):
    def test_explicit_verified_source_rejects_shadowing_api(self):
        from omega_geodepth.da3_adapter import _load_official_da3_class
        from omega_geodepth.da3_source import extract_da3_source
        import sys,types
        archive=Path(__file__).resolve().parents[1]/'vendor/Depth-Anything-3-main.zip'
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'source';extract_da3_source(archive,source)
            fake=types.ModuleType('depth_anything_3.api')
            fake.__file__='/tmp/malicious_api.py'
            with patch.dict(sys.modules,{'depth_anything_3.api':fake}):
                with self.assertRaisesRegex(RuntimeError,'DA3_SOURCE_IMPORT_COLLISION'):
                    _load_official_da3_class(source)
