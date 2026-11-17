import unittest
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'comfyui/custom_nodes/OCAF_GeoDepth_Omega'))
from nodes import OmegaGeoDepth, NODE_CLASS_MAPPINGS

class ComfyUITests(unittest.TestCase):
    def test_node_contract_exposed(self):
        self.assertIn('OCAF_Omega_GeoDepth',NODE_CLASS_MAPPINGS)
        self.assertEqual(OmegaGeoDepth.RETURN_TYPES,('IMAGE','MASK','STRING'))
    def test_batch_returns_correct_shape_with_real_checkpoint(self):
        image=torch.rand(2,72,88,3)
        out=OmegaGeoDepth().execute(image,model_directory=str(Path(__file__).resolve().parents[1]/'models'),
                                     processing='direct',tile_size=64,overlap=24,global_side=64)
        self.assertEqual(tuple(out[0].shape),tuple(image.shape))
        self.assertEqual(tuple(out[1].shape),(2,72,88))
        self.assertEqual(out[0].dtype,torch.float32)
        self.assertTrue(out[1].isfinite().all())
        self.assertIn('arbitrary_scale',out[2])

if __name__=='__main__': unittest.main()
