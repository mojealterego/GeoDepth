import tempfile
from pathlib import Path
import unittest
import json
import numpy as np
from omega_geodepth.calibration import conformal_quantile, load_verified_calibration

class CalibrationTests(unittest.TestCase):
    def test_quantile_exceeds_majority_scores(self):
        residuals=np.arange(1,101,dtype=np.float64)
        q=conformal_quantile(residuals,alpha=.1)
        self.assertGreaterEqual(q,91)

    def test_requires_matching_model_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            (p/'calibration.json').write_text(json.dumps({'model_sha256':'wrong','alpha':.1,'q':1.2}))
            with self.assertRaisesRegex(ValueError,'CALIBRATION_MODEL_MISMATCH'):
                load_verified_calibration(p,{'weights_sha256':{'omega_geodepth_synthetic_v0.1.safetensors':'right'}})

    def test_quantile_rejects_invalid(self):
        with self.assertRaises(ValueError): conformal_quantile(np.array([0,np.nan]),alpha=.1)

if __name__=='__main__':unittest.main()
