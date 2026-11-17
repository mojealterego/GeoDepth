# Engineering roadmap — verified status 2026-11-17

## Implemented / locally tested

- Integrity and architecture validation of user-supplied DA3-BASE SafeTensors.
- Strict offline prediction contract and defensive NPZ ingestion.
- Projection/backprojection, visibility-aware multiview validation and bounded comparisons.
- Confidence quantile, deterministic sampling, voxel fusion and PLY export.
- Optional independent metric scale anchors, scale and camera-translation correction.
- Fail-closed quality gates, report provenance, optional quality maps and safe output publication.
- SHA-256-verified multipart distribution, reproducible local tests.
- Optional reprojection-compatible multiview depth consensus refinement (synthetic tests only).
- Ground-truth evaluation CLI with AbsRel, RMSE, delta-1 and SILog, including explicit scale-alignment mode.

## Not implemented / requires further evidence or infrastructure

1. **Real-model inference integration E2E:** install and pin official upstream DA3 code, run on CPU/GPU, compare expected tensor outputs and memory use. Blocked by unavailable upstream package/network in the execution environment.
2. **Accuracy improvements to learned weights:** curated datasets, training budget, ablation studies, independent validation and reproducible benchmark; no retraining has been done.
3. **Quantized Android deployment:** select runtime supported by DA3 architecture, benchmark quantization drift and latency on physical target hardware; no Android native binary has been produced.
4. **Calibrated uncertainty:** learn/calibrate reliability on held-out ground truth; raw `conf` and geometric consistency are not calibrated probabilities.
5. **Dynamic-scene geometry:** segmentation/tracking and temporal filtering validated against real motion; no dynamic object handling beyond reprojection is implemented.
6. **True multiview global optimization:** robust bundle adjustment and dense depth refinement against held-out references; not present.
7. **Photogrammetric accuracy certification:** real calibrated cameras and surveyed metric reference points, controlled lighting and benchmark reporting.

Release gate: do not label the model more accurate, faster, metric or production-certified without corresponding independent evidence.
