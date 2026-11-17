# OMEGA GeoDepth Ω 0.3.0 — change log

- Preserved complete v0.2 source code, small trained Safetensors/TorchScript weights, examples, ComfyUI node, benchmarks and original reference PDF.
- Added fail-closed original DA3 Small/Base checkpoint contract with SHA-256 manifest and restricted local Hugging Face model config constructors.
- Added explicit opt-in downloader for official DA3 weights and direct `from-da3` multi-view geometry workflow (requires externally supplied original checkpoint and upstream code).
- Added 5D scene field: xyz + time + observed sRGB luminance, per-frame spatial voxel fusion, nearest-neighbor scene flow, binary time-aware PLY and NPZ.
- Added per-frame depth triangulation, projective TSDF and marching cubes, unknown-voxel mask, bounded memory limits.
- Added sparse observed-only time interpolation renderer, novel camera projection API and z-buffer.
- Added 3-frame moving-sphere raycast example including depth, K, poses, time, 3 volumetric TSDFs, OBJ meshes and temporal visualizations.
- Added tests for geometric invariants, frame ordering, path traversal, checkpoint supply chain and render visibility.
- No original DA3 weights downloaded in this environment; original checkpoint end-to-end inference, actual hardware benchmarks, optimised 4DGS and full 400 MP pipelines remain incomplete.
