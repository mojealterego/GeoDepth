#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# Avoid CPU oversubscription in chained PyTorch rendering jobs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python -m compileall -q omega_geodepth tests scripts
python -m pytest -q
python -m omega_geodepth.da3_source inspect --archive vendor/Depth-Anything-3-main.zip
DA3_VERIFY_TEMP="$(mktemp -d)"
trap 'rm -rf "$DA3_VERIFY_TEMP"' EXIT
python -m omega_geodepth.da3_source extract --archive vendor/Depth-Anything-3-main.zip --target "$DA3_VERIFY_TEMP/da3-src"
python -m omega_geodepth.da3_source verify --target "$DA3_VERIFY_TEMP/da3-src"
python -m omega_geodepth verify --model-dir models
python -m omega_geodepth.reconstruction5d from-npz \
  --manifest examples/sequence5d/manifest_from_npz.json \
  --output "${OMEGA_VERIFY_OUTPUT:-/tmp/omega-geodepth-v08-verify}" \
  --voxel-size 0.08 --neighbor-radius 0.30 --mesh --tsdf \
  --tsdf-voxel-size 0.12 --tsdf-truncation 0.36 --render-intervals 1 --gaussian-splats

# Verify the *trained* time-conditioned Gaussian checkpoint, not only its source.
python -m omega_geodepth.trainable_4dgs render \
  --checkpoint examples/v05_trainable_4dgs \
  --manifest examples/sequence5d/manifest_from_npz.json \
  --time 0.6 --max-side 32 \
  --output "${OMEGA_VERIFY_OUTPUT:-/tmp/omega-geodepth-v08-verify}/v05_render.png"

# V0.6 learned nonlinear deformation checkpoint: independent load + in-interval rendering.
python -m omega_geodepth.deformable_4dgs render \
  --checkpoint examples/v06_deformable/checkpoint \
  --manifest examples/sequence5d/manifest_from_npz.json \
  --time 0.6 --max-side 32 \
  --output "${OMEGA_VERIFY_OUTPUT:-/tmp/omega-geodepth-v08-verify}/v06_render.png"


# v0.8: authentic DA3-SMALL original weights, CPU, relative depth and native 5D.
python -m omega_geodepth.da3_real \
  --model-dir models/da3 --source-dir vendor/da3_src \
  --input examples/v08_real_da3/images/000.png examples/v08_real_da3/images/010.png \
  --process-res 280 --verify-reprojection \
  --output "${OMEGA_VERIFY_OUTPUT:-/tmp/omega-geodepth-v08-verify}/da3_real"
python -m omega_geodepth.reconstruction5d from-da3 \
  --runtime native_network --checkpoint models/da3 --official-source vendor/da3_src \
  --model DA3-SMALL --device cpu --manifest examples/v08_real_da3/manifest.json \
  --process-res 280 --verify-reprojection --output "${OMEGA_VERIFY_OUTPUT:-/tmp/omega-geodepth-v08-verify}/da3_5d" \
  --voxel-size .08 --neighbor-radius .12 --max-points-per-frame 24000 --mesh
