# OMEGA synthetic v0.2 checkpoint

The original Safetensors checkpoint is stored as 15 raw ordered parts in this directory. Reassemble from repository root:

```sh
cat models/omega_geodepth_synthetic_v0.2.parts/part-*.bin > models/omega_geodepth_synthetic_v0.2.safetensors.tmp
printf '%s  %s\n' '936d76c0b4cdcfb9854e143d3af5e605808b415bca683b78e63806cc26564d0d' 'models/omega_geodepth_synthetic_v0.2.safetensors.tmp' | sha256sum -c - && mv -n models/omega_geodepth_synthetic_v0.2.safetensors.tmp models/omega_geodepth_synthetic_v0.2.safetensors
```

Expected 5,187,384 bytes. Verify SHA-256 before loading. This is the synthetic OMEGA v0.2 model, not DA3 Small or Base. The official DA3 weights are not yet stored in the repo.
