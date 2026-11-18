# omega_geodepth_synthetic_v0.1.safetensors

Original trained OMEGA artifact; all raw bytes are stored in the numbered parts.

From the repository root on Linux or Termux:

```sh
cat models/omega_geodepth_synthetic_v0.1.safetensors.parts/part-*.bin > models/omega_geodepth_synthetic_v0.1.safetensors.tmp
printf '%s  %s\n' '93bd2c91ab36bb095c7f6bd950bbce1c332834bb27e22f20a6285e77ab349338' 'models/omega_geodepth_synthetic_v0.1.safetensors.tmp' | sha256sum -c - && mv -n models/omega_geodepth_synthetic_v0.1.safetensors.tmp models/omega_geodepth_synthetic_v0.1.safetensors
```

Expected length: 2313464 bytes. Only load after successful verification.
