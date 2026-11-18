# omega_geodepth_synthetic_v0.1.torchscript.pt

Original trained OMEGA artifact; all raw bytes are stored in the numbered parts.

From the repository root on Linux or Termux:

```sh
cat models/omega_geodepth_synthetic_v0.1.torchscript.pt.parts/part-*.bin > models/omega_geodepth_synthetic_v0.1.torchscript.pt.tmp
printf '%s  %s\n' 'aa34155dddd4286dd4d0ac5fad8659481ad340fd42b721a30d342cf0e4a26562' 'models/omega_geodepth_synthetic_v0.1.torchscript.pt.tmp' | sha256sum -c - && mv -n models/omega_geodepth_synthetic_v0.1.torchscript.pt.tmp models/omega_geodepth_synthetic_v0.1.torchscript.pt
```

Expected length: 2336433 bytes. Only load after successful verification.
