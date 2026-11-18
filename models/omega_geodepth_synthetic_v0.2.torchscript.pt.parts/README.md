# omega_geodepth_synthetic_v0.2.torchscript.pt

Original trained OMEGA artifact; all raw bytes are stored in the numbered parts.

From the repository root on Linux or Termux:

```sh
cat models/omega_geodepth_synthetic_v0.2.torchscript.pt.parts/part-*.bin > models/omega_geodepth_synthetic_v0.2.torchscript.pt.tmp
printf '%s  %s\n' '7df967f9303329935a9410ad381845eb3b15085414c42fe5433bd22540f81df5' 'models/omega_geodepth_synthetic_v0.2.torchscript.pt.tmp' | sha256sum -c - && mv -n models/omega_geodepth_synthetic_v0.2.torchscript.pt.tmp models/omega_geodepth_synthetic_v0.2.torchscript.pt
```

Expected length: 5222767 bytes. Only load after successful verification.
