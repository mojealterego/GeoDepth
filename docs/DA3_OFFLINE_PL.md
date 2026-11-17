# DA3 — model registry, contract, license and supply chain

**Scope:** `DA3-SMALL`, `DA3-BASE` — any-view models with original Apache-2.0 weight licenses, separate from the OMEGA synthetic backbone.

Official repo: https://github.com/ByteDance-Seed/Depth-Anything-3
API: https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md

- The source DA3 inference API accepts images and optionally `extrinsics` `[N,4,4]` w2c, `intrinsics` `[N,3,3]`. It returns `depth` `[N,H,W]`, `conf` `[N,H,W]`, `extrinsics` `[N,3,4]` w2c, `intrinsics` `[N,3,3]`, `processed_images` `[N,H,W,3]`.
- The OMEGA adapter transforms `[N,3,4]` w2c into strictly validated `SE(3)` c2w `[N,4,4]`; invalid camera poses fail, no silent coordinate reversal. A few invalid depth pixels become NaN, not artificial zero depth.
- Confidence raw is used **only** as a per-frame normalized quality *proxy*; never as a calibrated likelihood. Relative depth is `arbitrary_scale`.
- SHA-256 applies to the entire local file; model manifest explicitly records repository ID, model ID, and license. Not a signed supply-chain attestation.
- `config.json` is checked against an allowlist of known DA3 importable constructors to reduce arbitrary module imports. Official DA3 library itself remains an independently managed dependency; examine and pin its source before production deployment.
- Download is opt-in (`--allow-network --accept-license`); infer never downloads. `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set during inference; enforce network isolation externally in production.
- The official model artifacts are NOT bundled in v0.3 due to failed network availability in the build environment. For independent machines, download original weights from official Hugging Face with the provided provisioning script.

**Licensing:** DA3-LARGE, DA3-GIANT, DA3NESTED variants are not permitted by this adapter for default commercial use because their upstream model weight licenses are CC BY-NC 4.0, unlike DA3 Small/Base. The source code's Apache-2.0 license is not a license grant for all model weights.

**External integration proof:** local contract tests with synthetic mock `Prediction` pass; real upstream inference has **not** been run in this release.

## v0.5 — przeciwko samoatestacji wag

Weryfikacja wymaga **dwóch** zgodności: lokalny manifest SHA-256 oraz rzeczywisty SHA-256 pliku `model.safetensors` odpowiadający zarejestrowanemu publicznemu digestowi wag DA3 Small/Base. Fałszywa sieć w poprawnym kontenerze Safetensors nie przejdzie bramki. Piny: `DA3-SMALL 364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf`; `DA3-BASE e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5`. Wymagana osobna instalacja oficjalnego kodu; bez wag DA3 inferencja jest wprost blokowana.
