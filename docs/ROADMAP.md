# GeoDepth Ω — status v0.16.0

## Potwierdzone lokalnie
- Oryginalne DA3 Small i Base FP32, kontrola SHA-256 i strict-load; Small FP16/AQ4/AQ6, eksperymentalny dynamiczny INT8 CPU.
- TSDF, eksperymentalne Gaussy 4D/5D, Representations5D, KPlanes5D: nie wykazano przewagi pola RGB nad NN.
- ENGINE v1, secure NPZ, konsensus, odtwarzanie Base z sześciu wolumenów, fallback rozdzielczości.
- v0.14: benchmark referencyjny Sim(3), F1/Chamfer/AUC, rzeczywiste pose conditioning, 223 testy.
- v0.15: natywny ray decoder, audyt p90 kierunku/początku i osi Z, kontrola rotacji i centrów na rozłącznych kotwicach; 230 testów. 8 przebiegów 140 px, dodatkowy 280 px nieukończony (3/4).
- v0.16: fast_ray, graf Sim(3), weryfikacja nakładających się widoków, statyczna rekonstrukcja okienkowa i odmowa eksportu niespójnej sceny. 242 testy, 12 nowych. WHL z niezależną inferencją Small.
- Benchmark 140 px: Small 7,462→1,013 s, Small+kamery 7,743→0,960 s, Base 9,925→2,995 s, Base+kamery 9,877→3,114 s. Pojedyncze pomiary na jednej scenie. Domyślny token decoder bez zmian; F1 nie poprawiło się konsekwentnie.
- Graf: koszt Small 323,775→9,272, Base 81,158→3,438; oba zestawy realnych predykcji odrzucono. Dokładne transformacje syntetyczne odzyskane w testach.
- Raport: reports/v16/benchmark_140.json. Instrukcja: docs/ZMIANY_v0.16_PL.md. Zdalnego CI nie uruchamiano.

## Publikacja
Każda kolejna iteracja ma być zapisana na main wraz z testami i raportami. Weryfikować Git blob SHA i stan gałęzi po zapisie. Wagi i pliki binarne niewprowadzone do repozytorium są nadal otwartą bramką dystrybucji; docelowo użyć Git LFS lub Releases bez naruszania licencji.

## Otwarte bramki
- v0.17: trenowana głowica GS-DPT / osobny eksperymentalny moduł, walidacja NVS z rozłącznymi widokami.
- v0.18: pełniejsza dynamiczna rekonstrukcja 4D/5D.
- v0.19: niezależne realne RGB-D z fizyczną referencją, wieloscenowa dokładność, kalibracja niepewności.
- v0.20: natywne AQ4/AQ6, optymalizacja pamięci, GPU/Vulkan/Android i 400 MP.
- DA3 Giant/Large: brak checkpointów; ograniczenia licencyjne przed integracją.
- Dystrybucja binarna, niezależny audyt końcowy, CI, brak certyfikacji produkcyjnej.

### Aktualizacja dystrybucji
- OMEGA v0.1/v0.2 Safetensors i TorchScript: 44 części, wszystkie zapisane w GitHub; instrukcje odtworzenia i sumy SHA-256.
- DA3 FP32/FP16/AQ4/AQ6: pełne checkpointy oczekują na transfer. Rejestr: models/repository_inventory.json.

### 2026-11-24 — bezpieczna ścieżka publikacji dużych checkpointów
- IMPLEMENTED: `omega_geodepth/model_release.py` — sprawdza 5 plików DA3 względem `repository_inventory.json`, ogranicza ścieżki, rozmiary i wykrywa symlinki, oblicza strumieniowo SHA-256.
- IMPLEMENTED: publikacja oryginalnych plików jako GitHub Release assets bez Base64 i dzielenia na tysiące blobów; preflight bez sieci, jawne `--publish`, GitHub `main` HEAD lease, tryb draft i weryfikacja zdalnego `sha256:` przed upublicznieniem.
- TESTED LOCALLY: 16 testów jednostkowych offline (`pytest -q tests/test_model_release_v17.py`); 0 błędów; sprawdzono składnię `py_compile`.
- BLOCKED: 5 checkpointów DA3 o łącznej wielkości 810442158 bajtów nie zostało opublikowanych: ich oryginalne bajty nie są dostępne w obecnej sesji; nie przeprowadzono zewnętrznego E2E wysyłki.
- v0.17 GS-DPT pozostaje OPEN; poniższy moduł dystrybucji nie stanowi implementacji ani treningu GS-DPT.

### v0.17 — eksperymentalna głowica Gaussian
- IMPLEMENTED / TESTED LOCALLY: `omega_geodepth/gsdpt.py`, 6 testów CPU, kontrolowany trening na syntetycznej scenie z rozłączną kamerą walidacyjną. Raport `reports/v17/gsdpt_synthetic.json`.
- Nie jest to pełny GS-DPT: integracja z cechami DA3, rzeczywiste dane NVS, zapis wytrenowanej głowicy, pełna regresja i ocena jakości pozostają OPEN. Checkpointy DA3 pozostają BLOCKED.

### OMEGA GeoDepth v4 — Activation-Aware Quantization
- IMPLEMENTED / TESTED LOCALLY: referencyjna kwantyzacja grupowa 4/6/8 bitów z wagą błędu opartą o diagonalny drugi moment aktywacji; `omega_geodepth/da3_quant/activation_v4.py`, 6 testów CPU, w tym limit pamięci po paddingu. Raport `reports/v17/activation_v4.json`.
- OPEN: kalibracja na prawdziwych aktywacjach DA3, pakowanie bitowe, zachowanie warstw geometrycznych, pełne checkpointy oraz porównawcza walidacja jakości głębi. Nie opublikowano nowego modelu.
