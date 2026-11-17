# GeoDepth Ω v0.12.0 — integracja ENGINE v1

## Pochodzenie i rozgraniczenie
Dostarczony OMEGA_GeoDepth_ENGINE_v1.zip (samodzielna wersja 1.0.0, Apache-2.0) zawiera 38 wpisów, kod, przykładową predykcję, konfigurację DA3 Base oraz manifest, ale nie wagi. Jego 45 testów uruchomiono przed integracją i wszystkie przeszły. To **moduł uzupełniający**, a nie następca 0.11 z modelami Small i polami 5D. Nie zastąpiono wcześniejszych modułów `geometry`, `calibration` ani CLI. Dostosowany kod znajduje się w `omega_geodepth/quality_v1` wraz z oryginalną licencją. Oryginalny ZIP i dokumentację zachowano jako odrębny zasób referencyjny.

## Implementacja i ograniczenia
- Import NPZ dwóch schematów: `depth/conf/extrinsics/intrinsics` (W2C) i `depth_z/confidence_raw/pose_c2w/intrinsics_k` (C2W), walidacja nagłówków NPY i rozmiaru ładunku przed alokacją, limity, brak pickle, odrzucanie metryki deklarowanej przez niezaufany NPZ. Nie jest to izolacja systemowa; pliki wejściowe nie mogą być równolegle zmieniane.
- Geometria widoczności, konsensus w **scenach statycznych** lub zsynchronizowanych widokach, ponowny audyt dopiero po refinamencie, progi pokrycia, spójności i paralaksy; pusty PLY odrzucany. Progi CLI nie dowodzą dokładności wobec fizycznego wzorca.
- Opcjonalne kotwice Z w metrach, rozkład przestrzenny, brak podwójnej kalibracji, kontrola przepełnienia indeksu wokseli. Pochodzenie pomiarów pozostaje odpowiedzialnością dostawcy.
- Eksport PLY i raportu, opcjonalne mapy jakości, publikacja no-replace z markerem ukończenia; na platformach bez renameat2 zastosowano rezerwację katalogu z markerem. Nie jest to gwarancja odporności na zanik zasilania.
- Most `quality_v1.bridge.prepare_static_field` do KPlanes5D: wyłącznie jawne założenie statyczności, maski jakości, XYZ/czas (t=0)/skala pikselowa; bez wymyślania ruchu i bez metryzacji DA3.
- `quality_v1.metrics` i `evaluation`: AbsRel, RMSE, delta-1, SILog, pokrycie prawidłowej referencji przez predykcję. Mediana skali dopasowana do referencji testowej nie jest kalibracją metryczną.
- DA3 Base: dołączono dostarczoną konfigurację i manifest; pin wag odpowiada istniejącemu upstream pinowi. Rozszerzono `DA3RealCPU` i `from-da3` o parametr modelu; **Base tylko FP32**. Testy braku checkpointu i odmowy błędnego profilu; nie wykonano prawdziwej inferencji Base. Plik 541 518 028 B nie znajduje się w ZIP. Konfiguracja Base przypięta do dostarczonego pliku; oryginalny SHA wag nadal wymagany. Nie przeprowadzono nowego treningu DA3.

## Uruchomienie
Instaluj pełną dystrybucję wraz z opcjonalnym dodatkiem AQ do tego samego katalogu nadrzędnego. Zależności jak w README v0.11. Potok statyczny:

```bash
python -m omega_geodepth.quality_v1 reconstruct \
  --prediction examples/v08_real_da3/da3_real_results.npz \
  --output output/quality --assume-static --refine --quality-maps
python -m omega_geodepth.quality_v1 evaluate \
  --prediction-depth predicted.npy --ground-truth-depth truth.npy \
  --alignment none --output output/evaluation.json
```

Gdy użytkownik dostarczy oficjalny checkpoint Base, należy zarejestrować go standardowym `register-da3` (patrz wcześniejsze instrukcje `omega-geodepth-da3 --help`), a następnie uruchomić `python -m omega_geodepth.da3_real --model-id DA3-BASE --model-dir models/da3_base --source-dir vendor/da3_src --input photo.png --output output/base --profile fp32`. Wymagany SHA-256 wag: `e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5`. Nie używać manifestu ENGINE jako zamiennika rejestracji.

## Wyniki
49 testów modułów jakości (36 przeniesionych, 13 integracyjnych), całkowita regresja 201 testów (z dodatkiem AQ). Benchmark `examples/v12_quality/benchmark/benchmark.json`: dwie fotografie z archiwum DA3, autentyczny Small FP32 CPU, 2×154×280; zgodność reprojekcji 86,70% → 87,12%, 7006 wokseli, 3241 zaakceptowanych obserwacji dla pola statycznego. Bez fizycznych danych referencyjnych. Jedna analityczna płaszczyzna Z=2, równomiernie i przeciwnie zaburzone wejścia Z=1.95/2.05: RMSE 0.04999995 → 0.02757715 (w jednostkach sceny), pokrycie 100%; rezultat ilustracyjny, nie benchmark generalizacji.

## Otwarte bramki
Base/Giant (wagi i uruchomienie), natywne AQ4/AQ6, kontrola peak RSS, pełny dynamiczny 4DGS/5D, kalibracja niepewności i walidacja metryczna na zestawach rzeczywistych, Android/GPU/400 MP; brak certyfikacji produkcyjnej i zdalnego CI. API TorchScript i torch.ao zgłaszają ostrzeżenia deprecacji.
