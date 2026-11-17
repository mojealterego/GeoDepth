# OMEGA GeoDepth Ω v0.8.0 — rzeczywisty DA3 Small (offline)

## Dowód pochodzenia i uruchomienia

- **Wagi DA3 Small**: `models/da3/model.safetensors`, 137 248 940 bajtów, SHA-256 `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf`. Suma jest zgodna z publicznym wskaźnikiem SHA-256 Hugging Face dla `depth-anything/DA3-SMALL` (commit `e08cab65`).
- **Konfiguracja**: oryginalny plik przekazany przez użytkownika, SHA-256 `a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2`. Przed uruchomieniem wymaga zgodności digestu, schematu i listy konstruktorów.
- **Źródła sieci**: oryginalny, dostarczony ZIP `vendor/Depth-Anything-3-main.zip` (SHA-256 `6d2ba1d07a69f3d260394e49c7545aed8f3cc7df8633469ae79782ee34978d42`), bez zmian. `vendor/da3_src` zawiera kopię niezmienionego kodu `src`; wszystkie `.py` i `.yaml` są weryfikowane bajtowo względem ZIP przed załadowaniem.
- **Uruchomienie**: `omega_geodepth.da3_real.DA3RealCPU` konstruuje *oryginalną* architekturę `DepthAnything3Net` na podstawie oryginalnego configu i ładuje prawdziwe tensory `.safetensors` przez `load_state_dict(..., strict=True)` z rozwinięciem aliasów zapisanych w metadanych Safetensors.
- **Różnica wykonawcza względem upstream**: dla inferencji bez eksportów specjalnych pomijane są *opcjonalne* zależności importowane przez `api.py` (MoviePy 1.x, COLMAP, Open3D, gsplat); wykorzystane są natywne `InputProcessor`, sieć i `OutputProcessor`. Minimalny moduł kompatybilności `addict.Dict` znajduje się w `vendor/da3_runtime_compat/addict.py` i działa wyłącznie lokalnie. Nie jest to przebudowa wag ani destylacja.

## Przykład na obrazach rzeczywistych

W katalogu `examples/v08_real_da3/` znajdują się obrazy pochodzące z dostarczonego repozytorium DA3 (`assets/examples/SOH/000.png`, `010.png`) oraz raport rzeczywistej inferencji CPU. Nie dysponujemy ground truth głębi ani wiarygodnymi znacznikami czasu zdjęć. Dwa znaczniki `time_s=0` i `1` są **przykładowymi etykietami kolejności**, nie pomiarem czasu.

Dla `process_res=280`:
- wynik `[N,H,W] = [2,154,280]`, 100% dodatnich skończonych pikseli głębi;
- audyt spójności względnej reprojekcji: 72,83% widocznego pokrycia, 3,38% mediany błędu względnego, 80,28% próbek w limicie 10%; **bramka zaliczona** dla tych progów;
- rekonstrukcja sceny 5D (XYZ, czas/indeks, proxy luminancji): 2707 punktów, 958 wzajemnych dopasowań z heurystyki NN, 2 siatki OBJ;
- wszystkie jednostki: `arbitrary_scale` i `relative_estimated`. Nie wolno nazywać tego pomiarem w metrach.

**Nie** jest to walidacja jakości na rzeczywistych pomiarach odległości, walidacja pełnej geometrii 5D ani wytrenowany na tych zdjęciach 4D Gaussian Splatting. Zachowano wytrenowane, ale odrębne checkpointy OMEGA-Synthetic v0.2 i niewielkie modele scenowe v0.5/v0.6.

## Uruchomienie

Od katalogu głównego po rozpakowaniu ZIP (PyTorch CPU, Python 3.13, numpy, PIL, scipy, omegaconf, safetensors, huggingface_hub, torchvision, opencv-python, einops):

```bash
python -m omega_geodepth.da3_real \
  --model-dir models/da3 --source-dir vendor/da3_src \
  --input examples/v08_real_da3/images/000.png examples/v08_real_da3/images/010.png \
  --process-res 280 --output /tmp/da3_real_outputs --verify-reprojection
```

Wyniki: `prediction.npz`, `depth_relative_00.png`, `depth_relative_01.png`, `metadata.json`.

Rekonstrukcja 5D z prawdziwych predykcji:

```bash
python -m omega_geodepth.reconstruction5d from-da3 \
  --runtime native_network \
  --manifest examples/v08_real_da3/manifest.json \
  --checkpoint models/da3 --official-source vendor/da3_src \
  --model DA3-SMALL --device cpu --process-res 280 \
  --verify-reprojection --output /tmp/da3_scene_5d \
  --voxel-size 0.08 --neighbor-radius 0.12 --mesh
```

W przypadku instalacji WHL zamiast wykonania z katalogu źródłowego, wszystkie ścieżki `models` oraz `vendor` muszą jawnie wskazywać katalog rozpakowanego ZIP. WHL nie zawiera wag ani kodu źródłowego zewnętrznego DA3.

## Granice wydania

- Działa DA3-SMALL, **nie** DA3-BASE (jego wag nie dostarczono).
- Działa inferencja CPU i wielowidokowa rekonstrukcja 5D na przykładach; **nie wykonano testu GPU ani Androida**.
- Brak zapewnionej skali metrycznej; nie należy interpretować surowego `conf` jako skalibrowanego prawdopodobieństwa.
- Brak kompletnego trenowanego deformowalnego modelu 4DGS na danych rzeczywistych, bez gwarancji wierności niewidocznej geometrii.
- Runtime natywny **nie obsługuje** eksportów specjalnych upstream ani automatycznego pobierania z internetu.
- Brak filtra NSFW dla danych wejściowych; nie stosowano ablacji w sensie modelu językowego (Abliterated).

## Licencje i pochodzenie

Źródła Depth Anything 3 © ByteDance Seed Team i licencja Apache 2.0 (`vendor/da3_src/LICENSE`, `vendor/Depth-Anything-3-main.zip`). Przesłane wagi `depth-anything/DA3-SMALL` również są deklarowane na licencji Apache 2.0, zob. https://huggingface.co/depth-anything/DA3-SMALL. Kod OMEGA pozostaje odrębnym projektem; licencje komponentów muszą być przestrzegane niezależnie. SHA-256 potwierdza zgodność *bajtów*, nie podpis kryptograficzny tożsamości autora.
