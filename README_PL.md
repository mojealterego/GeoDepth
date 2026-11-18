# OMEGA GeoDepth Ω v0.16.0 — DA3 / 4D / 5D / globalna geometria

Aktualna instrukcja: **`docs/ZMIANY_v0.16_PL.md`**. Oficjalny FP32 i pochodny FP16 są w głównej paczce; AQ6/AQ4 w osobnym dodatku. Wszystkie profile wykonują inferencję FP32. Poniżej zachowano historię poprzednich wydań.

---

# OMEGA GeoDepth Ω v0.8.0 — prawdziwy DA3-SMALL × geometria 5D

**NOWOŚĆ v0.8:** dołączono oryginalny checkpoint `models/da3/model.safetensors` wraz
z `config.json` i manifestem SHA-256. **Zweryfikowana inferencja oryginalnego DA3
Small na CPU** działa przez `omega_geodepth.da3_real`, również na fotografiach
dostarczonych w oficjalnym archiwum DA3. Zachowano moduły OMEGA v0.2–v0.7.

Pobieranie wag DA3 nie jest już wymagane dla **Small**. Base nadal nie jest
dołączony. Poprzednie sekcje oznaczone v0.2–v0.7 przedstawiają historyczny stan
wydań i nie aktualny brak wag. Instrukcja aktualnego wydania wraz z komendami:
**`docs/ZMIANY_v0.8_PL.md`**.

**Ograniczenia:** wartości głębi mają skalę względną; 5D oznacza XYZ + czas +
cechę obrazu, a nie 5 wymiarów fizycznej przestrzeni. Nie przeprowadzono
walidacji metrycznej ani pomiarów na Androidzie.

---

# OMEGA GeoDepth Ω 0.7.0 — DA3 offline × 5D × Fused TSDF × nieliniowe gaussy 4D

**Data wydania:** 2026-11-16 • **tryb dostawy:** ZIP, bez zmian w repozytoriach • **wykonanie:** Python 3.10+ (lokalnie testowano Python 3.13 CPU) • **filtracja materiału NSFW:** brak klasyfikatora treści.



## NOWOŚĆ v0.7.0: dostarczone źródła DA3 + weryfikacja geometrii

Dołączono **niezmodyfikowany kod źródłowy DA3 przesłany w ZIP**, bez wag pretrained.
Jego ekstrakcja i sprawdzanie SHA-256 są dostępne przez `omega-geodepth-da3-source`.
Opcjonalna bramka `--verify-reprojection` sprawdza spójność między następującymi
po sobie widokami i *abstynuje* przy nieodpowiednim pokryciu.
`--official-source` ładuje źródła z katalogu zweryfikowanego względem przesłanego archiwum.
**Oryginalna inferencja DA3 nadal wymaga pobrania oficjalnych wag i zależności.**
Pełne instrukcje oraz zakres testów: `docs/ZMIANY_v0.7_PL.md`.

---

## NOWOŚĆ v0.6.0: uczona deformacja nieliniowa w czasie

Nowy moduł `omega_geodepth/deformable_4dgs.py` rozszerza stary model liniowy o
**smooth temporal RBF deformation**: każdemu obserwowanemu Gaussowi przypisuje
uczone, gładkie przesunięcia w węzłach czasu. Rekonstrukcja zachowuje geometrię
bazową przy `reference_time` i nie zezwala na ekstrapolację poza zakres węzłów.
Oryginalne checkpointy liniowe v0.5 działają bez zmian. Wagi nowego modelu
w `examples/v06_deformable/checkpoint/gaussians.safetensors` są **rzeczywiście trenowane**,
ale na syntetycznym, jednoobiektowym przykładzie.

Wydanie obejmuje **weryfikację z rozdzielonymi chwilami treningu, walidacji i audytu**
dla dwóch trajektorii analitycznych: zmian kierunku i przyspieszenia.
Raport `examples/v06_deformable/benchmark.json` zawiera pomiary RGB, głębi,
pokrycia, ograniczenia i konfigurację. Żadne z tych liczb nie są benchmarkiem
rzeczywistych fotografii. Porównanie PNG w `comparison.png`.

Trening własnego checkpointu z istniejącego manifestu RGB-D (lokalnie, CPU):

```bash
OMP_NUM_THREADS=1 python -m omega_geodepth.deformable_4dgs train \
 --manifest examples/sequence5d/manifest_from_npz.json \
 --output /tmp/omega_deformable_v06 --knots 5 --splats 48 --steps 150 \
 --max-side 32 --depth-weight 0.6 --deformation-regularization 0.05
python -m omega_geodepth.deformable_4dgs render \
 --checkpoint /tmp/omega_deformable_v06 \
 --manifest examples/sequence5d/manifest_from_npz.json \
 --time 0.6 --output /tmp/omega_v06_mid.png
```

Zachowanie wcześniejszych parametrów sceny liniowej: `train --upgrade-from
examples/v05_trainable_4dgs ...`. Wspomniane gaussy są sferyczne, z alpha
compositing i ruchem nieliniowym, **nie pełnym wielokamerowym 4DGS/5D**.
Oficjalnych wag DA3 nadal nie pobrano — brak DNS na połączeniu do HF.
Wersja v0.6 nie wprowadza nowej, niezweryfikowanej estymacji metrycznej.

---

## POPRZEDNIA ZMIANA v0.5.0: wytrenowany model dynamicznych Gaussów (eksperymentalny)

Prawdziwa optymalizacja na danych RGB-D: **72 splaty**, 150 kroków Adam, 3 syntetyczne klatki 32×32. Checkpoint: `examples/v05_trainable_4dgs/gaussians.safetensors` (nie jest to DA3). Wg `reports/v05_holdout/holdout_report.json` średni MAE RGB na dwóch dodatkowych chwilach analitycznej sceny 0.11564 (nieuczona inicjalizacja) → 0.01844 (wyuczone gaussy), **84.06% poprawy**. Nie wykazano generalizacji na rzeczywiste obrazy.

Trening i renderowanie z zachowanych manifestów:

```bash
OMP_NUM_THREADS=1 python -m omega_geodepth.trainable_4dgs train \
 --manifest examples/sequence5d/manifest_from_npz.json \
 --output /tmp/omega_v05_gaussians --splats 72 --steps 150 \
 --lr 0.026 --max-side 32
python -m omega_geodepth.trainable_4dgs render \
 --checkpoint /tmp/omega_v05_gaussians \
 --manifest examples/sequence5d/manifest_from_npz.json \
 --time 0.6 --max-side 32 --output /tmp/omega_v05_render.png
```

Zaostrzono weryfikację checkpointów DA3: oryginalny SHA-256 upstream jest obowiązkowy (nie wystarczy lokalny manifest), odrzucane są symlinki. **Oficjalne DA3 nadal nie zostało pobrane**. Połączenie `curl`/pobieranie pliku kończy się błędem DNS. Zobacz `docs/ZMIANY_v0.5_PL.md` oraz `docs/DA3_OFFLINE_PL.md`.

Nazwa „4DGS” oznacza tutaj ruch Gaussów liniowy w czasie, uczony z RGB-D. Nie jest to pełna architektura deformowalnych 4D Gaussian Splatting. Nazwa „5D” to starsze `XYZ+czas+cecha wyglądu`, nie 5 współrzędnych przestrzennych.

## Co rzeczywiście zawiera paczka

1. **Wagi OMEGA-Synthetic v0.2**: wytrenowane 1 295 234-parametrowe wagi `models/omega_geodepth_synthetic_v0.2.safetensors` oraz TorchScript. To sprawdzony lokalny model treningowy na syntetycznych danych, NIE DA3.
2. **Adapter rzeczywistego Depth Anything 3 (DA3-SMALL i DA3-BASE)**: ładowanie *wyłącznie z lokalnego folderu* zawierającego oficjalne `model.safetensors`, `config.json` i manifest SHA-256. Integruje oryginalne API DA3, normalizuje `[N,3,4]` world-to-camera do `[N,4,4]` camera-to-world oraz weryfikuje K, głębię i pozycje kamer.
3. **Własny wieloklatkowy rdzeń rekonstrukcji 5D**: 3D XYZ + czas + obserwowana luminancja sRGB jako cecha (nie dodatkowa fizyczna współrzędna). Łączy chmury punktów na siatce czasowo-przestrzennej, wylicza przybliżony scene-flow, udostępnia zachowujące nieobserwowane obszary eksporty 5D `NPZ` i binarny `PLY`.
4. **Rekonstrukcja powierzchni**: każda chwila ma niezależną siatkę OBJ po back-projection, filtrację krawędzi i osobny wolumen TSDF, z opcjonalnym wyciąganiem powierzchni (scikit-image marching cubes).
5. **Z-buffer i próbkowanie czasu**: oszczędny renderer danych zaobserwowanych, eksport RGB/depth/mask w chwilach pośrednich z jawnymi brakami geometrii.
6. **Testy, przykłady i dokumentacja**: zachowane wszystkie testy i materiały v0.2, nowe testy kontraktowe i pełny przykład ruchomej sceny 3 klatek.

**Nie zawiera** pobranych oficjalnych wag DA3: sesja wykonawcza nie miała dostępu sieciowego do Hugging Face. Uczciwie: adapter jest zintegrowany i testowany kontraktowo, ale nie sprawdzony end-to-end na oryginalnych wagach. Skrypt poniżej doinstalowuje je później w sieci użytkownika. Nie ma fikcyjnych checkpointów z etykietą DA3.

## Instalacja

Z zaufanym środowiskiem Python 3.10+:

```bash
python -m pip install -r requirements.txt
# dla TSDF + ekstrakcji siatki:
python -m pip install "scikit-image>=0.23,<1"
python -m pip install --no-deps dist/omega_geodepth-0.7.0-py3-none-any.whl
bash scripts_verify.sh
```

W przypadku instalacji wyłącznie offline wymagane są lokalne koła pakietów i kompatybilny PyTorch.

## A. Model syntetyczny — inferencja z zachowaną poprzednią wersją

```bash
python -m omega_geodepth verify --model-dir models
python -m omega_geodepth infer --image examples/input/synthetic_scene.png \
 --model-dir models --output /tmp/omega_depth --max-side 64
```

## B. Oficjalne wagi Depth Anything 3 — instalacja kontrolowana

Zewnętrzny model to **oddzielny folder**, nie wolno nadpisywać lokalnego modelu syntetycznego.

```bash
# Po zainstalowaniu oficjalnego pakietu Depth Anything 3 (wg upstream):
# https://github.com/ByteDance-Seed/Depth-Anything-3
python -m pip install huggingface_hub

# Pobranie modelu tylko na wyraźne polecenie i po zaakceptowaniu licencji Apache-2.0:
python -m omega_geodepth.download_da3 \
 --model DA3-SMALL --destination models/da3/small \
 --allow-network --accept-license

# Większy DA3 Base — osobny folder:
python -m omega_geodepth.download_da3 \
 --model DA3-BASE --destination models/da3/base \
 --allow-network --accept-license
```

Skrypt pobiera jedynie `config.json`, `model.safetensors`, `README.md`, ewentualnie `LICENSE`; sprawdza nagłówek Safetensors, liczy SHA-256 i zapisuje lokalny `omega_da3_manifest.json`. Dla ręcznie skopiowanych plików użyj `--register-existing --revision <hash_commitu_upstream>`. Opcja `--revision main` nie zapewnia niezmienności w czasie; do reprodukowalnych wdrożeń zalecamy dokładny commit. Lokalny SHA potwierdza spójność plików, **nie potwierdza kryptograficznie ich pochodzenia**.

Model DA3 wymaga uruchamiania `depth_anything_3` z oficjalnego repozytorium, którego **nie dołączono** do tego ZIP. Sama instalacja `huggingface_hub` nie instaluje DA3. Pozostałe użycie DA3 nie pobiera wag samoczynnie. Dodatkowa izolacja sieci workerów wymaga kontroli systemowej (np. kontener z no-egress).

Oficjalne źródła: https://github.com/ByteDance-Seed/Depth-Anything-3 oraz https://huggingface.co/depth-anything/DA3-SMALL i https://huggingface.co/depth-anything/DA3-BASE.

## C. Funkcjonująca rekonstrukcja 5D z istniejących map głębi

W archiwum znajduje się gotowy manifest `examples/sequence5d/manifest_from_npz.json` zawierający 3 klatki proceduralnego raycastu ruchomej kuli, ich mapy Z, obrazy RGB, K, T_c2w i znaczniki czasu.

```bash
python -m omega_geodepth.reconstruction5d from-npz \
 --manifest examples/sequence5d/manifest_from_npz.json \
 --output /tmp/omega_scene5d --voxel-size 0.08 --neighbor-radius 0.3 \
 --mesh --tsdf --tsdf-voxel-size 0.12 --tsdf-truncation 0.36 \
 --render-intervals 2
```

**Wyniki:** `scene5d.npz` (tensor `[N,5]` XYZ+t+luma), `scene5d.ply` (kolor, czas, przepływ, metadane proweniencji), `meshes/frame_XXXX.obj`, `tsdf/frame_XXXX.npz`, `tsdf/frame_XXXX.obj`, `renders/between_XXXX_YYYY.png|npz`, `report.json` z SHA-256 wszystkich artefaktów.

**Uwaga:** pliki `from-npz` muszą zawierać `depth_z`, `source_rgb`, `intrinsics_k`, `valid_mask`; wymaga się rzeczywistej lub jawnie oszacowanej macierzy camera-to-world. Brak kalibracji blokuje przeliczenie zamiast tworzenia fałszywej pozycji. Dane z paczki syntetycznej mają jednostki arbitralne.

## D. Bezpośrednie połączenie DA3 -> 5D (wymaga pobranych wag)

```bash
python -m omega_geodepth.reconstruction5d from-da3 \
 --manifest examples/sequence5d/manifest_from_da3.json \
 --checkpoint models/da3/small --model DA3-SMALL \
 --output /tmp/omega_da3_scene5d --process-res 504 \
 --voxel-size 0.05 --neighbor-radius 0.15 --mesh --render-intervals 2
```

Dopóki nie ma *oryginalnych* plików modelu, polecenie **celowo kończy się błędem** `DA3_CHECKPOINT_MANIFEST_MISSING`, a nie sukcesem zastępczym. DA3-Small/Base przewidują głębię względną, więc **wszystkie wyniki zachowują `arbitrary_scale`**. Nie ma fikcyjnego przeliczenia na metry.

## E. Testy i dowody

```bash
python -m compileall -q omega_geodepth tests
python -m unittest discover -s tests -v
python -m omega_geodepth verify --model-dir models
python -m pip wheel --no-deps --no-build-isolation . -w dist
```

- `reports/tests_v03_final.txt` — rzeczywiste wyniki testów Python
- `reports/reconstruction5d_demo.json` — dane rekonstrukcji ruchomej sceny
- `ARTIFACTS_SHA256.json` — integralność plików paczki (bez podpisu cyfrowego)
- `models/da3/checkpoint_status.json` — jawnie niedostarczone oficjalne wagi

## F. Nowe funkcje v0.4 — fuzja wielowidokowa i rasteryzacja gaussowska

Przy znanej kalibracji i geometrii *statycznej* (lub jawnie dostarczonych maskach statyczności):

```bash
python -m omega_geodepth.reconstruction5d from-npz \
 --manifest examples/sequence5d/manifest_from_npz.json \
 --output /tmp/omega_v04 --voxel-size .08 --neighbor-radius .30 \
 --mesh --tsdf --tsdf-voxel-size .12 --tsdf-truncation .36 \
 --gaussian-splats --render-intervals 2
```

Wynik: `gaussian5d.npz`, `gaussian_renders/*.png|npz`. **Uwaga:** przykład 3-klatkowy jest dynamiczny; w tym trybie celowo NIE stosuje się statycznej fuzji TSDF.

Dla sceny statycznej użyj analogicznego manifestu klatek oraz opcji `--fused-static-tsdf`. Powstają `static_fused_tsdf.npz|obj` oraz raport pokrycia przez różne kamery. Jeśli nie ma masek statyczności, raport jawnie oznacza `assumed_static_without_masks=true`.

Weryfikacja modułu i wydania: `python -m pytest -q` i `bash scripts_verify.sh`. Raport szczegółowy: `docs/ZMIANY_v0.4_PL.md`.

## Ograniczenia (nieredukowane sloganem „absolute maximum”)

- To **eksperymentalna rekonstrukcja 5D cech sceny**, nie pięć przestrzennych wymiarów i nie nauczony dynamiczny NeRF/4DGS. Przybliżony NN scene-flow nie jest zweryfikowanym śledzeniem obiektów. Renderer nie rekonstruuje powierzchni zasłoniętych.
- Integracja oryginalnego DA3 nie została zweryfikowana na jego rzeczywistych wagach (brak pobrania). Wagi OMEGA-v0.2 nadal są syntetycznie trenowane.
- Nie wykonano testów na Galaxy S24 Ultra, GPU, zdjęciach rzeczywistych ani wielokamerowej kalibracji metrycznej. TSDF ma z góry ograniczony budżet wokseli, a Python przetwarza klatki w pamięci.
- Nie dostarczono pełnego systemu rekonstrukcji 5D klasy produkcyjnej, 4D Gaussian Splatting, optymalizatora bundle adjustment, rekonstrukcji danych nieobserwowanych ani 400 MP out-of-core.
- Brak klasyfikatora NSFW i odmowy ze względu na temat obrazu nie zwiększa automatycznie jakości estymacji głębi; "abliteration" nie dotyczy tego modelu.

Dalsze szczegóły: `ARCHITECTURE_PL.md`, `MODEL_CARD_PL.md`, `docs/DA3_OFFLINE_PL.md`, `docs/REKONSTRUKCJA_5D_PL.md`.


## v0.10.0

Eksperymentalny profil dynamic INT8 CPU dla backbone DA3. Patrz [zmiany](docs/ZMIANY_v0.10_PL.md). FP32/FP16 pozostają profilami preferowanymi.

## v0.11 — wieloskalowe uczone pole 5D

Zobacz [dokumentację integracji](docs/ZMIANY_v0.11_PL.md). Pole RGB jest warunkowane dostarczoną geometrią. Zachowuje względne jednostki DA3 i odróżnia czas rzeczywisty od założenia statycznego. Dostarczono dwa nowo wytrenowane checkpointy, z raportami porównawczymi. Nie zastępuje dotychczasowej rekonstrukcji TSDF/Gaussians.

## v0.12 — ENGINE v1, statyczny audyt jakości i bezpieczny import NPZ

Nowy moduł `omega_geodepth.quality_v1` uzupełnia istniejący silnik DA3 i 5D. Nie zastępuje dotychczasowego CLI. Instrukcje, wyniki, ograniczenia oraz konfiguracja Base: [docs/ZMIANY_v0.12_PL.md](docs/ZMIANY_v0.12_PL.md). Oficjalne wagi Base nie zostały dostarczone.

### Aktualizacja 0.13.0 — DA3 Base
Weryfikacja oryginalnego Base, obustronna reprojekcja i kontrolowany fallback rozdzielczości. Instrukcje instalacji nakładki i odtwarzania sześciu części wag: [docs/ZMIANY_v0.13_PL.md](docs/ZMIANY_v0.13_PL.md). Base FP32 pozostaje opcjonalny; Small jest domyślny.

## v0.14 — niezależny audyt geometrii

Moduł Visual Geometry Benchmark, Sim(3), AUC3/30, Chamfer i F1, oraz opcjonalne pose conditioning DA3. Zobacz [instrukcję](docs/ZMIANY_v0.14_PL.md).

## v0.15.0 — Depth–Ray

Nowy opcjonalny dekoder promieni i audyt kierunku, początku i składowej Z; osobna rygorystyczna kalibracja Sim(3) z odłożonymi kamerami. Domyślna inferencja pozostaje tokenowa. 230 testów lokalnych; szczegóły i ograniczenia w [docs/ZMIANY_v0.15_PL.md](docs/ZMIANY_v0.15_PL.md). Nakładka kumulatywna zawiera również zmiany źródłowe v0.14.

## v0.16 — szybki dekoder i globalna synchronizacja

[Dokumentacja v0.16](docs/ZMIANY_v0.16_PL.md). Opcjonalny fast_ray, statyczny graf nakładających się okien, rygorystyczna kontrola kamer i głębi. Nowe testy 12, pełna regresja 242. Oba rzeczywiste benchmarki grafu zostały odrzucone, nie deklarujemy ukończonej globalnej rekonstrukcji.
