# OMEGA GeoDepth Ω v0.5.0 — dziennik implementacji i dowodów

Data: 2026-11-16. Kontynuacja źródeł v0.4.0, bez zmian w repozytorium GitHub.

## 1. Weryfikacja wejścia

Archiwum v0.4.0 rozpakowano do odrębnego folderu roboczego. Przed zmianami wykonano `python -m pytest -q`: **76/76 zaliczonych**. Zachowano istniejące pliki wag syntetycznych OMEGA v0.2.

## 2. Wykonane zmiany

**Trenowalna geometria czasowo-przestrzenna** (`omega_geodepth/trainable_4dgs.py`):

- Uczone pola: pozycje 3D, prędkości świata (ruch liniowy zależny od czasu), skalarne rozmiary Gaussów, kolory RGB i opacity (wszystkie mają gradient i optymalizator Adam).
- Differentiable spherical Gaussian splatting, jawna intrinsics K + camera-to-world SE(3), front-to-back alpha composition i optyczne Z.
- Strata fotometryczna RGB oraz pomocnicza depth/silhouette z maskowaniem obserwacji, clipping gradientów, budżet prymitywów i pikseli, limit czasu obliczeń.
- Bezpieczny, bez-pickle zapis `.safetensors` i manifest SHA-256, weryfikacja przed ponownym uruchomieniem.
- Interfejs CLI `train`/`render`, odczyt wcześniejszych manifestów RGB-D i poprawna aktualizacja K po resamplingu w konwencji środka piksela `0.5`.

**Ochrona oryginalnych wag DA3** (`omega_geodepth/da3_adapter.py`):

- Dotychczas manifest użytkownika mógł deklarować dowolny checkpoint jako DA3: usunięto możliwość uznania nieoryginalnych wag za oficjalne.
- SHA-256 lokalnego pliku jest porównywany z publikowanym digestem upstream Hugging Face, a nie tylko z lokalnym manifestem.
- Dodano odrzucanie symlinków w ścieżkach checkpointu.
- Źródło wzorców: `depth-anything/DA3-SMALL` (commit e08cab6, SHA-256 `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf`), `depth-anything/DA3-BASE` (commit 6544606, SHA-256 `e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5`). Licencje Apache-2.0 (stan na dzień weryfikacji).
- Test integracji z atrapą API DA3 izoluje symulowany pin wewnątrz testu; nie dopuszczono takiego odstępstwa w kodzie produkcyjnym.

## 3. Wykonany trening 4D + niezależne chwile czasowe

Dane: syntetyczny raycast kuli przemieszczającej się nad nieruchomym tłem, rozdzielczość po przeskalowaniu 32×32, klatki treningowe `t=[0.0, 0.4, 0.8] s`, 72 gaussy, 150 iteracji Adam, seed 42.

- `loss_initial = 0.1755337119`; `loss_final = 0.0231476557` (mieszana strata RGB/depth/silhouette, nie benchmark jakości rzeczywistych zdjęć).
- Własny raycaster obliczył dodatkowe dane **w chwilach nieobecnych w treningu** `t=[0.2,0.6]`.
- `RGB MAE` względem nieuczonej inicjalizacji: `0.11563649 -> 0.01843533` (zakres 0–1), względna poprawa **84.06%**.
- Wygenerowano render pośredni oraz porównawczy panel 4 obrazów. Szczegóły: `reports/v05_holdout/holdout_report.json`.

Wszystkie powyższe liczby dotyczą **syntetycznych danych na CPU**, nie znanych benchmarków 4DGS ani realnych obiektów.

## 4. Ograniczenia i status dowodów

- Nowy model to eksperymentalna **optymalizacja czasowo warunkowanych sferycznych splatów**, NIE pełna sieć deformacji z publikacji 4D Gaussian Splatting. Nie ma treningu rzeczywistego zbioru wielokamerowego ani sieci odtwarzającej obszary zasłonięte.
- Nazwa 5D w starszym module nadal oznacza `X,Y,Z,t,cecha wyglądu`; model uczony ma geometrię XYZ i parametr czasu jako warunek generowania wyglądu.
- **Wagi oficjalnego DA3 nie są załączone**, a jego rzeczywista inferencja nie została wykonana: środowisko wykonawcze nie rozwiązuje DNS Hugging Face. Adapter pozostaje testowany kontraktowo, a sprawdzenie oryginalnych sum SHA-256 testowano na negatywnych fixtures.
- Brak testów na rzeczywistych fotografiach, Androidzie, CPU/GPU urządzenia produkcyjnego; brak metrycznej kotwicy i pełnego 400MP out-of-core.
- SHA-256 jest kontrolą integralności, nie podpisem zaufanego wydawcy. Oficjalne digests są wpisane do kodu — aktualizacja upstream wymaga świadomego przeglądu.

## 5. Komendy odtwarzalne

```bash
OMP_NUM_THREADS=1 python -m omega_geodepth.trainable_4dgs train \
  --manifest examples/sequence5d/manifest_from_npz.json \
  --output examples/v05_trainable_4dgs --splats 72 --steps 150 \
  --lr .026 --seed 42 --max-side 32
OMP_NUM_THREADS=1 python scripts/evaluate_v05_holdout.py \
  --checkpoint examples/v05_trainable_4dgs \
  --manifest examples/sequence5d/manifest_from_npz.json \
  --output reports/v05_holdout
python -m pytest -q
bash scripts_verify.sh
```
