# GeoDepth Ω v0.11.0 — pole wieloskalowe 5D

Zintegrowano dostarczony Representations5D 0.2.0+omega.5d (Apache-2.0 dla upstream). Oryginalny ZIP, wheel, licencja i syntetyczny checkpoint są zachowane w vendor. Model pola sceny używa dziesięciu płaszczyzn KPlanes5D na znormalizowanej kostce jednostkowej; metryczne API upstream nie jest używane do zapisu względnych współrzędnych. Nowy, osobny schemat Safetensors `omega.geodepth.scene_field/1` przechowuje transformację, jednostki, pochodzenie, podstawę czasu i raport. Loader kontroluje kształty, typy, skończoność i stałe bufory normalizacji. Pole przewiduje RGB na dostarczonych współrzędnych, nie jest polem radiancji ani zajętości, nie odtwarza geometrii niewidzianej sceny. Współrzędne: XYZ, czas, log2(Z/(sqrt(fx*fy)*r_ref)). RGB stanowi cechę. Jednostki względne zachowane; dla trybu statycznego wszystkie czasy są równe 0. Warstwa align_baseline dopasowuje globalną skalę z co najmniej dwóch odległości kamer dostarczonych przez użytkownika, odrzuca niespójne kotwice i skaluje zarówno Z, jak i translacje kamer. Nie potwierdza autentyczności pomiarów ani prawdziwej dokładności metrycznej.

## Instalacja lokalna

```sh
python -m pip install --no-deps 'vendor/representations4d-0.2.0+omega.5d-py3-none-any.whl'
python -m pip install -e .
```

Zakres `--no-deps` dotyczy tylko wybranych podmodułów: NumPy/SciPy/PyTorch/Safetensors. Wymagania DA3 pozostają opisane w poprzednich wydaniach.

## Trening

```sh
python scripts/train_scene_field.py --source npz --manifest examples/sequence5d/manifest_from_npz.json --out /tmp/geo-syn --time-basis capture_seconds --reference-radius .025 --holdout 1 --split time --steps 400 --stride 2
python scripts/train_scene_field.py --source da3 --manifest examples/v08_real_da3/manifest.json --model-dir models/da3 --da3-source vendor/da3_src --out /tmp/geo-photo --time-basis static_assumption --reference-radius .025 --holdout 1 --steps 500 --stride 3
```

Opcjonalna kotwica `--baseline-evidence` ma postać JSON `{"evidence_id":"identyfikator pomiaru","baselines":[[0,1,0.5],[0,2,1.0]]}`; wymaga rzeczywistych pomiarów i spójnego układu względnego. Przykładowych liczb nie należy używać jako pomiaru realnych zdjęć. Timestamps w trybie capture_seconds muszą pochodzić od użytkownika. Wybranie czasu statycznego jest założeniem, nie detekcją statyczności.

## Wyniki, ograniczenia

- 152 testy GeoDepth i mostu z wszystkimi profilami DA3; 6 ostrzeżeń PyTorch dot. przestarzałego interfejsu kwantyzacji.
- Syntetyczna scena, całkowicie odłożona klatka: RGB MSE 0.0007100344 (najbliższy punkt: 0.0004155142).
- Dwa rzeczywiste zdjęcia DA3, jeden widok bez treningu RGB: MSE 0.0353029272 (najbliższy punkt: 0.0303406219). Reprojekcja DA3 przeszła. Geometria pochodzi z obu obrazów, a granice domeny ze wszystkich współrzędnych, więc nie jest to niezależna walidacja novel-view synthesis. Oba testy używają względnej geometrii.
- Modele scenowe 30387 parametrów, ponowny odczyt Safetensors sprawdzony. Nie przewyższają NN na tych przykładach; nie promować do domyślnego renderera. Oś czasu obrazu rzeczywistego jest stała; brak testu ruchu na rzeczywistym filmie.
- Brak nowej inferencji DA3 Base/Giant, natywnych kerneli AQ4/AQ6, walidacji GPU/Android/metryki fizycznej.
