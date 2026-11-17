# Karta modelu: OMEGA GeoDepth Ω 0.2.0

- **ID modelu:** `omega_geodepth_synthetic_v0.2`.
- **Architektura:** residual encoder-decoder (U-Net), przewidywanie `log_relative_depth` i `log_variance`; szerokość 24; 1 295 234 parametrów.
- **Trening:** od zera na 896 proceduralnych scenach raycast 64×64, 1500 iteracji AdamW; żadne zewnętrzne wagi nie były użyte.
- **Podział danych:** walidacja seed 50000–50127, benchmark niezależny seed 80000–80127, kalibracja 60000–60111, audyt kalibracji 70000–70111. Brak wspólnych seedów.
- **Błąd modelu na 128 scenach niezależnych:** MAE log-depth 0.102475, wobec 0.113674 dla modelu 0.1.
- **Cel:** eksperymentalna geometria głębi względnej i jej wizualizacja. Nie jest to SOTA ani pomiar metryczny bez niezależnych sensorów.
- **Materiał NSFW:** w kodzie brak klasyfikatora i blokowania treści. Jakość na zdjęciach osób, nagości, luster, ekranów, włosów i innych obiektach rzeczywistych nie została zmierzona.
- **Bezpieczeństwo:** wczytywanie Safetensors ze sprawdzeniem sumy SHA-256 i ścisłym zestawem wag; nie jest to podpis cyfrowy. Brak automatycznych pobrań sieciowych.
- **Niepewność:** log-wariancja po treningu + kalibracja *tylko* direct64 na syntetycznej dystrybucji. Inny rozmiar, dane rzeczywiste i tile to inne domeny.
- **Ograniczenia:** brak oceny OOD, brak walidacji metrycznej, 24 MP limit pliku źródłowego, przetwarzanie RGB w pamięci, brak full 400MP, tryb mono.
- **Niezależność:** nie jest modelem DA3 ani modelem językowym Abliterated; nie stosowano ablacji wektorów odmowy.

Artefakty: `models/manifest.json` oraz `reports/benchmark_v02.json` są wiążącymi dowodami liczb. Każdą deklarację skuteczności na zdjęciach rzeczywistych należy poprzedzić osobnym testem z odpowiednim ground truth.

## Dodatek 0.3.0

Paczka **nie trenuje ponownie** modelu OMEGA: identyczne checkpointy v0.2 zostały zachowane dla kompatybilności i testów regresji. Dodane warstwy DA3 i Scene5D nie nadają wagom OMEGA v0.2 zdolności rozumienia prawdziwych filmów.

Oficjalne DA3 Small/Base są obsługiwane przez osobny *adapter offline* z kontrolą kształtów, konwencji kamer i SHA-256, lecz ich plików wag nie pobrano ani nie przetestowano sprzętowo. Algorytm 5D to wynik deterministycznego back-project i integracji chmur oraz TSDF z obserwowanych map Z, nie dodatkowy wytrenowany model.


## Dodatek v0.4.0 — zakres modułów geometrycznych

Wagi bazowej sieci OMEGA v0.2 pozostały bez zmian; nie przeprowadzono nowego treningu modelu neuronowego. Dodano CPU Gaussian Surfel Rasterizer i statyczny Multiview TSDF, uruchamiane na wynikach z przetwornika geometrii. Nie dołączono oryginalnych wag DA3 i nie uruchomiono ich inferencji, pomimo dostępnego adaptera API. Wyniki błędów na syntetycznej płaszczyźnie dotyczą tylko poprawności rzutowania, nie generalizacji do realnych zdjęć.

## Dodatek v0.5.0 — uczony model dynamicznych Gaussów

Drugi checkpoint `examples/v05_trainable_4dgs/gaussians.safetensors` to **odrębny od OMEGA DepthNet i DA3**, trenowany od zera na 3 syntetycznych klatkach: 72 gaussy ze zmiennymi pozycjami, prędkością, wielkością, opacity i RGB. Jest to mała scenowa optymalizacja parametrów, nie duży model fundamentalny. Metryki treningowe i dwa punkty czasowe testu niewidziane podczas treningu są w `docs/ZMIANY_v0.5_PL.md` i `reports/v05_holdout/holdout_report.json`. Zdolności generalizacji na inne ujęcia nie wykazano.

## Dodatek v0.6.0 — nieliniowa reprezentacja czasowa

Dodano oddzielny model z `knots × gaussians × XYZ` uczonymi przesunięciami,
komponowanymi z liniowym wektorem prędkości. Względem v0.5 różnica jest
reprezentacyjna, **nie** nowy fundamentalny model głębi DA3. Checkpoint nowej
sceny pochodzi z syntetycznego przykładu i ma osobny identyfikator schematu
`omega-gaussian-deformable-v2`. Wsteczny odczyt v0.5 utrzymano.
Testy audytowe, metryki i ograniczenia: `examples/v06_deformable/benchmark.json`.
Brak gwarancji skali metrycznej, nieobserwowanej geometrii lub real-world quality.


## Dodatek v0.7.0 — DA3 user-supplied source ZIP
Dostarczone oryginalnie zapakowane źródła DA3 objęte deklarowaną licencją Apache-2.0 (szczegóły `vendor/Depth-Anything-3-main.zip`/LICENSE). SHA-256 paczki użytkownika nie jest poświadczeniem upstream. Kod zachowany bez modyfikacji. Oryginalnych wag DA3 nadal brak. Nowy audyt reprojekcji kontroluje wyłącznie względną spójność wielowidokową, nie dokładność metryczną. Patrz `docs/ZMIANY_v0.7_PL.md`.


## Dodatek v0.8.0 — DA3 Small z autentycznymi wagami

Oddzielny model `DA3-SMALL` (34,3 mln parametrów, Apache-2.0) został
wczytany w oryginalnej architekturze i uruchomiony na CPU na 2 obrazach
fotograficznych z repozytorium DA3. Przesłane `model.safetensors`
odpowiada upstream SHA-256 `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf`.
Oryginalne źródła w ZIP zachowano niezmienione. Szczegóły i ograniczenia:
`docs/ZMIANY_v0.8_PL.md` oraz `examples/v08_real_da3/run_report.json`.
Nie jest to model OMEGA Synthetic i jego wyników nie należy mieszać
z benchmarkiem v0.2 na syntetycznym raycaście.
