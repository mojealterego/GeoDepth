# GeoDepth Ω v0.15 — Depth–Ray, niezależna kontrola skali i kamer

Własne rozszerzenie audytowe na podstawie raportu DA3, sekcji 3.1 i 6 (13.11.2025). Pinned upstream ma w kodzie `ray_utils.camray_to_caminfo` kolejność **direction, origin** (raport notuje origin, direction). Zachowano natywną sieć oraz oficjalne checkpointy; nie trenowano DA3 i nie dodano GS-DPT.

## Zmiany
- `DA3RealCPU.inference(..., pose_decoder='ray', audit_depth_rays=True)` uruchamia oryginalną ścieżkę odzyskiwania kamer z promieni. Domyślnie `token` tak jak dotychczas. Opcjonalny audyt przechwytuje kopię surowych wyjść Dual-DPT przed ich usunięciem przez dekoder. Zwalnia hook również po wyjątku; blokada per instancja serializuje forward z hookiem. Ustawienie ziarna 42 i fork_rng ogranicza zmiany stanu RNG dla sekwencyjnego użycia; współbieżność pomiędzy różnymi instancjami/RNG wymaga zewnętrznej synchronizacji. GPU nie testowano.
- `audit_rays` raportuje per-view p90 odchylenia kątowego promienia od kamery, odległości początku promienia od centrum (w relacji do mediany Z) i odchylenia składowej osiowej od 1. Nie normalizujemy promieni w sposób ukrywający błąd Z. Próbkowanie deterministyczne, limit 4096 domyślnie, wymagane >=80% ważnych próbek na widok. Powyższe są kontrolami wewnętrznej spójności, nie kalibracją prawdopodobieństwa ani geometrią referencyjną.
- `reconcile_camera_gauge`: osobna operacja dopasowania Sim(3) wyłącznie na wskazanych co najmniej 3 niekolinearnych kotwicach i kontroli wszystkich kamer, w tym odłożonych, względem niezależnie zadanej tolerancji centrum i kąta rotacji. Zwraca przeskalowaną głębię i wyrównane przewidywane pozy (nie podstawia kamer wejściowych ani K), bez zmiany wyniku surowego. Jednostki `reference_units_estimated`, nie deklaracja metrów czy pomiaru. Nie dowodzi poprawności powierzchni; jakość zależy od referencji i kalibracji.
- CLI: `--pose-decoder ray --audit-depth-rays`, również przy istniejącym fallback rozdzielczości. Dane wyjściowe nadal `arbitrary_scale / relative_estimated`.

## Test i benchmark
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. pytest -q`: 230 passed, 6 warnings. Siedem nowych testów (w tym prawdziwy Small/ray). Zachowane checkpointy modeli, pełna regresja z Base i AQ.

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/benchmark_rays_v15.py --resolution 140 --seeds 11 --output reports/repeat/v15.json`

Jedna scena proceduralna, 8 kamer, ten sam seed 11 co w v0.14, wszystkie widoki są wejściem modelu, a w trybie conditioned także wszystkie znane kamery. Kalibracja 0–3, ocena 4–7. Wyrównanie referencyjne, F1@0,10 jednostki metrycznej sceny syntetycznej; kontrola centrum 0,25 i orientacji 5°. Nie jest to pomiar rzeczywistego świata ani oficjalny benchmark; brak generalizacji i NVS. Osiem przebiegów przy 140 px ukończono. `reports/v15/small.json` to **niekompletny** dodatkowy test 280 px (3/4 wariantów z powodu timeoutu), nie służy do agregacji z 140 px.

| Model | Dekoder | Znane kamery | F1 | najgorsze p90 kątowe | bramka kamer |
|---|---|---|---:|---:|---|
| Small | token | nie | .030022 | 9.920° | odmowa |
| Small | token | tak | .039814 | 14.862° | akceptacja |
| Small | ray | nie | .055411 | .731° | odmowa |
| Small | ray | tak | .074937 | .943° | akceptacja |
| Base | token | nie | .006855 | 12.930° | odmowa |
| Base | token | tak | .066276 | 15.989° | odmowa |
| Base | ray | nie | .011454 | .663° | odmowa |
| Base | ray | tak | .063414 | .741° | odmowa |

Wszystkie wartości F1 niskie. Dekoder ray poprawił wewnętrzny kąt, ale nie zawsze F1. Koszt RANSAC istotny: czasy procesu współdzielą CPU z regresją, więc nie są wiarygodnym benchmarkiem wydajności. Przy 280 px Small bez kamer: token F1 .025266, ray F1 0; kąty 4.598° -> .797°. Nie zmieniono progów ani domyślnego dekodera. Nie rozwiązano problemu dokładności metrycznej na realnych danych i nie osiągnięto certyfikacji produkcyjnej.

## Instalacja
Nakładka kumulatywna na v0.13 (zawiera również źródłowe zmiany v0.14). Wagi pozostają w istniejących dystrybucjach i wolumenach Base. Zainstaluj zależności według wcześniejszej instrukcji, rozpakuj aktualizację do katalogu projektu, uruchom testy. Stary manifest v0.12/0.13 opisuje wcześniejszą wersję; manifest nakładki v0.15 dotyczy aktualizowanych plików, nie całej dystrybucji. Nie uruchamiano zdalnego CI.
