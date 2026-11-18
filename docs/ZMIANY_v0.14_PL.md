# GeoDepth Ω v0.14 — Visual Geometry Benchmark i pose conditioning

## Podstawa i granice

Lin i in., *Depth Anything 3: Recovering the Visual Space from Any Views*,
przekazany raport z 13 listopada 2025 r. Sekcje 3.1–3.2, 6.1–6.3 (s. 6–7,
12–13): reprezentacja depth–ray, tokeny kamer, wyrównanie pozy i ocena chmur.
Sekcja 5 (s. 11–12) opisuje dodatkową trenowaną głowicę GS-DPT; nie jest ona
częścią dostarczonych checkpointów Small/Base. Sekcja 8 (s. 22) traktuje sceny
dynamiczne jako dalszy kierunek badań.

Implementacja jest własnym, ograniczonym audytem inspirowanym metodologią
publikacji, **nie reprodukcją oficjalnego benchmarku**. Nie wykorzystuje
oryginalnych zbiorów HiRoom/ETH3D/DTU/7Scenes/ScanNet++; nie porównywać
liczb z tabelami raportu. Nie ma nowego treningu DA3 ani GS-DPT.

## Implementacja

- `omega_geodepth.visual_benchmark`: deterministyczne RANSAC + Umeyama (Sim(3)),
  orientacja right-handed, minimalnie trzy niekolinearne kamery, konsensus
  co najmniej 60% i jawna tolerancja w jednostkach referencji. Brak dopasowania
  do chmury ocenianej, brak dopasowania kamer oceny do referencji.
- AUC3 i AUC30: całka empirycznej dystrybuanty maksymalnego błędu kątowego
  rotacji i kierunku translacji. Para bez referencyjnej bazy nie jest
  oceniana w RTA; brak bazy predykcji przy rzeczywistej bazie karany 180°.
  Kierunek translacji ze znakiem, w lokalnym układzie pierwszej kamery.
- Symetryczny nieskwadrowany Chamfer, accuracy, completeness, precision,
  recall i F1 z jawnym progiem odległości; brak próbek/NaN blokuje ocenę.
- `select_context_views`: deterministyczny podział context/target i farthest
  point sampling z odległością pozycji i kątową; limit 100 kamer w module.
  Nie implementuje renderowania NVS, nie zmienia bramek v0.13.
- `DA3RealCPU.inference(..., camera_poses_c2w=..., camera_intrinsics=...)`:
  wymagane oba wejścia lub żadne; pinhole bez dystorsji/skew. K w pikselach
  obrazu wejściowego, C2W right-handed. Przekazanie do oryginalnego procesora
  DA3 (aktualizacja K przez resize/crop), gauge pierwszej kamery i normalizacja
  medianą bazy zgodnie z upstream API, wywołanie rzeczywistej sieci. Wynik
  zachowuje **arbitrary_scale / relative_estimated** i pole `pose_conditioned`.
  Nie podmienia przewidzianych kamer danymi wejściowymi, nie obiecuje
  automatycznej kalibracji metrycznej. Funkcja dostępna w Python API.

## Odtworzenie

Nakładka na v0.13, bez ponownego dołączania wag. Uzupełnij zasoby z wydań
v0.12, v0.13 (w tym Base wg instrukcji), zainstaluj wymagane zależności
oraz WHL Representations5D. Zachowaj oryginalne źródła DA3 i manifesty.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/benchmark_reference_v14.py --output reports/repeat/free.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/benchmark_reference_v14.py --conditioned --output reports/repeat/conditioned.json
```

Eksperyment 3 warianty jednej rodziny proceduralnej, 8 kamer, 280×280;
4 kamery do kalibracji i 4 do oceny. Wszystkie obrazy są wejściem modelu.
W trybie conditioned znane kamery wszystkich widoków są także wejściem;
nie wolno traktować błędu pozy jako niezależnej predykcji pozy w tym trybie.
Próg konsensusu centrów 0,25 m, F1 0,10 m; chmury próbkowane co 4 piksele,
bez fuzji TSDF. Metry oznaczają jednostki sceny analitycznej po dopasowaniu
referencyjnym, nie pomiar realnego świata. Dane syntetyczne nie są oficjalnym
benchmarkiem DA3; test nie jest NVS ani testem nieobserwowanej geometrii.
Wyniki obu trybów w `reports/v14/`. Nie wybierać modelu po pojedynczej metryce.

## Wynik lokalny (CPU, 1 wątek, oryginalne wagi)

Średnie po trzech scenach; wszystkie sześć dopasowań w każdym trybie przeszło
konsensus centrów. Wartości F1 i AUC podano w [0,1]:

| Model / tryb | F1@0,10m ↑ | CD m ↓ | AbsRel ↓ | AUC30 ↑ |
|---|---:|---:|---:|---:|
| Small bez kamer | 0,018938 | 0,840725 | 0,168201 | 0,789822 |
| Small z kamerami | 0,096646 | 1,250523 | 0,273974 | 0,907224 |
| Base bez kamer | 0,037050 | 1,206757 | 0,291563 | 0,519369 |
| Base z kamerami | 0,000000 | 2,999681 | 0,730103 | 0,892624 |

Wyniki ujawniają słabą zgodność przestrzenną w tej rodzinie syntetycznej;
nie ma podstaw do automatycznego przełączenia na pose conditioning ani do
uznania któregokolwiek modelu za dokładny metrycznie. AUC w trybie z kamerami
nie jest niezależnym testem pozy. Wydanie nie podnosi deklaracji jakości.

Regresja pełnej lokalnej dystrybucji: 223 passed, 6 warnings. Jedenaście nowych
testów (9 benchmarku, 2 conditioning), rzeczywisty Small/Base, zbudowany WHL
0.14.0, instalacja i smoke test poza źródłami. Zdalnego CI nie uruchamiano.
