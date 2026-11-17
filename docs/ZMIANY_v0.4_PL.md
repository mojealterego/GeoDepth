# OMEGA GeoDepth Ω v0.4.0 — raport zmian i dowody

**Wydanie:** 2026-11-15. **Typ:** eksperymentalne offline; bez operacji na zewnętrznych repozytoriach. **Model bazowy:** syntetyczny checkpoint OMEGA v0.2, pozostawiony bez zmian. **Wagi DA3:** nieobecne (blokada pobrania przez DNS). **Pełny wytrenowany 4DGS/5D NeRF:** nieobecny.

## Zmiany rzeczywiście wprowadzone

1. `multiview_tsdf.py`: projektowanie przestrzeni z wielu skalibrowanych pozycji kamer do jednego **statycznego** wolumenu TSDF; pojedyncze obserwacje i wielokrotne pokrycie mierzone oddzielnie, ochrona nieobserwowanych wokseli, maski statyczności, kontrola sprzecznych pomiarów, granice pamięci, filtrowanie punktów za powierzchnią.
2. `reconstruction5d.py`: opcjonalne statyczne maski klatek, obustronne unikalne skojarzenia punktów (mutual nearest neighbor) i bramka fotometryczna przy scene-flow; wyjściowy strumień 5D zawiera XYZ + czas + przybliżoną cechę koloru.
3. `gaussian5d.py`: rzeczywista CPU rasteryzacja obserwowanych surfeli gaussowskich, Z-buffer i przezroczystość, punkty warpowane w czasie, eksport NPZ i PNG/NPZ dla chwil pośrednich; **to nie jest trenowana 4D Gaussian Splatting**.
4. `render5d.py`: dla czasu pośredniego wybiera najbliższą klatkę i warpuje tylko punkty z dopasowanym ruchem, nie dopowiada obszarów niewidzianych.
5. Regresja: wykryto i naprawiono mutowanie maski `valid` w oryginalnym `Frame5D` przez inplace `&=` w procedurze integracji.
6. `save_scene5d` i CLI: `--fused-static-tsdf`, `--gaussian-splats`, raport SHA-256, obserwacje wielowidokowe, raport rozdzielony na dane wejściowe i model.

## Metodyka

Przed implementacją dodano testy, które nie mogły się zebrać z powodu brakującego modułu. Oddzielny test mutowalności wykazał 1024 zmienione piksele w 32×32, następnie po poprawce przeszedł. Testy obejmują topologię i projekcję powierzchni, pokrycie kamer, maskowanie, bezpieczeństwo wejść i artefaktów, rasteryzację gaussowską i pełną sekwencję końcową.

**Zakres wiarygodności:** testy i pomiary wykonywane na syntetycznych klatkach. Nie dowodzą metrycznej dokładności rzeczywistych kamer, ani działania DA3 na rzeczywistych wagach. Testowanie oryginalnych wag jest otwartą bramką.

## Architektura wymiarów

W tej wersji 5D = `(world_x, world_y, world_z, time_seconds, normalized_sRGB_luma_proxy)`; piąty wymiar jest cechą obserwacyjną, nie wymiarem przestrzennym. Geometryczna rekonstrukcja dynamiczna nadal wymaga kalibracji, gęstych korespondencji, sprawdzonych masek ruchu i danych z wielu widoków.

## Obsługa NSFW i abliteracji

System **nie ma klasyfikatora treści NSFW**, wnioskowanie o głębi jest agnostyczne względem tematyki fotografii. Abliteracja wektorów odmowy modeli LLM nie ma odpowiadającej operacji treningowej dla DA3 i nie jest deklarowana.

## Blokady wydania

- Oryginalne wagi `depth-anything/DA3-SMALL` (~137MB) i `DA3-BASE` nie znalazły się w archiwum: `huggingface.co` nie rozwiązuje się z lokalnego środowiska wykonawczego. Downloader jest dostępny do użycia z siecią i sprawdza pliki SHA-256.
- Nie potwierdzono wydajności/kompatybilności GPU, Android, Termux; lokalna weryfikacja CPU, x86_64/Python 3.13.
- Brak treningu dopasowanego do rzeczywistych zdjęć i dynamicznego renderera 4DGS; rasterizer jest aproksymacją surfeli.
- Rekonstrukcja statycznej sceny z ruchomą kamerą jest testowana, ale nie zastępuje pełnego bundle adjustment lub prawdziwego rozdziału dynamic/static.
