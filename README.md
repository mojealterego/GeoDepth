# OCAF GeoDepth Ω

Silnik estymacji głębi, rekonstrukcji wielowidokowej i eksperymentalnej reprezentacji 5D (XYZ + czas + atrybut wyglądu).

## Stan
Wersja lokalna: 0.10.0. Oficjalny DA3 Small FP32, pochodne FP16/AQ6/AQ4 i eksperymentalna dynamiczna kwantyzacja INT8 CPU. Głowice głębi i kamery pozostają FP32. Głębia względna nie jest pomiarem metrycznym.

Repozytorium jest inicjalizowane na `main`. Duże modele i kompletne archiwa wykonawcze nie są przechowywane w zwykłym Git; pełna dystrybucja offline jest oddzielnym artefaktem. Nie nadano nowej licencji kodowi OMEGA. DA3: ByteDance Seed, Apache-2.0.

Otwarte bramki: DA3 Base/Giant, natywny AQ4/AQ6, pełna rekonstrukcja dynamiczna produkcyjna, fizyczna walidacja metryczna, GPU/Android.
