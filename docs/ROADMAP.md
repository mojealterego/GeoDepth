# GeoDepth Ω — status implementacji v0.10.0

## Wykonane lokalnie
- Oficjalny DA3 Small FP32; pochodne FP16, AQ6 i AQ4.
- Dynamic INT8 CPU: 48 warstw Linear backbone, głowice FP32.
- Walidacja źródła i checkpointów, eksperymentalna rekonstrukcja 5D, modele Gaussów 4D.
- Pytest: 139 zaliczonych testów w kompletnym wydaniu lokalnym, w tym 5 nowych.
- Dwie fotografie: odchylenie względnej głębi INT8 od FP32 3,30%, zmiana K maks. 8,53 px.
- Redukcja ładunku wag 48 warstw: 84 934 656 → 21 233 664 B. Nie stwierdzono istotnej redukcji szczytowego RSS całego procesu.

## Stan GitHub
Na main opublikowano moduł INT8, jego testy, dokumentację oraz patch integracyjny v0.9 → v0.10. Pełna paczka lokalna zawiera pozostały kod, zależności źródłowe i wagi. Samo repozytorium nie stanowi jeszcze kompletnego wydania i nie uruchamia pełnych testów bez importu źródeł i danych.

## Otwarte bramki
- Import reszty źródeł na main; bez publikowania dużych modeli w zwykłym Git.
- Redukcja pamięci szczytowej, migracja do utrzymywanego API kwantyzacji.
- Natywne kernele AQ4/AQ6, DA3 Base/Giant z prawdziwymi wagami.
- Produkcyjna rekonstrukcja dynamiczna, walidacja metryczna, Android/GPU.
- Niezależna walidacja na wielu scenach ze znaną geometrią i audyt wszystkich wymagań.
