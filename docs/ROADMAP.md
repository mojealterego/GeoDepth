# GeoDepth Ω — status v0.13.0

## Wykonane lokalnie
- Autentyczny DA3 Small FP32 oraz FP16/AQ6/AQ4 i eksperymentalny INT8 CPU, kontrolowane pochodzenie wag.
- **Autentyczny DA3 Base FP32**: sześć wolumenów odtworzono do checkpointu 541518028 B, SHA-256 e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5, zgodny z upstream. Strict load i rzeczywista inferencja CPU.
- TSDF, Gaussy 4D/5D, Representations5D, eksperymentalne pola RGB KPlanes5D. NN w v0.11 lepszy od pola na obu scenach.
- ENGINE v1: quality_v1, kontrola NPZ, konsensus statyczny, kotwice, raporty metryk i atomowy eksport, most do KPlanes5D.
- v0.13: wszystkie uporządkowane pary kamer, deterministyczny fallback rozdzielczości w CLI i bezpieczne odtworzenie Base.
- Benchmark dwóch zdjęć: Base zgodność najgorszego kierunku 83.56/73.76/81.28/82.58% dla 140/280/392/504; Small 90.98/80.28/85.83/83.60%. Próg 80%, Base 280 odrzucono; CLI 280→392 zaakceptowano. Przy 504 Base: 18815 punktów fuzji wokselowej, 10999 próbek pola 5D. Brak prawdziwego wzorca metrycznego; wynik nie ustanawia globalnej przewagi modelu.
- Pełna lokalna regresja 212/212 testów, 6 ostrzeżeń deprecacji; zbudowano WHL i sprawdzono import poza źródłami. Nakładka 0.13 na pełną dystrybucję 0.12. Wagi Base odtwarzane z sześciu oryginalnych wolumenów, poza zwykłym Git.

## Otwarte bramki
- DA3 Giant: pozyskanie oryginalnego checkpointu, licencja i rzeczywista inferencja.
- Natywne AQ4/AQ6, optymalizacja pamięci i migracja przestarzałych API.
- Dynamiczna radiancja/occupancy, nieobserwowana geometria z niezależną walidacją; przewaga nad NN.
- Rzeczywiste RGB-D z fizyczną referencją, kalibracja niepewności, walidacja wieloscenowa.
- GPU/Vulkan/Android, 400 MP, niezależny pełny audyt inżynieryjny.
- Zdalnego CI nie uruchomiono; brak certyfikacji produkcyjnej.
