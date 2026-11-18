# GeoDepth Ω — status v0.15.0

## Potwierdzone lokalnie
- Oryginalne DA3 Small i Base FP32, kontrola SHA-256 i strict-load; Small FP16/AQ4/AQ6, eksperymentalny dynamiczny INT8 CPU.
- TSDF, eksperymentalne Gaussy 4D/5D, Representations5D, KPlanes5D: nie wykazano przewagi pola RGB nad NN.
- ENGINE v1, secure NPZ, konsensus, odtwarzanie Base z sześciu wolumenów, fallback rozdzielczości.
- v0.14: benchmark referencyjny Sim(3), F1/Chamfer/AUC, rzeczywiste pose conditioning, 223 testy.
- v0.15: natywny ray decoder opcjonalnie, audyt p90 kierunku/początku i normalizacji osi Z, dodatkowa kontrola rotacji i centrów na rozłącznych kotwicach; 230 testów, 6 ostrzeżeń, rzeczywista inferencja z niezależnie zainstalowanego WHL.
- 8 przebiegów syntetycznych przy 140 px: native ray poprawia p90 kątowe, ale F1 pozostaje niskie. W obu trybach Small ze znanymi kamerami przeszedł nową bramkę kamer. Ustawienia domyślne nie zmienione. Dodatkowy 280 px nieukończony (3/4).
- Szczegółowe wyniki tabelaryczne: docs/ZMIANY_v0.15_PL.md; surowy JSON v0.15 w ZIP. Nie uruchamiano zdalnego CI.

## Otwarte bramki
- v0.16: ograniczony koszt dekodera promieni, globalna optymalizacja pozy/geometrii (bez obniżania progów i bez fałszywej skali).
- v0.17: trenowana głowica GS-DPT / osobny eksperymentalny moduł, walidacja NVS z rozłącznymi widokami.
- v0.18: pełniejsza dynamiczna rekonstrukcja 4D/5D.
- v0.19: niezależne realne RGB-D z fizyczną referencją, wieloscenowa dokładność, kalibracja niepewności.
- v0.20: natywne AQ4/AQ6, optymalizacja pamięci, GPU/Vulkan/Android i 400 MP.
- DA3 Giant/Large: brak checkpointów; ograniczenia licencyjne przed integracją.
- Niezależny audyt końcowy, CI, brak certyfikacji produkcyjnej.
