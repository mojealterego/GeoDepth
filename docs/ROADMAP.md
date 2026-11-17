# GeoDepth Ω — status v0.12.0

## Wykonane lokalnie
- DA3 Small FP32, FP16, AQ6/AQ4 oraz eksperymentalny INT8 CPU, TSDF i Gaussy 4D/5D.
- Reprezentacje 5D, mosty i eksperymentalne pole RGB KPlanes5D. W v0.11 NN był lepszy od neuronowego pola RGB na obu testowych scenach.
- ENGINE v1: osobny moduł quality_v1, przewidywania NPZ w dwóch schematach, walidacja przed alokacją, konsensus statyczny, per-view coverage, ponowny audyt po korekcie, kotwice, raporty i eksport bez nadpisania.
- Testy: oryginalny ENGINE v1 — 45; adaptacja — 49; pełna regresja v0.12 — 201. Lokalny build WHL i instalacja poza źródłami.
- Rzeczywisty Small FP32 na dwóch fotografiach: szacowana zgodność reprojekcji 86,70% → 87,12%, bez prawdziwej referencji metrycznej; 7006 wokseli, 3241 zaakceptowanych próbek statycznego pola. Pojedynczy syntetyczny test płaszczyzny RMSE 0,0500 → 0,0276. Nie utożsamiać z generalizacją na realne dane.
- Base FP32: dostarczona konfiguracja z pinem SHA i manifestem, wybór modelu oraz test odmowy bez checkpointu. Wagi nieobecne, nie wykonano Base E2E.

## Otwarte bramki
- Oryginalne wagi DA3 Base i Giant, ich właściwa inferencja oraz przestrzeganie odrębnych licencji.
- Natywna inferencja AQ4/AQ6, peak RSS, migracja przestarzałych API kwantyzacji.
- Dynamiczna radiancja/occupancy, nieobserwowana geometria z niezależną walidacją; poprawa ponad baseline NN.
- Dane RGB-D z fizyczną referencją, kalibracja niepewności, walidacja wieloscenowa.
- GPU/Vulkan/Android, 400 MP, niezależny audyt wymagań.
- CI: nieuruchomione; nie deklarujemy produkcyjnego wdrożenia.

Duże zasoby wykonawcze pozostają poza zwykłym Git.
