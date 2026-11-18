# OCAF GeoDepth Ω

DA3, rekonstrukcja wielowidokowa, TSDF, eksperymentalne Gaussy 4D oraz wieloskalowe pola 5D (XYZ, czas, skala pikselowa). RGB jest cechą, nie wymiarem przestrzennym.

## Stan v0.16.0
Oryginalne DA3 Small i Base działają lokalnie w FP32 CPU. Zachowano opcjonalne profile Small FP16/AQ6/AQ4/INT8, kontrolę metryki, TSDF, modele 4D/5D i bramki jakości. v0.16 dodaje eksperymentalny fast_ray, globalny graf Sim(3) i obsługę statycznej rekonstrukcji okienkowej. Dekoder tokenowy pozostaje domyślny. Na jednej scenie przy 140 px fast_ray przyspieszył inferencję 3,17–8,07× względem natywnego ray. Graf zredukował koszt optymalizacji, ale oba rzeczywiste zestawy predykcji zostały odrzucone przez bramki jakości. Nie jest to dowód dokładności metrycznej.

242 testy lokalnie, 12 nowych; WHL z niezależnym uruchomieniem Small. Kod, testy, dokumentacja i raport v0.16 są na main. Duże wagi i zależności upstream są nadal poza Git; sam checkout nie zawiera wszystkich zasobów pełnej lokalnej regresji. Zdalnego CI nie uruchomiono.

## Zasady publikacji
Każda kolejna iteracja ma trafiać na main: źródła, testy, dokumentacja, roadmapa i dostępne raporty. Publikacja wymaga weryfikacji plików i odczytu końcowego SHA gałęzi. Zasoby binarne wymagają osobnego mechanizmu transferu (Git LFS lub Releases); nie należy ich zastępować pustymi wskaźnikami. Nie deklarować publikacji plików, których repozytorium rzeczywiście nie zawiera.

- [Instrukcja v0.16](docs/ZMIANY_v0.16_PL.md)
- [Benchmark v0.16](reports/v16/benchmark_140.json)
- [Instrukcja v0.15](docs/ZMIANY_v0.15_PL.md)
- [README PL](README_PL.md)
- [Roadmap](docs/ROADMAP.md)

Licencje: NOTICE.md; DA3 i quality_v1 Apache-2.0. Nie nadano nowej licencji reszcie kodu OMEGA.

## Dystrybucja modeli
Oryginalne wagi i eksporty TorchScript OMEGA v0.1/v0.2 są już zapisane w repozytorium w numerowanych częściach. Zobacz [status dystrybucji](docs/MODEL_DISTRIBUTION.md) i [manifest](models/repository_inventory.json). Pięć dużych checkpointów DA3 nadal oczekuje na transfer pełnych bajtów.
