# OCAF GeoDepth Ω

DA3, rekonstrukcja wielowidokowa, TSDF, eksperymentalne Gaussy 4D i wieloskalowe pola 5D (XYZ, czas, skala pikselowa). RGB jest cechą, nie wymiarem przestrzennym.

## Stan v0.15.0
Autentyczne DA3 Small i Base są uruchomione lokalnie w FP32 CPU. Opcjonalne profile Small FP16/AQ6/AQ4/INT8, kontrola metryki, TSDF, modele 4D/5D i bramki jakości pozostają zachowane.
v0.14 wprowadziło benchmark referencyjny i pose conditioning; v0.15 dodaje natywny dekoder ray, audyt kierunku/origin/Z oraz ściślejsze dopasowanie Sim(3). Dekoder tokenowy pozostaje domyślny. 230 testów lokalnie, 6 ostrzeżeń; zbudowano WHL 0.15.0 i uruchomiono CLI ray na 2 fotografiach spoza źródeł. Wynik syntetyczny nie potwierdza fizycznej dokładności ani przewagi modelu.

Kod i testy są na main. Pełne raporty v0.15, wagi i zależności upstream pozostają w lokalnej dystrybucji; sam checkout nie wystarcza do pełnej regresji. Zdalnego CI nie uruchomiono.

- [Instrukcja v0.15](docs/ZMIANY_v0.15_PL.md)
- [Instrukcja v0.14](docs/ZMIANY_v0.14_PL.md)
- [README PL](README_PL.md)
- [Roadmap](docs/ROADMAP.md)

Licencje: NOTICE.md; DA3 i quality_v1 Apache-2.0. Nie nadano nowej licencji reszcie kodu OMEGA.
