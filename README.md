# OCAF GeoDepth Ω

DA3, rekonstrukcja wielowidokowa, TSDF, eksperymentalne Gaussy 4D oraz wieloskalowe pola 5D (XYZ, czas, skala pikselowa). RGB jest cechą, nie wymiarem przestrzennym.

## Stan v0.13.0
Oryginalne checkpointy DA3 Small i Base zostały zweryfikowane i uruchomione na CPU FP32. Base: 541518028 B, 135366599 parametrów. Dodano odtwarzanie sześciu wolumenów Base z weryfikacją SHA-256 i kontrolowany fallback rozdzielczości z obustronnym audytem wszystkich par kamer. Progi jakości pozostają niezmienione. Na dwóch fotografiach Base 280 odrzucono, 392 zaakceptowano. Nie jest to niezależna walidacja metryczna. Small pozostaje domyślnym modelem.

Pełna lokalna regresja z zasobami AQ/Representations5D i Base: 212 testów, 6 ostrzeżeń. Wagi, źródła upstream, dane przykładowe i archiwa wykonawcze są oddzielnymi artefaktami. Zdalnego CI nie uruchomiono; sam checkout repozytorium nie zawiera wszystkich zasobów pełnej regresji. Wydanie lokalne 0.13 jest nakładką na dystrybucję 0.12.

- [Instrukcja v0.13](docs/ZMIANY_v0.13_PL.md)
- [README PL](README_PL.md)
- [Roadmap](docs/ROADMAP.md)
- [Benchmark Base](reports/v13/base.json)
- [Benchmark Small](reports/v13/small.json)

Licencje: patrz NOTICE.md. DA3 i moduł quality_v1: Apache-2.0. Nie nadano nowej licencji reszcie kodu OMEGA.
