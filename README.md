# OCAF GeoDepth Ω

Silnik DA3, rekonstrukcji wielowidokowej, TSDF, eksperymentalnych Gaussów 4D i wieloskalowych pól 5D (XYZ, czas, skala pikselowa). RGB jest cechą, nie dodatkowym wymiarem przestrzennym.

## Stan v0.12.0
Zintegrowano dostarczony ENGINE v1 jako odrębny moduł jakości: kontrolowany import NPZ, statyczny konsensus głębi, kontrolę pokrycia i paralaksy, kotwice metryczne, atomowy eksport oraz most do KPlanes5D. Pełna lokalna regresja wraz z zasobami AQ: 201 testów (6 ostrzeżeń deprecacji). DA3 Small FP32, pochodne FP16/AQ6/AQ4 i eksperymentalny INT8 pozostają w projekcie. Dodano konfigurację DA3 Base i obsługę wyboru Base FP32; brak rzeczywistego checkpointu Base i jego inferencji.

Źródła pierwszoplanowe znajdują się na main. Duże wagi, zależności upstream, pliki przykładowe i kompletne archiwa wykonawcze są oddzielnymi artefaktami. Samo repozytorium nie zawiera wszystkich zasobów potrzebnych do pełnej regresji. Zdalnego CI nie uruchomiono.

- [Instrukcja v0.12](docs/ZMIANY_v0.12_PL.md)
- [README PL](README_PL.md)
- [Roadmap](docs/ROADMAP.md)

Licencje: patrz NOTICE.md. Moduł quality_v1 zachowuje licencję Apache-2.0 dostarczonego ENGINE v1. Nie nadano nowej licencji reszcie kodu OMEGA.
