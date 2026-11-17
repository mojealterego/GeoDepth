# OCAF GeoDepth Ω

Silnik estymacji głębi, rekonstrukcji wielowidokowej, TSDF, eksperymentalnych Gaussów 4D oraz wieloskalowych pól 5D (XYZ, czas, skala pikselowa). RGB jest cechą, nie wymiarem przestrzennym.

## Stan v0.11.0
Kod źródłowy GeoDepth, testy, narzędzia i dokumentacja są na main. Oficjalny DA3 Small FP32, pochodne FP16/AQ6/AQ4, eksperymentalny INT8 oraz KPlanes5D były uruchomione lokalnie. 152 testy GeoDepth w kompletnej dystrybucji przeszły. Dwa scenowe checkpointy KPlanes5D zostały wytrenowane lokalnie. Model nie przewyższył najbliższego punktu w obu testach RGB; nie stanowi pola radiancji ani modelu nieobserwowanej geometrii.

Pełne archiwa wykonawcze są oddzielnymi artefaktami; ciężkie wagi, dane przykładowe, upstream DA3 ZIP i Representation5D WHL nie są przechowywane w zwykłym Git. Do uruchomienia pełnych testów należy uzupełnić drzewo o te zasoby z dystrybucji v0.11 oraz zainstalować zależności zgodnie z [instrukcją](docs/ZMIANY_v0.11_PL.md). Nie uruchamiano zdalnego CI. Nie nadano nowej licencji kodowi OMEGA. DA3 i upstream Representations4D: Apache-2.0; patrz NOTICE.md.

Dalsze prace: [ROADMAP](docs/ROADMAP.md). Dokumentacja: [README_PL](README_PL.md).
