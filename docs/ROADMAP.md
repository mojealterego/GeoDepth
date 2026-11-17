# GeoDepth Ω — status v0.11.0

## Wykonane lokalnie
- DA3 Small FP32, FP16, AQ6/AQ4 i eksperymentalny INT8 CPU; zachowane TSDF, Gaussian 4D/5D i kontrole reprojekcji.
- Most Representations5D i pole RGB KPlanes5D, 30387 parametrów, odrzucone ekstrapolacje poza domenę, odrębne jednostki względne/metryczne, zewnętrzne kotwice bazowe.
- Trening na syntetycznej sekwencji (odłożony cały czas) i dwóch rzeczywistych zdjęciach DA3 (odłożony kolor widoku, współrzędne dostarczone). Wyniki RGB MSE 0.000710 i 0.035303; prosty NN lepszy w obu eksperymentach.
- Pytest w pełnym wydaniu: 152 testy; build i niezależna instalacja WHL 0.11.0. Nie oznacza pełnego audytu inżynieryjnego ani wyników produkcyjnych.

## Stan dystrybucji
- Źródła pierwszoplanowe i testy przeniesione na main; wagi, duże zasoby, licencjonowane upstream archiwa i WHL pozostają w wydaniu offline.
- Nie przeprowadzano zdalnego CI. Weryfikacja lokalna obejmuje pełne zasoby dystrybucji.

## Otwarte bramki
- Wydajność pamięci i migracja torch.ao do utrzymywanego API, natywne kernele AQ4/AQ6.
- Oficjalne checkpointy i inferencja DA3 Base/Giant.
- Fizyczna walidacja metryczna, wieloskalowe dane realne RGB-D z rzeczywistymi czasami i pomiarami kalibracyjnymi.
- Wytrenowana dynamiczna radiancja/occupancy, weryfikacja na całkowicie niewidzianych widokach bez geometrii z widoku testowego; model ma pokonać NN na oddzielnym zbiorze.
- GPU/Vulkan/Android i 400 MP.
- Niezależny audyt pełnej specyfikacji oraz CI.

Historia: v0.10 — dynamiczny INT8 48 Linear, redukcja wag tych warstw 75%, brak istotnej redukcji RSS procesu; odchylenie map 3,30% względem FP32. Profil nadal eksperymentalny.
