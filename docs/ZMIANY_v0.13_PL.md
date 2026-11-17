# GeoDepth Ω 0.13.0 — autentyczny DA3 Base i konserwatywny fallback

## Aktualizacja nakładkowa
Najpierw rozpakuj główne wydanie 0.12 oraz opcjonalny dodatek AQ. Rozpakuj ZIP aktualizacji 0.13 do tego samego katalogu nadrzędnego, nadpisując wymienione pliki w `OMEGA_GeoDepth_Omega_v0_12`. Wersja kodu po nakładce to 0.13.0, historyczna nazwa katalogu pozostaje dla zgodności. Sprawdź `UPDATE_SHA256.json`. Stare manifesty pełnej dystrybucji 0.12 opisują oryginalne wydanie, nie pliki zastąpione nakładką. Instalator WHL wymaga osobnych zasobów vendor, modeli i zależności z głównego wydania. Aktualizacja nie powiela dużych checkpointów.

## Base
Dostarczone części `model (2).z01`–`.z05` i `model (2).zip` zawierają jeden checkpoint. Rozmiar 541518028 B, SHA-256 e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5, zgodny z publicznym pinem Hugging Face (https://huggingface.co/depth-anything/DA3-BASE/blob/main/model.safetensors). 437 tensorów, 135366599 przechowywanych wartości FP32. Konfiguracja vitb identyczna z dostarczoną. Model nie był ponownie trenowany ani kwantyzowany. Licencja Apache 2.0. Przywracanie wymaga Info-ZIP (`zip`, w Termux `pkg install zip`) i około 1.1 GB dodatkowego wolnego miejsca w prywatnym katalogu lokalnym obsługującym hardlinki. Katalog modeli musi zawierać dostarczoną przypiętą konfigurację. Zapis no-replace; nie używaj katalogu współdzielonego Android bez obsługi hardlinków.

```bash
python -m omega_geodepth.base_restore --split-zip '/path/model (2).zip' --model-dir models/da3_base
python -m omega_geodepth.da3_real --model-id DA3-BASE --model-dir models/da3_base \
  --source-dir vendor/da3_src --input examples/v08_real_da3/images/000.png examples/v08_real_da3/images/010.png \
  --process-res 280 --verify-all-pairs --fallback-resolutions 392 504 --output output/base
```

Zachowano istniejący pin wag, konfiguracji, źródeł i strict state_dict. `--verify-all-pairs` kontroluje wszystkie uporządkowane pary (maks. 8 widoków), statyczna scena lub zsynchronizowane kamery. `--fallback-resolutions` przyjmuje maks. trzy dodatkowe, unikalne rozdzielczości, bez obniżania progów. Pierwszy zaakceptowany wariant wybrany w kolejności żądania. Wszystkie podjęte próby w raporcie. Przy braku akceptacji brak nowego eksportu. Nie używać jako walidacji ruchu; kontrola zgodności geometrii nie mierzy prawdziwego błędu metrycznego. `--verify-reprojection` nadal obsługuje stary kontrakt.

## Weryfikacja i benchmark
`reports/v13/{base,small,fallback}.json`. Wagi Base uruchomione na CPU FP32, ten sam zestaw dwóch fotografii z DA3, rozdzielczości 140/280/392/504. Base: 83.56/73.76/81.28/82.58% najmniejszej zgodności obustronnej. Small: 90.98/80.28/85.83/83.60%. Próg 80%, pokrycie min. 10%. Base 280 odrzucony. CLI fallback 280→392 zaakceptowany (2x224x392). Przy 504: 18815 punktów po fuzji wokselowej, 10999 kwalifikowanych próbek pola 5D. Wszystkie pozostałe rozdzielczości Base i Small przeszły pipeline jakości. Punktów z różnych gauge nie należy interpretować jako porównania fizycznej gęstości. Brak wzorca metrycznego; nie ma podstaw do ogólnego rankingu dokładności. Szczytowe RSS raportu procesu zależy od środowiska, jest kumulatywne, nie stanowi pomiaru telefonu.

## Otwarte bramki
Giant, natywne AQ4/AQ6, walidacja metryczna wieloscenowa, produkcyjna rekonstrukcja dynamiczna 5D, GPU/Vulkan/Android/400 MP. Brak zdalnego CI oraz niezależnej certyfikacji. Base nie jest domyślnym zamiennikiem Small. Nie zmieniono parametrów sieci ani wag. Skrypt pakowania pełnego wydania pomija duży plik Base; przy przekroczeniu limitu 250 MB konieczny podział dystrybucji. Nie przebudowano całej dystrybucji, wydanie 0.13 jest nakładką.
