# GeoDepth Ω v0.10.0 — CPU INT8 (eksperymentalny)

Nowy tryb `--profile int8 --allow-experimental` wykonuje dynamiczną kwantyzację INT8 warstw Linear backbone oryginalnego DA3 Small. Głowice kamery/głębi, konwolucje i pozostałe operacje nadal FP32. Tryb wymaga CPU i oficjalnego checkpointu FP32 z dokładnie przypiętym SHA-256. W rekonstrukcji 5D wymagany jest audyt reprojekcji. Wagi nie są zapisywane w formacie pickle. SHA-256 ładunku skwantyzowanych warstw obejmuje nazwy, rozmiary, parametry kwantyzacji, bajty INT8 i bias; nie jest to hash całej sieci.

```bash
python -m omega_geodepth.da3_real --model-dir models/da3 --source-dir vendor/da3_src --input examples/v08_real_da3/images/000.png examples/v08_real_da3/images/010.png --process-res 280 --profile int8 --allow-experimental --verify-reprojection --output /tmp/geodepth-int8
python -m omega_geodepth.reconstruction5d from-da3 --runtime native_network --checkpoint models/da3 --official-source vendor/da3_src --model DA3-SMALL --profile int8 --allow-experimental --verify-reprojection --manifest examples/v08_real_da3/manifest.json --process-res 280 --output /tmp/geodepth-int8-5d --voxel-size .08 --neighbor-radius .12 --mesh
```

Benchmark: `reports/v10_int8_geometry.json`. Dwie fotografie, jednakowe ustawienia, osobny proces dla FP32/INT8. To nie pomiar metryczny ani pełny benchmark mobilny. Pakowane wagi Linear: 84 934 656 → 21 233 664 B; nie obejmuje biasów, narzutu pakowania, innych warstw ani aktywacji. Peak RSS nie spada istotnie; konwersja wymaga początkowego załadowania FP32. Odchylenie geometrii wyklucza INT8 z domyślnej ścieżki. PyTorch 2.10.0 CPU: użyte API `torch.ao.quantization` jest oznaczone jako deprecated; migracja do torchao wymaga osobnej implementacji/testów. Nie twierdzimy, że AQ4/AQ6 mają natywny kernel ani że Android/GPU został zweryfikowany.
