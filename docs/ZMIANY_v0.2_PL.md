# OMEGA GeoDepth Ω — zmiany 0.2.0

## Rzeczywiście zaimplementowane

- **Rozszerzony checkpoint**: sieć U-Net o szerokości 24, trenowana od początku na proceduralnych scenach raycast; porównanie z 0.1 na identycznym zewnętrznym (syntetycznym) podzbiorze.
- **Kafelkowa inferencja global-local**: globalna mapa w skali roboczej + nakładające się lokalne predykcje, usuwanie gauge w przestrzeni log Z, okno cosinusowe, niezerowe pokrycie brzegów i jawne oszacowanie rozbieżności. Algorytm autorski inspirowany strategią wieloskalową, nie jest implementacją PatchFusion. Nie gwarantuje poprawy dokładności.
- **Bramka kotwic**: przestrzenna dystrybucja pomiarów, odporna estymacja skali, odrzucenie dużych residuów, logarytmiczna niepewność skali. Nie weryfikuje fizycznego pochodzenia pomiaru.
- **Konformalna kalibracja scen**: osobny kwantyl poziomu sceny dla 90% pokrycia 90% pikseli, osobna syntetyczna próbka audytowa. Nie dotyczy fotografii realnych, wysokich rozdzielczości ani kotwic metrycznych bez osobnej walidacji.
- **Bezpieczeństwo dekodowania**: sprawdzanie liczby pikseli przed EXIF-transpose; limit 24MP źródła. Nie jest to dekoder 400MP out-of-core.
- **ComfyUI V1**: jeden działający węzeł IMAGE → IMAGE/MASK/STRING, testowany jako moduł Python, bez zewnętrznego ComfyUI; interfejs GUI samego ComfyUI nie był sprawdzany.
- **Integralność**: lokalne sumy SHA-256 i atomowy zapis wyników. Wciąż brak niezależnego cyfrowego podpisu.

## Użycie

```bash
python -m omega_geodepth infer --image fotografia.jpg --output wynik --model-dir models \
  --max-side 1536 --tiled --tile-size 256 --overlap 64 --global-side 256
```

`depth_preview.png` to obraz wizualizacji, nie pomiar; numeryczny `depth_geometry.npz` jest w jednostkach arbitralnych. `--max-side > 1024` wymaga `--tiled`, maksymalnie 4096 i 24MP źródła. Dla więcej niż 1,1 miliona punktów wynik PLY zostaje celowo pominięty i jest to raportowane.

Aby zainstalować węzeł ComfyUI: zainstaluj koło Python `omega-geodepth` w tym samym interpreterze, skopiuj folder `comfyui/custom_nodes/OCAF_GeoDepth_Omega` do `ComfyUI/custom_nodes/`, a w polu `model_directory` wskaż katalog absolutny z `models/manifest.json` i checkpointem. Węzeł wykonuje się na CPU i nie pobiera wag.

## Ograniczenia pozostałe

Brak: modelu językowego Abliterated, wytrenowanych wag DA3, pewnych odległości metrycznych bez sensorów, 4D, fuzji czujników, natywnej kompilacji Android/Termux i pomiarów na zdjęciach realnych. Wyniki treningu wyłącznie syntetyczne. Brak filtra NSFW oznacza brak klasyfikatora treści w silniku geometrycznym, nie gwarancję jakości predykcji na wszystkich treściach.

## Źródła koncepcji

- Specyfikacja użytkownika `SYS-AETHERIS-VISION-GEO-003`, strony 1–7, 46–50, 63–64.
- Oficjalny Depth Anything 3: https://github.com/ByteDance-Seed/Depth-Anything-3
- PatchFusion, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/papers/Li_PatchFusion_An_End-to-End_Tile-Based_Framework_for_High-Resolution_Monocular_Metric_Depth_CVPR_2024_paper.pdf
