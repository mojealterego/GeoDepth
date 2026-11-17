# OMEGA GeoDepth Ω v0.6.0 — implementacja i audyt

**Data:** 16 listopada 2026. **Wydanie:** samodzielny ZIP, bez zmian w repozytoriach.

## Co zostało zrealizowane

- Rzeczywiste, trenowalne nieliniowe pole deformacji pozycji Gaussów: znormalizowane funkcje radialne RBF, oddzielne przemieszczenia per-splat i per-knot, zakotwiczenie względem czasu referencyjnego i uczona prędkość bazowa.
- Gładka regularyzacja temporalna (energia amplitud i drugiej różnicy), loss RGB + depth + silhouette, clipped gradients, Adam, ograniczenia zasobów.
- Model starej reprezentacji liniowej 0.5 pozostaje kompatybilny; `upgrade_linear_gaussians` przekształca go bez utraty dotychczasowych parametrów.
- Nowy format `omega-gaussian-deformable-v2` w Safetensors, z manifestem SHA-256, deklaracją jednostek i bez `pickle`.
- Loader odrzuca `NaN`/`Inf`, nieznane tensory, niepoprawne znaczniki jednostek, symbolic links i nadmierne rozmiary.
- Oddzielny CLI: `python -m omega_geodepth.deformable_4dgs train|render`; wejście: manifest RGB-D/K/camera-to-world z poprzednich wydań.

## Metodologia pomiaru

- **Syntetyka**: jedna obserwowana kula gaussowska 24×24 px. Ścieżki analityczne, niezależnie określone: `x(t)=0.30 sin(πt)` i `x(t)=0.19t²-0.18t`. To test interpolacji ruchu, nie jakości ogólnej rekonstrukcji fotografii.
- Trening: 5 chwil (0, 0.5, 1, 1.5, 2); po 160 kroków Adam na model, ten sam seed i dane dla v0.5 i v0.6; uczone jedynie parametry ruchu (v0.6 dodatkowo deformacje).
- Chwile użyte do wyboru regularyzacji: (0.25, 0.75, 1.25, 1.75). Dodatkowe, niewykorzystane do doboru parametrów chwile audytowe: (0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875).
- Obie wersje: waga głębi 0.6, lr 0.055; v0.6 reg. deformacji 0.05. Nie mierzono czasu rzeczywistego działania poza tym przebiegiem ani metryk FID/LPIPS.
- RGB MAE mierzono wyłącznie na pikselach obserwowanych i ważnych; głębię Z tam, gdzie obserwacja i przewidywana alpha >0.01 występowały jednocześnie. Coverage raportowano osobno.
- Surowe metryki i konfiguracja: `examples/v06_deformable/benchmark.json`, demonstracja obrazu: `examples/v06_deformable/comparison.png`.

## Granice wykonanej pracy

- **Nie** pobrano oficjalnych wag DA3, ponieważ lokalny runtime nie rozwiązuje DNS Hugging Face, a publiczna ścieżka downloadu nie dostarcza plików binarnych do środowiska. Do wydania dołączono bezpieczny adapter DA3 i skrypt z weryfikacją oficjalnych SHA-256.
- **Nie** zweryfikowano real-world depth/pose/4DGS, urządzeń Android, GPU ani obrazów 400 MP.
- Uczenie dotyczy parametrów konkretnej sceny, a nie uniwersalnej sieci transferowalnej między scenami.
- Nie ma anisotropic Gaussian rasterizer, SH lighting, pełnego 4DGS deformable network, multi-object segmentation ani gwarantowanej rekonstrukcji części zasłoniętych.

## Komendy weryfikacyjne

```bash
OMP_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 bash scripts_verify.sh
OMP_NUM_THREADS=1 PYTHONPATH=. python benchmarks/benchmark_deformable_v06.py \
  --output examples/v06_deformable --steps 160 --depth-weight .6 --deform-reg .05
python -m pip wheel --no-deps --no-build-isolation . -w dist
```

Potwierdzenie kompletności archiwum i instalacji WHL należy do oddzielnej bramki wydaniowej.
