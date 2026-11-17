# GeoDepth Ω 0.9.0 — profile DA3 i geometria 5D

Integracja addytywna z OMEGA DA3 v2.1 (archiwum użytkownika). Oficjalne wagi FP32 zachowano bez zmian. FP16 to 437 tensorów identycznych z `original.half()` i 6 aliasów. AQ6/AQ4 to stratne transportowe formaty SafeTensors; odtwarzane w prywatnym katalogu tymczasowym, usuwanym po załadowaniu do sieci. Wszystkie przebiegi używają obliczeń FP32; nie jest to natywne 4/6/16-bitowe wykonanie. Nie ma dowodu na zmniejszenie RAM ani przyspieszenie.

## Instalacja

Rozpakuj główny ZIP. Opcjonalny ZIP AQ rozpakuj do **tego samego katalogu nadrzędnego** (oba zawierają katalog `OMEGA_GeoDepth_Omega_v0_9`). Używaj zaufanego środowiska, bez innych kopii importowanych modułów DA3. Instalacja zależności zgodnie z `requirements.txt` i instrukcją v0.8; WHL nie obejmuje wag ani źródeł vendor.

```bash
python -m pip install -r requirements.txt
python -m pip install dist/omega_geodepth-0.9.0-py3-none-any.whl --no-deps
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m omega_geodepth.da3_real --model-dir models/da3_profiles/fp16 --profile fp16 \
 --source-dir vendor/da3_src --input examples/v08_real_da3/images/000.png examples/v08_real_da3/images/010.png \
 --process-res 280 --verify-reprojection --output /tmp/geodepth_fp16
python -m omega_geodepth.reconstruction5d from-da3 --runtime native_network \
 --model DA3-SMALL --profile fp16 --checkpoint models/da3_profiles/fp16 \
 --official-source vendor/da3_src --manifest examples/v08_real_da3/manifest.json \
 --process-res 280 --verify-reprojection --mesh --output /tmp/geodepth_5d_fp16
```

FP32: `--profile fp32 --checkpoint models/da3` (w inferencji `--model-dir`). AQ: po rozpakowaniu dodatku `--profile aq6 --checkpoint models/da3_profiles/aq6 --allow-experimental --verify-reprojection`; analogicznie aq4. Odtwarzanie odbywa się tylko w native_network, nie w upstream_api. W przypadku braku AQ testy odtwarzania są jawnie pomijane; instalacja dodatku umożliwia pełny zestaw.

## Bezpieczeństwo i pochodzenie

Przypięte w kodzie SHA-256 każdego checkpointu, konfiguracji i deskryptora AQ; kontrola rozmiaru, aliasów, licencji źródłowej i skrótu rodzica. Profil ma osobny identyfikator, skrót artefaktu transportowego oraz skrót odtworzonego runtime. Lokalny manifest nie zastępuje pinów. Chroniony shim `addict.py` ma osobny pin. Manifesty integralności nie są podpisane kluczem zewnętrznym. Nie gwarantują ochrony przed napastnikiem kontrolującym proces, system plików równolegle lub kod w zaufanym środowisku.

## Dowody

`reports/v09_profile_geometry.json`: dwie fotografie DA3, 280 px, ten sam potok 5D; wspólny współczynnik skali dla obu widoków, układ pierwszej kamery. Błąd = średnia bezwzględna względnych różnic głębokości względem FP32. Symetryczna odległość Chamfera (średnia odległość do najbliższych sąsiadów w obu kierunkach) podzielona przez medianę głębi FP32. To audyt regresji, nie metryczny benchmark. FP16 spełnia progi regresji 0,5% głębi, 0,5 px K, 0,02 znormalizowanej odległości Chamfera. Nie dokonano ponownego treningu modelu i nie testowano DA3 Base, Androida, CUDA, metrycznego ground truth ani pełnej deformacji produkcyjnej.

`examples/v09_profiles/` zawiera cztery rekonstrukcje i raporty. Czas klatek jest umowny, nie jest pomiarem czasu zdjęć. AQ4 i AQ6 nie są profilami produkcyjnymi.
