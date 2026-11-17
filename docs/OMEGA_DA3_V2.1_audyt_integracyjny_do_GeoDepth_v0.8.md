# Audyt integracyjny: OMEGA DA3-SMALL v2.1 → OCAF GeoDepth Ω v0.8

**Data:** 2026-11-16  
**Podstawa:** archiwum użytkownika `OMEGA_DA3_V2_COMPLETE_2026-11-16.zip` oraz archiwum v0.8 naszego projektu.

## Rozstrzygnięcie

**Warto włączyć konwerter i profil FP16, a AQ6/AQ4 zatrzymać w strefie eksperymentalnej.** Nowa paczka nie dodaje nowej architektury DA3, treningu czy rekonstrukcji 5D; dodaje adaptacyjną kwantyzację, przechowywanie wag i narzędzia odtwarzania.

## Weryfikacja wejścia

- 33 wpisy ZIP, 32 pliki objęte manifestem SHA-256: **32/32 zgodne**.
- SHA-256 ZIP: `900ebdcbc6fbb1f7fc09afa5b863b42df6613c27a4b415b8cfdfc9af6a4c9532`
- Testy paczki: **69/69 PASS** (`pytest -q`, CPU, torch 2.10.0+cpu, Python bieżącego środowiska).
- `validate_adaptive` AQ6/AQ4: **PASS**; `restore_adaptive`: **437 tensorów na wariant**, 6 aliasów zachowanych.
- Oryginalny model kontrolny FP32 SHA-256: `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf`.
- Nowa paczka **nie zawiera oryginalnego FP32**; dołączone są FP16, AQ6 i AQ4.

## Profile wag

| Model | Wielkość zapisanych wag (MB dziesiętne) | Rola |
|---|---:|---|
| DA3 Small FP32 z projektu v0.8 | 137,25 | Podstawa kontrolna, nie zastępować |
| FP16 z nowej paczki | 68,65 | Polecany profil oszczędności dysku |
| AQ6 | 33,88 | Eksperymentalny format transportowy |
| AQ4 | 29,14 | Wyłącznie eksperymentalny |

AQ6/AQ4 po rekonstrukcji zajmują **72,38 MB** pliku wag FP16/FP32, a model wykonawczy może zajmować jeszcze więcej pamięci. Nie istnieje tu natywny kernel inferencji 4- lub 6-bitowej.

## Test E2E na oryginalnej sieci DA3 Small

Dwie fotografie `examples/v08_real_da3/images/000.png` i `010.png` (każda 1208×680), CPU. Baseline = wagi FP32, następnie w tej samej architekturze podmieniono tensory modelu na profile FP16 oraz odtworzone AQ6/AQ4. Porównano mapy głębi między wariantami **po dopasowaniu medianowej skali na każdej klatce**; to nie jest metryka dokładności fizycznej.

| process_res | Wariant | Średnie względne MAE głębi vs FP32 | Max zmiany K (px) | Reprojekcja |
|---:|---|---:|---:|---|
| 140 | FP16 | 0.036% | 0.015 | NIE TESTOWANO |
| 140 | AQ6 | 5.679% | 5.131 | NIE TESTOWANO |
| 140 | AQ4 | 7.073% | 11.965 | NIE TESTOWANO |
| 280 | FP16 | 0.030% | 0.022 | PASS |
| 280 | AQ6 | 3.491% | 4.907 | PASS |
| 280 | AQ4 | 5.505% | 33.281 | PASS |

Dla `process_res=280` wszystkie cztery warianty (FP32/FP16/AQ6/AQ4) spełniły standardową kontrolę reprojekcji v0.8. **Nie dowodzi to równoważności modeli** — predykcje głębi i pozy mogą pozostawać wewnętrznie spójne pomimo błędów względem fizycznej sceny.

## Przeszkody integracyjne

1. `DA3RealCPU` w v0.8 akceptuje wyłącznie oficjalny SHA-256 FP32; nie wolno podmieniać pochodnych wag i oznaczać ich jako identycznego checkpointu oficjalnego.
2. Wszystkie trzy `config.json` z nowej paczki mają identyczną semantykę jak oryginał, lecz **inną bajtową sumę SHA-256** (inny zapis JSON). Oryginalna bramka pinująca bajty je odrzuci, dopóki nie zostanie udokumentowany osobny profil.
3. `omega_adaptive.py` zawiera kodek kwantyzacji i odtwarzania; nowe `infer.py` importuje oficjalne API DA3 i może wymagać zależności opcjonalnych, których v0.8 unikał izolowanym backendem.
4. Manifest i SHA-256 wykrywają niezamierzoną zmianę, ale nie autentyczność autorstwa. Powiązać wariant z przypiętym hashem oryginalnego FP32 i podpisanym manifestem wydania.
5. Nie ma potwierdzonego przyspieszenia CPU/GPU, natywnych kernelów niskobitowych, benchmarków terenowych ani testu Android.

## Rekomendowana integracja v0.9

1. Zmiany wyłącznie addytywne: `tools/da3_quant/`, bez nadpisania `models/da3/model.safetensors`.
2. Zarejestrować niezależne identyfikatory wag `DA3-SMALL-FP16`, `DA3-SMALL-AQ6-EXPERIMENTAL`, `DA3-SMALL-AQ4-EXPERIMENTAL` wraz z `parent_sha256`, `artifact_sha256`, formatem, skalą jednostek i ograniczeniami.
3. Po `validate_adaptive` odtwarzać w odizolowanym katalogu tymczasowym, bez automatycznego pobierania i bez wykonywania kodu z checkpointu.
4. Przyjęcie do potoku 5D dopiero po porównaniu reprojekcji, pozy, K i głębi na wielowidokowych zbiorach z ground truth; domyślnie FP32/FP16.
5. Profil AQ6/AQ4 udostępniać wyłącznie za jawnym przełącznikiem `experimental`, bez „metrycznego” podpisywania głębi względnej.

## Status wykonania audytu

**Wykonane:** inspekcja kodu i metadanych, manifesty, testy 69/69, odtwarzanie AQ4/AQ6, rzeczywista inferencja porównawcza przy 140 i 280 px, reprojekcja 280 px.

**Nie wykonano:** modyfikacji repozytorium, treningu lub fine-tuningu, benchmarku na ground truth, testów GPU/Android, pełnego porównania rekonstrukcji TSDF/4DGS.

**Źródła liczb:** raporty `e2e_comparison_da3.json` i `e2e_comparison_da3_280_reprojection.json` z bieżącego audytu. Dane wag i opis kwantyzacji potwierdzone przez zawartość dostarczonego ZIP.
