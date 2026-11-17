# OMEGA GeoDepth Ω v0.7.0 — integracja źródeł DA3, walidacja wielowidokowa

## Stan implementacji

v0.7 zachowuje działanie v0.6, rzeczywiste, uprzednio trenowane wagi OMEGA-v0.2 oraz eksperymentalne modele scenowe Gaussów. **Nie trenuje ani nie pobiera nowych wag DA3**. Dołączono nienaruszone ZIP z kodem źródłowym `Depth-Anything-3-main.zip` dostarczone przez użytkownika. Istnienie kodu nie dowodzi obecności wag pretrained.

Nowości:

1. **Oficjalny kod DA3 jako offline bundle** — `vendor/Depth-Anything-3-main.zip`, 163 rekordy ZIP. Integralność przez pełny SHA-256 ZIP oraz SHA-256 rozpakowanych plików. Źródło jest dostarczone przez użytkownika; niezależne potwierdzenie identyczności z konkretnym commitem upstream **nie zostało wykonane**.
2. **Bezpieczny ekstraktor źródeł** — kontrola zip-slip, symlinków, powtórzonych nazw, rozmiarów, wymaganych plików i kontraktu AST API. Rozpakowanie do osobnego katalogu z atomowym rename; bez uruchamiania kodu podczas inspekcji/ekstrakcji.
3. **Jawna ścieżka uruchomienia oficjalnego kodu** — `--official-source` wskazuje na uprzednio zweryfikowany katalog źródeł. Moduł importujący sprawdza położenie `depth_anything_3.api`; odrzuca kolizję z inną wersją.
4. **Weryfikacja spójności wielowidokowej** — opcjonalna kontrola reprojekcji głębi osiowej `Z` przy użyciu wewnętrznych `K` (w tym skew) i pozycji kamer `camera-to-world`. Zwraca pokrycie, frakcję zgodności, medianę residuum, liczbę zasłonięć, a przy braku pokrycia abstencję. Sprawdza *spójność względną* — nie poprawność skali w metrach.
5. **Bramka geometryczna CLI** — `omega-geodepth-5d from-da3 ... --verify-reprojection`. W przypadku niespójności nie powstają pliki 5D z błędnym oznaczeniem „zweryfikowano”.

## Identyfikacja dostarczonego źródła

- SHA-256 archiwum: `6d2ba1d07a69f3d260394e49c7545aed8f3cc7df8633469ae79782ee34978d42`.
- Widoczny kontrakt `DepthAnything3.inference`: obrazy jako lista, wynik `depth`/`conf` `[N,H,W]`, `extrinsics` `[N,3,4]`, `intrinsics` `[N,3,3]`.
- Dostarczony pakiet zawiera plik licencji Apache 2.0 i nie zawiera `.safetensors`, `.ckpt`, `.pth`, `.pt`, `.onnx`, `.bin`.
- Zawiera moduły `da3_streaming`, ale funkcjonalność streamingu **nie została przetestowana na rzeczywistych wagach ani sekwencji nagranej kamerą**.

## Wykonanie offline ze źródeł (bez pobierania modeli)

```bash
python -m pip install --no-deps dist/omega_geodepth-0.7.0-py3-none-any.whl
omega-geodepth-da3-source inspect --archive vendor/Depth-Anything-3-main.zip
omega-geodepth-da3-source extract --archive vendor/Depth-Anything-3-main.zip --target /tmp/verified-da3
omega-geodepth-da3-source verify --target /tmp/verified-da3
```

Te polecenia **nie uruchamiają sieci DA3** i **nie ładują wag**, ale przygotowują wiarygodny względem otrzymanego ZIP katalog źródeł. Zależności z `vendor` nie są automatycznie instalowane. Oryginalny projekt DA3 wymaga dodatkowych pakietów (m.in. `addict`, `moviepy`, `evo`, zależnie od ścieżki importu); przed właściwą inferencją trzeba zainstalować zgodne wersje w odizolowanym środowisku.

## Oryginalne wagi DA3, gdy będą dostępne lokalnie

Do katalogu `/models/DA3-SMALL/` wprowadź **oryginalne** `model.safetensors` i `config.json` pobrane z `https://huggingface.co/depth-anything/DA3-SMALL` (repozytorium Apache-2.0). Następnie:

```bash
omega-geodepth-da3 --register-existing --model DA3-SMALL --destination /models/DA3-SMALL --revision <ZWERYFIKOWANY_COMMIT>
omega-geodepth-5d from-da3 \
  --manifest examples/sequence5d/manifest_from_da3.json \
  --checkpoint /models/DA3-SMALL --official-source /tmp/verified-da3 \
  --model DA3-SMALL --process-res 504 \
  --verify-reprojection --max-relative-error 0.15 \
  --output /tmp/geodepth7-scene5d --mesh --tsdf
```

`register-existing` wymaga poprawnego, znanego SHA-256 **oryginalnych** wag, a nie samodzielnie podanego skrótu. W przypadku braku wag system kończy błędem, nie tworzy fałszywego modelu. `--verify-reprojection` ma charakter pomocniczej kontroli spójności, może odrzucić poprawne ujęcia ze względu na brak wspólnego pola widzenia lub zasłonięcia.

## Bezpieczeństwo, licencje, ograniczenia

- Oryginalne źródła są w ZIP, bez ingerencji w ich kod. Obejmują licencję Apache 2.0; prawa do modelu i jego wag należy traktować oddzielnie.
- SHA-256 dostarczonego ZIP dowodzi jedynie integralności względem bieżącego wydania. **Nie jest podpisem cyfrowym ByteDance ani audytem upstream.**
- Kod źródłowy DA3 może wykonywać operacje dynamiczne (np. ładowanie konfiguracji OmegaConf). Faktyczne uruchamianie obcego kodu zaleca się przeprowadzić w izolowanym kontenerze bez sekretów, z ograniczonym egress.
- Nieliniowe gaussy v0.6 oraz 5D `XYZ+t+luma` pozostają eksperymentalne i nie są pełnym wytrenowanym 4DGS klasy produkcyjnej.
- Nie wykonano pomiarów na rzeczywistych nagraniach, benchmarków DTU/ScanNet++, testów GPU/Android ani inferencji na oryginalnych wagach DA3.
- Brak klasyfikatora NSFW w OMEGA nie jest gwarancją poprawnego działania dla dowolnych fotografii.

## Kryteria odbioru v0.7

Potwierdzaj po uruchomieniu `pytest` po czystym rozpakowaniu paczki: zgodność API, testy zamykania błędów, test sceny z syntetycznym wyjściem DA3, deterministyczny checksum ZIP i instalacja WHL. Nie opisuj testów z zamiennikiem DA3 jako testu oryginalnego checkpointu.
