# Dystrybucja wag modeli

Zobacz [manifest stanu repozytorium](../models/repository_inventory.json). Cztery oryginalne artefakty OMEGA v0.1/v0.2 są w pełni zapisane w 44 numerowanych częściach binarnych, łącznie 15 060 048 bajtów. Instrukcje w katalogach `.parts` opisują odtworzenie plików i kontrolę SHA-256. Modele scenowe 4D/5D i checkpoint Representations5D są odrębnymi plikami w repozytorium.

Pięć oficjalnych lub pochodnych checkpointów DA3 pozostaje otwartą bramką dystrybucji. Nie należy interpretować manifestu jako publikacji ich bajtów. DA3 FP32 Small i Base przekraczają limit typowego pojedynczego pliku GitHub; zalecane Git LFS lub Release assets z zachowaniem licencji i integralności. Dostępny kanał API nie posiada tych operacji. Testowany transfer małych blobów jest możliwy; duży transfer wymaga dodatkowego stabilnego kanału.

Nie uruchamiano nowej regresji kodu — w tej iteracji zmieniono wyłącznie dystrybucję artefaktów i dokumentację.

## Publikator checkpointów DA3 (kod w `main`)

Po uzyskaniu oryginalnych plików lokalnie, z układem katalogów zgodnym z `models/repository_inventory.json`:

```sh
omega-geodepth-model-release --root .                    # preflight; nie używa sieci
omega-geodepth-model-release --root . --publish          # wymaga GEODEPTH_GITHUB_TOKEN w środowisku
```

Publikator wymaga wszystkich pięciu plików (810442158 bajtów). Przed jakimkolwiek zapisem sprawdza rozmiary i sumy SHA-256. Tworzy roboczy GitHub Release (draft) na bieżącym SHA `main`, wysyła assety strumieniowo, a następnie porównuje zwrócony przez GitHub digest SHA-256. Dopiero po kompletnej weryfikacji pięciu assetów oraz niezmienionego HEAD publikuje Release. Uruchomiony ponownie rozpoznaje już zweryfikowane zasoby; istniejącego błędnego assetu nie nadpisuje.

**Zakres dowodów:** testy offline potwierdzają kontrakty walidacji i automat stanów w stubie API. Nie potwierdzają rzeczywistego uploadu; rzeczywiste wagi nadal nie są na GitHub. Nie umieszczać tokenu w poleceniach, logach, manifestach ani commitach.
