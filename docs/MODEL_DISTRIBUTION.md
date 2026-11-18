# Dystrybucja wag modeli

Zobacz [manifest stanu repozytorium](../models/repository_inventory.json). Cztery oryginalne artefakty OMEGA v0.1/v0.2 są w pełni zapisane w 44 numerowanych częściach binarnych, łącznie 15 060 048 bajtów. Instrukcje w katalogach `.parts` opisują odtworzenie plików i kontrolę SHA-256. Modele scenowe 4D/5D i checkpoint Representations5D są odrębnymi plikami w repozytorium.

Pięć oficjalnych lub pochodnych checkpointów DA3 pozostaje otwartą bramką dystrybucji. Nie należy interpretować manifestu jako publikacji ich bajtów. DA3 FP32 Small i Base przekraczają limit typowego pojedynczego pliku GitHub; zalecane Git LFS lub Release assets z zachowaniem licencji i integralności. Dostępny kanał API nie posiada tych operacji. Testowany transfer małych blobów jest możliwy; duży transfer wymaga dodatkowego stabilnego kanału.

Nie uruchamiano nowej regresji kodu — w tej iteracji zmieniono wyłącznie dystrybucję artefaktów i dokumentację.
