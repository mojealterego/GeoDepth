# Eksperymentalny eksport backbone DA3 (v0.17)

Opcjonalny kontrakt Python: `engine.inference(images, export_backbone=True)`.
Dotyczy wyłącznie oryginalnych profili FP32 Small i Base. Domyślny pipeline nie eksportuje cech.
Przetworzony obraz musi mieć boki podzielne przez 14 i maksymalnie 504 px.
Warstwa 11: Small 384 cechy, Base 768 cech na patch. Po dołączeniu RGB i promieni głowica przyjmuje odpowiednio 389 lub 773 kanały.

```python
from omega_geodepth.da3_features import prepare_backbone_gaussians
from omega_geodepth.gsdpt import GaussianHead

prediction = engine.inference(training_images, export_backbone=True)
sample = prepare_backbone_gaussians(prediction, 0)
head = GaussianHead(sample.features.shape[1])
gaussians = head(sample.features, sample.rays, sample.depth, sample.pose)
```

Nie należy podawać kamer walidacyjnych w `training_images`: backbone DA3 agreguje widoki.
Nie należy ładować starej pięciokanałowej głowicy do nowego kontraktu.
Powyższy kod inicjalizuje głowicę, nie zastępuje jej treningu. Głębia względna, nie metryczna.

Status: helper i most przetestowane na kontrolowanej sieci testowej, nie na rzeczywistych wagach.
Integracja runtime oraz zgodność z przypiętym archiwum źródłowym wymagają rzeczywistego E2E.
