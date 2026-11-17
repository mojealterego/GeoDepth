# Licensing and provenance of OMEGA GeoDepth Ω v0.4.0

This archive contains OMEGA-authored experimental reconstruction code and the previous independently trained OMEGA synthetic-only checkpoint. The copyrighted Depth Anything 3 architecture and original weights are NOT redistributed here: DA3-SMALL/DA3-BASE are provided exclusively as optional external checkpoint integrations. Fetch them from the original depth-anything model repository and respect their Apache-2.0 license, NOTICE and attribution requirements.

The previous OMEGA synthetic model was trained only on synthetic procedural raycast scenes. It is not the official Depth Anything 3 and is not validated for physical measurements. The project is not an uncensored or abliterated LLM; it is a content-agnostic geometric estimator without an explicit NSFW filtering stage.

Dependencies: PyTorch, NumPy, Pillow, SciPy, Safetensors, optional scikit-image, and separately the official DA3 source and Hugging Face Hub. Verify each upstream license and its supply chain before reuse. No new license for OMEGA-authored files is asserted by this NOTICE.

### v0.5

Oficjalne sumy SHA-256 plików `model.safetensors` DA3 Small/Base pochodzą z publicznych wskaźników Hugging Face, nie z lokalnie dostarczonych wag. Projekt nie zawiera oryginalnych checkpointów DA3 i nie przypisuje sobie ich autorstwa. Nowy `gaussians.safetensors` to autorski checkpoint nauczony na syntetycznych scenach.


## Dodatek v0.7.0 — DA3 user-supplied source ZIP
Dostarczone oryginalnie zapakowane źródła DA3 objęte deklarowaną licencją Apache-2.0 (szczegóły `vendor/Depth-Anything-3-main.zip`/LICENSE). SHA-256 paczki użytkownika nie jest poświadczeniem upstream. Kod zachowany bez modyfikacji. Oryginalnych wag DA3 nadal brak. Nowy audyt reprojekcji kontroluje wyłącznie względną spójność wielowidokową, nie dokładność metryczną. Patrz `docs/ZMIANY_v0.7_PL.md`.

## Aktualizacja v0.8 — oficjalny checkpoint DA3 Small
Zawartość `models/da3/model.safetensors` jest zgodna z publicznym SHA-256 modelu
`depth-anything/DA3-SMALL` (Apache-2.0), licencja i autorstwo modelu: ByteDance Seed Team.
Model i jego źródła stanowią niezależne komponenty licencjonowane na Apache-2.0.
Niezmienione źródła: `vendor/Depth-Anything-3-main.zip`; skopiowany kod wykonawczy
`vendor/da3_src/src` jest weryfikowany względem archiwum. Własny mały shim
`addict.Dict` nie jest częścią oryginalnej biblioteki `addict`.

## v0.9 — pochodne DA3
Kod kodeka omega_da3 i omega_adaptive skopiowano z dostarczonego archiwum OMEGA DA3 v2.1; zmieniono import na względny. Wagi pochodne FP16/AQ6/AQ4 nie są oryginalnymi checkpointami ByteDance. Licencja upstream Apache-2.0 zachowana w models/da3_profiles i vendor. Autorstwo kodu OMEGA pozostaje przy dostawcy; nie nadano nowej licencji.


## v0.10

Dynamiczna kwantyzacja INT8 jest wykonywana lokalnie na przypiętych oryginalnych wagach FP32 DA3 Small. Nie zmienia licencji upstream.

## v0.11 — Representations5D
Zachowano przekazane przez użytkownika oryginalne archiwum Representations5D, instalator, checkpoint i licencję w vendor. KPlanes5D jest osobnym komponentem, a nowe wagi `examples/v11_scene_field/*/scene.safetensors` są modelami scenowymi wytrenowanymi w GeoDepth. Nie są oryginalnym checkpointem DA3 ani checkpointem autorów Representations4D.

### v0.12 — ENGINE v1
Zaadaptowano kod `OMEGA_GeoDepth_ENGINE_v1.zip` dostarczony przez użytkownika, w odrębnym module `omega_geodepth/quality_v1`, z modyfikacjami opisanymi w `docs/ZMIANY_v0.12_PL.md`. Licencja tego modułu: Apache-2.0, oryginał w `docs/engine_v1/LICENSE` i w pakiecie Python. Nie nadano nowej licencji pozostałym kodom OMEGA. Konfiguracja i manifest DA3 Base nie zawierają wag; nie deklarujemy uruchomienia Base.
