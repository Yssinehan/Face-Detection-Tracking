# Documentation technique

## Objectif

Détecter des visages dans des images, vidéos et flux webcam, puis associer les
boîtes consécutives par un tracker géométrique.

## Modules

- `detector.py` : MediaPipe Tasks, validation et boîtes pixel.
- `tracker.py` : NMS, coût d'association et cycle de vie.
- `visualization.py` : annotations OpenCV.
- `video_processor.py` : vidéo CLI.
- `camera_processor.py` : webcam CLI.
- `ui/` : pages Streamlit et processeur WebRTC.

## Pipeline

OpenCV lit une frame BGR. Le détecteur la convertit en RGB, appelle MediaPipe
et convertit les coordonnées normalisées en boîtes pixels écrêtées. Le tracker
dédoublonne puis associe les détections aux pistes. La visualisation dessine
uniquement les pistes confirmées et visibles.

## Association

Pour deux boîtes A et B :

`IoU(A,B) = aire(A ∩ B) / aire(A ∪ B)`

La distance centroïde est :

`d = sqrt((x2-x1)^2 + (y2-y1)^2)`

Le coût implémenté combine distance normalisée, `1 - IoU` et une pénalité de
variation de taille. Les poids principaux sont `TRACK_ALPHA = 0.6` et
`TRACK_BETA = 0.4`; la pénalité de taille vaut `0.15`. Les gates dépendent de
la distance maximale, des frames manquantes et du ratio de surface.

## Cycle de vie

Une piste est `tentative` puis `confirmed` après `MIN_TRACK_HITS`. Lorsqu'une
détection manque, elle devient `missing` et reste interne pour réassociation.
Elle est supprimée après la limite de frames ou `MAX_MISSING_SECONDS`.

## WebRTC

`WebcamVideoProcessor` est créé par session. `recv()` convertit `av.VideoFrame`
en BGR, réutilise un détecteur et un tracker, puis renvoie une frame annotée.
Un `Lock` protège l'état et `on_ended()` ferme le détecteur.

## Tests et performance

La suite comporte 46 tests. Les résultats mesurés sont détaillés dans
`docs/performance_report.md`. Aucun score de précision n'est fourni sans
annotations de vérité terrain.
