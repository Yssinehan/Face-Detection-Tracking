# Rapport de performance

## Conditions

- Python 3.13.7, environnement `venv`.
- Modèle `blaze_face_full_range.tflite`.
- Vidéo `videos/demo.mp4`, 1920×1080, 214 frames, 23,98 FPS source.
- Mesures réalisées avec `time.perf_counter()`.
- Les valeurs sont des mesures locales et ne représentent pas une garantie
  temps réel sur un autre matériel.

## Résultats

| Mode | Frames | Temps | FPS |
|---|---:|---:|---:|
| Détection image (`images/test.jpg`) | 1 | 66,52 ms | N/A |
| Vidéo CLI sans tracking | 214 | 14,530 s | 14,73 |
| Vidéo CLI avec tracking | 214 | 16,250 s | 13,17 |
| Vidéo Streamlit UI avec tracking | 214 | 18,571 s | 11,52 |
| Webcam locale | 30 frames | 2,303 s | 13,03 |
| WebRTC navigateur | N/A | N/A | N/A |

## Profil par frame

Sur une boucle directe de 214 frames :

- détection sans tracking : moyenne 26,86 ms ;
- détection avec tracking : moyenne 33,34 ms ;
- mise à jour tracker : moyenne 0,494 ms ;
- annotation : environ 3,31 à 3,86 ms.

Le détecteur et l'encodage vidéo dominent le coût. Le tracker ne justifie pas
une réécriture ou une optimisation risquée.

L'échantillon webcam local a traité 30 frames, avec une détection sur les 30
frames et un ID confirmé `[1]`. Ce résultat mesure le pipeline local, pas le
transport WebRTC navigateur.

## Métriques absentes

Aucun jeu de vérité terrain annoté n'est fourni. Precision, recall, F1, mAP,
IDF1 et MOTA n'ont donc pas été calculés.
