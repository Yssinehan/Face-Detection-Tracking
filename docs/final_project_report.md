# Rapport final — AI Face Detection & Tracking

## Page de garde

- **Projet :** AI Face Detection & Tracking
- **Année académique :** 2025–2026
- **Spécialisation :** Artificial Intelligence & Big Data
- **Institution :** EMSI

## Résumé

Ce projet met en œuvre une détection de visages basée sur MediaPipe, un suivi
géométrique multi-visages et des interfaces CLI et Streamlit. Les résultats
sont visualisés et exportables. Les IDs sont temporaires et ne réalisent pas
d'identification biométrique.

## Abstract

This project implements MediaPipe face detection, geometric multi-face
tracking, command-line processing, and a Streamlit interface with image,
video, local webcam, and embedded WebRTC workflows. Results are visualized
and exportable. Tracking identifiers are temporary geometric labels and are
not biometric identities.

## Introduction générale

La vision par ordinateur permet d'extraire des informations visuelles à partir
de flux image. Le projet traite le besoin de localiser des visages et de
maintenir une association temporelle exploitable pour une démonstration
interactive.

## Chapitre 1 — Présentation

Le périmètre couvre images, vidéos, webcam locale et navigateur. Les
contraintes principales sont la latence, le nettoyage des ressources, la
lisibilité des exports et l'absence de reconnaissance faciale.

## Chapitre 2 — Analyse et conception

Le système sépare détecteur, tracker, visualisation, processeurs médias et UI.
Le flux principal est : capture, conversion, inférence, filtrage, association,
annotation et export.

## Chapitre 3 — Technologies

Python orchestre l'application, OpenCV lit et écrit les médias, MediaPipe
fournit le modèle BlazeFace full-range, NumPy représente les frames, Streamlit
fournit l'interface et `streamlit-webrtc` transporte le flux navigateur.

## Chapitre 4 — Implémentation

Les boîtes sont écrêtées aux dimensions de la frame. Le tracker applique NMS,
distance centroïde, IoU, cohérence de taille, confirmation et expiration
temporelle. WebRTC conserve un processeur par session et libère le détecteur
à l'arrêt.

## Chapitre 5 — Tests et résultats

La validation finale a exécuté 46 tests avec 46 réussites. Sur `demo.mp4`
(214 frames, 1920×1080), le pipeline CLI atteint 14,73 FPS sans tracking et
13,17 FPS avec tracking. Le pipeline Streamlit mesuré atteint 11,52 FPS.
Un échantillon de webcam locale de 30 frames a atteint 13,03 FPS, avec une
détection sur chaque frame et un ID confirmé.
L'image réelle contient une détection valide. Aucun dataset annoté n'étant
disponible, précision, rappel, F1, mAP, IDF1 et MOTA ne sont pas calculés.

## Chapitre 6 — Optimisations

Le profil montre que la détection domine le coût et que le tracker représente
environ 0,494 ms par frame. Aucune optimisation invasive n'a été appliquée :
le gain potentiel ne justifiait pas un risque sur la stabilité du tracking.
Le détecteur est déjà réutilisé par session.

## Conclusion générale

Le projet fournit une démonstration complète de détection et de suivi
géométrique, avec exports et interface Streamlit.

## Perspectives

Créer un dataset annoté, ajouter un benchmark reproductible, améliorer la
gestion de scènes occultées et intégrer WebRTC STUN/TURN pour un déploiement
distant constituent des évolutions possibles.

## Annexes

```text
app.py / streamlit_app.py
detector.py / tracker.py / visualization.py
video_processor.py / camera_processor.py
ui/
tests/
docs/
models/ images/ videos/ output/
```
