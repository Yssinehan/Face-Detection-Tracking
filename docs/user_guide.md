# Guide utilisateur

## A. Installation

Dans PowerShell :

```powershell
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## B. Lancement

```powershell
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## C. Dashboard

Utilisez la barre latérale pour choisir Dashboard, Image Detection, Video
Tracking ou Live Webcam. Le dashboard affiche l'état du modèle et les
statistiques de la session courante.

## D. Image detection

1. Importer une image JPG, JPEG ou PNG.
2. Régler le seuil de confiance.
3. Cliquer sur `Detect Faces`.
4. Examiner les boîtes et scores.
5. Télécharger `detected_faces.png`.

## E. Video tracking

1. Importer une vidéo MP4, AVI ou MOV.
2. Activer ou désactiver le tracking.
3. Régler confiance, expiration, hits et association.
4. Choisir les overlays.
5. Cliquer sur `Start Video Processing`.
6. Suivre la progression.
7. Télécharger la vidéo annotée.

## F. Webcam WebRTC

1. Ouvrir `Live Webcam`.
2. Cliquer sur `Start` dans le composant vidéo.
3. Autoriser la caméra.
4. Observer boîtes, scores, IDs et FPS.
5. Cliquer sur `Stop`.

## G. Dépannage

- **Caméra refusée** : autoriser la caméra pour localhost ou utiliser HTTPS.
- **Aucun visage** : améliorer l'éclairage et vérifier la distance.
- **ID changé** : vérifier les occlusions et les longues disparitions.
- **FPS faible** : utiliser une résolution raisonnable ou traiter hors UI.
- **Vidéo refusée** : vérifier le codec supporté par OpenCV.
- **Modèle absent** : restaurer `models/blaze_face_full_range.tflite`.
- **CLI webcam** : `.\venv\Scripts\python.exe app.py --camera`.
