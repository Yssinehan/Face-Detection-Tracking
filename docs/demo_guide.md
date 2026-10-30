# Guide de démonstration (5–7 minutes)

## Préparation

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Conserver `images/test.jpg` et `videos/demo.mp4` dans leurs dossiers.

## Séquence

1. **Introduction (30 s)** : expliquer détection et IDs temporaires.
2. **Dashboard (45 s)** : montrer le modèle disponible et les pages.
3. **Image (1 min)** : importer `images/test.jpg`, détecter, télécharger.
4. **Vidéo (2 min)** : importer `videos/demo.mp4`, activer tracking, montrer
   progression, IDs et export.
5. **Webcam (1–2 min)** : ouvrir Live Webcam, cliquer Start, autoriser la
   caméra, bouger la tête, cliquer Stop.
6. **Conclusion (30 s)** : rappeler les limites géométriques et la confidentialité.

## Résultats attendus

Les boîtes et scores doivent apparaître sur l'image. La vidéo doit produire un
fichier lisible et des IDs temporaires. WebRTC doit afficher la frame dans le
composant intégré lorsque la permission caméra est accordée.

## Plan de secours

Si la permission navigateur échoue, montrer le mode CLI :

```powershell
.\venv\Scripts\python.exe app.py --camera
```

Ou présenter les exports déjà vérifiés dans `output/`. Ne pas prétendre avoir
validé une session WebRTC si le navigateur n'a pas fourni de frames.
