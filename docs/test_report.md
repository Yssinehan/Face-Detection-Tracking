# Rapport de tests

## Environnement

- Python 3.13.7
- OpenCV 5.0.0
- MediaPipe 1.2.0
- NumPy 2.5.4
- Streamlit 1.66.1
- streamlit-webrtc 0.79.0
- aiortc 1.15.0
- av 17.1.0

## Commande

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Résultat final : **46 tests exécutés, 46 réussis, 0 échec, 0 ignoré**.

La suite couvre la détection, les boîtes, le tracker, les pipelines vidéo,
les validations UI, les temporaires et le processeur WebRTC.

## Vérifications complémentaires

- `python -m pip check` : aucune dépendance cassée.
- `py_compile` sur les modules UI : réussi.
- Streamlit démarre et `/_stcore/health` répond `ok`.
- Le processeur WebRTC a été testé avec des frames `av.VideoFrame` simulées.
- La webcam locale a été ouverte avec succès en 640×480 ; 30/30 frames ont
  contenu une détection et l'ID confirmé observé était `1`.

## Limites de validation

Les tests automatisés ne remplacent pas une permission caméra réelle dans un
navigateur. La présence du composant WebRTC dans Streamlit a été vérifiée,
mais une session caméra complète avec visage réel n'a pas été confirmée dans
le navigateur automatisé disponible.
