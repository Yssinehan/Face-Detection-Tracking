# Confidentialité et limitations

## Nature du système

Le projet détecte des boîtes de visages et leur attribue des IDs temporaires
géométriques. Ces IDs ne sont pas des identités humaines et ne constituent
pas une reconnaissance faciale.

Le projet ne calcule pas d'embeddings, d'âge, de genre, d'émotion ou de base
biométrique.

## Stockage

- Les images et vidéos importées peuvent être écrites dans `output/` lorsque
  l'utilisateur demande un export.
- Les fichiers vidéo UI sont d'abord écrits dans un fichier temporaire puis
  supprimés après traitement.
- La webcam ne s'enregistre pas automatiquement.
- Le flux WebRTC n'ajoute pas de service cloud d'analyse.

Supprimer les artefacts exportés de `output/` lorsque leur conservation n'est
plus nécessaire.

## Limitations

- La qualité dépend de l'éclairage, de la taille et de l'orientation du visage.
- Les occlusions, croisements et longues disparitions peuvent changer un ID.
- Les codecs disponibles dépendent d'OpenCV et du navigateur.
- WebRTC nécessite localhost ou HTTPS et une permission caméra.
- Un serveur distant peut nécessiter STUN/TURN.
