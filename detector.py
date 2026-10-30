"""Détection de visages avec l'API MediaPipe Tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config


@dataclass(frozen=True)
class FaceDetection:
    """Résultat indépendant de l'API MediaPipe."""

    x: int
    y: int
    width: int
    height: int
    confidence: float


def _clip_box(
    origin_x: float,
    origin_y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """Convertit une boîte MediaPipe en coordonnées pixels valides."""
    left = max(0, min(image_width, int(origin_x)))
    top = max(0, min(image_height, int(origin_y)))
    right = max(0, min(image_width, int(origin_x + width)))
    bottom = max(0, min(image_height, int(origin_y + height)))
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top


class FaceDetector:
    """Détecteur réutilisable pour des images OpenCV au format BGR."""

    def __init__(
        self,
        model_path: Path = config.MODEL_PATH,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Le seuil de confiance doit être compris entre 0 et 1.")
        if not model_path.is_file():
            raise FileNotFoundError(f"Modèle MediaPipe introuvable : {model_path}")
        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=confidence_threshold,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        self.confidence_threshold = confidence_threshold

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """Détecte les visages d'une image BGR et renvoie des boîtes pixels."""
        self._validate_image(image)
        image_height, image_width = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self._detector.detect(mp_image)
        detections: list[FaceDetection] = []
        for detection in result.detections:
            if not detection.categories:
                continue
            confidence = float(detection.categories[0].score)
            if not 0.0 <= confidence <= 1.0 or confidence < self.confidence_threshold:
                continue
            box = detection.bounding_box
            clipped = _clip_box(
                box.origin_x,
                box.origin_y,
                box.width,
                box.height,
                image_width,
                image_height,
            )
            if clipped is None:
                continue
            x, y, width, height = clipped
            detections.append(FaceDetection(x, y, width, height, confidence))
        return detections

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("L'image doit être un numpy.ndarray.")
        if image.size == 0:
            raise ValueError("L'image ne peut pas être vide.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("L'image doit avoir la forme hauteur x largeur x 3.")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("Les dimensions de l'image doivent être positives.")

    def close(self) -> None:
        """Libère les ressources natives MediaPipe."""
        self._detector.close()

    def __enter__(self) -> FaceDetector:
        return self

    def __exit__(self, *args) -> None:
        del args
        self.close()


def load_image(path: Path) -> np.ndarray:
    """Charge une image supportée et signale clairement les erreurs."""
    if not path.is_file():
        raise FileNotFoundError(f"Image introuvable : {path}")
    if path.suffix.lower() not in config.SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Extension d'image non supportée : {path.suffix}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Image illisible ou corrompue : {path}")
    return image
