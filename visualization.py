"""Fonctions d'annotation OpenCV sans logique de détection ou de suivi."""

from __future__ import annotations

import cv2
import numpy as np

import config
from detector import FaceDetection
from tracker import Track


def draw_detections(
    frame: np.ndarray,
    detections: list[FaceDetection],
    frame_number: int | None = None,
    processing_fps: float | None = None,
    show_boxes: bool = True,
    show_confidence: bool = True,
    show_fps: bool = True,
) -> np.ndarray:
    """Dessine les boîtes, scores et statistiques sur une copie de l'image."""
    annotated = frame.copy()
    for detection in detections:
        if show_boxes:
            cv2.rectangle(
                annotated,
                (detection.x, detection.y),
                (detection.x + detection.width, detection.y + detection.height),
                config.BOUNDING_BOX_COLOR,
                config.ANNOTATION_THICKNESS,
            )
        if show_confidence:
            cv2.putText(
                annotated,
                f"Face: {detection.confidence:.0%}",
                (detection.x, max(20, detection.y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                config.TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
    lines = [f"Faces detected: {len(detections)}"]
    if frame_number is not None:
        lines.append(f"Frame: {frame_number}")
    if processing_fps is not None and show_fps:
        lines.append(f"Processing FPS: {processing_fps:.1f}")
    for index, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (10, 25 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            config.TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
    return annotated


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    detected_faces: int,
    frame_number: int | None = None,
    processing_fps: float | None = None,
    show_boxes: bool = True,
    show_ids: bool = True,
    show_confidence: bool = True,
    show_fps: bool = True,
) -> np.ndarray:
    """Dessine les pistes et distingue une piste momentanément absente."""
    annotated = frame.copy()
    visible_tracks = [
        track for track in tracks if track.detected and track.state == "confirmed"
    ]
    for track in visible_tracks:
        color = config.TRACK_COLORS[(track.track_id - 1) % len(config.TRACK_COLORS)]
        x, y, width, height = track.bbox
        if show_boxes:
            cv2.rectangle(
                annotated,
                (x, y),
                (x + width, y + height),
                color,
                config.ANNOTATION_THICKNESS,
            )
        labels = []
        if show_ids:
            labels.append(f"ID: {track.track_id}")
        if show_confidence:
            labels.append(f"{track.confidence:.0%}")
        if labels:
            cv2.putText(
                annotated,
                " | ".join(labels),
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                config.TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
    lines = [
        f"FPS: {processing_fps:.1f}"
        if processing_fps is not None and show_fps
        else "FPS: --",
        f"Detected Faces: {detected_faces}",
        f"Active Tracks: {len(visible_tracks)}",
    ]
    if frame_number is not None:
        lines.append(f"Frame: {frame_number}")
    for index, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (10, 25 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            config.TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
    return annotated
