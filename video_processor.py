"""Pipeline de détection indépendante image par image pour les vidéos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2

import config
from detector import FaceDetector
from tracker import FaceTracker
from visualization import draw_detections, draw_tracks


@dataclass(frozen=True)
class VideoProcessingStats:
    """Mesures produites par un traitement vidéo."""

    frames_processed: int
    duration_seconds: float
    processing_fps: float
    output_path: Path


def process_video(
    input_path: Path,
    output_path: Path,
    detector: FaceDetector,
    *,
    display: bool = False,
    overwrite: bool = False,
    track: bool = False,
) -> VideoProcessingStats:
    """Détecte et exporte une vidéo sans associer les frames entre elles."""
    _validate_video_path(input_path)
    if output_path.suffix.lower() not in config.SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"Extension de sortie vidéo non supportée : {output_path.suffix}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"La sortie existe déjà : {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Vidéo illisible ou impossible à ouvrir : {input_path}")
    writer: cv2.VideoWriter | None = None
    tracker = (
        FaceTracker(
            max_missing_frames=config.MAX_MISSING_FRAMES,
            min_track_hits=config.MIN_TRACK_HITS,
            max_centroid_distance=config.MAX_CENTROID_DISTANCE,
            min_iou=config.MIN_TRACK_IOU,
            association_threshold=config.TRACK_ASSOCIATION_THRESHOLD,
            alpha=config.TRACK_ALPHA,
            beta=config.TRACK_BETA,
            max_missing_seconds=config.MAX_MISSING_SECONDS,
            debug=config.TRACKING_DEBUG,
        )
        if track
        else None
    )
    frames_processed = 0
    started = perf_counter()
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = config.VIDEO_FPS_FALLBACK
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame is None or frame.ndim != 3 or frame.shape[0] <= 0:
                raise ValueError("Une frame vidéo décodée est invalide.")
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise OSError(f"Impossible d'initialiser la sortie : {output_path}")
            detections = detector.detect(frame)
            elapsed = perf_counter() - started
            throughput = (frames_processed + 1) / elapsed if elapsed > 0 else 0.0
            if tracker is None:
                annotated = draw_detections(
                    frame,
                    detections,
                    frame_number=frames_processed + 1,
                    processing_fps=throughput,
                )
            else:
                tracks = tracker.update(
                    detections,
                    frame.shape,
                    timestamp=(frames_processed + 1) / fps,
                )
                annotated = draw_tracks(
                    frame,
                    tracks,
                    detected_faces=len(detections),
                    frame_number=frames_processed + 1,
                    processing_fps=throughput,
                )
            writer.write(annotated)
            frames_processed += 1
            if display:
                cv2.imshow("Face detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    if frames_processed == 0:
        raise ValueError(f"La vidéo ne contient aucune frame lisible : {input_path}")
    duration = perf_counter() - started
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise OSError(f"La sortie vidéo n'a pas été générée : {output_path}")
    verification = cv2.VideoCapture(str(output_path))
    try:
        if not verification.isOpened() or not verification.read()[0]:
            raise OSError(f"La sortie vidéo est illisible : {output_path}")
    finally:
        verification.release()
    return VideoProcessingStats(
        frames_processed,
        duration,
        frames_processed / duration if duration > 0 else 0.0,
        output_path,
    )


def _validate_video_path(input_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {input_path}")
    if input_path.suffix.lower() not in config.SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"Extension vidéo non supportée : {input_path.suffix}")
