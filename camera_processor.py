"""Traitement webcam avec détection et suivi géométrique local."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2

import config
from detector import FaceDetector
from tracker import FaceTracker
from visualization import draw_tracks


@dataclass(frozen=True)
class CameraProcessingStats:
    """Mesures de la session webcam."""

    frames_processed: int
    duration_seconds: float
    processing_fps: float
    output_path: Path | None


def process_camera(
    detector: FaceDetector,
    *,
    camera_index: int = 0,
    record: bool = False,
    output_path: Path | None = None,
    camera: cv2.VideoCapture | None = None,
) -> CameraProcessingStats:
    """Lance une session webcam jusqu'à la touche Q."""
    capture = camera or cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Impossible d'ouvrir la caméra {camera_index}.")

    writer: cv2.VideoWriter | None = None
    tracker = FaceTracker(
        max_missing_frames=config.CAMERA_MAX_MISSING_FRAMES,
        min_track_hits=config.MIN_TRACK_HITS,
        max_centroid_distance=config.CAMERA_MAX_CENTROID_DISTANCE,
        min_iou=config.MIN_TRACK_IOU,
        association_threshold=config.CAMERA_TRACK_ASSOCIATION_THRESHOLD,
        alpha=config.TRACK_ALPHA,
        beta=config.TRACK_BETA,
        max_missing_seconds=config.CAMERA_MAX_MISSING_SECONDS,
        debug=config.TRACKING_DEBUG,
    )
    frames_processed = 0
    started = perf_counter()
    actual_output = output_path
    try:
        if record and actual_output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            actual_output = config.OUTPUT_DIRECTORY / f"webcam_tracking_{timestamp}.mp4"
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                raise OSError("La caméra n'a pas fourni de frame valide.")
            if writer is None and record:
                height, width = frame.shape[:2]
                actual_output.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(actual_output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    config.VIDEO_FPS_FALLBACK,
                    (width, height),
                )
                if not writer.isOpened():
                    raise OSError(f"Impossible d'initialiser l'enregistrement : {actual_output}")
            detections = detector.detect(frame)
            elapsed = perf_counter() - started
            processing_fps = (frames_processed + 1) / elapsed if elapsed > 0 else 0.0
            tracks = tracker.update(
                detections,
                frame.shape,
                timestamp=elapsed,
            )
            annotated = draw_tracks(
                frame,
                tracks,
                detected_faces=len(detections),
                processing_fps=processing_fps,
            )
            cv2.putText(
                annotated,
                "Q: quit | IDs temporaires a la session",
                (10, max(25, annotated.shape[0] - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                config.TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
            if writer is not None:
                writer.write(annotated)
            cv2.imshow("AI FACE DETECTION & TRACKING", annotated)
            frames_processed += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    duration = perf_counter() - started
    if writer is not None and (
        actual_output is None
        or not actual_output.is_file()
        or actual_output.stat().st_size == 0
    ):
        raise OSError("L'enregistrement webcam n'a pas été généré.")
    return CameraProcessingStats(
        frames_processed,
        duration,
        frames_processed / duration if duration > 0 else 0.0,
        actual_output,
    )
