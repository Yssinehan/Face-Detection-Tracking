"""Page de traitement vidéo avec progression et export."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from uuid import uuid4

import cv2
import streamlit as st

import config
from detector import FaceDetector
from tracker import FaceTracker
from visualization import draw_detections, draw_tracks

from .components import download_file, page_header


def _save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in config.SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError("Format vidéo non supporté.")
    if not uploaded_file.getvalue():
        raise ValueError("Le fichier vidéo est vide.")
    temporary = NamedTemporaryFile(delete=False, suffix=suffix)
    temporary.write(uploaded_file.getvalue())
    temporary.close()
    return Path(temporary.name)


def _video_info(path: Path) -> tuple[cv2.VideoCapture, dict[str, float | int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("La vidéo ne peut pas être décodée par OpenCV.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return capture, {
        "fps": fps if fps > 0 else config.VIDEO_FPS_FALLBACK,
        "frames": total,
        "width": width,
        "height": height,
    }


def _process(
    source: Path,
    output: Path,
    confidence: float,
    tracking: bool,
    max_missing_seconds: float,
    min_track_hits: int,
    association_threshold: float,
    show_boxes: bool,
    show_ids: bool,
    show_confidence: bool,
    show_fps: bool,
    progress,
    status,
) -> dict[str, float | int]:
    capture, info = _video_info(source)
    fps = float(info["fps"])
    writer = None
    tracker = None
    if tracking:
        tracker = FaceTracker(
            max_missing_frames=config.MAX_MISSING_FRAMES,
            max_centroid_distance=config.MAX_CENTROID_DISTANCE,
            association_threshold=association_threshold,
            min_track_hits=min_track_hits,
            max_missing_seconds=max_missing_seconds,
        )
    frames = 0
    detections_total = 0
    max_faces = 0
    started = perf_counter()
    try:
        with FaceDetector(confidence_threshold=confidence) as detector:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if writer is None:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise OSError("Impossible d'initialiser la vidéo de sortie.")
                detections = detector.detect(frame)
                detections_total += len(detections)
                max_faces = max(max_faces, len(detections))
                if tracker is None:
                    annotated = draw_detections(
                        frame,
                        detections,
                        show_boxes=show_boxes,
                        show_confidence=show_confidence,
                        show_fps=show_fps,
                    )
                else:
                    tracks = tracker.update(
                        detections,
                        frame.shape,
                        timestamp=frames / fps,
                    )
                    annotated = draw_tracks(
                        frame,
                        tracks,
                        detected_faces=len(detections),
                        frame_number=frames + 1,
                        show_boxes=show_boxes,
                        show_ids=show_ids,
                        show_confidence=show_confidence,
                        show_fps=show_fps,
                    )
                writer.write(annotated)
                frames += 1
                if info["frames"]:
                    progress(min(frames / int(info["frames"]), 1.0))
                status(f"Frames : {frames} | FPS : {frames / max(perf_counter() - started, 1e-6):.1f}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    duration = perf_counter() - started
    if frames == 0 or not output.is_file() or output.stat().st_size == 0:
        raise OSError("Aucune sortie vidéo valide n'a été produite.")
    return {
        "frames": frames,
        "duration": duration,
        "fps": frames / duration if duration else 0.0,
        "faces": detections_total,
        "max_faces": max_faces,
    }


def render() -> None:
    page_header("Video Tracking", "Traitement séquentiel, progression et export.")
    uploaded = st.file_uploader("Importer une vidéo", type=["mp4", "avi", "mov"])
    tracking = st.checkbox("Enable Face Tracking", value=True)
    show_boxes = st.checkbox("Show Bounding Boxes", value=True)
    show_ids = st.checkbox("Show Tracking IDs", value=True)
    show_confidence = st.checkbox("Show Confidence Scores", value=True)
    show_fps = st.checkbox("Show FPS Overlay", value=True)
    confidence = st.slider(
        "Detection confidence",
        0.1,
        0.95,
        float(config.CONFIDENCE_THRESHOLD),
        0.05,
    )
    max_missing_seconds = st.number_input(
        "Maximum missing time (seconds)",
        min_value=0.1,
        max_value=30.0,
        value=float(config.MAX_MISSING_SECONDS),
        step=0.1,
    )
    min_track_hits = st.number_input(
        "Minimum track hits",
        min_value=1,
        max_value=20,
        value=int(config.MIN_TRACK_HITS),
        step=1,
    )
    association_threshold = st.number_input(
        "Association threshold",
        min_value=0.1,
        max_value=3.0,
        value=float(config.TRACK_ASSOCIATION_THRESHOLD),
        step=0.05,
    )
    if uploaded is None:
        return
    try:
        source = _save_upload(uploaded)
        capture, info = _video_info(source)
        capture.release()
    except (OSError, ValueError) as error:
        if "source" in locals():
            source.unlink(missing_ok=True)
        st.error(str(error))
        return
    st.write(
        f"{uploaded.name} · {info['width']}×{info['height']} · "
        f"{info['frames']} frames · {info['fps']:.2f} FPS · "
        f"{info['frames'] / info['fps']:.2f} s"
    )
    if not st.button("Start Video Processing", type="primary"):
        return
    output = config.OUTPUT_DIRECTORY / (
        f"tracked_{Path(uploaded.name).stem}_{uuid4().hex[:8]}.mp4"
    )
    progress = st.progress(0.0)
    status = st.empty()
    try:
        result = _process(
            source,
            output,
            confidence,
            tracking,
            max_missing_seconds,
            int(min_track_hits),
            association_threshold,
            show_boxes,
            show_ids,
            show_confidence,
            show_fps,
            progress.progress,
            status.write,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        st.error(f"Échec du traitement : {error}")
        return
    finally:
        source.unlink(missing_ok=True)
    st.success("Traitement terminé.")
    st.video(str(output))
    columns = st.columns(4)
    columns[0].metric("Frames", result["frames"])
    columns[1].metric("Processing FPS", f"{result['fps']:.2f}")
    columns[2].metric("Detections", result["faces"])
    columns[3].metric("Max faces", result["max_faces"])
    download_file("Download Processed Video", output, "video/mp4", "tracked_video.mp4")
