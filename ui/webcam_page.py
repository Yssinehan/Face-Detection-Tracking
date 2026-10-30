"""Webcam navigateur intégrée avec streamlit-webrtc."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Callable

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer

import config
from detector import FaceDetector
from tracker import FaceTracker
from visualization import draw_tracks

from .components import page_header


@dataclass(frozen=True)
class WebcamStats:
    """Snapshot thread-safe des métriques du flux courant."""

    frames: int = 0
    detected_faces: int = 0
    confirmed_tracks: int = 0
    processing_fps: float = 0.0
    error: str | None = None


class WebcamVideoProcessor(VideoProcessorBase):
    """Traite les frames d'une seule session navigateur WebRTC."""

    def __init__(
        self,
        detector_factory: Callable[[], FaceDetector] = FaceDetector,
        tracker_factory: Callable[[], FaceTracker] | None = None,
    ) -> None:
        self._detector = detector_factory()
        self._tracker = (
            tracker_factory()
            if tracker_factory is not None
            else FaceTracker(
                max_missing_frames=config.CAMERA_MAX_MISSING_FRAMES,
                min_track_hits=config.MIN_TRACK_HITS,
                max_centroid_distance=config.CAMERA_MAX_CENTROID_DISTANCE,
                association_threshold=config.CAMERA_TRACK_ASSOCIATION_THRESHOLD,
                max_missing_seconds=config.CAMERA_MAX_MISSING_SECONDS,
                alpha=config.TRACK_ALPHA,
                beta=config.TRACK_BETA,
            )
        )
        self._lock = Lock()
        self._stats = WebcamStats()
        self._started_at = monotonic()
        self._frame_number = 0
        self._closed = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Détecte, suit et renvoie une frame BGR annotée."""
        with self._lock:
            if self._closed:
                return frame
            try:
                image = frame.to_ndarray(format="bgr24")
                if image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError("La frame WebRTC doit être une image BGR 3 canaux.")
                self._frame_number += 1
                detections = self._detector.detect(image)
                tracks = self._tracker.update(
                    detections,
                    image.shape,
                    timestamp=monotonic() - self._started_at,
                )
                visible_tracks = [
                    track
                    for track in tracks
                    if track.detected and track.state == "confirmed"
                ]
                elapsed = max(monotonic() - self._started_at, 1e-6)
                self._stats = WebcamStats(
                    frames=self._frame_number,
                    detected_faces=len(detections),
                    confirmed_tracks=len(visible_tracks),
                    processing_fps=self._frame_number / elapsed,
                )
                annotated = draw_tracks(
                    image,
                    tracks,
                    detected_faces=len(detections),
                    processing_fps=self._stats.processing_fps,
                )
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")
            except (TypeError, ValueError, RuntimeError) as error:
                self._stats = WebcamStats(
                    frames=self._frame_number,
                    error=str(error),
                )
                return frame

    def get_stats(self) -> WebcamStats:
        """Retourne une copie cohérente des métriques."""
        with self._lock:
            return self._stats

    def on_ended(self) -> None:
        """Libère le modèle lorsque le navigateur arrête ou perd le flux."""
        with self._lock:
            if not self._closed:
                self._detector.close()
                self._closed = True


def render() -> None:
    page_header("Live Webcam", "Flux temps réel intégré dans Streamlit avec WebRTC.")
    st.info(
        "Autorisez l'accès caméra dans votre navigateur. Sur un déploiement distant, "
        "utilisez HTTPS ou localhost ; les frames sont traitées localement par cette session."
    )
    st.caption(
        "Les boutons Start/Stop du composant contrôlent le flux. Aucun enregistrement "
        "n'est automatique et les IDs restent temporaires à la session."
    )
    context = webrtc_streamer(
        key="embedded-face-webcam",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=WebcamVideoProcessor,
        async_processing=True,
        media_toggle_controls=True,
    )
    processor = context.video_processor
    if processor is not None:
        stats = processor.get_stats()
        columns = st.columns(4)
        columns[0].metric("Streaming", "Active" if context.state.playing else "Stopped")
        columns[1].metric("Processing FPS", f"{stats.processing_fps:.1f}")
        columns[2].metric("Detected Faces", stats.detected_faces)
        columns[3].metric("Visible Tracks", stats.confirmed_tracks)
        if stats.error:
            st.error(f"Erreur de traitement webcam : {stats.error}")
    elif context.state.playing:
        st.warning("Initialisation du processeur webcam…")

