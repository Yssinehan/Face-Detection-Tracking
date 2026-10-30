"""Page de détection sur image."""

from io import BytesIO
from time import perf_counter

import cv2
import numpy as np
import streamlit as st

import config
from detector import FaceDetector
from visualization import draw_detections

from .components import page_header


def _decode_upload(uploaded_file) -> np.ndarray:
    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("Le fichier image est vide.")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("L'image est illisible ou corrompue.")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("Les dimensions de l'image sont invalides.")
    return image


def render() -> None:
    page_header("Image Detection", "Détection MediaPipe et export de l'annotation.")
    uploaded = st.file_uploader("Importer une image", type=["jpg", "jpeg", "png"])
    threshold = st.slider(
        "Seuil de confiance",
        min_value=0.1,
        max_value=0.95,
        value=float(config.CONFIDENCE_THRESHOLD),
        step=0.05,
    )
    if uploaded is None:
        return
    try:
        image = _decode_upload(uploaded)
    except (TypeError, ValueError) as error:
        st.error(str(error))
        return
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Image originale")
    if not st.button("Detect Faces", type="primary"):
        return
    started = perf_counter()
    try:
        with FaceDetector(confidence_threshold=threshold) as detector:
            detections = detector.detect(image)
        annotated = draw_detections(image, detections)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        st.error(f"Échec de la détection : {error}")
        return
    elapsed = perf_counter() - started
    st.session_state.stats = {
        **st.session_state.get("stats", {}),
        "images": st.session_state.get("stats", {}).get("images", 0) + 1,
        "faces": len(detections),
    }
    left, right = st.columns(2)
    with left:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Résultat")
    with right:
        st.metric("Faces Detected", len(detections))
        st.metric("Processing Time", f"{elapsed * 1000:.1f} ms")
        for index, detection in enumerate(detections, start=1):
            st.write(f"Face {index}: {detection.confidence:.1%}")
    success, encoded = cv2.imencode(".png", annotated)
    if success:
        st.download_button(
            "Download Annotated Image",
            data=BytesIO(encoded.tobytes()),
            file_name="detected_faces.png",
            mime="image/png",
        )
