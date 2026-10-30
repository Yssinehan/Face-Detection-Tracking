"""Page d'accueil et état de session."""

from pathlib import Path

import config
import streamlit as st

from .components import page_header


def render() -> None:
    page_header("AI FACE DETECTION & TRACKING", "Intelligent Computer Vision System")
    st.write(
        "Application locale de détection et de suivi géométrique de visages "
        "avec MediaPipe, OpenCV et NumPy."
    )
    cards = [
        ("IMAGE DETECTION", "Détecter des visages dans une image importée."),
        ("VIDEO TRACKING", "Analyser une vidéo et conserver des IDs temporaires."),
        ("LIVE WEBCAM", "Utiliser la webcam locale avec le pipeline existant."),
        ("RESULT EXPORT", "Télécharger les images et vidéos annotées."),
    ]
    columns = st.columns(4)
    for column, (title, text) in zip(columns, cards):
        with column:
            st.markdown(
                f'<div class="card"><h3>{title}</h3><p>{text}</p></div>',
                unsafe_allow_html=True,
            )
    st.subheader("Technologies")
    st.write("Python · OpenCV · MediaPipe Tasks · NumPy · Streamlit")
    st.subheader("État de la session")
    stats = st.session_state.get("stats", {})
    if not stats:
        st.info("Aucune donnée de traitement disponible pour le moment.")
    else:
        columns = st.columns(4)
        values = (
            ("Images traitées", stats.get("images", 0)),
            ("Vidéos traitées", stats.get("videos", 0)),
            ("Dernier FPS", f"{stats.get('fps', 0.0):.2f}"),
            ("Derniers visages", stats.get("faces", 0)),
        )
        for column, (label, value) in zip(columns, values):
            column.metric(label, value)
    st.caption(f"Modèle disponible : {'oui' if Path(config.MODEL_PATH).is_file() else 'non'}")
