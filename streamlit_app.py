"""Point d'entrée de l'interface Streamlit Phase 4."""

import config
import streamlit as st

from ui import dashboard, image_page, video_page, webcam_page
from ui.styles import apply_styles


def main() -> None:
    st.set_page_config(
        page_title="AI Face Detection & Tracking",
        page_icon="👁",
        layout="wide",
    )
    apply_styles()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Section",
        ["Dashboard", "Image Detection", "Video Tracking", "Live Webcam"],
    )
    st.sidebar.caption(f"MediaPipe model: {config.MODEL_PATH.name}")
    if "stats" not in st.session_state:
        st.session_state.stats = {}
    if page == "Dashboard":
        dashboard.render()
    elif page == "Image Detection":
        image_page.render()
    elif page == "Video Tracking":
        video_page.render()
    else:
        webcam_page.render()


if __name__ == "__main__":
    main()
