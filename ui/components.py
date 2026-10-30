"""Composants Streamlit réutilisables."""

from pathlib import Path

import streamlit as st


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str) -> None:
    st.metric(label, value)


def download_file(label: str, path: Path, mime: str, file_name: str) -> None:
    if path.is_file() and path.stat().st_size > 0:
        st.download_button(
            label,
            data=path.read_bytes(),
            file_name=file_name,
            mime=mime,
        )
    else:
        st.warning("Le fichier de sortie n'est pas disponible.")
