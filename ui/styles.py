"""Styles visuels partagés par l'interface Streamlit."""

import streamlit as st


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #0B1220; color: #F8FAFC; }
        [data-testid="stSidebar"] { background: #111C2E; }
        .hero { padding: 1.5rem 0 1rem; }
        .hero h1 { color: #F8FAFC; letter-spacing: .04em; }
        .hero p, .muted { color: #94A3B8; }
        .card {
            background: #111C2E; border: 1px solid #24344D;
            border-radius: 14px; padding: 1.1rem; min-height: 120px;
        }
        .card h3 { color: #22D3EE; margin-top: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
