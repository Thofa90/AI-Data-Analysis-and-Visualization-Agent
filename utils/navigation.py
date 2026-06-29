"""Single-source Streamlit navigation state."""

from __future__ import annotations

import streamlit as st


def navigate_to(page_name: str) -> None:
    """Update the radio-backed page before the next script render."""
    st.session_state.current_page = page_name
