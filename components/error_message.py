"""Consistent user-facing error rendering."""

from __future__ import annotations

import streamlit as st


def render_error(message: str) -> None:
    """Render a safe error without exposing internals."""
    st.error(message, icon=":material/error:")
