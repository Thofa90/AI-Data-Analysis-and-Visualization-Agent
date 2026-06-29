"""Consistent Streamlit rendering for assistant chat narrative text."""

from __future__ import annotations

import streamlit as st

from agent.schemas import ChatNarrativeResponse
from utils.chat_text import (
    escape_streamlit_math,
    render_chat_narrative_markdown,
    sanitize_markdown,
)


def render_chat_narrative(response: ChatNarrativeResponse | dict | None, fallback: str) -> None:
    """Render structured narrative when available, otherwise sanitized Markdown."""
    if response:
        narrative = (
            ChatNarrativeResponse.model_validate(response)
            if isinstance(response, dict)
            else response
        )
        st.markdown(escape_streamlit_math(render_chat_narrative_markdown(narrative)))
        return
    st.markdown(escape_streamlit_math(sanitize_markdown(fallback or "")))
