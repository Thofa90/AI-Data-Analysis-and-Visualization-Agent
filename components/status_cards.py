"""Status and feature card components."""

from __future__ import annotations

import streamlit as st


def render_foundation_status() -> None:
    """Summarize currently available capabilities."""
    st.markdown("### Analysis Readiness")
    columns = st.columns(3)
    items = [
        ("Structured profiling", "Schema, missingness, uniqueness, ranges, and outliers are computed locally."),
        ("Explainable quality", "Every quality deduction is deterministic and includes a recommended action."),
        ("Safe metric detection", "Likely identifiers are excluded before meaningful measures are selected."),
    ]
    for column, (title, body) in zip(columns, items, strict=True):
        with column:
            st.markdown(
                f'<div class="feature-card"><h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True,
            )
