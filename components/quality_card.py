"""Data quality score UI."""

from __future__ import annotations

import streamlit as st

from services.profile_models import DataQualityScore


def render_quality_card(quality: DataQualityScore) -> None:
    """Render an explainable score and its deductions."""
    score_class = quality.rating.lower()
    st.markdown(
        f"""
        <div class="quality-card {score_class}">
            <div>
                <div class="metric-label">Data Quality Score</div>
                <div class="quality-score">{quality.score:.1f}<span>/100</span></div>
            </div>
            <div class="quality-rating">{quality.rating}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("How this score was calculated"):
        if not quality.issues:
            st.success("No scored data-quality issues were detected.")
        for issue in quality.issues:
            st.markdown(f"**{issue.category}: -{issue.deduction:.1f} points**")
            st.write(issue.detail)
            st.caption(issue.recommendation)
