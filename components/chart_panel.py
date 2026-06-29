"""Shared visualization insight presentation."""

from __future__ import annotations

from html import escape

import streamlit as st

from services.chart_insight_service import ChartInsight


def _section(label: str, value: str | None) -> str:
    if not value:
        return ""
    return (
        f'<div class="chart-insight-section">'
        f'<span>{escape(label)}</span><p>{escape(value)}</p></div>'
    )


def render_chart_insight(insight: ChartInsight, key_prefix: str | None = None) -> None:
    """Render a structured evidence-based insight below a chart."""
    base_key = key_prefix or str(abs(hash(insight.chart_title + insight.key_finding)))
    mode = st.radio(
        "Insight detail",
        ["Compact", "Detailed"],
        horizontal=True,
        key=f"insight_mode_{base_key}",
    )
    if mode == "Compact":
        body = "".join((
            _section("Key finding", insight.key_finding),
            _section("Why it matters", insight.interpretation),
            _section("Recommended next step", insight.recommended_next_step),
        ))
    else:
        body = "".join((
            _section("Key finding", insight.key_finding),
            _section("Supporting evidence", insight.supporting_evidence),
            _section("Interpretation", insight.interpretation),
            _section("Caution", insight.caution),
            _section("Recommended next step", insight.recommended_next_step),
        ))
    strength = insight.evidence_strength.title()
    st.markdown(
        f"""
        <div class="chart-insight-card">
            <div class="chart-insight-label">Chart Insight · Evidence: {escape(strength)}</div>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if insight.evidence and st.checkbox(
        "Show statistical evidence",
        key=f"show_evidence_{base_key}",
    ):
        st.json(insight.evidence.model_dump(mode="json"))
