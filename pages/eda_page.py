"""Automated EDA dashboard."""

from __future__ import annotations

import streamlit as st

from components.quality_card import render_quality_card
from components.chart_panel import render_chart_insight
from services.chart_service import create_chart, figure_to_png
from services.chart_insight_service import generate_chart_insight
from services.eda_service import generate_eda_summary


def render_eda_page() -> None:
    """Render observations, distributions, outliers, and correlations."""
    st.title("EDA Dashboard")
    dataframe = st.session_state.active_dataframe
    profile = st.session_state.dataset_profile
    if dataframe is None or profile is None:
        st.info("Upload a dataset to run automated exploratory analysis.")
        return
    summary = generate_eda_summary(dataframe, profile)
    st.session_state.eda_summary = summary
    left, right = st.columns([1.5, 1])
    with left:
        st.markdown("### Automated Observations")
        for observation in summary.observations:
            st.markdown(f"- {observation}")
    with right:
        render_quality_card(profile.quality)

    st.markdown("### Recommended Analysis")
    for index, spec in enumerate(summary.chart_specs):
        try:
            spec = spec.model_copy(update={
                "currency_symbol": st.session_state.currency_symbol,
            })
            figure, result = create_chart(dataframe, spec)
            st.plotly_chart(figure, width="stretch", key=f"eda_chart_{index}")
            render_chart_insight(generate_chart_insight(result))
            st.session_state.chart_history = [
                *st.session_state.chart_history,
                result.model_dump(mode="json"),
            ][-30:]
            try:
                png = figure_to_png(figure)
                st.download_button(
                    "Download chart PNG",
                    png,
                    file_name=f"eda_chart_{index + 1}.png",
                    mime="image/png",
                    key=f"eda_download_{index}",
                )
            except ValueError:
                st.caption("PNG export is unavailable until Kaleido can access Chrome.")
        except ValueError as exc:
            st.warning(str(exc))
