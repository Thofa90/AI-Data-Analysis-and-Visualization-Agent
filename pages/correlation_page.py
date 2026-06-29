"""Correlation exploration page."""

from __future__ import annotations

import streamlit as st

from components.chart_panel import render_chart_insight
from services.chart_service import ChartSpec, create_chart
from services.chart_insight_service import generate_chart_insight


def render_correlation_page() -> None:
    """Render a correlation matrix and pair explorer."""
    st.title("Correlations")
    dataframe = st.session_state.active_dataframe
    profile = st.session_state.dataset_profile
    if dataframe is None or profile is None:
        st.info("Upload a dataset to analyze correlations.")
        return
    if len(profile.numeric_columns) < 2:
        st.info("At least two non-identifier numeric columns are required.")
        return
    figure, heatmap_result = create_chart(
        dataframe,
        ChartSpec(
            chart_type="heatmap",
            title="Correlation Matrix",
            currency_symbol=st.session_state.currency_symbol,
        ),
    )
    st.plotly_chart(figure, width="stretch")
    render_chart_insight(generate_chart_insight(heatmap_result))
    row = st.columns(2)
    first = row[0].selectbox("First numeric column", profile.numeric_columns)
    second_options = [column for column in profile.numeric_columns if column != first]
    second = row[1].selectbox("Second numeric column", second_options)
    correlation = dataframe[[first, second]].corr().iloc[0, 1]
    st.metric("Pearson correlation", f"{correlation:.3f}")
    scatter, scatter_result = create_chart(dataframe, ChartSpec(
        chart_type="scatter",
        x=first,
        y=second,
        title=f"{second} versus {first}",
        currency_symbol=st.session_state.currency_symbol,
    ))
    st.plotly_chart(scatter, width="stretch")
    render_chart_insight(generate_chart_insight(scatter_result))
