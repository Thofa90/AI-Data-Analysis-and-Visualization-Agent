"""Automatic and manually customized key metric cards."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from services.metric_detector import MetricResult, calculate_manual_metric
from services.profile_models import DatasetProfile
from utils.formatting import format_compact_number, format_currency


def format_metric_value(metric: MetricResult, currency_symbol: str = "$") -> str:
    """Format a metric according to its explicit display setting."""
    value = metric.value
    if metric.number_format == "currency":
        return format_currency(value, symbol=currency_symbol)
    if metric.number_format == "percentage":
        return f"{value * 100:.2f}%" if abs(value) <= 1 else f"{value:.2f}%"
    if isinstance(value, int) or float(value).is_integer():
        return format_compact_number(value, decimals=0)
    return format_compact_number(value)


def render_key_metric_cards(metrics: list[MetricResult]) -> None:
    """Render verified metrics in responsive accent cards."""
    if not metrics:
        st.info("No metrics are available for this dataset.")
        return
    columns = st.columns(min(len(metrics), 4))
    for index, metric in enumerate(metrics):
        with columns[index % len(columns)]:
            st.markdown(
                f"""
                <div class="key-metric-card accent-{index % 4}">
                    <div class="metric-label">{escape(metric.label)}</div>
                    <div class="metric-value">{escape(format_metric_value(
                        metric, st.session_state.currency_symbol
                    ))}</div>
                    <div class="metric-detail">{escape(metric.aggregation.title())}
                    {escape(metric.column or "dataset")}</div>
                    {f'<div class="metric-comparison">{metric.comparison_percentage:+.1f}% {escape(metric.comparison_label or "")}</div>'
                     if metric.comparison_percentage is not None else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_metric_customizer(dataframe: pd.DataFrame, profile: DatasetProfile) -> None:
    """Render an allowlisted manual metric form."""
    st.session_state.metric_mode = st.radio(
        "Metric selection",
        ["Automatic", "Manual"],
        horizontal=True,
        index=0 if st.session_state.metric_mode == "Automatic" else 1,
    )
    if st.session_state.metric_mode == "Automatic":
        return

    columns = list(dataframe.columns)
    with st.form("manual_metric_form", border=False):
        row = st.columns([2, 1.4, 2, 1.3])
        column = row[0].selectbox("Column", columns)
        is_numeric = pd.api.types.is_numeric_dtype(dataframe[column])
        aggregations = (
            ["sum", "mean", "median", "minimum", "maximum", "count", "unique count"]
            if is_numeric else ["count", "unique count"]
        )
        aggregation = row[1].selectbox("Aggregation", aggregations)
        label = row[2].text_input("Display label", placeholder="Optional custom label")
        number_format = row[3].selectbox("Format", ["number", "currency", "percentage"])
        compare = st.checkbox(
            "Compare the latest period with the previous period",
            disabled=not profile.datetime_columns,
            help="Available only when the profile contains a valid datetime column.",
        )
        date_column = None
        comparison_period = None
        if compare:
            comparison_row = st.columns(2)
            date_column = comparison_row[0].selectbox("Date column", profile.datetime_columns)
            comparison_period = comparison_row[1].selectbox("Period", ["month", "quarter", "year"])
        submitted = st.form_submit_button("Add metric", width="stretch")
    if submitted:
        try:
            metric = calculate_manual_metric(
                dataframe,
                column,
                aggregation,
                label=label or None,
                number_format=number_format,
                date_column=date_column,
                comparison_period=comparison_period,
            )
            st.session_state.custom_metrics = [*st.session_state.custom_metrics, metric]
        except ValueError as exc:
            st.error(str(exc))

    if st.session_state.custom_metrics and st.button("Clear custom metrics"):
        st.session_state.custom_metrics = []
        st.rerun()
