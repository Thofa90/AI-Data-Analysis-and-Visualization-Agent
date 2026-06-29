"""Dynamic dataset metric cards."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.profile_models import DatasetProfile
from utils.formatting import format_bytes, format_number


def render_dataset_metrics(
    dataframe: pd.DataFrame | None,
    profile: DatasetProfile | None = None,
) -> None:
    """Render real dataset measurements or a clear empty state."""
    if dataframe is None:
        st.info("Upload a CSV or Excel dataset to populate the dashboard.")
        return

    total_cells = max(dataframe.shape[0] * dataframe.shape[1], 1)
    missing = profile.total_missing if profile else int(dataframe.isna().sum().sum())
    numeric = len(profile.numeric_columns) if profile else len(dataframe.select_dtypes(include="number").columns)
    categorical = (
        len(profile.categorical_columns)
        if profile else len(dataframe.select_dtypes(include=["object", "category", "string"]).columns)
    )
    memory = profile.memory_bytes if profile else int(dataframe.memory_usage(deep=True).sum())
    cards = [
        ("Total Rows", format_number(len(dataframe)), "Dataset records"),
        ("Total Columns", format_number(len(dataframe.columns)), "Detected fields"),
        ("Missing Values", f"{missing / total_cells:.2%}", f"{missing:,} cells"),
        ("Numeric Columns", str(numeric), "Number-compatible"),
        ("Categorical Columns", str(categorical), "Text and categories"),
        ("Dataset Memory", format_bytes(memory), "In-memory footprint"),
    ]
    columns = st.columns(3)
    for index, (label, value, detail) in enumerate(cards):
        with columns[index % 3]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-detail">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
