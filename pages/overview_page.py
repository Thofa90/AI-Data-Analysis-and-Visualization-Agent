"""Structured dataset overview."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.metric_cards import render_dataset_metrics
from components.quality_card import render_quality_card
from services.profile_models import DatasetProfile
from utils.formatting import format_bytes


def _column_table(profile: DatasetProfile) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Column": column.name,
            "Type": column.kind.title(),
            "Pandas dtype": column.pandas_dtype,
            "Missing": column.missing_count,
            "Missing %": round(column.missing_percentage, 2),
            "Unique": column.unique_count,
            "Unique %": round(column.unique_percentage, 2),
            "Outliers": column.outlier_count,
            "Flags": ", ".join(filter(None, [
                "Likely ID" if column.is_likely_id else "",
                "Constant" if column.is_constant else "",
                "High cardinality" if column.is_high_cardinality else "",
                "Type issue" if column.potential_type_issue else "",
            ])),
        }
        for column in profile.columns
    ])


def render_overview_page() -> None:
    """Render schema, quality, missingness, and summary statistics."""
    st.title("Data Overview")
    dataframe = st.session_state.active_dataframe
    profile: DatasetProfile | None = st.session_state.dataset_profile
    if dataframe is None or profile is None:
        st.info("Upload a dataset from the sidebar to generate its profile.")
        return

    render_dataset_metrics(dataframe, profile)
    left, right = st.columns([1, 1.6])
    with left:
        render_quality_card(profile.quality)
    with right:
        st.markdown("### Profile Summary")
        summary = pd.DataFrame({
            "Measure": [
                "Dataset memory", "Duplicate rows", "Likely ID columns",
                "Constant columns", "High-cardinality columns", "Potential type problems",
            ],
            "Value": [
                format_bytes(profile.memory_bytes),
                f"{profile.duplicate_rows:,} ({profile.duplicate_percentage:.2f}%)",
                str(len(profile.id_columns)),
                str(len(profile.constant_columns)),
                str(len(profile.high_cardinality_columns)),
                str(len(profile.potential_type_problems)),
            ],
        })
        st.dataframe(summary, width="stretch", hide_index=True)

    st.markdown("### Column Schema")
    st.dataframe(_column_table(profile), width="stretch", hide_index=True)

    numeric_tab, category_tab, missing_tab, date_tab = st.tabs(
        ["Numeric Summary", "Categorical Summary", "Missing Values", "Date Ranges"]
    )
    with numeric_tab:
        if profile.numeric_columns:
            st.dataframe(dataframe[profile.numeric_columns].describe().T, width="stretch")
        else:
            st.info("No non-identifier numeric columns were detected.")
    with category_tab:
        if profile.categorical_columns:
            st.dataframe(dataframe[profile.categorical_columns].describe().T, width="stretch")
        else:
            st.info("No categorical columns were detected.")
    with missing_tab:
        missing = _column_table(profile)
        missing = missing[missing["Missing"] > 0][["Column", "Missing", "Missing %"]]
        if missing.empty:
            st.success("No missing values were detected.")
        else:
            st.dataframe(missing, width="stretch", hide_index=True)
    with date_tab:
        date_rows = [
            {"Column": column.name, "Minimum": column.minimum, "Maximum": column.maximum}
            for column in profile.columns if column.kind == "datetime"
        ]
        if date_rows:
            st.dataframe(pd.DataFrame(date_rows), width="stretch", hide_index=True)
        else:
            st.info("No datetime columns were detected.")
