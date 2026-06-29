"""Confirmed, reversible data cleaning UI."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from services.cleaning_service import apply_cleaning, preview_cleaning
from services.dataset_profiler import profile_dataset
from services.metric_detector import detect_key_metrics


ACTION_LABELS = {
    "Remove duplicate rows": "remove_duplicates",
    "Fill numeric missing values with mean": "fill_numeric_mean",
    "Fill numeric missing values with median": "fill_numeric_median",
    "Fill categorical missing values with mode": "fill_categorical_mode",
    "Fill missing values with custom value": "fill_custom",
    "Remove selected columns": "remove_columns",
    "Rename a column": "rename_column",
    "Convert a data type": "convert_type",
    "Trim whitespace": "trim_whitespace",
    "Standardize category text": "standardize_category_text",
    "Remove IQR outliers": "remove_outliers",
}


def _parameters(action: str, dataframe) -> dict:
    numeric = list(dataframe.select_dtypes(include="number").columns)
    text = list(dataframe.select_dtypes(include=["object", "string", "category"]).columns)
    if action in {"fill_numeric_mean", "fill_numeric_median", "remove_outliers"}:
        return {"column": st.selectbox("Numeric column", numeric)} if numeric else {}
    if action in {"fill_categorical_mode", "trim_whitespace", "standardize_category_text"}:
        params = {"column": st.selectbox("Text column", text)} if text else {}
        if action == "standardize_category_text":
            params["case"] = st.selectbox("Text case", ["title", "lower", "upper"])
        return params
    if action == "fill_custom":
        return {
            "column": st.selectbox("Column", list(dataframe.columns)),
            "value": st.text_input("Custom fill value"),
        }
    if action == "remove_columns":
        return {"columns": st.multiselect("Columns to remove", list(dataframe.columns))}
    if action == "rename_column":
        return {
            "old": st.selectbox("Column to rename", list(dataframe.columns)),
            "new": st.text_input("New column name"),
        }
    if action == "convert_type":
        return {
            "column": st.selectbox("Column", list(dataframe.columns)),
            "target": st.selectbox("Target type", ["string", "integer", "float", "boolean", "datetime"]),
        }
    return {}


def _refresh_profile() -> None:
    profile = profile_dataset(st.session_state.active_dataframe)
    st.session_state.dataset_profile = profile
    st.session_state.detected_metrics = detect_key_metrics(st.session_state.active_dataframe, profile)


def render_cleaning_page() -> None:
    st.title("Data Cleaning")
    dataframe = st.session_state.active_dataframe
    if dataframe is None:
        st.info("Upload a dataset before applying cleaning operations.")
        return
    st.warning("Cleaning affects the active working copy. The original upload remains available for reset.")
    label = st.selectbox("Cleaning operation", list(ACTION_LABELS))
    action = ACTION_LABELS[label]
    params = _parameters(action, dataframe)
    if st.button("Preview planned change"):
        try:
            st.session_state.pending_cleaning_plan = preview_cleaning(dataframe, action, params)
        except ValueError as exc:
            st.error(str(exc))
    plan = st.session_state.get("pending_cleaning_plan")
    if plan:
        st.markdown("### Change Preview")
        st.write(plan.description)
        st.write(f"Affected rows: **{plan.affected_rows:,}**")
        st.write(f"Affected columns: **{', '.join(plan.affected_columns)}**")
        confirmed = st.checkbox("I confirm this change")
        if st.button("Apply confirmed change", type="primary", disabled=not confirmed):
            try:
                before = st.session_state.active_dataframe.copy(deep=True)
                cleaned = apply_cleaning(before, plan)
                st.session_state.cleaning_undo_stack = [
                    *st.session_state.cleaning_undo_stack, before
                ][-10:]
                st.session_state.active_dataframe = cleaned
                st.session_state.cleaning_history = [
                    *st.session_state.cleaning_history,
                    {
                        **plan.model_dump(mode="json"),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "rows_after": len(cleaned),
                    },
                ]
                st.session_state.pending_cleaning_plan = None
                _refresh_profile()
                st.success("Cleaning operation applied.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    controls = st.columns(2)
    if controls[0].button("Undo last cleaning", disabled=not st.session_state.cleaning_undo_stack):
        st.session_state.active_dataframe = st.session_state.cleaning_undo_stack[-1]
        st.session_state.cleaning_undo_stack = st.session_state.cleaning_undo_stack[:-1]
        if st.session_state.cleaning_history:
            st.session_state.cleaning_history = st.session_state.cleaning_history[:-1]
        _refresh_profile()
        st.rerun()
    if controls[1].button("Reset to original"):
        st.session_state.active_dataframe = st.session_state.original_dataframe.copy(deep=True)
        st.session_state.cleaning_history = []
        st.session_state.cleaning_undo_stack = []
        _refresh_profile()
        st.rerun()
    st.markdown("### Active Data Preview")
    st.dataframe(st.session_state.active_dataframe.head(30), width="stretch", hide_index=True)
    if st.session_state.cleaning_history:
        st.markdown("### Cleaning History")
        st.dataframe(st.session_state.cleaning_history, width="stretch", hide_index=True)
