"""Reports and export center."""

from __future__ import annotations

import streamlit as st

from services.eda_service import generate_eda_summary
from services.export_service import dataframe_to_csv, dataframe_to_excel, history_to_json
from services.report_service import generate_html_report, generate_pdf_report


def render_reports_page() -> None:
    st.title("Reports & Exports")
    dataframe = st.session_state.active_dataframe
    profile = st.session_state.dataset_profile
    if dataframe is None or profile is None:
        st.info("Upload a dataset before generating exports.")
        return
    metadata = st.session_state.uploaded_file or {}
    dataset_name = metadata.get("name", "dataset")
    summary = generate_eda_summary(dataframe, profile)
    metrics = st.session_state.detected_metrics
    st.markdown("### Data Exports")
    columns = st.columns(2)
    columns[0].download_button(
        "Download cleaned CSV",
        dataframe_to_csv(dataframe),
        file_name="cleaned_data.csv",
        mime="text/csv",
        width="stretch",
    )
    columns[1].download_button(
        "Download cleaned Excel",
        dataframe_to_excel(dataframe),
        file_name="cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    history_columns = st.columns(2)
    history_columns[0].download_button(
        "Download analysis history",
        history_to_json(st.session_state.analysis_history),
        file_name="analysis_history.json",
        mime="application/json",
        width="stretch",
    )
    history_columns[1].download_button(
        "Download cleaning history",
        history_to_json(st.session_state.cleaning_history),
        file_name="cleaning_history.json",
        mime="application/json",
        width="stretch",
    )
    st.markdown("### Summary Reports")
    try:
        html_report = generate_html_report(
            dataset_name, profile, metrics, summary.observations,
            st.session_state.saved_insights, st.session_state.analysis_history,
            st.session_state.evaluation_history, dataframe,
        )
        pdf_report = generate_pdf_report(
            dataset_name, profile, metrics, summary.observations, st.session_state.saved_insights,
            st.session_state.evaluation_history, dataframe,
        )
        report_columns = st.columns(2)
        report_columns[0].download_button(
            "Download HTML report", html_report, file_name="analysis_report.html",
            mime="text/html", width="stretch",
        )
        report_columns[1].download_button(
            "Download PDF report", pdf_report, file_name="analysis_report.pdf",
            mime="application/pdf", width="stretch",
        )
    except Exception:
        st.error("The report could not be generated. Check the application logs for details.")
