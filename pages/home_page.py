"""Main application dashboard."""

from __future__ import annotations

import streamlit as st

from components.metric_cards import render_dataset_metrics
from components.key_metrics import render_key_metric_cards, render_metric_customizer
from components.quality_card import render_quality_card
from components.status_cards import render_foundation_status
from config.settings import Settings
from services.profile_models import DatasetProfile
from utils.navigation import navigate_to


@st.dialog("Documentation")
def _documentation_dialog() -> None:
    st.write("Upload, profile, analyze, visualize, clean, export, and evaluate datasets locally.")
    st.code("streamlit run app.py", language="bash")


@st.dialog("Tutorial")
def _tutorial_dialog() -> None:
    st.write("Upload a CSV or Excel file from the sidebar. For workbooks, choose a worksheet, then review the live dataset cards.")


@st.dialog("About This App")
def _about_dialog(settings: Settings) -> None:
    st.write(f"{settings.app_name} keeps calculations local and will use approved deterministic tools for analysis.")
    st.caption("Calculations are verified with pandas or SciPy; Ollama is limited to planning and explanation.")


def render_home_page(settings: Settings) -> None:
    """Render the upload-first home dashboard."""
    header, actions = st.columns([3, 2], vertical_alignment="center")
    with header:
        st.title("Hello! I'm your Data Analysis Agent")
        st.caption("Upload a dataset to inspect its structure, quality, and automatically detected key metrics.")
    with actions:
        buttons = st.columns(3)
        if buttons[0].button("Docs", width="stretch"):
            _documentation_dialog()
        if buttons[1].button("Tutorial", width="stretch"):
            _tutorial_dialog()
        if buttons[2].button("About", width="stretch"):
            _about_dialog(settings)

    dataframe = st.session_state.active_dataframe
    profile: DatasetProfile | None = st.session_state.dataset_profile
    render_dataset_metrics(dataframe, profile)

    if dataframe is not None and profile is not None:
        metric_header, quality_link = st.columns([4, 1], vertical_alignment="center")
        with metric_header:
            st.markdown("### Detected Key Metrics")
        with quality_link:
            st.button(
                "Open Data Overview",
                width="stretch",
                on_click=navigate_to,
                args=("Data Overview",),
            )
        with st.expander("Customize metrics"):
            render_metric_customizer(dataframe, profile)
        metrics = (
            st.session_state.detected_metrics
            if st.session_state.metric_mode == "Automatic"
            else st.session_state.custom_metrics
        )
        render_key_metric_cards(metrics)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### Dataset Workspace")
        if dataframe is None:
            st.markdown(
                '<div class="empty-panel"><h3>Your data workspace is ready</h3>'
                "<p>Upload a supported dataset from the sidebar to begin.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            st.dataframe(dataframe.head(20), width="stretch", hide_index=True)
            st.caption("Previewing the first 20 rows. Profiling is deterministic and local.")
    with right:
        if profile is not None:
            st.markdown("### Data Quality")
            render_quality_card(profile.quality)
        else:
            st.markdown("### System Readiness")
            status = st.session_state.ollama_status
            if status and status.online:
                st.success(f"Ollama is online with {len(status.models)} installed model(s).")
            else:
                st.warning("Ollama is offline. File inspection remains available.")
            st.markdown(
                """
                - File validation: active
                - Dataset profiling: active
                - Dynamic metrics: active
                - Conversational analysis: active
                - Evaluation engine: active
                """
            )

    render_foundation_status()
    st.markdown("### Explore Features")
    actions = st.columns(5)
    destinations = [
        ("Conversational Analysis", "Chat & Ask"),
        ("Auto EDA", "EDA Dashboard"),
        ("Smart Visualizations", "Visualizations"),
        ("Export & Share", "Reports"),
        ("Evaluation", "Evaluation"),
    ]
    for column, (label, destination) in zip(actions, destinations, strict=True):
        column.button(
            label,
            key=f"feature_{destination}",
            width="stretch",
            on_click=navigate_to,
            args=(destination,),
        )
