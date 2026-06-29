"""Application sidebar, upload flow, navigation, and local AI status."""

from __future__ import annotations

import logging
import json
from pathlib import Path

import streamlit as st

from config.settings import Settings
from services.file_loader import FileLoadError, get_excel_sheet_names, load_dataset
from services.dataset_profiler import profile_dataset
from services.metric_detector import detect_key_metrics
from services.ollama_service import check_ollama
from agent.tool_registry import active_tool_names
from utils.formatting import format_bytes
from utils.navigation import navigate_to
from utils.session_state import clear_conversation_state, clear_dataset_state

LOGGER = logging.getLogger(__name__)

NAVIGATION = [
    "Home",
    "Chat & Ask",
    "EDA Dashboard",
    "Data Overview",
    "Visualizations",
    "Correlations",
    "Data Cleaning",
    "Reports",
    "Saved Insights",
    "Evaluation",
    "Settings",
]

CURRENCIES = {
    "USD ($)": ("USD", "$"),
    "EUR (€)": ("EUR", "€"),
    "GBP (£)": ("GBP", "£"),
    "JPY (¥)": ("JPY", "¥"),
    "INR (₹)": ("INR", "₹"),
    "BDT (৳)": ("BDT", "৳"),
}


@st.cache_data(show_spinner=False)
def _load_cached(filename: str, data: bytes, sheet_name: str | None, max_size_mb: int):
    return load_dataset(filename, data, sheet_name=sheet_name, max_size_mb=max_size_mb)


@st.cache_data(show_spinner=False)
def _sheet_names_cached(filename: str, data: bytes, max_size_mb: int) -> list[str]:
    return get_excel_sheet_names(filename, data, max_size_mb)


def _render_ollama(settings: Settings) -> None:
    status = check_ollama(settings.ollama_base_url)
    st.session_state.ollama_status = status
    status_class = "online" if status.online else "offline"
    status_label = "Ollama Online" if status.online else "Ollama Offline"
    st.markdown(f'<div class="status-pill {status_class}">{status_label}</div>', unsafe_allow_html=True)

    available_supported = [model for model in settings.supported_models if model in status.models]
    model_options = available_supported or list(settings.supported_models)
    current = st.session_state.selected_model
    if current not in model_options:
        current = model_options[0]
    st.session_state.selected_model = st.selectbox(
        "Model",
        model_options,
        index=model_options.index(current),
        help="Only installed supported models are selectable when Ollama is online.",
    )
    if status.online and not available_supported:
        st.caption(
            "Supported models are not installed. "
            f"Pull {settings.default_model} or {settings.fallback_model}."
        )
    elif not status.online:
        st.caption(status.message)


def _activate_dataset(uploaded_file, file_bytes: bytes, sheet_name: str | None, settings: Settings) -> None:
    loaded = _load_cached(uploaded_file.name, file_bytes, sheet_name, settings.max_upload_size_mb)
    dataset_changed = loaded.fingerprint != st.session_state.active_dataset_id
    sheet_changed = loaded.sheet_name != st.session_state.selected_sheet
    if dataset_changed or sheet_changed:
        clear_dataset_state(st.session_state, preserve_chat=True)
        LOGGER.info("Active dataset changed: %s [%s]", loaded.filename, loaded.sheet_name or "CSV")
    if dataset_changed or sheet_changed:
        st.session_state.active_dataframe = loaded.dataframe.copy()
        st.session_state.original_dataframe = loaded.dataframe.copy(deep=True)
        st.session_state.active_dataset_id = loaded.fingerprint
        st.session_state.selected_sheet = loaded.sheet_name
        st.session_state.uploaded_file = {
            "name": loaded.filename,
            "size": loaded.file_size,
        }
        profile = profile_dataset(loaded.dataframe)
        st.session_state.dataset_profile = profile
        st.session_state.detected_metrics = detect_key_metrics(loaded.dataframe, profile)
        st.session_state.file_error = None


def _render_upload(settings: Settings) -> None:
    st.markdown("#### Dataset")
    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
        help=f"CSV or Excel, up to {settings.max_upload_size_mb} MB.",
    )
    if uploaded_file is not None:
        data = uploaded_file.getvalue()
        extension = Path(uploaded_file.name).suffix.lower()
        try:
            selected_sheet = None
            if extension in {".xlsx", ".xls"}:
                sheets = _sheet_names_cached(uploaded_file.name, data, settings.max_upload_size_mb)
                st.session_state.excel_sheets = sheets
                current = st.session_state.selected_sheet
                index = sheets.index(current) if current in sheets else 0
                selected_sheet = st.selectbox("Worksheet", sheets, index=index)
            _activate_dataset(uploaded_file, data, selected_sheet, settings)
        except FileLoadError as exc:
            st.session_state.file_error = str(exc)
            LOGGER.warning("File upload rejected: %s", exc)

    metadata = st.session_state.uploaded_file
    dataframe = st.session_state.active_dataframe
    if metadata and dataframe is not None:
        st.markdown(
            f"""
            <div class="dataset-panel">
                <strong>{metadata["name"]}</strong><br>
                <span>{len(dataframe):,} rows · {len(dataframe.columns):,} columns ·
                {format_bytes(metadata["size"])}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.selected_sheet:
            st.caption(f"Worksheet: {st.session_state.selected_sheet}")
        current_currency = next(
            (
                label for label, value in CURRENCIES.items()
                if value[0] == st.session_state.currency_code
            ),
            "USD ($)",
        )
        selected_currency = st.selectbox(
            "Dataset currency",
            list(CURRENCIES),
            index=list(CURRENCIES).index(current_currency),
            help="Used for monetary chart axes, hover values, and displayed results.",
        )
        (
            st.session_state.currency_code,
            st.session_state.currency_symbol,
        ) = CURRENCIES[selected_currency]
    if st.session_state.file_error:
        st.error(st.session_state.file_error)


def render_sidebar(settings: Settings) -> str:
    """Render sidebar and return the selected page."""
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">DI</div>'
            '<div><strong>Data Insight Agent</strong><span>AI-Powered Data Analyst</span></div></div>',
            unsafe_allow_html=True,
        )
        _render_ollama(settings)
        st.markdown("#### Navigation")
        selected_page = st.radio(
            "Navigation",
            NAVIGATION,
            key="current_page",
            label_visibility="collapsed",
        )
        _render_upload(settings)

        st.markdown("#### Quick Actions")
        action_columns = st.columns(2)
        with action_columns[0]:
            st.button(
                "Run Auto EDA",
                disabled=st.session_state.active_dataframe is None,
                width="stretch",
                on_click=navigate_to,
                args=("EDA Dashboard",),
            )
        with action_columns[1]:
            st.button(
                "Export Results",
                disabled=st.session_state.active_dataframe is None,
                width="stretch",
                on_click=navigate_to,
                args=("Reports",),
            )
        if st.button("Clear Chat", width="stretch"):
            clear_conversation_state(st.session_state)
            st.toast("Conversation memory cleared.")
        if st.button("Clear Memory", width="stretch"):
            clear_conversation_state(st.session_state)
            st.session_state.saved_insights = []
            st.session_state.evaluation_history = []
            st.toast("Session analysis memory cleared.")
        if st.button("Reset Application", width="stretch"):
            clear_dataset_state(st.session_state, preserve_chat=False)
            st.session_state.uploaded_file = None
            st.rerun()

        with st.expander("Agent & Memory"):
            st.write(f"Model: `{st.session_state.selected_model}`")
            st.write("Active tools:")
            for tool_name in active_tool_names():
                st.caption(tool_name)
            st.write(f"Stored messages: {len(st.session_state.chat_messages)}")
            st.write(f"Analysis history: {len(st.session_state.analysis_history)}")
            memory_bytes = len(json.dumps(
                {
                    "messages": st.session_state.chat_messages,
                    "analysis": st.session_state.analysis_history,
                },
                default=str,
            ).encode("utf-8"))
            st.write(f"Conversation memory estimate: {format_bytes(memory_bytes)}")
            if st.session_state.analysis_history:
                st.write("Recent questions:")
                for item in st.session_state.analysis_history[-5:][::-1]:
                    st.caption(item.get("question", ""))
            st.caption("Memory is session-only and is not persisted.")
    return selected_page
