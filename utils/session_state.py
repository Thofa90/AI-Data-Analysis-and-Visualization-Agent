"""Centralized Streamlit session-state lifecycle."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


DATASET_DEPENDENT_DEFAULTS: dict[str, Any] = {
    "active_dataframe": None,
    "original_dataframe": None,
    "active_dataset_id": None,
    "dataset_profile": None,
    "detected_metrics": [],
    "custom_metrics": [],
    "metric_mode": "Automatic",
    "analysis_history": [],
    "tool_history": [],
    "chart_history": [],
    "saved_insights": [],
    "cleaning_history": [],
    "evaluation_history": [],
    "eda_summary": None,
    "current_chart": None,
    "pending_cleaning_plan": None,
    "cleaning_undo_stack": [],
    "pending_question": None,
    "pending_clarification": None,
    "dataset_query_guide": None,
    "query_builder_draft": "",
    "benchmark_results": [],
    "selected_sheet": None,
    "excel_sheets": [],
    "currency_code": "USD",
    "currency_symbol": "$",
}


def _fresh(value: Any) -> Any:
    return value.copy() if isinstance(value, (list, dict, set)) else value


def initialize_state_mapping(state: MutableMapping[str, Any], default_model: str) -> None:
    """Initialize all application state keys in a testable mapping."""
    defaults = {
        "uploaded_file": None,
        "chat_messages": [],
        "current_page": "Home",
        "selected_model": default_model,
        "ollama_status": None,
        "file_error": None,
        **DATASET_DEPENDENT_DEFAULTS,
    }
    for key, value in defaults.items():
        if key not in state:
            state[key] = _fresh(value)


def initialize_session_state(default_model: str) -> None:
    """Initialize Streamlit session state."""
    import streamlit as st

    initialize_state_mapping(st.session_state, default_model)


def clear_dataset_state(state: MutableMapping[str, Any], preserve_chat: bool = True) -> None:
    """Clear results that cannot safely cross dataset boundaries."""
    for key, default in DATASET_DEPENDENT_DEFAULTS.items():
        state[key] = _fresh(default)
    if not preserve_chat:
        state["chat_messages"] = []
    state["file_error"] = None


def clear_conversation_state(state: MutableMapping[str, Any]) -> None:
    """Clear chat and analysis context without unloading the dataset."""
    state["chat_messages"] = []
    for key in ("analysis_history", "tool_history", "chart_history"):
        state[key] = []
