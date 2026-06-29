"""Phase 1 tests for session-state boundaries."""

from __future__ import annotations

from utils.session_state import clear_dataset_state, initialize_state_mapping


def test_state_initialization_is_centralized() -> None:
    state: dict = {}
    initialize_state_mapping(state, "qwen3:4b")

    assert state["selected_model"] == "qwen3:4b"
    assert state["chat_messages"] == []
    assert "evaluation_history" in state
    assert state["detected_metrics"] == []
    assert state["metric_mode"] == "Automatic"
    assert state["currency_code"] == "USD"
    assert state["currency_symbol"] == "$"


def test_dataset_reset_preserves_general_chat_by_default() -> None:
    state = {
        "chat_messages": [{"role": "user", "content": "hello"}],
        "analysis_history": [{"result": 10}],
        "active_dataframe": object(),
    }
    clear_dataset_state(state, preserve_chat=True)

    assert state["chat_messages"]
    assert state["analysis_history"] == []
    assert state["active_dataframe"] is None
