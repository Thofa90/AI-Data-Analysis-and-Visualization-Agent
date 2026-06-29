"""Session-memory helpers for follow-up analytical questions."""

from __future__ import annotations

from typing import Any

from agent.schemas import AgentResponse


def recent_context(history: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """Return compact recent plans without stale result payloads."""
    compact = []
    for item in history[-limit:]:
        plan = item.get("plan", {})
        compact.append({
            "question": item.get("question"),
            "tool_name": plan.get("tool_name"),
            "arguments": plan.get("arguments", {}),
            "chart_spec": plan.get("chart_spec"),
            "clarification": plan.get("clarification"),
        })
    return compact


def store_response(history: list[dict[str, Any]], response: AgentResponse) -> list[dict[str, Any]]:
    """Append a JSON-safe response with a bounded history."""
    return [*history, response.model_dump(mode="json")][-100:]
