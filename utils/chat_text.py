"""Safe text normalization and Markdown rendering helpers for chat answers."""

from __future__ import annotations

import logging
import re

from agent.schemas import ChatNarrativeResponse

LOGGER = logging.getLogger(__name__)


def normalize_chat_text(text: str) -> str:
    """Normalize generated plain text without touching tables, code, or URLs."""
    if not text:
        return ""
    normalized = str(text).replace("\u00a0", " ")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"([,;:!?])(?=\S)", r"\1 ", normalized)
    normalized = re.sub(r"(?<!\d)\.(?=[A-Za-z0-9])", ". ", normalized)
    normalized = re.sub(r"(?<=\*\*)\(", " (", normalized)
    normalized = re.sub(r"\)(?=[A-Za-z])", ") ", normalized)
    normalized = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", normalized)
    normalized = re.sub(r"\bhas(?=the)", "has ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bthe(?=highest|lowest|largest|smallest)", "the ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(highest|lowest)(?=[A-Za-z])", r"\1 ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bin(?=\d{4}\b)", "in ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def has_unbalanced_bold_markers(text: str) -> bool:
    """Return True when Markdown bold markers cannot be paired safely."""
    return str(text).count("**") % 2 != 0


def _has_malformed_emphasis(text: str) -> bool:
    return any(
        re.search(pattern, text)
        for pattern in (
            r"\*\*\s*[.;:,!?]\s*\*\*",
            r"\*\*[^*\n]*\*\*\(",
            r"\*\*[A-Za-z0-9]+(?=[A-Z])",
            r"\b(?:has|is|was|were|from|to)\*\*",
        )
    )


def sanitize_markdown(text: str) -> str:
    """Normalize chat Markdown and remove broken emphasis markers."""
    sanitized = normalize_chat_text(text)
    original = sanitized
    if has_unbalanced_bold_markers(sanitized) or _has_malformed_emphasis(sanitized):
        LOGGER.warning("Removed unbalanced bold markers from chat answer: %s", original)
        sanitized = sanitized.replace("**", "")
        sanitized = normalize_chat_text(sanitized)
    if sanitized.count("*") % 2 != 0:
        LOGGER.warning("Removed unbalanced emphasis markers from chat answer: %s", original)
        sanitized = sanitized.replace("*", "")
        sanitized = normalize_chat_text(sanitized)
    sanitized = re.sub(r"\*\*\s*([^*\n]+?)\s*\*\*", r"**\1**", sanitized)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized.strip()


def escape_streamlit_math(text: str) -> str:
    """Escape dollar signs so currency is not rendered as inline math."""
    if not text:
        return ""
    return re.sub(r"(?<!\\)\$", r"\\$", str(text))


def _clean_field(value: str | None) -> str:
    return sanitize_markdown(value or "").replace("*", "")


def render_chat_narrative_markdown(response: ChatNarrativeResponse) -> str:
    """Render a structured narrative into predictable, valid Markdown."""
    sections: list[str] = []
    summary = _clean_field(response.summary)
    if summary:
        sections.append(f"**Summary**\n\n{summary}")
    if response.key_findings:
        findings = "\n".join(f"- {_clean_field(item)}" for item in response.key_findings if _clean_field(item))
        if findings:
            sections.append(f"**Key findings**\n\n{findings}")
    for metric in response.metric_summaries:
        heading = _clean_field(metric.metric_label)
        lines: list[str] = []
        if metric.total_value:
            lines.append(f"Total: {_clean_field(metric.total_value)}")
        if metric.highest_period and metric.highest_value:
            lines.append(
                f"Highest: {_clean_field(metric.highest_period)} "
                f"({_clean_field(metric.highest_value)})"
            )
        if metric.lowest_period and metric.lowest_value:
            lines.append(
                f"Lowest: {_clean_field(metric.lowest_period)} "
                f"({_clean_field(metric.lowest_value)})"
            )
        if metric.trend_text:
            lines.append(_clean_field(metric.trend_text))
        if heading and lines:
            sections.append(f"**{heading}**\n\n" + "\n\n".join(lines))
    if response.caution:
        sections.append(f"**Caution**\n\n{_clean_field(response.caution)}")
    if response.recommended_next_step:
        sections.append(
            f"**Recommended next step**\n\n{_clean_field(response.recommended_next_step)}"
        )
    return "\n\n".join(sections).strip()
