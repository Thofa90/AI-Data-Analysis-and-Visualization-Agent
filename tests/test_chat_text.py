"""Chat text normalization and deterministic narrative rendering tests."""

from agent.schemas import ChatNarrativeResponse, MetricSummary
from utils.chat_text import (
    escape_streamlit_math,
    has_unbalanced_bold_markers,
    normalize_chat_text,
    render_chat_narrative_markdown,
    sanitize_markdown,
)


def test_normalize_chat_text_fixes_spacing_and_blank_lines() -> None:
    raw = "highestprofit:**2022**(23.2M).\n\n\nRevenue decreased in2023"

    normalized = normalize_chat_text(raw)

    assert "highest profit: **2022** (23.2M)." in normalized
    assert "in 2023" in normalized
    assert "\n\n\n" not in normalized


def test_sanitize_markdown_removes_unbalanced_bold_and_single_asterisks() -> None:
    raw = "Revenue total: 167.7M **; decreased from **52M * in2023"

    sanitized = sanitize_markdown(raw)

    assert "*" not in sanitized
    assert "in 2023" in sanitized
    assert "  " not in sanitized


def test_sanitize_markdown_removes_malformed_balanced_stars_and_repairs_words() -> None:
    raw = (
        "Sub-Saharan Africa has the highest Total Revenue: 356.7M **. "
        "**Sub-SaharanAfrica **hasthehighestTotalProfit : **101.7M. "
        "Sub-Saharan Africa has the highest Units Sold: 1.4M."
    )

    sanitized = sanitize_markdown(raw)

    assert "*" not in sanitized
    assert "Sub-Saharan Africa has the highest Total Profit: 101.7M" in sanitized
    assert "hasthehighest" not in sanitized
    assert "Sub-SaharanAfrica" not in sanitized


def test_escape_streamlit_math_preserves_currency_as_text() -> None:
    text = "Revenue decreased from $515.7M in 2021 to $336M in 2023."

    escaped = escape_streamlit_math(text)

    assert escaped == "Revenue decreased from \\$515.7M in 2021 to \\$336M in 2023."
    assert escape_streamlit_math(escaped) == escaped


def test_render_chat_narrative_markdown_uses_consistent_sections() -> None:
    narrative = ChatNarrativeResponse(
        summary="Asia generated $167.7M in total revenue and $50.8M in total profit across 2021-2023.",
        key_findings=[
            "Total revenue was highest in 2022 at $72.1M.",
            "Total profit was highest in 2022 at $23.2M.",
            "Both metrics were lowest in 2023.",
        ],
        metric_summaries=[
            MetricSummary(
                metric_label="Revenue",
                total_value="$167.7M",
                highest_period="2022",
                highest_value="$72.1M",
                lowest_period="2023",
                lowest_value="$43.6M",
                trend_text="Revenue decreased from $52.0M in 2021 to $43.6M in 2023.",
            ),
            MetricSummary(
                metric_label="Profit",
                total_value="$50.8M",
                highest_period="2022",
                highest_value="$23.2M",
                lowest_period="2023",
                lowest_value="$13.1M",
                trend_text="Profit decreased from $14.4M in 2021 to $13.1M in 2023.",
            ),
        ],
    )

    markdown = render_chat_narrative_markdown(narrative)

    assert markdown.startswith("**Summary**")
    assert "**Key findings**" in markdown
    assert "**Revenue**" in markdown
    assert "**Profit**" in markdown
    assert "Highest: 2022 ($72.1M)" in markdown
    assert "highestprofit" not in markdown
    assert "in2023" not in markdown
    assert markdown.count("**") % 2 == 0
