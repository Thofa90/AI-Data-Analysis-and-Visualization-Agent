"""Safe analytical agent with deterministic fallback and optional Ollama planning."""

from __future__ import annotations

from difflib import SequenceMatcher
import logging
import re
from time import perf_counter
from typing import Any

import pandas as pd

from agent.memory import recent_context
from agent.schemas import (
    AgentPlan,
    AgentResponse,
    ChatNarrativeResponse,
    MetricSummary,
    ToolResult,
)
from agent.tool_registry import TOOL_REGISTRY, execute_tool, validate_tool_arguments
from config.settings import Settings
from services.chart_service import ChartSpec, create_chart
from services.date_aggregation_service import period_bounds_from_text
from services.dataset_profiler import profile_dataset
from services.eda_service import generate_eda_summary
from services.llm_service import (
    explain_with_ollama,
    plan_with_ollama,
    structured_dataset_context,
)
from services.profile_models import DatasetProfile
from utils.chat_text import render_chat_narrative_markdown, sanitize_markdown
from utils.formatting import format_metric

LOGGER = logging.getLogger(__name__)


def _current_date() -> pd.Timestamp:
    return pd.Timestamp.now()


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _category_value_aliases(value: Any) -> set[str]:
    """Return normalized aliases for a dataset category value."""
    text = str(value)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    aliases = {_normalize(text)}
    significant = [
        token
        for token in tokens
        if token not in {"and", "of", "the", "a", "an"}
    ]
    if len(significant) >= 2:
        acronym = "".join(token[0] for token in significant if token)
        if len(acronym) >= 2:
            aliases.add(acronym)
    return {alias for alias in aliases if alias}


def _category_value_matches_query(value: Any, query_text: str, normalized_query: str) -> bool:
    lowered = query_text.lower()
    value_tokens = re.findall(r"[a-z0-9]+", str(value).lower())
    full_alias = _normalize(str(value))
    for alias in _category_value_aliases(value):
        if alias != full_alias:
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lowered):
                return True
        elif len(value_tokens) == 1 or len(alias) <= 2:
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lowered):
                return True
        elif alias in normalized_query:
            return True
    return False


def _display_column_name(value: str) -> str:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    words = spaced.replace("_", " ").lower().split()
    if len(words) > 1 and words[0] == "total":
        words = words[1:]
    return " ".join(words)


SEMANTIC_SYNONYMS = {
    "revenue": {"revenue", "sales", "turnover", "income"},
    "sales": {"sales", "revenue", "turnover"},
    "profit": {"profit", "earnings", "margin"},
    "country": {"country", "countries", "nation", "nations"},
    "region": {"region", "area", "territory"},
    "quantity": {"quantity", "unit", "units", "volume"},
}

ANALYTICAL_VOCABULARY = {
    "above",
    "across",
    "after",
    "aggregate",
    "all",
    "also",
    "analysis",
    "analyze",
    "average",
    "below",
    "between",
    "calculate",
    "category",
    "chart",
    "compare",
    "correlation",
    "count",
    "data",
    "dataset",
    "delivery",
    "difference",
    "display",
    "duplicate",
    "each",
    "filter",
    "finally",
    "find",
    "frequency",
    "graph",
    "highest",
    "identify",
    "invoice",
    "insight",
    "largest",
    "lowest",
    "maximum",
    "mean",
    "median",
    "minimum",
    "missing",
    "most",
    "outlier",
    "plot",
    "region",
    "regions",
    "show",
    "smallest",
    "status",
    "sum",
    "summarize",
    "summary",
    "then",
    "total",
    "trend",
    "value",
    "values",
    "visualize",
}
for _synonyms in SEMANTIC_SYNONYMS.values():
    ANALYTICAL_VOCABULARY.update(_synonyms)


def _column_tokens(value: str) -> set[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return set(re.findall(r"[a-z0-9]+", spaced.lower().replace("_", " ")))


def _typo_vocabulary(
    columns: list[str] | None = None,
    dataframe: pd.DataFrame | None = None,
) -> set[str]:
    vocabulary = set(ANALYTICAL_VOCABULARY)
    for column in columns or []:
        vocabulary.update(_column_tokens(column))
        vocabulary.update(_column_aliases(column))
    if dataframe is not None:
        for column in dataframe.select_dtypes(
            include=["object", "string", "category"]
        ).columns:
            values = dataframe[column].dropna().astype(str).unique()
            if len(values) > 100:
                continue
            for value in values:
                if len(value) <= 40:
                    vocabulary.update(re.findall(r"[a-z0-9]+", value.lower()))
    return vocabulary


def _correct_question_typos(
    question: str,
    columns: list[str] | None = None,
    dataframe: pd.DataFrame | None = None,
) -> str:
    """Correct only high-confidence typos from a constrained vocabulary."""
    vocabulary = _typo_vocabulary(columns, dataframe)

    def is_adjacent_transposition(first: str, second: str) -> bool:
        if len(first) != len(second):
            return False
        differences = [
            index for index, pair in enumerate(zip(first, second)) if pair[0] != pair[1]
        ]
        return (
            len(differences) == 2
            and differences[1] == differences[0] + 1
            and first[differences[0]] == second[differences[1]]
            and first[differences[1]] == second[differences[0]]
        )

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        lowered = token.lower()
        if columns is None and dataframe is None and re.search(r"[a-z][A-Z]", token):
            return token
        if len(lowered) < 4 or lowered in vocabulary:
            return token
        candidates = []
        for candidate in vocabulary:
            if abs(len(candidate) - len(lowered)) > 2:
                continue
            ratio = SequenceMatcher(None, lowered, candidate).ratio()
            if is_adjacent_transposition(lowered, candidate):
                ratio = 0.95
            if ratio >= 0.82:
                candidates.append((ratio, candidate))
        candidates.sort(reverse=True)
        if not candidates:
            return token
        best_ratio, best = candidates[0]
        second_ratio = candidates[1][0] if len(candidates) > 1 else 0.0
        if best_ratio - second_ratio < 0.06:
            return token
        if token.isupper():
            return best.upper()
        if token[:1].isupper():
            return best.capitalize()
        return best

    return re.sub(r"\b[a-zA-Z][a-zA-Z0-9_]*\b", replace, question)


def _question_tokens(question: str) -> set[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", question)
    tokens = set(re.findall(r"[a-z0-9]+", spaced.lower()))
    expanded = set(tokens)
    for canonical, synonyms in SEMANTIC_SYNONYMS.items():
        if tokens & synonyms:
            expanded.add(canonical)
            expanded.update(synonyms)
    return expanded


def _has_date_intent(text: str, mentioned: list[str], date_columns: list[str]) -> bool:
    if any(column in mentioned for column in date_columns):
        return True
    date_terms = (
        "date",
        "period",
        "daily",
        "weekly",
        "monthly",
        "quarterly",
        "yearly",
        "annual",
        "month",
        "week",
        "quarter",
        "year",
        "season",
        "calendar",
        "recent",
        "latest",
        "last year",
        "this year",
        "last month",
        "this month",
        "over time",
        "time trend",
        "date wise",
        "date-wise",
        "by date",
        "by month",
        "by year",
        "by week",
        "by quarter",
    )
    return any(term in text for term in date_terms)


def _time_frequency_from_text(text: str) -> str:
    if any(term in text for term in ("daily", "day", "by date")):
        return "day"
    if any(term in text for term in ("weekly", "week")):
        return "week"
    if any(term in text for term in ("monthly", "month", "by month")):
        return "month"
    if any(term in text for term in ("quarterly", "quarter")):
        return "quarter"
    if any(term in text for term in ("yearly", "annual", "year")):
        return "year"
    return "month"


def _preferred_date_column(mentioned: list[str], date_columns: list[str]) -> str | None:
    mentioned_date = next((item for item in mentioned if item in date_columns), None)
    if mentioned_date:
        return mentioned_date
    if len(date_columns) == 1:
        return date_columns[0]
    preferred_tokens = ("order", "transaction", "invoice", "sales", "sale", "event", "date")
    ranked = [
        column
        for column in date_columns
        if any(token in column.lower() for token in preferred_tokens)
    ]
    return ranked[0] if len(ranked) == 1 else None


def _date_aggregation_from_text(text: str) -> str:
    if any(word in text for word in ("average", "mean")):
        return "mean"
    if "median" in text:
        return "median"
    if any(word in text for word in ("minimum", "lowest", "smallest", "min")):
        return "min"
    if any(word in text for word in ("maximum", "highest", "largest", "max")):
        return "max"
    if re.search(r"\bcount\b", text) or any(
        phrase in text for phrase in ("number of", "how many")
    ):
        return "count"
    return "sum"


def _date_breakdown_requested(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "for each day",
            "for each week",
            "for each month",
            "for each quarter",
            "for each year",
            "each day",
            "each week",
            "each month",
            "each quarter",
            "each year",
            "per day",
            "per week",
            "per month",
            "per quarter",
            "per year",
            "by day",
            "by date",
            "by week",
            "by month",
            "by quarter",
            "by year",
            "all day",
            "all days",
            "all week",
            "all weeks",
            "all month",
            "all months",
            "all quarter",
            "all quarters",
            "all year",
            "all years",
            "over all day",
            "over all days",
            "over all week",
            "over all weeks",
            "over all month",
            "over all months",
            "over all the month",
            "over all the months",
            "over all quarter",
            "over all quarters",
            "over all year",
            "over all years",
            "over all the year",
            "over all the years",
            "over time",
            "daily",
            "weekly",
            "monthly",
            "quarterly",
            "yearly",
        )
    )


def _column_phrase(column: str) -> str:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", column)
    return " ".join(re.findall(r"[a-z0-9]+", spaced.lower().replace("_", " ")))


def _column_role_pattern(column: str) -> str:
    """Return a regex for user-facing column roles, tolerating simple plurals."""
    parts = _column_phrase(column).split()
    if not parts:
        return re.escape(_column_phrase(column))
    escaped = [re.escape(part) for part in parts]
    escaped[-1] = f"{escaped[-1]}s?"
    return r"\s+".join(escaped)


def _find_longest_column_value_match(
    query_text: str,
    dataframe: pd.DataFrame,
    column: str,
) -> Any | None:
    normalized_query = _normalize(query_text)
    candidates = sorted(
        dataframe[column].dropna().unique(),
        key=lambda value: len(_normalize(str(value))),
        reverse=True,
    )
    for candidate in candidates:
        aliases = _category_value_aliases(candidate)
        if not aliases:
            continue
        if _category_value_matches_query(candidate, query_text, normalized_query):
            return candidate
        candidate_alias = _normalize(str(candidate))
        if (
            len(candidate_alias) >= 5
            and len(re.findall(r"[a-z0-9]+", str(candidate).lower())) >= 2
            and abs(len(candidate_alias) - len(normalized_query)) <= 2
            and SequenceMatcher(None, candidate_alias, normalized_query).ratio() >= 0.84
        ):
            return candidate
    return None


def _explicit_categorical_filter(
    question: str,
    dataframe: pd.DataFrame | None,
    categorical_columns: list[str],
    excluded_columns: set[str] | None = None,
) -> tuple[str, Any] | None:
    """Resolve role-value phrases such as "region Asia" before broad lookup."""
    if dataframe is None:
        return None
    excluded = set(excluded_columns or ())
    for column in categorical_columns:
        if column in excluded or column not in dataframe.columns:
            continue
        phrase = _column_role_pattern(column)
        patterns = (
            rf"\b(?:for|in|within|only|through|from|via)\s+{phrase}\s+(?P<value>[^,.;]+)",
            rf"\b{phrase}\s+(?:is|equals)\s+(?P<value>[^,.;]+)",
            rf"\b{phrase}\s+(?P<value>[^,.;]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if not match:
                continue
            candidate = _find_longest_column_value_match(
                match.group("value"),
                dataframe,
                column,
            )
            if candidate is not None:
                return column, candidate
    return None


def _breakdown_column_from_text(text: str, categorical_columns: list[str]) -> str | None:
    """Resolve category roles used as breakdowns, not filters."""
    for column in categorical_columns:
        phrase = _column_role_pattern(column)
        compact = re.escape(_normalize(column))
        patterns = (
            rf"\bby\s+{phrase}\b",
            rf"\bby\s+{compact}\b",
            rf"\bvs\.?\s+{phrase}\b",
            rf"\bvs\.?\s+{compact}\b",
            rf"\bversus\s+{phrase}\b",
            rf"\bversus\s+{compact}\b",
            rf"\bper\s+{phrase}\b",
            rf"\bper\s+{compact}\b",
            rf"\beach\s+{phrase}\b",
            rf"\beach\s+{compact}\b",
            rf"\bevery\s+{phrase}\b",
            rf"\bevery\s+{compact}\b",
            rf"\bfor\s+all\s+{phrase}\b",
            rf"\bfor\s+all\s+{compact}\b",
            rf"\bacross\s+all\s+{phrase}\b",
            rf"\bacross\s+all\s+{compact}\b",
            rf"\bfor\s+each\s+{phrase}\b",
            rf"\bfor\s+each\s+{compact}\b",
            rf"\bwithin\s+each\s+{phrase}\b",
            rf"\bwithin\s+each\s+{compact}\b",
            rf"\b{phrase}\s*[- ]wise\b",
            rf"\b{compact}\s*[- ]wise\b",
        )
        if any(re.search(pattern, text) for pattern in patterns):
            return column
    return None


def _which_column_from_text(text: str, categorical_columns: list[str]) -> str | None:
    """Resolve the primary category in questions such as "which sales channel..."."""
    for column in categorical_columns:
        phrase = _column_role_pattern(column)
        compact = re.escape(_normalize(column))
        patterns = (
            rf"\bwhich\s+{phrase}\b",
            rf"\bwhich\s+{compact}\b",
            rf"\bwhat\s+{phrase}\b",
            rf"\bwhat\s+{compact}\b",
            rf"\bwhich\s+(?:\w+\s+){{0,2}}{phrase}\b",
        )
        if any(re.search(pattern, text) for pattern in patterns):
            return column
    return None


def _categorical_value_count_requested(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "value count",
            "value counts",
            "count distinct",
            "count unique",
            "distinct count",
            "unique count",
            "frequency",
            "frequencies",
            "frequency distribution",
            "distribution",
            "distribution of",
            "count each",
            "count the",
            "counts by",
            "count by category",
            "how many of each",
            "number of records for each",
            "occurrences",
            "category counts",
        )
    ) or bool(
        re.search(r"\b\w+\s+counts?\s+by\s+\w+", text)
        or re.search(r"\bcount\s+(?:distinct|unique)\b.+\b(?:for\s+each|by|per)\b", text)
        or re.search(r"\b[\w\s]+?\s+counts\b", text)
        or re.search(r"\b(?:show|display|list|give)(?:\s+me)?\s+[\w\s]+?\s+count\b", text)
        or re.search(r"\bhow\s+many\b.+\b(?:are|were|is)?\s*(?:there\s+)?(?:in|for|within)\s+each\b", text)
    )


def _paired_columns_for_frequency_distribution(
    text: str,
    categorical_columns: list[str],
) -> tuple[str, str] | None:
    if "frequency" not in text and "distribution" not in text:
        return None
    mentioned = [
        column
        for column in _mentioned_columns(text, categorical_columns)
        if column in categorical_columns
    ]
    if len(mentioned) < 2:
        return None
    if re.search(r"\buse\b.+\band\b.+\b(?:frequency|distribution)\b", text):
        return mentioned[0], mentioned[1]
    return None


def _column_from_fragment(fragment: str, columns: list[str]) -> str | None:
    cleaned = re.sub(
        r"\b(show|show me|a|bar|chart|of|the|value|values|count|counts|frequency|distribution|for|each|by|per|within|what|are|is|there|in)\b",
        " ",
        fragment.lower(),
    )
    return next(iter(_mentioned_columns(cleaned or fragment, columns)), None)


def _counted_column_for_value_counts(
    text: str,
    categorical_columns: list[str],
    excluded_columns: set[str] | None = None,
) -> str | None:
    excluded = set(excluded_columns or ())
    patterns = (
        r"value\s+counts?\s+of\s+(?P<counted>.+?)(?:\s+(?:for\s+each|by|per)\s+.+)?$",
        r"distribution\s+of\s+(?P<counted>.+?)(?:\s+by\s+.+)?$",
        r"frequency(?:\s+distribution)?\s+of\s+(?P<counted>.+?)(?:\s+by\s+.+)?$",
        r"count\s+each\s+(?P<counted>.+?)\s+values?\b",
        r"count\s+(?:the\s+)?(?P<counted>.+?)\s+values?\b",
        r"(?:show|display|list|give)(?:\s+me)?\s+(?P<counted>.+?)\s+count\b",
        r"(?P<counted>.+?)\s+counts?\s+by\s+.+$",
        r"(?P<counted>.+?)\s+counts?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        column = _column_from_fragment(match.group("counted"), categorical_columns)
        if column and column not in excluded:
            return column
    for column in _mentioned_columns(text, categorical_columns):
        if column not in excluded:
            return column
    return None


def _categorical_value_count_plan(
    question: str,
    text: str,
    dataframe: pd.DataFrame | None,
    columns: list[str],
    categorical_columns: list[str],
) -> AgentPlan | None:
    if not _categorical_value_count_requested(text):
        return None
    unique_measure = bool(
        re.search(r"\bunique\b|\bdistinct\b|\bnunique\b", text)
        and re.search(r"\bcounts?\b", text)
    )
    paired_columns = _paired_columns_for_frequency_distribution(text, categorical_columns)
    primary_group = _breakdown_column_from_text(text, categorical_columns)
    counted_from_values = _mentioned_category_values(question, dataframe, categorical_columns)
    counted_value_column = counted_from_values[0] if counted_from_values else None
    if paired_columns and not primary_group:
        primary_group, counted_value_column = paired_columns
    filter_match = _detect_categorical_filter(
        question,
        dataframe,
        categorical_columns,
        excluded_columns={
            column
            for column in (primary_group, counted_value_column)
            if column
        },
    )
    filter_column, filter_value = filter_match or (None, None)
    excluded = {column for column in (primary_group, filter_column) if column}
    counted_column = counted_value_column or _counted_column_for_value_counts(
        text,
        categorical_columns,
        excluded,
    )
    if not counted_column:
        unknown_column = _unknown_value_count_column_text(text, columns)
        if unknown_column:
            suggestions = _column_suggestions(unknown_column, columns)
            available = ", ".join(suggestions or columns[:8])
            return AgentPlan(
                tool_name="",
                clarification=(
                    f'I could not find a column named "{unknown_column}". '
                    f"Available similar columns: {available}. No calculation was performed."
                ),
            )
    distinct_column = None
    if unique_measure:
        mentioned_all = _mentioned_columns(question, columns)
        distinct_column = next(
            (
                column
                for column in mentioned_all
                if column != counted_column and column != primary_group and column != filter_column
            ),
            None,
        )
        if primary_group and distinct_column and counted_column is None:
            counted_column = primary_group
            primary_group = None
    if not counted_column:
        return AgentPlan(
            tool_name="",
            clarification=(
                "Which categorical column should I count? "
                f"Categorical columns: {', '.join(categorical_columns) or 'none'}."
            ),
        )
    if unique_measure and not distinct_column:
        return AgentPlan(
            tool_name="",
            clarification="Which column should I count uniquely?",
        )
    filters = (
        [{"column": filter_column, "operator": "equals", "value": filter_value}]
        if filter_column
        else []
    )
    include_missing = any(word in text for word in ("missing", "unknown", "null", "blank"))
    percentage_requested = any(word in text for word in ("percent", "percentage", "share", "composition"))
    stacked_requested = "stacked" in text or "composition" in text
    chart_type = (
        "percentage_stacked_bar"
        if primary_group and percentage_requested
        else "stacked_bar"
        if primary_group and stacked_requested
        else "grouped_bar"
        if primary_group
        else "bar"
    )
    y_column = f"Unique {distinct_column} Count" if unique_measure else "Count"
    title_filter = f" for {filter_value}" if filter_column else ""
    title = (
        f"{_display_column_name(counted_column).title()} Counts"
        + (f" by {primary_group}" if primary_group else title_filter)
    )
    return AgentPlan(
        tool_name="analyze_categorical_value_counts",
        arguments={
            "counted_column": counted_column,
            "primary_group_column": primary_group,
            "filters": filters,
            "include_missing": include_missing,
            "normalization": "within_primary_group" if primary_group else "overall",
            "chart_type": chart_type,
            "sort_mode": "category" if "alphabet" in text else "count_descending",
            "measure_type": "distinct_count" if unique_measure else "row_count",
            "distinct_column": distinct_column,
            "original_query": question,
        },
        chart_spec=ChartSpec(
            chart_type=(
                "stacked_bar"
                if chart_type in {"stacked_bar", "percentage_stacked_bar"}
                else "grouped_bar"
                if primary_group
                else "bar"
            ),
            x=primary_group or counted_column,
            y=y_column,
            color=counted_column if primary_group else None,
            aggregation=None,
            sort_descending=False,
            title=title,
        ),
    )


def _primary_group_for_extrema(text: str, categorical_columns: list[str]) -> str | None:
    for column in categorical_columns:
        phrase = _column_role_pattern(column)
        patterns = (
            rf"\b(?:in|for|within|inside)\s+each\s+{phrase}\b",
            rf"\bevery\s+{phrase}\b",
            rf"\bper\s+{phrase}\b",
            rf"\bby\s+{phrase}\b",
            rf"\b{phrase}\s+on\s+the\s+x[- ]?axis\b",
        )
        if any(re.search(pattern, text) for pattern in patterns):
            return column
    return None


def _secondary_group_for_extrema(
    text: str,
    categorical_columns: list[str],
    primary_group: str | None,
) -> str | None:
    for column in categorical_columns:
        if column == primary_group:
            continue
        phrase = _column_role_pattern(column)
        patterns = (
            rf"\b(?:top|best(?:-performing)?|highest|lowest)\s+(?:\w+\s+){{0,3}}{phrase}\b",
            rf"\b(?:which|what)\s+{phrase}\s+(?:generates?|generated|brings?|brought|contributes?|contributed|produces?|produced|drives?|drove)\b",
            rf"\b{phrase}\s+(?:with|responsible|based|having)\b",
            rf"\b{phrase}\s+(?:has|have)\s+the\s+(?:highest|lowest|top)\b",
            rf"\bshow\s+the\s+{phrase}\b",
        )
        if any(re.search(pattern, text) for pattern in patterns):
            return column
    mentioned_secondary = [
        column
        for column in categorical_columns
        if column != primary_group and _column_tokens(column) & _question_tokens(text)
    ]
    return mentioned_secondary[0] if len(mentioned_secondary) == 1 else None


def _grouped_extrema_intent(text: str) -> tuple[str, str] | None:
    if not any(word in text for word in ("highest", "lowest", "top", "best")):
        return None
    if not any(phrase in text for phrase in ("each ", "every ", " per ", "x-axis", "x axis")):
        return None
    extremum = "min" if "lowest" in text else "max"
    return "grouped_extrema", extremum


def _column_aliases(column: str) -> set[str]:
    """Return compact singular/plural aliases for a column name."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", column)
    tokens = re.findall(r"[a-z0-9]+", spaced.lower().replace("_", " "))
    aliases = {_normalize(column)}
    for index, token in enumerate(tokens):
        if token.endswith("s") and len(token) > 3:
            singular = [*tokens]
            singular[index] = token[:-1]
            aliases.add("".join(singular))
        if len(token) > 2:
            plural = [*tokens]
            plural[index] = f"{token[:-1]}ies" if token.endswith("y") else f"{token}s"
            aliases.add("".join(plural))
    return aliases


def _mentioned_columns(question: str, columns: list[str]) -> list[str]:
    normalized_question = _normalize(question)
    exact = [
        column
        for column in columns
        if any(alias in normalized_question for alias in _column_aliases(column))
    ]
    suppressed: set[str] = set()
    lowered = question.lower()
    for column in exact:
        column_tokens = _column_tokens(column)
        if len(column_tokens) != 1:
            continue
        for other in exact:
            other_tokens = _column_tokens(other)
            if column == other or not column_tokens < other_tokens:
                continue
            token = next(iter(column_tokens))
            explicit_column_pattern = (
                rf"\b(?:and|by|per|each|every|for\s+each|within\s+each|"
                rf"vs\.?|versus)\s+{re.escape(token)}s?\b"
            )
            if not re.search(explicit_column_pattern, lowered):
                suppressed.add(column)
                break
    exact = [column for column in exact if column not in suppressed]
    question_tokens = _question_tokens(question)
    scored: list[tuple[float, int, str]] = []
    for index, column in enumerate(columns):
        if column in suppressed:
            continue
        column_tokens = _column_tokens(column)
        overlap = column_tokens & question_tokens
        if not overlap and column not in exact:
            continue
        score = len(overlap) / len(column_tokens) if column_tokens else 0
        if column in exact:
            score += 2
        exact_position = normalized_question.find(_normalize(column))
        position = (
            exact_position if exact_position >= 0 else len(normalized_question) + index
        )
        scored.append((score, -position, column))
    return [column for _, _, column in sorted(scored, reverse=True)]


def _column_profile_requested(text: str) -> bool:
    if re.search(r"\b(?:explain|describe|profile|summarize)\b.+\b(?:column|field)?\b", text):
        return True
    if re.search(r"\bwhat\s+is\b.+\b(?:column|field)\b", text):
        return True
    return any(
        phrase in text
        for phrase in (
            "tell me about",
            "what type of column",
            "what kind of column",
            "what values are in",
            "what values are stored in",
            "what values does",
            "what does",
            "column statistics",
            "show column statistics",
            "how many unique values",
            "minimum and maximum",
            "min and max",
        )
    )


def _column_lookup_candidates(column: str) -> set[str]:
    display = _display_column_name(column)
    candidates = {
        column,
        display,
        column.replace("_", " "),
        display.replace(" ", ""),
    }
    tokens = _column_tokens(column)
    if len(tokens) > 1:
        candidates.add(" ".join(tokens))
        generic = {"total", "unit", "units", "value", "amount"}
        meaningful = tokens - generic
        if meaningful:
            candidates.add(" ".join(meaningful))
    return {_normalize(candidate) for candidate in candidates if candidate}


def _resolve_column_reference(question: str, columns: list[str]) -> str | None:
    normalized = _normalize(question)
    lookup: dict[str, str] = {}
    for column in columns:
        for candidate in _column_lookup_candidates(column):
            lookup.setdefault(candidate, column)
    exact_mentions = _mentioned_columns(question, columns)
    if exact_mentions:
        return exact_mentions[0]
    matches = [
        (len(alias), column)
        for alias, column in lookup.items()
        if alias and alias in normalized
    ]
    if matches:
        return max(matches, key=lambda item: item[0])[1]
    return None


def _column_suggestions(question: str, columns: list[str], limit: int = 5) -> list[str]:
    tokens = list(_question_tokens(question))
    candidates: list[tuple[float, str]] = []
    normalized_question = _normalize(question)
    for column in columns:
        score = SequenceMatcher(None, normalized_question, _normalize(column)).ratio()
        for token in tokens:
            score = max(score, SequenceMatcher(None, token, _normalize(column)).ratio())
        candidates.append((score, column))
    return [column for score, column in sorted(candidates, reverse=True)[:limit] if score >= 0.35]


def _column_profile_plan(question: str, text: str, columns: list[str]) -> AgentPlan | None:
    if not _column_profile_requested(text):
        return None
    if any(phrase in text for phrase in ("previous result", "previous analysis", "this mean", "that mean")):
        if not any(word in text for word in ("column", "field")):
            return None
    column = _resolve_column_reference(question, columns)
    if not column:
        if not any(
            phrase in text
            for phrase in (
                "column",
                "field",
                "what type",
                "what kind",
                "what values",
                "unique values",
                "column statistics",
                "minimum and maximum",
                "min and max",
            )
        ):
            return None
        suggestions = _column_suggestions(question, columns)
        available = ", ".join(suggestions or columns[:8])
        return AgentPlan(
            tool_name="",
            clarification=(
                "I could not find a column matching your request. "
                f"Available similar columns: {available}."
            ),
        )
    display = _display_column_name(column)
    return AgentPlan(
        tool_name="profile_column",
        arguments={
            "column_name": column,
            "include_table": True,
            "include_chart": True,
            "include_examples": True,
            "include_semantic_explanation": True,
            "original_query": question,
        },
    )


def _requested_summary_statistics(text: str) -> list[str]:
    """Return requested describe-style statistics in a stable display order."""
    requested: set[str] = set()
    patterns = {
        "min": (r"\bmin\b", r"\bminimum\b", r"\blowest\b", r"\bsmallest\b"),
        "max": (r"\bmax\b", r"\bmaximum\b", r"\bhighest\b", r"\blargest\b"),
        "mean": (r"\bmean\b", r"\baverage\b", r"\bavg\b"),
        "median": (r"\bmedian\b",),
        "std": (r"\bstd\b", r"\bstandard deviation\b"),
        "count": (r"\bcount\b", r"\bnon-null\b", r"\bnon null\b"),
    }
    for stat, stat_patterns in patterns.items():
        if any(re.search(pattern, text) for pattern in stat_patterns):
            requested.add(stat)
    return [
        stat
        for stat in ("min", "max", "mean", "median", "std", "count")
        if stat in requested
    ]


def _numeric_summary_statistics_plan(
    question: str,
    text: str,
    numeric_columns: list[str],
) -> AgentPlan | None:
    requested = _requested_summary_statistics(text)
    if len(requested) < 2:
        return None
    selected = _explicit_numeric_metrics(question, numeric_columns)
    if not selected:
        selected = [
            column
            for column in _mentioned_columns(question, numeric_columns)
            if column in numeric_columns
        ]
    if not selected and len(numeric_columns) == 1:
        selected = [numeric_columns[0]]
    if not selected:
        return AgentPlan(
            tool_name="",
            clarification=(
                "Which numeric column should I summarize? "
                f"Numeric columns: {', '.join(numeric_columns) or 'none'}."
            ),
        )
    return AgentPlan(
        tool_name="calculate_summary_statistics",
        arguments={"columns": selected},
        response_mode="text",
    )


def _date_distinct_count_plan(
    question: str,
    text: str,
    columns: list[str],
    categorical_columns: list[str],
    date_columns: list[str],
    dataframe: pd.DataFrame | None,
) -> AgentPlan | None:
    if not re.search(r"\b(?:unique|distinct|nunique)\b", text):
        return None
    if not any(phrase in text for phrase in ("how many", "number of", "count", "frequency", "show")):
        return None
    period_start, period_end, period_type, period_value = period_bounds_from_text(
        text,
        current_date=_current_date(),
    )
    mentioned_columns = _mentioned_columns(question, columns)
    if not _has_date_intent(text, mentioned_columns, date_columns):
        return None
    date = _preferred_date_column(mentioned_columns, date_columns)
    if not date:
        return AgentPlan(
            tool_name="",
            clarification=f"Which date column should I use? Date columns: {', '.join(date_columns) or 'none'}.",
        )
    target = next(
        (
            column
            for column in mentioned_columns
            if column != date and column not in date_columns
        ),
        None,
    )
    if not target and re.search(r"\borders?\b", text):
        target = next(
            (
                column
                for column in columns
                if "order" in _column_tokens(column)
                and ("id" in _column_tokens(column) or "number" in _column_tokens(column))
            ),
            None,
        )
    if not target and "unique number" in text:
        id_like = [
            column
            for column in columns
            if {"id"} & _column_tokens(column)
            or {"number", "no"} & _column_tokens(column)
        ]
        if len(id_like) == 1:
            target = id_like[0]
    if not target:
        return AgentPlan(
            tool_name="",
            clarification=f"Which column should I count uniquely? Columns: {', '.join(columns)}.",
        )
    category_filter = _detect_categorical_filter(
        question,
        dataframe,
        categorical_columns,
        excluded_columns={target, date},
    )
    filter_column, filter_value = category_filter or (None, None)
    frequency = _time_frequency_from_text(text)
    if period_start is None and (
        _date_breakdown_requested(text)
        or any(word in text for word in ("daily", "weekly", "monthly", "yearly", "quarterly", "month", "year", "week", "day"))
    ):
        return AgentPlan(
            tool_name="calculate_time_trend",
            arguments={
                "date_column": date,
                "value_column": target,
                "aggregation": "nunique",
                "frequency": frequency,
                "start_date": None,
                "end_date": None,
                "filter_column": filter_column,
                "filter_value": filter_value,
            },
            chart_spec=ChartSpec(
                chart_type="line",
                x=date,
                y=target,
                aggregation="nunique",
                title=f"Unique {_display_column_name(target).title()} Count by {frequency.title()}",
                time_grain=frequency,
                time_column=date,
                filter_column=filter_column,
                filter_value=filter_value,
            ),
        )
    if period_start is None:
        return None
    return AgentPlan(
        tool_name="calculate_date_aggregate",
        arguments={
            "date_column": date,
            "value_column": target,
            "aggregation": "nunique",
            "start_date": period_start,
            "end_date": period_end,
            "period_type": period_type,
            "period_value": period_value,
            "filter_column": filter_column,
            "filter_value": filter_value,
        },
        response_mode="text",
    )


def _scalar_extrema_requested(text: str, mentioned: list[str], categorical_columns: list[str]) -> bool:
    if not any(
        re.search(rf"\b{word}\b", text)
        for word in ("highest", "maximum", "max", "largest", "lowest", "minimum", "min", "smallest")
    ):
        return False
    if any(column in mentioned for column in categorical_columns):
        return False
    grouped_phrases = (
        " by ",
        " per ",
        "for each",
        "for all",
        " each ",
        "across",
        "which category",
        "which region",
        "which country",
    )
    if any(phrase in f" {text} " for phrase in grouped_phrases):
        return False
    return any(
        phrase in text
        for phrase in (
            "what is",
            "what's",
            "single",
            "recorded",
            "value",
            "record",
            "entry",
            "observation",
        )
    )


def _share_of_total_requested(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "percentage of",
            "percent of",
            "share of",
            "revenue share",
            "profit share",
            "share",
            "contributes",
            "contribution",
            "percentage distribution",
            "comes from",
        )
    )


def _share_of_total_plan(
    question: str,
    text: str,
    dataframe: pd.DataFrame | None,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> AgentPlan | None:
    if not _share_of_total_requested(text):
        return None
    metric = next(iter(_explicit_numeric_metrics(question, numeric_columns)), None)
    if not metric:
        return None
    mentioned_categories = [
        item for item in _mentioned_columns(question, categorical_columns)
        if item in categorical_columns
    ]
    group = _breakdown_column_from_text(text, categorical_columns)
    top_match = re.search(r"top\s+(\d+|three|five|ten)", text)
    top_words = {"three": 3, "five": 5, "ten": 10}
    ranking_limit = None
    if top_match:
        raw_limit = top_match.group(1)
        ranking_limit = top_words.get(raw_limit, int(raw_limit) if raw_limit.isdigit() else 5)
    category_filter = _detect_categorical_filter(
        question,
        dataframe,
        categorical_columns,
        excluded_columns={group} if group else None,
    )
    focus_column = None
    focus_value = None
    matches = _all_categorical_value_matches(question, dataframe, categorical_columns)
    if "comes from" in text and len(matches) >= 2:
        parent = matches[0]
        focus = matches[-1]
        category_filter = (parent[0], parent[1])
        focus_column, focus_value = focus[0], focus[1]
        group = focus_column
    if not group:
        if mentioned_categories:
            group = next(
                (item for item in mentioned_categories if not category_filter or item != category_filter[0]),
                mentioned_categories[0],
            )
        elif focus_column:
            group = focus_column
    if not group:
        return None
    filter_column, filter_value = category_filter or (None, None)
    return AgentPlan(
        tool_name="group_and_aggregate",
        arguments={
            "group_by": group,
            "secondary_group_by": None,
            "value_column": metric,
            "value_columns": None,
            "aggregation": "sum",
            "limit": ranking_limit,
            "filter_column": filter_column,
            "filter_value": filter_value,
            "include_percentage": True,
            **({"sort_descending": True} if ranking_limit or "largest" in text or "highest" in text else {}),
            **({"focus_column": focus_column, "focus_value": focus_value} if focus_column else {}),
        },
        chart_spec=ChartSpec(
            chart_type="bar",
            x=group,
            y="PercentageOfTotal",
            aggregation=None,
            sort_descending=True,
            limit=ranking_limit,
            title=f"Percentage of Total {_display_column_name(metric)} by {group}",
        ),
    )


def _explicit_chart_type_plan(
    question: str,
    text: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> AgentPlan | None:
    chart_requested = any(
        phrase in text
        for phrase in (
            "scatter",
            "histogram",
            "box plot",
            "boxplot",
            "pie chart",
            "correlation heatmap",
            "heatmap",
        )
    )
    if not chart_requested:
        return None
    mentioned = _mentioned_columns(question, numeric_columns + categorical_columns)
    metrics = _explicit_numeric_metrics(question, numeric_columns)
    if not metrics:
        metrics = [item for item in mentioned if item in numeric_columns]
    categories = [item for item in mentioned if item in categorical_columns]

    if "scatter" in text:
        if len(metrics) < 2:
            return AgentPlan(
                tool_name="",
                clarification=f"A scatter plot needs two numeric columns. Numeric columns: {', '.join(numeric_columns) or 'none'}.",
            )
        x, y = metrics[:2]
        return AgentPlan(
            tool_name="create_scatter_plot",
            arguments={},
            chart_spec=ChartSpec(
                chart_type="scatter",
                x=x,
                y=y,
                title=f"Scatter Plot: {_display_column_name(y).title()} versus {_display_column_name(x).title()}",
            ),
        )

    if "histogram" in text:
        metric = metrics[0] if metrics else (numeric_columns[0] if len(numeric_columns) == 1 else None)
        if not metric:
            return AgentPlan(
                tool_name="",
                clarification=f"A histogram needs one numeric column. Numeric columns: {', '.join(numeric_columns) or 'none'}.",
            )
        return AgentPlan(
            tool_name="create_histogram",
            arguments={},
            chart_spec=ChartSpec(
                chart_type="histogram",
                x=metric,
                title=f"Histogram: {_display_column_name(metric).title()}",
            ),
        )

    if "box plot" in text or "boxplot" in text:
        metric = metrics[0] if metrics else None
        category = categories[0] if categories else None
        if not metric:
            return AgentPlan(
                tool_name="",
                clarification=f"A box plot needs one numeric column. Numeric columns: {', '.join(numeric_columns) or 'none'}.",
            )
        return AgentPlan(
            tool_name="create_box_plot",
            arguments={},
            chart_spec=ChartSpec(
                chart_type="box",
                x=category,
                y=metric,
                color=category,
                title=(
                    f"Box Plot: {_display_column_name(metric).title()}"
                    + (f" by {_display_column_name(category).title()}" if category else "")
                ),
            ),
        )

    if "pie chart" in text or re.search(r"\bpie\b", text):
        metric = metrics[0] if metrics else None
        category = _breakdown_column_from_text(text, categorical_columns) or (categories[0] if categories else None)
        if not metric or not category:
            return AgentPlan(
                tool_name="",
                clarification=(
                    "A pie chart needs one categorical column and one numeric column. "
                    f"Categorical columns: {', '.join(categorical_columns) or 'none'}. "
                    f"Numeric columns: {', '.join(numeric_columns) or 'none'}."
                ),
            )
        return AgentPlan(
            tool_name="create_pie_chart",
            arguments={},
            chart_spec=ChartSpec(
                chart_type="pie",
                x=category,
                y=metric,
                aggregation="sum",
                sort_descending=True,
                title=f"Pie Chart: {_display_column_name(metric).title()} Share by {_display_column_name(category).title()}",
            ),
        )

    if "heatmap" in text and "correlat" in text:
        selected = metrics or [item for item in mentioned if item in numeric_columns]
        if not selected:
            selected = numeric_columns
        if len(selected) < 2:
            return AgentPlan(
                tool_name="",
                clarification=f"A correlation heatmap needs at least two numeric columns. Numeric columns: {', '.join(numeric_columns) or 'none'}.",
            )
        return AgentPlan(
            tool_name="create_heatmap",
            arguments={},
            chart_spec=ChartSpec(
                chart_type="heatmap",
                value_columns=selected,
                title=f"Correlation Heatmap: {', '.join(selected)}",
            ),
        )

    return None


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def _ranking_limit_from_text(text: str) -> int | None:
    match = re.search(r"\b(?:top|bottom)\s+(\d+|" + "|".join(NUMBER_WORDS) + r")\b", text)
    if not match:
        match = re.search(r"\b(\d+|" + "|".join(NUMBER_WORDS) + r")\s+(?:least|most|lowest|highest|largest|smallest|profitable)\b", text)
    if not match:
        return None
    raw = match.group(1)
    return int(raw) if raw.isdigit() else NUMBER_WORDS.get(raw)


def _ranking_direction_from_text(text: str) -> str | None:
    if any(word in text for word in ("bottom", "lowest", "least", "smallest", "worst", "minimum")):
        return "lowest"
    if any(word in text for word in ("top", "highest", "largest", "most", "best", "greatest", "maximum")):
        return "highest"
    return None


def _resolve_metric_alias(text: str, columns: list[str]) -> str | None:
    alias_groups = [
        (("sales", "sale", "revenue", "revenues"), ("Sales", "TotalRevenue")),
        (("profit", "profits", "loss"), ("Profit", "TotalProfit")),
        (("discount", "discounts", "discount rate"), ("Discount",)),
        (("quantity", "quantities", "unit", "units", "units sold"), ("Quantity", "UnitsSold")),
        (("order id", "orders", "order", "unique orders"), ("Order ID", "OrderID")),
        (("customer id", "customers", "customer", "unique customers"), ("Customer ID", "CustomerID")),
        (("product id", "products", "product", "unique products"), ("Product ID", "ProductID")),
    ]
    for aliases, candidates in alias_groups:
        if any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases):
            column = next((candidate for candidate in candidates if candidate in columns), None)
            if column:
                return column
    return None


def _resolve_category_alias(text: str, categorical_columns: list[str]) -> str | None:
    aliases = [
        (("states", "state"), "State"),
        (("cities", "city"), "City"),
        (("regions", "region"), "Region"),
        (("segments", "segment", "customer segments", "customer segment"), "Segment"),
        (("ship modes", "ship mode", "shipping mode"), "Ship Mode"),
        (("sub-categories", "sub-category", "subcategory", "sub category"), "Sub-Category"),
        (("categories", "category"), "Category"),
        (("customers", "customer"), "Customer ID"),
        (("products", "product"), "Product ID"),
    ]
    for names, column in aliases:
        if column in categorical_columns and any(re.search(rf"\b{re.escape(name)}\b", text) for name in names):
            return column
    return None


def _metric_aggregation_for_text(text: str, metric: str) -> str:
    if re.search(r"\b(?:average|mean)\b", text):
        return "mean"
    if re.search(r"\b(?:unique|distinct)\b", text) or metric.lower().endswith(" id") or metric.lower().endswith("id"):
        return "nunique"
    if "count" in text and (metric.lower().endswith(" id") or metric.lower().endswith("id")):
        return "nunique"
    return "sum"


def _entity_group_columns(text: str, columns: list[str], categorical_columns: list[str]) -> list[str] | None:
    if re.search(r"\bcustomer\s+segments?\b", text) and "Segment" in categorical_columns:
        return ["Segment"]
    if re.search(r"\bcustomers?\b", text) and "Customer ID" in columns:
        return [column for column in ("Customer ID", "Customer Name") if column in columns]
    if re.search(r"\bproducts?\b", text) and "Product ID" in columns:
        return [column for column in ("Product ID", "Product Name") if column in columns]
    group = _resolve_category_alias(text, categorical_columns)
    return [group] if group else None


def _requested_group_columns(
    text: str,
    mentioned: list[str],
    columns: list[str],
    categorical_columns: list[str],
) -> list[str] | None:
    entity_columns = _entity_group_columns(text, columns, categorical_columns)
    if entity_columns and any(column.endswith(" ID") or column.endswith("ID") for column in entity_columns):
        return entity_columns
    explicit = [
        column
        for column in categorical_columns
        if _category_column_explicit_in_text(column, text)
    ]
    if explicit:
        return list(dict.fromkeys(explicit))
    if entity_columns:
        return entity_columns
    alias_group = _resolve_category_alias(text, categorical_columns)
    return [alias_group] if alias_group else None


def _category_column_explicit_in_text(column: str, text: str) -> bool:
    lowered = text.lower()
    if column == "Sub-Category":
        return bool(re.search(r"\b(?:sub[-\s]?categor(?:y|ies)|subcategor(?:y|ies))\b", lowered))
    if column == "Category":
        without_subcategory = re.sub(
            r"\b(?:sub[-\s]?categor(?:y|ies)|subcategor(?:y|ies))\b",
            "",
            lowered,
        )
        return bool(re.search(r"\bcategor(?:y|ies)\b", without_subcategory))
    if column == "Segment":
        return bool(re.search(r"\b(?:segments?|customer\s+segments?)\b", lowered))
    if column == "Ship Mode":
        return bool(re.search(r"\bship\s+modes?\b", lowered))
    role = _column_role_pattern(column)
    return bool(re.search(rf"\b{role}\b", lowered))


def _metric_objective_from_text(text: str, column: str, default: str) -> str:
    label = re.escape(_column_phrase(column))
    before_patterns = {
        "lowest": rf"\b(?:lowest|least|minimum|min|smallest)\b[^.?,;]{{0,40}}\b{label}\b",
        "highest": rf"\b(?:highest|top|maximum|max|largest|most|best)\b[^.?,;]{{0,40}}\b{label}\b",
    }
    after_patterns = {
        "lowest": rf"\b{label}\b[^.?,;]{{0,30}}\b(?:lowest|least|minimum|min|smallest)\b",
        "highest": rf"\b{label}\b[^.?,;]{{0,30}}\b(?:highest|top|maximum|max|largest|most|best)\b",
    }
    for objective, pattern in before_patterns.items():
        if re.search(pattern, text):
            return objective
    for objective, pattern in after_patterns.items():
        if re.search(pattern, text):
            return objective
    return default


def _advanced_analytics_plan(
    question: str,
    text: str,
    columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> AgentPlan | None:
    mentioned = _mentioned_columns(question, columns)
    metric = next((item for item in mentioned if item in numeric_columns), None) or _resolve_metric_alias(text, columns)
    direction = _ranking_direction_from_text(text)
    limit = _ranking_limit_from_text(text)

    if "distribution" in text and metric and not any(word in text for word in ("percentage", "percent", "share")):
        return AgentPlan(
            tool_name="analyze_advanced_request",
            arguments={"operation": "distribution", "metric_column": metric},
            chart_spec=ChartSpec(chart_type="histogram", x=metric, title=f"Distribution of {metric}"),
        )

    if "associated" in text or "association" in text or ("higher" in text and "lower" in text and len([m for m in mentioned if m in numeric_columns]) >= 2):
        metrics = [{"column": item} for item in (mentioned if mentioned else []) if item in numeric_columns]
        if len(metrics) < 2 and {"Discount", "Profit"} <= set(columns):
            metrics = [{"column": "Discount"}, {"column": "Profit"}]
        if len(metrics) >= 2:
            return AgentPlan(
                tool_name="analyze_advanced_request",
                arguments={"operation": "relationship", "metrics": metrics[:2]},
                chart_spec=ChartSpec(chart_type="scatter", x=metrics[0]["column"], y=metrics[1]["column"], title=f"{metrics[1]['column']} versus {metrics[0]['column']}"),
            )

    if "percentage of records" in text and "negative" in text and metric:
        return AgentPlan(
            tool_name="analyze_advanced_request",
            arguments={"operation": "negative_record_percentage", "metric_column": metric},
            response_mode="text",
        )

    if ("negative" in text or "loss-making" in text) and metric:
        if "records" in text or "orders by" in text or "loss-making orders" in text:
            group_columns = (
                _requested_group_columns(text, mentioned, columns, categorical_columns)
                or [_resolve_category_alias(text, categorical_columns) or categorical_columns[0]]
            )
            return AgentPlan(
                tool_name="analyze_advanced_request",
                arguments={"operation": "loss_by_group", "group_by": group_columns, "metric_column": metric},
                chart_spec=ChartSpec(chart_type="bar", x=group_columns[-1], y="Total loss", aggregation="sum", sort_descending=False, title=f"Loss by {group_columns[-1]}"),
            )
        group_columns = _requested_group_columns(text, mentioned, columns, categorical_columns)
        if group_columns:
            return AgentPlan(
                tool_name="analyze_advanced_request",
                arguments={"operation": "negative_groups", "group_by": group_columns, "metric_column": metric, "aggregation": "sum", "result_column": f"Total {metric}"},
                chart_spec=ChartSpec(chart_type="bar", x=group_columns[-1], y=f"Total {metric}", aggregation="sum", sort_descending=False, title=f"Negative Total {metric} by {group_columns[-1]}"),
            )

    if "share" in text and ("unique" in text or "orders" in text):
        group_columns = _requested_group_columns(text, mentioned, columns, categorical_columns)
        if not group_columns:
            group = _resolve_category_alias(text, categorical_columns)
            group_columns = [group] if group else None
        entity_metric = _resolve_metric_alias("orders", columns) or metric
        if group_columns and entity_metric:
            return AgentPlan(
                tool_name="analyze_advanced_request",
                arguments={"operation": "share_unique", "group_by": group_columns, "metric_column": entity_metric, "result_column": f"Unique {entity_metric} Count"},
                chart_spec=ChartSpec(chart_type="pie", x=group_columns[-1], y=f"Unique {entity_metric} Count", aggregation="sum", title=f"Share of Unique {entity_metric} by {group_columns[-1]}"),
            )

    if " per " in text or "margin" in text:
        group_columns = _requested_group_columns(text, mentioned, columns, categorical_columns)
        if "margin" in text and {"Profit", "Sales"} <= set(columns):
            group_columns = group_columns or [_resolve_category_alias(text, categorical_columns) or categorical_columns[0]]
            return AgentPlan(
                tool_name="analyze_advanced_request",
                arguments={"operation": "derived_ratio", "group_by": group_columns, "numerator_column": "Profit", "denominator_column": "Sales", "denominator_aggregation": "sum", "result_column": "Profit Margin"},
                chart_spec=ChartSpec(chart_type="bar", x=group_columns[-1], y="Profit Margin", aggregation="sum", title=f"Profit Margin by {group_columns[-1]}"),
            )
        numerator = "Profit" if "profit" in text and "Profit" in columns else "Sales" if "Sales" in columns else metric
        denominator = None
        denominator_agg = "sum"
        if re.search(r"\b(?:unit|quantity)\b", text):
            denominator = "Quantity" if "Quantity" in columns else None
            denominator_agg = "sum"
        elif "order" in text:
            denominator = "Order ID" if "Order ID" in columns else None
            denominator_agg = "nunique"
        elif "customer" in text:
            denominator = "Customer ID" if "Customer ID" in columns else None
            denominator_agg = "nunique"
        if numerator and denominator and group_columns:
            alias = f"{numerator} per {'Order' if denominator == 'Order ID' else 'Customer' if denominator == 'Customer ID' else denominator}"
            return AgentPlan(
                tool_name="analyze_advanced_request",
                arguments={"operation": "derived_ratio", "group_by": group_columns, "numerator_column": numerator, "denominator_column": denominator, "denominator_aggregation": denominator_agg, "result_column": alias, "rank_by": alias, "direction": direction or "highest", "limit": limit},
                chart_spec=ChartSpec(chart_type="bar", x=group_columns[-1], y=alias, aggregation="sum", sort_descending=(direction != "lowest"), title=f"{alias} by {group_columns[-1]}"),
            )

    if "minus" in text and len([item for item in mentioned if item in numeric_columns]) >= 2:
        metrics_found = [item for item in mentioned if item in numeric_columns]
        group_columns = _requested_group_columns(text, mentioned, columns, categorical_columns)
        if not group_columns:
            alias_group = _resolve_category_alias(text, categorical_columns)
            group_columns = [alias_group] if alias_group else None
        if group_columns:
            alias = f"{metrics_found[0]} minus {metrics_found[1]}"
            return AgentPlan(
                tool_name="analyze_advanced_request",
                arguments={"operation": "derived_difference", "group_by": group_columns, "numerator_column": metrics_found[0], "denominator_column": metrics_found[1], "result_column": alias},
                chart_spec=ChartSpec(chart_type="bar", x=group_columns[-1], y=alias, aggregation="sum", title=f"{alias} by {group_columns[-1]}"),
            )

    if direction and ("for each" in text or "in each" in text):
        primary = None
        secondary = None
        if re.search(r"\b(?:for|in|within)\s+each\s+customer\s+segments?\b", text) and "Segment" in categorical_columns:
            primary = "Segment"
        for column in categorical_columns:
            if primary:
                break
            role = _column_role_pattern(column)
            if re.search(rf"\b(?:for|in|within)\s+each\s+{role}\b", text):
                primary = column
                break
        secondary_candidates = [item for item in mentioned if item in categorical_columns and item != primary]
        if not secondary_candidates:
            secondary_alias = _resolve_category_alias(text, categorical_columns)
            if secondary_alias and secondary_alias != primary:
                secondary_candidates = [secondary_alias]
        secondary = secondary_candidates[0] if secondary_candidates else None
        if primary and secondary and metric:
            aggregation = _metric_aggregation_for_text(text, metric)
            extremum = "min" if direction == "lowest" else "max"
            return AgentPlan(
                tool_name="calculate_grouped_extrema",
                arguments={"primary_group_column": primary, "secondary_group_column": secondary, "metric_column": metric, "aggregation": aggregation, "extremum": extremum},
                chart_spec=ChartSpec(chart_type="grouped_extrema_bar", x=primary, y=metric, color=secondary, aggregation=aggregation, sort_descending=False, title=f"{'Highest' if extremum == 'max' else 'Lowest'} {metric} {secondary} by {primary}"),
            )

    if (
        direction
        and (" and " in text)
        and len([item for item in mentioned if item in numeric_columns]) >= 2
        and not ("compare" in text and limit)
    ):
        group_columns = _requested_group_columns(text, mentioned, columns, categorical_columns)
        metrics_found = [item for item in mentioned if item in numeric_columns]
        if group_columns:
            metric_specs = []
            for column in metrics_found[:2]:
                objective = _metric_objective_from_text(text, column, direction or "highest")
                metric_specs.append({"column": column, "aggregation": "sum", "direction": objective, "alias": f"Total {column}"})
            return AgentPlan(
                tool_name="analyze_advanced_request",
                arguments={"operation": "multi_metric_extrema", "group_by": group_columns, "metrics": metric_specs},
                chart_spec=ChartSpec(chart_type="bar", x=group_columns[-1], y="Total " + metrics_found[0], value_columns=[f"Total {m}" for m in metrics_found[:2]], aggregation="sum", title=f"{' and '.join(metrics_found[:2])} by {group_columns[-1]}"),
            )

    metrics_found = [item for item in mentioned if item in numeric_columns]
    if (
        ("compare" in text or len(metrics_found) >= 2)
        and limit
        and len(metrics_found) >= 2
    ):
        group_columns = _requested_group_columns(text, mentioned, columns, categorical_columns)
        if group_columns:
            metrics_specs = [
                {"column": column, "aggregation": "sum", "alias": f"Total {column}"}
                for column in metrics_found[:3]
            ]
            return AgentPlan(
                tool_name="analyze_advanced_request",
                arguments={
                    "operation": "grouped_aggregation",
                    "group_by": group_columns,
                    "metrics": metrics_specs,
                    "rank_by": metrics_specs[0]["alias"],
                    "limit": limit,
                },
                chart_spec=ChartSpec(
                    chart_type="bar",
                    x=group_columns[-1],
                    y=metrics_specs[0]["alias"],
                    value_columns=[spec["alias"] for spec in metrics_specs],
                    aggregation="sum",
                    title=f"Top {limit} {group_columns[-1]} by {metrics_found[0]}",
                ),
            )

    if (direction or limit) and metric:
        group_columns = _requested_group_columns(text, mentioned, columns, categorical_columns)
        if group_columns:
            aggregation = _metric_aggregation_for_text(text, metric)
            return AgentPlan(
                tool_name="analyze_advanced_request",
                arguments={"operation": "ranking", "group_by": group_columns, "metric_column": metric, "aggregation": aggregation, "direction": direction or "highest", "limit": limit, "result_column": f"{aggregation.title()} {metric}"},
                chart_spec=ChartSpec(chart_type="bar", x=group_columns[-1], y=f"{aggregation.title()} {metric}", aggregation="sum", sort_descending=(direction != "lowest"), limit=limit, title=f"{direction or 'Top'} {group_columns[-1]} by {metric}"),
            )

    if ("compare" in text or " by " in text or " per " in text or "across" in text) and metric:
        group_columns = _requested_group_columns(text, mentioned, columns, categorical_columns)
        metrics_found = [item for item in mentioned if item in numeric_columns]
        if "discount" in text and "profit" in text and "Discount" in columns and "Profit" in columns:
            metrics_specs = [{"column": "Profit", "aggregation": "sum", "alias": "Total Profit"}, {"column": "Discount", "aggregation": "mean", "alias": "Average Discount"}]
        elif "sales" in text and "discount" in text and "Sales" in columns and "Discount" in columns:
            metrics_specs = [{"column": "Sales", "aggregation": "sum", "alias": "Total Sales"}, {"column": "Discount", "aggregation": "mean", "alias": "Average Discount"}]
        elif metrics_found:
            agg = "mean" if "average" in text or "mean" in text else "sum"
            metrics_specs = [{"column": metrics_found[0], "aggregation": agg, "alias": f"{agg.title()} {metrics_found[0]}"}]
        else:
            metrics_specs = []
        if group_columns and metrics_specs:
            different_units = any(spec["column"] == "Discount" for spec in metrics_specs) and any(spec["column"] in {"Sales", "Profit"} for spec in metrics_specs)
            chart = ChartSpec(
                chart_type="dual_axis" if different_units and len(metrics_specs) >= 2 else "grouped_bar" if len(group_columns) > 1 else "bar",
                x=group_columns[0],
                y=metrics_specs[0]["alias"],
                secondary_y=metrics_specs[1]["alias"] if different_units and len(metrics_specs) >= 2 else None,
                color=group_columns[1] if len(group_columns) > 1 and not different_units else None,
                value_columns=[spec["alias"] for spec in metrics_specs] if len(metrics_specs) > 1 and not different_units else [],
                aggregation="sum",
                title=f"Metrics by {' and '.join(group_columns)}",
            )
            return AgentPlan(
                tool_name="analyze_advanced_request",
                arguments={"operation": "grouped_aggregation", "group_by": group_columns, "metrics": metrics_specs, "rank_by": metrics_specs[0]["alias"] if limit else None, "limit": limit},
                chart_spec=chart,
            )

    return None


def _period_over_period_requested(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "compared with the previous",
            "compare yearly",
            "previous month",
            "previous year",
            "percentage change",
            "growth",
            "month-over-month",
            "year-over-year",
            "period-over-period",
            "monthly revenue change",
            "monthly profit change",
            "yearly profit growth",
            "monthly revenue change",
        )
    ) or bool(re.search(r"\b(?:monthly|yearly|weekly|quarterly)\s+\w+\s+change\b", text))


def _period_over_period_plan(
    question: str,
    text: str,
    dataframe: pd.DataFrame | None,
    numeric_columns: list[str],
    categorical_columns: list[str],
    date_columns: list[str],
) -> AgentPlan | None:
    if not _period_over_period_requested(text):
        return None
    metric = next(iter(_explicit_numeric_metrics(question, numeric_columns)), None)
    if not metric:
        return None
    date = _preferred_date_column(_mentioned_columns(question, date_columns), date_columns)
    if not date:
        return AgentPlan(
            tool_name="",
            clarification=(
                "A period-over-period analysis requires a valid date column. "
                f"Date columns: {', '.join(date_columns) or 'none'}."
            ),
        )
    category_filter = _detect_categorical_filter(question, dataframe, categorical_columns)
    filter_column, filter_value = category_filter or (None, None)
    period_start, period_end, _, _ = period_bounds_from_text(text, current_date=_current_date())
    frequency = _time_frequency_from_text(text)
    aggregation = "mean" if any(word in text for word in ("average", "mean")) else "sum"
    comparison_basis = "previous_year" if "previous year" in text or "year-over-year" in text or "yoy" in text else "previous_period"
    return AgentPlan(
        tool_name="calculate_period_over_period",
        arguments={
            "date_column": date,
            "value_column": metric,
            "aggregation": aggregation,
            "frequency": frequency,
            "comparison_basis": comparison_basis,
            "start_date": period_start,
            "end_date": period_end,
            "filter_column": filter_column,
            "filter_value": filter_value,
        },
        chart_spec=ChartSpec(
            chart_type="bar",
            x=date,
            y="PercentageChange",
            aggregation=None,
            title=f"{_display_column_name(metric)} {frequency.title()} Change",
            time_grain=frequency,
            comparison_basis="previous_period",
        ),
    )


def _followup_time_trend_plan(
    question: str,
    text: str,
    dataframe: pd.DataFrame | None,
    previous: list[dict[str, Any]],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> AgentPlan | None:
    followup_requested = any(
        phrase in text
        for phrase in ("now", "instead", "only", "just", "make it", "use ", "compare ")
    )
    if not followup_requested:
        return None
    previous_plan = next(
        (
            item for item in reversed(previous)
            if item.get("tool_name") == "calculate_time_trend"
        ),
        None,
    )
    if previous_plan is None:
        if _period_over_period_requested(text):
            return None
        if previous:
            return None
        if not (text.startswith("now ") or text.startswith("make it") or text.startswith("same but")):
            return None
        return AgentPlan(
            tool_name="",
            clarification="I need a previous time-series analysis to update. Please specify the metric and scope.",
        )
    args = dict(previous_plan.get("arguments", {}))
    metrics = _explicit_numeric_metrics(question, numeric_columns)
    if len(metrics) >= 2 or (metrics and "compare" in text):
        args["value_column"] = metrics[0]
        args["value_columns"] = metrics[:2]
    elif metrics:
        args["value_column"] = metrics[0]
        args.pop("value_columns", None)
    if any(word in text for word in ("monthly", "month")):
        args["frequency"] = "month"
    elif any(word in text for word in ("yearly", "year")) and "2021" not in text and "2022" not in text and "2023" not in text:
        args["frequency"] = "year"
    elif any(word in text for word in ("weekly", "week")):
        args["frequency"] = "week"
    category_filter = _detect_categorical_filter(question, dataframe, categorical_columns)
    if category_filter:
        args["filter_column"], args["filter_value"] = category_filter
    period_start, period_end, _, _ = period_bounds_from_text(text, current_date=_current_date())
    if period_start is not None:
        args["start_date"] = period_start
        args["end_date"] = period_end
    value_columns = args.get("value_columns") or []
    chart_type = "line"
    chart_spec = ChartSpec(
        chart_type=chart_type,
        x=args["date_column"],
        y="Value" if value_columns else args["value_column"],
        value_columns=value_columns,
        color="Metric" if value_columns else None,
        aggregation=args.get("aggregation", "sum"),
        title=(
            f"{', '.join(value_columns) if value_columns else args['value_column']} "
            f"by {str(args.get('frequency', 'month')).title()}"
        ),
        time_grain=args.get("frequency", "month"),
        time_column=args["date_column"],
        date_range_start=args.get("start_date"),
        date_range_end=args.get("end_date"),
        filter_column=args.get("filter_column"),
        filter_value=args.get("filter_value"),
    )
    return AgentPlan(
        tool_name="calculate_time_trend",
        arguments=args,
        chart_spec=chart_spec,
    )


def _unresolved_filter_value_text(
    text: str,
    columns: list[str],
) -> str | None:
    match = re.search(r"\b(?:for|in|within)\s+(?P<value>[a-z][a-z\s']{1,40})\b", text)
    if not match:
        return None
    value = match.group("value").strip(" .'")
    if not value or value.startswith(
        ("all ", "each ", "every ", "year ", "month ", "week ", "quarter ")
    ):
        return None
    blocked = {
        "a valid date column",
        "valid date column",
        "a column",
        "the column",
        "each region",
        "each country",
        "the uploaded excel file",
        "uploaded excel file",
        "the uploaded file",
        "uploaded file",
        "the dataset",
        "dataset",
    }
    if value in blocked:
        return None
    called_match = re.search(r"\b(?:a|an|the)?\s*\w+\s+called\s+(?P<value>.+)$", value)
    if called_match:
        value = called_match.group("value").strip(" .'")
    normalized_value = _normalize(value)
    if any(normalized_value == _normalize(column) for column in columns):
        return None
    return value


def _unknown_value_count_column_text(text: str, columns: list[str]) -> str | None:
    patterns = (
        r"value\s+counts?\s+(?:for|of)\s+(?P<column>.+?)$",
        r"counts?\s+(?:for|of)\s+(?P<column>.+?)$",
        r"frequency\s+(?:for|of)\s+(?P<column>.+?)$",
        r"distribution\s+(?:for|of)\s+(?P<column>.+?)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        raw = match.group("column").strip(" .?")
        raw = re.sub(r"\b(?:column|field|values?)\b", " ", raw).strip()
        if not raw:
            continue
        if _mentioned_columns(raw, columns):
            return None
        return raw
    return None


def _unknown_requested_date_column_text(text: str, date_columns: list[str]) -> str | None:
    patterns = (
        r"\busing\s+(?P<column>[a-z][a-z0-9_\-\s]{0,40}date)\b",
        r"\bwith\s+(?P<column>[a-z][a-z0-9_\-\s]{0,40}date)\b",
        r"\bby\s+(?P<column>[a-z][a-z0-9_\-\s]{0,40}date)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        raw = match.group("column").strip(" .?")
        if raw in {"a valid date", "valid date"}:
            return "valid date column"
        if any(_normalize(raw) == _normalize(column) for column in date_columns):
            return None
        return raw
    if "valid date column" in text and _has_date_intent(text, [], date_columns):
        return "valid date column"
    return None


def _invalid_non_numeric_average_plan(
    question: str,
    text: str,
    columns: list[str],
    numeric_columns: list[str],
) -> AgentPlan | None:
    if not any(word in text for word in ("average", "mean")):
        return None
    if any(word in text for word in ("above", "below")):
        return None
    if _explicit_numeric_metrics(question, numeric_columns):
        return None
    mentioned = _mentioned_columns(question, columns)
    non_numeric = next((column for column in mentioned if column not in numeric_columns), None)
    if not non_numeric:
        return None
    reason = "a categorical/text column"
    if "id" in _column_tokens(non_numeric):
        reason = "an identifier column"
    return AgentPlan(
        tool_name="",
        clarification=(
            f"I cannot calculate an average for {non_numeric} because it is {reason}, "
            "not a numeric measure. No calculation was performed."
        ),
    )


def _date_clarification_plan(
    question: str,
    metric: str | None,
    date_columns: list[str],
    frequency: str = "month",
) -> AgentPlan:
    options = list(date_columns)
    metric_text = metric or "the metric"
    suggested = [
        f"Using {date}, {question.strip().rstrip('.')}"
        for date in options
    ]
    return AgentPlan(
        tool_name="",
        clarification=(
            f"This dataset contains multiple date columns: {', '.join(options)}. "
            f"Which one should be used for the {frequency}ly {metric_text} analysis?"
        ),
        arguments={
            "clarification_type": "ambiguous_date_column",
            "options": options,
            "original_query": question,
            "metric": metric,
            "suggested_queries": suggested,
        },
    )


def _ambiguous_date_plan(
    question: str,
    text: str,
    mentioned: list[str],
    numeric_columns: list[str],
    date_columns: list[str],
) -> AgentPlan | None:
    if len(date_columns) < 2:
        return None
    if any(item in date_columns for item in mentioned):
        return None
    if not _has_date_intent(text, mentioned, date_columns):
        return None
    metric = next(iter(_explicit_numeric_metrics(question, numeric_columns)), None)
    if not metric:
        metric = next((item for item in mentioned if item in numeric_columns), None)
    if not metric and len(numeric_columns) == 1:
        metric = numeric_columns[0]
    if not metric:
        return None
    return _date_clarification_plan(
        question,
        metric,
        date_columns,
        frequency=_time_frequency_from_text(text),
    )


def _ambiguous_filter_value_plan(
    question: str,
    dataframe: pd.DataFrame | None,
    categorical_columns: list[str],
) -> AgentPlan | None:
    if dataframe is None:
        return None
    explicit = _explicit_categorical_filter(question, dataframe, categorical_columns)
    if explicit:
        return None
    matches = _all_categorical_value_matches(question, dataframe, categorical_columns)
    grouped: dict[str, list[str]] = {}
    for column, value, _position in matches:
        grouped.setdefault(str(value).casefold(), []).append(column)
    for value_key, columns in grouped.items():
        unique_columns = list(dict.fromkeys(columns))
        if len(unique_columns) < 2:
            continue
        display_value = next(str(value) for column, value, _ in matches if str(value).casefold() == value_key)
        options = [f"{column} = {display_value}" for column in unique_columns]
        return AgentPlan(
            tool_name="",
            clarification=(
                f'"{display_value}" appears in multiple columns. Which filter do you mean?'
            ),
            arguments={
                "clarification_type": "ambiguous_filter_value",
                "options": options,
                "original_query": question,
                "suggested_queries": [
                    f"{question.strip().rstrip('.')} where {option}"
                    for option in options
                ],
            },
        )
    return None


def _explicit_numeric_metrics(question: str, numeric_columns: list[str]) -> list[str]:
    """Return numeric columns with a meaningful metric token in the question."""
    question_tokens = _question_tokens(question)
    normalized_question = _normalize(question)
    lowered = question.lower()
    mentioned = _mentioned_columns(question, numeric_columns)
    generic_tokens = {"total", "value", "amount", "metric", "number"}
    explicit: list[tuple[int, str, bool]] = []
    for column in mentioned:
        significant = _column_tokens(column) - generic_tokens
        if not significant:
            significant = _column_tokens(column)
        compact_aliases = {
            alias for alias in _column_aliases(column) if alias not in generic_tokens
        }
        alias_positions = [
            normalized_question.find(alias)
            for alias in compact_aliases
            if alias in normalized_question
        ]
        token_positions = [
            match.start()
            for token in significant
            for match in re.finditer(rf"\b{re.escape(token)}\b", lowered)
        ]
        if alias_positions:
            explicit.append((min(alias_positions), column, True))
        elif significant.issubset(question_tokens) and token_positions:
            explicit.append((min(token_positions), column, False))
    filtered: list[tuple[int, str, bool]] = []
    for position, column, alias_matched in explicit:
        significant = _column_tokens(column) - generic_tokens
        if not significant:
            significant = _column_tokens(column)
        broader_than_requested = False
        for _, other_column, other_alias_matched in explicit:
            if other_column == column:
                continue
            other_significant = _column_tokens(other_column) - generic_tokens
            if not other_significant:
                other_significant = _column_tokens(other_column)
            if (
                not alias_matched
                and other_alias_matched
                and significant < other_significant
            ):
                broader_than_requested = True
                break
        if not broader_than_requested:
            filtered.append((position, column, alias_matched))
    return [column for position, column, _ in sorted(filtered, key=lambda item: item[0])]


def _directed_metric_pair(
    question: str,
    numeric_columns: list[str],
) -> tuple[str, str] | None:
    """Return the metrics described as high and low, in that order."""
    high_words = ("high", "higher", "many", "most", "large", "larger", "above")
    low_words = ("low", "lower", "few", "small", "smaller", "below")
    clauses = re.split(r"\b(?:but|while|whereas|although|and)\b|[,;]", question.lower())
    high_metric = None
    low_metric = None

    for clause in clauses:
        clause_metrics = _mentioned_columns(clause, numeric_columns)
        if not clause_metrics:
            continue
        if any(re.search(rf"\b{word}\b", clause) for word in high_words):
            high_metric = high_metric or clause_metrics[0]
        if any(re.search(rf"\b{word}\b", clause) for word in low_words):
            low_metric = low_metric or clause_metrics[0]

    if high_metric and low_metric and high_metric != low_metric:
        return high_metric, low_metric

    mentioned = _mentioned_columns(question, numeric_columns)
    text = question.lower()
    high_positions = [text.find(word) for word in high_words if word in text]
    low_positions = [text.find(word) for word in low_words if word in text]
    if len(mentioned) >= 2 and high_positions and low_positions:
        if min(high_positions) < min(low_positions):
            return mentioned[0], mentioned[1]
        return mentioned[1], mentioned[0]
    return None


def _benchmark_group_column(
    question: str,
    categorical_columns: list[str],
) -> str | None:
    """Identify the parent category that defines a local benchmark."""
    text = question.lower()
    for column in categorical_columns:
        stems = _column_tokens(column)
        if not stems:
            continue
        stem_pattern = "|".join(re.escape(stem) for stem in stems)
        patterns = (
            rf"\b(?:average|mean|median)\s+(?:\w+\s+){{0,2}}(?:{stem_pattern})\w*\b",
            rf"\b(?:{stem_pattern})\w*\s+(?:average|mean|median)\b",
            rf"\b(?:within|inside|for)\s+(?:each|every)\s+(?:{stem_pattern})\w*\b",
        )
        if any(re.search(pattern, text) for pattern in patterns):
            return column
    return None


def _referenced_filter_context(
    text: str,
    previous: list[dict[str, Any]],
    categorical_columns: list[str],
) -> tuple[str, Any] | None:
    """Resolve phrases such as "this region" from recent analysis scope."""
    referenced_columns = [
        column
        for column in categorical_columns
        if any(
            phrase in text
            for phrase in (
                f"this {_display_column_name(column)}",
                f"that {_display_column_name(column)}",
                f"same {_display_column_name(column)}",
                f"the same {_display_column_name(column)}",
            )
        )
    ]
    for column in referenced_columns:
        for item in reversed(previous):
            arguments = item.get("arguments", {})
            if (
                arguments.get("filter_column") == column
                and arguments.get("filter_value") is not None
            ):
                return column, arguments["filter_value"]
            if (
                arguments.get("category_column") == column
                and arguments.get("category_value") is not None
            ):
                return column, arguments["category_value"]
    return None


def _detect_categorical_filter(
    question: str,
    dataframe: pd.DataFrame | None,
    categorical_columns: list[str],
    excluded_column: str | None = None,
    excluded_columns: set[str] | None = None,
) -> tuple[str, Any] | None:
    if dataframe is None:
        return None
    excluded = set(excluded_columns or ())
    if excluded_column:
        excluded.add(excluded_column)
    explicit = _explicit_categorical_filter(
        question,
        dataframe,
        categorical_columns,
        excluded_columns=excluded,
    )
    if explicit:
        return explicit
    normalized_question = _normalize(question)
    matches: list[tuple[int, str, Any]] = []
    for column in categorical_columns:
        if column in excluded or column not in dataframe.columns:
            continue
        for value in dataframe[column].dropna().unique():
            aliases = _category_value_aliases(value)
            if aliases and _category_value_matches_query(value, question, normalized_question):
                matches.append((max(len(alias) for alias in aliases), column, value))
    if not matches:
        return None
    _, column, value = max(matches, key=lambda item: item[0])
    return column, value


def _detect_categorical_filters(
    question: str,
    dataframe: pd.DataFrame | None,
    categorical_columns: list[str],
    excluded_columns: set[str] | None = None,
) -> list[tuple[str, Any]]:
    if dataframe is None:
        return []
    excluded = set(excluded_columns or ())
    matches: list[tuple[int, int, str, Any, bool]] = []
    for column in categorical_columns:
        if column in excluded or column not in dataframe.columns:
            continue
        explicit = _explicit_categorical_filter(
            question,
            dataframe,
            [column],
        )
        if explicit:
            position = question.lower().find(str(explicit[1]).lower())
            matches.append((position if position >= 0 else len(question), 10_000, *explicit, True))
            continue
        normalized_question = _normalize(question)
        for value in dataframe[column].dropna().unique():
            aliases = _category_value_aliases(value)
            if aliases and _category_value_matches_query(value, question, normalized_question):
                position = question.lower().find(str(value).lower())
                matches.append(
                    (
                        position if position >= 0 else len(question),
                        max(len(alias) for alias in aliases),
                        column,
                        value,
                        False,
                    )
                )
    selected: dict[str, tuple[int, int, Any, bool]] = {}
    for position, score, column, value, explicit in sorted(matches, key=lambda item: (item[0], -item[1])):
        current = selected.get(column)
        if current is None or score > current[1]:
            selected[column] = (position, score, value, explicit)
    filtered: dict[str, tuple[int, int, Any, bool]] = {}
    selected_items = list(selected.items())
    for column, item in selected_items:
        position, score, value, explicit = item
        normalized_value = _normalize(str(value))
        if not explicit and any(
            other_column != column
            and len(_normalize(str(other_item[2]))) > len(normalized_value)
            and normalized_value
            and normalized_value in _normalize(str(other_item[2]))
            for other_column, other_item in selected_items
        ):
            continue
        filtered[column] = item
    return [
        (column, value)
        for column, (position, _score, value, _explicit) in sorted(
            filtered.items(), key=lambda item: item[1][0]
        )
    ]


def _mentioned_category_values(
    question: str,
    dataframe: pd.DataFrame | None,
    categorical_columns: list[str],
) -> tuple[str, list[Any]] | None:
    """Find two or more explicit values belonging to one categorical column."""
    if dataframe is None:
        return None
    normalized_question = _normalize(question)
    for column in categorical_columns:
        values = []
        for value in dataframe[column].dropna().unique():
            if _category_value_matches_query(value, question, normalized_question):
                values.append(value)
        if len(values) >= 2:
            return column, values
    return None


def _all_categorical_value_matches(
    question: str,
    dataframe: pd.DataFrame | None,
    categorical_columns: list[str],
) -> list[tuple[str, Any, int]]:
    if dataframe is None:
        return []
    normalized_question = _normalize(question)
    matches: list[tuple[str, Any, int]] = []
    for column in categorical_columns:
        if column not in dataframe.columns:
            continue
        for value in dataframe[column].dropna().unique():
            if _category_value_matches_query(value, question, normalized_question):
                position = question.lower().find(str(value).lower())
                matches.append((column, value, position if position >= 0 else len(question)))
    matches.sort(key=lambda item: item[2])
    return matches


def _safe_code(plan: AgentPlan) -> str:
    args = plan.arguments
    templates = {
        "inspect_dataset": "df.shape",
        "get_column_information": (
            f"df[{args.get('column')!r}].dtype"
            if args.get("column")
            else "df.dtypes"
        ),
        "profile_column": f"profile one column: df[{args.get('column_name')!r}]",
        "calculate_summary_statistics": (
            f"df[{args.get('columns')!r}].describe().T"
            if args.get("columns")
            else "df.select_dtypes(include='number').describe().T"
        ),
        "calculate_average_numeric_columns": "df.select_dtypes(include='number').mean()",
        "analyze_missing_values": "df.isna().sum()",
        "analyze_duplicates": "df.duplicated().sum()",
        "calculate_correlation": f"df[[{args.get('first_column')!r}, {args.get('second_column')!r}]].corr()",
        "detect_outliers": f"# IQR outlier detection for {args.get('column') or 'all numeric columns'}",
        "group_and_aggregate": (
            (
                f"df.loc[df[{args.get('filter_column')!r}].astype('string').str.casefold() == "
                f"{str(args.get('filter_value')).casefold()!r}]"
                if args.get("filter_column")
                else "df"
            )
            + (
                f".loc[pd.to_datetime(df[{args.get('date_column')!r}], errors='coerce').between"
                f"({args.get('start_date')!r}, {args.get('end_date')!r})]"
                if args.get("date_column")
                else ""
            )
            + (
                f".groupby([{args.get('group_by')!r}, "
                f"{args.get('secondary_group_by')!r}])"
                if args.get("secondary_group_by")
                else f".groupby({args.get('group_by')!r})"
            )
            + (
                f"[{args.get('value_columns')!r}]"
                if args.get("value_columns")
                else f"[{args.get('value_column')!r}]"
            )
            + f".agg({args.get('aggregation')!r})"
            + (
                f".sort_values(by={args['value_columns'][0]!r}, ascending=False)"
                if args.get("value_columns")
                else ".sort_values(ascending=False)"
            )
            + (f".head({args.get('limit')})" if args.get("limit") else "")
        ),
        "calculate_grouped_extrema": (
            f"aggregated = df.groupby([{args.get('primary_group_column')!r}, "
            f"{args.get('secondary_group_column')!r}])[{args.get('metric_column')!r}]"
            f".agg({args.get('aggregation', 'sum')!r}).reset_index()\n"
            f"# Select tied {'maximum' if args.get('extremum', 'max') == 'max' else 'minimum'} "
            f"{args.get('secondary_group_column')!r} rows inside each "
            f"{args.get('primary_group_column')!r}"
        ),
        "compare_grouped_to_benchmark": (
            (
                f"working = df.loc[df[{args.get('filter_column')!r}].astype('string').str.casefold() == "
                f"{str(args.get('filter_value')).casefold()!r}]\n"
                if args.get("filter_column")
                else "working = df\n"
            )
            + f"totals = working.groupby("
            f"{[item for item in (args.get('benchmark_group_by'), args.get('category_column')) if item]!r}"
            f")[{args.get('value_column')!r}].agg({args.get('aggregation', 'sum')!r})\n"
            f"# Compare each total with the {args.get('benchmark', 'mean')} "
            f"{'within its parent group' if args.get('benchmark_group_by') else 'across categories'}"
        ),
        "sort_and_limit": (
            f"df.sort_values({args.get('sort_by')!r}, ascending={args.get('ascending', False)})"
            f".head({args.get('limit', 5)})"
        ),
        "calculate_time_trend": (
            "working = df\n"
            + (
                f"working = working.loc[working[{args.get('filter_column')!r}].astype('string').str.casefold() == "
                f"{str(args.get('filter_value')).casefold()!r}]\n"
                if args.get("filter_column")
                else ""
            )
            + (
                f"dates = pd.to_datetime(working[{args.get('date_column')!r}], errors='coerce')\n"
                f"working = working.loc[dates.between({args.get('start_date')!r}, {args.get('end_date')!r})]\n"
                if args.get("start_date") is not None or args.get("end_date") is not None
                else f"dates = pd.to_datetime(working[{args.get('date_column')!r}], errors='coerce')\n"
            )
            + f"working.assign(period=dates.dt.to_period({args.get('frequency', 'month')!r}))"
            f".groupby({[item for item in ('period', args.get('breakdown_column')) if item]!r})"
            f"[{args.get('value_columns') or args.get('value_column')!r}]"
            f".agg({args.get('aggregation', 'sum')!r})"
        ),
        "calculate_date_aggregate": (
            f"df.loc[pd.to_datetime(df[{args.get('date_column')!r}], errors='coerce').between"
            f"({args.get('start_date')!r}, {args.get('end_date')!r})]"
            f"[{args.get('value_column')!r}].agg({args.get('aggregation', 'sum')!r})"
        ),
        "calculate_value_counts": f"df[{args.get('column')!r}].value_counts().head({args.get('limit', 20)})",
        "analyze_categorical_value_counts": (
            f"df.groupby([{args.get('primary_group_column')!r}, {args.get('counted_column')!r}], dropna=False).size()"
            if args.get("primary_group_column")
            else f"df[{args.get('counted_column')!r}].value_counts(dropna={not args.get('include_missing', False)!r})"
        ),
        "compare_category_values": (
            f"df[df[{args.get('category_column')!r}].isin("
            f"{[args.get('first_value'), args.get('second_value')]!r})]"
            f".groupby({args.get('category_column')!r})"
            f"[{args.get('value_column')!r}].agg({args.get('aggregation', 'sum')!r})"
        ),
        "calculate_filtered_aggregate": (
            f"df.loc[df[{args.get('category_column')!r}].astype('string').str.casefold() == "
            f"{str(args.get('category_value')).casefold()!r}, "
            f"{args.get('value_columns') or args.get('value_column')!r}].agg({args.get('aggregation', 'sum')!r})"
        ),
        "calculate_scalar_aggregate": (
            f"df[{args.get('value_column')!r}].agg("
            f"{args.get('aggregation', 'sum')!r})"
        ),
        "calculate_multi_scalar_aggregate": (
            f"df[{args.get('value_columns')!r}].agg("
            f"{args.get('aggregation', 'sum')!r})"
        ),
        "list_distinct_values": (
            f"df.loc[df[{args.get('filter_column')!r}].astype('string').str.casefold() == "
            f"{str(args.get('filter_value')).casefold()!r}, "
            f"{args.get('target_column')!r}].dropna().unique()"
            if args.get("filter_column")
            else f"df[{args.get('target_column')!r}].dropna().unique()"
        ),
        "count_distinct_values": (f"df[{args.get('column')!r}].nunique(dropna=True)"),
        "analyze_high_volume_low_outcome": (
            f"df.groupby({args.get('category_column')!r})"
            f"[[{args.get('volume_column')!r}, {args.get('outcome_column')!r}]]"
            f".agg({args.get('aggregation', 'sum')!r})"
        ),
    }
    return templates.get(
        plan.tool_name, "# Executed through an approved validated tool"
    )


def _requested_summary_lines(text: str, default: int = 2) -> int:
    match = re.search(r"\b(\d+|one|two|three|four|five)\s+lines?\b", text)
    if not match:
        return default
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    raw = match.group(1)
    return max(1, min(5, words.get(raw, int(raw) if raw.isdigit() else default)))


def _chat_narrative_from_result(
    plan: AgentPlan,
    result: ToolResult | None,
) -> ChatNarrativeResponse | None:
    """Build structured chat prose for complex verified result types."""
    if (
        not result
        or plan.tool_name != "calculate_time_trend"
        or not isinstance(result.data, list)
        or not result.data
    ):
        return None
    metrics = plan.arguments.get("value_columns") or []
    if len(metrics) < 2:
        return None
    date_column = plan.arguments.get("date_column")
    if not date_column:
        return None
    rows = [
        row
        for row in result.data
        if isinstance(row, dict)
        and date_column in row
        and all(isinstance(row.get(metric), (int, float)) for metric in metrics)
    ]
    if not rows:
        return None
    aggregation = plan.arguments.get("aggregation", "sum")
    frequency = plan.arguments.get("frequency", "period")
    filter_column = plan.arguments.get("filter_column")
    filter_value = plan.arguments.get("filter_value")
    filter_text = f" where {filter_column} = {filter_value}" if filter_column else ""
    chart_label = "dual-line" if len(metrics) == 2 else "multi-line"
    metric_labels = [_display_column_name(metric) for metric in metrics]
    summary = (
        f"Compared {', '.join(metric_labels)}{filter_text} across "
        f"{len(rows):,} {frequency} period(s). The table and {chart_label} chart "
        f"show one verified row per {frequency}."
    )
    key_findings: list[str] = []
    metric_summaries: list[MetricSummary] = []
    for metric, label in zip(metrics, metric_labels, strict=False):
        overall_value = (
            sum(float(row[metric]) for row in rows)
            if aggregation in {"sum", "count"}
            else sum(float(row[metric]) for row in rows) / len(rows)
        )
        first = rows[0]
        latest = rows[-1]
        peak = max(rows, key=lambda row: row[metric])
        trough = min(rows, key=lambda row: row[metric])
        change = float(latest[metric]) - float(first[metric])
        direction = "increased" if change > 0 else "decreased" if change < 0 else "stayed flat"
        key_findings.append(
            f"Highest {label} was {peak[date_column]} at {format_metric(peak[metric], metric)}."
        )
        trend = (
            f"{label.title()} {direction} from {format_metric(first[metric], metric)} "
            f"in {first[date_column]} to {format_metric(latest[metric], metric)} "
            f"in {latest[date_column]}."
        )
        metric_summaries.append(
            MetricSummary(
                metric_label=label.title(),
                total_value=format_metric(overall_value, metric),
                highest_period=str(peak[date_column]),
                highest_value=format_metric(peak[metric], metric),
                lowest_period=str(trough[date_column]),
                lowest_value=format_metric(trough[metric], metric),
                first_period=str(first[date_column]),
                first_value=format_metric(first[metric], metric),
                latest_period=str(latest[date_column]),
                latest_value=format_metric(latest[metric], metric),
                trend_text=trend,
            )
        )
    return ChatNarrativeResponse(
        summary=summary,
        key_findings=key_findings,
        metric_summaries=metric_summaries,
    )


def _is_summary_request(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "summarize",
            "summary",
            "describe",
            "explain",
            "tell me about",
            "insight",
            "finding",
            "takeaway",
            "conclusion",
            "what is this dataset about",
            "what is the dataset about",
            "what does this mean",
            "what does it mean",
            "interpret this",
        )
    )


def _summary_target(text: str) -> str | None:
    if any(
        phrase in text
        for phrase in (
            "tell me about this dataset",
            "tell me about the dataset",
            "what is this dataset about",
            "what is the dataset about",
            "describe this dataset",
            "describe the dataset",
        )
    ):
        return "dataset"
    if not _is_summary_request(text):
        return None
    if any(
        phrase in text
        for phrase in (
            "above chart",
            "previous chart",
            "last chart",
            "this chart",
            "the chart",
        )
    ):
        return "chart"
    if any(
        phrase in text
        for phrase in (
            "previous result",
            "last result",
            "above result",
            "this result",
            "previous analysis",
            "last analysis",
            "above analysis",
            "what does this mean",
            "what does it mean",
            "interpret this",
        )
    ):
        return "previous"
    if any(
        phrase in text
        for phrase in (
            "the data",
            "this data",
            "dataset",
            "entire data",
            "overall data",
            "insight",
            "finding",
            "takeaway",
            "conclusion",
        )
    ):
        return "dataset"
    return "previous"


def split_user_questions(text: str, limit: int = 10) -> list[str]:
    """Split a chat message into independent analytical requests."""
    cleaned = text.strip()
    if not cleaned:
        return []

    cleaned = re.sub(
        r"(?:^|\n)\s*(?:\d+[.)]|[-*])\s+",
        "\n",
        cleaned,
    )
    request_starters = (
        r"what|which|who|where|when|why|how|find|show|give|tell|"
        r"calculate|compute|plot|chart|graph|compare|list|summarize|"
        r"analyse|analyze|display|identify"
    )
    cleaned = re.sub(
        rf"\.\s*(?=(?:{request_starters})\b)",
        "\n",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        rf"\s+\b(?:and\s+(?:then|also|finally|finaly)|and|also|then|finally|finaly)\b\s+"
        rf"(?=(?:{request_starters})\b)",
        "\n",
        cleaned,
        flags=re.IGNORECASE,
    )
    parts = re.split(r"\?+|[;\n]+", cleaned)
    questions = []
    for part in parts:
        question = re.sub(
            r"^\s*(?:and|also|then)\s+",
            "",
            part,
            flags=re.IGNORECASE,
        ).strip(" \t\r\n.,")
        if question:
            questions.append(question)
    return questions[:limit]


def deterministic_plan(
    question: str,
    profile: DatasetProfile,
    history: list[dict[str, Any]] | None = None,
    dataframe: pd.DataFrame | None = None,
) -> AgentPlan:
    """Interpret common analysis requests without relying on an LLM."""
    columns = [column.name for column in profile.columns]
    question = _correct_question_typos(question, columns, dataframe)
    text = question.lower().strip()
    mentioned = _mentioned_columns(question, columns)
    numeric = profile.numeric_columns
    categorical = profile.categorical_columns
    dates = profile.datetime_columns
    previous = recent_context(history or [], 3)
    previous_args = previous[-1]["arguments"] if previous else {}
    data_type_requested = any(
        phrase in text
        for phrase in (
            "data type",
            "data types",
            "datatype",
            "datatypes",
            "dtype",
            "dtypes",
            "column type",
            "column types",
            "types of all column",
            "types of all columns",
        )
    )
    if data_type_requested:
        requested_column = (
            mentioned[0]
            if mentioned and not any(word in text for word in ("all", "every", "columns"))
            else None
        )
        arguments = {"column": requested_column} if requested_column else {}
        return AgentPlan(
            tool_name="get_column_information",
            arguments=arguments,
            response_mode="text",
        )
    row_column_count_requested = (
        any(phrase in text for phrase in ("how many", "number of", "count of"))
        and any(word in text for word in ("row", "rows", "record", "records"))
        and any(word in text for word in ("column", "columns", "field", "fields"))
    )
    if row_column_count_requested:
        return AgentPlan(tool_name="inspect_dataset", response_mode="text")
    ambiguous_date_plan = _ambiguous_date_plan(question, text, mentioned, numeric, dates)
    if ambiguous_date_plan:
        return ambiguous_date_plan
    ambiguous_filter_plan = _ambiguous_filter_value_plan(question, dataframe, categorical)
    if ambiguous_filter_plan:
        return ambiguous_filter_plan
    date_distinct_plan = _date_distinct_count_plan(
        question,
        text,
        columns,
        categorical,
        dates,
        dataframe,
    )
    if date_distinct_plan:
        return date_distinct_plan
    if not text:
        return AgentPlan(
            tool_name="",
            clarification="Please enter a question about the active dataset.",
        )
    pre_summary_advanced_plan = _advanced_analytics_plan(
        question,
        text,
        columns,
        numeric,
        categorical,
    )
    if pre_summary_advanced_plan and (
        pre_summary_advanced_plan.tool_name == "calculate_grouped_extrema"
        and pre_summary_advanced_plan.arguments.get("aggregation") == "nunique"
        or (
            pre_summary_advanced_plan.tool_name == "analyze_advanced_request"
            and pre_summary_advanced_plan.arguments.get("operation")
            in {
                "derived_difference",
                "derived_ratio",
                "distribution",
                "multi_metric_extrema",
                "negative_groups",
                "negative_record_percentage",
                "loss_by_group",
                "relationship",
            }
        )
        or (
            pre_summary_advanced_plan.tool_name == "analyze_advanced_request"
            and pre_summary_advanced_plan.arguments.get("operation") == "grouped_aggregation"
            and bool(pre_summary_advanced_plan.arguments.get("limit"))
            and len(pre_summary_advanced_plan.arguments.get("metrics") or []) > 1
        )
    ):
        return pre_summary_advanced_plan
    summary_statistics_plan = _numeric_summary_statistics_plan(question, text, numeric)
    if summary_statistics_plan:
        return summary_statistics_plan
    frequency_requested = any(
        phrase in text for phrase in ("most common", "most frequent")
    )
    ranking_requested = any(
        word in text
        for word in (
            "highest",
            "higher",
            "largest",
            "maximum",
            "max",
            "top",
            "best",
            "brings",
            "generated",
            "generates",
            "brought",
            "contributes",
            "contributed",
            "produces",
            "produced",
            "drove",
            "driven",
            "lowest",
            "lower",
            "smallest",
            "minimum",
            "min",
            "least",
        )
    ) or ("most" in text and not frequency_requested)
    ranking_limit = 1 if ranking_requested else None

    if (
        previous
        and previous[-1].get("clarification")
        and len(text.split()) <= 4
        and mentioned
    ):
        previous_question = str(previous[-1].get("question") or "").strip()
        if previous_question:
            return deterministic_plan(
                f"{previous_question} using {question}",
                profile,
                (history or [])[:-1],
                dataframe,
            )
    followup_plan = _followup_time_trend_plan(
        question,
        text,
        dataframe,
        previous,
        numeric,
        categorical,
    )
    if followup_plan:
        return followup_plan
    if "summary statistic" in text or text in {
        "summary statistics",
        "describe the data statistically",
    }:
        return AgentPlan(tool_name="calculate_summary_statistics")
    column_profile_plan = _column_profile_plan(question, text, columns)
    if column_profile_plan:
        return column_profile_plan
    invalid_average_plan = _invalid_non_numeric_average_plan(question, text, columns, numeric)
    if invalid_average_plan:
        return invalid_average_plan
    unknown_date_column = _unknown_requested_date_column_text(text, dates)
    if unknown_date_column:
        return AgentPlan(
            tool_name="",
            clarification=(
                f'I could not find the requested date column "{unknown_date_column}". '
                f"Available date columns: {', '.join(dates) or 'none'}. No calculation was performed."
            ),
        )
    explicit_chart_plan = _explicit_chart_type_plan(
        question,
        text,
        numeric,
        categorical,
    )
    if explicit_chart_plan:
        return explicit_chart_plan
    share_plan = _share_of_total_plan(
        question,
        text,
        dataframe,
        numeric,
        categorical,
    )
    if share_plan:
        return share_plan
    categorical_count_plan = _categorical_value_count_plan(
        question,
        text,
        dataframe,
        columns,
        categorical,
    )
    if categorical_count_plan:
        return categorical_count_plan
    period_change_plan = _period_over_period_plan(
        question,
        text,
        dataframe,
        numeric,
        categorical,
        dates,
    )
    if period_change_plan:
        return period_change_plan
    explicit_grouped_analysis = (
        any(
            phrase in f" {text} "
            for phrase in (
                " by ",
                " per ",
                " vs ",
                " vs. ",
                " versus ",
                " for all ",
                " across all ",
            )
        )
        and any(item in numeric for item in mentioned)
        and any(item in categorical for item in mentioned)
    )
    summary_target = None if explicit_grouped_analysis else _summary_target(text)
    if summary_target == "chart":
        if not history or not any(item.get("chart_spec") for item in reversed(history)):
            return AgentPlan(
                tool_name="",
                clarification="There is no previous chart in this conversation to summarize.",
            )
        return AgentPlan(
            tool_name="summarize_previous",
            arguments={"line_count": _requested_summary_lines(text), "target": "chart"},
        )
    if summary_target == "previous":
        if not history:
            return AgentPlan(
                tool_name="",
                clarification="There is no previous analysis in this conversation to summarize.",
            )
        return AgentPlan(
            tool_name="summarize_previous",
            arguments={
                "line_count": _requested_summary_lines(text, default=3),
                "target": "analysis",
            },
        )
    if summary_target == "dataset":
        return AgentPlan(
            tool_name="summarize_dataset",
            arguments={"line_count": _requested_summary_lines(text, default=5)},
        )
    if "missing" in text or "null" in text:
        return AgentPlan(tool_name="analyze_missing_values")
    if "duplicate" in text:
        return AgentPlan(tool_name="analyze_duplicates")
    if "outlier" in text:
        column = next((item for item in mentioned if item in numeric), None)
        return AgentPlan(tool_name="detect_outliers", arguments={"column": column})
    if "average of each numeric" in text or "average of all numeric" in text:
        return AgentPlan(tool_name="calculate_average_numeric_columns")
    distinct_count_requested = any(
        phrase in text for phrase in ("how many", "number of", "count of")
    )
    if distinct_count_requested:
        explicit_metrics = _explicit_numeric_metrics(question, numeric)
        metric = explicit_metrics[0] if explicit_metrics else None
        category_filter = _detect_categorical_filter(question, dataframe, categorical)
        if metric and (
            any(word in text for word in ("total", "all", "overall", "sold"))
            or category_filter
        ):
            if category_filter:
                filter_column, filter_value = category_filter
                return AgentPlan(
                    tool_name="calculate_filtered_aggregate",
                    arguments={
                        "category_column": filter_column,
                        "category_value": filter_value,
                        "value_column": metric,
                        "aggregation": "sum",
                    },
                    response_mode="text",
                )
            return AgentPlan(
                tool_name="calculate_scalar_aggregate",
                arguments={
                    "value_column": metric,
                    "aggregation": "sum",
                },
                response_mode="text",
            )
        unique_measure = bool(re.search(r"\bunique\b|\bdistinct\b|\bnunique\b", text))
        if unique_measure and category_filter:
            distinct_column = next(
                (
                    column
                    for column in _mentioned_columns(question, columns)
                    if column != category_filter[0]
                ),
                None,
            )
            if distinct_column:
                filter_column, filter_value = category_filter
                return AgentPlan(
                    tool_name="analyze_categorical_value_counts",
                    arguments={
                        "counted_column": filter_column,
                        "primary_group_column": None,
                        "filters": [
                            {
                                "column": filter_column,
                                "operator": "equals",
                                "value": filter_value,
                            }
                        ],
                        "include_missing": False,
                        "normalization": "overall",
                        "chart_type": "bar",
                        "sort_mode": "count_descending",
                        "measure_type": "distinct_count",
                        "distinct_column": distinct_column,
                        "original_query": question,
                    },
                    response_mode="text",
                )
        occurrence_requested = any(
            phrase in text
            for phrase in (
                "how many time",
                "how many times",
                "appear",
                "appears",
                "appeared",
                "occurs",
                "occur",
                "occurred",
                "occurrence",
                "occurrences",
            )
        )
        category_filter = (
            category_filter
            if occurrence_requested
            else None
        )
        if category_filter:
            filter_column, filter_value = category_filter
            return AgentPlan(
                tool_name="analyze_categorical_value_counts",
                arguments={
                    "counted_column": filter_column,
                    "primary_group_column": None,
                    "filters": [
                        {
                            "column": filter_column,
                            "operator": "equals",
                            "value": filter_value,
                        }
                    ],
                    "include_missing": False,
                    "normalization": "overall",
                    "chart_type": "bar",
                    "sort_mode": "count_descending",
                    "measure_type": "row_count",
                    "distinct_column": None,
                    "original_query": question,
                },
                response_mode="text",
            )
        column = next(
            (item for item in mentioned if item in categorical),
            None,
        )
        if column:
            return AgentPlan(
                tool_name="count_distinct_values",
                arguments={"column": column},
                response_mode="text",
            )
    filtered_list_requested = (
        any(
            phrase in text
            for phrase in (
                "list",
                "show me all",
                "show all",
                "name all",
                "names of",
                "give me all",
                "tell me all",
            )
        )
        or bool(re.search(r"\b(?:which|what)\s+.+?\s+(?:are|is)\s+in\b", text))
        or bool(re.search(r"\b(?:which|what)\s+.+?\s+(?:are|is)\s+(?:under|within|for)\b", text))
        or any(phrase in text for phrase in (" names", " values"))
    )
    analytical_list_blocked = any(
        re.search(rf"\b{re.escape(word)}\b", text)
        for word in (
            "highest",
            "lowest",
            "higher",
            "lower",
            "top",
            "bottom",
            "above",
            "below",
            "average",
            "mean",
            "median",
            "sum",
            "total",
            "revenue",
            "profit",
            "sales",
            "count",
            "brings",
            "generating",
            "generated",
            "generate",
            "generates",
        )
    )
    if filtered_list_requested and not analytical_list_blocked:
        category_filter = _detect_categorical_filter(question, dataframe, categorical)
        if category_filter:
            filter_column, filter_value = category_filter
            target_column = next(
                (item for item in mentioned if item in categorical and item != filter_column),
                None,
            )
            if target_column:
                return AgentPlan(
                    tool_name="list_distinct_values",
                    arguments={
                        "target_column": target_column,
                        "filter_column": filter_column,
                        "filter_value": filter_value,
                    },
                    response_mode="text",
                )
    distinct_values_requested = any(
        phrase in text
        for phrase in (
            "what are",
            "which are",
            "list",
            "show",
            "names",
            "values",
            "unique",
            "distinct",
        )
    ) and any(
        phrase in text
        for phrase in (
            " names",
            " values",
            " in column",
            " column",
            "unique",
            "distinct",
        )
    )
    if distinct_values_requested:
        column = next((item for item in mentioned if item in categorical), None)
        if column:
            return AgentPlan(
                tool_name="list_distinct_values",
                arguments={"target_column": column},
                response_mode="text",
            )
    availability_requested = any(
        phrase in text for phrase in ("which ", "what ")
    ) and any(
        phrase in text
        for phrase in ("available", "availability", "offered", "present", "found")
    )
    if availability_requested:
        category_filter = _detect_categorical_filter(question, dataframe, categorical)
        if category_filter:
            filter_column, filter_value = category_filter
            target_column = next(
                (
                    item
                    for item in mentioned
                    if item in categorical and item != filter_column
                ),
                None,
            )
            if target_column:
                return AgentPlan(
                    tool_name="list_distinct_values",
                    arguments={
                        "target_column": target_column,
                        "filter_column": filter_column,
                        "filter_value": filter_value,
                    },
                    response_mode="text",
                )
    benchmark_comparison_requested = any(
        re.search(rf"\b{word}\b", text) for word in ("below", "under", "above", "over")
    ) and any(word in text for word in ("average", "mean", "median"))
    if benchmark_comparison_requested:
        grouped_context = next(
            (
                item.get("arguments", {})
                for item in reversed(previous)
                if item.get("arguments", {}).get("value_column") in numeric
                and item.get("arguments", {}).get("group_by") in categorical
            ),
            {},
        )
        metric = next(
            (item for item in mentioned if item in numeric),
            grouped_context.get("value_column"),
        )
        mentioned_categories = [item for item in mentioned if item in categorical]
        benchmark_group = _benchmark_group_column(question, mentioned_categories)
        category = next(
            (item for item in mentioned_categories if item != benchmark_group),
            mentioned_categories[0] if len(mentioned_categories) == 1 else None,
        )
        if not category and grouped_context.get("group_by") in categorical:
            category = grouped_context["group_by"]
        if not category and len(categorical) == 1:
            category = categorical[0]
        if metric and category:
            excluded = {category}
            if benchmark_group:
                excluded.add(benchmark_group)
            category_filter = _detect_categorical_filter(
                question,
                dataframe,
                categorical,
                excluded_columns=excluded,
            )
            filter_column, filter_value = category_filter or (None, None)
            if (
                not filter_column
                and grouped_context.get("filter_column") in categorical
            ):
                filter_column = grouped_context["filter_column"]
                filter_value = grouped_context.get("filter_value")
            benchmark = "median" if "median" in text else "mean"
            comparison = (
                "below"
                if any(re.search(rf"\b{word}\b", text) for word in ("below", "under"))
                else "above"
            )
            aggregation = (
                "mean"
                if any(
                    phrase in text
                    for phrase in ("average value", "mean value", "average amount")
                )
                else grouped_context.get("aggregation", "sum")
            )
            arguments = {
                "category_column": category,
                "value_column": metric,
                "aggregation": aggregation,
                "benchmark": benchmark,
                "comparison": comparison,
                "benchmark_group_by": benchmark_group,
            }
            if filter_column:
                arguments.update(
                    {
                        "filter_column": filter_column,
                        "filter_value": filter_value,
                    }
                )
            return AgentPlan(
                tool_name="compare_grouped_to_benchmark",
                arguments=arguments,
            )
        return AgentPlan(
            tool_name="",
            clarification=(
                "This comparison needs one numeric metric and at least one "
                "categorical column. For a local benchmark, also name the "
                "parent category, such as Country within Region."
            ),
        )
    directed_metrics = _directed_metric_pair(question, numeric)
    if directed_metrics:
        category = next(
            (item for item in mentioned if item in categorical),
            categorical[0] if len(categorical) == 1 else None,
        )
        volume, outcome = directed_metrics
        if category and volume and outcome:
            return AgentPlan(
                tool_name="analyze_high_volume_low_outcome",
                arguments={
                    "category_column": category,
                    "volume_column": volume,
                    "outcome_column": outcome,
                    "aggregation": "sum",
                },
                chart_spec=ChartSpec(
                    chart_type="dual_axis",
                    x=category,
                    y=volume,
                    secondary_y=outcome,
                    aggregation="sum",
                    sort_descending=True,
                    title=f"{volume} and {outcome} by {category}",
                ),
            )
    multi_value_category = _mentioned_category_values(question, dataframe, categorical)
    if ranking_requested and multi_value_category:
        primary_group, selected_values = multi_value_category
        secondary_group = next(
            (
                column
                for column in _mentioned_columns(question, categorical)
                if column != primary_group
            ),
            None,
        )
        metric = next(iter(_explicit_numeric_metrics(question, numeric)), None)
        if not metric:
            metric = next((item for item in mentioned if item in numeric), None)
        if primary_group and secondary_group and metric:
            if any(word in text for word in ("average", "mean")):
                aggregation = "mean"
            elif "median" in text:
                aggregation = "median"
            else:
                aggregation = "sum"
            extremum = "min" if any(word in text for word in ("lowest", "smallest", "minimum", "min")) else "max"
            return AgentPlan(
                tool_name="calculate_grouped_extrema",
                arguments={
                    "primary_group_column": primary_group,
                    "secondary_group_column": secondary_group,
                    "metric_column": metric,
                    "aggregation": aggregation,
                    "extremum": extremum,
                    "filter_column": primary_group,
                    "filter_values": selected_values,
                },
                chart_spec=ChartSpec(
                    chart_type="grouped_extrema_bar",
                    x=primary_group,
                    y=metric,
                    color=secondary_group,
                    aggregation=aggregation,
                    sort_descending=False,
                    title=(
                        f"{'Highest' if extremum == 'max' else 'Lowest'} "
                        f"{_display_column_name(metric).title()} {secondary_group} "
                        f"by {primary_group}"
                    ),
                ),
            )
    grouped_extrema = _grouped_extrema_intent(text)
    if grouped_extrema:
        _, extremum = grouped_extrema
        primary_group = _primary_group_for_extrema(text, categorical)
        secondary_group = _secondary_group_for_extrema(text, categorical, primary_group)
        if not secondary_group:
            secondary_group = next(
                (
                    column
                    for column in _mentioned_columns(question, categorical)
                    if column != primary_group
                ),
                None,
            )
        metric = next(iter(_explicit_numeric_metrics(question, numeric)), None)
        if not metric:
            metric = next((item for item in mentioned if item in numeric), None)
        if primary_group and secondary_group and metric:
            if any(word in text for word in ("average", "mean")):
                aggregation = "mean"
            elif "median" in text:
                aggregation = "median"
            else:
                aggregation = "sum"
            return AgentPlan(
                tool_name="calculate_grouped_extrema",
                arguments={
                    "primary_group_column": primary_group,
                    "secondary_group_column": secondary_group,
                    "metric_column": metric,
                    "aggregation": aggregation,
                    "extremum": extremum,
                },
                chart_spec=ChartSpec(
                    chart_type="grouped_extrema_bar",
                    x=primary_group,
                    y=metric,
                    color=secondary_group,
                    aggregation=aggregation,
                    sort_descending=False,
                    title=(
                        f"{'Highest' if extremum == 'max' else 'Lowest'} "
                        f"{_display_column_name(metric).title()} {secondary_group} "
                        f"by {primary_group}"
                    ),
                ),
            )
        return AgentPlan(
            tool_name="",
            clarification=(
                "I could not resolve the requested primary group, secondary group, "
                "or metric from the dataset."
            ),
        )
    visualization_requested = any(
        word in text
        for word in (
            "plot",
            "chart",
            "graph",
            "visualize",
            "visualisation",
            "visualization",
        )
    )
    period_start, period_end, period_type, period_value = period_bounds_from_text(text, current_date=_current_date())
    date_breakdown_requested = _date_breakdown_requested(text)
    categorical_breakdown_requested = any(
        phrase in text
        for phrase in (
            "for each",
            "for all",
            " each ",
            " by ",
            " per ",
            "across",
        )
    ) and any(item in categorical for item in mentioned)
    date_total_requested = (
        (period_start is not None or _has_date_intent(text, mentioned, dates))
        and period_start is not None
        and not date_breakdown_requested
        and not categorical_breakdown_requested
        and any(
            phrase in text
            for phrase in (
                "total",
                "sum",
                "average",
                "mean",
                "count",
                "how much",
                "what is",
                "what was",
                "give me",
                "show",
            )
        )
    )
    if date_total_requested:
        date = _preferred_date_column(mentioned, dates)
        metric = next(
            (item for item in mentioned if item in numeric),
            numeric[0] if len(numeric) == 1 else None,
        )
        if not date or not metric:
            return AgentPlan(
                tool_name="",
                clarification=(
                    f"Which date and metric should I use? Date columns: {', '.join(dates) or 'none'}. "
                    f"Numeric columns: {', '.join(numeric) or 'none'}."
                ),
            )
        category_filter = _detect_categorical_filter(question, dataframe, categorical)
        filter_column, filter_value = category_filter or (None, None)
        aggregation = _date_aggregation_from_text(text)
        return AgentPlan(
            tool_name="calculate_date_aggregate",
            arguments={
                "date_column": date,
                "value_column": metric,
                "aggregation": aggregation,
                "start_date": period_start,
                "end_date": period_end,
                "period_type": period_type,
                "period_value": period_value,
                "filter_column": filter_column,
                "filter_value": filter_value,
            },
            response_mode="text",
        )
    scalar_fact_requested = (
        (
            not visualization_requested
            and not date_breakdown_requested
            and not (ranking_requested and any(item in categorical for item in mentioned))
            and not any(
                phrase in text
                for phrase in (
                    " by ",
                    " per ",
                    "across",
                    "for each",
                    " each ",
                    "difference",
                    "compare",
                    "highest",
                    "lowest",
                    "top ",
                    "best",
                    "trend",
                )
            )
            and any(
                phrase in text
                for phrase in (
                    "what is",
                    "what's",
                    "how much",
                    "tell me",
                    "show ",
                    "show me",
                    "total ",
                    "average ",
                    "mean ",
                )
            )
        )
        or _scalar_extrema_requested(text, mentioned, categorical)
    )
    if scalar_fact_requested:
        category_filters = _detect_categorical_filters(question, dataframe, categorical)
        category_filter = category_filters[0] if category_filters else None
        mentioned_metrics = [item for item in mentioned if item in numeric]
        explicit_metrics = _explicit_numeric_metrics(question, numeric)
        metric = explicit_metrics[0] if explicit_metrics else (mentioned_metrics[0] if mentioned_metrics else None)
        if any(word in text for word in ("average", "mean")):
            aggregation = "mean"
        elif "median" in text:
            aggregation = "median"
        elif any(word in text for word in ("minimum", "lowest", "smallest", "min")):
            aggregation = "min"
        elif any(word in text for word in ("maximum", "highest", "largest", "max")):
            aggregation = "max"
        else:
            aggregation = "sum"
        if category_filter and metric:
            category_column, category_value = category_filter
            value_columns = explicit_metrics if len(explicit_metrics) > 1 else None
            return AgentPlan(
                tool_name="calculate_filtered_aggregate",
                arguments={
                    "category_column": category_column,
                    "category_value": category_value,
                    "value_column": metric,
                    "value_columns": value_columns,
                    "aggregation": aggregation,
                    **(
                        {
                            "filters": [
                                {"column": column, "operator": "equals", "value": value}
                                for column, value in category_filters
                            ]
                        }
                        if len(category_filters) > 1
                        else {}
                    ),
                },
                response_mode="text",
            )
        if metric:
            unresolved_value = _unresolved_filter_value_text(text, columns)
            if unresolved_value:
                return AgentPlan(
                    tool_name="",
                    clarification=(
                        f'I could not find "{unresolved_value}" in the uploaded dataset. '
                        "No calculation was performed."
                    ),
                )
        if len(explicit_metrics) > 1:
            return AgentPlan(
                tool_name="calculate_multi_scalar_aggregate",
                arguments={
                    "value_columns": explicit_metrics,
                    "aggregation": aggregation,
                },
                response_mode="text",
            )
        if metric:
            return AgentPlan(
                tool_name="calculate_scalar_aggregate",
                arguments={
                    "value_column": metric,
                    "aggregation": aggregation,
                },
                response_mode="text",
            )
    comparison_requested = any(
        phrase in text
        for phrase in (
            "difference",
            "gap",
            "versus",
            " vs ",
            "compare between",
            "comparison between",
            "how much more",
            "how much less",
        )
    )
    if comparison_requested:
        category_match = _mentioned_category_values(question, dataframe, categorical)
        metric = next((item for item in mentioned if item in numeric), None)
        if category_match and metric:
            category_column, values = category_match
            aggregation = (
                "mean" if any(word in text for word in ("average", "mean")) else "sum"
            )
            return AgentPlan(
                tool_name="compare_category_values",
                arguments={
                    "category_column": category_column,
                    "value_column": metric,
                    "first_value": values[0],
                    "second_value": values[1],
                    "aggregation": aggregation,
                },
                chart_spec=ChartSpec(
                    chart_type="bar",
                    x=category_column,
                    y=metric,
                    aggregation=aggregation,
                    include_values=values[:2],
                    title=f"{aggregation.title()} {metric}: {values[0]} vs {values[1]}",
                ),
            )
        if category_match and not metric:
            return AgentPlan(
                tool_name="",
                clarification=f"Which numeric metric should I compare? {', '.join(numeric)}.",
            )
    mentioned_metrics = [item for item in mentioned if item in numeric]
    explicit_mentioned_metrics = _explicit_numeric_metrics(question, numeric)
    mentioned_categories = [item for item in mentioned if item in categorical]
    multi_metric_date_requested = (
        len(explicit_mentioned_metrics) >= 2
        and _has_date_intent(text, mentioned, dates)
        and (
            date_breakdown_requested
            or visualization_requested
            or ranking_requested
            or any(phrase in text for phrase in ("over time", "over the year", "over the years"))
        )
    )
    if multi_metric_date_requested:
        date = _preferred_date_column(mentioned, dates)
        if not date:
            return AgentPlan(
                tool_name="",
                clarification=f"Which date column should I use? Date columns: {', '.join(dates) or 'none'}.",
            )
        metrics = explicit_mentioned_metrics
        primary = metrics[0]
        category_filter = _detect_categorical_filter(question, dataframe, categorical)
        filter_column, filter_value = category_filter or (None, None)
        frequency = _time_frequency_from_text(text)
        aggregation = "mean" if "average" in text or "mean" in text else "sum"
        filter_title = (
            f" for {filter_column} {filter_value}"
            if filter_column is not None
            else ""
        )
        if len(metrics) == 2:
            secondary = metrics[1]
            chart_spec = ChartSpec(
                chart_type="dual_line",
                x=date,
                y=primary,
                secondary_y=secondary,
                aggregation=aggregation,
                title=(
                    f"{aggregation.title()} {primary} and {secondary} "
                    f"by {frequency.title()}{filter_title}"
                ),
                time_grain=frequency,
                time_column=date,
                date_range_start=period_start,
                date_range_end=period_end,
                filter_column=filter_column,
                filter_value=filter_value,
            )
        else:
            chart_spec = ChartSpec(
                chart_type="line",
                x=date,
                y="Value",
                color="Metric",
                value_columns=metrics,
                aggregation=aggregation,
                title=(
                    f"{aggregation.title()} {', '.join(metrics)} "
                    f"by {frequency.title()}{filter_title}"
                ),
                time_grain=frequency,
                time_column=date,
                date_range_start=period_start,
                date_range_end=period_end,
                filter_column=filter_column,
                filter_value=filter_value,
            )
        return AgentPlan(
            tool_name="calculate_time_trend",
            arguments={
                "date_column": date,
                "value_column": primary,
                "value_columns": metrics,
                "aggregation": aggregation,
                "frequency": frequency,
                "start_date": period_start,
                "end_date": period_end,
                "filter_column": filter_column,
                "filter_value": filter_value,
            },
            chart_spec=chart_spec,
        )
    dual_axis_requested = any(
        phrase in text
        for phrase in (
            "dual axis",
            "dual-axis",
            "two axes",
            "two axis",
            "secondary axis",
            "secondary y-axis",
            "secondary y axis",
        )
    )
    multi_metric_comparison = (
        len(explicit_mentioned_metrics) >= 2
        and bool(mentioned_categories)
        and (
            dual_axis_requested
            or visualization_requested
            or any(
                phrase in text
                for phrase in ("compare ", " versus ", " vs ", "together", "alongside")
            )
        )
    )
    if multi_metric_comparison:
        group = mentioned_categories[0]
        primary, secondary = explicit_mentioned_metrics[:2]
        aggregation = (
            "mean" if any(word in text for word in ("average", "mean")) else "sum"
        )
        return AgentPlan(
            tool_name="group_and_aggregate",
            arguments={
                "group_by": group,
                "value_column": primary,
                "value_columns": [primary, secondary],
                "aggregation": aggregation,
            },
            chart_spec=ChartSpec(
                chart_type="dual_axis",
                x=group,
                y=primary,
                secondary_y=secondary,
                aggregation=aggregation,
                sort_descending=True,
                title=f"{primary} and {secondary} by {group}",
            ),
        )
    if dual_axis_requested:
        return AgentPlan(
            tool_name="",
            clarification=(
                "A dual-axis chart needs one categorical column and two numeric columns. "
                f"Numeric columns: {', '.join(numeric) or 'none'}. "
                f"Categorical columns: {', '.join(categorical) or 'none'}."
            ),
        )
    if "correlat" in text:
        selected = [item for item in mentioned if item in numeric]
        if len(selected) >= 2:
            return AgentPlan(
                tool_name="calculate_correlation",
                arguments={"first_column": selected[0], "second_column": selected[1]},
                chart_spec=ChartSpec(
                    chart_type="scatter",
                    x=selected[0],
                    y=selected[1],
                    title=f"{selected[1]} versus {selected[0]}",
                ),
            )
        if len(numeric) >= 2 and not mentioned:
            return AgentPlan(
                tool_name="calculate_correlation",
                chart_spec=ChartSpec(chart_type="heatmap", title="Correlation Matrix"),
            )
        return AgentPlan(
            tool_name="",
            clarification=f"Choose two numeric columns: {', '.join(numeric)}.",
        )
    period_start, period_end, period_type, period_value = period_bounds_from_text(text, current_date=_current_date())
    date_breakdown_requested = _date_breakdown_requested(text)
    categorical_breakdown_requested = any(
        phrase in text
        for phrase in (
            "for each",
            "for all",
            " each ",
            " by ",
            " per ",
            "across",
        )
    ) and any(item in categorical for item in mentioned)
    date_total_requested = (
        _has_date_intent(text, mentioned, dates)
        and period_start is not None
        and not date_breakdown_requested
        and not categorical_breakdown_requested
        and any(
            phrase in text
            for phrase in (
                "total",
                "sum",
                "average",
                "mean",
                "count",
                "how much",
                "what is",
                "what was",
                "give me",
                "show",
            )
        )
    )
    if date_total_requested:
        date = _preferred_date_column(mentioned, dates)
        metric = next(
            (item for item in mentioned if item in numeric),
            numeric[0] if len(numeric) == 1 else None,
        )
        if not date or not metric:
            return AgentPlan(
                tool_name="",
                clarification=(
                    f"Which date and metric should I use? Date columns: {', '.join(dates) or 'none'}. "
                    f"Numeric columns: {', '.join(numeric) or 'none'}."
                ),
            )
        category_filter = _detect_categorical_filter(question, dataframe, categorical)
        filter_column, filter_value = category_filter or (None, None)
        aggregation = _date_aggregation_from_text(text)
        return AgentPlan(
            tool_name="calculate_date_aggregate",
            arguments={
                "date_column": date,
                "value_column": metric,
                "aggregation": aggregation,
                "start_date": period_start,
                "end_date": period_end,
                "period_type": period_type,
                "period_value": period_value,
                "filter_column": filter_column,
                "filter_value": filter_value,
            },
            response_mode="text",
        )

    breakdown_column = _breakdown_column_from_text(text, categorical)
    time_series_breakdown_requested = (
        breakdown_column is not None
        and _has_date_intent(text, mentioned, dates)
        and (
            date_breakdown_requested
            or any(word in text for word in ("daily", "weekly", "monthly", "yearly"))
            or "over time" in text
        )
    )
    if time_series_breakdown_requested:
        date = _preferred_date_column(mentioned, dates)
        metric = next(
            (item for item in mentioned if item in numeric),
            numeric[0] if len(numeric) == 1 else None,
        )
        if not date or not metric:
            return AgentPlan(
                tool_name="",
                clarification=(
                    f"Which date and metric should I use? Date columns: {', '.join(dates) or 'none'}. "
                    f"Numeric columns: {', '.join(numeric) or 'none'}."
                ),
            )
        category_filter = _detect_categorical_filter(
            question,
            dataframe,
            categorical,
            excluded_columns={breakdown_column},
        )
        filter_column, filter_value = category_filter or (None, None)
        frequency = _time_frequency_from_text(text)
        aggregation = _date_aggregation_from_text(text)
        filter_title = (
            f" for {filter_column} {filter_value}"
            if filter_column is not None
            else ""
        )
        return AgentPlan(
            tool_name="calculate_time_trend",
            arguments={
                "date_column": date,
                "value_column": metric,
                "breakdown_column": breakdown_column,
                "aggregation": aggregation,
                "frequency": frequency,
                "start_date": period_start,
                "end_date": period_end,
                "filter_column": filter_column,
                "filter_value": filter_value,
            },
            chart_spec=ChartSpec(
                chart_type="line",
                x=date,
                y=metric,
                color=breakdown_column,
                aggregation=aggregation,
                title=(
                    f"{aggregation.title()} {metric} by {frequency.title()} "
                    f"and {breakdown_column}{filter_title}"
                ),
                time_grain=frequency,
                time_column=date,
                date_range_start=period_start,
                date_range_end=period_end,
                filter_column=filter_column,
                filter_value=filter_value,
            ),
        )

    date_question_requested = _has_date_intent(text, mentioned, dates) and (
        any(item in numeric for item in mentioned)
        or len(numeric) == 1
        or any(word in text for word in ("trend", "over time", "monthly", "yearly", "weekly"))
    ) and not categorical_breakdown_requested and not (
        ranking_requested and any(item in categorical for item in mentioned)
    )
    if date_question_requested:
        date = next(
            (item for item in mentioned if item in dates),
            dates[0] if len(dates) == 1 else None,
        )
        metric = next(
            (item for item in mentioned if item in numeric),
            numeric[0] if len(numeric) == 1 else None,
        )
        if not date or not metric:
            return AgentPlan(
                tool_name="",
                clarification=(
                    f"Which date and metric should I use? Date columns: {', '.join(dates) or 'none'}. "
                    f"Numeric columns: {', '.join(numeric) or 'none'}."
                ),
            )
        category_filter = _detect_categorical_filter(question, dataframe, categorical)
        filter_column, filter_value = category_filter or (None, None)
        frequency = _time_frequency_from_text(text)
        aggregation = "mean" if "average" in text or "mean" in text else "sum"
        filter_title = (
            f" for {filter_column} {filter_value}"
            if filter_column is not None
            else ""
        )
        return AgentPlan(
            tool_name="calculate_time_trend",
            arguments={
                "date_column": date,
                "value_column": metric,
                "aggregation": aggregation,
                "frequency": frequency,
                "start_date": period_start,
                "end_date": period_end,
                "filter_column": filter_column,
                "filter_value": filter_value,
            },
            chart_spec=ChartSpec(
                chart_type="line",
                x=date,
                y=metric,
                aggregation=aggregation,
                title=f"{aggregation.title()} {metric} by {frequency.title()}{filter_title}",
                time_grain=frequency,
                time_column=date,
                date_range_start=period_start,
                date_range_end=period_end,
                filter_column=filter_column,
                filter_value=filter_value,
            ),
        )
    top_match = re.search(r"top\s+(\d+|three|five|ten)", text)
    top_words = {"three": 3, "five": 5, "ten": 10}
    if top_match:
        raw_limit = top_match.group(1)
        limit = top_words.get(raw_limit, int(raw_limit) if raw_limit.isdigit() else 5)
        metric = next((item for item in mentioned if item in numeric), None)
        mentioned_categories = [item for item in mentioned if item in categorical]
        group = mentioned_categories[0] if mentioned_categories else None
        category_filter = _detect_categorical_filter(
            question,
            dataframe,
            categorical,
            excluded_columns={group} if group else None,
        )
        if metric and group:
            filter_column, filter_value = category_filter or (None, None)
            aggregation = (
                "mean" if any(word in text for word in ("average", "mean")) else "sum"
            )
            return AgentPlan(
                tool_name="group_and_aggregate",
                arguments={
                    "group_by": group,
                    "value_column": metric,
                    "aggregation": aggregation,
                    "limit": limit,
                    "filter_column": filter_column,
                    "filter_value": filter_value,
                },
                chart_spec=ChartSpec(
                    chart_type="bar",
                    x=group,
                    y=metric,
                    aggregation=aggregation,
                    sort_descending=True,
                    limit=limit,
                    filter_column=filter_column,
                    filter_value=filter_value,
                    title=(
                        f"Top {limit} {group} by {metric}"
                        + (f" in {filter_value}" if filter_column else "")
                    ),
                ),
            )
        if previous_args.get("group_by") and previous_args.get("value_column"):
            group = previous_args["group_by"]
            metric = next(
                (item for item in mentioned if item in numeric),
                previous_args["value_column"],
            )
            aggregation = previous_args.get("aggregation", "sum")
            return AgentPlan(
                tool_name="group_and_aggregate",
                arguments={
                    "group_by": group,
                    "value_column": metric,
                    "aggregation": aggregation,
                    "limit": limit,
                },
                chart_spec=ChartSpec(
                    chart_type="bar",
                    x=group,
                    y=metric,
                    aggregation=aggregation,
                    sort_descending=True,
                    limit=limit,
                    title=f"Top {limit} {group} by {metric}",
                ),
            )
        sort_by = next(
            (item for item in mentioned if item in numeric),
            previous_args.get("sort_by"),
        )
        if sort_by:
            display_columns = mentioned or previous_args.get("columns")
            return AgentPlan(
                tool_name="sort_and_limit",
                arguments={
                    "sort_by": sort_by,
                    "limit": limit,
                    "ascending": False,
                    "columns": display_columns or None,
                },
                chart_spec=None,
            )
    aggregation = "mean" if any(word in text for word in ("average", "mean")) else "sum"
    if ranking_requested or any(
        word in text
        for word in (
            "total",
            "average",
            "mean",
            "sum",
            "compare",
            " by ",
            " per ",
            "performance",
            "across",
            "for all",
            "for each",
            "find ",
            "show ",
            "give me",
        )
    ):
        mentioned_metrics = [item for item in mentioned if item in numeric]
        explicit_metrics = _explicit_numeric_metrics(question, numeric)
        metric = mentioned_metrics[0] if mentioned_metrics else None
        question_tokens = _question_tokens(question)
        normalized_question = _normalize(question)
        mentioned_categories = [
            item
            for item in mentioned
            if item in categorical
            and (
                _column_tokens(item).issubset(question_tokens)
                or any(alias in normalized_question for alias in _column_aliases(item))
            )
        ]
        if not mentioned_categories:
            mentioned_categories = [item for item in mentioned if item in categorical]
        group = _which_column_from_text(text, categorical) or (
            mentioned_categories[0] if mentioned_categories else None
        )
        category_filter = _detect_categorical_filter(
            question,
            dataframe,
            categorical,
            excluded_columns={group} if group else None,
        )
        if not category_filter:
            category_filter = _referenced_filter_context(text, previous, categorical)
        if not category_filter and metric:
            unresolved_value = _unresolved_filter_value_text(text, columns)
            if unresolved_value:
                return AgentPlan(
                    tool_name="",
                    clarification=(
                        f'I could not find "{unresolved_value}" in the uploaded dataset. '
                        "No calculation was performed."
                    ),
                )
        filter_column, filter_value = category_filter or (None, None)
        if filter_column and filter_column != group:
            mentioned_categories = [
                item for item in mentioned_categories if item != filter_column
            ]
            group = mentioned_categories[0] if mentioned_categories else group
        secondary_group = (
            next(
                (item for item in mentioned_categories[1:] if item != filter_column),
                None,
            )
            if len(mentioned_categories) > 1
            else None
        )
        if not metric and previous_args.get("value_column") in numeric:
            metric = previous_args["value_column"]
        if not group and previous_args.get("group_by") in categorical:
            group = previous_args["group_by"]
        count_ranking_requested = (
            ranking_requested
            and group
            and not metric
            and (
                re.search(r"\bcounts?\b", text)
                or any(phrase in text for phrase in ("how many", "number of", "frequency"))
            )
        )
        if count_ranking_requested:
            ascending = any(word in text for word in ("lowest", "smallest", "least"))
            return AgentPlan(
                tool_name="group_and_aggregate",
                arguments={
                    "group_by": group,
                    "secondary_group_by": secondary_group,
                    "value_column": "Count",
                    "value_columns": None,
                    "aggregation": "count",
                    "limit": ranking_limit,
                    "filter_column": filter_column,
                    "filter_value": filter_value,
                    "sort_descending": not ascending,
                },
                chart_spec=ChartSpec(
                    chart_type=(
                        "stacked_bar"
                        if secondary_group
                        else "bar"
                    ),
                    x=group,
                    y="Count",
                    color=secondary_group,
                    aggregation="count",
                    sort_descending=not ascending,
                    limit=ranking_limit,
                    filter_column=filter_column,
                    filter_value=filter_value,
                    title=(
                        f"Count by {group}"
                        + (f" and {secondary_group}" if secondary_group else "")
                        + (f" in {filter_value}" if filter_column else "")
                    ),
                ),
            )
        if metric and group:
            value_columns = explicit_metrics if len(explicit_metrics) > 1 else None
            if value_columns:
                metric = value_columns[0]
            ascending = any(word in text for word in ("lowest", "smallest"))
            percentage_of_total_requested = (
                any(word in text for word in ("percentage", "percent", "share"))
                and any(phrase in text for phrase in ("of total", "from each", "comes from", "contribution"))
                and not value_columns
                and secondary_group is None
            )
            date = _preferred_date_column(mentioned, dates)
            date_args = (
                {
                    "date_column": date,
                    "start_date": period_start,
                    "end_date": period_end,
                }
                if date and period_start is not None
                else {}
            )
            percentage_args = (
                {"include_percentage": True}
                if percentage_of_total_requested
                else {}
            )
            return AgentPlan(
                tool_name="group_and_aggregate",
                arguments={
                    "group_by": group,
                    "secondary_group_by": secondary_group,
                    "value_column": metric,
                    "value_columns": value_columns,
                    "aggregation": aggregation,
                    "limit": ranking_limit,
                    "filter_column": filter_column,
                    "filter_value": filter_value,
                    **percentage_args,
                    **({"sort_descending": not ascending} if ranking_requested else {}),
                    **date_args,
                },
                chart_spec=(
                    ChartSpec(
                        chart_type=(
                            "stacked_bar"
                            if secondary_group and not value_columns
                            else "bar"
                        ),
                        x=group,
                        y="PercentageOfTotal" if percentage_of_total_requested else metric,
                        color=secondary_group,
                        value_columns=value_columns or [],
                        aggregation=None if percentage_of_total_requested else aggregation,
                        sort_descending=not ascending,
                        limit=ranking_limit,
                        filter_column=filter_column,
                        filter_value=filter_value,
                        time_column=date if date_args else None,
                        date_range_start=period_start if date_args else None,
                        date_range_end=period_end if date_args else None,
                        title=(
                            (
                                f"Percentage of Total {_display_column_name(metric)} by {group}"
                                if percentage_of_total_requested
                                else f"{aggregation.title()} "
                                f"{' and '.join(value_columns) if value_columns else metric} "
                                f"by {group}"
                            )
                        )
                        + (f" and {secondary_group}" if secondary_group else "")
                        + (f" in {filter_value}" if filter_column else ""),
                    )
                ),
            )
        advanced_plan = _advanced_analytics_plan(
            question,
            text,
            columns,
            numeric,
            categorical,
        )
        if advanced_plan:
            return advanced_plan
        if len(numeric) > 1 or len(categorical) > 1:
            return AgentPlan(
                tool_name="",
                clarification=(
                    f"Which metric and category should I use? Numeric columns: {', '.join(numeric)}. "
                    f"Categorical columns: {', '.join(categorical)}."
                ),
            )
    if any(word in text for word in ("frequency", "value count", "most common")):
        column = (
            mentioned[0]
            if mentioned
            else (categorical[0] if len(categorical) == 1 else None)
        )
        if column:
            return AgentPlan(
                tool_name="calculate_value_counts",
                arguments={"column": column, "limit": 20},
            )
    if any(
        word in text
        for word in ("important finding", "overview", "inspect", "rows", "columns")
    ):
        return AgentPlan(tool_name="inspect_dataset")
    return AgentPlan(
        tool_name="",
        clarification=(
            "I could not map that request to a safe analytical operation. Try asking about totals by category, "
            "missing values, correlations, outliers, rankings, or time trends."
        ),
    )


def _deterministic_explanation(plan: AgentPlan, result: ToolResult) -> str:
    if plan.tool_name in {
        "create_scatter_plot",
        "create_histogram",
        "create_box_plot",
        "create_pie_chart",
        "create_heatmap",
    } and plan.chart_spec:
        spec = plan.chart_spec
        chart_name = spec.chart_type.replace("_", " ")
        details = []
        if spec.x:
            details.append(f"x-axis: **{spec.x}**")
        if spec.y:
            details.append(f"y-axis: **{spec.y}**")
        if spec.value_columns:
            details.append(f"columns: **{', '.join(spec.value_columns)}**")
        if spec.aggregation:
            details.append(f"aggregation: **{spec.aggregation}**")
        detail_text = f" ({'; '.join(details)})" if details else ""
        return f"Created a **{chart_name}** chart{detail_text}."
    if plan.tool_name == "analyze_advanced_request":
        operation = str(plan.arguments.get("operation") or "")
        data = result.data
        if operation == "negative_record_percentage" and isinstance(data, dict):
            metric = data.get("metric_column") or plan.arguments.get("metric_column")
            percentage = float(data.get("percentage", 0) or 0)
            return (
                f"{data.get('negative_count', 0):,} of {data.get('valid_count', 0):,} "
                f"valid {metric} records are negative "
                f"({percentage:.1f}%)."
            )
        if operation == "distribution" and isinstance(data, dict):
            metric = data.get("metric_column") or plan.arguments.get("metric_column")
            return (
                f"For **{metric}**, min is {format_metric(data.get('min'), metric)}, "
                f"median is {format_metric(data.get('median'), metric)}, mean is "
                f"{format_metric(data.get('mean'), metric)}, and max is "
                f"{format_metric(data.get('max'), metric)} across "
                f"{data.get('count', 0):,} valid value(s)."
            )
        if operation == "relationship" and isinstance(data, dict):
            first = data.get("first_column")
            second = data.get("second_column")
            return (
                f"Measured the relationship between **{first}** and **{second}**. "
                f"Pearson correlation is {data.get('pearson', 0):.3f}; "
                f"Spearman correlation is {data.get('spearman', 0):.3f}."
            )
        if operation == "multi_metric_extrema" and isinstance(data, dict):
            winners = data.get("winners") or []
            pieces = []
            for row in winners[:4]:
                group_items = [
                    f"{key} = {value}"
                    for key, value in row.items()
                    if key not in {"Metric", "Aggregation", "Objective", "Value"}
                ]
                pieces.append(
                    f"{str(row.get('Objective', '')).title()} {row.get('Metric')} is "
                    f"{format_metric(row.get('Value'), str(row.get('Metric') or ''))}"
                    f" at {', '.join(group_items)}"
                )
            return ". ".join(pieces) + "." if pieces else result.summary
        if isinstance(data, list):
            return f"{result.summary} Returned **{len(data):,} row(s)**."
        if isinstance(data, dict) and data.get("table_rows"):
            return f"{result.summary} Returned **{len(data['table_rows']):,} row(s)**."
        return result.summary
    if plan.tool_name == "inspect_dataset" and isinstance(result.data, dict):
        rows = result.data.get("rows")
        columns = result.data.get("columns")
        column_names = result.data.get("column_names") or []
        preview = (
            f" Columns include: {', '.join(map(str, column_names[:8]))}."
            if column_names
            else ""
        )
        return (
            f"The dataset contains **{rows:,} rows** and **{columns:,} columns**."
            f"{preview}"
        )
    if plan.tool_name == "compare_grouped_to_benchmark" and isinstance(
        result.data, list
    ):
        category = plan.arguments["category_column"]
        metric = plan.arguments["value_column"]
        comparison = plan.arguments["comparison"]
        parent = plan.arguments.get("benchmark_group_by")
        filter_column = plan.arguments.get("filter_column")
        filter_value = plan.arguments.get("filter_value")
        filter_text = (
            f" for {filter_column} = **{filter_value}**" if filter_column else ""
        )
        if not result.data:
            scope = f" within any {parent}" if parent else ""
            return (
                f"No {category} values are {comparison} the requested "
                f"{plan.arguments['benchmark']} benchmark{scope}{filter_text}."
            )
        details = []
        for row in result.data[:10]:
            difference = abs(float(row["DifferenceFromBenchmark"]))
            parent_text = f" in {row[parent]}" if parent else ""
            details.append(
                f"**{row[category]}**{parent_text}: "
                f"{format_metric(row[metric], metric)} versus "
                f"{format_metric(row['Benchmark'], metric)} "
                f"({format_metric(difference, metric)} {comparison})"
            )
        remaining = len(result.data) - len(details)
        suffix = f" Plus {remaining} more." if remaining else ""
        prefix = f"Using only records{filter_text}: " if filter_column else ""
        return prefix + "; ".join(details) + "." + suffix
    if plan.tool_name == "analyze_high_volume_low_outcome" and isinstance(
        result.data, dict
    ):
        data = result.data
        candidates = data.get("candidates", [])
        volume = data["volume_column"]
        outcome = data["outcome_column"]
        category = data["category_column"]
        volume_threshold = format_metric(data["volume_threshold"], volume)
        outcome_threshold = format_metric(data["outcome_threshold"], outcome)
        if not candidates:
            return (
                f"No {category} meets both conditions: {volume} at or above "
                f"**{volume_threshold}** and {outcome} below "
                f"**{outcome_threshold}**."
            )
        details = "; ".join(
            f"**{row[category]}** ({volume}: "
            f"{format_metric(row[volume], volume)}, {outcome}: "
            f"{format_metric(row[outcome], outcome)})"
            for row in candidates
        )
        return (
            f"{details}. High {volume} means at or above its median "
            f"(**{volume_threshold}**), while low {outcome} means below its "
            f"median (**{outcome_threshold}**)."
        )
    if plan.tool_name == "get_column_information" and isinstance(result.data, list):
        if not result.data:
            return "No column information was available."
        details = []
        for row in result.data:
            details.append(
                f"**{row['name']}**: {row['pandas_dtype']} ({row['kind']})"
            )
        return "Column data types: " + "; ".join(details) + "."
    if plan.tool_name == "profile_column" and isinstance(result.data, dict):
        data = result.data
        profile = data.get("profile", {})
        display = profile.get("display_name", profile.get("column_name", "Column"))
        semantic_type = str(profile.get("semantic_type", "unknown")).replace("_", " ")
        examples = profile.get("example_values") or []
        example_text = (
            f" Example values: {', '.join(map(str, examples[:5]))}."
            if examples
            else ""
        )
        parts = [
            "Column overview",
            (
                f"{display} is a {semantic_type} column with pandas dtype "
                f"{profile.get('pandas_dtype')}. It has {profile.get('row_count', 0):,} row(s), "
                f"{profile.get('non_null_count', 0):,} non-null value(s), "
                f"{profile.get('missing_count', 0):,} missing value(s) "
                f"({profile.get('missing_percentage', 0):.1f}%), and "
                f"{profile.get('unique_count', 0):,} unique value(s).{example_text}"
            ),
        ]
        if data.get("table_rows"):
            parts.extend([
                "Statistics or value distribution",
                "The table below contains the verified type-specific statistics or value distribution.",
            ])
        if profile.get("meaning"):
            confidence = profile.get("meaning_confidence", "low")
            parts.extend([
                "Meaning",
                f"{profile['meaning']} Meaning confidence: {confidence}.",
            ])
        if data.get("caution"):
            parts.extend(["Caution", str(data["caution"])])
        if data.get("recommended_next_step"):
            parts.extend(["Recommended next step", str(data["recommended_next_step"])])
        return "\n\n".join(parts)
    if plan.tool_name == "calculate_summary_statistics" and isinstance(
        result.data, list
    ):
        if not result.data:
            return "No numeric summary statistics were available."
        parts = []
        for row in result.data:
            column = row.get("column", "column")
            stat_values = [
                ("minimum", row.get("min")),
                ("maximum", row.get("max")),
                ("mean", row.get("mean")),
                ("median", row.get("50%")),
            ]
            formatted = [
                f"{label}: **{format_metric(value, str(column))}**"
                for label, value in stat_values
                if value is not None and not pd.isna(value)
            ]
            if formatted:
                parts.append(
                    f"{_display_column_name(str(column)).title()} statistics: "
                    + "; ".join(formatted)
                )
        return ". ".join(parts) + "."
    if plan.tool_name == "calculate_scalar_aggregate" and isinstance(result.data, dict):
        data = result.data
        aggregation_label = {
            "sum": "total",
            "mean": "mean",
            "median": "median",
            "min": "minimum",
            "max": "maximum",
        }.get(data["aggregation"], data["aggregation"])
        return (
            f"The {aggregation_label} "
            f"{_display_column_name(data['value_column'])} is "
            f"**{format_metric(data['value'], data['value_column'])}**."
        )
    if plan.tool_name == "calculate_multi_scalar_aggregate" and isinstance(
        result.data, list
    ):
        statements = []
        for row in result.data:
            aggregation_label = {
                "sum": "total",
                "mean": "mean",
                "median": "median",
                "min": "minimum",
                "max": "maximum",
            }.get(row["aggregation"], row["aggregation"])
            statements.append(
                f"{aggregation_label.capitalize()} "
                f"{_display_column_name(row['value_column'])}: "
                f"**{format_metric(row['value'], row['value_column'])}**"
            )
        return "; ".join(statements) + "."
    if plan.tool_name == "calculate_date_aggregate" and isinstance(result.data, dict):
        data = result.data
        metric = data.get("metric", plan.arguments.get("value_column"))
        aggregation = data.get("aggregation", plan.arguments.get("aggregation", "sum"))
        value = data.get("value")
        period_label = data.get("period_label", "the selected period")
        row_count = int(data.get("row_count", 0) or 0)
        date_column = data.get("date_column", plan.arguments.get("date_column"))
        start = data.get("start_date")
        end = data.get("end_date")
        if aggregation == "mean":
            lead = f"The average {_display_column_name(metric)} for {period_label} is **{format_metric(value, metric)}**"
        elif aggregation == "count":
            lead = f"The record count for {period_label} is **{int(value or 0):,}**"
        elif aggregation in {"nunique", "unique count", "distinct_count"}:
            lead = (
                f"The unique {_display_column_name(metric)} count for {period_label} "
                f"is **{int(value or 0):,}**"
            )
        elif aggregation in {"median", "min", "max"}:
            lead = f"The {aggregation} {_display_column_name(metric)} for {period_label} is **{format_metric(value, metric)}**"
        else:
            lead = f"Total {_display_column_name(metric)} for {period_label} is **{format_metric(value, metric)}**"
        date_detail = ""
        if start and end:
            date_detail = f" Dates included: {pd.Timestamp(start).date()} to {pd.Timestamp(end).date()}."
        return (
            f"{lead}, based on {row_count:,} record(s) using the {date_column} column."
            f"{date_detail}"
        )
    if plan.tool_name == "list_distinct_values" and isinstance(result.data, dict):
        data = result.data
        values = [str(value) for value in data.get("values", [])]
        if not values:
            if data.get("filter_column"):
                return (
                    f"No {data['target_column']} values were found for "
                    f"{data['filter_column']} = **{data['filter_value']}**."
                )
            return f"No non-empty {data['target_column']} values were found."
        if not data.get("filter_column"):
            return (
                f"{data['target_column']} values are **{', '.join(values)}** "
                f"({data['count']} distinct value(s))."
            )
        return (
            f"The **{data['filter_value']}** {data['filter_column']} is available in "
            f"**{', '.join(values)}** ({data['count']} {data['target_column']} values)."
        )
    if plan.tool_name == "count_distinct_values" and isinstance(result.data, dict):
        data = result.data
        return f"There are **{data['count']}** distinct {data['column']} values."
    if plan.tool_name == "analyze_categorical_value_counts" and isinstance(result.data, dict):
        data = result.data
        request = data.get("request", {})
        counted = request.get("counted_column", plan.arguments.get("counted_column"))
        primary = request.get("primary_group_column")
        rows = data.get("table_rows", [])
        total = int(data.get("total_matching_rows", 0) or 0)
        filters = request.get("filters", [])
        filter_text = ""
        if filters:
            parts = [
                f"{item.get('column')} = **{item.get('value')}**"
                for item in filters
                if isinstance(item, dict)
            ]
            if parts:
                filter_text = " after applying " + ", ".join(parts)
        count_column = (
            f"Unique {request.get('distinct_column')} Count"
            if request.get("measure_type") == "distinct_count"
            else "Count"
        )
        if not rows:
            return f"No value-count rows were available for {counted}{filter_text}."
        if primary:
            dominant = []
            for group_value in dict.fromkeys(row.get(primary) for row in rows):
                group_rows = [row for row in rows if row.get(primary) == group_value]
                if not group_rows:
                    continue
                leader = max(group_rows, key=lambda row: row.get(count_column, 0))
                dominant.append(
                    f"**{group_value}**: **{leader.get(counted)}** "
                    f"({int(leader.get(count_column, 0)):,}, {leader.get('Percentage', 0):.1f}%)"
                )
            shown = "; ".join(dominant[:5])
            remaining = len(dominant) - min(len(dominant), 5)
            suffix = f" Plus {remaining} more group(s)." if remaining else ""
            return (
                f"Calculated "
                f"{'unique ' + str(request.get('distinct_column')) + ' counts' if request.get('measure_type') == 'distinct_count' else 'row-count frequencies'} "
                f"for **{counted}** by **{primary}**"
                f"{filter_text}, using **{total:,}** matching record(s). "
                f"Dominant values by group: {shown}.{suffix}\n\n"
                + (
                    "Caution: this counts distinct IDs within each category, not revenue or profit."
                    if request.get("measure_type") == "distinct_count"
                    else "Caution: these are dataset row counts, not revenue, profit, or unique-order totals."
                )
            )
        if len(rows) == 1:
            row = rows[0]
            value = row.get(counted)
            count = int(row.get(count_column, 0))
            percent = row.get("Percentage", 0)
            if request.get("measure_type") == "distinct_count":
                distinct = request.get("distinct_column")
                scope = filter_text or f" for **{counted} = {value}**"
                return (
                    f"There are **{count:,}** unique {distinct} value(s){scope} "
                    f"({percent:.1f}% of the matching records)."
                )
            return (
                f"**{value}** appears **{count:,}** time(s) in **{counted}**"
                f"{filter_text} ({percent:.1f}% of the matching records).\n\n"
                + (
                    "Caution: this counts distinct IDs within the filtered records."
                    if request.get("measure_type") == "distinct_count"
                    else "Caution: this is a dataset row count unless the question explicitly asks for a unique count."
                )
            )
        leader = max(rows, key=lambda row: row.get(count_column, 0))
        trailer = min(rows, key=lambda row: row.get(count_column, 0))
        return (
            f"Calculated value counts for **{counted}**{filter_text}, using "
            f"**{total:,}** matching record(s). **{leader.get(counted)}** is most frequent "
            f"with **{int(leader.get(count_column, 0)):,}** record(s) "
            f"({leader.get('Percentage', 0):.1f}%). **{trailer.get(counted)}** is least frequent "
            f"with **{int(trailer.get(count_column, 0)):,}** record(s).\n\n"
            + (
                "Caution: this counts distinct IDs within each category, not revenue or profit."
                if request.get("measure_type") == "distinct_count"
                else "Caution: these are dataset row counts unless the question explicitly asks for a unique count."
            )
        )
    if plan.tool_name == "calculate_filtered_aggregate" and isinstance(
        result.data, dict
    ):
        data = result.data
        aggregation_label = (
            "total" if data["aggregation"] == "sum" else data["aggregation"]
        )
        filters = data.get("filters") or []
        filter_label = (
            ", ".join(
                f"{item.get('column')} = **{item.get('value')}**"
                for item in filters
                if isinstance(item, dict)
            )
            if len(filters) > 1
            else f"{data.get('category_column')} = **{data['category_value']}**"
        )
        if data.get("values"):
            statements = []
            for row in data["values"]:
                statements.append(
                    f"{aggregation_label.capitalize()} "
                    f"{_display_column_name(row['value_column'])}: "
                    f"**{format_metric(row['value'], row['value_column'])}**"
                )
            return (
                f"Applied filter: {filter_label}. "
                + "; ".join(statements)
                + "."
            )
        return (
            f"Applied filter: {filter_label}. The {aggregation_label} "
            f"{_display_column_name(data['value_column'])} is "
            f"**{format_metric(data['value'], data['value_column'])}**."
        )
    if plan.tool_name == "compare_category_values" and isinstance(result.data, dict):
        data = result.data
        first = data["first_value"]
        second = data["second_value"]
        first_total = data["first_total"]
        second_total = data["second_total"]
        difference = data["absolute_difference"]
        if data["higher_value"] is None:
            comparison = f"**{first} and {second} are equal**"
        else:
            lower = second if data["higher_value"] == first else first
            comparison = (
                f"**{data['higher_value']} exceeds {lower} by "
                f"{format_metric(difference, data['value_column'])}**"
            )
        percentage = data.get("percentage_difference")
        percentage_text = (
            f", equivalent to {percentage:,.2f}% of {second}'s value"
            if percentage is not None
            else ""
        )
        return (
            f"{first}: **{format_metric(first_total, data['value_column'])}**; "
            f"{second}: **{format_metric(second_total, data['value_column'])}**. "
            f"{comparison}{percentage_text}."
        )
    value_columns = plan.arguments.get("value_columns") or []
    if (
        plan.tool_name == "group_and_aggregate"
        and value_columns
        and isinstance(result.data, list)
        and result.data
    ):
        group = plan.arguments["group_by"]
        statements = []
        for metric in value_columns:
            rows = [
                row for row in result.data if isinstance(row.get(metric), (int, float))
            ]
            if rows:
                leader = max(rows, key=lambda row: row[metric])
                statements.append(
                    f"{leader[group]} has the highest {_display_column_name(metric)}: "
                    f"{format_metric(leader[metric], metric)}."
                )
        return " ".join(statements)
    if (
        plan.tool_name == "group_and_aggregate"
        and plan.arguments.get("include_percentage")
        and isinstance(result.data, list)
        and result.data
    ):
        group = plan.arguments["group_by"]
        metric = plan.arguments["value_column"]
        focus_value = plan.arguments.get("focus_value")
        rows = result.data
        if focus_value is not None:
            focused = [
                row for row in result.data
                if str(row.get(group)).casefold() == str(focus_value).casefold()
            ]
            rows = focused or result.data
        details = []
        for row in rows[:8]:
            details.append(
                f"**{row[group]}**: **{format_metric(row[metric], metric)}** "
                f"({float(row.get('PercentageOfTotal', 0)):.1f}% of total)"
            )
        remaining = len(rows) - len(details)
        suffix = f" Plus {remaining} more group(s)." if remaining else ""
        filter_text = (
            f" within {plan.arguments['filter_column']} = {plan.arguments['filter_value']}"
            if plan.arguments.get("filter_column")
            else ""
        )
        return (
            f"Calculated each {group}'s share of total {_display_column_name(metric)}{filter_text}: "
            + "; ".join(details)
            + "."
            + suffix
        )
    if plan.tool_name == "calculate_period_over_period" and isinstance(result.data, list) and result.data:
        metric = plan.arguments["value_column"]
        frequency = plan.arguments.get("frequency", "period")
        comparable = [
            row for row in result.data
            if isinstance(row.get("PercentageChange"), (int, float))
        ]
        if not comparable:
            return (
                f"Calculated {frequency}ly {_display_column_name(metric)} changes, "
                "but no comparable previous-period values were available."
            )
        latest = comparable[-1]
        largest_increase = max(comparable, key=lambda row: row["PercentageChange"])
        largest_decline = min(comparable, key=lambda row: row["PercentageChange"])
        period_col = plan.arguments["date_column"]
        return (
            f"Calculated {frequency}ly {_display_column_name(metric)} change versus the previous period. "
            f"Latest comparable period {latest[period_col]} changed by "
            f"**{format_metric(latest['AbsoluteChange'], metric)}** "
            f"({latest['PercentageChange']:.1f}%). "
            f"Largest increase: {largest_increase[period_col]} ({largest_increase['PercentageChange']:.1f}%). "
            f"Largest decline: {largest_decline[period_col]} ({largest_decline['PercentageChange']:.1f}%)."
        )
    if plan.tool_name == "calculate_grouped_extrema" and isinstance(result.data, list):
        if not result.data:
            return "No grouped winners were found."
        primary = plan.arguments["primary_group_column"]
        secondary = plan.arguments["secondary_group_column"]
        metric = plan.arguments["metric_column"]
        extremum = plan.arguments.get("extremum", "max")
        leaders = result.data[:3]
        details = [
            f"**{row[primary]}**: **{row[secondary]}** at "
            f"**{format_metric(row[metric], metric)}**"
            + (" (tie)" if row.get("Tie") else "")
            for row in leaders
        ]
        remaining = len(result.data) - len(leaders)
        suffix = f" Plus {remaining} more winner row(s)." if remaining else ""
        objective = "highest" if extremum == "max" else "lowest"
        return (
            f"Selected the {objective} {_display_column_name(metric)} {secondary} inside each "
            f"{primary} after first aggregating {_display_column_name(metric)} by "
            f"{primary} and {secondary}. "
            + "; ".join(details)
            + "."
            + suffix
            + "\n\nThe table and bar chart show the winning "
            f"{secondary} for each {primary}; tied winners are preserved."
        )
    if (
        plan.tool_name == "group_and_aggregate"
        and plan.arguments.get("limit") == 1
        and isinstance(result.data, list)
        and result.data
    ):
        row = result.data[0]
        group = plan.arguments["group_by"]
        metric = plan.arguments["value_column"]
        objective = "highest" if plan.arguments.get("sort_descending", True) else "lowest"
        filter_text = (
            f' for {plan.arguments["filter_column"]} = {plan.arguments["filter_value"]}'
            if plan.arguments.get("filter_column")
            else ""
        )
        metric_text = (
            "record count"
            if plan.arguments.get("aggregation") == "count" and metric == "Count"
            else f"{plan.arguments.get('aggregation', 'sum')} {metric}"
        )
        return (
            f"**{row[group]}** has the {objective} {metric_text}"
            f"{filter_text}: **{format_metric(row[metric], metric)}**."
        )
    if plan.tool_name == "calculate_time_trend" and isinstance(result.data, list) and result.data:
        date_column = plan.arguments["date_column"]
        metrics = plan.arguments.get("value_columns") or [plan.arguments["value_column"]]
        breakdown_column = plan.arguments.get("breakdown_column")
        if breakdown_column:
            metric = metrics[0]
            rows = [
                row
                for row in result.data
                if date_column in row
                and breakdown_column in row
                and isinstance(row.get(metric), (int, float))
            ]
            if rows:
                aggregation = plan.arguments.get("aggregation", "sum")
                frequency = plan.arguments.get("frequency", "month")
                filter_text = (
                    f" where {plan.arguments['filter_column']} = **{plan.arguments['filter_value']}**"
                    if plan.arguments.get("filter_column")
                    else ""
                )
                total_value = sum(float(row[metric]) for row in rows)
                top_row = max(rows, key=lambda row: row[metric])
                periods = {row[date_column] for row in rows}
                breakdown_values = {row[breakdown_column] for row in rows}
                return (
                    f"{aggregation.title()} {_display_column_name(metric)} by {frequency} and "
                    f"{_display_column_name(breakdown_column)}{filter_text}: "
                    f"**{format_metric(total_value, metric)}** across "
                    f"{len(periods):,} period(s) and {len(breakdown_values):,} "
                    f"{_display_column_name(breakdown_column)} value(s). "
                    f"The table and chart show one verified row per {frequency} and "
                    f"{_display_column_name(breakdown_column)}.\n\n"
                    f"Highest row: **{top_row[breakdown_column]}** in **{top_row[date_column]}** "
                    f"at **{format_metric(top_row[metric], metric)}**."
                )
        if len(metrics) > 1:
            frequency = plan.arguments.get("frequency", "month")
            aggregation = plan.arguments.get("aggregation", "sum")
            filter_text = (
                f" where {plan.arguments['filter_column']} = {plan.arguments['filter_value']}"
                if plan.arguments.get("filter_column")
                else ""
            )
            rows = [
                row
                for row in result.data
                if date_column in row
                and all(isinstance(row.get(metric), (int, float)) for metric in metrics)
            ]
            if rows:
                metric_summaries = []
                peak_summaries = []
                for metric in metrics:
                    display_metric = _display_column_name(metric)
                    overall_value = (
                        sum(float(row[metric]) for row in rows)
                        if aggregation in {"sum", "count"}
                        else sum(float(row[metric]) for row in rows) / len(rows)
                    )
                    first = rows[0]
                    last = rows[-1]
                    peak = max(rows, key=lambda row: row[metric])
                    trough = min(rows, key=lambda row: row[metric])
                    change = float(last[metric]) - float(first[metric])
                    direction = "increased" if change > 0 else "decreased" if change < 0 else "stayed flat"
                    peak_summaries.append(
                        f"Highest {display_metric}: {peak[date_column]} "
                        f"({format_metric(peak[metric], metric)})"
                    )
                    metric_summaries.append(
                        f"{display_metric.title()}: total {format_metric(overall_value, metric)}; "
                        f"{direction} from {format_metric(first[metric], metric)} in {first[date_column]} "
                        f"to {format_metric(last[metric], metric)} in {last[date_column]}; "
                        f"peak {peak[date_column]} ({format_metric(peak[metric], metric)}); "
                        f"lowest {trough[date_column]} ({format_metric(trough[metric], metric)})."
                    )
                chart_label = "dual-line" if len(metrics) == 2 else "multi-line"
                return (
                    f"Compared {', '.join(_display_column_name(metric) for metric in metrics)}"
                    f"{filter_text} across {len(rows):,} {frequency} period(s).\n\n"
                    f"Chart: {chart_label} chart with one verified row per {frequency}.\n\n"
                    f"Key result: {'; '.join(peak_summaries)}.\n\n"
                    "Metric details:\n"
                    + "\n".join(f"- {summary}" for summary in metric_summaries)
                )
        metric = metrics[0]
        rows = [
            row
            for row in result.data
            if date_column in row
            and metric in row
            and isinstance(row.get(metric), (int, float))
        ]
        if rows:
            first = rows[0]
            last = rows[-1]
            peak = max(rows, key=lambda row: row[metric])
            trough = min(rows, key=lambda row: row[metric])
            change = float(last[metric]) - float(first[metric])
            aggregation = plan.arguments.get("aggregation", "sum")
            frequency = plan.arguments.get("frequency", "month")
            display_metric = _display_column_name(metric)
            filter_text = (
                f" where {plan.arguments['filter_column']} = **{plan.arguments['filter_value']}**"
                if plan.arguments.get("filter_column")
                else ""
            )
            overall_value = (
                sum(float(row[metric]) for row in rows)
                if aggregation in {"sum", "count"}
                else sum(float(row[metric]) for row in rows) / len(rows)
            )
            if aggregation == "sum":
                overall_label = f"Overall {display_metric}"
                period_metric_label = f"{frequency}ly sum {display_metric}"
            elif aggregation in {"nunique", "unique count", "distinct_count"}:
                overall_value = sum(float(row[metric]) for row in rows)
                overall_label = f"Total displayed unique {display_metric} counts"
                period_metric_label = f"{frequency}ly unique {display_metric} count"
            else:
                overall_label = f"Overall {aggregation} {display_metric}"
                period_metric_label = f"{frequency}ly {aggregation} {display_metric}"
            if first[metric]:
                change_text = f" ({change / abs(float(first[metric])) * 100:,.1f}% from the first period)"
            else:
                change_text = ""
            direction = "increased" if change > 0 else "decreased" if change < 0 else "stayed flat"
            return (
                f"{overall_label}{filter_text}: **{format_metric(overall_value, metric)}** across "
                f"{len(rows):,} {frequency} period(s). The table and chart show one verified row per {frequency}.\n\n"
                f"The {period_metric_label} {direction} from "
                f"**{format_metric(first[metric], metric)}** in {first[date_column]} "
                f"to **{format_metric(last[metric], metric)}** in {last[date_column]}, a change of "
                f"**{format_metric(change, metric)}**{change_text}.\n\n"
                f"Peak period: **{peak[date_column]}** at **{format_metric(peak[metric], metric)}**. "
                f"Lowest period: **{trough[date_column]}** at **{format_metric(trough[metric], metric)}**.\n\n"
                "This answers the date pattern from verified data. It can show when the metric changed, "
                "but it cannot prove why without comparing related drivers such as region, product, cost, units, or channel."
            )
    if isinstance(result.data, list) and result.data:
        preview = result.data[:5]
        return f"{result.summary}\n\nVerified result preview: {preview}"
    if isinstance(result.data, dict):
        return f"{result.summary}\n\nVerified result: {result.data}"
    return result.summary


def _column_profile_chart_spec(data: dict[str, Any]) -> ChartSpec | None:
    profile = data.get("profile", {})
    column = profile.get("column_name")
    display = profile.get("display_name") or (str(column) if column else "Column")
    chart_type = data.get("chart_type")
    if not chart_type or not column:
        return None
    if chart_type == "histogram":
        return ChartSpec(
            chart_type="histogram",
            x=column,
            title=f"{display} Distribution",
        )
    if profile.get("semantic_type") == "datetime":
        return ChartSpec(
            chart_type="bar",
            x="Period",
            y="Count",
            title=f"{display} Records Over Time",
        )
    if profile.get("semantic_type") == "boolean":
        return ChartSpec(
            chart_type="bar",
            x="Value",
            y="Count",
            title=f"{display} Counts",
        )
    return ChartSpec(
        chart_type="bar",
        x=column,
        y="Count",
        title=f"{display} Value Counts",
    )


def _previous_analysis_item(
    history: list[dict[str, Any]] | None,
    require_chart: bool = False,
) -> dict[str, Any] | None:
    for item in reversed(history or []):
        if require_chart and item.get("chart_spec"):
            return item
        if not require_chart and (
            item.get("result") or item.get("chart_spec") or item.get("answer")
        ):
            return item
    return None


def _format_summary_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _summarize_previous_analysis(item: dict[str, Any], line_count: int) -> str:
    spec = item.get("chart_spec") or {}
    data = item.get("chart_data") or []
    result = item.get("result") or {}
    if not data and isinstance(result.get("data"), list):
        data = result["data"]
    title = spec.get("title") or "The previous chart"
    lines: list[str] = []

    if data and isinstance(data[0], dict):
        x_column = spec.get("x")
        y_column = spec.get("y")
        usable = (
            [
                row
                for row in data
                if x_column in row
                and y_column in row
                and isinstance(row.get(y_column), (int, float))
            ]
            if x_column and y_column
            else []
        )
        if usable:
            highest = max(usable, key=lambda row: row[y_column])
            lowest = min(usable, key=lambda row: row[y_column])
            lines.append(
                f"{highest[x_column]} has the highest {y_column} at "
                f"{_format_summary_value(highest[y_column])}."
            )
            if len(usable) > 1:
                lines.append(
                    f"{lowest[x_column]} has the lowest {y_column} at "
                    f"{_format_summary_value(lowest[y_column])}."
                )
            if len(usable) > 2:
                total = sum(float(row[y_column]) for row in usable)
                lines.append(
                    f"The chart compares {len(usable)} values with a combined {y_column} of "
                    f"{_format_summary_value(total)}."
                )
        else:
            missing_rows = [
                row
                for row in data
                if "missing_count" in row
                and isinstance(row.get("missing_count"), (int, float))
            ]
            outlier_rows = [
                row
                for row in data
                if "outlier_count" in row
                and isinstance(row.get("outlier_count"), (int, float))
            ]
            if missing_rows:
                affected = [row for row in missing_rows if row["missing_count"] > 0]
                total = sum(row["missing_count"] for row in affected)
                lines.append(
                    f"{len(affected)} column(s) contain {total:,} missing value(s) in total."
                )
                if affected:
                    worst = max(affected, key=lambda row: row["missing_count"])
                    lines.append(
                        f"{worst.get('column', 'The most affected column')} has the most missing values "
                        f"({worst['missing_count']:,})."
                    )
            elif outlier_rows:
                affected = [row for row in outlier_rows if row["outlier_count"] > 0]
                lines.append(
                    f"{len(affected)} numeric column(s) contain potential IQR outliers."
                )
                if affected:
                    worst = max(affected, key=lambda row: row["outlier_count"])
                    lines.append(
                        f"{worst.get('column', 'The most affected column')} has "
                        f"{worst['outlier_count']:,} potential outlier(s)."
                    )
            else:
                lines.append(f"{title} contains {len(data)} verified result record(s).")
    elif isinstance(result.get("data"), dict):
        values = result["data"]
        if "correlation" in values:
            strength = abs(float(values["correlation"]))
            description = (
                "strong"
                if strength >= 0.7
                else "moderate" if strength >= 0.4 else "weak"
            )
            direction = "positive" if values["correlation"] > 0 else "negative"
            lines.append(
                f"{values.get('first_column')} and {values.get('second_column')} have a "
                f"{description} {direction} correlation ({values['correlation']:.3f})."
            )
        elif "duplicate_rows" in values:
            lines.append(
                f"The dataset contains {values['duplicate_rows']:,} duplicate row(s), "
                f"representing {values.get('duplicate_percentage', 0):.2f}% of records."
            )
        elif "rows" in values and "columns" in values:
            lines.append(
                f"The dataset contains {values['rows']:,} rows and {values['columns']:,} columns."
            )
    if not lines:
        previous_answer = str(item.get("answer", "")).strip()
        lines.append(
            previous_answer or f"{title} is based on the previous verified analysis."
        )
    while len(lines) < line_count:
        filter_column = spec.get("filter_column")
        if filter_column:
            lines.append(
                f"The chart is filtered to {filter_column} = {spec.get('filter_value')}."
            )
        else:
            lines.append(
                f"The chart uses the verified {spec.get('aggregation') or 'raw'} values shown above."
            )
    return "\n".join(
        f"{index + 1}. {line}" for index, line in enumerate(lines[:line_count])
    )


def _summarize_dataset(
    dataframe: pd.DataFrame,
    profile: DatasetProfile,
    line_count: int,
) -> str:
    observations = generate_eda_summary(dataframe, profile).observations
    lines = list(observations)
    if profile.numeric_columns:
        ranked = sorted(
            (
                column
                for column in profile.columns
                if column.kind == "numeric" and column.mean is not None
            ),
            key=lambda column: abs(column.skewness or 0),
            reverse=True,
        )
        if ranked:
            column = ranked[0]
            lines.append(
                f"{column.name} has an average of {_format_summary_value(column.mean)} "
                f"and ranges from {_format_summary_value(column.minimum)} "
                f"to {_format_summary_value(column.maximum)}."
            )
    if profile.categorical_columns:
        column_name = profile.categorical_columns[0]
        counts = dataframe[column_name].value_counts(dropna=False)
        if not counts.empty:
            lines.append(
                f"The most common {column_name} is {counts.index[0]} "
                f"with {int(counts.iloc[0]):,} record(s)."
            )
    return "\n".join(
        f"{index + 1}. {line}" for index, line in enumerate(lines[:line_count])
    )


def run_agent(
    question: str,
    dataframe: pd.DataFrame,
    profile: DatasetProfile,
    settings: Settings,
    model_name: str,
    dataset_name: str,
    history: list[dict[str, Any]] | None = None,
    ollama_online: bool = False,
) -> AgentResponse:
    """Plan, validate, execute, chart, and explain one user question."""
    total_started = perf_counter()
    interpretation_started = perf_counter()
    if list(dataframe.columns) != [column.name for column in profile.columns]:
        profile = profile_dataset(dataframe)
    plan = deterministic_plan(question, profile, history, dataframe)
    interpretation_seconds = perf_counter() - interpretation_started
    if ollama_online and (plan.clarification or not plan.tool_name):
        context = structured_dataset_context(
            dataset_name,
            profile,
            dataframe.head(settings.max_sample_rows_for_llm).to_dict(orient="records"),
        )
        try:
            llm_plan, elapsed = plan_with_ollama(
                question, context, list(TOOL_REGISTRY), settings, model_name
            )
            if llm_plan.clarification:
                plan = llm_plan
                interpretation_seconds = elapsed
            elif llm_plan.tool_name in TOOL_REGISTRY:
                validate_tool_arguments(llm_plan.tool_name, llm_plan.arguments)
                plan = llm_plan
                interpretation_seconds = elapsed
        except Exception:
            LOGGER.exception("Ollama planning failed for model %s", model_name)
    if plan.clarification or not plan.tool_name:
        plan.safe_code = None
        return AgentResponse(
            question=question,
            answer=plan.clarification or "Please clarify the requested analysis.",
            plan=plan,
            total_seconds=perf_counter() - total_started,
            interpretation_seconds=interpretation_seconds,
        )
    if plan.tool_name == "summarize_previous":
        previous_item = _previous_analysis_item(
            history,
            require_chart=plan.arguments.get("target") == "chart",
        )
        if previous_item is None:
            return AgentResponse(
                question=question,
                answer="There is no previous chart in this conversation to summarize.",
                plan=AgentPlan(
                    tool_name="",
                    clarification="There is no previous chart in this conversation to summarize.",
                ),
                total_seconds=perf_counter() - total_started,
                interpretation_seconds=interpretation_seconds,
            )
        return AgentResponse(
            question=question,
            answer=_summarize_previous_analysis(
                previous_item,
                int(plan.arguments.get("line_count", 2)),
            ),
            plan=plan,
            chart_spec=None,
            chart_data=[],
            suggested_questions=[
                "Explain the main difference.",
                "Summarize it in one line.",
            ],
            total_seconds=perf_counter() - total_started,
            interpretation_seconds=interpretation_seconds,
        )
    if plan.tool_name == "summarize_dataset":
        return AgentResponse(
            question=question,
            answer=_summarize_dataset(
                dataframe,
                profile,
                int(plan.arguments.get("line_count", 5)),
            ),
            plan=plan,
            chart_spec=None,
            chart_data=[],
            suggested_questions=[
                "Explain the biggest data-quality issue.",
                "Show the most important numeric trend.",
            ],
            total_seconds=perf_counter() - total_started,
            interpretation_seconds=interpretation_seconds,
        )
    plan.safe_code = plan.safe_code or _safe_code(plan)
    try:
        result = execute_tool(dataframe, plan.tool_name, plan.arguments)
    except (TypeError, ValueError) as exc:
        failed = ToolResult(
            tool_name=plan.tool_name,
            success=False,
            summary=str(exc),
            warnings=["The requested operation was not executed."],
        )
        return AgentResponse(
            question=question,
            answer=str(exc),
            plan=plan,
            result=failed,
            total_seconds=perf_counter() - total_started,
            interpretation_seconds=interpretation_seconds,
        )
    if plan.tool_name == "profile_column" and isinstance(result.data, dict):
        plan.chart_spec = _column_profile_chart_spec(result.data)
    chart_data: list[dict[str, Any]] = []
    if plan.chart_spec:
        if plan.tool_name in {"analyze_categorical_value_counts", "profile_column"} and isinstance(result.data, dict):
            chart_data = [
                row for row in result.data.get("chart_rows", [])
                if isinstance(row, dict)
            ]
        elif plan.tool_name == "calculate_grouped_extrema" and isinstance(result.data, list):
            primary = plan.arguments["primary_group_column"]
            secondary = plan.arguments["secondary_group_column"]
            metric = plan.arguments["metric_column"]
            chart_data = [
                {
                    primary: row.get(primary),
                    secondary: row.get(secondary),
                    metric: row.get(metric),
                    "Tie": row.get("Tie", False),
                }
                for row in result.data
                if isinstance(row, dict)
            ]
        elif (
            (
                plan.tool_name == "group_and_aggregate"
                and plan.arguments.get("include_percentage")
            )
            or plan.tool_name == "calculate_period_over_period"
        ) and isinstance(result.data, list):
            chart_data = [row for row in result.data if isinstance(row, dict)]
        elif plan.tool_name == "analyze_advanced_request" and result.data is not None:
            if isinstance(result.data, list):
                chart_data = [row for row in result.data if isinstance(row, dict)]
            elif isinstance(result.data, dict):
                rows = (
                    result.data.get("table_rows")
                    or result.data.get("rows")
                    or result.data.get("winners")
                    or []
                )
                chart_data = [row for row in rows if isinstance(row, dict)]
        else:
            _, chart_result = create_chart(dataframe, plan.chart_spec)
            chart_data = chart_result.data
    explanation_started = perf_counter()
    answer = _deterministic_explanation(plan, result)
    deterministic_only = (
        plan.tool_name == "group_and_aggregate"
    )
    if (
        ollama_online
        and not deterministic_only
        and not plan.arguments.get("value_columns")
        and plan.tool_name
        not in {
            "compare_category_values",
            "calculate_filtered_aggregate",
            "calculate_scalar_aggregate",
            "calculate_summary_statistics",
            "list_distinct_values",
            "count_distinct_values",
            "analyze_high_volume_low_outcome",
            "compare_grouped_to_benchmark",
            "profile_column",
        }
    ):
        try:
            answer, explanation_seconds = explain_with_ollama(
                question, result, settings, model_name
            )
        except Exception:
            LOGGER.exception("Ollama explanation failed for model %s", model_name)
            explanation_seconds = perf_counter() - explanation_started
    else:
        explanation_seconds = perf_counter() - explanation_started
    narrative = _chat_narrative_from_result(plan, result)
    if narrative:
        answer = render_chat_narrative_markdown(narrative)
    else:
        answer = sanitize_markdown(answer)
    return AgentResponse(
        question=question,
        answer=answer,
        narrative=narrative,
        plan=plan,
        result=result,
        chart_spec=plan.chart_spec,
        chart_data=chart_data,
        suggested_questions=[
            "Which columns contain missing values?",
            "Find potential outliers.",
            "Show a correlation matrix.",
        ],
        total_seconds=perf_counter() - total_started,
        interpretation_seconds=interpretation_seconds,
        tool_seconds=result.execution_seconds,
        explanation_seconds=explanation_seconds,
    )
