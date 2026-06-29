"""Dataset-aware guidance for natural-language analytical questions."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from services.profile_models import DatasetProfile


class ColumnMetadata(BaseModel):
    column_name: str
    display_name: str
    pandas_dtype: str
    semantic_type: str
    is_date: bool = False
    is_numeric: bool = False
    is_categorical: bool = False
    is_identifier: bool = False
    unique_count: int = 0
    example_values: list[str] = Field(default_factory=list)


class DatasetQueryGuide(BaseModel):
    fingerprint: str
    preferred_date_column: str | None = None
    date_columns: list[ColumnMetadata] = Field(default_factory=list)
    numeric_columns: list[ColumnMetadata] = Field(default_factory=list)
    categorical_columns: list[ColumnMetadata] = Field(default_factory=list)
    identifier_columns: list[ColumnMetadata] = Field(default_factory=list)


class QueryResolutionIssue(BaseModel):
    issue_type: Literal[
        "missing_date_column",
        "ambiguous_date_column",
        "ambiguous_filter_value",
        "unknown_column",
        "unknown_value",
        "missing_metric",
        "missing_grouping",
    ]
    message: str
    options: list[str] = Field(default_factory=list)


class QueryResolutionResult(BaseModel):
    status: Literal["ready", "needs_clarification", "invalid"]
    issues: list[QueryResolutionIssue] = Field(default_factory=list)
    suggested_query: str | None = None


def dataset_fingerprint(dataframe: pd.DataFrame) -> str:
    columns = "|".join(f"{column}:{dataframe[column].dtype}" for column in dataframe.columns)
    payload = f"{len(dataframe)}::{columns}".encode("utf-8")
    return sha256(payload).hexdigest()[:16]


def _display_name(column_name: str) -> str:
    import re

    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(column_name).replace("_", " "))
    return " ".join(spaced.split())


def _examples(dataframe: pd.DataFrame, column: str, limit: int = 5) -> list[str]:
    if column not in dataframe.columns:
        return []
    values = dataframe[column].dropna().astype(str).drop_duplicates().head(limit)
    return [str(value) for value in values.tolist()]


def build_dataset_query_guide(
    dataframe: pd.DataFrame,
    profile: DatasetProfile,
    preferred_date_column: str | None = None,
) -> DatasetQueryGuide:
    guide = DatasetQueryGuide(
        fingerprint=dataset_fingerprint(dataframe),
        preferred_date_column=preferred_date_column,
    )
    for column in profile.columns:
        metadata = ColumnMetadata(
            column_name=column.name,
            display_name=_display_name(column.name),
            pandas_dtype=column.pandas_dtype,
            semantic_type=column.kind,
            is_date=column.kind == "datetime",
            is_numeric=column.kind == "numeric",
            is_categorical=column.kind == "categorical",
            is_identifier=column.kind == "identifier",
            unique_count=column.unique_count,
            example_values=_examples(dataframe, column.name, limit=5),
        )
        if metadata.is_date:
            guide.date_columns.append(metadata)
        elif metadata.is_numeric:
            guide.numeric_columns.append(metadata)
        elif metadata.is_categorical:
            guide.categorical_columns.append(metadata)
        elif metadata.is_identifier:
            guide.identifier_columns.append(metadata)
    if guide.preferred_date_column not in {column.column_name for column in guide.date_columns}:
        guide.preferred_date_column = None
    return guide


def build_query_examples(guide: DatasetQueryGuide, limit: int = 9) -> list[str]:
    examples: list[str] = []
    dates = guide.date_columns
    metrics = guide.numeric_columns
    categories = [column for column in guide.categorical_columns if column.example_values]
    if dates and metrics:
        metric = _pick(metrics, "Profit") or metrics[0]
        for date in dates[:2]:
            examples.append(f"Using {date.display_name}, show monthly {metric.display_name} for 2017.")
    if metrics and categories:
        metric = _pick(metrics, "Sales") or metrics[0]
        category = _pick(categories, "Region") or categories[0]
        value = category.example_values[0]
        examples.append(f"Show total {metric.display_name} where {category.display_name} = {value}.")
        examples.append(f"Show total {metric.display_name} by {category.display_name}.")
    if categories:
        counted = _pick(categories, "Ship Mode") or categories[0]
        examples.append(f"Show value counts of {counted.display_name}.")
        if len(categories) > 1:
            filter_column = _pick(categories, "Region") or categories[0]
            if filter_column.column_name != counted.column_name and filter_column.example_values:
                examples.append(
                    f"Show {counted.display_name} counts where {filter_column.display_name} = {filter_column.example_values[0]}."
                )
    if metrics and categories:
        metric = _pick(metrics, "Sales") or metrics[0]
        category = _pick(categories, "Region") or categories[0]
        examples.append(f"Show each {category.display_name}'s share of total {metric.display_name}.")
    if dates and metrics:
        date = dates[0]
        metric = _pick(metrics, "Sales") or metrics[0]
        examples.append(
            f"Using {date.display_name}, show monthly {metric.display_name} change compared with the previous month."
        )
    return _dedupe(examples)[:limit]


def build_hint_chips(guide: DatasetQueryGuide) -> dict[str, list[str]]:
    return {
        "Date field": [column.display_name for column in guide.date_columns[:4]],
        "Metric": [column.display_name for column in guide.numeric_columns[:6]],
        "Filter": [column.display_name for column in guide.categorical_columns[:6]],
        "Group by": [column.display_name for column in guide.categorical_columns[:6]],
        "Calculation": ["Total", "Average", "Count", "Frequency", "Share %"],
        "Chart": ["Monthly", "Yearly", "Bar", "Line", "Pie", "Scatter"],
    }


def suggest_columns(partial: str, guide: DatasetQueryGuide, limit: int = 6) -> list[str]:
    needle = _normalize(partial)
    if not needle:
        return []
    columns = [
        *guide.date_columns,
        *guide.numeric_columns,
        *guide.categorical_columns,
        *guide.identifier_columns,
    ]
    scored: list[tuple[int, str]] = []
    aliases = {
        "revenue": {"sales"},
        "sales": {"sales", "revenue"},
        "profit": {"profit"},
        "order": {"orderdate", "orderid"},
        "ship": {"shipdate", "shipmode"},
    }
    for column in columns:
        normalized = _normalize(column.display_name)
        score = 0
        if normalized.startswith(needle):
            score = 3
        elif needle in normalized:
            score = 2
        elif any(alias.startswith(needle) or needle in alias for alias in aliases.get(normalized, set())):
            score = 1
        if score:
            scored.append((score, column.display_name))
    return [name for _, name in sorted(scored, reverse=True)[:limit]]


def _pick(columns: list[ColumnMetadata], name: str) -> ColumnMetadata | None:
    normalized = _normalize(name)
    return next((column for column in columns if _normalize(column.display_name) == normalized), None)


def _normalize(value: Any) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
