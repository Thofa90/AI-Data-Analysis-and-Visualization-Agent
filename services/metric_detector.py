"""Automatic and manual dataset metric calculation."""

from __future__ import annotations

import re
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from services.profile_models import DatasetProfile


Aggregation = Literal["sum", "mean", "median", "minimum", "maximum", "count", "unique count"]
NumberFormat = Literal["number", "currency", "percentage"]

MEASURE_HINTS = (
    "revenue", "sales", "profit", "income", "salary", "cost", "price", "amount",
    "value", "quantity", "units", "tenure", "duration", "age", "score", "rating",
    "efficiency", "output", "energy", "power", "distance", "weight",
)


class MetricResult(BaseModel):
    """A verified metric ready for display."""

    label: str
    column: str | None = None
    aggregation: Aggregation
    value: float | int
    number_format: NumberFormat = "number"
    is_automatic: bool = True
    comparison_percentage: float | None = None
    comparison_label: str | None = None


def _friendly_name(column: str) -> str:
    return re.sub(r"[_\-]+", " ", column).strip().title()


def _aggregate(series: pd.Series, aggregation: Aggregation) -> float | int:
    operations = {
        "sum": series.sum,
        "mean": series.mean,
        "median": series.median,
        "minimum": series.min,
        "maximum": series.max,
        "count": series.count,
        "unique count": series.nunique,
    }
    if aggregation not in operations:
        raise ValueError(f'Unsupported aggregation: "{aggregation}".')
    value = operations[aggregation]()
    return int(value) if isinstance(value, int) or float(value).is_integer() else float(value)


def calculate_manual_metric(
    dataframe: pd.DataFrame,
    column: str,
    aggregation: Aggregation,
    label: str | None = None,
    number_format: NumberFormat = "number",
    date_column: str | None = None,
    comparison_period: Literal["month", "quarter", "year"] | None = None,
) -> MetricResult:
    """Calculate one allowlisted metric against a validated column."""
    if column not in dataframe.columns:
        raise ValueError(f'Column "{column}" was not found.')
    series = dataframe[column]
    if aggregation not in {"count", "unique count"} and not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f'Aggregation "{aggregation}" requires a numeric column.')
    default_label = f"{aggregation.title()} {_friendly_name(column)}"
    metric = MetricResult(
        label=(label or default_label).strip(),
        column=column,
        aggregation=aggregation,
        value=_aggregate(series, aggregation),
        number_format=number_format,
        is_automatic=False,
    )
    if date_column and comparison_period:
        if date_column not in dataframe.columns:
            raise ValueError(f'Date column "{date_column}" was not found.')
        dates = pd.to_datetime(dataframe[date_column], errors="coerce")
        frequency = {"month": "M", "quarter": "Q", "year": "Y"}[comparison_period]
        periods = dates.dt.to_period(frequency)
        valid_periods = sorted(periods.dropna().unique())
        if len(valid_periods) < 2:
            raise ValueError(f"At least two comparable {comparison_period}s are required.")
        previous_period, latest_period = valid_periods[-2:]
        previous_value = _aggregate(dataframe.loc[periods == previous_period, column], aggregation)
        latest_value = _aggregate(dataframe.loc[periods == latest_period, column], aggregation)
        metric.value = latest_value
        metric.comparison_percentage = (
            None if previous_value == 0 else (latest_value - previous_value) / abs(previous_value) * 100
        )
        metric.comparison_label = f"vs. {previous_period}"
    return metric


def detect_key_metrics(
    dataframe: pd.DataFrame,
    profile: DatasetProfile,
    limit: int = 4,
) -> list[MetricResult]:
    """Select useful metrics while excluding identifiers and constants."""
    eligible = [
        column
        for column in profile.numeric_columns
        if column not in profile.id_columns and column not in profile.constant_columns
    ]

    def priority(column: str) -> tuple[int, int]:
        lowered = column.lower()
        hint_index = next((index for index, hint in enumerate(MEASURE_HINTS) if hint in lowered), 999)
        return hint_index, list(dataframe.columns).index(column)

    metrics: list[MetricResult] = []
    for column in sorted(eligible, key=priority)[:limit]:
        lowered = column.lower()
        aggregation: Aggregation = "sum" if any(
            hint in lowered for hint in ("revenue", "sales", "profit", "income", "cost", "amount", "quantity", "units")
        ) else "mean"
        prefix = "Total" if aggregation == "sum" else "Average"
        number_format: NumberFormat = "percentage" if any(
            hint in lowered for hint in ("rate", "percent", "ratio", "efficiency")
        ) else "currency" if any(
            hint in lowered for hint in ("revenue", "sales", "profit", "income", "salary", "cost", "price", "amount")
        ) else "number"
        metrics.append(MetricResult(
            label=f"{prefix} {_friendly_name(column)}",
            column=column,
            aggregation=aggregation,
            value=_aggregate(dataframe[column], aggregation),
            number_format=number_format,
        ))

    fallbacks = [
        MetricResult(label="Total Records", aggregation="count", value=len(dataframe)),
        MetricResult(label="Missing Values", aggregation="count", value=profile.total_missing),
        MetricResult(label="Duplicate Rows", aggregation="count", value=profile.duplicate_rows),
        MetricResult(label="Numeric Features", aggregation="count", value=len(profile.numeric_columns)),
    ]
    for fallback in fallbacks:
        if len(metrics) >= limit:
            break
        metrics.append(fallback)
    return metrics
