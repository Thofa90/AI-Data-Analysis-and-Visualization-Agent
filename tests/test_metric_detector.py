"""Tests for automatic and manual key metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from services.dataset_profiler import profile_dataset
from services.metric_detector import calculate_manual_metric, detect_key_metrics


def test_automatic_metrics_exclude_identifier_columns() -> None:
    dataframe = pd.DataFrame({
        "TransactionID": [1001, 1002, 1003],
        "Revenue": [100.0, 150.0, 50.0],
        "Satisfaction Score": [4.0, 5.0, 3.0],
    })
    profile = profile_dataset(dataframe)

    metrics = detect_key_metrics(dataframe, profile)

    assert all(metric.column != "TransactionID" for metric in metrics)
    revenue = next(metric for metric in metrics if metric.column == "Revenue")
    assert revenue.aggregation == "sum"
    assert revenue.value == 300


def test_generic_metrics_fill_when_no_numeric_measure_exists() -> None:
    dataframe = pd.DataFrame({"Category": ["A", "B", "C"]})
    profile = profile_dataset(dataframe)

    metrics = detect_key_metrics(dataframe, profile)

    assert metrics[0].label == "Total Records"
    assert metrics[0].value == 3
    assert len(metrics) == 4


def test_manual_metric_uses_allowlisted_aggregation() -> None:
    dataframe = pd.DataFrame({"Revenue": [10.0, 20.0, 30.0]})

    metric = calculate_manual_metric(dataframe, "Revenue", "median", label="Typical Revenue")

    assert metric.label == "Typical Revenue"
    assert metric.value == 20
    assert metric.is_automatic is False


def test_manual_metric_period_comparison_uses_real_periods() -> None:
    dataframe = pd.DataFrame({
        "Order Date": ["2025-01-02", "2025-01-10", "2025-02-03", "2025-02-18"],
        "Revenue": [10.0, 20.0, 30.0, 30.0],
    })

    metric = calculate_manual_metric(
        dataframe,
        "Revenue",
        "sum",
        date_column="Order Date",
        comparison_period="month",
    )

    assert metric.value == 60
    assert metric.comparison_percentage == 100
    assert metric.comparison_label == "vs. 2025-01"


def test_manual_metric_rejects_invalid_column_or_text_sum() -> None:
    dataframe = pd.DataFrame({"Category": ["A", "B"]})

    with pytest.raises(ValueError, match="was not found"):
        calculate_manual_metric(dataframe, "Missing", "count")
    with pytest.raises(ValueError, match="requires a numeric column"):
        calculate_manual_metric(dataframe, "Category", "sum")
