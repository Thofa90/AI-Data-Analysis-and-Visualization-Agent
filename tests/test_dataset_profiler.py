"""Tests for deterministic Phase 2 dataset profiling."""

from __future__ import annotations

import pandas as pd

from services.dataset_profiler import profile_dataset


def test_profile_detects_schema_quality_and_dates() -> None:
    dataframe = pd.DataFrame({
        "CustomerID": [1, 2, 3, 4, 5, 5],
        "Order Date": [
            "2025-01-01", "2025-01-02", "2025-01-03",
            "2025-01-04", "2025-01-05", "2025-01-05",
        ],
        "Region": ["West", "East", "West", None, "North", "North"],
        "Sales": [10.0, 12.0, 11.0, 13.0, 1000.0, 1000.0],
        "Constant": ["same"] * 6,
    })
    dataframe.loc[5] = dataframe.loc[3]

    profile = profile_dataset(dataframe)

    assert "CustomerID" in profile.id_columns
    assert "CustomerID" not in profile.numeric_columns
    assert "Order Date" in profile.datetime_columns
    assert "Region" in profile.columns_with_missing
    assert "Constant" in profile.constant_columns
    assert profile.duplicate_rows == 1
    assert "Sales" in profile.outlier_columns
    assert profile.quality.score < 100
    assert profile.quality.issues


def test_profile_detects_mixed_numeric_text() -> None:
    dataframe = pd.DataFrame({"Amount": ["10", "20", "unknown", "40"], "Category": ["A", "B", "A", "B"]})

    profile = profile_dataset(dataframe)

    assert "Amount" in profile.potential_type_problems
    assert any(issue.category == "Mixed data types" for issue in profile.quality.issues)


def test_clean_dataset_receives_excellent_score() -> None:
    dataframe = pd.DataFrame({
        "Category": ["A", "A", "B", "B"],
        "Value": [10.0, 11.0, 12.0, 13.0],
    })

    profile = profile_dataset(dataframe)

    assert profile.quality.score == 100
    assert profile.quality.rating == "Excellent"
