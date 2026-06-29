"""Tests for cleaning previews, application, and source preservation."""

from __future__ import annotations

import pandas as pd

from services.cleaning_service import apply_cleaning, preview_cleaning


def test_remove_duplicates_and_fill_missing_without_mutation() -> None:
    dataframe = pd.DataFrame({"Value": [1.0, None, 1.0], "Category": [" A ", "B", " A "]})
    original = dataframe.copy(deep=True)
    fill_plan = preview_cleaning(dataframe, "fill_numeric_mean", {"column": "Value"})
    cleaned = apply_cleaning(dataframe, fill_plan)
    assert cleaned["Value"].isna().sum() == 0
    pd.testing.assert_frame_equal(dataframe, original)
    trim_plan = preview_cleaning(cleaned, "trim_whitespace", {"column": "Category"})
    trimmed = apply_cleaning(cleaned, trim_plan)
    assert trimmed["Category"].iloc[0] == "A"


def test_remove_columns_and_rename() -> None:
    dataframe = pd.DataFrame({"A": [1], "B": [2]})
    removed = apply_cleaning(dataframe, preview_cleaning(dataframe, "remove_columns", {"columns": ["B"]}))
    assert list(removed.columns) == ["A"]
    renamed = apply_cleaning(dataframe, preview_cleaning(dataframe, "rename_column", {"old": "A", "new": "Value"}))
    assert "Value" in renamed.columns
