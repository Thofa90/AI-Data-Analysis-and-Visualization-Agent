"""Phase 1 tests for dataset file loading."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from services.file_loader import FileLoadError, get_excel_sheet_names, load_dataset


def test_load_csv() -> None:
    loaded = load_dataset("sample.csv", b"name,value\nA,10\nB,20\n")

    assert loaded.dataframe.shape == (2, 2)
    assert loaded.dataframe["value"].sum() == 30
    assert loaded.sheet_name is None


def test_load_excel_selected_sheet() -> None:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"value": [1]}).to_excel(writer, sheet_name="First", index=False)
        pd.DataFrame({"value": [2]}).to_excel(writer, sheet_name="Second", index=False)
    data = buffer.getvalue()

    assert get_excel_sheet_names("book.xlsx", data) == ["First", "Second"]
    loaded = load_dataset("book.xlsx", data, sheet_name="Second")
    assert loaded.sheet_name == "Second"
    assert loaded.dataframe.iloc[0, 0] == 2


@pytest.mark.parametrize(
    ("filename", "data", "message"),
    [
        ("notes.txt", b"hello", "Unsupported file type"),
        ("empty.csv", b"", "empty"),
        ("headers.csv", b"name,value\n", "no data rows"),
        ("duplicate.csv", b"name,name\nA,B\n", "Duplicate column names"),
        ("missing-header.csv", b"name,\nA,B\n", "invalid column headers"),
    ],
)
def test_invalid_uploads(filename: str, data: bytes, message: str) -> None:
    with pytest.raises(FileLoadError, match=message):
        load_dataset(filename, data)


def test_missing_excel_sheet() -> None:
    buffer = io.BytesIO()
    pd.DataFrame({"value": [1]}).to_excel(buffer, index=False)

    with pytest.raises(FileLoadError, match="was not found"):
        load_dataset("book.xlsx", buffer.getvalue(), sheet_name="Missing")


def test_file_size_limit() -> None:
    with pytest.raises(FileLoadError, match="exceeds"):
        load_dataset("large.csv", b"a\n1\n", max_size_mb=0)
