"""In-memory exports for data, tables, and JSON history."""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd


def dataframe_to_csv(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


def dataframe_to_excel(dataframe: pd.DataFrame, sheet_name: str = "Cleaned Data") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return buffer.getvalue()


def history_to_json(history: list[dict[str, Any]]) -> bytes:
    return json.dumps(history, indent=2, default=str).encode("utf-8")
