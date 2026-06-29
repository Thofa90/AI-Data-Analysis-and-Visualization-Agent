"""Display formatting helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def is_currency_column(column_name: str | None) -> bool:
    """Identify common monetary metric names."""
    if not column_name:
        return False
    normalized = column_name.lower().replace("_", "").replace(" ", "")
    return any(
        token in normalized
        for token in (
            "revenue", "profit", "sales", "cost", "price",
            "income", "amount", "turnover", "earnings",
        )
    )


def format_compact_number(value: int | float, decimals: int = 1) -> str:
    """Format a number with compact K, M, B, and T suffixes."""
    numeric = float(value)
    for divisor, suffix in (
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ):
        if abs(numeric) >= divisor:
            compact = numeric / divisor
            rendered = f"{compact:.{decimals}f}".rstrip("0").rstrip(".")
            return f"{rendered}{suffix}"
    if numeric.is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:,.2f}".rstrip("0").rstrip(".")


def format_currency(
    value: int | float,
    symbol: str = "$",
    unit: str = "auto",
) -> str:
    """Format currency using compact display units."""
    numeric = float(value)
    sign = "-" if numeric < 0 else ""
    divisors = {
        "unit": (1, ""),
        "K": (1_000, "K"),
        "M": (1_000_000, "M"),
        "B": (1_000_000_000, "B"),
    }
    if unit == "auto":
        rendered = format_compact_number(abs(numeric))
    else:
        divisor, suffix = divisors.get(unit, divisors["unit"])
        scaled = abs(numeric) / divisor
        rendered = f"{scaled:,.2f}".rstrip("0").rstrip(".") + suffix
    return f"{sign}{symbol}{rendered}"


def format_metric(value: int | float, column_name: str | None = None) -> str:
    """Format a value according to its metric name."""
    if is_currency_column(column_name):
        return format_currency(value)
    return format_compact_number(value)


def format_bytes(size_bytes: int | float) -> str:
    """Format a byte count using binary units."""
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def format_number(value: int | float) -> str:
    """Format a number with thousands separators."""
    return f"{value:,.0f}"


def format_period(value: Any, granularity: str | None = None) -> str:
    """Format dates and periods without exposing raw timestamps."""
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return str(value)
    if granularity == "year":
        return timestamp.strftime("%Y")
    if granularity == "quarter":
        quarter = (timestamp.month - 1) // 3 + 1
        return f"Q{quarter} {timestamp.year}"
    if granularity == "month":
        return timestamp.strftime("%B %Y")
    if granularity == "week":
        return f"Week of {timestamp.strftime('%-d %b %Y')}"
    return timestamp.strftime("%-d %b %Y")
