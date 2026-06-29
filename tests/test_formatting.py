"""Tests for compact display formatting."""

from __future__ import annotations

from utils.formatting import (
    format_compact_number,
    format_currency,
    format_metric,
    is_currency_column,
)


def test_compact_number_units() -> None:
    assert format_compact_number(950) == "950"
    assert format_compact_number(24_900) == "24.9K"
    assert format_compact_number(356_724_250.12) == "356.7M"
    assert format_compact_number(1_250_000_000) == "1.2B"


def test_currency_formatting_and_detection() -> None:
    assert format_currency(356_724_250.12) == "$356.7M"
    assert format_currency(-24_900) == "-$24.9K"
    assert format_currency(2_500_000, symbol="€", unit="M") == "€2.5M"
    assert format_currency(2_500, symbol="£", unit="unit") == "£2,500"
    assert format_metric(1_200_000, "TotalProfit") == "$1.2M"
    assert is_currency_column("TotalRevenue")
    assert not is_currency_column("UnitsSold")
