"""Tests for Smart Visualization title generation."""

import pandas as pd

from pages.visualization_page import (
    CHART_OPTIONS,
    build_chart_title,
    category_limit_max,
    date_candidate_columns,
    discrete_date_period_options,
    normalize_symbol_map_color,
    reset_invalid_symbol_map_color_state,
    symbol_map_color_options,
)


def test_date_candidate_columns_include_parseable_date_values() -> None:
    dataframe = pd.DataFrame({
        "Period": ["2026-01-01", "2026-02-01", "2026-03-01"],
        "Region": ["West", "East", "West"],
        "Sales": [10.0, 20.0, 30.0],
    })

    assert date_candidate_columns(dataframe) == ["Period"]


def test_discrete_date_period_options_include_year_month_week_day() -> None:
    dates = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-15", "2026-02-01"]))

    assert [label for label, _ in discrete_date_period_options(dates, "year")] == ["2026"]
    assert [label for label, _ in discrete_date_period_options(dates, "month")] == ["January 2026", "February 2026"]
    assert [label for label, _ in discrete_date_period_options(dates, "week")] == [
        "2026-W01",
        "2026-W03",
        "2026-W05",
    ]
    assert [label for label, _ in discrete_date_period_options(dates, "day")] == [
        "01 January 2026",
        "15 January 2026",
        "01 February 2026",
    ]


def test_scatter_category_limit_max_uses_observations_not_unique_x_values() -> None:
    dataframe = pd.DataFrame({
        "UnitsSold": [1, 1, 1, 2, 2, 3],
        "TotalRevenue": [10, 12, 11, 20, 21, 30],
        "Region": ["A", "A", "B", "B", "C", "C"],
    })

    assert category_limit_max("scatter", dataframe, "UnitsSold") == 6
    assert category_limit_max("bar", dataframe, "Region") == 3


def test_chart_title_includes_metric_group_and_aggregation() -> None:
    assert CHART_OPTIONS["Sorted Percentage Bar"][0] == "sorted_percentage_bar"
    assert CHART_OPTIONS["Period-over-Period % Change"][0] == "period_over_period_change"
    assert "Gantt View" not in CHART_OPTIONS
    assert "target column" in CHART_OPTIONS["Bullet Graph"][1]
    assert build_chart_title(
        "Sorted Percentage Bar",
        "sorted_percentage_bar",
        "Region",
        "TotalRevenue",
        None,
        None,
        "sum",
    ) == "Share of TotalRevenue by Region"
    assert build_chart_title(
        "Symbol Map",
        "symbol_map",
        "Region",
        "TotalRevenue",
        None,
        None,
        "mean",
    ) == "Average Total Revenue by Region"
    assert build_chart_title(
        "Symbol Map",
        "symbol_map",
        "Region",
        "TotalRevenue",
        None,
        "Region",
        "sum",
    ) == "Total Revenue by Region"
    assert build_chart_title(
        "Symbol Map",
        "symbol_map",
        "Country",
        "TotalRevenue",
        None,
        "Region",
        "sum",
    ) == "Total Revenue by Country and Region"


def test_symbol_map_color_options_exclude_location_and_reset_invalid_state() -> None:
    assert symbol_map_color_options(["Region", "Country", "Segment"], "Region") == ["__none__", "__location__", "Country", "Segment"]
    assert normalize_symbol_map_color("Region", "Region") == "__location__"
    assert normalize_symbol_map_color("Region", "__location__") == "__location__"
    assert normalize_symbol_map_color("Country", "Region") == "Region"
    assert build_chart_title(
        "Symbol Map",
        "symbol_map",
        "Region",
        "TotalRevenue",
        None,
        "__location__",
        "sum",
    ) == "Total Revenue by Region"

    state = {"viz_map_color": "Region"}
    reset_invalid_symbol_map_color_state(state, "viz_map_color", ["__none__", "__location__", "Country", "Segment"], "Region")
    assert state["viz_map_color"] == "__location__"

    state = {"viz_map_color": "Region"}
    reset_invalid_symbol_map_color_state(state, "viz_map_color", ["__none__", "__location__", "Country", "Segment"], "Country")
    assert state["viz_map_color"] == "__none__"

    state = {"viz_map_color": "Segment"}
    reset_invalid_symbol_map_color_state(state, "viz_map_color", ["__none__", "__location__", "Country", "Segment"])
    assert state["viz_map_color"] == "Segment"


def test_dual_and_circle_titles_include_all_selected_measures() -> None:
    assert build_chart_title(
        "Dual Lines",
        "dual_line",
        "Month",
        "Sales",
        "Profit",
        None,
        "sum",
    ) == "Dual Lines: Sum Sales and Profit by Month"
    assert build_chart_title(
        "Circle View",
        "circle_view",
        "Sales",
        "Profit",
        "Orders",
        "Segment",
        None,
    ) == "Circle View: Profit vs Sales, Sized by Orders, Colored by Segment"


def test_dual_line_title_includes_independent_aggregations() -> None:
    assert build_chart_title(
        "Dual Lines",
        "dual_line",
        "Region",
        "TotalRevenue",
        "UnitPrice",
        None,
        None,
        "sum",
        "mean",
    ) == "Dual Lines: TotalRevenue (Sum) and UnitPrice (Mean) by Region"


def test_dual_axis_title_includes_independent_aggregations() -> None:
    assert build_chart_title(
        "Dual Combination",
        "dual_axis",
        "Region",
        "UnitsSold",
        "TotalProfit",
        None,
        None,
        "sum",
        "mean",
    ) == "Dual Combination: UnitsSold (Sum) and TotalProfit (Mean) by Region"


def test_gantt_and_count_titles_describe_selected_fields() -> None:
    assert build_chart_title(
        "Gantt View",
        "gantt",
        "StartDate",
        "Task",
        "EndDate",
        None,
        None,
    ) == "Gantt View: Task from StartDate to EndDate"
    assert build_chart_title(
        "Bar",
        "bar",
        "Region",
        None,
        None,
        None,
        "count",
    ) == "Bar: Count Records by Region"
