"""Tests for approved analytical tools and safe execution."""

from __future__ import annotations

import pandas as pd
import pytest

from agent.tool_registry import execute_tool


def test_group_aggregate_and_ranking() -> None:
    dataframe = pd.DataFrame({
        "Region": ["West", "West", "East"],
        "Sales": [10.0, 20.0, 5.0],
    })
    result = execute_tool(dataframe, "group_and_aggregate", {
        "group_by": "Region", "value_column": "Sales", "aggregation": "sum",
    })
    assert result.data[0] == {"Region": "West", "Sales": 30.0}
    ranked = execute_tool(dataframe, "sort_and_limit", {"sort_by": "Sales", "limit": 2})
    assert ranked.data[0]["Sales"] == 20.0


def test_group_aggregate_supports_two_categories() -> None:
    dataframe = pd.DataFrame({
        "Region": ["West", "West", "East", "East"],
        "SalesChannel": ["Online", "Offline", "Online", "Offline"],
        "TotalRevenue": [30.0, 20.0, 15.0, 10.0],
    })
    result = execute_tool(dataframe, "group_and_aggregate", {
        "group_by": "Region",
        "secondary_group_by": "SalesChannel",
        "value_column": "TotalRevenue",
        "aggregation": "sum",
    })

    assert result.data[0] == {
        "Region": "West",
        "SalesChannel": "Online",
        "TotalRevenue": 30.0,
    }


def test_group_aggregate_supports_multiple_metrics() -> None:
    dataframe = pd.DataFrame({
        "Region": ["West", "West", "East"],
        "TotalRevenue": [30.0, 20.0, 15.0],
        "TotalProfit": [8.0, 5.0, 4.0],
    })
    result = execute_tool(dataframe, "group_and_aggregate", {
        "group_by": "Region",
        "value_columns": ["TotalRevenue", "TotalProfit"],
        "aggregation": "sum",
    })

    assert result.data[0] == {
        "Region": "West",
        "TotalRevenue": 50.0,
        "TotalProfit": 13.0,
    }


def test_group_aggregate_counts_rows_without_numeric_metric() -> None:
    dataframe = pd.DataFrame({
        "Country": ["Turkey", "Turkey", "Turkey", "Canada"],
        "SalesChannel": ["Online", "Offline", "Online", "Online"],
    })

    result = execute_tool(dataframe, "group_and_aggregate", {
        "group_by": "SalesChannel",
        "value_column": "Count",
        "aggregation": "count",
        "limit": 1,
        "filter_column": "Country",
        "filter_value": "Turkey",
    })

    assert result.data == [{"SalesChannel": "Online", "Count": 2}]


def test_categorical_value_counts_simple_grouped_filtered_and_unique() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Europe", "Europe", "Europe", "Asia", "Asia"],
        "Country": ["Spain", "India", "Spain", "France", "Spain", "Thailand", "Spain"],
        "OrderPriority": ["High", "Low", "Medium", "High", "Critical", "High", None],
        "OrderID": [1, 2, 3, 4, 3, 6, 7],
    })

    simple = execute_tool(dataframe, "analyze_categorical_value_counts", {
        "counted_column": "OrderPriority",
    })
    assert simple.data["table_rows"][0] == {
        "OrderPriority": "High",
        "Count": 3,
        "Percentage": 50.0,
    }
    assert simple.data["total_matching_rows"] == 6

    grouped = execute_tool(dataframe, "analyze_categorical_value_counts", {
        "counted_column": "OrderPriority",
        "primary_group_column": "Region",
        "normalization": "within_primary_group",
        "chart_type": "grouped_bar",
    })
    europe_rows = [
        row for row in grouped.data["table_rows"] if row["Region"] == "Europe"
    ]
    assert sum(row["Count"] for row in europe_rows) == 3
    assert sum(row["Percentage"] for row in europe_rows) == pytest.approx(100.0)

    filtered = execute_tool(dataframe, "analyze_categorical_value_counts", {
        "counted_column": "OrderPriority",
        "filters": [{"column": "Country", "operator": "equals", "value": "Spain"}],
    })
    assert filtered.data["total_matching_rows"] == 3
    assert {row["OrderPriority"] for row in filtered.data["table_rows"]} == {
        "Critical",
        "High",
        "Medium",
    }

    unique = execute_tool(dataframe, "analyze_categorical_value_counts", {
        "counted_column": "OrderPriority",
        "filters": [{"column": "Country", "operator": "equals", "value": "Spain"}],
        "measure_type": "distinct_count",
        "distinct_column": "OrderID",
    })
    assert unique.data["table_rows"][0]["Unique OrderID Count"] == 1


def test_profile_column_returns_type_aware_profiles() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe", "Asia", None],
        "OrderID": [1, 2, 2, 4],
        "TotalRevenue": [100.0, 200.0, 0.0, -50.0],
        "Date": ["2021-01-01", "2021-02-01", "bad", "2022-01-01"],
        "IsActive": [True, False, True, True],
        "Notes": [
            "Customer requested a detailed delivery note.",
            "Long descriptive text for a second order.",
            "Another sentence with enough length to be text.",
            "Final free text value for the profile.",
        ],
    })

    categorical = execute_tool(dataframe, "profile_column", {"column_name": "Region"})
    numerical = execute_tool(dataframe, "profile_column", {"column_name": "TotalRevenue"})
    datetime = execute_tool(dataframe, "profile_column", {"column_name": "Date"})
    identifier = execute_tool(dataframe, "profile_column", {"column_name": "OrderID"})
    boolean = execute_tool(dataframe, "profile_column", {"column_name": "IsActive"})
    free_text = execute_tool(dataframe, "profile_column", {"column_name": "Notes"})

    assert categorical.data["profile"]["semantic_type"] == "categorical"
    assert categorical.data["profile"]["missing_count"] == 1
    assert categorical.data["table_rows"][0] == {
        "Region": "Asia",
        "Count": 2,
        "Percentage": pytest.approx(66.66666666666666),
    }
    assert categorical.data["chart_type"] == "bar"

    assert numerical.data["profile"]["semantic_type"] == "numerical"
    assert numerical.data["profile"]["minimum"] == -50.0
    assert numerical.data["profile"]["maximum"] == 200.0
    assert numerical.data["profile"]["zero_count"] == 1
    assert numerical.data["profile"]["negative_count"] == 1
    assert numerical.data["chart_type"] == "histogram"

    assert datetime.data["profile"]["semantic_type"] == "datetime"
    assert datetime.data["profile"]["earliest_date"] == "2021-01-01"
    assert datetime.data["profile"]["latest_date"] == "2022-01-01"
    assert datetime.data["profile"]["missing_count"] == 1
    assert datetime.data["chart_rows"][0] == {"Period": "2021-01-01", "Count": 1}

    assert identifier.data["profile"]["semantic_type"] == "identifier"
    assert identifier.data["profile"]["duplicate_count"] == 1
    assert identifier.data["chart_type"] is None
    assert all(row["Statistic"] != "Mean" for row in identifier.data["table_rows"])

    assert boolean.data["profile"]["semantic_type"] == "boolean"
    assert boolean.data["profile"]["true_count"] == 3
    assert boolean.data["profile"]["false_count"] == 1

    assert free_text.data["profile"]["semantic_type"] == "free_text"
    assert free_text.data["profile"]["average_length"] is not None


def test_grouped_benchmark_comparison_supports_parent_groups() -> None:
    dataframe = pd.DataFrame({
        "City": ["North", "North", "North", "South", "South"],
        "Store": ["A", "A", "B", "C", "D"],
        "Revenue": [40.0, 60.0, 300.0, 50.0, 150.0],
    })
    result = execute_tool(dataframe, "compare_grouped_to_benchmark", {
        "category_column": "Store",
        "value_column": "Revenue",
        "aggregation": "sum",
        "benchmark": "mean",
        "comparison": "below",
        "benchmark_group_by": "City",
    })

    assert result.data == [
        {
            "City": "North",
            "Store": "A",
            "Revenue": 100.0,
            "Benchmark": 200.0,
            "DifferenceFromBenchmark": -100.0,
        },
        {
            "City": "South",
            "Store": "C",
            "Revenue": 50.0,
            "Benchmark": 100.0,
            "DifferenceFromBenchmark": -50.0,
        },
    ]


def test_grouped_extrema_filters_to_multiple_primary_values() -> None:
    dataframe = pd.DataFrame({
        "Country": ["Canada", "Canada", "USA", "USA", "Mexico"],
        "ItemType": ["Meat", "Fruit", "Meat", "Fruit", "Meat"],
        "TotalRevenue": [10.0, 30.0, 50.0, 20.0, 999.0],
    })

    result = execute_tool(dataframe, "calculate_grouped_extrema", {
        "primary_group_column": "Country",
        "secondary_group_column": "ItemType",
        "metric_column": "TotalRevenue",
        "aggregation": "sum",
        "extremum": "max",
        "filter_column": "Country",
        "filter_values": ["Canada", "USA"],
    })

    assert result.data[0]["Country"] == "Canada"
    assert result.data[0]["ItemType"] == "Fruit"
    assert result.data[0]["TotalRevenue"] == 30.0
    assert result.data[1]["Country"] == "USA"
    assert result.data[1]["ItemType"] == "Meat"
    assert result.data[1]["TotalRevenue"] == 50.0
    assert {row["Country"] for row in result.data} == {"Canada", "USA"}


def test_grouped_benchmark_comparison_supports_global_benchmark() -> None:
    dataframe = pd.DataFrame({
        "Product": ["A", "B", "C"],
        "Units": [10, 30, 20],
    })
    result = execute_tool(dataframe, "compare_grouped_to_benchmark", {
        "category_column": "Product",
        "value_column": "Units",
        "benchmark": "mean",
        "comparison": "above",
    })

    assert result.data == [{
        "Product": "B",
        "Units": 30,
        "Benchmark": 20.0,
        "DifferenceFromBenchmark": 10.0,
    }]


def test_grouped_benchmark_applies_filter_before_calculation() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Europe"],
        "Country": ["Japan", "India", "France"],
        "Revenue": [100.0, 50.0, 5_000.0],
    })
    result = execute_tool(dataframe, "compare_grouped_to_benchmark", {
        "category_column": "Country",
        "value_column": "Revenue",
        "benchmark": "mean",
        "comparison": "below",
        "filter_column": "Region",
        "filter_value": "Asia",
    })

    assert result.data == [{
        "Country": "India",
        "Revenue": 50.0,
        "Benchmark": 75.0,
        "DifferenceFromBenchmark": -25.0,
    }]


def test_compare_category_values_calculates_verified_difference() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Europe", "Asia", "Europe", "Asia"],
        "TotalProfit": [30.0, 10.0, 20.0, 15.0],
    })
    result = execute_tool(dataframe, "compare_category_values", {
        "category_column": "Region",
        "value_column": "TotalProfit",
        "first_value": "Europe",
        "second_value": "Asia",
        "aggregation": "sum",
    })

    assert result.data["first_total"] == 50.0
    assert result.data["second_total"] == 25.0
    assert result.data["absolute_difference"] == 25.0
    assert result.data["percentage_difference"] == 100.0
    assert result.data["higher_value"] == "Europe"


def test_filtered_aggregate_returns_one_verified_scalar() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe", "Asia"],
        "TotalProfit": [10.0, 30.0, 4.0],
    })
    result = execute_tool(dataframe, "calculate_filtered_aggregate", {
        "category_column": "Region",
        "category_value": "Asia",
        "value_column": "TotalProfit",
        "aggregation": "sum",
    })

    assert result.data["value"] == 14.0
    assert result.data["category_value"] == "Asia"


def test_filtered_aggregate_returns_multiple_verified_scalars() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe", "Asia"],
        "TotalProfit": [10.0, 30.0, 4.0],
        "TotalRevenue": [100.0, 300.0, 40.0],
    })
    result = execute_tool(dataframe, "calculate_filtered_aggregate", {
        "category_column": "Region",
        "category_value": "Asia",
        "value_column": "TotalProfit",
        "value_columns": ["TotalProfit", "TotalRevenue"],
        "aggregation": "sum",
    })

    assert result.data["value"] == 14.0
    assert result.data["value_columns"] == ["TotalProfit", "TotalRevenue"]
    assert result.data["values"] == [
        {"value_column": "TotalProfit", "aggregation": "sum", "value": 14.0},
        {"value_column": "TotalRevenue", "aggregation": "sum", "value": 140.0},
    ]


def test_dataset_wide_scalar_aggregate() -> None:
    dataframe = pd.DataFrame({"TotalRevenue": [10.0, 20.0, 30.0]})
    result = execute_tool(dataframe, "calculate_scalar_aggregate", {
        "value_column": "TotalRevenue",
        "aggregation": "mean",
    })

    assert result.data == {
        "value_column": "TotalRevenue",
        "aggregation": "mean",
        "value": 20.0,
    }


def test_dataset_wide_multi_scalar_aggregate() -> None:
    dataframe = pd.DataFrame({
        "TotalRevenue": [10.0, 20.0, 30.0],
        "TotalProfit": [1.0, 3.0, 5.0],
        "UnitsSold": [2.0, 4.0, 6.0],
    })
    result = execute_tool(dataframe, "calculate_multi_scalar_aggregate", {
        "value_columns": ["TotalRevenue", "TotalProfit", "UnitsSold"],
        "aggregation": "mean",
    })

    assert result.data == [
        {"value_column": "TotalRevenue", "aggregation": "mean", "value": 20.0},
        {"value_column": "TotalProfit", "aggregation": "mean", "value": 3.0},
        {"value_column": "UnitsSold", "aggregation": "mean", "value": 4.0},
    ]


def test_categorical_lookup_and_distinct_count() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe", "Africa", "Asia"],
        "SalesChannel": ["Offline", "Online", "Offline", "Online"],
    })
    available = execute_tool(dataframe, "list_distinct_values", {
        "target_column": "Region",
        "filter_column": "SalesChannel",
        "filter_value": "Offline",
    })
    count = execute_tool(dataframe, "count_distinct_values", {
        "column": "Region",
    })

    assert available.data["values"] == ["Africa", "Asia"]
    assert available.data["count"] == 2
    assert count.data == {"column": "Region", "count": 3}


def test_high_volume_low_profit_analysis_uses_median_thresholds() -> None:
    dataframe = pd.DataFrame({
        "Country": ["A", "B", "C", "D"],
        "UnitsSold": [100, 80, 40, 20],
        "TotalProfit": [10.0, 50.0, 5.0, 30.0],
    })
    result = execute_tool(dataframe, "analyze_high_volume_low_outcome", {
        "category_column": "Country",
        "volume_column": "UnitsSold",
        "outcome_column": "TotalProfit",
        "aggregation": "sum",
    })

    assert result.data["volume_threshold"] == 60.0
    assert result.data["outcome_threshold"] == 20.0
    assert result.data["candidates"] == [{
        "Country": "A",
        "UnitsSold": 100,
        "TotalProfit": 10.0,
    }]


def test_missing_duplicates_correlation_and_outliers() -> None:
    dataframe = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, 100.0, 100.0],
        "B": [2.0, 4.0, 6.0, 200.0, 200.0],
        "C": ["x", None, "y", "z", "z"],
    })
    assert execute_tool(dataframe, "analyze_missing_values", {}).data[0]["column"] == "C"
    assert execute_tool(dataframe, "analyze_duplicates", {}).data["duplicate_rows"] == 1
    correlation = execute_tool(dataframe, "calculate_correlation", {
        "first_column": "A", "second_column": "B",
    })
    assert correlation.data["correlation"] == pytest.approx(1.0)
    assert execute_tool(dataframe, "detect_outliers", {"column": "A"}).data[0]["outlier_count"] >= 0


def test_invalid_tool_or_column_is_rejected() -> None:
    dataframe = pd.DataFrame({"Sales": [1, 2]})
    with pytest.raises(ValueError, match="Unsupported analytical tool"):
        execute_tool(dataframe, "exec_python", {})
    with pytest.raises(ValueError, match="not found"):
        execute_tool(dataframe, "group_and_aggregate", {
            "group_by": "Region", "value_column": "Sales", "aggregation": "sum",
        })
    with pytest.raises(ValueError, match="Invalid arguments"):
        execute_tool(dataframe, "inspect_dataset", {
            "dataset_path": "/invented/path.csv",
        })


def test_filter_time_trend_and_value_counts() -> None:
    dataframe = pd.DataFrame({
        "Date": ["2025-01-01", "2025-02-01", "2025-02-15"],
        "Region": ["West", "East", "East"],
        "Sales": [10.0, 20.0, 30.0],
    })
    filtered = execute_tool(dataframe, "filter_dataset", {
        "column": "Region", "operator": "equals", "value": "East",
    })
    assert len(filtered.data) == 2
    trend = execute_tool(dataframe, "calculate_time_trend", {
        "date_column": "Date", "value_column": "Sales", "aggregation": "sum", "frequency": "month",
    })
    assert trend.data[-1]["Sales"] == 50.0
    filtered_trend = execute_tool(dataframe, "calculate_time_trend", {
        "date_column": "Date",
        "value_column": "Sales",
        "aggregation": "sum",
        "frequency": "month",
        "start_date": "2025-02-01",
        "end_date": "2025-02-28",
    })
    assert filtered_trend.data == [{"Date": "2025-02", "Sales": 50.0}]
    category_filtered_trend = execute_tool(dataframe, "calculate_time_trend", {
        "date_column": "Date",
        "value_column": "Sales",
        "aggregation": "sum",
        "frequency": "month",
        "filter_column": "Region",
        "filter_value": "East",
    })
    assert category_filtered_trend.data == [{"Date": "2025-02", "Sales": 50.0}]
    multi_metric_trend = execute_tool(dataframe.assign(Profit=[1.0, 2.0, 3.0]), "calculate_time_trend", {
        "date_column": "Date",
        "value_column": "Sales",
        "value_columns": ["Sales", "Profit"],
        "aggregation": "sum",
        "frequency": "month",
    })
    assert multi_metric_trend.data == [
        {"Date": "2025-01", "Sales": 10.0, "Profit": 1.0},
        {"Date": "2025-02", "Sales": 50.0, "Profit": 5.0},
    ]
    breakdown_trend = execute_tool(dataframe, "calculate_time_trend", {
        "date_column": "Date",
        "value_column": "Sales",
        "breakdown_column": "Region",
        "aggregation": "sum",
        "frequency": "month",
    })
    assert breakdown_trend.data == [
        {"Date": "2025-01", "Region": "West", "Sales": 10.0},
        {"Date": "2025-02", "Region": "East", "Sales": 50.0},
    ]
    counts = execute_tool(dataframe, "calculate_value_counts", {"column": "Region"})
    assert counts.data[0]["count"] == 2


def test_filtered_group_ranking() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Europe", "Europe", "Asia"],
        "Country": ["France", "Germany", "Japan"],
        "TotalRevenue": [20.0, 60.0, 120.0],
    })
    result = execute_tool(dataframe, "group_and_aggregate", {
        "group_by": "Country",
        "value_column": "TotalRevenue",
        "aggregation": "sum",
        "limit": 1,
        "filter_column": "Region",
        "filter_value": "europe",
    })
    assert result.data == [{"Country": "Germany", "TotalRevenue": 60.0}]


def test_filtered_group_ranking_can_return_lowest() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Europe", "Europe", "Europe", "Asia"],
        "ItemType": ["Baby Food", "Office Supplies", "Meat", "Meat"],
        "TotalRevenue": [32.319, 89.3, 12.0, 120.0],
    })
    result = execute_tool(dataframe, "group_and_aggregate", {
        "group_by": "ItemType",
        "value_column": "TotalRevenue",
        "aggregation": "sum",
        "limit": 1,
        "filter_column": "Region",
        "filter_value": "europe",
        "sort_descending": False,
    })
    assert result.data == [{"ItemType": "Meat", "TotalRevenue": 12.0}]


def test_group_and_aggregate_supports_category_and_date_filters() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Asia", "Europe", "Asia"],
        "Country": ["Thailand", "India", "Thailand", "France", "India"],
        "Date": ["2021-01-01", "2021-02-01", "2022-01-01", "2021-01-01", "2021-03-01"],
        "TotalRevenue": [10.0, 20.0, 999.0, 40.0, 30.0],
    })
    result = execute_tool(dataframe, "group_and_aggregate", {
        "group_by": "Country",
        "value_column": "TotalRevenue",
        "aggregation": "sum",
        "filter_column": "Region",
        "filter_value": "Asia",
        "date_column": "Date",
        "start_date": "2021-01-01",
        "end_date": "2021-12-31",
    })

    assert result.data == [
        {"Country": "India", "TotalRevenue": 50.0},
        {"Country": "Thailand", "TotalRevenue": 10.0},
    ]


def test_group_and_aggregate_can_include_percentage_of_total() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Europe"],
        "TotalRevenue": [100.0, 300.0, 600.0],
    })
    result = execute_tool(dataframe, "group_and_aggregate", {
        "group_by": "Region",
        "value_column": "TotalRevenue",
        "aggregation": "sum",
        "include_percentage": True,
    })

    assert result.data == [
        {"Region": "Europe", "TotalRevenue": 600.0, "PercentageOfTotal": 60.0},
        {"Region": "Asia", "TotalRevenue": 400.0, "PercentageOfTotal": 40.0},
    ]


def test_calculate_period_over_period_uses_previous_period_baseline() -> None:
    dataframe = pd.DataFrame({
        "Date": ["2020-12-01", "2021-01-01", "2021-02-01"],
        "Region": ["Asia", "Asia", "Asia"],
        "TotalRevenue": [50.0, 100.0, 150.0],
    })

    result = execute_tool(dataframe, "calculate_period_over_period", {
        "date_column": "Date",
        "value_column": "TotalRevenue",
        "aggregation": "sum",
        "frequency": "month",
        "start_date": "2021-01-01",
        "end_date": "2021-12-31",
        "filter_column": "Region",
        "filter_value": "Asia",
    })

    assert result.data[:2] == [
        {
            "Date": "2021-01",
            "TotalRevenue": 100.0,
            "PreviousPeriodValue": 50.0,
            "AbsoluteChange": 50.0,
            "PercentageChange": 100.0,
        },
        {
            "Date": "2021-02",
            "TotalRevenue": 150.0,
            "PreviousPeriodValue": 100.0,
            "AbsoluteChange": 50.0,
            "PercentageChange": 50.0,
        },
    ]


def test_grouped_extrema_sums_before_ranking_and_preserves_ties() -> None:
    dataframe = pd.DataFrame({
        "Region": [
            "Asia", "Asia", "Asia", "Europe", "Europe", "North America",
            "Africa", "Africa",
        ],
        "ItemType": [
            "Household", "Cosmetics", "Household", "Cosmetics", "Office Supplies",
            "Office Supplies", "Food", "Beverages",
        ],
        "TotalRevenue": [100.0, 70.0, 50.0, 120.0, 80.0, 90.0, 40.0, 40.0],
    })

    result = execute_tool(dataframe, "calculate_grouped_extrema", {
        "primary_group_column": "Region",
        "secondary_group_column": "ItemType",
        "metric_column": "TotalRevenue",
        "aggregation": "sum",
        "extremum": "max",
    })

    assert result.data == [
        {
            "Region": "Africa",
            "ItemType": "Beverages",
            "TotalRevenue": 40.0,
            "Rank": 1,
            "Tie": True,
            "TieCount": 2,
            "SecondPlace": None,
            "SecondPlaceValue": None,
            "AbsoluteGap": None,
            "PercentageGap": None,
            "WinnerShareOfGroup": 50.0,
        },
        {
            "Region": "Africa",
            "ItemType": "Food",
            "TotalRevenue": 40.0,
            "Rank": 1,
            "Tie": True,
            "TieCount": 2,
            "SecondPlace": None,
            "SecondPlaceValue": None,
            "AbsoluteGap": None,
            "PercentageGap": None,
            "WinnerShareOfGroup": 50.0,
        },
        {
            "Region": "Asia",
            "ItemType": "Household",
            "TotalRevenue": 150.0,
            "Rank": 1,
            "Tie": False,
            "TieCount": 1,
            "SecondPlace": "Cosmetics",
            "SecondPlaceValue": 70.0,
            "AbsoluteGap": 80.0,
            "PercentageGap": 114.28571428571428,
            "WinnerShareOfGroup": 68.18181818181817,
        },
        {
            "Region": "Europe",
            "ItemType": "Cosmetics",
            "TotalRevenue": 120.0,
            "Rank": 1,
            "Tie": False,
            "TieCount": 1,
            "SecondPlace": "Office Supplies",
            "SecondPlaceValue": 80.0,
            "AbsoluteGap": 40.0,
            "PercentageGap": 50.0,
            "WinnerShareOfGroup": 60.0,
        },
        {
            "Region": "North America",
            "ItemType": "Office Supplies",
            "TotalRevenue": 90.0,
            "Rank": 1,
            "Tie": False,
            "TieCount": 1,
            "SecondPlace": None,
            "SecondPlaceValue": None,
            "AbsoluteGap": None,
            "PercentageGap": None,
            "WinnerShareOfGroup": 100.0,
        },
    ]
