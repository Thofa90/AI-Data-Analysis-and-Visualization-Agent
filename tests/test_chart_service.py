"""Tests for deterministic chart recommendation and specifications."""

from __future__ import annotations

import pandas as pd
import pytest

from services.chart_service import (
    SYMBOL_MAP_COLOR_GROUP,
    SYMBOL_MAP_COLOR_LOCATION,
    ChartSpec,
    create_chart,
    effective_symbol_map_color,
    prepare_symbol_map_data,
    recommend_chart,
)


def test_chart_recommendation_rules() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
        "Category": ["A", "B"],
        "Sales": [10.0, 20.0],
        "Profit": [2.0, 4.0],
    })
    assert recommend_chart(dataframe, "Date", "Sales") == "line"
    assert recommend_chart(dataframe, "Category", "Sales") == "bar"
    assert recommend_chart(dataframe, "Sales", "Profit") == "scatter"
    assert recommend_chart(dataframe, "Sales") == "histogram"


def test_bar_chart_uses_verified_aggregation() -> None:
    dataframe = pd.DataFrame({"Region": ["W", "W", "E"], "Sales": [10, 20, 5]})
    spec = ChartSpec(
        chart_type="bar",
        x="Region",
        y="Sales",
        aggregation="sum",
        sort_descending=True,
        title="Sales by Region",
    )
    _, result = create_chart(dataframe, spec)
    assert result.data[0] == {"Region": "W", "Sales": 30}


def test_limited_ascending_bar_chart_returns_lowest_category() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Asia"],
        "ItemType": ["Baby Food", "Fruits", "Meat"],
        "TotalRevenue": [14_200_000.0, 332_900.0, 8_000_000.0],
    })

    _, result = create_chart(dataframe, ChartSpec(
        chart_type="bar",
        x="ItemType",
        y="TotalRevenue",
        aggregation="sum",
        sort_descending=False,
        limit=1,
        filter_column="Region",
        filter_value="Asia",
        title="Sum TotalRevenue by ItemType in Asia",
    ))

    assert result.data == [{"ItemType": "Fruits", "TotalRevenue": 332_900.0}]


def test_every_chart_has_a_centered_visible_title() -> None:
    dataframe = pd.DataFrame({
        "Region": ["West", "East"],
        "Sales": [10.0, 20.0],
    })
    figure, _ = create_chart(dataframe, ChartSpec(
        chart_type="bar",
        x="Region",
        y="Sales",
        aggregation="sum",
        title="Regional Sales",
    ))

    assert figure.layout.title.text == "Regional Sales"
    assert figure.layout.title.x == 0.5
    assert figure.layout.title.xanchor == "center"
    assert figure.layout.title.font.color == "#ffffff"
    assert figure.layout.font.size == 12


def test_chart_currency_unit_scales_display_without_changing_result() -> None:
    dataframe = pd.DataFrame({
        "Region": ["West", "East"],
        "TotalRevenue": [2_500_000.0, 1_000_000.0],
    })
    figure, result = create_chart(dataframe, ChartSpec(
        chart_type="bar",
        x="Region",
        y="TotalRevenue",
        aggregation="sum",
        title="Revenue by Region",
        currency_symbol="€",
        currency_unit="M",
    ))

    assert list(figure.data[0].y) == [1.0, 2.5]
    assert figure.layout.yaxis.tickprefix == "€"
    assert figure.layout.yaxis.ticksuffix == "M"
    assert result.data == [
        {"Region": "East", "TotalRevenue": 1_000_000.0},
        {"Region": "West", "TotalRevenue": 2_500_000.0},
    ]


def test_categorical_bar_chart_counts_records_without_numeric_metric() -> None:
    dataframe = pd.DataFrame({
        "Region": ["West", "West", "East"],
    })
    figure, result = create_chart(dataframe, ChartSpec(
        chart_type="bar",
        x="Region",
        aggregation="count",
        sort_descending=True,
        title="Records by Region",
    ))

    assert result.data == [
        {"Region": "West", "Count": 2},
        {"Region": "East", "Count": 1},
    ]
    assert list(figure.data[0].y) == [2, 1]


def test_categorical_count_chart_supports_second_category() -> None:
    dataframe = pd.DataFrame({
        "Region": ["West", "West", "East"],
        "Channel": ["Online", "Offline", "Online"],
    })
    figure, result = create_chart(dataframe, ChartSpec(
        chart_type="bar",
        x="Region",
        color="Channel",
        aggregation="count",
        title="Records by Region and Channel",
    ))

    assert len(result.data) == 3
    assert len(figure.data) == 2


def test_categorical_pie_chart_counts_records() -> None:
    dataframe = pd.DataFrame({
        "Status": ["Open", "Open", "Closed"],
    })
    figure, result = create_chart(dataframe, ChartSpec(
        chart_type="pie",
        x="Status",
        aggregation="count",
        title="Records by Status",
    ))

    assert {row["Status"]: row["Count"] for row in result.data} == {
        "Open": 2,
        "Closed": 1,
    }
    assert set(figure.data[0].labels) == {"Open", "Closed"}


def test_chart_rejects_missing_column() -> None:
    dataframe = pd.DataFrame({"Sales": [1, 2]})
    with pytest.raises(ValueError, match="not found"):
        create_chart(
            dataframe,
            ChartSpec(chart_type="bar", x="Region", y="Sales", aggregation="sum", title="Bad"),
        )


def test_multi_metric_bar_chart_uses_grouped_long_form_data() -> None:
    dataframe = pd.DataFrame({
        "Region": ["West", "West", "East"],
        "TotalRevenue": [30.0, 20.0, 15.0],
        "TotalProfit": [8.0, 5.0, 4.0],
    })
    spec = ChartSpec(
        chart_type="bar",
        x="Region",
        y="TotalRevenue",
        value_columns=["TotalRevenue", "TotalProfit"],
        aggregation="sum",
        sort_descending=True,
        title="Revenue and Profit by Region",
    )

    figure, result = create_chart(dataframe, spec)

    assert len(figure.data) == 2
    assert result.data[0] == {
        "Region": "West",
        "Metric": "TotalRevenue",
        "Value": 50.0,
    }


def test_bar_chart_can_include_only_requested_category_values() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Europe", "Asia", "Africa"],
        "TotalProfit": [50.0, 25.0, 40.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="bar",
        x="Region",
        y="TotalProfit",
        aggregation="sum",
        include_values=["Europe", "Asia"],
        title="Europe vs Asia Profit",
    ))

    assert {row["Region"] for row in result.data} == {"Europe", "Asia"}
    assert result.spec.y == "TotalProfit"


def test_dual_axis_chart_aggregates_two_metrics() -> None:
    dataframe = pd.DataFrame({
        "Country": ["A", "A", "B"],
        "UnitsSold": [60, 40, 80],
        "TotalProfit": [6.0, 4.0, 50.0],
    })
    figure, result = create_chart(dataframe, ChartSpec(
        chart_type="dual_axis",
        x="Country",
        y="UnitsSold",
        secondary_y="TotalProfit",
        aggregation="sum",
        title="Units and Profit by Country",
    ))

    assert len(figure.data) == 2
    assert result.data[0] == {
        "Country": "A",
        "UnitsSold": 100,
        "TotalProfit": 10.0,
    }
    assert figure.layout.legend.font.color == "#ffffff"
    assert figure.layout.yaxis2.tickprefix == "$"


def test_dual_axis_chart_uses_independent_axis_aggregations() -> None:
    dataframe = pd.DataFrame({
        "Country": ["A", "A", "B", "B"],
        "UnitsSold": [60, 40, 80, 120],
        "TotalProfit": [6.0, 14.0, 50.0, 70.0],
    })
    figure, result = create_chart(dataframe, ChartSpec(
        chart_type="dual_axis",
        x="Country",
        y="UnitsSold",
        secondary_y="TotalProfit",
        primary_aggregation="sum",
        secondary_aggregation="mean",
        title="Units and Profit by Country",
    ))

    assert result.data[0] == {
        "Country": "A",
        "UnitsSold": 100,
        "TotalProfit": 10.0,
    }
    assert result.data[1] == {
        "Country": "B",
        "UnitsSold": 200,
        "TotalProfit": 60.0,
    }
    assert figure.layout.yaxis.title.text == "Units Sold (Sum)"
    assert figure.layout.yaxis2.title.text == "Total Profit (Mean)"


def test_chart_spec_allows_max_category_limit() -> None:
    spec = ChartSpec(
        chart_type="bar",
        x="Region",
        y="Sales",
        aggregation="sum",
        limit=1000,
        title="Sales by Region",
    )

    assert spec.limit == 1000


@pytest.mark.parametrize(
    ("chart_type", "expected_mode"),
    [
        ("stacked_bar", "stack"),
        ("grouped_bar", "group"),
    ],
)
def test_advanced_bar_modes(
    chart_type: str,
    expected_mode: str,
) -> None:
    dataframe = pd.DataFrame({
        "Region": ["West", "West", "East", "East"],
        "Channel": ["Online", "Offline", "Online", "Offline"],
        "Sales": [10.0, 20.0, 5.0, 15.0],
    })
    figure, result = create_chart(dataframe, ChartSpec(
        chart_type=chart_type,
        x="Region",
        y="Sales",
        color="Channel",
        aggregation="sum",
        title="Sales by Region and Channel",
    ))

    assert figure.layout.barmode == expected_mode
    assert len(result.data) == 4


def test_stacked_bar_sort_descending_uses_combined_category_total() -> None:
    dataframe = pd.DataFrame({
        "Region": ["A", "A", "B", "B", "C", "C"],
        "Channel": ["Online", "Offline", "Online", "Offline", "Online", "Offline"],
        "Sales": [100.0, 1.0, 60.0, 60.0, 80.0, 10.0],
    })

    figure, result = create_chart(dataframe, ChartSpec(
        chart_type="stacked_bar",
        x="Region",
        y="Sales",
        color="Channel",
        aggregation="sum",
        sort_descending=True,
        title="Stacked Sales",
    ))

    assert [row["Region"] for row in result.data[:2]] == ["B", "B"]
    assert [row["Region"] for row in result.data[2:4]] == ["A", "A"]
    assert list(figure.layout.xaxis.categoryarray) == ["B", "A", "C"]


def test_area_dual_line_and_circle_views() -> None:
    dataframe = pd.DataFrame({
        "Period": ["Q1", "Q2", "Q3"],
        "Sales": [10.0, 20.0, 30.0],
        "Profit": [2.0, 8.0, 12.0],
        "Orders": [5, 10, 15],
        "Segment": ["A", "A", "B"],
    })
    area, _ = create_chart(dataframe, ChartSpec(
        chart_type="area",
        x="Period",
        y="Sales",
        color="Segment",
        aggregation="sum",
        title="Sales Area",
    ))
    dual, _ = create_chart(dataframe, ChartSpec(
        chart_type="dual_line",
        x="Period",
        y="Sales",
        secondary_y="Profit",
        aggregation="sum",
        title="Sales and Profit",
    ))
    circles, _ = create_chart(dataframe, ChartSpec(
        chart_type="circle_view",
        x="Sales",
        y="Profit",
        secondary_y="Orders",
        color="Segment",
        title="Performance Bubbles",
    ))

    assert area.data[0].type == "scatter"
    assert len(dual.data) == 2
    assert dual.data[0].mode == "lines+markers"
    assert circles.data[0].marker.sizemode == "area"
    assert "Sales=" in circles.data[0].hovertemplate
    assert "Profit=" in circles.data[0].hovertemplate
    assert "Orders=" in circles.data[0].hovertemplate
    assert "Segment=" in circles.data[0].hovertemplate


def test_dual_line_supports_separate_money_units_and_shared_scale() -> None:
    dataframe = pd.DataFrame({
        "Period": ["Q1", "Q2", "Q3"],
        "TotalRevenue": [1_000_000.0, 2_000_000.0, 3_000_000.0],
        "TotalProfit": [100_000.0, 200_000.0, 300_000.0],
    })

    figure, result = create_chart(dataframe, ChartSpec(
        chart_type="dual_line",
        x="Period",
        y="TotalRevenue",
        secondary_y="TotalProfit",
        aggregation="sum",
        title="Revenue and Profit",
        primary_currency_unit="M",
        secondary_currency_unit="K",
        same_y_axis_scale=True,
    ))

    assert result.data[0]["TotalRevenue"] == 1_000_000.0
    assert list(figure.data[0].y) == [1.0, 2.0, 3.0]
    assert list(figure.data[1].y) == [100.0, 200.0, 300.0]
    assert figure.layout.yaxis.ticksuffix == "M"
    assert figure.layout.yaxis2.ticksuffix == "K"
    assert tuple(figure.layout.yaxis.range) == tuple(figure.layout.yaxis2.range)


def test_dual_line_aggregates_each_metric_independently() -> None:
    dataframe = pd.DataFrame({
        "Region": ["West", "West", "East", "East"],
        "TotalRevenue": [100.0, 200.0, 50.0, None],
        "UnitPrice": [10.0, 30.0, 20.0, 40.0],
    })

    figure, result = create_chart(dataframe, ChartSpec(
        chart_type="dual_line",
        x="Region",
        y="TotalRevenue",
        secondary_y="UnitPrice",
        primary_aggregation="sum",
        secondary_aggregation="mean",
        title="Revenue and Price",
    ))

    rows = {row["Region"]: row for row in result.data}
    assert rows["West"]["TotalRevenue"] == pytest.approx(300.0)
    assert rows["West"]["UnitPrice"] == pytest.approx(20.0)
    assert rows["East"]["TotalRevenue"] == pytest.approx(50.0)
    assert rows["East"]["UnitPrice"] == pytest.approx(30.0)
    assert figure.layout.yaxis.title.text == "Total Revenue (Sum)"
    assert figure.layout.yaxis2.title.text == "Unit Price (Mean)"


def test_dual_line_count_and_median_aggregation() -> None:
    dataframe = pd.DataFrame({
        "Month": ["Jan", "Jan", "Feb", "Feb"],
        "OrderID": [1.0, 2.0, 3.0, None],
        "DeliveryTime": [4.0, 8.0, 10.0, 30.0],
    })

    _, result = create_chart(dataframe, ChartSpec(
        chart_type="dual_line",
        x="Month",
        y="OrderID",
        secondary_y="DeliveryTime",
        primary_aggregation="count",
        secondary_aggregation="median",
        title="Orders and Delivery",
    ))

    rows = {row["Month"]: row for row in result.data}
    assert rows["Jan"]["OrderID"] == 2
    assert rows["Jan"]["DeliveryTime"] == pytest.approx(6.0)
    assert rows["Feb"]["OrderID"] == 1
    assert rows["Feb"]["DeliveryTime"] == pytest.approx(20.0)


def test_time_series_line_uses_all_dates_and_date_sort_order() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=120, freq="D"),
        "Sales": [float(index) for index in range(120)],
    })

    figure, result = create_chart(dataframe, ChartSpec(
        chart_type="line",
        x="Date",
        y="Sales",
        aggregation="sum",
        sort_descending=True,
        limit=20,
        title="Sales over Time",
    ))

    assert len(result.data) == 120
    assert result.data[0]["Date"] == pd.Timestamp("2026-04-30")
    assert result.data[-1]["Date"] == pd.Timestamp("2026-01-01")
    assert list(figure.data[0].x)[0] == pd.Timestamp("2026-04-30")


def test_time_series_area_uses_all_string_dates() -> None:
    dataframe = pd.DataFrame({
        "OrderDate": [f"2026-01-{day:02d}" for day in range(1, 31)],
        "Revenue": [float(day) for day in range(1, 31)],
    })

    _, result = create_chart(dataframe, ChartSpec(
        chart_type="area",
        x="OrderDate",
        y="Revenue",
        aggregation="sum",
        limit=5,
        title="Revenue over Time",
    ))

    assert len(result.data) == 30
    assert result.data[0]["OrderDate"] == "2026-01-01"
    assert result.data[-1]["OrderDate"] == "2026-01-30"


def test_time_series_continuous_date_range_filters_inclusively() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-01-01 10:00",
            "2026-01-02 12:00",
            "2026-01-03 18:00",
            "2026-01-04 09:00",
        ]),
        "Sales": [10.0, 20.0, 30.0, 40.0],
    })

    _, result = create_chart(dataframe, ChartSpec(
        chart_type="line",
        x="Date",
        y="Sales",
        aggregation="sum",
        date_range_start="2026-01-02",
        date_range_end="2026-01-03",
        title="Sales Range",
    ))

    assert len(result.data) == 2
    assert [row["Sales"] for row in result.data] == [20.0, 30.0]


def test_time_series_discrete_month_buckets_all_dates() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-01-01",
            "2026-01-15",
            "2026-02-01",
            "2026-02-20",
        ]),
        "Sales": [10.0, 20.0, 30.0, 40.0],
    })

    _, result = create_chart(dataframe, ChartSpec(
        chart_type="line",
        x="Date",
        y="Sales",
        aggregation="sum",
        time_grain="month",
        limit=1,
        title="Monthly Sales",
    ))

    assert len(result.data) == 2
    assert result.data == [
        {"Date": pd.Timestamp("2026-01-01"), "Sales": 30.0},
        {"Date": pd.Timestamp("2026-02-01"), "Sales": 70.0},
    ]


def test_time_series_can_combine_continuous_range_and_discrete_buckets() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-01-01",
            "2026-01-20",
            "2026-02-01",
            "2026-02-15",
            "2026-03-01",
        ]),
        "Sales": [5.0, 10.0, 20.0, 30.0, 100.0],
    })

    _, result = create_chart(dataframe, ChartSpec(
        chart_type="bar",
        x="Date",
        y="Sales",
        aggregation="sum",
        time_column="Date",
        time_grain="month",
        date_range_start="2026-01-15",
        date_range_end="2026-02-28",
        title="Monthly Sales Range",
    ))

    assert result.data == [
        {"Date": pd.Timestamp("2026-01-01"), "Sales": 10.0},
        {"Date": pd.Timestamp("2026-02-01"), "Sales": 50.0},
    ]


def test_date_range_filter_works_for_standard_bar_chart() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01", "2026-01-10", "2026-02-01"]),
        "Region": ["West", "West", "East"],
        "Sales": [10.0, 20.0, 100.0],
    })

    _, result = create_chart(dataframe, ChartSpec(
        chart_type="bar",
        x="Region",
        y="Sales",
        aggregation="sum",
        time_column="Date",
        date_range_start="2026-01-01",
        date_range_end="2026-01-31",
        title="January Sales by Region",
    ))

    assert result.data == [{"Region": "West", "Sales": 30.0}]


def test_discrete_date_period_filter_works_with_non_date_x_axis() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-01-01",
            "2026-02-01",
            "2026-02-15",
            "2026-03-01",
        ]),
        "Region": ["West", "West", "East", "East"],
        "Sales": [10.0, 20.0, 30.0, 100.0],
    })

    _, result = create_chart(dataframe, ChartSpec(
        chart_type="bar",
        x="Region",
        y="Sales",
        aggregation="sum",
        time_column="Date",
        time_grain="month",
        date_period_values=[pd.Timestamp("2026-02-01")],
        title="February Sales by Region",
    ))

    assert result.data == [
        {"Region": "East", "Sales": 30.0},
        {"Region": "West", "Sales": 20.0},
    ]


def test_discrete_time_bucket_works_for_bar_chart_with_date_x_axis() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime(["2025-12-31", "2026-01-01", "2026-02-01"]),
        "Sales": [10.0, 20.0, 30.0],
    })

    _, result = create_chart(dataframe, ChartSpec(
        chart_type="bar",
        x="Date",
        y="Sales",
        aggregation="sum",
        time_column="Date",
        time_grain="year",
        title="Sales by Year",
    ))

    assert result.data == [
        {"Date": pd.Timestamp("2025-01-01"), "Sales": 10.0},
        {"Date": pd.Timestamp("2026-01-01"), "Sales": 50.0},
    ]


def test_discrete_week_filter_uses_iso_week_around_new_year() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime(["2025-12-29", "2026-01-01", "2026-01-05"]),
        "Sales": [10.0, 20.0, 30.0],
    })

    _, result = create_chart(dataframe, ChartSpec(
        chart_type="line",
        x="Date",
        y="Sales",
        aggregation="sum",
        time_grain="week",
        date_period_values=[pd.Timestamp("2025-12-29")],
        title="Weekly Sales",
    ))

    assert result.data == [{"Date": pd.Timestamp("2025-12-29"), "Sales": 30.0}]
    assert result.metadata["date_summary"]["value"] == pytest.approx(30.0)


def test_date_summary_metadata_matches_filtered_source_total() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01", "2026-01-15", "2026-02-01"]),
        "Sales": [10.0, 20.0, 100.0],
    })

    _, result = create_chart(dataframe, ChartSpec(
        chart_type="line",
        x="Date",
        y="Sales",
        aggregation="sum",
        time_grain="month",
        date_period_values=[pd.Timestamp("2026-01-01")],
        title="January Sales",
    ))

    assert result.metadata["date_summary"]["value"] == pytest.approx(30.0)
    assert result.metadata["date_summary"]["period_label"] == "January 2026"


def test_treemap_and_symbol_map() -> None:
    dataframe = pd.DataFrame({
        "Continent": ["Asia", "Asia", "Europe"],
        "Country": ["Japan", "India", "France"],
        "Revenue": [100.0, 80.0, 90.0],
    })
    treemap, _ = create_chart(dataframe, ChartSpec(
        chart_type="treemap",
        x="Country",
        y="Revenue",
        color="Continent",
        aggregation="sum",
        title="Revenue Hierarchy",
    ))
    symbol_map, result = create_chart(dataframe, ChartSpec(
        chart_type="symbol_map",
        x="Country",
        y="Revenue",
        aggregation="sum",
        title="Revenue Map",
    ))

    assert treemap.data[0].type == "treemap"
    assert symbol_map.data[0].type == "scattergeo"
    assert {row["Country"] for row in result.data} == {
        "Japan", "India", "France",
    }


def test_symbol_map_places_all_standard_world_regions() -> None:
    dataframe = pd.DataFrame({
        "Region": [
            "Asia",
            "Australia and Oceania",
            "Central America and the Caribbean",
            "Europe",
            "Middle East and North Africa",
            "North America",
            "Sub-Saharan Africa",
        ],
        "TotalRevenue": [100.0, 80.0, 60.0, 110.0, 70.0, 90.0, 50.0],
    })
    figure, result = create_chart(dataframe, ChartSpec(
        chart_type="symbol_map",
        x="Region",
        y="TotalRevenue",
        color="Region",
        aggregation="sum",
        title="Revenue by Region",
    ))

    plotted_points = sum(len(trace.lat) for trace in figure.data)
    assert plotted_points == 7
    assert len(result.data) == 7
    marker_sizes = {
        float(size)
        for trace in figure.data
        for size in trace.marker.size
    }
    assert len(marker_sizes) == 7
    assert all(trace.marker.sizemode == "diameter" for trace in figure.data)
    assert all(
        "TotalRevenue:" in trace.hovertemplate
        and "%{hovertext}" in trace.hovertemplate
        for trace in figure.data
    )


def test_symbol_map_same_location_and_color_uses_virtual_location_color() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe", "North America"],
        "TotalRevenue": [100.0, 90.0, 80.0],
    })
    spec = ChartSpec(
        chart_type="symbol_map",
        x="Region",
        y="TotalRevenue",
        color="Region",
        aggregation="sum",
        title="Symbol Map: Sum TotalRevenue by Region, by Region",
    )
    map_data, effective_color = prepare_symbol_map_data(
        dataframe,
        "Region",
        "TotalRevenue",
        "Region",
    )
    figure, result = create_chart(dataframe, spec)

    assert effective_symbol_map_color("Region", "Region") == SYMBOL_MAP_COLOR_LOCATION
    assert effective_color == SYMBOL_MAP_COLOR_LOCATION
    assert map_data.columns.tolist() == ["Region", "TotalRevenue", "_location_label", SYMBOL_MAP_COLOR_GROUP]
    assert map_data["_location_label"].tolist() == ["Asia", "Europe", "North America"]
    assert map_data[SYMBOL_MAP_COLOR_GROUP].tolist() == ["Asia", "Europe", "North America"]
    assert spec.color == SYMBOL_MAP_COLOR_LOCATION
    assert spec.title == "Total Revenue by Region"
    assert figure.layout.title.text == "Total Revenue by Region"
    assert sum(len(trace.lat) for trace in figure.data) == 3
    assert len(figure.data) == 3
    assert len(result.data) == 3


def test_symbol_map_country_with_region_color_keeps_breakdown() -> None:
    dataframe = pd.DataFrame({
        "Country": ["Japan", "India", "France"],
        "Region": ["Asia", "Asia", "Europe"],
        "Revenue": [100.0, 80.0, 90.0],
    })
    spec = ChartSpec(
        chart_type="symbol_map",
        x="Country",
        y="Revenue",
        color="Region",
        aggregation="sum",
        title="Revenue by Country and Region",
    )
    map_data, effective_color = prepare_symbol_map_data(
        dataframe,
        "Country",
        "Revenue",
        "Region",
    )
    figure, result = create_chart(dataframe, spec)

    assert effective_color == "Region"
    assert map_data.columns.tolist() == ["Country", "Revenue", "Region", "_location_label", SYMBOL_MAP_COLOR_GROUP]
    assert map_data[SYMBOL_MAP_COLOR_GROUP].tolist() == ["Asia", "Asia", "Europe"]
    assert spec.color == "Region"
    assert figure.data[0].type == "scattergeo"
    assert {row["Region"] for row in result.data} == {"Asia", "Europe"}


def test_symbol_map_rejects_non_geographic_region_labels() -> None:
    dataframe = pd.DataFrame({
        "Region": ["East", "West"],
        "Revenue": [10.0, 20.0],
    })

    with pytest.raises(ValueError, match="not geographic locations"):
        create_chart(dataframe, ChartSpec(
            chart_type="symbol_map",
            x="Region",
            y="Revenue",
            aggregation="sum",
            title="Revenue Map",
        ))


def test_gantt_and_bullet_graphs() -> None:
    dataframe = pd.DataFrame({
        "Task": ["Design", "Build"],
        "StartDate": ["2026-01-01", "2026-01-05"],
        "EndDate": ["2026-01-04", "2026-01-12"],
        "Team": ["UX", "Engineering"],
        "Actual": [80.0, 60.0],
        "Target": [100.0, 75.0],
    })
    gantt, gantt_result = create_chart(dataframe, ChartSpec(
        chart_type="gantt",
        x="StartDate",
        y="Task",
        secondary_y="EndDate",
        color="Team",
        title="Project Plan",
    ))
    bullet, bullet_result = create_chart(dataframe, ChartSpec(
        chart_type="bullet",
        x="Task",
        y="Actual",
        secondary_y="Target",
        aggregation="sum",
        title="Actual vs Target",
    ))

    assert gantt.data[0].type == "bar"
    assert len(gantt_result.data) == 2
    assert len(bullet.data) == 2
    assert bullet.data[0].type == "indicator"
    assert bullet.data[0].number.font.size == 28
    assert bullet.data[0].title.font.size == 13
    assert {row["Task"]: row["Actual"] for row in bullet_result.data} == {
        "Design": 80.0,
        "Build": 60.0,
    }
