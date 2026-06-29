"""Tests for written explanations generated from verified chart values."""

from __future__ import annotations

import re

import pandas as pd

import pytest

from services.chart_insight_service import (
    BoxPlotEvidence,
    ChartInsight,
    CircleViewEvidence,
    CorrelationHeatmapEvidence,
    DualCombinationEvidence,
    DualLineEvidence,
    GroupedBarEvidence,
    HistogramEvidence,
    PeriodOverPeriodChangeEvidence,
    PieChartEvidence,
    ScatterEvidence,
    SingleBarEvidence,
    SingleAreaEvidence,
    SingleLineEvidence,
    SortedPercentageBarEvidence,
    StackedAreaEvidence,
    StackedBarEvidence,
    SymbolMapEvidence,
    TreemapEvidence,
    display_name,
    extract_chart_evidence,
    generate_chart_insight,
    normalize_column_name,
)
from services.chart_service import ChartResult, ChartSpec, create_chart
from config.settings import Settings


def _grouped_priority_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "Region": [
            "Europe", "Europe", "Europe", "Europe",
            "Asia", "Asia", "Asia", "Asia",
            "Sub-Saharan Africa", "Sub-Saharan Africa", "Sub-Saharan Africa",
            "North America", "North America", "North America", "North America",
        ],
        "OrderPriority": [
            "C", "H", "L", "M",
            "C", "H", "L", "M",
            "C", "H", "L",
            "C", "H", "L", "M",
        ],
        "TotalProfit": [
            10.0, 20.0, 70.0, 5.0,
            25.0, 25.0, 24.0, 26.0,
            50.0, 40.0, 0.0,
            1.0, 2.0, 3.0, 4.0,
        ],
    })


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def _standard_bar_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "Region": ["Sub-Saharan Africa", "Europe", "Asia", "North America"],
        "TotalRevenue": [356_700_000.0, 353_100_000.0, 250_000_000.0, 25_000_000.0],
        "TotalProfit": [101_000_000.0, 106_000_000.0, 50_000_000.0, 7_700_000.0],
        "UnitsSold": [5000, 4900, 3200, 500],
        "Channel": ["Offline", "Online", "Offline", "Online"],
    })


def test_column_normalization_and_display_name_helpers() -> None:
    assert normalize_column_name("Units Sold") == normalize_column_name("units_sold")
    assert normalize_column_name("UnitsSold") == normalize_column_name("units-sold")
    assert display_name("total_revenue") == "Total Revenue"
    assert display_name("orderPriority") == "Order Priority"
    assert display_name("customer_URL") == "Customer URL"


def test_chart_insight_model_rejects_empty_required_sections() -> None:
    with pytest.raises(ValueError):
        ChartInsight(
            chart_title="Chart",
            key_finding="",
            supporting_evidence="Evidence",
            interpretation="Interpretation",
            recommended_next_step="Next",
        )


def test_bar_chart_insight_reports_highest_and_lowest() -> None:
    dataframe = pd.DataFrame({"Region": ["West", "East", "South"], "Sales": [30, 20, 10]})
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="bar",
        x="Region",
        y="Sales",
        aggregation="sum",
        sort_descending=True,
        title="Sales by Region",
    ))

    insight = generate_chart_insight(result)

    assert "West" in insight.headline
    assert any("South" in item for item in insight.observations)
    assert insight.evidence
    assert insight.evidence.calculated_metrics["top_share_pct"] == 50.0


def test_line_and_scatter_insights_explain_direction() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
        "Sales": [10.0, 20.0, 30.0],
        "Profit": [2.0, 4.0, 6.0],
    })
    _, line_result = create_chart(dataframe, ChartSpec(
        chart_type="line", x="Date", y="Sales", title="Sales Trend"
    ))
    _, scatter_result = create_chart(dataframe, ChartSpec(
        chart_type="scatter", x="Sales", y="Profit", title="Profit vs Sales"
    ))

    assert "increased" in generate_chart_insight(line_result).headline
    assert "strong positive" in generate_chart_insight(scatter_result).headline
    scatter_evidence = extract_chart_evidence(scatter_result)
    assert scatter_evidence.calculated_metrics["pearson_correlation"] == pytest.approx(1.0)
    assert scatter_evidence.calculated_metrics["r_squared"] == pytest.approx(1.0)


def test_single_line_evidence_uses_displayed_sorted_friendly_monthly_series() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime(["2026-03-01", "2026-01-01", "2026-02-01", "2026-04-01"]),
        "TotalRevenue": [130.0, 100.0, 170.0, 120.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="line",
        x="Date",
        y="TotalRevenue",
        aggregation="sum",
        title="Revenue Trend",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, SingleLineEvidence)
    assert evidence.time_granularity == "month"
    assert evidence.start_period_label == "January 2026"
    assert evidence.end_period_label == "April 2026"
    assert evidence.endpoint_change == pytest.approx(20.0)
    assert evidence.endpoint_change_percent == pytest.approx(20.0)
    assert evidence.endpoint_change_basis == "starting value"
    assert evidence.peak_period_label == "February 2026"
    assert evidence.peak_value == pytest.approx(170.0)
    assert evidence.trough_period_label == "January 2026"
    assert evidence.trough_value == pytest.approx(100.0)
    assert evidence.strongest_increase_value == pytest.approx(70.0)
    assert evidence.strongest_decline_value == pytest.approx(-40.0)
    assert "00:00:00" not in insight.supporting_evidence
    assert "January 2026" in insight.supporting_evidence
    assert "TotalRevenue" not in insight.key_finding


def test_single_line_volatile_series_is_not_called_steady_growth() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=6, freq="MS"),
        "TotalRevenue": [100.0, 250.0, 90.0, 260.0, 80.0, 180.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="line",
        x="Date",
        y="TotalRevenue",
        aggregation="sum",
        title="Volatile Revenue",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, SingleLineEvidence)
    assert evidence.volatility_level == "high"
    assert evidence.direction_reversal_count >= 4
    assert "steady upward trend" not in insight.headline.lower()
    assert "increased across the displayed range" not in insight.headline.lower()
    assert "profitability" in insight.caution
    assert "total profit" in insight.recommended_next_step.lower()


def test_single_line_smooth_and_flat_volatile_classifications() -> None:
    smooth = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=6, freq="MS"),
        "Sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    _, smooth_result = create_chart(smooth, ChartSpec(
        chart_type="line", x="Date", y="Sales", title="Smooth Sales"
    ))
    smooth_evidence = extract_chart_evidence(smooth_result)

    flat_volatile = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=6, freq="MS"),
        "Orders": [100.0, 180.0, 20.0, 175.0, 25.0, 100.0],
    })
    _, flat_result = create_chart(flat_volatile, ChartSpec(
        chart_type="line", x="Date", y="Orders", title="Flat Volatile Orders"
    ))
    flat_evidence = extract_chart_evidence(flat_result)

    assert isinstance(smooth_evidence, SingleLineEvidence)
    assert smooth_evidence.pattern_classification == "steady upward trend"
    assert smooth_evidence.volatility_level == "low"
    assert isinstance(flat_evidence, SingleLineEvidence)
    assert flat_evidence.pattern_classification == "mostly flat but volatile"
    assert flat_evidence.volatility_level == "high"


def test_single_line_missing_months_zero_start_and_short_history() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01", "2026-03-01", "2026-04-01"]),
        "OrderCount": [0.0, 20.0, 10.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="line",
        x="Date",
        y="OrderCount",
        aggregation="sum",
        title="Order Count Trend",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, SingleLineEvidence)
    assert evidence.endpoint_change == pytest.approx(10.0)
    assert evidence.endpoint_change_percent is None
    assert evidence.missing_period_labels == ["February 2026"]
    assert evidence.irregular_intervals is True
    assert evidence.evidence_strength == "low"
    assert "seasonality" in insight.caution
    assert "average order value" in insight.recommended_next_step.lower()


def test_line_area_and_dual_dispatch_are_isolated() -> None:
    dataframe = pd.DataFrame({
        "Period": ["Q1", "Q2", "Q3"],
        "Sales": [10.0, 20.0, 30.0],
        "Profit": [2.0, 4.0, 12.0],
    })
    _, line_result = create_chart(dataframe.assign(Date=pd.date_range("2026-01-01", periods=3, freq="MS")), ChartSpec(
        chart_type="line", x="Date", y="Sales", title="Sales Line"
    ))
    _, area_result = create_chart(dataframe, ChartSpec(
        chart_type="area", x="Period", y="Sales", title="Sales Area"
    ))
    _, dual_result = create_chart(dataframe, ChartSpec(
        chart_type="dual_line", x="Period", y="Sales", secondary_y="Profit", title="Dual"
    ))

    assert isinstance(extract_chart_evidence(line_result), SingleLineEvidence)
    assert not isinstance(extract_chart_evidence(area_result), SingleLineEvidence)
    assert not isinstance(extract_chart_evidence(dual_result), SingleLineEvidence)


def test_single_line_llm_failure_uses_line_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import llm_service

    dataframe = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=4, freq="MS"),
        "TotalRevenue": [100.0, 140.0, 120.0, 180.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="line",
        x="Date",
        y="TotalRevenue",
        aggregation="sum",
        title="Revenue Trend",
    ))
    evidence = extract_chart_evidence(result)
    fallback = generate_chart_insight(result)
    prompts = []

    class FakeResponse:
        content = "not valid json"

    class FakeModel:
        def invoke(self, prompt: str) -> FakeResponse:
            prompts.append(prompt)
            return FakeResponse()

    monkeypatch.setattr(llm_service, "_model", lambda settings, model_name: FakeModel())
    insight, _ = llm_service.explain_chart_insight_with_ollama(
        evidence,
        fallback,
        Settings(),
        "fake-model",
    )

    assert isinstance(evidence, SingleLineEvidence)
    assert insight == fallback
    assert len(prompts) == 2
    assert "single-series line chart" in prompts[0]
    assert "Do not describe a volatile series as steadily increasing or decreasing" in prompts[0]


def test_single_area_evidence_explains_magnitude_and_sustained_periods() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-04-01",
            "2026-01-01",
            "2026-02-01",
            "2026-03-01",
            "2026-05-01",
        ]),
        "TotalRevenue": [220.0, 100.0, 120.0, 240.0, 150.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="area",
        x="Date",
        y="TotalRevenue",
        aggregation="sum",
        title="Revenue Area",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, SingleAreaEvidence)
    assert evidence.start_period_label == "January 2026"
    assert evidence.end_period_label == "May 2026"
    assert evidence.endpoint_change == pytest.approx(50.0)
    assert evidence.peak_period_label == "March 2026"
    assert evidence.trough_period_label == "January 2026"
    assert evidence.longest_above_average_run == 2
    assert evidence.longest_below_average_run == 2
    assert evidence.area_interpretation_valid is True
    assert "filled area" in insight.interpretation.lower()
    assert "00:00:00" not in insight.supporting_evidence
    assert "profitability" in insight.caution


def test_single_area_unordered_negative_and_missing_safeguards() -> None:
    unordered = pd.DataFrame({"Category": ["B", "A", "C"], "Score": [2.0, 5.0, 3.0]})
    _, unordered_result = create_chart(unordered, ChartSpec(
        chart_type="area",
        x="Category",
        y="Score",
        title="Category Area",
    ))
    unordered_evidence = extract_chart_evidence(unordered_result)
    unordered_insight = generate_chart_insight(unordered_result)

    assert isinstance(unordered_evidence, SingleAreaEvidence)
    assert unordered_evidence.evidence_strength == "low"
    assert any("unordered" in warning for warning in unordered_evidence.warnings)
    assert "bar chart" in unordered_insight.recommended_next_step.lower()

    dataframe = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01", "2026-03-01", "2026-04-01"]),
        "Score": [10.0, -5.0, 12.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="area",
        x="Date",
        y="Score",
        title="Score Area",
    ))
    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, SingleAreaEvidence)
    assert evidence.negative_value_count == 1
    assert evidence.missing_period_labels == ["February 2026"]
    assert "negative values" in insight.caution.lower()
    assert "seasonality" in insight.caution.lower()


def test_stacked_area_evidence_tracks_total_and_component_contribution() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-01-01", "2026-01-01",
            "2026-02-01", "2026-02-01",
            "2026-03-01", "2026-03-01",
        ]),
        "BuildingType": ["Office", "Retail", "Office", "Retail", "Office", "Retail"],
        "EnergyConsumption": [70.0, 30.0, 80.0, 60.0, 90.0, 110.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="area",
        x="Date",
        y="EnergyConsumption",
        color="BuildingType",
        aggregation="sum",
        title="Energy by Building Type",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, StackedAreaEvidence)
    assert evidence.start_total == pytest.approx(100.0)
    assert evidence.end_total == pytest.approx(200.0)
    assert evidence.dominant_stack_overall == "Office"
    assert evidence.stack_with_largest_growth == "Retail"
    assert evidence.missing_combinations == []
    assert "component" in insight.interpretation.lower()
    assert "bottom stacked layer" in insight.caution.lower()


def test_stacked_area_missing_combinations_and_area_llm_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import llm_service

    dataframe = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-02-01", "2026-03-01"]),
        "Type": ["A", "B", "A", "B"],
        "Activity": [10.0, 5.0, 20.0, 15.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="area",
        x="Date",
        y="Activity",
        color="Type",
        aggregation="sum",
        title="Activity Mix",
    ))
    evidence = extract_chart_evidence(result)
    fallback = generate_chart_insight(result)
    prompts = []

    class FakeResponse:
        content = "not valid json"

    class FakeModel:
        def invoke(self, prompt: str) -> FakeResponse:
            prompts.append(prompt)
            return FakeResponse()

    monkeypatch.setattr(llm_service, "_model", lambda settings, model_name: FakeModel())
    insight, _ = llm_service.explain_chart_insight_with_ollama(
        evidence,
        fallback,
        Settings(),
        "fake-model",
    )

    assert isinstance(evidence, StackedAreaEvidence)
    assert evidence.missing_combinations
    assert insight == fallback
    assert len(prompts) == 2
    assert "area chart" in prompts[0]
    assert "distinguish total movement from component contribution" in prompts[0]


def test_distribution_and_heatmap_insights_are_written() -> None:
    dataframe = pd.DataFrame({
        "Sales": [10.0, 12.0, 20.0, 30.0],
        "Profit": [2.0, 3.0, 5.0, 8.0],
    })
    _, histogram = create_chart(dataframe, ChartSpec(
        chart_type="histogram", x="Sales", title="Sales Distribution"
    ))
    _, heatmap = create_chart(dataframe, ChartSpec(
        chart_type="heatmap", title="Correlation Matrix"
    ))

    histogram_insight = generate_chart_insight(histogram)
    assert isinstance(histogram_insight.evidence, HistogramEvidence)
    assert "observations" in histogram_insight.supporting_evidence.lower()
    assert "median" in histogram_insight.supporting_evidence.lower()
    heatmap_insight = generate_chart_insight(heatmap)
    assert isinstance(heatmap_insight.evidence, CorrelationHeatmapEvidence)
    assert "unique non-diagonal pair" in heatmap_insight.supporting_evidence.lower()


def test_correlation_heatmap_evidence_summarizes_structure_without_duplicate_pairs() -> None:
    dataframe = pd.DataFrame({
        "Order ID": range(1, 9),
        "Units Sold": [10, 20, 30, 40, 50, 60, 70, 80],
        "Unit Price": [2.0, 4.0, 6.0, 8.0, 11.0, 12.0, 14.0, 16.0],
        "Unit Cost": [1.0, 2.0, 3.0, 4.0, 5.5, 6.0, 7.0, 8.0],
        "Total Revenue": [20.0, 80.0, 180.0, 320.0, 550.0, 720.0, 980.0, 1280.0],
        "Total Cost": [10.0, 40.0, 90.0, 160.0, 275.0, 360.0, 490.0, 640.0],
        "Total Profit": [10.0, 40.0, 90.0, 160.0, 275.0, 360.0, 490.0, 640.0],
        "Inverse Demand": [80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="heatmap",
        title="Correlation Matrix",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, CorrelationHeatmapEvidence)
    assert evidence.correlation_method == "pearson"
    assert evidence.displayed_variable_count == 8
    assert evidence.unique_pair_count == 28
    assert len(evidence.pairs) == 28
    assert not any(pair.variable_x == pair.variable_y for pair in evidence.pairs)
    assert not any(
        pair.variable_x == "Total Cost" and pair.variable_y == "Total Revenue"
        for pair in evidence.pairs
    )
    assert evidence.strongest_positive_pair is not None
    assert evidence.strongest_negative_pair is not None
    assert evidence.strongest_negative_pair.correlation < 0
    assert "Order ID" in evidence.identifier_like_columns
    assert evidence.formula_relationships
    assert evidence.high_multicollinearity_pairs
    assert evidence.correlation_clusters
    assert "single strongest pair" not in insight.key_finding.lower()
    assert "Pearson" in insight.supporting_evidence
    assert "formula" in insight.caution.lower()


def test_correlation_heatmap_reports_pairwise_missing_and_constant_columns() -> None:
    dataframe = pd.DataFrame({
        "Metric A": [1.0, 2.0, 3.0, 4.0, None, None],
        "Metric B": [2.0, 4.0, 6.0, 8.0, 10.0, None],
        "Metric C": [10.0, None, 8.0, None, 6.0, 5.0],
        "Constant Metric": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="heatmap",
        title="Correlation Matrix",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, CorrelationHeatmapEvidence)
    assert "Constant Metric" in evidence.constant_columns
    assert evidence.minimum_pairwise_count is not None
    assert evidence.maximum_pairwise_count is not None
    assert evidence.minimum_pairwise_count < evidence.maximum_pairwise_count
    assert evidence.unequal_pairwise_counts is True
    assert any("different observation counts" in warning.lower() for warning in evidence.warnings)
    assert "observation counts" in insight.caution.lower() or "constant" in insight.caution.lower()


def test_correlation_heatmap_llm_failure_uses_heatmap_prompt_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import llm_service

    dataframe = pd.DataFrame({
        "Sales": [10.0, 20.0, 30.0, 40.0],
        "Profit": [2.0, 4.0, 8.0, 12.0],
        "Cost": [8.0, 16.0, 22.0, 28.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="heatmap",
        title="Correlation Matrix",
    ))
    evidence = extract_chart_evidence(result)
    fallback = generate_chart_insight(result)
    prompts = []

    class FakeResponse:
        content = "not valid json"

    class FakeModel:
        def invoke(self, prompt: str) -> FakeResponse:
            prompts.append(prompt)
            return FakeResponse()

    monkeypatch.setattr(llm_service, "_model", lambda settings, model_name: FakeModel())
    insight, _ = llm_service.explain_chart_insight_with_ollama(
        evidence,
        fallback,
        Settings(),
        "fake-model",
    )

    assert isinstance(evidence, CorrelationHeatmapEvidence)
    assert insight == fallback
    assert len(prompts) == 2
    assert "correlation Heatmap" in prompts[0]
    assert "Do not repeat symmetric pairs" in prompts[0]


def test_histogram_evidence_stats_shape_bins_and_outlier_wording() -> None:
    values = [0, 1, 1, 2, 2, 2, 3, 4, 5, 30]
    dataframe = pd.DataFrame({"TotalProfit": values})
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="histogram",
        x="TotalProfit",
        limit=3,
        title="Profit Distribution",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, HistogramEvidence)
    assert evidence.displayed_value_count == 10
    assert evidence.top_n is None
    assert evidence.minimum == 0
    assert evidence.maximum == 30
    assert evidence.mean == pytest.approx(pd.Series(values).mean())
    assert evidence.median == pytest.approx(pd.Series(values).median())
    assert evidence.standard_deviation == pytest.approx(pd.Series(values).std(ddof=1))
    assert evidence.q1 == pytest.approx(pd.Series(values).quantile(0.25))
    assert evidence.q3 == pytest.approx(pd.Series(values).quantile(0.75))
    assert evidence.iqr == pytest.approx(evidence.q3 - evidence.q1)
    assert evidence.p90 == pytest.approx(pd.Series(values).quantile(0.90))
    assert evidence.p95 == pytest.approx(pd.Series(values).quantile(0.95))
    assert evidence.skew_direction == "right-skewed"
    assert evidence.modal_bin_count is not None
    assert evidence.modal_bin_share is not None
    assert evidence.upper_outlier_count == 1
    assert evidence.potential_outlier_count == 1
    assert evidence.outlier_method == "1.5 x IQR rule"
    assert "potential" in insight.supporting_evidence.lower()
    assert "displayed categories" not in insight.supporting_evidence.lower()
    assert "top" not in insight.supporting_evidence.lower()
    assert "right-skewed distribution" not in insight.interpretation.lower()
    assert "bin width" in insight.caution.lower()


def test_histogram_handles_left_symmetric_zero_negative_and_constant_cases() -> None:
    left_result = create_chart(pd.DataFrame({"Metric": [-30, -5, -4, -3, -2, -2, -1, 0, 0, 1]}), ChartSpec(
        chart_type="histogram",
        x="Metric",
        title="Left",
    ))[1]
    symmetric_result = create_chart(pd.DataFrame({"Metric": [-2, -1, 0, 1, 2]}), ChartSpec(
        chart_type="histogram",
        x="Metric",
        title="Symmetric",
    ))[1]
    constant_result = create_chart(pd.DataFrame({"Metric": [5, 5, 5]}), ChartSpec(
        chart_type="histogram",
        x="Metric",
        title="Constant",
    ))[1]

    left = extract_chart_evidence(left_result)
    symmetric = extract_chart_evidence(symmetric_result)
    constant = extract_chart_evidence(constant_result)

    assert isinstance(left, HistogramEvidence)
    assert left.skew_direction == "left-skewed"
    assert left.negative_count > 0
    assert left.zero_count > 0
    assert isinstance(symmetric, HistogramEvidence)
    assert symmetric.skew_direction == "approximately symmetric"
    assert isinstance(constant, HistogramEvidence)
    assert constant.evidence_strength == "low"
    assert constant.bin_method == "constant"


def test_histogram_group_summary_and_llm_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import llm_service

    dataframe = pd.DataFrame({
        "UnitPrice": [10, 10, 12, 12, 50, 52, 54, 55],
        "ItemType": ["A", "A", "A", "A", "B", "B", "B", "B"],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="histogram",
        x="UnitPrice",
        color="ItemType",
        title="Price Distribution",
    ))
    evidence = extract_chart_evidence(result)
    fallback = generate_chart_insight(result)
    prompts = []

    class FakeResponse:
        content = "not valid json"

    class FakeModel:
        def invoke(self, prompt: str) -> FakeResponse:
            prompts.append(prompt)
            return FakeResponse()

    monkeypatch.setattr(llm_service, "_model", lambda settings, model_name: FakeModel())
    insight, _ = llm_service.explain_chart_insight_with_ollama(
        evidence,
        fallback,
        Settings(),
        "fake-model",
    )

    assert isinstance(evidence, HistogramEvidence)
    assert set(evidence.group_summary) == {"A", "B"}
    assert evidence.multimodal is True
    assert "item type" in fallback.recommended_next_step.lower()
    assert insight == fallback
    assert len(prompts) == 2
    assert "histogram" in prompts[0].lower()


def test_box_plot_evidence_separates_x_axis_and_breakdown_groups() -> None:
    dataframe = pd.DataFrame({
        "ItemType": ["Baby Food"] * 9 + ["Office Supplies"] * 8,
        "Region": ["Europe"] * 5 + ["Asia"] * 4 + ["Europe"] * 4 + ["Asia"] * 4,
        "TotalRevenue": [
            100.0, 120.0, 140.0, 200.0, 500.0,
            50.0, 60.0, 70.0, 80.0,
            90.0, 95.0, 100.0, 105.0,
            110.0, 125.0, 130.0, 140.0,
        ],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="box",
        x="ItemType",
        y="TotalRevenue",
        color="Region",
        title="Revenue Distribution by Item and Region",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, BoxPlotEvidence)
    assert evidence.x_column == "ItemType"
    assert evidence.y_column == "TotalRevenue"
    assert evidence.breakdown_column == "Region"
    assert evidence.x_category_count == 2
    assert evidence.breakdown_category_count == 2
    assert evidence.box_count == 4
    assert {group.display_label for group in evidence.groups} == {
        "Baby Food / Asia",
        "Baby Food / Europe",
        "Office Supplies / Asia",
        "Office Supplies / Europe",
    }
    assert evidence.highest_median_combination == "Baby Food / Europe"
    assert evidence.lowest_median_combination == "Baby Food / Asia"
    assert evidence.widest_iqr_combination == "Baby Food / Europe"
    assert evidence.breakdown_leader_by_x == {
        "Baby Food": "Europe",
        "Office Supplies": "Asia",
    }
    assert evidence.breakdown_lead_counts == {"Asia": 1, "Europe": 1}
    assert evidence.x_categories_with_ranking_changes
    assert evidence.total_potential_outlier_count == 1
    assert "global median" not in insight.key_finding.lower()
    assert "Baby Food / Europe" in insight.supporting_evidence
    assert "leader" in insight.supporting_evidence.lower()
    assert "units sold and unit price" in insight.recommended_next_step.lower()


def test_box_plot_single_x_category_focuses_on_within_category_breakdown() -> None:
    dataframe = pd.DataFrame({
        "ItemType": ["Baby Food"] * 8,
        "Region": ["Europe"] * 4 + ["Asia"] * 4,
        "TotalProfit": [20.0, 25.0, 28.0, 30.0, 8.0, 10.0, 12.0, 14.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="box",
        x="ItemType",
        y="TotalProfit",
        color="Region",
        title="Profit Distribution by Item and Region",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, BoxPlotEvidence)
    assert evidence.x_category_count == 1
    assert evidence.box_count == 2
    assert "Within Baby Food" in insight.interpretation
    assert "across item type categories" not in insight.interpretation.lower()
    assert "profit margin and total cost" in insight.recommended_next_step.lower()


def test_box_plot_missing_combinations_and_unequal_sample_sizes_are_reported() -> None:
    dataframe = pd.DataFrame({
        "ItemType": ["A"] * 8 + ["B"] * 3,
        "Region": ["East"] * 6 + ["West"] * 2 + ["East"] * 3,
        "Sales": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 30.0, 32.0, 20.0, 22.0, 24.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="box",
        x="ItemType",
        y="Sales",
        color="Region",
        title="Sales Distribution by Item and Region",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, BoxPlotEvidence)
    assert evidence.x_category_count == 2
    assert evidence.breakdown_category_count == 2
    assert evidence.box_count == 3
    assert evidence.unequal_sample_sizes is True
    assert any("fewer observations" in warning.lower() for warning in evidence.warnings)
    assert "fewer observations" in insight.caution.lower()


def test_box_plot_llm_failure_uses_box_prompt_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import llm_service

    dataframe = pd.DataFrame({
        "ItemType": ["A"] * 4 + ["B"] * 4,
        "Region": ["East"] * 2 + ["West"] * 2 + ["East"] * 2 + ["West"] * 2,
        "TotalRevenue": [10.0, 12.0, 20.0, 22.0, 30.0, 32.0, 15.0, 17.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="box",
        x="ItemType",
        y="TotalRevenue",
        color="Region",
        title="Revenue Box",
    ))
    evidence = extract_chart_evidence(result)
    fallback = generate_chart_insight(result)
    prompts = []

    class FakeResponse:
        content = "not valid json"

    class FakeModel:
        def invoke(self, prompt: str) -> FakeResponse:
            prompts.append(prompt)
            return FakeResponse()

    monkeypatch.setattr(llm_service, "_model", lambda settings, model_name: FakeModel())
    insight, _ = llm_service.explain_chart_insight_with_ollama(
        evidence,
        fallback,
        Settings(),
        "fake-model",
    )

    assert isinstance(evidence, BoxPlotEvidence)
    assert insight == fallback
    assert len(prompts) == 2
    assert "Box Plot" in prompts[0]
    assert "Do not use a global median" in prompts[0]


def test_multi_metric_insight_explains_each_metric_separately() -> None:
    dataframe = pd.DataFrame({
        "Region": ["West", "East"],
        "TotalRevenue": [100.0, 80.0],
        "TotalProfit": [10.0, 20.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="bar",
        x="Region",
        y="TotalRevenue",
        value_columns=["TotalRevenue", "TotalProfit"],
        aggregation="sum",
        title="Revenue and Profit by Region",
    ))

    insight = generate_chart_insight(result)

    assert any("West" in item and "TotalRevenue" in item for item in insight.observations)
    assert any("East" in item and "TotalProfit" in item for item in insight.observations)


def test_pie_dispatch_uses_dedicated_part_to_whole_evidence() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Sub-Saharan Africa", "Europe", "Asia", "North America"],
        "TotalRevenue": [356_700_000.0, 353_100_000.0, 250_000_000.0, 25_000_000.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="pie",
        x="Region",
        y="TotalRevenue",
        aggregation="sum",
        sort_descending=True,
        title="Total Revenue Share by Region",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, PieChartEvidence)
    assert not isinstance(evidence, SingleBarEvidence)
    assert evidence.largest_category == "Sub-Saharan Africa"
    assert evidence.second_category == "Europe"
    assert evidence.smallest_category == "North America"
    assert evidence.largest_share == pytest.approx(36.22, rel=0.01)
    assert evidence.second_share == pytest.approx(35.85, rel=0.01)
    assert evidence.top_two_share == pytest.approx(72.07, rel=0.01)
    assert evidence.top_three_share == pytest.approx(97.46, rel=0.01)
    assert evidence.leader_to_second_gap == pytest.approx(3_600_000.0)
    assert evidence.leader_to_second_gap_percent == pytest.approx(1.0195, rel=0.01)
    assert evidence.lead_strength == "narrow"
    assert evidence.part_to_whole_valid is True
    assert "contributes the largest share" in insight.key_finding
    assert "narrowly ahead of Europe" in insight.key_finding
    assert "highest sum Total Revenue" not in " ".join(insight.observations)
    assert "displayed Region are" not in " ".join(insight.observations)
    assert "TotalRevenue" not in " ".join(insight.observations)
    assert "profit margin" in insight.recommended_next_step.lower()


def test_pie_concentration_small_slices_and_other_slice() -> None:
    dataframe = pd.DataFrame({
        "Segment": ["A", "B", "C", "Other", "Tiny"],
        "Sales": [80.0, 10.0, 5.0, 3.0, 2.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="pie",
        x="Segment",
        y="Sales",
        aggregation="sum",
        sort_descending=True,
        title="Sales Share by Segment",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, PieChartEvidence)
    assert evidence.lead_strength == "clear"
    assert evidence.concentration_level == "highly concentrated"
    assert evidence.small_slice_categories == ["Tiny"]
    assert evidence.other_category_present is True
    assert evidence.other_category_share == pytest.approx(3.0)
    assert "highly concentrated" in insight.interpretation
    assert "Other" in insight.supporting_evidence
    assert "small slices" in insight.caution.lower()


def test_pie_top_n_warning_only_when_categories_are_removed() -> None:
    dataframe = pd.DataFrame({
        "Region": ["A", "B", "C", "D", "E"],
        "Revenue": [50.0, 40.0, 30.0, 20.0, 10.0],
    })
    _, limited_result = create_chart(dataframe, ChartSpec(
        chart_type="pie",
        x="Region",
        y="Revenue",
        aggregation="sum",
        sort_descending=True,
        limit=3,
        title="Top Revenue Share",
    ))
    _, full_result = create_chart(dataframe.head(3), ChartSpec(
        chart_type="pie",
        x="Region",
        y="Revenue",
        aggregation="sum",
        sort_descending=True,
        limit=3,
        title="Revenue Share",
    ))

    limited = generate_chart_insight(limited_result)
    full = generate_chart_insight(full_result)

    assert isinstance(limited.evidence, PieChartEvidence)
    assert limited.evidence.top_n_applied == 3
    assert "top 3 of 5" in limited.supporting_evidence.lower()
    assert isinstance(full.evidence, PieChartEvidence)
    assert full.evidence.top_n_applied is None
    assert "top 3" not in full.supporting_evidence.lower()


def test_pie_non_additive_negative_and_zero_total_safeguards() -> None:
    mean_result = create_chart(pd.DataFrame({
        "Region": ["A", "B", "C"],
        "Score": [10.0, 20.0, 30.0],
    }), ChartSpec(
        chart_type="pie",
        x="Region",
        y="Score",
        aggregation="mean",
        title="Average Score Share",
    ))[1]
    negative_result = create_chart(pd.DataFrame({
        "Region": ["A", "B", "C"],
        "Profit": [100.0, -40.0, 20.0],
    }), ChartSpec(
        chart_type="pie",
        x="Region",
        y="Profit",
        aggregation="sum",
        title="Profit Share",
    ))[1]
    zero_result = create_chart(pd.DataFrame({
        "Region": ["A", "B"],
        "Profit": [0.0, 0.0],
    }), ChartSpec(
        chart_type="pie",
        x="Region",
        y="Profit",
        aggregation="sum",
        title="Zero Profit Share",
    ))[1]

    mean_evidence = extract_chart_evidence(mean_result)
    negative_evidence = extract_chart_evidence(negative_result)
    zero_evidence = extract_chart_evidence(zero_result)
    mean_insight = generate_chart_insight(mean_result)

    assert isinstance(mean_evidence, PieChartEvidence)
    assert mean_evidence.part_to_whole_valid is False
    assert mean_evidence.evidence_strength == "low"
    assert "not additive" in " ".join(mean_evidence.warnings).lower()
    assert "sorted bar chart" in mean_insight.recommended_next_step.lower()
    assert isinstance(negative_evidence, PieChartEvidence)
    assert negative_evidence.part_to_whole_valid is False
    assert negative_evidence.evidence_strength == "low"
    assert any("negative" in warning.lower() for warning in negative_evidence.warnings)
    assert isinstance(zero_evidence, PieChartEvidence)
    assert zero_evidence.part_to_whole_valid is False
    assert zero_evidence.evidence_strength == "low"
    assert any("zero or negative" in warning.lower() for warning in zero_evidence.warnings)


def test_pie_llm_failure_uses_pie_prompt_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import llm_service

    dataframe = pd.DataFrame({
        "Priority": ["High", "Medium", "Low"],
        "OrderID": [10.0, 7.0, 3.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="pie",
        x="Priority",
        y="OrderID",
        aggregation="count",
        title="Orders by Priority",
    ))
    evidence = extract_chart_evidence(result)
    fallback = generate_chart_insight(result)
    prompts = []

    class FakeResponse:
        content = "not valid json"

    class FakeModel:
        def invoke(self, prompt: str) -> FakeResponse:
            prompts.append(prompt)
            return FakeResponse()

    monkeypatch.setattr(llm_service, "_model", lambda settings, model_name: FakeModel())
    insight, _ = llm_service.explain_chart_insight_with_ollama(
        evidence,
        fallback,
        Settings(),
        "fake-model",
    )

    assert isinstance(evidence, PieChartEvidence)
    assert insight == fallback
    assert len(prompts) == 2
    assert "Pie or Donut chart" in prompts[0]
    assert "highest sum Total Revenue" in prompts[0]


def test_treemap_uses_dedicated_area_share_evidence_and_wording() -> None:
    dataframe = pd.DataFrame({
        "Region": [
            "Sub-Saharan Africa",
            "Europe",
            "Middle East and North Africa",
            "Asia",
            "Central America and the Caribbean",
            "Australia and Oceania",
            "North America",
        ],
        "TotalRevenue": [
            356_700_000.0,
            353_100_000.0,
            250_000_000.0,
            210_000_000.0,
            190_000_000.0,
            120_000_000.0,
            25_000_000.0,
        ],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="treemap",
        x="Region",
        y="TotalRevenue",
        aggregation="sum",
        sort_descending=True,
        title="Tree Map: Sum TotalRevenue by Region",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, TreemapEvidence)
    assert evidence.largest_category == "Sub-Saharan Africa"
    assert evidence.second_category == "Europe"
    assert evidence.smallest_category == "North America"
    assert evidence.largest_share == pytest.approx(23.70, rel=0.01)
    assert evidence.second_share == pytest.approx(23.46, rel=0.01)
    assert evidence.top_two_share == pytest.approx(47.16, rel=0.01)
    assert evidence.lead_strength == "narrow"
    assert "largest area share" in insight.key_finding
    assert "narrowly ahead of Europe" in insight.key_finding
    assert "rectangles" in insight.supporting_evidence.lower()
    assert "highest sum Total Revenue" not in " ".join(insight.observations)
    assert "displayed Region are" not in " ".join(insight.observations)
    assert "TotalRevenue" not in " ".join(insight.observations)


def test_treemap_hierarchy_top_n_and_non_additive_safeguards() -> None:
    dataframe = pd.DataFrame({
        "Continent": ["Asia", "Asia", "Europe", "Europe", "Africa"],
        "Country": ["Japan", "India", "France", "Germany", "Kenya"],
        "Revenue": [100.0, 80.0, 90.0, 70.0, 5.0],
        "Score": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    _, hierarchical = create_chart(dataframe, ChartSpec(
        chart_type="treemap",
        x="Country",
        y="Revenue",
        color="Continent",
        aggregation="sum",
        sort_descending=True,
        limit=3,
        title="Revenue Hierarchy",
    ))
    _, mean_result = create_chart(dataframe, ChartSpec(
        chart_type="treemap",
        x="Country",
        y="Score",
        aggregation="mean",
        title="Score Hierarchy",
    ))

    hierarchy_evidence = extract_chart_evidence(hierarchical)
    hierarchy_insight = generate_chart_insight(hierarchical)
    mean_evidence = extract_chart_evidence(mean_result)
    mean_insight = generate_chart_insight(mean_result)

    assert isinstance(hierarchy_evidence, TreemapEvidence)
    assert hierarchy_evidence.top_n_applied == 3
    assert hierarchy_evidence.largest_group == "Asia"
    assert hierarchy_evidence.largest_group_share == pytest.approx(66.67, rel=0.01)
    assert "top 3 of 5" in hierarchy_insight.supporting_evidence.lower()
    assert "largest parent group" in hierarchy_insight.supporting_evidence.lower()
    assert isinstance(mean_evidence, TreemapEvidence)
    assert mean_evidence.part_to_whole_valid is False
    assert mean_evidence.evidence_strength == "low"
    assert "not additive" in " ".join(mean_evidence.warnings).lower()
    assert "sorted bar chart" in mean_insight.recommended_next_step.lower()


def test_treemap_llm_failure_uses_treemap_prompt_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import llm_service

    dataframe = pd.DataFrame({
        "Region": ["A", "B", "C"],
        "Revenue": [50.0, 30.0, 20.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="treemap",
        x="Region",
        y="Revenue",
        aggregation="sum",
        title="Revenue Tree",
    ))
    evidence = extract_chart_evidence(result)
    fallback = generate_chart_insight(result)
    prompts = []

    class FakeResponse:
        content = "not valid json"

    class FakeModel:
        def invoke(self, prompt: str) -> FakeResponse:
            prompts.append(prompt)
            return FakeResponse()

    monkeypatch.setattr(llm_service, "_model", lambda settings, model_name: FakeModel())
    insight, _ = llm_service.explain_chart_insight_with_ollama(
        evidence,
        fallback,
        Settings(),
        "fake-model",
    )

    assert isinstance(evidence, TreemapEvidence)
    assert insight == fallback
    assert len(prompts) == 2
    assert "Treemap" in prompts[0]
    assert "highest sum Total Revenue" in prompts[0]


def test_symbol_map_uses_dedicated_geospatial_evidence_and_wording() -> None:
    dataframe = pd.DataFrame({
        "Country": ["Cuba", "France", "Germany", "Japan", "India", "Kenya"],
        "Region": [
            "Central America and the Caribbean",
            "Europe",
            "Europe",
            "Asia",
            "Asia",
            "Africa",
        ],
        "TotalRevenue": [100.0, 90.0, 80.0, 70.0, 60.0, 10.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="symbol_map",
        x="Country",
        y="TotalRevenue",
        color="Region",
        aggregation="sum",
        sort_descending=True,
        title="Total Revenue by Country and Region",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)
    text = " ".join(insight.observations)

    assert isinstance(evidence, SymbolMapEvidence)
    assert evidence.largest_location == "Cuba"
    assert evidence.second_location == "France"
    assert evidence.third_location == "Germany"
    assert evidence.smallest_location == "Kenya"
    assert evidence.displayed_location_count == 6
    assert evidence.largest_share == pytest.approx(24.39, rel=0.01)
    assert evidence.top_two_share == pytest.approx(46.34, rel=0.01)
    assert evidence.top_three_share == pytest.approx(65.85, rel=0.01)
    assert evidence.top_five_share == pytest.approx(97.56, rel=0.01)
    assert evidence.largest_group == "Central America and the Caribbean"
    assert evidence.highest_total_group == "Europe"
    assert evidence.highest_total_group_value == pytest.approx(170.0)
    assert evidence.group_with_most_top_locations_count is not None
    assert "country" in text.lower() or "countries" in text.lower()
    assert "map" in text.lower()
    assert "highest sum Total Revenue" not in text
    assert "displayed Country are" not in text
    assert "displayed categories" not in text
    assert "location(s)" not in text


def test_symbol_map_top_n_and_non_additive_safeguards() -> None:
    dataframe = pd.DataFrame({
        "Country": ["Cuba", "France", "Germany", "Japan", "India"],
        "Region": ["Caribbean", "Europe", "Europe", "Asia", "Asia"],
        "Revenue": [100.0, 90.0, 80.0, 70.0, 60.0],
        "Score": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    _, top_result = create_chart(dataframe, ChartSpec(
        chart_type="symbol_map",
        x="Country",
        y="Revenue",
        aggregation="sum",
        sort_descending=True,
        limit=3,
        title="Revenue Map",
    ))
    _, mean_result = create_chart(dataframe, ChartSpec(
        chart_type="symbol_map",
        x="Country",
        y="Score",
        aggregation="mean",
        title="Score Map",
    ))

    top_evidence = extract_chart_evidence(top_result)
    top_insight = generate_chart_insight(top_result)
    mean_evidence = extract_chart_evidence(mean_result)
    mean_insight = generate_chart_insight(mean_result)

    assert isinstance(top_evidence, SymbolMapEvidence)
    assert top_evidence.top_n_applied == 3
    assert top_evidence.original_location_count == 5
    assert "top 3 of 5" in top_insight.supporting_evidence.lower()
    assert isinstance(mean_evidence, SymbolMapEvidence)
    assert mean_evidence.largest_share is None
    assert mean_evidence.evidence_strength == "low"
    assert "not additive" in " ".join(mean_evidence.warnings).lower()
    assert "bubble shares" in mean_insight.caution.lower()


def test_symbol_map_same_location_and_color_insight_uses_virtual_location_color() -> None:
    dataframe = pd.DataFrame({
        "Region": [
            "Sub-Saharan Africa",
            "Europe",
            "Asia",
            "North America",
        ],
        "TotalRevenue": [356_700_000.0, 352_700_000.0, 250_000_000.0, 25_000_000.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="symbol_map",
        x="Region",
        y="TotalRevenue",
        color="Region",
        aggregation="sum",
        sort_descending=True,
        title="Symbol Map: Sum TotalRevenue by Region, by Region",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)
    text = " ".join(insight.observations)

    assert isinstance(evidence, SymbolMapEvidence)
    assert result.spec.color == "__location__"
    assert result.spec.title == "Total Revenue by Region"
    assert evidence.color_column == "__location__"
    assert evidence.color_groups == []
    assert evidence.highest_total_group is None
    assert evidence.largest_location == "Sub-Saharan Africa"
    assert evidence.second_location == "Europe"
    assert evidence.top_two_share == pytest.approx(72.05, rel=0.01)
    assert "map" in text.lower()
    assert "by Region, by Region" not in result.spec.title
    assert "Sum TotalRevenue" not in result.spec.title
    assert "highest sum Total Revenue" not in text
    assert "displayed categories" not in text


def test_symbol_map_llm_failure_uses_symbol_prompt_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import llm_service

    dataframe = pd.DataFrame({
        "Country": ["Cuba", "France", "Germany"],
        "Revenue": [100.0, 90.0, 80.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="symbol_map",
        x="Country",
        y="Revenue",
        aggregation="sum",
        title="Revenue Map",
    ))
    evidence = extract_chart_evidence(result)
    fallback = generate_chart_insight(result)
    prompts = []

    class FakeResponse:
        content = "not valid json"

    class FakeModel:
        def invoke(self, prompt: str) -> FakeResponse:
            prompts.append(prompt)
            return FakeResponse()

    monkeypatch.setattr(llm_service, "_model", lambda settings, model_name: FakeModel())
    insight, _ = llm_service.explain_chart_insight_with_ollama(
        evidence,
        fallback,
        Settings(),
        "fake-model",
    )

    assert isinstance(evidence, SymbolMapEvidence)
    assert insight == fallback
    assert len(prompts) == 2
    assert "Symbol Map" in prompts[0]
    assert "geographic distribution" in prompts[0]


def test_sorted_percentage_bar_evidence_sorting_shares_and_insight() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Sub-Saharan Africa", "Europe", "Asia", "North America"],
        "TotalRevenue": [356_700_000.0, 353_100_000.0, 250_000_000.0, 25_000_000.0],
    })
    figure, result = create_chart(dataframe, ChartSpec(
        chart_type="sorted_percentage_bar",
        x="Region",
        y="TotalRevenue",
        aggregation="sum",
        sort_descending=True,
        title="Share of Total Revenue by Region",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert result.data[0]["Region"] == "Sub-Saharan Africa"
    assert result.data[0]["percentage_share"] == pytest.approx(36.22, rel=0.01)
    assert figure.data[0].orientation != "h"
    assert isinstance(evidence, SortedPercentageBarEvidence)
    assert evidence.largest_category == "Sub-Saharan Africa"
    assert evidence.second_category == "Europe"
    assert evidence.smallest_category == "North America"
    assert evidence.leader_to_second_gap_percentage_points == pytest.approx(0.37, rel=0.05)
    assert evidence.lead_strength == "narrow"
    assert evidence.top_two_share == pytest.approx(72.07, rel=0.01)
    assert "contributes the largest share" in insight.key_finding
    assert "percentage points" in insight.supporting_evidence
    assert "highest sum Total Revenue" not in " ".join(insight.observations)
    assert "TotalRevenue" not in " ".join(insight.observations)


def test_sorted_percentage_bar_date_filter_top_n_and_denominator_modes() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-01-01", "2026-01-02", "2026-01-03",
            "2026-02-01", "2026-02-02",
        ]),
        "Region": ["A", "B", "C", "A", "B"],
        "Revenue": [50.0, 30.0, 20.0, 100.0, 100.0],
    })
    _, full_result = create_chart(dataframe, ChartSpec(
        chart_type="sorted_percentage_bar",
        x="Region",
        y="Revenue",
        aggregation="sum",
        sort_descending=True,
        limit=2,
        time_column="Date",
        date_range_start="2026-01-01",
        date_range_end="2026-01-31",
        percentage_denominator_mode="full_filtered",
        title="January Revenue Share",
    ))
    _, displayed_result = create_chart(dataframe, ChartSpec(
        chart_type="sorted_percentage_bar",
        x="Region",
        y="Revenue",
        aggregation="sum",
        sort_descending=True,
        limit=2,
        time_column="Date",
        date_range_start="2026-01-01",
        date_range_end="2026-01-31",
        percentage_denominator_mode="displayed",
        include_other=True,
        title="January Revenue Share",
    ))

    full_evidence = extract_chart_evidence(full_result)
    displayed_evidence = extract_chart_evidence(displayed_result)

    assert isinstance(full_evidence, SortedPercentageBarEvidence)
    assert [row["Region"] for row in full_result.data] == ["A", "B"]
    assert full_result.data[0]["percentage_share"] == pytest.approx(50.0)
    assert full_result.data[1]["percentage_share"] == pytest.approx(30.0)
    assert full_evidence.top_n_applied == 2
    assert full_evidence.date_column == "Date"
    assert "Top-N is active" in " ".join(full_evidence.warnings)
    assert isinstance(displayed_evidence, SortedPercentageBarEvidence)
    assert displayed_evidence.other_category_present is True
    assert displayed_evidence.other_share == pytest.approx(20.0)
    assert sum(row["percentage_share"] for row in displayed_result.data) == pytest.approx(100.0)


def test_sorted_percentage_bar_ascending_horizontal_and_invalid_percentages() -> None:
    many = pd.DataFrame({
        "Channel": [f"C{i}" for i in range(7)],
        "UnitsSold": [7, 6, 5, 4, 3, 2, 1],
    })
    figure, ascending = create_chart(many, ChartSpec(
        chart_type="sorted_percentage_bar",
        x="Channel",
        y="UnitsSold",
        aggregation="sum",
        sort_descending=False,
        title="Unit Share",
    ))
    negative = create_chart(pd.DataFrame({
        "Region": ["A", "B", "C"],
        "Profit": [100.0, -50.0, 25.0],
    }), ChartSpec(
        chart_type="sorted_percentage_bar",
        x="Region",
        y="Profit",
        aggregation="sum",
        title="Profit Share",
    ))[1]
    mean_result = create_chart(pd.DataFrame({
        "Region": ["A", "B", "C"],
        "UnitPrice": [10.0, 20.0, 30.0],
    }), ChartSpec(
        chart_type="sorted_percentage_bar",
        x="Region",
        y="UnitPrice",
        aggregation="mean",
        title="Average Price Share",
    ))[1]

    assert figure.data[0].orientation == "h"
    assert figure.data[0].x[0] == pytest.approx(1 / 28 * 100)
    negative_evidence = extract_chart_evidence(negative)
    mean_evidence = extract_chart_evidence(mean_result)
    assert isinstance(negative_evidence, SortedPercentageBarEvidence)
    assert negative_evidence.percentage_valid is False
    assert negative_evidence.evidence_strength == "low"
    assert any("negative" in warning.lower() for warning in negative_evidence.warnings)
    assert isinstance(mean_evidence, SortedPercentageBarEvidence)
    assert mean_evidence.additive_aggregation is False
    assert mean_evidence.evidence_strength == "low"
    assert "sorted bar chart" in generate_chart_insight(mean_result).recommended_next_step.lower()


def test_sorted_percentage_bar_llm_failure_uses_dedicated_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import llm_service

    dataframe = pd.DataFrame({
        "Region": ["A", "B", "C"],
        "Revenue": [50.0, 30.0, 20.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="sorted_percentage_bar",
        x="Region",
        y="Revenue",
        aggregation="sum",
        title="Share of Revenue by Region",
    ))
    evidence = extract_chart_evidence(result)
    fallback = generate_chart_insight(result)
    prompts = []

    class FakeResponse:
        content = "not valid json"

    class FakeModel:
        def invoke(self, prompt: str) -> FakeResponse:
            prompts.append(prompt)
            return FakeResponse()

    monkeypatch.setattr(llm_service, "_model", lambda settings, model_name: FakeModel())
    insight, _ = llm_service.explain_chart_insight_with_ollama(
        evidence,
        fallback,
        Settings(),
        "fake-model",
    )

    assert isinstance(evidence, SortedPercentageBarEvidence)
    assert insight == fallback
    assert len(prompts) == 2
    assert "Sorted Percentage Bar" in prompts[0]
    assert "percentage points" in prompts[0]


def test_period_over_period_change_uses_baseline_before_display_range() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime(["2025-12-01", "2026-01-01", "2026-02-01", "2026-03-01"]),
        "TotalRevenue": [100.0, 125.0, 100.0, 150.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="period_over_period_change",
        x="Date",
        y="TotalRevenue",
        aggregation="sum",
        time_grain="month",
        comparison_basis="previous_period",
        date_range_start="2026-01-01",
        date_range_end="2026-03-31",
        title="Monthly Revenue Change",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, PeriodOverPeriodChangeEvidence)
    assert result.data[0]["percentage_change"] == pytest.approx(25.0)
    assert evidence.period_count == 3
    assert evidence.comparable_period_count == 3
    assert evidence.latest_period == "March 2026"
    assert evidence.latest_percent_change == pytest.approx(50.0)
    assert evidence.largest_decline_percent == pytest.approx(-20.0)
    assert "versus the previous period" in insight.key_finding
    assert "line chart" not in insight.interpretation.lower()


def test_period_over_period_same_period_last_year_and_zero_baseline() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-01", "2025-02-01", "2026-01-01", "2026-02-01"]),
        "TotalProfit": [100.0, 0.0, 150.0, 10.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="period_over_period_change",
        x="Date",
        y="TotalProfit",
        aggregation="sum",
        time_grain="month",
        comparison_basis="same_period_last_year",
        date_range_start="2026-01-01",
        date_range_end="2026-02-28",
        title="YoY Profit Change",
    ))

    evidence = extract_chart_evidence(result)

    assert isinstance(evidence, PeriodOverPeriodChangeEvidence)
    assert result.data[0]["percentage_change"] == pytest.approx(50.0)
    assert pd.isna(result.data[1]["percentage_change"])
    assert evidence.zero_baseline_count == 1
    assert evidence.comparable_period_count == 1
    assert any("zero" in warning.lower() for warning in evidence.warnings)


def test_period_over_period_llm_failure_uses_dedicated_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import llm_service

    dataframe = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=4, freq="MS"),
        "UnitsSold": [10.0, 15.0, 12.0, 18.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="period_over_period_change",
        x="Date",
        y="UnitsSold",
        aggregation="sum",
        time_grain="month",
        title="Monthly Unit Change",
    ))
    evidence = extract_chart_evidence(result)
    fallback = generate_chart_insight(result)
    prompts = []

    class FakeResponse:
        content = "not valid json"

    class FakeModel:
        def invoke(self, prompt: str) -> FakeResponse:
            prompts.append(prompt)
            return FakeResponse()

    monkeypatch.setattr(llm_service, "_model", lambda settings, model_name: FakeModel())
    insight, _ = llm_service.explain_chart_insight_with_ollama(
        evidence,
        fallback,
        Settings(),
        "fake-model",
    )

    assert isinstance(evidence, PeriodOverPeriodChangeEvidence)
    assert insight == fallback
    assert len(prompts) == 2
    assert "Period-over-Period % Change" in prompts[0]
    assert "normal line-chart wording" in prompts[0]


def test_standard_bar_dispatch_uses_single_bar_handler_only_for_plain_bar() -> None:
    dataframe = _standard_bar_dataframe()
    _, plain_result = create_chart(dataframe, ChartSpec(
        chart_type="bar",
        x="Region",
        y="TotalRevenue",
        aggregation="sum",
        title="Total Revenue by Region",
    ))
    _, grouped_result = create_chart(dataframe, ChartSpec(
        chart_type="grouped_bar",
        x="Region",
        y="TotalRevenue",
        color="Channel",
        aggregation="sum",
        title="Revenue by Region and Channel",
    ))
    _, stacked_result = create_chart(dataframe, ChartSpec(
        chart_type="stacked_bar",
        x="Region",
        y="TotalRevenue",
        color="Channel",
        aggregation="sum",
        title="Stacked Revenue",
    ))

    assert isinstance(extract_chart_evidence(plain_result), SingleBarEvidence)
    assert isinstance(extract_chart_evidence(grouped_result), GroupedBarEvidence)
    assert not isinstance(extract_chart_evidence(stacked_result), SingleBarEvidence)


def test_standard_bar_ranking_gap_and_concentration_evidence() -> None:
    _, result = create_chart(_standard_bar_dataframe(), ChartSpec(
        chart_type="bar",
        x="Region",
        y="TotalRevenue",
        aggregation="sum",
        title="Total Revenue by Region",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, SingleBarEvidence)
    assert evidence.highest_category == "Sub-Saharan Africa"
    assert evidence.second_highest_category == "Europe"
    assert evidence.lowest_category == "North America"
    assert evidence.lead_strength == "narrow"
    assert evidence.leader_to_second_gap == pytest.approx(3_600_000.0)
    assert evidence.leader_to_second_gap_percent == pytest.approx(1.0195, rel=0.01)
    assert evidence.leader_to_second_gap_basis == "Europe"
    assert evidence.highest_share_percent == pytest.approx(36.22, rel=0.01)
    assert evidence.top_two_share_percent == pytest.approx(72.07, rel=0.01)
    assert evidence.concentration_level == "high"
    assert "narrowly ahead of Europe" in insight.key_finding
    assert "relative to Europe" in insight.supporting_evidence
    assert "Together, Sub-Saharan Africa and Europe" in insight.supporting_evidence
    assert "highest sum Total Revenue" not in " ".join(insight.observations)
    assert "spread across multiple categories" not in " ".join(insight.observations)
    assert "TotalRevenue" not in " ".join(insight.observations)


def test_standard_bar_clear_lead_and_non_additive_share_handling() -> None:
    dataframe = pd.DataFrame({
        "Region": ["A", "B", "C"],
        "UnitPrice": [100.0, 50.0, 45.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="bar",
        x="Region",
        y="UnitPrice",
        aggregation="mean",
        title="Average Price by Region",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, SingleBarEvidence)
    assert evidence.lead_strength == "clear"
    assert evidence.highest_share_percent is None
    assert evidence.top_two_share_percent is None
    assert "average unit price" in insight.key_finding
    assert "representing" not in insight.supporting_evidence
    assert "average can be affected" in insight.caution


def test_standard_bar_top_n_filter_and_metric_specific_text() -> None:
    dataframe = pd.DataFrame({
        "Country": [f"Country {index}" for index in range(6)],
        "TotalProfit": [100.0, 90.0, 80.0, 70.0, 60.0, 50.0],
        "Channel": ["Online"] * 6,
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="bar",
        x="Country",
        y="TotalProfit",
        aggregation="sum",
        sort_descending=True,
        limit=3,
        filter_column="Channel",
        filter_value="Online",
        title="Profit by Country",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, SingleBarEvidence)
    assert evidence.top_n_applied == 3
    assert "top 3" in insight.supporting_evidence.lower()
    assert "Channel = Online" in insight.supporting_evidence
    assert "profit margin" in insight.caution
    assert "total cost" in insight.recommended_next_step


def test_standard_bar_units_caution_and_next_step() -> None:
    _, result = create_chart(_standard_bar_dataframe(), ChartSpec(
        chart_type="bar",
        x="Region",
        y="UnitsSold",
        aggregation="sum",
        title="Units Sold by Region",
    ))

    insight = generate_chart_insight(result)

    assert "Units sold measures volume" in insight.caution
    assert "revenue per unit" in insight.recommended_next_step


def test_stacked_bar_insight_uses_combined_totals_and_stack_composition() -> None:
    dataframe = pd.DataFrame({
        "Region": ["A", "A", "B", "B", "C", "C"],
        "SalesChannel": ["Online", "Offline", "Online", "Offline", "Online", "Offline"],
        "TotalRevenue": [100.0, 1.0, 60.0, 60.0, 80.0, 10.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="stacked_bar",
        x="Region",
        y="TotalRevenue",
        color="SalesChannel",
        aggregation="sum",
        sort_descending=True,
        title="Revenue by Region and Channel",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, StackedBarEvidence)
    assert evidence.highest_combined_category == "B"
    assert evidence.highest_combined_value == pytest.approx(120.0)
    assert evidence.highest_category_dominant_stack in {"Offline", "Online"}
    assert "B records the highest combined revenue" in insight.key_finding
    assert "across all sales channel segments" in insight.supporting_evidence
    assert "bar height as the combined total" in insight.interpretation
    assert "highest sum TotalRevenue" not in " ".join(insight.observations)


def test_standard_bar_llm_failure_uses_single_bar_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import llm_service

    _, result = create_chart(_standard_bar_dataframe(), ChartSpec(
        chart_type="bar",
        x="Region",
        y="TotalRevenue",
        aggregation="sum",
        title="Total Revenue by Region",
    ))
    evidence = extract_chart_evidence(result)
    fallback = generate_chart_insight(result)
    prompts = []

    class FakeResponse:
        content = "not valid json"

    class FakeModel:
        def invoke(self, prompt: str) -> FakeResponse:
            prompts.append(prompt)
            return FakeResponse()

    monkeypatch.setattr(llm_service, "_model", lambda settings, model_name: FakeModel())
    insight, _ = llm_service.explain_chart_insight_with_ollama(
        evidence,
        fallback,
        Settings(),
        "fake-model",
    )

    assert isinstance(evidence, SingleBarEvidence)
    assert insight == fallback
    assert len(prompts) == 2
    assert "standard single-series bar chart" in prompts[0]
    assert "Do not introduce grouped-series or stacked-bar language" in prompts[0]


def test_grouped_bar_evidence_separates_combined_totals_from_segments() -> None:
    _, result = create_chart(_grouped_priority_dataframe(), ChartSpec(
        chart_type="grouped_bar",
        x="Region",
        y="TotalProfit",
        color="OrderPriority",
        aggregation="sum",
        title="Side-by-Side Bar: Sum Total Profit by Region and Order Priority",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, GroupedBarEvidence)
    assert evidence.category_totals["Europe"] == pytest.approx(105.0)
    assert evidence.category_totals["North America"] == pytest.approx(10.0)
    assert evidence.highest_combined_category == "Europe"
    assert evidence.highest_combined_value == pytest.approx(105.0)
    assert evidence.lowest_combined_category == "North America"
    assert evidence.lowest_combined_value == pytest.approx(10.0)
    assert evidence.highest_individual_category == "Europe"
    assert evidence.highest_individual_group == "Low"
    assert evidence.highest_individual_value == pytest.approx(70.0)
    assert "$105" in insight.key_finding or "$105" in insight.supporting_evidence
    assert "$70" in insight.supporting_evidence
    assert "compared with $10 in North America" in insight.supporting_evidence
    assert "North America records the lowest" in insight.key_finding
    assert "Europe-Low" not in insight.supporting_evidence


def test_grouped_bar_winners_gaps_balance_and_missing_combinations() -> None:
    _, result = create_chart(_grouped_priority_dataframe(), ChartSpec(
        chart_type="grouped_bar",
        x="Region",
        y="TotalProfit",
        color="OrderPriority",
        aggregation="sum",
        title="Profit by Region and Priority",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, GroupedBarEvidence)
    assert evidence.winner_by_category == {
        "Asia": "Medium",
        "Europe": "Low",
        "North America": "Medium",
        "Sub-Saharan Africa": "Critical",
    }
    assert evidence.group_win_counts == {"Medium": 2, "Low": 1, "Critical": 1}
    assert evidence.largest_gap_category == "Europe"
    assert evidence.largest_gap_groups == ["Low", "High"]
    assert evidence.largest_gap_value == pytest.approx(50.0)
    assert evidence.largest_gap_percent == pytest.approx(250.0)
    assert evidence.most_balanced_category == "Asia"
    assert evidence.smallest_gap_value == pytest.approx(1.0)
    assert {"category": "Sub-Saharan Africa", "group": "Medium"} in evidence.missing_group_combinations
    assert evidence.evidence_strength == "medium"
    assert "not present in the displayed data" in insight.caution
    assert "Low" in insight.supporting_evidence
    assert "varies by region" in insight.interpretation.lower()


def test_grouped_bar_gap_percent_handles_zero_runner_up_safely() -> None:
    result = ChartResult(
        spec=ChartSpec(
            chart_type="grouped_bar",
            x="Region",
            y="TotalProfit",
            color="OrderPriority",
            aggregation="sum",
            title="Zero Gap",
        ),
        data=[
            {"Region": "A", "OrderPriority": "C", "TotalProfit": 10.0},
            {"Region": "A", "OrderPriority": "H", "TotalProfit": 0.0},
        ],
    )

    evidence = extract_chart_evidence(result)

    assert isinstance(evidence, GroupedBarEvidence)
    assert evidence.winner_gap_by_category["A"] == pytest.approx(10.0)
    assert evidence.winner_gap_percent_by_category["A"] is None
    assert evidence.largest_gap_percent is None


def test_grouped_bar_top_n_and_filters_are_reported_only_when_active() -> None:
    default_result = create_chart(_grouped_priority_dataframe(), ChartSpec(
        chart_type="grouped_bar",
        x="Region",
        y="TotalProfit",
        color="OrderPriority",
        aggregation="sum",
        limit=99,
        title="No Top 99 Warning",
    ))[1]
    default_insight = generate_chart_insight(default_result)
    assert "top 99" not in default_insight.supporting_evidence.lower()

    many_rows = pd.DataFrame({
        "Region": [f"Region {index:02d}" for index in range(30) for _ in range(2)],
        "OrderPriority": ["C", "H"] * 30,
        "TotalProfit": [float(index + 1) for index in range(60)],
        "Channel": ["Online"] * 60,
    })
    limited_result = create_chart(many_rows, ChartSpec(
        chart_type="grouped_bar",
        x="Region",
        y="TotalProfit",
        color="OrderPriority",
        aggregation="sum",
        sort_descending=True,
        limit=20,
        filter_column="Channel",
        filter_value="Online",
        title="Limited Grouped Profit",
    ))[1]
    limited_insight = generate_chart_insight(limited_result)
    limited_evidence = extract_chart_evidence(limited_result)

    assert isinstance(limited_evidence, GroupedBarEvidence)
    assert limited_evidence.top_n_applied == 20
    assert "top 20" in limited_insight.supporting_evidence.lower()
    assert "Channel = Online" in limited_insight.supporting_evidence


def test_grouped_bar_llm_uses_structured_evidence_and_invalid_output_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import llm_service

    _, result = create_chart(_grouped_priority_dataframe(), ChartSpec(
        chart_type="grouped_bar",
        x="Region",
        y="TotalProfit",
        color="OrderPriority",
        aggregation="sum",
        title="Profit by Region and Priority",
    ))
    evidence = extract_chart_evidence(result)
    fallback = generate_chart_insight(result)
    prompts = []

    class FakeResponse:
        content = "not valid json"

    class FakeModel:
        def invoke(self, prompt: str) -> FakeResponse:
            prompts.append(prompt)
            return FakeResponse()

    monkeypatch.setattr(llm_service, "_model", lambda settings, model_name: FakeModel())

    insight, _ = llm_service.explain_chart_insight_with_ollama(
        evidence,
        fallback,
        Settings(),
        "fake-model",
    )

    assert insight == fallback
    assert len(prompts) == 2
    assert "concise business data-analysis writer" in prompts[0]
    assert "values_by_category_and_group" in prompts[0]
    assert "Use only the supplied grouped-bar evidence" in prompts[0]


def test_grouped_bar_written_insight_is_clean_and_non_repetitive() -> None:
    _, result = create_chart(_grouped_priority_dataframe(), ChartSpec(
        chart_type="grouped_bar",
        x="Region",
        y="TotalProfit",
        color="OrderPriority",
        aggregation="sum",
        title="Profit by Region and Priority",
    ))

    insight = generate_chart_insight(result)
    combined_text = " ".join([
        insight.key_finding,
        insight.supporting_evidence,
        insight.interpretation,
        insight.caution or "",
        insight.recommended_next_step,
    ])

    assert ChartInsight.model_validate(insight.model_dump())
    assert all(sentence[0].isupper() for sentence in _sentences(combined_text))
    assert "where Low over High" not in combined_text
    assert "TotalProfit" not in combined_text
    assert "OrderPriority" not in combined_text
    assert "win counts" not in combined_text.lower()
    assert "relative to High profit" in insight.supporting_evidence
    assert "spread across multiple categories" not in combined_text
    assert "cause" not in combined_text.lower()
    assert "profit margin" in insight.caution
    assert "profit per order" in insight.recommended_next_step


def test_grouped_bar_metric_specific_caution_and_next_step_change_by_measure() -> None:
    dataframe = pd.DataFrame({
        "Region": ["A", "A", "B", "B"],
        "SalesChannel": ["Online", "Offline", "Online", "Offline"],
        "TotalRevenue": [100.0, 150.0, 90.0, 70.0],
        "UnitsSold": [10, 20, 12, 6],
    })
    _, revenue_result = create_chart(dataframe, ChartSpec(
        chart_type="grouped_bar",
        x="Region",
        y="TotalRevenue",
        color="SalesChannel",
        aggregation="sum",
        title="Revenue by Region and Channel",
    ))
    _, units_result = create_chart(dataframe, ChartSpec(
        chart_type="grouped_bar",
        x="Region",
        y="UnitsSold",
        color="SalesChannel",
        aggregation="sum",
        title="Units by Region and Channel",
    ))

    revenue_insight = generate_chart_insight(revenue_result)
    units_insight = generate_chart_insight(units_result)

    assert "unit prices" in revenue_insight.caution
    assert "profit margin" in revenue_insight.recommended_next_step
    assert "revenue per unit" in units_insight.recommended_next_step
    assert "average price" in units_insight.caution


def test_top_n_filter_and_missing_rows_are_reported_in_evidence() -> None:
    _, result = create_chart(pd.DataFrame({
        "Region": ["A", "B", "C"],
        "Revenue": [3.0, 2.0, 1.0],
    }), ChartSpec(
        chart_type="bar",
        x="Region",
        y="Revenue",
        aggregation="sum",
        limit=2,
        filter_column="Region",
        filter_value="A",
        title="Filtered Revenue",
    ))

    insight = generate_chart_insight(result)

    assert insight.evidence
    assert insight.evidence.top_n == 2
    assert insight.evidence.filters == {"Region": "A"}
    assert "top 2" not in insight.supporting_evidence
    assert "Region = A" in insight.supporting_evidence


def test_sales_formula_dependency_detection_for_scatter_evidence() -> None:
    result = ChartResult(
        spec=ChartSpec(
            chart_type="scatter",
            x="UnitsSold",
            y="TotalRevenue",
            title="Revenue vs Units",
        ),
        data=[
            {
                "UnitsSold": 10,
                "UnitPrice": 2.5,
                "UnitCost": 1.5,
                "TotalRevenue": 25.0,
                "TotalCost": 15.0,
                "TotalProfit": 10.0,
            },
            {
                "UnitsSold": 20,
                "UnitPrice": 3.0,
                "UnitCost": 1.0,
                "TotalRevenue": 60.0,
                "TotalCost": 20.0,
                "TotalProfit": 40.0,
            },
        ],
    )

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert evidence.calculated_metrics["revenue_formula_match_pct"] == pytest.approx(100.0)
    assert evidence.calculated_metrics["cost_formula_match_pct"] == pytest.approx(100.0)
    assert evidence.calculated_metrics["profit_formula_match_pct"] == pytest.approx(100.0)
    assert any("Revenue may be mathematically derived" in warning for warning in evidence.warnings)
    assert "cause" not in insight.interpretation.lower()


def test_scatter_uses_dedicated_evidence_and_observation_wording_without_false_top_n() -> None:
    dataframe = pd.DataFrame({
        "MetricX": [1, 2, 3, 4, 5],
        "MetricY": [2, 4, 6, 8, 10],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="scatter",
        x="MetricX",
        y="MetricY",
        limit=3,
        title="Y vs X",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, ScatterEvidence)
    assert evidence.displayed_point_count == 5
    assert evidence.top_n is None
    assert evidence.pearson_correlation == pytest.approx(1.0)
    assert evidence.spearman_correlation == pytest.approx(1.0)
    assert evidence.r_squared == pytest.approx(1.0)
    assert "displayed observations" in insight.supporting_evidence.lower()
    assert "displayed categories" not in insight.supporting_evidence.lower()
    assert "top" not in insight.supporting_evidence.lower()
    assert "observed variation" in insight.supporting_evidence.lower()


def test_scatter_constant_axis_omits_correlation() -> None:
    dataframe = pd.DataFrame({
        "MetricX": [1, 1, 1, 1],
        "MetricY": [1, 2, 3, 4],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="scatter",
        x="MetricX",
        y="MetricY",
        title="Y vs X",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, ScatterEvidence)
    assert evidence.pearson_correlation is None
    assert evidence.spearman_correlation is None
    assert evidence.r_squared is None
    assert evidence.evidence_strength == "low"
    assert "correlation requires" in " ".join(evidence.warnings).lower()
    assert "pearson correlation is" not in insight.supporting_evidence.lower()


def test_scatter_detects_monotonic_non_linear_relationship() -> None:
    dataframe = pd.DataFrame({
        "MetricX": [1, 2, 3, 4, 5, 6, 7, 8],
        "MetricY": [1, 4, 9, 16, 25, 36, 49, 64],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="scatter",
        x="MetricX",
        y="MetricY",
        title="Curved",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, ScatterEvidence)
    assert evidence.spearman_correlation == pytest.approx(1.0)
    assert evidence.relationship_form in {"monotonic but non-linear", "approximately linear"}
    if evidence.relationship_form == "monotonic but non-linear":
        assert "rank" in insight.supporting_evidence.lower() or "rank" in insight.interpretation.lower()


def test_scatter_detects_bands_and_formula_dependency() -> None:
    rows = []
    for unit_price in (2.0, 3.0, 4.0):
        for units in range(1, 9):
            rows.append({
                "UnitsSold": units,
                "UnitPrice": unit_price,
                "TotalRevenue": units * unit_price,
            })
    result = ChartResult(
        spec=ChartSpec(
            chart_type="scatter",
            x="UnitsSold",
            y="TotalRevenue",
            title="Revenue vs Units",
        ),
        data=rows,
    )

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, ScatterEvidence)
    assert evidence.banding_detected is True
    assert evidence.band_count and evidence.band_count >= 3
    assert evidence.mathematical_dependency
    assert "structural" in insight.caution.lower()
    assert "band" in insight.key_finding.lower() or "band" in insight.supporting_evidence.lower()
    assert "unit price" in insight.interpretation.lower() or "unit price" in insight.recommended_next_step.lower()


def test_scatter_detects_outliers_heteroscedasticity_and_color_groups() -> None:
    dataframe = pd.DataFrame({
        "MetricX": list(range(1, 41)) + [45],
        "MetricY": [x + ((x % 5) - 2) * (x / 5) for x in range(1, 41)] + [140],
        "Segment": ["A"] * 20 + ["B"] * 21,
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="scatter",
        x="MetricX",
        y="MetricY",
        color="Segment",
        title="Spread",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, ScatterEvidence)
    assert evidence.color_group_summary
    assert evidence.outlier_count >= 1
    assert evidence.heteroscedasticity_detected is True
    assert "spread" in insight.key_finding.lower() or "spread" in insight.supporting_evidence.lower()
    assert "groups" in insight.supporting_evidence.lower() or "group" in insight.caution.lower()


def test_scatter_uses_dedicated_llm_prompt_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import llm_service

    dataframe = pd.DataFrame({
        "MetricX": [1, 2, 3, 4, 5],
        "MetricY": [5, 4, 3, 2, 1],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="scatter",
        x="MetricX",
        y="MetricY",
        title="Negative",
    ))
    evidence = extract_chart_evidence(result)
    fallback = generate_chart_insight(result)
    prompts = []

    class FakeResponse:
        content = "not valid json"

    class FakeModel:
        def invoke(self, prompt: str) -> FakeResponse:
            prompts.append(prompt)
            return FakeResponse()

    monkeypatch.setattr(llm_service, "_model", lambda settings, model_name: FakeModel())
    insight, _ = llm_service.explain_chart_insight_with_ollama(
        evidence,
        fallback,
        Settings(),
        "fake-model",
    )

    assert isinstance(evidence, ScatterEvidence)
    assert insight == fallback
    assert len(prompts) == 2
    assert "scatter plot" in prompts[0].lower()


def test_currency_chart_insight_uses_compact_units() -> None:
    dataframe = pd.DataFrame({
        "Region": ["West", "East"],
        "TotalRevenue": [356_724_250.12, 24_900_000.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="bar",
        x="Region",
        y="TotalRevenue",
        aggregation="sum",
        title="Revenue by Region",
    ))

    insight = generate_chart_insight(result)

    assert any("$356.7M" in item for item in insight.observations)
    assert any("$24.9M" in item for item in insight.observations)


def test_dual_axis_insight_identifies_high_volume_low_profit() -> None:
    dataframe = pd.DataFrame({
        "Country": ["A", "B", "C", "D"],
        "UnitsSold": [100, 80, 40, 20],
        "TotalProfit": [10.0, 50.0, 5.0, 30.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="dual_axis",
        x="Country",
        y="UnitsSold",
        secondary_y="TotalProfit",
        aggregation="sum",
        title="Units and Profit by Country",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, DualCombinationEvidence)
    assert evidence.chart_type == "dual_axis"
    assert evidence.bar_y_column == "UnitsSold"
    assert evidence.line_y_column == "TotalProfit"
    assert evidence.bar_aggregation == "sum"
    assert evidence.line_aggregation == "sum"
    assert evidence.unit_relationship == "different unit"
    assert evidence.bar_highest_x == "A"
    assert evidence.line_highest_x == "B"
    assert evidence.largest_normalized_divergence_x is not None
    assert "bars show" in insight.supporting_evidence.lower()
    assert "line shows" in insight.supporting_evidence.lower()
    assert "separate y-axis scales" in insight.caution.lower()
    assert "normalized" in insight.recommended_next_step.lower()


def test_dual_axis_datetime_insight_uses_ordered_relationship_and_independent_aggregations() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=6, freq="MS").repeat(2),
        "TotalRevenue": [100.0, 120.0, 130.0, 140.0, 125.0, 135.0, 160.0, 170.0, 155.0, 165.0, 190.0, 200.0],
        "ProfitMargin": [0.10, 0.14, 0.13, 0.15, 0.12, 0.13, 0.18, 0.17, 0.16, 0.15, 0.21, 0.22],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="dual_axis",
        x="Date",
        y="TotalRevenue",
        secondary_y="ProfitMargin",
        primary_aggregation="sum",
        secondary_aggregation="mean",
        title="Revenue and Margin",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, DualCombinationEvidence)
    assert evidence.x_axis_type == "datetime"
    assert evidence.bar_aggregation == "sum"
    assert evidence.line_aggregation == "mean"
    assert evidence.aggregation_relationship != "same aggregation"
    assert evidence.bar_endpoint_direction == "higher"
    assert evidence.line_endpoint_direction == "higher"
    assert evidence.comparable_transition_count == 5
    assert evidence.aligned_direction_count is not None
    assert evidence.peaks_aligned is True
    assert "shared direction" in insight.supporting_evidence.lower()
    assert "total cost" in insight.recommended_next_step.lower()
    combined_text = " ".join([
        insight.key_finding,
        insight.supporting_evidence,
        insight.interpretation,
    ]).lower()
    assert "chart mechanics" not in combined_text
    assert "00:00:00" not in combined_text


def test_dual_axis_uses_dedicated_llm_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import llm_service

    dataframe = pd.DataFrame({
        "Region": ["A", "B", "C"],
        "UnitsSold": [100, 80, 40],
        "UnitPrice": [10.0, 12.0, 15.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="dual_axis",
        x="Region",
        y="UnitsSold",
        secondary_y="UnitPrice",
        primary_aggregation="sum",
        secondary_aggregation="mean",
        title="Units and Price",
    ))
    evidence = extract_chart_evidence(result)
    fallback = generate_chart_insight(result)
    prompts = []

    class FakeResponse:
        content = "not valid json"

    class FakeModel:
        def invoke(self, prompt: str) -> FakeResponse:
            prompts.append(prompt)
            return FakeResponse()

    monkeypatch.setattr(llm_service, "_model", lambda settings, model_name: FakeModel())
    insight, _ = llm_service.explain_chart_insight_with_ollama(
        evidence,
        fallback,
        Settings(),
        "fake-model",
    )

    assert isinstance(evidence, DualCombinationEvidence)
    assert insight == fallback
    assert len(prompts) == 2
    assert "Dual Combination chart" in prompts[0]


def test_count_and_hierarchical_charts_receive_written_insights() -> None:
    dataframe = pd.DataFrame({
        "Continent": ["Asia", "Asia", "Europe"],
        "Country": ["Japan", "India", "France"],
        "Revenue": [100.0, 80.0, 90.0],
    })
    _, count_result = create_chart(dataframe, ChartSpec(
        chart_type="bar",
        x="Continent",
        aggregation="count",
        title="Records by Continent",
    ))
    _, treemap_result = create_chart(dataframe, ChartSpec(
        chart_type="treemap",
        x="Country",
        y="Revenue",
        color="Continent",
        aggregation="sum",
        title="Revenue Hierarchy",
    ))
    _, map_result = create_chart(dataframe, ChartSpec(
        chart_type="symbol_map",
        x="Country",
        y="Revenue",
        aggregation="sum",
        title="Revenue Map",
    ))

    assert "Asia" in generate_chart_insight(count_result).headline
    assert any(
        "represents" in item
        for item in generate_chart_insight(treemap_result).observations
    )
    map_text = " ".join(generate_chart_insight(map_result).observations).lower()
    assert "map" in map_text
    assert "country" in map_text or "countries" in map_text


def test_area_dual_line_and_circle_views_receive_written_insights() -> None:
    dataframe = pd.DataFrame({
        "Period": ["Q1", "Q2", "Q3"],
        "Sales": [10.0, 20.0, 30.0],
        "Profit": [2.0, 4.0, 12.0],
        "Orders": [5, 10, 20],
        "Segment": ["A", "A", "B"],
    })
    _, area_result = create_chart(dataframe, ChartSpec(
        chart_type="area",
        x="Period",
        y="Sales",
        aggregation="sum",
        title="Sales Area",
    ))
    _, dual_result = create_chart(dataframe, ChartSpec(
        chart_type="dual_line",
        x="Period",
        y="Sales",
        secondary_y="Profit",
        aggregation="sum",
        title="Sales and Profit",
    ))
    _, circle_result = create_chart(dataframe, ChartSpec(
        chart_type="circle_view",
        x="Sales",
        y="Profit",
        secondary_y="Orders",
        color="Segment",
        title="Performance Bubbles",
    ))

    circle_insight = generate_chart_insight(circle_result)

    assert "increased" in generate_chart_insight(area_result).headline
    assert generate_chart_insight(dual_result).evidence.chart_type == "dual_line"
    assert isinstance(circle_insight.evidence, CircleViewEvidence)
    assert "bubble" in circle_insight.supporting_evidence.lower()
    assert "bubble area" in circle_insight.caution.lower()


def test_circle_view_evidence_tracks_size_color_quadrants_and_observation_wording() -> None:
    dataframe = pd.DataFrame({
        "UnitsSold": [10, 20, 30, 40, 50, 60],
        "TotalRevenue": [100, 180, 320, 410, 530, 650],
        "TotalProfit": [8, 20, 45, 60, 140, 180],
        "ItemType": ["Office", "Office", "Tech", "Tech", "Household", "Household"],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="circle_view",
        x="UnitsSold",
        y="TotalRevenue",
        secondary_y="TotalProfit",
        color="ItemType",
        title="Revenue Profit Circles",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, CircleViewEvidence)
    assert evidence.x_column == "UnitsSold"
    assert evidence.y_column == "TotalRevenue"
    assert evidence.size_column == "TotalProfit"
    assert evidence.color_column == "ItemType"
    assert evidence.displayed_point_count == 6
    assert evidence.pearson_xy == pytest.approx(pd.Series(dataframe["UnitsSold"]).corr(pd.Series(dataframe["TotalRevenue"])))
    assert evidence.spearman_xy == pytest.approx(pd.Series(dataframe["UnitsSold"]).corr(pd.Series(dataframe["TotalRevenue"]), method="spearman"))
    assert evidence.pearson_size_x == pytest.approx(pd.Series(dataframe["UnitsSold"]).corr(pd.Series(dataframe["TotalProfit"])))
    assert evidence.pearson_size_y == pytest.approx(pd.Series(dataframe["TotalRevenue"]).corr(pd.Series(dataframe["TotalProfit"])))
    assert evidence.largest_bubble_size == 180
    assert evidence.largest_bubble_group == "Household"
    assert evidence.largest_bubble_quadrant == "upper-right"
    assert evidence.large_bubble_concentration == "upper-right"
    assert evidence.group_with_largest_total_size == "Household"
    assert evidence.group_with_largest_single_bubble == "Household"
    assert "displayed observations" in insight.supporting_evidence.lower()
    assert "displayed categories" not in insight.supporting_evidence.lower()
    assert "bubble area" in insight.supporting_evidence.lower()
    assert "color" not in insight.recommended_next_step.lower()
    assert "item type" in insight.recommended_next_step.lower()


def test_circle_view_detects_similar_positions_overlap_and_dependency() -> None:
    rows = []
    for unit_price in (2.0, 3.0, 4.0):
        for units in range(1, 9):
            revenue = units * unit_price
            rows.append({
                "UnitsSold": units,
                "UnitPrice": unit_price,
                "UnitCost": unit_price - 0.5,
                "TotalRevenue": revenue,
                "TotalCost": units * (unit_price - 0.5),
                "TotalProfit": units * 0.5,
                "ItemType": "A" if unit_price < 4 else "B",
            })
    rows.extend([
        {"UnitsSold": 4, "UnitPrice": 3.0, "UnitCost": 2.9, "TotalRevenue": 12.0, "TotalCost": 11.6, "TotalProfit": 1.0, "ItemType": "A"},
        {"UnitsSold": 4, "UnitPrice": 3.0, "UnitCost": 1.0, "TotalRevenue": 12.0, "TotalCost": 4.0, "TotalProfit": 8.0, "ItemType": "B"},
    ])
    result = ChartResult(
        spec=ChartSpec(
            chart_type="circle_view",
            x="UnitsSold",
            y="TotalRevenue",
            secondary_y="TotalProfit",
            color="ItemType",
            title="Formula Circles",
        ),
        data=rows,
    )

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, CircleViewEvidence)
    assert evidence.banding_detected is True
    assert evidence.similar_position_different_size_count >= 1
    assert evidence.mathematical_dependency
    assert "structural" in insight.caution.lower()
    assert "unit price" in insight.recommended_next_step.lower() or "revenue per unit" in insight.recommended_next_step.lower()


def test_circle_view_constant_size_low_evidence() -> None:
    dataframe = pd.DataFrame({
        "MetricX": [1, 2, 3, 4],
        "MetricY": [2, 4, 6, 8],
        "BubbleSize": [5, 5, 5, 5],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="circle_view",
        x="MetricX",
        y="MetricY",
        secondary_y="BubbleSize",
        title="Constant Size",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, CircleViewEvidence)
    assert evidence.evidence_strength == "low"
    assert evidence.pearson_size_x is None
    assert "bubble" in insight.supporting_evidence.lower()


def test_circle_view_uses_dedicated_llm_prompt_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import llm_service

    dataframe = pd.DataFrame({
        "MetricX": [1, 2, 3, 4],
        "MetricY": [2, 3, 5, 8],
        "BubbleSize": [1, 3, 9, 20],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="circle_view",
        x="MetricX",
        y="MetricY",
        secondary_y="BubbleSize",
        title="Circle LLM",
    ))
    evidence = extract_chart_evidence(result)
    fallback = generate_chart_insight(result)
    prompts = []

    class FakeResponse:
        content = "not valid json"

    class FakeModel:
        def invoke(self, prompt: str) -> FakeResponse:
            prompts.append(prompt)
            return FakeResponse()

    monkeypatch.setattr(llm_service, "_model", lambda settings, model_name: FakeModel())
    insight, _ = llm_service.explain_chart_insight_with_ollama(
        evidence,
        fallback,
        Settings(),
        "fake-model",
    )

    assert isinstance(evidence, CircleViewEvidence)
    assert insight == fallback
    assert len(prompts) == 2
    assert "circle view chart" in prompts[0].lower()


def test_dual_line_categorical_evidence_uses_rankings_not_temporal_change() -> None:
    dataframe = pd.DataFrame({
        "Region": ["West", "West", "East", "East", "North", "North"],
        "TotalRevenue": [100.0, 200.0, 50.0, 80.0, 20.0, 30.0],
        "UnitsSold": [10.0, 20.0, 6.0, 9.0, 2.0, 3.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="dual_line",
        x="Region",
        y="TotalRevenue",
        secondary_y="UnitsSold",
        primary_aggregation="sum",
        secondary_aggregation="sum",
        title="Revenue and Units",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, DualLineEvidence)
    assert evidence.x_axis_type == "unordered categorical"
    assert evidence.primary_highest_x == "West"
    assert evidence.secondary_highest_x == "West"
    assert evidence.unit_relationship == "different unit"
    assert evidence.pearson_correlation == pytest.approx(0.9959556576)
    assert "changed from" not in insight.supporting_evidence.lower()
    assert "temporal trend" in insight.caution.lower()
    assert "revenue per unit" in insight.recommended_next_step.lower()


def test_dual_line_same_unit_and_different_aggregation_cautions() -> None:
    dataframe = pd.DataFrame({
        "Region": ["West", "West", "East", "East", "North", "North"],
        "TotalRevenue": [100.0, 200.0, 50.0, 80.0, 20.0, 30.0],
        "TotalCost": [70.0, 140.0, 40.0, 60.0, 10.0, 15.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="dual_line",
        x="Region",
        y="TotalRevenue",
        secondary_y="TotalCost",
        primary_aggregation="sum",
        secondary_aggregation="mean",
        title="Revenue and Cost",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, DualLineEvidence)
    assert evidence.unit_relationship == "same unit"
    assert evidence.derived_metric_available == "profit"
    assert evidence.aggregation_warning
    assert "shared y-axis" in insight.caution.lower()
    assert "total profit" in insight.recommended_next_step.lower()


def test_dual_line_datetime_evidence_uses_temporal_movement() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=4, freq="MS"),
        "Revenue": [100.0, 120.0, 115.0, 150.0],
        "SatisfactionScore": [80.0, 82.0, 81.0, 85.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="dual_line",
        x="Date",
        y="Revenue",
        secondary_y="SatisfactionScore",
        primary_aggregation="sum",
        secondary_aggregation="mean",
        title="Revenue and Satisfaction",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, DualLineEvidence)
    assert evidence.x_axis_type == "datetime"
    assert evidence.primary_change == pytest.approx(50.0)
    assert evidence.secondary_change == pytest.approx(5.0)
    assert "was" in insight.supporting_evidence.lower()
    assert "changed across the ordered range" not in insight.key_finding.lower()
    assert "move" not in insight.key_finding.lower()
    assert "move" not in insight.supporting_evidence.lower()
    assert "different y-axis scales" in insight.caution.lower()


def test_dual_line_datetime_evidence_adds_endpoint_alignment_volatility_and_divergence() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=6, freq="MS"),
        "TotalRevenue": [100.0, 130.0, 120.0, 160.0, 150.0, 190.0],
        "TotalProfit": [20.0, 30.0, 25.0, 45.0, 35.0, 50.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="dual_line",
        x="Date",
        y="TotalRevenue",
        secondary_y="TotalProfit",
        primary_aggregation="sum",
        secondary_aggregation="sum",
        title="Revenue and Profit",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, DualLineEvidence)
    assert evidence.x_axis_type == "datetime"
    assert evidence.primary_change == pytest.approx(90.0)
    assert evidence.primary_change_percent == pytest.approx(90.0)
    assert evidence.secondary_change == pytest.approx(30.0)
    assert evidence.secondary_change_percent == pytest.approx(150.0)
    assert evidence.primary_endpoint_direction == "higher"
    assert evidence.secondary_endpoint_direction == "higher"
    assert evidence.peaks_aligned is True
    assert evidence.troughs_aligned is True
    assert evidence.comparable_transition_count == 5
    assert evidence.aligned_direction_count == 5
    assert evidence.aligned_direction_percent == pytest.approx(100.0)
    assert evidence.opposite_direction_count == 0
    assert evidence.pearson_correlation is not None
    assert evidence.primary_volatility_level is not None
    assert evidence.secondary_volatility_level is not None
    assert "same period-to-period direction in 5 of 5" in insight.supporting_evidence
    assert "Profit reflects the portion of revenue remaining after costs." in insight.supporting_evidence
    assert "profit margin" in insight.recommended_next_step.lower()
    assert "total cost" in insight.recommended_next_step.lower()
    combined_text = " ".join([
        insight.key_finding,
        insight.supporting_evidence,
        insight.interpretation,
        insight.recommended_next_step,
    ]).lower()
    assert "00:00:00" not in combined_text
    assert "changed across the ordered range" not in combined_text
    assert "move" not in combined_text


def test_dual_line_datetime_divergence_recommends_revenue_per_unit() -> None:
    dataframe = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=6, freq="MS"),
        "TotalRevenue": [100.0, 150.0, 110.0, 160.0, 140.0, 180.0],
        "UnitsSold": [10.0, 8.0, 12.0, 9.0, 15.0, 11.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="dual_line",
        x="Date",
        y="TotalRevenue",
        secondary_y="UnitsSold",
        primary_aggregation="sum",
        secondary_aggregation="sum",
        title="Revenue and Units",
    ))

    evidence = extract_chart_evidence(result)
    insight = generate_chart_insight(result)

    assert isinstance(evidence, DualLineEvidence)
    assert evidence.unit_relationship == "different unit"
    assert evidence.opposite_direction_count == 5
    assert evidence.opposite_direction_percent == pytest.approx(100.0)
    assert evidence.divergence_periods
    assert evidence.largest_normalized_divergence_period is not None
    assert evidence.largest_normalized_divergence_value is not None
    assert "opposite directions in 5" in insight.supporting_evidence
    assert "different y-axis scales" in insight.caution.lower()
    assert "revenue per unit" in insight.recommended_next_step.lower()


def test_dual_line_constant_series_omits_correlation_and_llm_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import llm_service

    dataframe = pd.DataFrame({
        "Region": ["A", "B", "C"],
        "MetricA": [1.0, 1.0, 1.0],
        "MetricB": [2.0, 3.0, 4.0],
    })
    _, result = create_chart(dataframe, ChartSpec(
        chart_type="dual_line",
        x="Region",
        y="MetricA",
        secondary_y="MetricB",
        primary_aggregation="sum",
        secondary_aggregation="sum",
        title="Metrics",
    ))
    evidence = extract_chart_evidence(result)
    fallback = generate_chart_insight(result)
    prompts = []

    class FakeResponse:
        content = "not valid json"

    class FakeModel:
        def invoke(self, prompt: str) -> FakeResponse:
            prompts.append(prompt)
            return FakeResponse()

    monkeypatch.setattr(llm_service, "_model", lambda settings, model_name: FakeModel())
    insight, _ = llm_service.explain_chart_insight_with_ollama(
        evidence,
        fallback,
        Settings(),
        "fake-model",
    )

    assert isinstance(evidence, DualLineEvidence)
    assert evidence.pearson_correlation is None
    assert insight == fallback
    assert len(prompts) == 2
    assert "dual-line chart" in prompts[0]


def test_gantt_and_bullet_views_receive_written_insights() -> None:
    dataframe = pd.DataFrame({
        "Task": ["Design", "Build"],
        "StartDate": ["2026-01-01", "2026-01-05"],
        "EndDate": ["2026-01-04", "2026-01-12"],
        "Actual": [110.0, 60.0],
        "Target": [100.0, 75.0],
    })
    _, gantt_result = create_chart(dataframe, ChartSpec(
        chart_type="gantt",
        x="StartDate",
        y="Task",
        secondary_y="EndDate",
        title="Project Plan",
    ))
    _, bullet_result = create_chart(dataframe, ChartSpec(
        chart_type="bullet",
        x="Task",
        y="Actual",
        secondary_y="Target",
        aggregation="sum",
        title="Actual vs Target",
    ))

    gantt_insight = generate_chart_insight(gantt_result)
    bullet_insight = generate_chart_insight(bullet_result)

    assert "2 task(s)" in gantt_insight.headline
    assert any("longest task" in item for item in gantt_insight.observations)
    assert "1 of 2" in bullet_insight.headline
    assert any("below target" in item for item in bullet_insight.observations)
