"""Evidence-based written insights derived from verified chart data."""

from __future__ import annotations

from math import isfinite
import re
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from services.chart_service import (
    SYMBOL_MAP_COLOR_LOCATION,
    ChartResult,
    REGION_CENTROIDS,
    effective_symbol_map_color,
)
from utils.formatting import format_metric, format_period


EvidenceStrength = Literal["high", "medium", "low"]


ABBREVIATIONS = {"ID", "URL", "GDP", "KPI", "ROI", "USD", "EUR"}
VALUE_LABELS = {
    "orderpriority": {
        "C": "Critical",
        "H": "High",
        "L": "Low",
        "M": "Medium",
    },
    "orderprioritycode": {
        "C": "Critical",
        "H": "High",
        "L": "Low",
        "M": "Medium",
    },
}

CORRELATION_STRENGTH_THRESHOLDS = (
    (0.10, "negligible"),
    (0.30, "weak"),
    (0.50, "moderate"),
    (0.70, "strong"),
    (0.90, "very strong"),
)
HIGH_MULTICOLLINEARITY_THRESHOLD = 0.80
CORRELATION_CLUSTER_THRESHOLD = 0.70
ID_NAME_PATTERN = re.compile(
    r"(^|[\s_\-])(id|key|code|index|identifier|number|no)([\s_\-]|$)",
    re.IGNORECASE,
)


class ChartEvidence(BaseModel):
    """Calculated chart evidence that may be safely explained by rules or an LLM."""

    chart_type: str
    chart_title: str
    x_column: str | None = None
    y_columns: list[str] = Field(default_factory=list)
    category_column: str | None = None
    group_column: str | None = None
    color_column: str | None = None
    size_column: str | None = None
    aggregation: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    sorting: str | None = None
    top_n: int | None = None
    total_rows: int
    valid_rows: int
    excluded_rows: int = 0
    calculated_metrics: dict[str, Any] = Field(default_factory=dict)
    detected_patterns: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    analysis_context: dict[str, Any] = Field(default_factory=dict)
    evidence_strength: EvidenceStrength = "medium"


class CorrelationPairEvidence(BaseModel):
    """Evidence for one unique non-diagonal correlation pair."""

    variable_x: str
    variable_y: str
    correlation: float
    absolute_correlation: float
    direction: str
    strength: str
    paired_observation_count: int | None = None
    missing_pair_count: int | None = None
    known_relationship: str | None = None
    formula_derived: bool = False


class CorrelationHeatmapEvidence(ChartEvidence):
    """Evidence for numeric Pearson correlation heatmaps."""

    chart_type: str = "correlation_heatmap"
    heatmap_type: str = "correlation"
    correlation_method: str = "pearson"
    selected_columns: list[str] = Field(default_factory=list)
    displayed_variable_count: int = 0
    unique_pair_count: int = 0
    raw_row_count: int = 0
    filtered_row_count: int = 0
    sampled: bool = False
    sampling_method: str | None = None
    pairs: list[CorrelationPairEvidence] = Field(default_factory=list)
    strongest_positive_pair: CorrelationPairEvidence | None = None
    strongest_negative_pair: CorrelationPairEvidence | None = None
    strongest_pairs: list[CorrelationPairEvidence] = Field(default_factory=list)
    strong_positive_pairs: list[CorrelationPairEvidence] = Field(default_factory=list)
    strong_negative_pairs: list[CorrelationPairEvidence] = Field(default_factory=list)
    moderate_pairs: list[CorrelationPairEvidence] = Field(default_factory=list)
    weak_pairs: list[CorrelationPairEvidence] = Field(default_factory=list)
    near_zero_pairs: list[CorrelationPairEvidence] = Field(default_factory=list)
    identifier_like_columns: list[str] = Field(default_factory=list)
    constant_columns: list[str] = Field(default_factory=list)
    near_constant_columns: list[str] = Field(default_factory=list)
    formula_relationships: list[dict[str, Any]] = Field(default_factory=list)
    correlation_clusters: list[dict[str, Any]] = Field(default_factory=list)
    high_multicollinearity_pairs: list[CorrelationPairEvidence] = Field(default_factory=list)
    minimum_pairwise_count: int | None = None
    maximum_pairwise_count: int | None = None
    unequal_pairwise_counts: bool = False
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class SingleBarEvidence(ChartEvidence):
    """Evidence for one-series bar charts with exactly one value per category."""

    value_column: str | None = None
    category_count: int = 0
    original_category_count: int = 0
    category_values: dict[str, float] = Field(default_factory=dict)
    highest_category: str | None = None
    highest_value: float | None = None
    second_highest_category: str | None = None
    second_highest_value: float | None = None
    lowest_category: str | None = None
    lowest_value: float | None = None
    leader_to_second_gap: float | None = None
    leader_to_second_gap_percent: float | None = None
    leader_to_second_gap_basis: str | None = None
    highest_to_lowest_gap: float | None = None
    highest_to_lowest_gap_percent: float | None = None
    highest_to_lowest_gap_basis: str | None = None
    highest_share_percent: float | None = None
    top_two_share_percent: float | None = None
    top_three_share_percent: float | None = None
    displayed_total: float | None = None
    displayed_mean: float | None = None
    displayed_median: float | None = None
    concentration_level: str | None = None
    lead_strength: str | None = None
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    top_n_applied: int | None = None


class PieChartEvidence(ChartEvidence):
    """Evidence for Pie/Donut part-to-whole charts."""

    chart_type: str = "pie"
    value_column: str | None = None
    category_column: str | None = None
    aggregation: str | None = None
    original_category_count: int = 0
    displayed_category_count: int = 0
    values_by_category: dict[str, float] = Field(default_factory=dict)
    shares_by_category: dict[str, float] = Field(default_factory=dict)
    displayed_total: float | None = None
    largest_category: str | None = None
    largest_value: float | None = None
    largest_share: float | None = None
    second_category: str | None = None
    second_value: float | None = None
    second_share: float | None = None
    smallest_category: str | None = None
    smallest_value: float | None = None
    smallest_share: float | None = None
    leader_to_second_gap: float | None = None
    leader_to_second_gap_percent: float | None = None
    leader_to_second_gap_basis: str | None = None
    lead_strength: str | None = None
    top_two_share: float | None = None
    top_three_share: float | None = None
    remaining_share: float | None = None
    concentration_level: str | None = None
    effective_category_count: float | None = None
    herfindahl_index: float | None = None
    small_slice_categories: list[str] = Field(default_factory=list)
    small_slice_threshold_percent: float = 3.0
    other_category_present: bool = False
    other_category_label: str | None = None
    other_category_share: float | None = None
    top_n_applied: int | None = None
    aggregation_additive: bool = True
    part_to_whole_valid: bool = True
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class TreemapEvidence(ChartEvidence):
    """Evidence for Treemap area-share charts."""

    chart_type: str = "treemap"
    value_column: str | None = None
    category_column: str | None = None
    group_column: str | None = None
    aggregation: str | None = None
    original_category_count: int = 0
    displayed_category_count: int = 0
    displayed_total: float | None = None
    values_by_category: dict[str, float] = Field(default_factory=dict)
    shares_by_category: dict[str, float] = Field(default_factory=dict)
    group_totals: dict[str, float] = Field(default_factory=dict)
    group_shares: dict[str, float] = Field(default_factory=dict)
    largest_category: str | None = None
    largest_value: float | None = None
    largest_share: float | None = None
    second_category: str | None = None
    second_value: float | None = None
    second_share: float | None = None
    smallest_category: str | None = None
    smallest_value: float | None = None
    smallest_share: float | None = None
    largest_group: str | None = None
    largest_group_value: float | None = None
    largest_group_share: float | None = None
    leader_to_second_gap: float | None = None
    leader_to_second_gap_percent: float | None = None
    lead_strength: str | None = None
    top_two_share: float | None = None
    top_three_share: float | None = None
    remaining_share: float | None = None
    concentration_level: str | None = None
    small_rectangle_categories: list[str] = Field(default_factory=list)
    small_rectangle_threshold_percent: float = 3.0
    top_n_applied: int | None = None
    part_to_whole_valid: bool = True
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class SymbolMapLocationEvidence(BaseModel):
    """Evidence for one displayed Symbol Map location."""

    location: str
    latitude: float | None = None
    longitude: float | None = None
    aggregated_value: float
    share_of_displayed_total: float | None = None
    rank: int | None = None
    color_group: str | None = None
    valid_coordinates: bool = True
    coordinate_source: str | None = None


class SymbolMapGroupEvidence(BaseModel):
    """Evidence for one color group on a Symbol Map."""

    group_name: str
    location_count: int
    total_value: float
    share_of_displayed_total: float | None = None
    mean_location_value: float | None = None
    median_location_value: float | None = None
    largest_location: str | None = None
    largest_location_value: float | None = None
    top_location_count: int = 0


class SymbolMapEvidence(ChartEvidence):
    """Evidence for Symbol Maps with geographic location and bubble-size values."""

    chart_type: str = "symbol_map"
    location_column: str | None = None
    location_type: str | None = None
    value_column: str | None = None
    aggregation: str | None = None
    color_column: str | None = None
    raw_row_count: int = 0
    aggregated_location_count: int = 0
    displayed_location_count: int = 0
    unresolved_location_count: int = 0
    unresolved_locations: list[str] = Field(default_factory=list)
    displayed_total: float | None = None
    locations: list[SymbolMapLocationEvidence] = Field(default_factory=list)
    largest_location: str | None = None
    largest_value: float | None = None
    largest_share: float | None = None
    largest_group: str | None = None
    second_location: str | None = None
    second_value: float | None = None
    second_share: float | None = None
    third_location: str | None = None
    third_value: float | None = None
    third_share: float | None = None
    smallest_location: str | None = None
    smallest_value: float | None = None
    smallest_share: float | None = None
    top_two_share: float | None = None
    top_three_share: float | None = None
    top_five_share: float | None = None
    leader_to_second_gap: float | None = None
    leader_to_second_gap_percentage_points: float | None = None
    lead_strength: str | None = None
    concentration_level: str | None = None
    herfindahl_index: float | None = None
    effective_location_count: float | None = None
    color_groups: list[SymbolMapGroupEvidence] = Field(default_factory=list)
    highest_total_group: str | None = None
    highest_total_group_value: float | None = None
    highest_total_group_share: float | None = None
    highest_median_group: str | None = None
    highest_median_group_value: float | None = None
    group_with_most_top_locations: str | None = None
    group_with_most_top_locations_count: int | None = None
    geographic_distribution: str | None = None
    spatial_concentration_level: str | None = None
    marker_overlap_level: str | None = None
    dense_areas: list[str] = Field(default_factory=list)
    top_n_applied: int | None = None
    original_location_count: int | None = None
    date_column: str | None = None
    start_date: Any | None = None
    end_date: Any | None = None
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class SortedPercentageBarEvidence(ChartEvidence):
    """Evidence for sorted percentage contribution bars."""

    chart_type: str = "sorted_percentage_bar"
    category_column: str | None = None
    value_column: str | None = None
    aggregation: str | None = None
    original_category_count: int = 0
    displayed_category_count: int = 0
    percentage_denominator_mode: str = "full_filtered"
    aggregated_values: dict[str, float] = Field(default_factory=dict)
    percentage_shares: dict[str, float] = Field(default_factory=dict)
    category_ranking: list[str] = Field(default_factory=list)
    displayed_total: float | None = None
    full_filtered_total: float | None = None
    largest_category: str | None = None
    largest_value: float | None = None
    largest_share: float | None = None
    second_category: str | None = None
    second_value: float | None = None
    second_share: float | None = None
    third_category: str | None = None
    third_value: float | None = None
    third_share: float | None = None
    smallest_category: str | None = None
    smallest_value: float | None = None
    smallest_share: float | None = None
    leader_to_second_gap_percentage_points: float | None = None
    leader_to_second_relative_gap_percent: float | None = None
    lead_strength: str | None = None
    top_two_share: float | None = None
    top_three_share: float | None = None
    remaining_share: float | None = None
    concentration_level: str | None = None
    herfindahl_index: float | None = None
    effective_category_count: float | None = None
    small_share_categories: list[str] = Field(default_factory=list)
    top_n_applied: int | None = None
    other_category_present: bool = False
    other_share: float | None = None
    additive_aggregation: bool = True
    percentage_valid: bool = True
    negative_value_count: int = 0
    date_column: str | None = None
    start_date: Any | None = None
    end_date: Any | None = None
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PeriodOverPeriodChangeEvidence(ChartEvidence):
    """Evidence for period-over-period percentage change charts."""

    chart_type: str = "period_over_period_change"
    date_column: str | None = None
    value_column: str | None = None
    aggregation: str | None = None
    granularity: str | None = None
    comparison_basis: str = "previous_period"
    period_count: int = 0
    comparable_period_count: int = 0
    unavailable_period_count: int = 0
    periods: list[str] = Field(default_factory=list)
    current_values: dict[str, float] = Field(default_factory=dict)
    comparison_values: dict[str, float] = Field(default_factory=dict)
    percentage_changes: dict[str, float] = Field(default_factory=dict)
    absolute_changes: dict[str, float] = Field(default_factory=dict)
    largest_increase_period: str | None = None
    largest_increase_percent: float | None = None
    largest_decline_period: str | None = None
    largest_decline_percent: float | None = None
    latest_period: str | None = None
    latest_value: float | None = None
    latest_comparison_value: float | None = None
    latest_percent_change: float | None = None
    latest_absolute_change: float | None = None
    increase_count: int = 0
    decline_count: int = 0
    no_change_count: int = 0
    average_percent_change: float | None = None
    median_percent_change: float | None = None
    volatility_level: str | None = None
    missing_period_count: int = 0
    zero_baseline_count: int = 0
    start_date: Any | None = None
    end_date: Any | None = None
    calculation_start: Any | None = None
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class StackedBarEvidence(ChartEvidence):
    """Evidence for stacked bars with combined category and segment composition facts."""

    value_column: str | None = None
    category_count: int = 0
    stack_count: int = 0
    category_totals: dict[str, float] = Field(default_factory=dict)
    stack_totals: dict[str, float] = Field(default_factory=dict)
    values_by_category_and_stack: dict[str, dict[str, float]] = Field(default_factory=dict)
    highest_combined_category: str | None = None
    highest_combined_value: float | None = None
    lowest_combined_category: str | None = None
    lowest_combined_value: float | None = None
    dominant_stack_by_category: dict[str, str] = Field(default_factory=dict)
    dominant_stack_value_by_category: dict[str, float] = Field(default_factory=dict)
    dominant_stack_share_by_category: dict[str, float | None] = Field(default_factory=dict)
    strongest_stack: str | None = None
    strongest_stack_value: float | None = None
    highest_category_dominant_stack: str | None = None
    highest_category_dominant_value: float | None = None
    highest_category_dominant_share: float | None = None
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    top_n_applied: int | None = None


class GroupedBarEvidence(ChartEvidence):
    """Evidence for side-by-side/grouped bars with separate total and segment facts."""

    category_count: int = 0
    group_count: int = 0
    value_column: str | None = None
    category_totals: dict[str, float] = Field(default_factory=dict)
    group_totals: dict[str, float] = Field(default_factory=dict)
    values_by_category_and_group: dict[str, dict[str, float]] = Field(default_factory=dict)
    missing_group_combinations: list[dict[str, str]] = Field(default_factory=list)
    highest_combined_category: str | None = None
    highest_combined_value: float | None = None
    lowest_combined_category: str | None = None
    lowest_combined_value: float | None = None
    highest_individual_category: str | None = None
    highest_individual_group: str | None = None
    highest_individual_value: float | None = None
    lowest_individual_category: str | None = None
    lowest_individual_group: str | None = None
    lowest_individual_value: float | None = None
    winner_by_category: dict[str, str] = Field(default_factory=dict)
    winner_value_by_category: dict[str, float] = Field(default_factory=dict)
    winner_gap_by_category: dict[str, float] = Field(default_factory=dict)
    winner_gap_percent_by_category: dict[str, float | None] = Field(default_factory=dict)
    group_win_counts: dict[str, int] = Field(default_factory=dict)
    largest_gap_category: str | None = None
    largest_gap_groups: list[str] = Field(default_factory=list)
    largest_gap_value: float | None = None
    largest_gap_percent: float | None = None
    largest_gap_percent_basis: str | None = None
    most_balanced_category: str | None = None
    smallest_gap_value: float | None = None
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    top_n_applied: int | None = None


class SingleLineEvidence(ChartEvidence):
    """Evidence for a standard single-series line chart."""

    chart_type: str = "line"
    x_column: str
    y_column: str
    aggregation: str | None = None
    time_granularity: str | None = None
    point_count: int = 0
    start_period: Any | None = None
    start_period_label: str | None = None
    start_value: float | None = None
    end_period: Any | None = None
    end_period_label: str | None = None
    end_value: float | None = None
    endpoint_change: float | None = None
    endpoint_change_percent: float | None = None
    endpoint_change_basis: str | None = None
    peak_period: Any | None = None
    peak_period_label: str | None = None
    peak_value: float | None = None
    trough_period: Any | None = None
    trough_period_label: str | None = None
    trough_value: float | None = None
    value_range: float | None = None
    strongest_increase_start: Any | None = None
    strongest_increase_start_label: str | None = None
    strongest_increase_end: Any | None = None
    strongest_increase_end_label: str | None = None
    strongest_increase_value: float | None = None
    strongest_increase_percent: float | None = None
    strongest_decline_start: Any | None = None
    strongest_decline_start_label: str | None = None
    strongest_decline_end: Any | None = None
    strongest_decline_end_label: str | None = None
    strongest_decline_value: float | None = None
    strongest_decline_percent: float | None = None
    mean_period_change: float | None = None
    median_period_change: float | None = None
    period_change_std: float | None = None
    mean_absolute_period_change: float | None = None
    median_absolute_period_change: float | None = None
    mean_absolute_percent_change: float | None = None
    coefficient_of_variation: float | None = None
    volatility_level: str | None = None
    linear_trend_slope: float | None = None
    linear_trend_r_squared: float | None = None
    trend_direction: str | None = None
    trend_strength: str | None = None
    pattern_classification: str | None = None
    positive_change_count: int = 0
    negative_change_count: int = 0
    unchanged_count: int = 0
    direction_reversal_count: int = 0
    missing_periods: list[Any] = Field(default_factory=list)
    missing_period_labels: list[str] = Field(default_factory=list)
    irregular_intervals: bool = False
    seasonality_evidence: str | None = None
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class SingleAreaEvidence(ChartEvidence):
    """Evidence for a single-series area chart."""

    chart_type: str = "area"
    x_column: str
    y_column: str
    aggregation: str | None = None
    x_axis_type: str = "unknown"
    time_granularity: str | None = None
    point_count: int = 0
    start_period: Any | None = None
    start_period_label: str | None = None
    start_value: float | None = None
    end_period: Any | None = None
    end_period_label: str | None = None
    end_value: float | None = None
    endpoint_change: float | None = None
    endpoint_change_percent: float | None = None
    endpoint_change_basis: str | None = None
    peak_period: Any | None = None
    peak_period_label: str | None = None
    peak_value: float | None = None
    trough_period: Any | None = None
    trough_period_label: str | None = None
    trough_value: float | None = None
    value_range: float | None = None
    mean_value: float | None = None
    median_value: float | None = None
    strongest_increase_start: Any | None = None
    strongest_increase_start_label: str | None = None
    strongest_increase_end: Any | None = None
    strongest_increase_end_label: str | None = None
    strongest_increase_value: float | None = None
    strongest_increase_percent: float | None = None
    strongest_decline_start: Any | None = None
    strongest_decline_start_label: str | None = None
    strongest_decline_end: Any | None = None
    strongest_decline_end_label: str | None = None
    strongest_decline_value: float | None = None
    strongest_decline_percent: float | None = None
    linear_trend_slope: float | None = None
    linear_trend_r_squared: float | None = None
    trend_direction: str | None = None
    trend_strength: str | None = None
    volatility_level: str | None = None
    coefficient_of_variation: float | None = None
    direction_reversal_count: int | None = None
    high_periods: list[Any] = Field(default_factory=list)
    high_period_labels: list[str] = Field(default_factory=list)
    low_periods: list[Any] = Field(default_factory=list)
    low_period_labels: list[str] = Field(default_factory=list)
    longest_above_average_run: int | None = None
    longest_below_average_run: int | None = None
    approximate_area_under_curve: float | None = None
    area_interpretation_valid: bool = False
    baseline_value: float | None = 0.0
    baseline_is_zero: bool = True
    missing_periods: list[Any] = Field(default_factory=list)
    missing_period_labels: list[str] = Field(default_factory=list)
    irregular_intervals: bool = False
    negative_value_count: int = 0
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    pattern_classification: str | None = None


class StackedAreaEvidence(ChartEvidence):
    """Evidence for stacked area charts represented by area charts with color."""

    chart_type: str = "stacked_area"
    x_column: str
    stack_column: str
    y_column: str
    aggregation: str | None = None
    time_granularity: str | None = None
    point_count: int = 0
    stack_count: int = 0
    total_by_period: dict[str, float] = Field(default_factory=dict)
    values_by_period_and_stack: dict[str, dict[str, float]] = Field(default_factory=dict)
    start_total: float | None = None
    end_total: float | None = None
    total_change: float | None = None
    total_change_percent: float | None = None
    peak_period: Any | None = None
    peak_period_label: str | None = None
    peak_total: float | None = None
    trough_period: Any | None = None
    trough_period_label: str | None = None
    trough_total: float | None = None
    overall_stack_totals: dict[str, float] = Field(default_factory=dict)
    overall_stack_shares: dict[str, float] = Field(default_factory=dict)
    dominant_stack_overall: str | None = None
    dominant_stack_share: float | None = None
    dominant_stack_by_period: dict[str, str] = Field(default_factory=dict)
    stack_share_by_period: dict[str, dict[str, float]] = Field(default_factory=dict)
    stack_with_largest_growth: str | None = None
    stack_with_largest_decline: str | None = None
    composition_shift_periods: list[str] = Field(default_factory=list)
    missing_combinations: list[dict[str, Any]] = Field(default_factory=list)
    baseline_is_zero: bool = True
    negative_value_count: int = 0
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class DualLineEvidence(ChartEvidence):
    """Evidence for dual-line charts with one primary and one secondary metric."""

    chart_type: str = "dual_line"
    x_column: str
    x_axis_type: str = "unknown"
    primary_y_column: str
    primary_aggregation: str
    primary_unit: str | None = None
    secondary_y_column: str
    secondary_aggregation: str
    secondary_unit: str | None = None
    unit_relationship: str = "unknown unit"
    category_count: int = 0
    valid_primary_count: int = 0
    valid_secondary_count: int = 0
    primary_values: dict[str, float] = Field(default_factory=dict)
    secondary_values: dict[str, float] = Field(default_factory=dict)
    primary_highest_x: Any | None = None
    primary_highest_value: float | None = None
    primary_lowest_x: Any | None = None
    primary_lowest_value: float | None = None
    primary_range: float | None = None
    secondary_highest_x: Any | None = None
    secondary_highest_value: float | None = None
    secondary_lowest_x: Any | None = None
    secondary_lowest_value: float | None = None
    secondary_range: float | None = None
    same_highest_category: bool | None = None
    same_lowest_category: bool | None = None
    primary_ranking: list[str] = Field(default_factory=list)
    secondary_ranking: list[str] = Field(default_factory=list)
    pearson_correlation: float | None = None
    spearman_correlation: float | None = None
    relationship_strength: str | None = None
    relationship_direction: str | None = None
    rank_agreement_count: int | None = None
    rank_disagreement_categories: list[str] = Field(default_factory=list)
    normalized_primary_values: dict[str, float] = Field(default_factory=dict)
    normalized_secondary_values: dict[str, float] = Field(default_factory=dict)
    primary_start_x: Any | None = None
    primary_start_value: float | None = None
    primary_end_x: Any | None = None
    primary_end_value: float | None = None
    primary_change: float | None = None
    primary_change_percent: float | None = None
    primary_endpoint_direction: str | None = None
    secondary_start_x: Any | None = None
    secondary_start_value: float | None = None
    secondary_end_x: Any | None = None
    secondary_end_value: float | None = None
    secondary_change: float | None = None
    secondary_change_percent: float | None = None
    secondary_endpoint_direction: str | None = None
    primary_peak_period: Any | None = None
    primary_peak_value: float | None = None
    primary_trough_period: Any | None = None
    primary_trough_value: float | None = None
    secondary_peak_period: Any | None = None
    secondary_peak_value: float | None = None
    secondary_trough_period: Any | None = None
    secondary_trough_value: float | None = None
    peaks_aligned: bool | None = None
    troughs_aligned: bool | None = None
    primary_strongest_increase_start: Any | None = None
    primary_strongest_increase_end: Any | None = None
    primary_strongest_increase_value: float | None = None
    primary_strongest_decline_start: Any | None = None
    primary_strongest_decline_end: Any | None = None
    primary_strongest_decline_value: float | None = None
    secondary_strongest_increase_start: Any | None = None
    secondary_strongest_increase_end: Any | None = None
    secondary_strongest_increase_value: float | None = None
    secondary_strongest_decline_start: Any | None = None
    secondary_strongest_decline_end: Any | None = None
    secondary_strongest_decline_value: float | None = None
    paired_point_count: int = 0
    comparable_transition_count: int = 0
    aligned_direction_count: int | None = None
    aligned_direction_percent: float | None = None
    opposite_direction_count: int | None = None
    opposite_direction_percent: float | None = None
    unchanged_transition_count: int = 0
    primary_volatility_level: str | None = None
    secondary_volatility_level: str | None = None
    primary_coefficient_of_variation: float | None = None
    secondary_coefficient_of_variation: float | None = None
    more_volatile_metric: str | None = None
    divergence_periods: list[Any] = Field(default_factory=list)
    largest_normalized_divergence_period: Any | None = None
    largest_normalized_divergence_value: float | None = None
    missing_periods: list[Any] = Field(default_factory=list)
    irregular_intervals: bool = False
    known_metric_relationship: str | None = None
    derived_metric_available: str | None = None
    aggregation_warning: str | None = None
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class DualCombinationEvidence(ChartEvidence):
    """Evidence for dual-combination charts with bars on the left axis and a line on the right axis."""

    chart_type: str = "dual_axis"
    x_column: str
    x_axis_type: str = "unknown"
    time_granularity: str | None = None
    bar_y_column: str
    bar_aggregation: str
    bar_unit: str | None = None
    line_y_column: str
    line_aggregation: str
    line_unit: str | None = None
    unit_relationship: str = "unknown unit"
    aggregation_relationship: str = "unknown aggregation relationship"
    point_count: int = 0
    paired_point_count: int = 0
    valid_bar_count: int = 0
    valid_line_count: int = 0
    missing_bar_count: int = 0
    missing_line_count: int = 0
    bar_values: dict[str, float] = Field(default_factory=dict)
    line_values: dict[str, float] = Field(default_factory=dict)
    bar_highest_x: Any | None = None
    bar_highest_value: float | None = None
    bar_lowest_x: Any | None = None
    bar_lowest_value: float | None = None
    bar_range: float | None = None
    line_highest_x: Any | None = None
    line_highest_value: float | None = None
    line_lowest_x: Any | None = None
    line_lowest_value: float | None = None
    line_range: float | None = None
    same_highest_x: bool | None = None
    same_lowest_x: bool | None = None
    bar_ranking: list[str] = Field(default_factory=list)
    line_ranking: list[str] = Field(default_factory=list)
    pearson_correlation: float | None = None
    spearman_correlation: float | None = None
    relationship_strength: str | None = None
    relationship_direction: str | None = None
    normalized_bar_values: dict[str, float] = Field(default_factory=dict)
    normalized_line_values: dict[str, float] = Field(default_factory=dict)
    largest_positive_divergence_x: Any | None = None
    largest_negative_divergence_x: Any | None = None
    largest_normalized_divergence_x: Any | None = None
    bar_start_x: Any | None = None
    bar_start_value: float | None = None
    bar_end_x: Any | None = None
    bar_end_value: float | None = None
    bar_change: float | None = None
    bar_change_percent: float | None = None
    bar_endpoint_direction: str | None = None
    line_start_x: Any | None = None
    line_start_value: float | None = None
    line_end_x: Any | None = None
    line_end_value: float | None = None
    line_change: float | None = None
    line_change_percent: float | None = None
    line_endpoint_direction: str | None = None
    bar_peak_x: Any | None = None
    bar_peak_value: float | None = None
    bar_trough_x: Any | None = None
    bar_trough_value: float | None = None
    line_peak_x: Any | None = None
    line_peak_value: float | None = None
    line_trough_x: Any | None = None
    line_trough_value: float | None = None
    peaks_aligned: bool | None = None
    troughs_aligned: bool | None = None
    aligned_direction_count: int | None = None
    comparable_transition_count: int | None = None
    aligned_direction_percent: float | None = None
    opposite_direction_count: int | None = None
    opposite_direction_percent: float | None = None
    bar_volatility_level: str | None = None
    line_volatility_level: str | None = None
    more_volatile_metric: str | None = None
    known_metric_relationship: str | None = None
    derived_metric_available: str | None = None
    missing_x_values: list[Any] = Field(default_factory=list)
    irregular_intervals: bool = False
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    top_n_applied: int | None = None
    warnings: list[str] = Field(default_factory=list)


class ScatterEvidence(ChartEvidence):
    """Evidence for raw scatter plots using only displayed observations."""

    chart_type: str = "scatter"
    x_column: str
    y_column: str
    color_column: str | None = None
    size_column: str | None = None
    x_unit: str | None = None
    y_unit: str | None = None
    raw_row_count: int = 0
    valid_point_count: int = 0
    displayed_point_count: int = 0
    sampled: bool = False
    sampling_method: str | None = None
    sampling_fraction: float | None = None
    x_min: float | None = None
    x_max: float | None = None
    x_mean: float | None = None
    x_median: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    y_mean: float | None = None
    y_median: float | None = None
    pearson_correlation: float | None = None
    spearman_correlation: float | None = None
    r_squared: float | None = None
    relationship_direction: str | None = None
    relationship_strength: str | None = None
    relationship_form: str | None = None
    linear_slope: float | None = None
    linear_intercept: float | None = None
    outlier_count: int = 0
    influential_point_count: int = 0
    outlier_indices: list[int] = Field(default_factory=list)
    cluster_count: int | None = None
    cluster_summary: list[dict[str, Any]] = Field(default_factory=list)
    band_count: int | None = None
    banding_detected: bool = False
    heteroscedasticity_detected: bool = False
    variance_pattern: str | None = None
    color_group_summary: dict[str, dict[str, float]] = Field(default_factory=dict)
    group_relationships: dict[str, dict[str, Any]] = Field(default_factory=dict)
    size_relationship_with_x: float | None = None
    size_relationship_with_y: float | None = None
    known_metric_relationship: str | None = None
    mathematical_dependency: str | None = None
    derived_metric_available: str | None = None
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class CircleViewEvidence(ChartEvidence):
    """Evidence for Circle View charts with x, y, bubble-size, and optional color encodings."""

    chart_type: str = "circle_view"
    x_column: str
    y_column: str
    size_column: str
    color_column: str | None = None
    x_unit: str | None = None
    y_unit: str | None = None
    size_unit: str | None = None
    raw_row_count: int = 0
    valid_point_count: int = 0
    displayed_point_count: int = 0
    sampled: bool = False
    sampling_method: str | None = None
    sampling_fraction: float | None = None
    aggregated: bool = False
    aggregation_metadata: dict[str, Any] = Field(default_factory=dict)
    x_min: float | None = None
    x_max: float | None = None
    x_mean: float | None = None
    x_median: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    y_mean: float | None = None
    y_median: float | None = None
    size_min: float | None = None
    size_max: float | None = None
    size_mean: float | None = None
    size_median: float | None = None
    largest_bubble_index: int | None = None
    largest_bubble_x: float | None = None
    largest_bubble_y: float | None = None
    largest_bubble_size: float | None = None
    largest_bubble_group: str | None = None
    top_size_observations: list[dict[str, Any]] = Field(default_factory=list)
    pearson_xy: float | None = None
    spearman_xy: float | None = None
    r_squared_xy: float | None = None
    pearson_size_x: float | None = None
    spearman_size_x: float | None = None
    pearson_size_y: float | None = None
    spearman_size_y: float | None = None
    xy_relationship_direction: str | None = None
    xy_relationship_strength: str | None = None
    xy_relationship_form: str | None = None
    size_x_relationship_direction: str | None = None
    size_x_relationship_strength: str | None = None
    size_y_relationship_direction: str | None = None
    size_y_relationship_strength: str | None = None
    bubble_quadrant_summary: dict[str, dict[str, Any]] = Field(default_factory=dict)
    largest_bubble_quadrant: str | None = None
    large_bubble_concentration: str | None = None
    banding_detected: bool = False
    band_count: int | None = None
    cluster_count: int | None = None
    cluster_summary: list[dict[str, Any]] = Field(default_factory=list)
    outlier_count: int = 0
    influential_point_count: int = 0
    outlier_summary: list[dict[str, Any]] = Field(default_factory=list)
    overlap_level: str | None = None
    heteroscedasticity_detected: bool = False
    color_group_count: int = 0
    color_group_summary: dict[str, dict[str, Any]] = Field(default_factory=dict)
    group_with_largest_total_size: str | None = None
    group_with_largest_average_size: str | None = None
    group_with_largest_single_bubble: str | None = None
    known_xy_relationship: str | None = None
    known_size_relationship: str | None = None
    mathematical_dependency: str | None = None
    derived_metric_available: str | None = None
    similar_position_different_size_count: int = 0
    high_xy_small_size_count: int = 0
    moderate_xy_large_size_count: int = 0
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class HistogramEvidence(ChartEvidence):
    """Evidence for histogram distributions using displayed numeric values."""

    chart_type: str = "histogram"
    value_column: str
    color_column: str | None = None
    unit: str | None = None
    raw_row_count: int = 0
    valid_value_count: int = 0
    displayed_value_count: int = 0
    missing_value_count: int = 0
    excluded_value_count: int = 0
    sampled: bool = False
    sampling_method: str | None = None
    sampling_fraction: float | None = None
    bin_count: int = 0
    bin_width: float | None = None
    bin_edges: list[float] = Field(default_factory=list)
    bin_counts: list[int] = Field(default_factory=list)
    bin_method: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    value_range: float | None = None
    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None
    q1: float | None = None
    q3: float | None = None
    iqr: float | None = None
    p05: float | None = None
    p10: float | None = None
    p90: float | None = None
    p95: float | None = None
    skewness: float | None = None
    skew_direction: str | None = None
    skew_strength: str | None = None
    kurtosis: float | None = None
    tail_description: str | None = None
    modal_bin_start: float | None = None
    modal_bin_end: float | None = None
    modal_bin_count: int | None = None
    modal_bin_share: float | None = None
    lower_half_share: float | None = None
    upper_tail_share: float | None = None
    zero_count: int = 0
    zero_share: float | None = None
    negative_count: int = 0
    negative_share: float | None = None
    potential_outlier_count: int = 0
    potential_outlier_share: float | None = None
    lower_outlier_count: int = 0
    upper_outlier_count: int = 0
    outlier_method: str | None = None
    lower_outlier_threshold: float | None = None
    upper_outlier_threshold: float | None = None
    mode_count_estimate: int | None = None
    multimodal: bool = False
    multimodal_evidence: str | None = None
    sparse_intervals: list[dict[str, float]] = Field(default_factory=list)
    group_summary: dict[str, dict[str, Any]] = Field(default_factory=dict)
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class BoxGroupEvidence(BaseModel):
    """Distribution evidence for one displayed box."""

    x_value: str
    breakdown_value: str | None = None
    display_label: str
    observation_count: int
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    q1: float | None = None
    q3: float | None = None
    iqr: float | None = None
    lower_whisker: float | None = None
    upper_whisker: float | None = None
    lower_outlier_count: int = 0
    upper_outlier_count: int = 0
    potential_outlier_count: int = 0
    potential_outlier_share: float | None = None
    skew_direction: str | None = None


class BoxPlotEvidence(ChartEvidence):
    """Evidence for grouped box plots with x categories and optional breakdown groups."""

    chart_type: str = "box"
    x_column: str
    y_column: str
    breakdown_column: str | None = None
    x_category_count: int = 0
    breakdown_category_count: int = 0
    box_count: int = 0
    groups: list[BoxGroupEvidence] = Field(default_factory=list)
    highest_median_combination: str | None = None
    highest_median_value: float | None = None
    lowest_median_combination: str | None = None
    lowest_median_value: float | None = None
    widest_iqr_combination: str | None = None
    widest_iqr_value: float | None = None
    narrowest_iqr_combination: str | None = None
    narrowest_iqr_value: float | None = None
    breakdown_leader_by_x: dict[str, str] = Field(default_factory=dict)
    breakdown_laggard_by_x: dict[str, str] = Field(default_factory=dict)
    breakdown_median_gap_by_x: dict[str, float] = Field(default_factory=dict)
    breakdown_lead_counts: dict[str, int] = Field(default_factory=dict)
    x_categories_with_ranking_changes: list[str] = Field(default_factory=list)
    total_potential_outlier_count: int = 0
    groups_with_outliers: list[str] = Field(default_factory=list)
    unequal_sample_sizes: bool = False
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ChartInsight(BaseModel):
    """Validated insight displayed in the chart insight card."""

    chart_title: str
    key_finding: str
    supporting_evidence: str
    interpretation: str
    caution: str | None = None
    recommended_next_step: str
    evidence_strength: EvidenceStrength = "medium"
    evidence: ChartEvidence | CorrelationHeatmapEvidence | PieChartEvidence | TreemapEvidence | SymbolMapEvidence | SortedPercentageBarEvidence | PeriodOverPeriodChangeEvidence | SingleBarEvidence | StackedBarEvidence | GroupedBarEvidence | SingleLineEvidence | SingleAreaEvidence | StackedAreaEvidence | DualLineEvidence | DualCombinationEvidence | ScatterEvidence | CircleViewEvidence | HistogramEvidence | BoxPlotEvidence | None = None

    @model_validator(mode="after")
    def validate_readable_sections(self) -> "ChartInsight":
        for field_name in (
            "chart_title",
            "key_finding",
            "supporting_evidence",
            "interpretation",
            "recommended_next_step",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} cannot be empty.")
        return self

    @property
    def headline(self) -> str:
        """Backward-compatible alias used by older tests/components."""
        return self.key_finding

    @property
    def observations(self) -> list[str]:
        """Backward-compatible summary list."""
        if self.evidence and self.evidence.chart_type == "dual_line":
            return [self.supporting_evidence, self.interpretation]
        return [
            item for item in (
                self.key_finding,
                self.supporting_evidence,
                self.interpretation,
                self.caution,
                self.recommended_next_step,
            )
            if item
        ]

    @property
    def caveat(self) -> str | None:
        """Backward-compatible alias for caution."""
        return self.caution


def normalize_column_name(value: str | None) -> str:
    """Normalize a column name so spelling variants compare consistently."""
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def display_name(value: str | None) -> str:
    """Render a technical column name as a friendly label."""
    if not value:
        return ""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(value))
    tokens = re.split(r"[\s_\-]+", spaced)
    rendered = []
    for token in tokens:
        if not token:
            continue
        upper = token.upper()
        rendered.append(upper if upper in ABBREVIATIONS else token[:1].upper() + token[1:])
    return " ".join(rendered)


def friendly_value(column: str | None, value: Any) -> str:
    """Render compact categorical codes with known semantic labels."""
    text = str(value)
    mapping = VALUE_LABELS.get(normalize_column_name(column))
    if not mapping:
        return text
    return mapping.get(text, text)


def _fmt(value: Any, metric: str | None = None) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)) and isfinite(float(value)):
        return format_metric(float(value), metric)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return str(value)


def _pct(value: float | None) -> str:
    if value is None or not isfinite(float(value)):
        return "not available"
    return f"{float(value):.1f}%"


def _date_summary_sentence(evidence: ChartEvidence, metric_column: str | None = None) -> str | None:
    summary = evidence.calculated_metrics.get("date_summary")
    if not isinstance(summary, dict):
        return None
    metric = metric_column or summary.get("metric")
    aggregation = summary.get("aggregation")
    value = summary.get("value")
    label = summary.get("period_label") or "the selected period"
    if value is None:
        return None
    if aggregation == "mean":
        return f"The overall average {display_name(metric)} for {label} is {_fmt(value, metric)}."
    if aggregation == "count":
        return f"The selected period contains {int(value):,} record(s)."
    if aggregation in {"min", "max", "median"}:
        return f"The overall {aggregation} {display_name(metric)} for {label} is {_fmt(value, metric)}."
    return f"Total {display_name(metric)} for {label} is {_fmt(value, metric)}."


def _records(result: ChartResult) -> list[dict[str, Any]]:
    return [row for row in result.data if isinstance(row, dict)]


def _metric_column(result: ChartResult) -> str | None:
    spec = result.spec
    if spec.y:
        return spec.y
    if spec.aggregation == "count":
        return "Count"
    return None


def _numeric_series(rows: list[dict[str, Any]], column: str | None) -> pd.Series:
    if not column:
        return pd.Series(dtype="float64")
    return pd.to_numeric(pd.Series([row.get(column) for row in rows]), errors="coerce").dropna()


def _base_evidence(result: ChartResult) -> ChartEvidence:
    spec = result.spec
    rows = _records(result)
    y_columns = [column for column in (spec.y, spec.secondary_y) if column]
    if spec.value_columns:
        y_columns = list(spec.value_columns)
    elif spec.aggregation == "count" and not y_columns:
        y_columns = ["Count"]
    filters = {}
    if spec.filter_column:
        filters[spec.filter_column] = spec.filter_value
    base = ChartEvidence(
        chart_type=spec.chart_type,
        chart_title=spec.title,
        x_column=spec.x,
        y_columns=y_columns,
        category_column=spec.x,
        group_column=spec.color,
        color_column=spec.color,
        size_column=spec.secondary_y if spec.chart_type == "circle_view" else None,
        aggregation=spec.aggregation,
        filters=filters,
        sorting="descending" if spec.sort_descending else None,
        top_n=spec.limit,
        total_rows=len(rows),
        valid_rows=len(rows),
        excluded_rows=0,
        evidence_strength="medium" if len(rows) >= 3 else "low",
    )
    if result.metadata.get("date_summary"):
        base.calculated_metrics["date_summary"] = result.metadata["date_summary"]
    return base


def _share_metrics(values: pd.Series) -> dict[str, Any]:
    total = float(values.sum()) if not values.empty else 0.0
    top_values = values.sort_values(ascending=False)
    return {
        "displayed_total": total,
        "top_share_pct": float(top_values.iloc[0] / total * 100) if total else None,
        "top_two_share_pct": float(top_values.head(2).sum() / total * 100) if total else None,
        "top_three_share_pct": float(top_values.head(3).sum() / total * 100) if total else None,
        "average_value": float(values.mean()) if not values.empty else None,
    }


def _bar_like_evidence(result: ChartResult, evidence: ChartEvidence) -> ChartEvidence:
    rows = _records(result)
    spec = result.spec
    y_col = _metric_column(result)
    if spec.value_columns and {"Metric", "Value"}.issubset(pd.DataFrame(rows).columns):
        metrics = {}
        frame = pd.DataFrame(rows)
        for metric, group in frame.groupby("Metric", dropna=False):
            values = pd.to_numeric(group["Value"], errors="coerce")
            leader = group.loc[values.idxmax()]
            metrics[str(metric)] = {
                "highest_category": leader.get(spec.x),
                "highest_value": float(leader["Value"]),
            }
        evidence.calculated_metrics["multi_metric_leaders"] = metrics
        evidence.detected_patterns.append("multiple metrics compared across one category")
        return evidence
    values = _numeric_series(rows, y_col)
    if values.empty or not y_col or not spec.x:
        evidence.warnings.append("No numeric plotted values were available for bar evidence.")
        evidence.evidence_strength = "low"
        return evidence
    frame = pd.DataFrame(rows).loc[values.index]
    highest = frame.loc[values.idxmax()]
    lowest = frame.loc[values.idxmin()]
    gap = float(highest[y_col]) - float(lowest[y_col])
    pct_gap = gap / abs(float(lowest[y_col])) * 100 if float(lowest[y_col]) else None
    evidence.calculated_metrics.update({
        "category_count": int(frame[spec.x].nunique(dropna=True)),
        "highest_category": highest[spec.x],
        "highest_value": float(highest[y_col]),
        "lowest_category": lowest[spec.x],
        "lowest_value": float(lowest[y_col]),
        "absolute_gap": gap,
        "percentage_gap": pct_gap,
        **_share_metrics(values),
    })
    if spec.color:
        totals = frame.groupby(spec.x, dropna=False)[y_col].sum()
        segments = frame.groupby(spec.color, dropna=False)[y_col].sum()
        evidence.calculated_metrics.update({
            "largest_total_category": totals.idxmax(),
            "largest_total_value": float(totals.max()),
            "dominant_segment": segments.idxmax(),
            "dominant_segment_value": float(segments.max()),
        })
        evidence.detected_patterns.append("segment composition is available")
    if evidence.calculated_metrics.get("top_three_share_pct", 0) and evidence.calculated_metrics["top_three_share_pct"] > 70:
        evidence.detected_patterns.append("values are concentrated in the top categories")
    return evidence


def is_single_series_bar(result: ChartResult) -> bool:
    """Return true only for standard one-series bar charts."""
    spec = result.spec
    y_columns = [column for column in (spec.y,) if column]
    if spec.aggregation == "count" and not y_columns:
        y_columns = ["Count"]
    return (
        spec.chart_type == "bar"
        and not spec.color
        and not spec.value_columns
        and not spec.secondary_y
        and len(y_columns) == 1
    )


def is_single_series_line(result: ChartResult) -> bool:
    """Return true only for standard one-series line charts."""
    spec = result.spec
    return (
        spec.chart_type == "line"
        and bool(spec.x)
        and bool(spec.y)
        and not spec.color
        and not spec.value_columns
        and not spec.secondary_y
    )


def is_single_area(result: ChartResult) -> bool:
    """Return true for area charts with one value series and no stack/color field."""
    spec = result.spec
    return (
        spec.chart_type == "area"
        and bool(spec.x)
        and bool(spec.y)
        and not spec.color
        and not spec.value_columns
        and not spec.secondary_y
    )


def is_stacked_area(result: ChartResult) -> bool:
    """Return true for area charts where color represents stacked components."""
    spec = result.spec
    return (
        spec.chart_type == "area"
        and bool(spec.x)
        and bool(spec.y)
        and bool(spec.color)
        and not spec.value_columns
        and not spec.secondary_y
    )


def _infer_time_granularity(values: pd.Series, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    dates = pd.to_datetime(values, errors="coerce").dropna().sort_values()
    if len(dates) < 2:
        return None
    deltas = dates.diff().dropna().dt.days
    if deltas.empty:
        return None
    median_days = float(deltas.median())
    if (dates.dt.day == 1).all() and 27 <= median_days <= 62:
        return "month"
    if 6 <= median_days <= 8:
        return "week"
    if 27 <= median_days <= 32:
        return "month"
    if 88 <= median_days <= 93:
        return "quarter"
    if 360 <= median_days <= 370:
        return "year"
    return "day"


def _area_axis_order(axis: pd.Series, x_column: str | None) -> tuple[pd.Series, str | None, str, bool]:
    dates = pd.to_datetime(axis, errors="coerce", format="mixed")
    axis_name = str(x_column or "").lower()
    if dates.notna().all() or (
        dates.notna().any()
        and any(token in axis_name for token in ("date", "time", "period", "month", "year", "week"))
    ):
        return dates, _infer_time_granularity(dates), "datetime", dates.notna().all()
    numeric = pd.to_numeric(axis, errors="coerce")
    if numeric.notna().all():
        return numeric, None, "numeric", True
    labels = axis.astype("string")
    quarter_match = labels.str.fullmatch(r"(?i)q[1-4]").fillna(False)
    if bool(quarter_match.all()):
        return labels.str.extract(r"([1-4])", expand=False).astype(float), "quarter", "ordered_period", True
    return pd.Series(range(len(axis)), index=axis.index, dtype="float64"), None, "categorical", False


def _period_labels(values: pd.Series, granularity: str | None) -> list[str]:
    return [format_period(value, granularity) for value in values]


def _is_ordered_line_axis(axis: pd.Series, x_column: str | None) -> tuple[pd.Series, str | None, bool]:
    dates = pd.to_datetime(axis, errors="coerce")
    axis_name = str(x_column or "").lower()
    looks_temporal = (
        dates.notna().all()
        or any(token in axis_name for token in ("date", "time", "period", "month", "year", "week"))
    )
    if looks_temporal and dates.notna().all():
        return dates, _infer_time_granularity(dates), True
    numeric = pd.to_numeric(axis, errors="coerce")
    if numeric.notna().all():
        return numeric, None, True
    return pd.Series(pd.NA, index=axis.index), None, False


def _period_change_percent(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return (current - previous) / abs(previous) * 100


def _trend_strength(r_squared: float | None) -> str | None:
    if r_squared is None:
        return None
    if r_squared < 0.10:
        return "weak"
    if r_squared < 0.35:
        return "moderate"
    return "strong"


def _volatility_level(coefficient_of_variation: float | None) -> str | None:
    if coefficient_of_variation is None:
        return None
    if coefficient_of_variation < 0.10:
        return "low"
    if coefficient_of_variation < 0.20:
        return "moderate"
    return "high"


def _direction_reversals(changes: pd.Series) -> int:
    signs = np.sign(changes[changes != 0])
    if len(signs) < 2:
        return 0
    return int((signs.shift() != signs).iloc[1:].sum())


def _classify_line_pattern(
    slope: float | None,
    values: pd.Series,
    volatility: str | None,
    trend_strength: str | None,
    endpoint_change: float | None,
    value_range: float | None,
) -> str:
    if slope is None or values.empty:
        return "mixed movement with no clear trend"
    value_scale = max(abs(float(values.mean())), 1.0)
    flat_slope = abs(slope) / value_scale < 0.01
    endpoint_flat = (
        endpoint_change is not None
        and value_range is not None
        and value_range > 0
        and abs(endpoint_change) / value_range < 0.15
    )
    if endpoint_flat and volatility == "high":
        return "mostly flat but volatile"
    if flat_slope and volatility == "high":
        return "mostly flat but volatile"
    if flat_slope:
        return "mostly flat and stable" if volatility == "low" else "mixed movement with no clear trend"
    if slope > 0 and trend_strength == "strong" and volatility == "low":
        return "steady upward trend"
    if slope < 0 and trend_strength == "strong" and volatility == "low":
        return "steady downward trend"
    if slope > 0 and volatility == "high":
        return "upward but volatile"
    if slope < 0 and volatility == "high":
        return "downward but volatile"
    return "mixed movement with no clear trend"


def _seasonality_evidence(point_count: int, granularity: str | None) -> str:
    minimums = {"month": 24, "quarter": 12, "week": 104}
    if granularity not in minimums:
        return "seasonality not established"
    if point_count < minimums[granularity]:
        return "insufficient history"
    return "no seasonality test performed"


def _longest_run(mask: pd.Series) -> int:
    longest = current = 0
    for value in mask.fillna(False):
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _area_volatility_label(coefficient_of_variation: float | None) -> str | None:
    if coefficient_of_variation is None:
        return None
    if coefficient_of_variation < 0.10:
        return "stable"
    if coefficient_of_variation < 0.20:
        return "moderately variable"
    return "highly volatile"


def _classify_area_pattern(
    slope: float | None,
    values: pd.Series,
    volatility: str | None,
    trend_strength: str | None,
) -> str:
    if slope is None or values.empty:
        return "mixed movement with no clear trend"
    value_scale = max(abs(float(values.mean())), 1.0)
    flat_slope = abs(slope) / value_scale < 0.01
    if flat_slope:
        return "broadly stable" if volatility != "highly volatile" else "mixed movement with no clear trend"
    if slope > 0 and trend_strength == "strong" and volatility == "stable":
        return "steady upward movement"
    if slope < 0 and trend_strength == "strong" and volatility == "stable":
        return "steady downward movement"
    if slope > 0 and volatility == "highly volatile":
        return "upward but volatile"
    if slope < 0 and volatility == "highly volatile":
        return "downward but volatile"
    return "mixed movement with no clear trend"


def _metric_unit(column: str | None) -> str | None:
    normalized = normalize_column_name(column)
    if any(token in normalized for token in ("revenue", "sales", "cost", "profit", "price", "amount", "income")):
        return "currency"
    if any(token in normalized for token in ("margin", "rate", "percent", "pct", "ratio")):
        return "percentage"
    if any(token in normalized for token in ("unit", "quantity", "volume", "count", "orders")):
        return "quantity"
    if any(token in normalized for token in ("duration", "time", "days", "hours", "minutes")):
        return "duration"
    if "temperature" in normalized or "temp" in normalized:
        return "temperature"
    if any(token in normalized for token in ("energy", "consumption", "kwh")):
        return "energy"
    if any(token in normalized for token in ("score", "rating", "satisfaction")):
        return "score"
    return None


def _unit_relationship(primary_unit: str | None, secondary_unit: str | None) -> str:
    if not primary_unit or not secondary_unit:
        return "unknown unit"
    if primary_unit == secondary_unit:
        return "same unit"
    return "different unit"


def _relationship_strength(correlation: float | None) -> str | None:
    if correlation is None or not isfinite(float(correlation)):
        return None
    absolute = abs(float(correlation))
    if absolute < 0.20:
        return "very weak"
    if absolute < 0.40:
        return "weak"
    if absolute < 0.60:
        return "moderate"
    if absolute < 0.80:
        return "strong"
    return "very strong"


def _endpoint_direction_label(change: float | None, start_value: float | None) -> str | None:
    if change is None or start_value is None:
        return None
    denominator = max(abs(start_value), 1.0)
    if abs(change) / denominator < 0.01:
        return "approximately unchanged"
    return "higher" if change > 0 else "lower"


def _series_volatility(values: pd.Series) -> tuple[float | None, str | None]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2:
        return None, None
    mean = float(values.mean())
    if mean == 0:
        return None, None
    coefficient = abs(float(values.std()) / mean)
    return coefficient, _volatility_level(coefficient)


def _strongest_change(values: pd.Series, labels: list[str]) -> dict[str, Any]:
    changes = pd.to_numeric(values, errors="coerce").diff().dropna()
    result: dict[str, Any] = {}
    if changes.empty:
        return result
    increase_index = changes.idxmax()
    decline_index = changes.idxmin()
    if float(changes.loc[increase_index]) > 0:
        result["increase_start"] = labels[int(increase_index - 1)]
        result["increase_end"] = labels[int(increase_index)]
        result["increase_value"] = float(changes.loc[increase_index])
    if float(changes.loc[decline_index]) < 0:
        result["decline_start"] = labels[int(decline_index - 1)]
        result["decline_end"] = labels[int(decline_index)]
        result["decline_value"] = float(changes.loc[decline_index])
    return result


def _index_to_100(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    first = values.dropna().iloc[0] if values.notna().any() else np.nan
    if pd.isna(first) or first == 0:
        minimum = values.min()
        maximum = values.max()
        if pd.isna(minimum) or maximum == minimum:
            return pd.Series(100.0, index=values.index)
        return (values - minimum) / (maximum - minimum) * 100
    return values / abs(float(first)) * 100


def _known_metric_relationship(primary: str | None, secondary: str | None) -> tuple[str | None, str | None]:
    normalized = {normalize_column_name(primary), normalize_column_name(secondary)}
    has_revenue = any("revenue" in item or "sales" in item for item in normalized)
    has_cost = any("cost" in item for item in normalized)
    has_profit = any("profit" in item for item in normalized)
    if has_revenue and has_profit:
        return "Profit reflects the portion of revenue remaining after costs.", "profit margin"
    if has_revenue and has_cost:
        return "The difference between revenue and cost represents profit.", "profit"
    if any("actual" in item for item in normalized) and any("target" in item for item in normalized):
        return "The difference represents performance against target.", "variance"
    if any("revenue" in item or "sales" in item for item in normalized) and any("unit" in item or "quantity" in item for item in normalized):
        return "Revenue divided by units can indicate revenue per unit.", "revenue per unit"
    return None, None


KNOWN_FORMULAS = {
    "totalrevenue": {
        "inputs": {"unitssold", "unitprice"},
        "formula": "Total Revenue = Units Sold x Unit Price",
        "derived_metric": "revenue per unit",
    },
    "totalcost": {
        "inputs": {"unitssold", "unitcost"},
        "formula": "Total Cost = Units Sold x Unit Cost",
        "derived_metric": "cost per unit",
    },
    "totalprofit": {
        "inputs": {"totalrevenue", "totalcost"},
        "formula": "Total Profit = Total Revenue - Total Cost",
        "derived_metric": "profit margin",
    },
}


def _scatter_formula_dependency(frame: pd.DataFrame, x_col: str, y_col: str) -> tuple[str | None, str | None, str | None]:
    normalized_columns = {normalize_column_name(column): column for column in frame.columns}
    x_norm = normalize_column_name(x_col)
    y_norm = normalize_column_name(y_col)
    plotted = {x_norm, y_norm}
    for output, metadata in KNOWN_FORMULAS.items():
        inputs = set(metadata["inputs"])
        if output in plotted and plotted & inputs and inputs.issubset(normalized_columns):
            output_name = display_name(normalized_columns.get(output, output))
            input_names = ", ".join(display_name(normalized_columns[item]) for item in sorted(inputs))
            dependency = (
                f"{output_name} is mathematically dependent on {input_names}, so the association is partly structural."
            )
            return metadata["formula"], dependency, str(metadata["derived_metric"])
        if plotted == {output, "totalrevenue"} and output == "totalprofit" and inputs.issubset(normalized_columns):
            return metadata["formula"], "Total Profit is derived from Total Revenue and Total Cost, so the association partly reflects an accounting identity.", str(metadata["derived_metric"])
    if {x_norm, y_norm} == {"unitcost", "unitprice"}:
        return "Unit Price - Unit Cost = Unit Margin", "The difference between Unit Price and Unit Cost represents unit margin.", "unit margin"
    return _known_metric_relationship(x_col, y_col)[0], None, _known_metric_relationship(x_col, y_col)[1]


def _finite_pair(frame: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    pair = frame[[x_col, y_col]].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(pair[x_col]) & np.isfinite(pair[y_col])
    return pair.loc[finite].dropna()


def _scatter_relationship_form(
    pearson: float | None,
    spearman: float | None,
    r_squared: float | None,
    banding: bool,
    heteroscedasticity: bool,
    cluster_count: int | None,
) -> str:
    if heteroscedasticity:
        return "fan-shaped"
    if banding:
        return "banded"
    if cluster_count and cluster_count >= 2:
        return "clustered"
    if pearson is None or spearman is None:
        return "no clear relationship"
    if abs(pearson) < 0.20 and abs(spearman) < 0.20:
        return "weak or diffuse"
    if abs(spearman) - abs(pearson) >= 0.15 and abs(spearman) >= 0.40:
        return "monotonic but non-linear"
    if abs(pearson) >= 0.60 and abs(spearman) + 0.15 < abs(pearson):
        return "approximately linear with influential points"
    if r_squared is not None and r_squared >= 0.35 and abs(pearson - spearman) < 0.15:
        return "approximately linear"
    return "approximately linear" if abs(pearson - spearman) < 0.15 else "monotonic but non-linear"


def _detect_banding(pair: pd.DataFrame, x_col: str, y_col: str) -> tuple[bool, int | None]:
    nonzero = pair.loc[pair[x_col] != 0].copy()
    if len(nonzero) < 12:
        return False, None
    ratios = (nonzero[y_col] / nonzero[x_col]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratios) < 12:
        return False, None
    rounded = ratios.round(2)
    counts = rounded.value_counts()
    repeated = counts[counts >= max(3, int(len(ratios) * 0.03))]
    coverage = float(repeated.sum() / len(ratios)) if len(ratios) else 0.0
    detected = len(repeated) >= 3 and coverage >= 0.25
    return detected, int(len(repeated)) if detected else None


def _detect_outliers(pair: pd.DataFrame, x_col: str, y_col: str) -> tuple[int, int, list[int]]:
    if len(pair) < 8 or pair[x_col].nunique() < 2:
        return 0, 0, []
    slope, intercept = np.polyfit(pair[x_col], pair[y_col], 1)
    residuals = pair[y_col] - (slope * pair[x_col] + intercept)
    median = float(residuals.median())
    mad = float((residuals - median).abs().median())
    if mad == 0:
        q1, q3 = residuals.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        mask = (residuals < q1 - 1.5 * iqr) | (residuals > q3 + 1.5 * iqr) if iqr else pd.Series(False, index=residuals.index)
    else:
        robust_z = 0.6745 * (residuals - median).abs() / mad
        mask = robust_z > 3.5
    outlier_indices = [int(index) for index in residuals.loc[mask].index[:10]]
    leverage_threshold = 2 / max(len(pair), 1)
    x_mean = float(pair[x_col].mean())
    leverage = ((pair[x_col] - x_mean) ** 2) / float(((pair[x_col] - x_mean) ** 2).sum()) if len(pair) > 1 else pd.Series(0, index=pair.index)
    influential = int(((mask) & (leverage > leverage_threshold)).sum())
    return int(mask.sum()), influential, outlier_indices


def _detect_heteroscedasticity(pair: pd.DataFrame, x_col: str, y_col: str) -> tuple[bool, str | None]:
    if len(pair) < 30 or pair[x_col].nunique() < 10:
        return False, None
    try:
        bins = pd.qcut(pair[x_col], q=4, duplicates="drop")
    except ValueError:
        return False, None
    grouped_std = pair.groupby(bins, observed=False)[y_col].std().dropna()
    if len(grouped_std) < 3 or float(grouped_std.min()) == 0:
        return False, None
    ratio = float(grouped_std.max() / grouped_std.min())
    if ratio >= 2.0:
        increasing = grouped_std.iloc[-1] > grouped_std.iloc[0]
        pattern = "vertical spread widens at higher x-values" if increasing else "vertical spread changes across x-values"
        return True, pattern
    return False, None


def _color_group_evidence(frame: pd.DataFrame, pair: pd.DataFrame, x_col: str, y_col: str, color_col: str | None) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]], int | None]:
    if not color_col or color_col not in frame:
        return {}, {}, None
    valid = frame.loc[pair.index, [color_col]].join(pair)
    summaries: dict[str, dict[str, float]] = {}
    relationships: dict[str, dict[str, Any]] = {}
    counts = valid[color_col].astype(str).value_counts()
    for group in counts.head(3).index:
        subset = valid.loc[valid[color_col].astype(str) == group]
        summaries[str(group)] = {
            "point_count": float(len(subset)),
            "x_median": float(subset[x_col].median()),
            "y_median": float(subset[y_col].median()),
        }
        if len(subset) >= 3 and subset[x_col].nunique() > 1 and subset[y_col].nunique() > 1:
            corr = float(subset[x_col].corr(subset[y_col]))
            relationships[str(group)] = {
                "pearson_correlation": corr,
                "relationship_strength": _relationship_strength(corr),
                "relationship_direction": "positive" if corr > 0 else "negative" if corr < 0 else "none or unclear",
            }
    cluster_count = int(min(len(counts), 3)) if len(counts) >= 2 else None
    return summaries, relationships, cluster_count


def _correlation_summary(frame: pd.DataFrame, first: str, second: str) -> tuple[float | None, float | None, str | None, str | None]:
    pair = _finite_pair(frame, first, second)
    if len(pair) < 3 or pair[first].nunique() <= 1 or pair[second].nunique() <= 1:
        return None, None, None, None
    pearson = float(pair[first].corr(pair[second]))
    spearman = float(pair[first].corr(pair[second], method="spearman"))
    if not isfinite(pearson) or not isfinite(spearman):
        return None, None, None, None
    direction = "positive" if pearson > 0 else "negative" if pearson < 0 else "none or unclear"
    return pearson, spearman, _relationship_strength(pearson), direction


def _circle_quadrants(frame: pd.DataFrame, x_col: str, y_col: str, size_col: str, color_col: str | None) -> tuple[dict[str, dict[str, Any]], str | None, str | None]:
    if frame.empty:
        return {}, None, None
    x_ref = float(frame[x_col].median())
    y_ref = float(frame[y_col].median())
    size_threshold = float(frame[size_col].quantile(0.75))

    def quadrant(row: pd.Series) -> str:
        high_x = float(row[x_col]) >= x_ref
        high_y = float(row[y_col]) >= y_ref
        if high_x and high_y:
            return "upper-right"
        if high_x and not high_y:
            return "lower-right"
        if not high_x and high_y:
            return "upper-left"
        return "lower-left"

    working = frame.copy()
    working["_quadrant"] = working.apply(quadrant, axis=1)
    total = len(working)
    summary: dict[str, dict[str, Any]] = {}
    for name, group in working.groupby("_quadrant", dropna=False):
        largest = group.loc[group[size_col].idxmax()]
        details: dict[str, Any] = {
            "point_count": int(len(group)),
            "point_share_percent": float(len(group) / total * 100) if total else 0.0,
            "total_size": float(group[size_col].sum()),
            "average_size": float(group[size_col].mean()),
            "largest_size": float(largest[size_col]),
        }
        if color_col and color_col in group:
            details["dominant_group"] = str(group[color_col].astype(str).value_counts().idxmax())
        summary[str(name)] = details
    largest_idx = frame[size_col].idxmax()
    largest_quadrant = str(working.loc[largest_idx, "_quadrant"])
    large = working.loc[working[size_col] >= size_threshold]
    concentration = None
    if not large.empty:
        counts = large["_quadrant"].value_counts()
        concentration = str(counts.idxmax()) if counts.iloc[0] / len(large) >= 0.5 else "distributed"
    return summary, largest_quadrant, concentration


def _circle_color_groups(frame: pd.DataFrame, x_col: str, y_col: str, size_col: str, color_col: str | None) -> tuple[dict[str, dict[str, Any]], str | None, str | None, str | None]:
    if not color_col or color_col not in frame or frame.empty:
        return {}, None, None, None
    summary: dict[str, dict[str, Any]] = {}
    for group_name, group in frame.groupby(color_col, dropna=False):
        group_label = str(group_name)
        largest = group.loc[group[size_col].idxmax()]
        pearson, spearman, _, _ = _correlation_summary(group, x_col, y_col)
        summary[group_label] = {
            "point_count": int(len(group)),
            "median_x": float(group[x_col].median()),
            "median_y": float(group[y_col].median()),
            "total_size": float(group[size_col].sum()),
            "median_size": float(group[size_col].median()),
            "average_size": float(group[size_col].mean()),
            "largest_bubble": float(largest[size_col]),
            "pearson_xy": pearson,
            "spearman_xy": spearman,
        }
    total_group = max(summary, key=lambda item: summary[item]["total_size"]) if summary else None
    average_group = max(summary, key=lambda item: summary[item]["average_size"]) if summary else None
    single_group = max(summary, key=lambda item: summary[item]["largest_bubble"]) if summary else None
    return summary, total_group, average_group, single_group


def _circle_overlap_level(frame: pd.DataFrame, x_col: str, y_col: str) -> str | None:
    if len(frame) < 20:
        return None
    x_range = float(frame[x_col].max() - frame[x_col].min())
    y_range = float(frame[y_col].max() - frame[y_col].min())
    if x_range == 0 or y_range == 0:
        return "high"
    rounded = pd.DataFrame({
        "x": ((frame[x_col] - frame[x_col].min()) / x_range * 20).round(),
        "y": ((frame[y_col] - frame[y_col].min()) / y_range * 20).round(),
    })
    collision_share = float(rounded.duplicated(keep=False).mean())
    if collision_share >= 0.35:
        return "high"
    if collision_share >= 0.15:
        return "moderate"
    return "low"


def _histogram_skew_labels(skewness: float | None) -> tuple[str | None, str | None]:
    if skewness is None or not isfinite(float(skewness)):
        return None, None
    absolute = abs(float(skewness))
    if absolute < 0.25:
        return "approximately symmetric", "approximately symmetric"
    strength = "mildly skewed" if absolute < 0.75 else "moderately skewed" if absolute < 1.5 else "strongly skewed"
    direction = "right-skewed" if skewness > 0 else "left-skewed"
    return direction, strength


def _histogram_bin_summary(values: pd.Series) -> tuple[list[float], list[int], str]:
    count = len(values)
    if count <= 1 or float(values.max()) == float(values.min()):
        minimum = float(values.min()) if count else 0.0
        maximum = float(values.max()) if count else 1.0
        return [minimum, maximum], [count], "constant"
    edges = np.histogram_bin_edges(values.to_numpy(dtype=float), bins="auto")
    counts, edges = np.histogram(values.to_numpy(dtype=float), bins=edges)
    return [float(edge) for edge in edges], [int(item) for item in counts], "numpy auto"


def _histogram_multimodal(counts: list[int], edges: list[float]) -> tuple[bool, int | None, str | None]:
    if len(counts) < 5:
        return False, None, None
    max_count = max(counts)
    if max_count <= 1:
        return False, None, None
    peaks = []
    for index in range(1, len(counts) - 1):
        if counts[index] >= counts[index - 1] and counts[index] >= counts[index + 1] and counts[index] >= max_count * 0.5:
            peaks.append(index)
    if len(peaks) >= 2:
        ranges = [f"{_fmt(edges[index])} to {_fmt(edges[index + 1])}" for index in peaks[:3]]
        return True, len(peaks), f"Possible multiple peaks near {', '.join(ranges)}."
    return False, None, None


def _histogram_group_summary(frame: pd.DataFrame, value_col: str, color_col: str | None) -> dict[str, dict[str, Any]]:
    if not color_col or color_col not in frame:
        return {}
    summary: dict[str, dict[str, Any]] = {}
    for group, subset in frame.groupby(color_col, dropna=False):
        values = pd.to_numeric(subset[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        upper_threshold = q3 + 1.5 * iqr
        summary[str(group)] = {
            "count": int(len(values)),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "iqr": float(iqr),
            "p90": float(values.quantile(0.90)),
            "p95": float(values.quantile(0.95)),
            "outlier_share": float((values > upper_threshold).sum() / len(values) * 100) if len(values) else 0.0,
        }
    return summary


def _aggregation_warning(primary: str | None, secondary: str | None) -> str | None:
    additive = {"sum", "count", "nunique"}
    average_like = {"mean", "median"}
    extrema = {"min", "max"}
    if primary == secondary:
        return None
    if (primary in additive and secondary in average_like) or (secondary in additive and primary in average_like):
        return "The chart compares a total or count with an average-like statistic, so the two series summarize the data differently."
    if primary in extrema or secondary in extrema:
        return "The chart compares an extreme value with another aggregation concept, so the series should not be interpreted as directly proportional."
    return "The chart uses different aggregation concepts for the two metrics."


def _aggregation_relationship(primary: str | None, secondary: str | None) -> str:
    if primary == secondary:
        return "same aggregation"
    warning = _aggregation_warning(primary, secondary)
    return warning or "different aggregation"


def _missing_periods(dates: pd.Series, granularity: str | None) -> tuple[list[Any], bool]:
    if granularity not in {"day", "week", "month", "quarter", "year"} or len(dates) < 3:
        return [], False
    frequency = {
        "day": "D",
        "week": "W",
        "month": "M",
        "quarter": "Q",
        "year": "Y",
    }[granularity]
    periods = dates.dt.to_period({
        "day": "D",
        "week": "W",
        "month": "M",
        "quarter": "Q",
        "year": "Y",
    }[granularity])
    unique_periods = pd.PeriodIndex(periods.dropna().unique()).sort_values()
    if len(unique_periods) < 3:
        return [], False
    expected = pd.period_range(unique_periods[0], unique_periods[-1], freq=frequency)
    missing = [period.start_time for period in expected.difference(unique_periods)]
    interval_steps = pd.Series([period.ordinal for period in unique_periods]).diff().dropna()
    irregular = bool((interval_steps != 1).any())
    return missing, irregular


def _aggregate_displayed_values(frame: pd.DataFrame, x_col: str, y_col: str, aggregation: str | None) -> pd.Series:
    grouped = frame.groupby(x_col, dropna=False)[y_col]
    if aggregation == "mean":
        values = grouped.mean()
    elif aggregation == "median":
        values = grouped.median()
    elif aggregation == "min":
        values = grouped.min()
    elif aggregation == "max":
        values = grouped.max()
    else:
        values = grouped.sum()
    return values.sort_values(ascending=False)


def _lead_strength(gap_percent: float | None) -> str:
    if gap_percent is None:
        return "unknown"
    if gap_percent < 5:
        return "narrow"
    if gap_percent < 15:
        return "moderate"
    return "clear"


def _concentration_level(top_two_share: float | None, top_three_share: float | None) -> str | None:
    if top_two_share is None and top_three_share is None:
        return None
    if top_two_share is not None and top_two_share >= 60:
        return "high"
    if top_three_share is not None and top_three_share >= 60:
        return "moderate"
    return "distributed"


def _single_bar_evidence(result: ChartResult, evidence: ChartEvidence) -> SingleBarEvidence:
    rows = _records(result)
    spec = result.spec
    y_col = _metric_column(result)
    single = SingleBarEvidence(**evidence.model_dump(), value_column=y_col)
    if not is_single_series_bar(result) or not spec.x or not y_col:
        single.warnings.append("Single-bar evidence requires one category axis and one value column.")
        single.evidence_strength = "low"
        return single
    frame = pd.DataFrame(rows)
    if spec.x not in frame.columns or y_col not in frame.columns:
        single.warnings.append("The displayed bar data is missing the category or value column.")
        single.evidence_strength = "low"
        return single
    frame = frame[[spec.x, y_col]].copy()
    frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
    single.excluded_rows = int(frame[y_col].isna().sum())
    frame = frame.dropna(subset=[y_col])
    single.valid_rows = int(len(frame))
    if frame.empty:
        single.warnings.append("No numeric bar values were available.")
        single.evidence_strength = "low"
        return single

    frame["_category_label"] = frame[spec.x].map(lambda value: friendly_value(spec.x, value))
    values = _aggregate_displayed_values(frame, "_category_label", y_col, spec.aggregation)
    single.category_count = int(len(values))
    single.original_category_count = int(len(values))
    single.filters_applied = dict(single.filters)
    if spec.limit and len(frame) >= spec.limit:
        single.top_n_applied = spec.limit
        single.original_category_count = max(single.original_category_count, spec.limit)

    single.category_values = {str(key): float(value) for key, value in values.items()}
    single.displayed_mean = float(values.mean()) if not values.empty else None
    single.displayed_median = float(values.median()) if not values.empty else None
    highest = values.iloc[0]
    lowest = values.iloc[-1]
    single.highest_category = str(values.index[0])
    single.highest_value = float(highest)
    single.lowest_category = str(values.index[-1])
    single.lowest_value = float(lowest)
    if len(values) > 1:
        single.second_highest_category = str(values.index[1])
        single.second_highest_value = float(values.iloc[1])
        second = float(values.iloc[1])
        gap = float(highest) - second
        single.leader_to_second_gap = gap
        single.leader_to_second_gap_percent = gap / abs(second) * 100 if second else None
        single.leader_to_second_gap_basis = single.second_highest_category
        single.lead_strength = _lead_strength(single.leader_to_second_gap_percent)
    else:
        single.lead_strength = "unknown"
    high_low_gap = float(highest) - float(lowest)
    single.highest_to_lowest_gap = high_low_gap
    single.highest_to_lowest_gap_percent = high_low_gap / abs(float(lowest)) * 100 if float(lowest) else None
    single.highest_to_lowest_gap_basis = single.lowest_category

    shares_allowed = (
        (spec.aggregation or "sum") in {"sum", "count"}
        and not values.empty
        and (values >= 0).all()
        and float(values.sum()) > 0
    )
    if shares_allowed:
        total = float(values.sum())
        single.displayed_total = total
        single.highest_share_percent = float(values.iloc[0] / total * 100)
        single.top_two_share_percent = float(values.head(2).sum() / total * 100)
        single.top_three_share_percent = float(values.head(3).sum() / total * 100)
        single.concentration_level = _concentration_level(
            single.top_two_share_percent,
            single.top_three_share_percent,
        )
    else:
        single.displayed_total = float(values.sum()) if not values.empty else None
        single.concentration_level = None

    if single.category_count < 2:
        single.evidence_strength = "low"
    elif single.category_count < 3 or single.top_n_applied or single.excluded_rows:
        single.evidence_strength = "medium"
    else:
        single.evidence_strength = "high"
    single.calculated_metrics.update({
        "category_count": single.category_count,
        "category_values": single.category_values,
        "highest_category": single.highest_category,
        "highest_value": single.highest_value,
        "second_highest_category": single.second_highest_category,
        "second_highest_value": single.second_highest_value,
        "lowest_category": single.lowest_category,
        "lowest_value": single.lowest_value,
        "leader_to_second_gap": single.leader_to_second_gap,
        "leader_to_second_gap_percent": single.leader_to_second_gap_percent,
        "leader_to_second_gap_basis": single.leader_to_second_gap_basis,
        "highest_to_lowest_gap": single.highest_to_lowest_gap,
        "highest_to_lowest_gap_percent": single.highest_to_lowest_gap_percent,
        "highest_to_lowest_gap_basis": single.highest_to_lowest_gap_basis,
        "highest_share_percent": single.highest_share_percent,
        "top_share_pct": single.highest_share_percent,
        "top_two_share_percent": single.top_two_share_percent,
        "top_three_share_percent": single.top_three_share_percent,
        "displayed_total": single.displayed_total,
        "displayed_mean": single.displayed_mean,
        "displayed_median": single.displayed_median,
        "concentration_level": single.concentration_level,
        "lead_strength": single.lead_strength,
    })
    return single


def _pie_concentration_level(largest_share: float | None, top_two_share: float | None, top_three_share: float | None) -> str | None:
    if largest_share is None:
        return None
    if largest_share >= 50 or (top_two_share is not None and top_two_share >= 70):
        return "highly concentrated"
    if largest_share < 30 and (top_three_share is None or top_three_share < 65):
        return "balanced"
    return "moderately concentrated"


def _pie_chart_evidence(result: ChartResult, evidence: ChartEvidence) -> PieChartEvidence:
    rows = _records(result)
    spec = result.spec
    y_col = _metric_column(result)
    pie = PieChartEvidence(
        **evidence.model_dump(exclude={"chart_type", "category_column", "aggregation", "warnings"}),
        chart_type="pie",
        category_column=spec.x,
        value_column=y_col,
        aggregation=spec.aggregation or "sum",
        warnings=list(evidence.warnings),
    )
    if spec.chart_type != "pie" or not spec.x or not y_col:
        pie.warnings.append("Pie evidence requires one category and one displayed value column.")
        pie.evidence_strength = "low"
        return pie
    frame = pd.DataFrame(rows)
    if spec.x not in frame.columns or y_col not in frame.columns:
        pie.warnings.append("The displayed pie data is missing the category or value column.")
        pie.evidence_strength = "low"
        return pie
    frame = frame[[spec.x, y_col]].copy()
    frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
    pie.excluded_rows = int(frame[y_col].isna().sum())
    frame = frame.dropna(subset=[y_col])
    pie.valid_rows = int(len(frame))
    pie.filters_applied = dict(pie.filters)
    if frame.empty:
        pie.warnings.append("No numeric pie slice values were available.")
        pie.evidence_strength = "low"
        return pie

    frame["_category_label"] = frame[spec.x].map(lambda value: friendly_value(spec.x, value))
    values = frame.groupby("_category_label", dropna=False)[y_col].sum().sort_values(ascending=False)
    pie.displayed_category_count = int(len(values))
    pie.original_category_count = int(result.metadata.get("original_category_count") or len(values))
    if spec.limit and pie.original_category_count > pie.displayed_category_count:
        pie.top_n_applied = spec.limit

    pie.values_by_category = {str(key): float(value) for key, value in values.items()}
    pie.displayed_total = float(values.sum()) if not values.empty else None
    aggregation = pie.aggregation or "sum"
    pie.aggregation_additive = aggregation in {"sum", "count", "nunique"}
    has_negative = bool((values < 0).any())
    has_positive = bool((values > 0).any())
    mixed_sign = has_negative and has_positive
    zero_or_negative_total = pie.displayed_total is None or pie.displayed_total <= 0
    pie.part_to_whole_valid = bool(pie.aggregation_additive and not has_negative and not mixed_sign and not zero_or_negative_total)

    pie.largest_category = str(values.index[0])
    pie.largest_value = float(values.iloc[0])
    pie.smallest_category = str(values.index[-1])
    pie.smallest_value = float(values.iloc[-1])
    if len(values) > 1:
        pie.second_category = str(values.index[1])
        pie.second_value = float(values.iloc[1])
        pie.leader_to_second_gap = pie.largest_value - pie.second_value
        pie.leader_to_second_gap_percent = (
            pie.leader_to_second_gap / abs(pie.second_value) * 100
            if pie.second_value else None
        )
        pie.leader_to_second_gap_basis = pie.second_category
        pie.lead_strength = _lead_strength(pie.leader_to_second_gap_percent)
    else:
        pie.lead_strength = "unknown"

    if pie.part_to_whole_valid and pie.displayed_total:
        shares = values / pie.displayed_total * 100
        pie.shares_by_category = {str(key): float(value) for key, value in shares.items()}
        pie.largest_share = float(shares.iloc[0])
        pie.smallest_share = float(shares.iloc[-1])
        if len(shares) > 1:
            pie.second_share = float(shares.iloc[1])
        pie.top_two_share = float(shares.head(2).sum()) if len(shares) >= 2 else None
        pie.top_three_share = float(shares.head(3).sum()) if len(shares) >= 3 else None
        pie.remaining_share = float(max(0.0, 100.0 - (pie.top_three_share or shares.sum())))
        proportions = shares / 100
        pie.herfindahl_index = float((proportions ** 2).sum())
        pie.effective_category_count = float(1 / pie.herfindahl_index) if pie.herfindahl_index else None
        pie.concentration_level = _pie_concentration_level(pie.largest_share, pie.top_two_share, pie.top_three_share)
        pie.small_slice_categories = [
            str(category)
            for category, share in shares.items()
            if float(share) < pie.small_slice_threshold_percent
        ]
        other_matches = [str(category) for category in shares.index if str(category).strip().casefold() == "other"]
        if other_matches:
            pie.other_category_present = True
            pie.other_category_label = other_matches[0]
            pie.other_category_share = float(shares.loc[other_matches[0]])
    else:
        pie.concentration_level = None
        if not pie.aggregation_additive:
            pie.warnings.append("The selected aggregation is not additive, so pie slices do not represent parts of one meaningful total.")
        if has_negative or mixed_sign:
            pie.warnings.append("Pie charts are not suitable when displayed values include negative contributions.")
        if zero_or_negative_total:
            pie.warnings.append("Slice percentages cannot be interpreted because the displayed total is zero or negative.")
    if aggregation == "nunique":
        pie.warnings.append("Unique counts may not be additive if the same entity appears in more than one category.")
    if pie.small_slice_categories:
        pie.warnings.append("Several small slices may be difficult to compare precisely in a Pie chart.")
    if pie.displayed_category_count > 8:
        pie.warnings.append("Many displayed slices can make Pie charts hard to read.")
    if pie.top_n_applied:
        pie.warnings.append(f"The chart shows the top {pie.displayed_category_count} of {pie.original_category_count} categories.")

    if not pie.part_to_whole_valid or pie.displayed_category_count <= 1:
        pie.evidence_strength = "low"
    elif pie.top_n_applied or pie.other_category_present or pie.small_slice_categories:
        pie.evidence_strength = "medium"
    else:
        pie.evidence_strength = "high"
    pie.calculated_metrics.update({
        "largest_category": pie.largest_category,
        "largest_value": pie.largest_value,
        "largest_share": pie.largest_share,
        "second_category": pie.second_category,
        "second_value": pie.second_value,
        "second_share": pie.second_share,
        "smallest_category": pie.smallest_category,
        "smallest_value": pie.smallest_value,
        "smallest_share": pie.smallest_share,
        "top_two_share": pie.top_two_share,
        "top_three_share": pie.top_three_share,
        "leader_to_second_gap": pie.leader_to_second_gap,
        "leader_to_second_gap_percent": pie.leader_to_second_gap_percent,
        "lead_strength": pie.lead_strength,
        "concentration_level": pie.concentration_level,
        "part_to_whole_valid": pie.part_to_whole_valid,
        "top_n_applied": pie.top_n_applied,
    })
    return pie


def _treemap_evidence(result: ChartResult, evidence: ChartEvidence) -> TreemapEvidence:
    rows = _records(result)
    spec = result.spec
    y_col = _metric_column(result)
    tree = TreemapEvidence(
        **evidence.model_dump(exclude={"chart_type", "category_column", "group_column", "aggregation", "warnings"}),
        chart_type="treemap",
        category_column=spec.x,
        group_column=spec.color,
        value_column=y_col,
        aggregation=spec.aggregation or "sum",
        warnings=list(evidence.warnings),
    )
    if spec.chart_type != "treemap" or not spec.x or not y_col:
        tree.warnings.append("Treemap evidence requires one category and one displayed value column.")
        tree.evidence_strength = "low"
        return tree
    frame = pd.DataFrame(rows)
    if spec.x not in frame.columns or y_col not in frame.columns:
        tree.warnings.append("The displayed Treemap data is missing the category or value column.")
        tree.evidence_strength = "low"
        return tree
    columns = [spec.x, y_col] + ([spec.color] if spec.color and spec.color in frame.columns else [])
    frame = frame[columns].copy()
    frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
    tree.excluded_rows = int(frame[y_col].isna().sum())
    frame = frame.dropna(subset=[y_col])
    tree.valid_rows = int(len(frame))
    tree.filters_applied = dict(tree.filters)
    if frame.empty:
        tree.warnings.append("No numeric Treemap values were available.")
        tree.evidence_strength = "low"
        return tree

    frame["_category_label"] = frame[spec.x].map(lambda value: friendly_value(spec.x, value))
    values = frame.groupby("_category_label", dropna=False)[y_col].sum().sort_values(ascending=False)
    tree.displayed_category_count = int(len(values))
    tree.original_category_count = int(result.metadata.get("original_category_count") or len(values))
    if spec.limit and tree.original_category_count > tree.displayed_category_count:
        tree.top_n_applied = spec.limit

    tree.values_by_category = {str(key): float(value) for key, value in values.items()}
    tree.displayed_total = float(values.sum())
    aggregation = tree.aggregation or "sum"
    has_negative = bool((values < 0).any())
    tree.part_to_whole_valid = bool(aggregation in {"sum", "count", "nunique"} and not has_negative and tree.displayed_total > 0)
    tree.largest_category = str(values.index[0])
    tree.largest_value = float(values.iloc[0])
    tree.smallest_category = str(values.index[-1])
    tree.smallest_value = float(values.iloc[-1])
    if len(values) > 1:
        tree.second_category = str(values.index[1])
        tree.second_value = float(values.iloc[1])
        tree.leader_to_second_gap = tree.largest_value - tree.second_value
        tree.leader_to_second_gap_percent = tree.leader_to_second_gap / abs(tree.second_value) * 100 if tree.second_value else None
        tree.lead_strength = _lead_strength(tree.leader_to_second_gap_percent)
    else:
        tree.lead_strength = "unknown"

    if tree.part_to_whole_valid and tree.displayed_total:
        shares = values / tree.displayed_total * 100
        tree.shares_by_category = {str(key): float(value) for key, value in shares.items()}
        tree.largest_share = float(shares.iloc[0])
        tree.smallest_share = float(shares.iloc[-1])
        if len(shares) > 1:
            tree.second_share = float(shares.iloc[1])
        tree.top_two_share = float(shares.head(2).sum()) if len(shares) >= 2 else None
        tree.top_three_share = float(shares.head(3).sum()) if len(shares) >= 3 else None
        tree.remaining_share = float(max(0.0, 100.0 - (tree.top_three_share or shares.sum())))
        tree.concentration_level = _pie_concentration_level(tree.largest_share, tree.top_two_share, tree.top_three_share)
        tree.small_rectangle_categories = [
            str(category)
            for category, share in shares.items()
            if float(share) < tree.small_rectangle_threshold_percent
        ]
        if spec.color and spec.color in frame.columns:
            frame["_group_label"] = frame[spec.color].map(lambda value: friendly_value(spec.color, value))
            group_totals = frame.groupby("_group_label", dropna=False)[y_col].sum().sort_values(ascending=False)
            group_shares = group_totals / tree.displayed_total * 100
            tree.group_totals = {str(key): float(value) for key, value in group_totals.items()}
            tree.group_shares = {str(key): float(value) for key, value in group_shares.items()}
            if not group_totals.empty:
                tree.largest_group = str(group_totals.index[0])
                tree.largest_group_value = float(group_totals.iloc[0])
                tree.largest_group_share = float(group_shares.iloc[0])
    else:
        if aggregation not in {"sum", "count", "nunique"}:
            tree.warnings.append("The selected aggregation is not additive, so Treemap areas do not represent parts of one meaningful total.")
        if has_negative:
            tree.warnings.append("Treemaps are not suitable when displayed values include negative contributions.")
        if tree.displayed_total <= 0:
            tree.warnings.append("Area shares cannot be interpreted because the displayed total is zero or negative.")
    if aggregation == "nunique":
        tree.warnings.append("Unique counts may not be additive if the same entity appears in more than one category.")
    if tree.small_rectangle_categories:
        tree.warnings.append("Very small rectangles may be difficult to read or compare precisely.")
    if tree.displayed_category_count > 12:
        tree.warnings.append("Many displayed rectangles can make the Treemap hard to read.")
    if tree.top_n_applied:
        tree.warnings.append(f"The chart shows the top {tree.displayed_category_count} of {tree.original_category_count} categories.")

    if not tree.part_to_whole_valid or tree.displayed_category_count <= 1:
        tree.evidence_strength = "low"
    elif tree.top_n_applied or tree.small_rectangle_categories:
        tree.evidence_strength = "medium"
    else:
        tree.evidence_strength = "high"
    tree.calculated_metrics.update({
        "largest_category": tree.largest_category,
        "largest_value": tree.largest_value,
        "largest_share": tree.largest_share,
        "second_category": tree.second_category,
        "second_value": tree.second_value,
        "second_share": tree.second_share,
        "top_two_share": tree.top_two_share,
        "top_three_share": tree.top_three_share,
        "largest_group": tree.largest_group,
        "largest_group_share": tree.largest_group_share,
        "concentration_level": tree.concentration_level,
        "top_n_applied": tree.top_n_applied,
    })
    return tree


def _normalized_geo_label(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum() or character == "-")


def _location_type(column: str | None) -> str:
    normalized = normalize_column_name(column)
    if "country" in normalized:
        return "country"
    if "city" in normalized:
        return "city"
    if "state" in normalized:
        return "state"
    if "province" in normalized:
        return "province"
    if "region" in normalized:
        return "region"
    if "postal" in normalized or "zip" in normalized:
        return "postal code"
    return "geographic label"


def _single_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    values = dataframe.loc[:, column]
    if isinstance(values, pd.DataFrame):
        return values.iloc[:, 0]
    return values


def _symbol_map_evidence(result: ChartResult, evidence: ChartEvidence) -> SymbolMapEvidence:
    rows = _records(result)
    spec = result.spec
    y_col = _metric_column(result)
    color_col = effective_symbol_map_color(spec.x, spec.color)
    symbol = SymbolMapEvidence(
        **evidence.model_dump(exclude={
            "chart_type",
            "category_column",
            "group_column",
            "color_column",
            "aggregation",
            "warnings",
        }),
        chart_type="symbol_map",
        location_column=spec.x,
        location_type=_location_type(spec.x),
        value_column=y_col,
        aggregation=spec.aggregation or "sum",
        color_column=color_col,
        warnings=list(evidence.warnings),
        filters_applied=dict(evidence.filters),
        raw_row_count=int(result.metadata.get("filtered_row_count") or len(rows)),
        original_location_count=int(result.metadata.get("original_category_count") or 0) or None,
        date_column=spec.time_column,
        start_date=spec.date_range_start,
        end_date=spec.date_range_end,
    )
    if spec.chart_type != "symbol_map" or not spec.x or not y_col:
        symbol.warnings.append("Symbol Map evidence requires one location and one displayed value column.")
        symbol.evidence_strength = "low"
        return symbol
    frame = pd.DataFrame(rows)
    if spec.x not in frame.columns or y_col not in frame.columns:
        symbol.warnings.append("The displayed Symbol Map data is missing the location or value column.")
        symbol.evidence_strength = "low"
        return symbol
    real_color_col = color_col if color_col and color_col != SYMBOL_MAP_COLOR_LOCATION else None
    columns = list(dict.fromkeys(
        column
        for column in [spec.x, y_col, real_color_col]
        if column and column in frame.columns
    ))
    frame = frame[columns].copy()
    frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
    frame = frame.dropna(subset=[spec.x, y_col])
    if frame.empty:
        symbol.warnings.append("No resolved numeric Symbol Map values were available.")
        symbol.evidence_strength = "low"
        return symbol

    frame["_location_label"] = _single_series(frame, spec.x).map(lambda value: friendly_value(spec.x, value))
    if real_color_col and real_color_col in frame:
        frame["_group_label"] = _single_series(frame, real_color_col).map(lambda value: friendly_value(real_color_col, value))
        ambiguous = (
            frame.groupby("_location_label")["_group_label"].nunique(dropna=False)
            .loc[lambda values: values > 1]
        )
        if not ambiguous.empty:
            symbol.warnings.append("Some displayed locations appear in multiple color groups, so group assignment is ambiguous.")
    location_values = frame.groupby("_location_label", dropna=False)[y_col].sum().sort_values(ascending=False)
    symbol.displayed_location_count = int(len(location_values))
    symbol.aggregated_location_count = symbol.displayed_location_count
    if symbol.original_location_count and spec.limit and symbol.original_location_count > symbol.displayed_location_count:
        symbol.top_n_applied = spec.limit

    additive = (symbol.aggregation or "sum") in {"sum", "count"}
    has_negative = bool((location_values < 0).any())
    total = float(location_values.sum())
    shares_valid = bool(additive and total > 0 and not has_negative)
    symbol.displayed_total = total
    shares = location_values / total * 100 if shares_valid else pd.Series(index=location_values.index, dtype="float64")
    if not shares_valid:
        if not additive:
            symbol.warnings.append("The selected aggregation is not additive, so location shares should not be interpreted as parts of one total.")
        if has_negative:
            symbol.warnings.append("Location contribution shares are not meaningful when values include negative amounts.")
        if total <= 0:
            symbol.warnings.append("Location contribution shares cannot be interpreted because the displayed total is zero or negative.")

    locations = []
    group_by_location = {}
    if color_col and "_group_label" in frame:
        group_by_location = (
            frame.sort_values(y_col, ascending=False)
            .drop_duplicates("_location_label")
            .set_index("_location_label")["_group_label"]
            .to_dict()
        )
    for rank, (location, value) in enumerate(location_values.items(), start=1):
        normalized = _normalized_geo_label(location)
        coords = REGION_CENTROIDS.get(normalized)
        locations.append(SymbolMapLocationEvidence(
            location=str(location),
            latitude=float(coords[0]) if coords else None,
            longitude=float(coords[1]) if coords else None,
            aggregated_value=float(value),
            share_of_displayed_total=float(shares.loc[location]) if shares_valid and location in shares else None,
            rank=rank,
            color_group=str(group_by_location.get(location)) if group_by_location.get(location) is not None else None,
            valid_coordinates=True,
            coordinate_source="standard region centroid" if coords else "plotly location name",
        ))
    symbol.locations = locations
    first = locations[0]
    last = locations[-1]
    symbol.largest_location = first.location
    symbol.largest_value = first.aggregated_value
    symbol.largest_share = first.share_of_displayed_total
    symbol.largest_group = first.color_group
    symbol.smallest_location = last.location
    symbol.smallest_value = last.aggregated_value
    symbol.smallest_share = last.share_of_displayed_total
    if len(locations) > 1:
        second = locations[1]
        symbol.second_location = second.location
        symbol.second_value = second.aggregated_value
        symbol.second_share = second.share_of_displayed_total
        symbol.leader_to_second_gap = symbol.largest_value - symbol.second_value
        if symbol.largest_share is not None and symbol.second_share is not None:
            symbol.leader_to_second_gap_percentage_points = symbol.largest_share - symbol.second_share
            symbol.lead_strength = _share_lead_strength(symbol.leader_to_second_gap_percentage_points)
    if len(locations) > 2:
        third = locations[2]
        symbol.third_location = third.location
        symbol.third_value = third.aggregated_value
        symbol.third_share = third.share_of_displayed_total
    if shares_valid:
        symbol.top_two_share = float(shares.head(2).sum()) if len(shares) >= 2 else None
        symbol.top_three_share = float(shares.head(3).sum()) if len(shares) >= 3 else None
        symbol.top_five_share = float(shares.head(5).sum()) if len(shares) >= 5 else None
        proportions = shares / 100
        symbol.herfindahl_index = float((proportions ** 2).sum())
        symbol.effective_location_count = float(1 / symbol.herfindahl_index) if symbol.herfindahl_index else None
        if symbol.largest_share is not None and symbol.largest_share < 5 and (symbol.top_five_share or 0) < 25:
            symbol.concentration_level = "widely dispersed"
        elif (symbol.top_five_share or 0) >= 50 or (symbol.largest_share or 0) >= 25:
            symbol.concentration_level = "highly concentrated"
        else:
            symbol.concentration_level = "moderately concentrated"

    if color_col and "_group_label" in frame:
        top_threshold = min(10, max(1, int(np.ceil(symbol.displayed_location_count * 0.1))))
        top_locations = set(location_values.head(top_threshold).index)
        group_rows = []
        for group_name, group in frame.groupby("_group_label", dropna=False):
            group_location_values = group.groupby("_location_label", dropna=False)[y_col].sum().sort_values(ascending=False)
            group_total = float(group_location_values.sum())
            largest_location = str(group_location_values.index[0])
            group_rows.append(SymbolMapGroupEvidence(
                group_name=str(group_name),
                location_count=int(len(group_location_values)),
                total_value=group_total,
                share_of_displayed_total=group_total / total * 100 if shares_valid else None,
                mean_location_value=float(group_location_values.mean()),
                median_location_value=float(group_location_values.median()),
                largest_location=largest_location,
                largest_location_value=float(group_location_values.iloc[0]),
                top_location_count=int(sum(location in top_locations for location in group_location_values.index)),
            ))
        symbol.color_groups = sorted(group_rows, key=lambda item: item.total_value, reverse=True)
        if symbol.color_groups:
            highest_total = symbol.color_groups[0]
            symbol.highest_total_group = highest_total.group_name
            symbol.highest_total_group_value = highest_total.total_value
            symbol.highest_total_group_share = highest_total.share_of_displayed_total
            highest_median = max(symbol.color_groups, key=lambda item: item.median_location_value or float("-inf"))
            symbol.highest_median_group = highest_median.group_name
            symbol.highest_median_group_value = highest_median.median_location_value
            most_top = max(symbol.color_groups, key=lambda item: item.top_location_count)
            symbol.group_with_most_top_locations = most_top.group_name
            symbol.group_with_most_top_locations_count = most_top.top_location_count
            if most_top.top_location_count >= max(2, top_threshold // 2):
                symbol.geographic_distribution = f"many top locations are concentrated in {most_top.group_name}"
            else:
                symbol.geographic_distribution = "top locations are distributed across several color groups"
    if not symbol.geographic_distribution:
        symbol.geographic_distribution = "displayed locations are compared geographically without a selected color group"

    symbol.spatial_concentration_level = symbol.concentration_level
    if symbol.displayed_location_count >= 150:
        symbol.marker_overlap_level = "high"
    elif symbol.displayed_location_count >= 50:
        symbol.marker_overlap_level = "moderate"
    else:
        symbol.marker_overlap_level = "low"
    if symbol.marker_overlap_level in {"moderate", "high"}:
        symbol.warnings.append("Markers may overlap in dense geographic areas, making smaller locations harder to see.")
    if symbol.top_n_applied:
        symbol.warnings.append(f"The map displays the top {symbol.displayed_location_count} of {symbol.original_location_count} locations.")
    if symbol.date_column:
        symbol.warnings.append("The map values reflect the active date-filter configuration.")
    if symbol.displayed_location_count < 8:
        symbol.evidence_strength = "low"
    elif symbol.displayed_location_count < 20 or symbol.top_n_applied or symbol.marker_overlap_level == "moderate" or not shares_valid:
        symbol.evidence_strength = "medium"
    else:
        symbol.evidence_strength = "high"
    symbol.calculated_metrics.update({
        "largest_location": symbol.largest_location,
        "largest_value": symbol.largest_value,
        "largest_share": symbol.largest_share,
        "top_five_share": symbol.top_five_share,
        "highest_total_group": symbol.highest_total_group,
        "marker_overlap_level": symbol.marker_overlap_level,
    })
    return symbol


def _share_lead_strength(gap_points: float | None) -> str:
    if gap_points is None:
        return "unknown"
    if gap_points < 1:
        return "narrow"
    if gap_points < 5:
        return "moderate"
    return "clear"


def _sorted_percentage_bar_evidence(result: ChartResult, evidence: ChartEvidence) -> SortedPercentageBarEvidence:
    rows = _records(result)
    spec = result.spec
    pct = SortedPercentageBarEvidence(
        **evidence.model_dump(exclude={"chart_type", "category_column", "aggregation", "warnings"}),
        chart_type="sorted_percentage_bar",
        category_column=spec.x,
        value_column=spec.y,
        aggregation=spec.aggregation or "sum",
        percentage_denominator_mode=spec.percentage_denominator_mode,
        warnings=list(evidence.warnings),
    )
    if spec.chart_type != "sorted_percentage_bar" or not spec.x or not spec.y:
        pct.warnings.append("Sorted Percentage Bar evidence requires one category and one numeric metric.")
        pct.evidence_strength = "low"
        return pct
    frame = pd.DataFrame(rows)
    required = {spec.x, "aggregated_value", "percentage_share"}
    if not required.issubset(frame.columns):
        pct.warnings.append("The displayed percentage-bar data is missing required columns.")
        pct.evidence_strength = "low"
        return pct
    frame = frame[[spec.x, "aggregated_value", "percentage_share"]].copy()
    frame["aggregated_value"] = pd.to_numeric(frame["aggregated_value"], errors="coerce")
    frame["percentage_share"] = pd.to_numeric(frame["percentage_share"], errors="coerce")
    frame = frame.dropna(subset=["aggregated_value"])
    pct.valid_rows = int(len(frame))
    pct.filters_applied = dict(pct.filters)
    pct.original_category_count = int(result.metadata.get("original_category_count") or len(frame))
    pct.displayed_category_count = int(len(frame))
    pct.displayed_total = float(result.metadata.get("displayed_total") or frame["aggregated_value"].sum())
    pct.full_filtered_total = float(result.metadata.get("full_filtered_total") or pct.displayed_total or 0)
    pct.percentage_valid = bool(result.metadata.get("percentage_valid", frame["percentage_share"].notna().all()))
    pct.negative_value_count = int(result.metadata.get("negative_value_count") or 0)
    pct.other_category_present = bool(result.metadata.get("other_category_present") or frame[spec.x].astype(str).str.casefold().eq("other").any())
    pct.additive_aggregation = (pct.aggregation or "sum") in {"sum", "count"}
    pct.date_column = result.metadata.get("date_column") or spec.time_column
    pct.start_date = result.metadata.get("date_range_start") or spec.date_range_start
    pct.end_date = result.metadata.get("date_range_end") or spec.date_range_end
    if spec.limit and pct.original_category_count > pct.displayed_category_count - int(pct.other_category_present):
        pct.top_n_applied = spec.limit

    frame["_category_label"] = frame[spec.x].map(lambda value: friendly_value(spec.x, value))
    pct.aggregated_values = {
        str(row["_category_label"]): float(row["aggregated_value"])
        for _, row in frame.iterrows()
    }
    valid_shares = frame.dropna(subset=["percentage_share"]).copy()
    pct.percentage_shares = {
        str(row["_category_label"]): float(row["percentage_share"])
        for _, row in valid_shares.iterrows()
    }
    pct.category_ranking = [str(value) for value in frame["_category_label"].tolist()]
    if not frame.empty:
        first = frame.iloc[0]
        last = frame.iloc[-1]
        pct.largest_category = str(first["_category_label"])
        pct.largest_value = float(first["aggregated_value"])
        pct.largest_share = float(first["percentage_share"]) if pd.notna(first["percentage_share"]) else None
        pct.smallest_category = str(last["_category_label"])
        pct.smallest_value = float(last["aggregated_value"])
        pct.smallest_share = float(last["percentage_share"]) if pd.notna(last["percentage_share"]) else None
    if len(frame) > 1:
        second = frame.iloc[1]
        pct.second_category = str(second["_category_label"])
        pct.second_value = float(second["aggregated_value"])
        pct.second_share = float(second["percentage_share"]) if pd.notna(second["percentage_share"]) else None
    if len(frame) > 2:
        third = frame.iloc[2]
        pct.third_category = str(third["_category_label"])
        pct.third_value = float(third["aggregated_value"])
        pct.third_share = float(third["percentage_share"]) if pd.notna(third["percentage_share"]) else None
    if pct.largest_share is not None and pct.second_share is not None:
        pct.leader_to_second_gap_percentage_points = pct.largest_share - pct.second_share
        pct.leader_to_second_relative_gap_percent = (
            pct.leader_to_second_gap_percentage_points / abs(pct.second_share) * 100
            if pct.second_share else None
        )
        pct.lead_strength = _share_lead_strength(pct.leader_to_second_gap_percentage_points)
    else:
        pct.lead_strength = "unknown"
    if pct.percentage_valid and not valid_shares.empty:
        shares = valid_shares["percentage_share"]
        pct.top_two_share = float(shares.head(2).sum()) if len(shares) >= 2 else None
        pct.top_three_share = float(shares.head(3).sum()) if len(shares) >= 3 else None
        pct.remaining_share = float(max(0.0, 100.0 - (pct.top_three_share or shares.sum())))
        proportions = shares / 100
        pct.herfindahl_index = float((proportions ** 2).sum())
        pct.effective_category_count = float(1 / pct.herfindahl_index) if pct.herfindahl_index else None
        pct.concentration_level = _pie_concentration_level(pct.largest_share, pct.top_two_share, pct.top_three_share)
        pct.small_share_categories = [
            str(row["_category_label"])
            for _, row in valid_shares.iterrows()
            if float(row["percentage_share"]) < 3
        ]
        if pct.other_category_present and "Other" in pct.percentage_shares:
            pct.other_share = pct.percentage_shares["Other"]

    if not pct.additive_aggregation:
        pct.warnings.append("The selected aggregation is not additive, so percentage shares are shares of aggregated displayed values, not true contributions.")
    if pct.negative_value_count:
        pct.warnings.append("Percentage-of-total contribution is not valid because aggregated values include negative amounts.")
    if pct.full_filtered_total is not None and pct.full_filtered_total <= 0:
        pct.warnings.append("Percentage shares cannot be interpreted because the denominator is zero or negative.")
    if pct.top_n_applied:
        pct.warnings.append("Top-N is active; excluded categories remain part of the full filtered denominator unless shown as Other.")
    if pct.small_share_categories:
        pct.warnings.append("Several categories contribute very small shares and may be easier to compare with labels.")

    if not pct.percentage_valid or not pct.additive_aggregation or pct.negative_value_count or pct.displayed_category_count <= 1:
        pct.evidence_strength = "low"
    elif pct.top_n_applied or pct.other_category_present or pct.small_share_categories:
        pct.evidence_strength = "medium"
    else:
        pct.evidence_strength = "high"
    pct.calculated_metrics.update({
        "largest_category": pct.largest_category,
        "largest_share": pct.largest_share,
        "second_category": pct.second_category,
        "second_share": pct.second_share,
        "top_two_share": pct.top_two_share,
        "top_three_share": pct.top_three_share,
        "lead_strength": pct.lead_strength,
        "percentage_valid": pct.percentage_valid,
    })
    return pct


def _period_label(value: Any, granularity: str | None) -> str:
    return format_period(value, granularity) if granularity else _fmt(value)


def _period_over_period_evidence(result: ChartResult, evidence: ChartEvidence) -> PeriodOverPeriodChangeEvidence:
    rows = _records(result)
    spec = result.spec
    pop = PeriodOverPeriodChangeEvidence(
        **evidence.model_dump(exclude={"chart_type", "aggregation", "warnings"}),
        chart_type="period_over_period_change",
        date_column=spec.x,
        value_column=spec.y,
        aggregation=spec.aggregation or "sum",
        granularity=spec.time_grain or result.metadata.get("granularity") or "month",
        comparison_basis=spec.comparison_basis,
        warnings=list(evidence.warnings),
        filters_applied=dict(evidence.filters),
        start_date=result.metadata.get("display_start") or spec.date_range_start,
        end_date=result.metadata.get("display_end") or spec.date_range_end,
        calculation_start=result.metadata.get("calculation_start"),
        missing_period_count=int(result.metadata.get("missing_period_count") or 0),
    )
    if spec.chart_type != "period_over_period_change" or not spec.x or not spec.y:
        pop.warnings.append("Period-over-Period evidence requires a date field and metric.")
        pop.evidence_strength = "low"
        return pop
    frame = pd.DataFrame(rows)
    required = {spec.x, spec.y, "comparison_value", "absolute_change", "percentage_change"}
    if frame.empty or not required.issubset(frame.columns):
        pop.warnings.append("The displayed period-over-period data is missing required fields.")
        pop.evidence_strength = "low"
        return pop
    frame[spec.x] = pd.to_datetime(frame[spec.x], errors="coerce")
    for column in (spec.y, "comparison_value", "absolute_change", "percentage_change"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[spec.x]).sort_values(spec.x)
    pop.period_count = int(len(frame))
    valid = frame.dropna(subset=["percentage_change"]).copy()
    pop.comparable_period_count = int(len(valid))
    pop.unavailable_period_count = pop.period_count - pop.comparable_period_count
    pop.zero_baseline_count = int((frame["comparison_value"] == 0).sum())
    pop.periods = [_period_label(value, pop.granularity) for value in frame[spec.x]]
    pop.current_values = {
        _period_label(row[spec.x], pop.granularity): float(row[spec.y])
        for _, row in frame.dropna(subset=[spec.y]).iterrows()
    }
    pop.comparison_values = {
        _period_label(row[spec.x], pop.granularity): float(row["comparison_value"])
        for _, row in frame.dropna(subset=["comparison_value"]).iterrows()
    }
    pop.absolute_changes = {
        _period_label(row[spec.x], pop.granularity): float(row["absolute_change"])
        for _, row in frame.dropna(subset=["absolute_change"]).iterrows()
    }
    pop.percentage_changes = {
        _period_label(row[spec.x], pop.granularity): float(row["percentage_change"])
        for _, row in valid.iterrows()
    }
    if not valid.empty:
        latest = valid.iloc[-1]
        pop.latest_period = _period_label(latest[spec.x], pop.granularity)
        pop.latest_value = float(latest[spec.y])
        pop.latest_comparison_value = float(latest["comparison_value"])
        pop.latest_percent_change = float(latest["percentage_change"])
        pop.latest_absolute_change = float(latest["absolute_change"])
        increase = valid.loc[valid["percentage_change"].idxmax()]
        decline = valid.loc[valid["percentage_change"].idxmin()]
        pop.largest_increase_period = _period_label(increase[spec.x], pop.granularity)
        pop.largest_increase_percent = float(increase["percentage_change"])
        pop.largest_decline_period = _period_label(decline[spec.x], pop.granularity)
        pop.largest_decline_percent = float(decline["percentage_change"])
        pop.increase_count = int((valid["percentage_change"] > 0).sum())
        pop.decline_count = int((valid["percentage_change"] < 0).sum())
        pop.no_change_count = int((valid["percentage_change"] == 0).sum())
        pop.average_percent_change = float(valid["percentage_change"].mean())
        pop.median_percent_change = float(valid["percentage_change"].median())
        spread = float(valid["percentage_change"].std(ddof=0)) if len(valid) > 1 else 0.0
        pop.volatility_level = "high" if spread >= 25 else "medium" if spread >= 10 else "low"
    if pop.unavailable_period_count:
        pop.warnings.append("Some displayed periods do not have a comparable baseline period.")
    if pop.zero_baseline_count:
        pop.warnings.append("Percentage change is unavailable when the comparison baseline is zero.")
    if pop.comparable_period_count < 2:
        pop.evidence_strength = "low"
    elif pop.unavailable_period_count or pop.comparable_period_count < 6:
        pop.evidence_strength = "medium"
    else:
        pop.evidence_strength = "high"
    pop.calculated_metrics.update({
        "latest_percent_change": pop.latest_percent_change,
        "largest_increase_percent": pop.largest_increase_percent,
        "largest_decline_percent": pop.largest_decline_percent,
        "increase_count": pop.increase_count,
        "decline_count": pop.decline_count,
    })
    return pop


def _stacked_bar_evidence(result: ChartResult, evidence: ChartEvidence) -> StackedBarEvidence:
    rows = _records(result)
    spec = result.spec
    y_col = _metric_column(result)
    stacked = StackedBarEvidence(**evidence.model_dump(), value_column=y_col)
    if spec.chart_type != "stacked_bar" or not spec.x or not spec.color or not y_col:
        stacked.warnings.append("Stacked-bar evidence requires a category, stack, and numeric value.")
        stacked.evidence_strength = "low"
        return stacked
    frame = pd.DataFrame(rows)
    required = {spec.x, spec.color, y_col}
    if not required.issubset(frame.columns):
        stacked.warnings.append("The displayed stacked-bar data is missing required columns.")
        stacked.evidence_strength = "low"
        return stacked
    frame = frame[[spec.x, spec.color, y_col]].copy()
    frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
    stacked.excluded_rows = int(frame[y_col].isna().sum())
    frame = frame.dropna(subset=[y_col])
    stacked.valid_rows = int(len(frame))
    if frame.empty:
        stacked.warnings.append("No numeric stacked-bar values were available.")
        stacked.evidence_strength = "low"
        return stacked

    frame["_category_label"] = frame[spec.x].map(lambda value: friendly_value(spec.x, value))
    frame["_stack_label"] = frame[spec.color].map(lambda value: friendly_value(spec.color, value))
    category_totals = (
        frame.groupby("_category_label", dropna=False)[y_col]
        .sum()
        .sort_values(ascending=False)
    )
    stack_totals = (
        frame.groupby("_stack_label", dropna=False)[y_col]
        .sum()
        .sort_values(ascending=False)
    )
    pivot = frame.pivot_table(
        index="_category_label",
        columns="_stack_label",
        values=y_col,
        aggfunc="sum",
        fill_value=0,
    )
    stacked.category_count = int(len(category_totals))
    stacked.stack_count = int(len(stack_totals))
    stacked.category_totals = {str(key): float(value) for key, value in category_totals.items()}
    stacked.stack_totals = {str(key): float(value) for key, value in stack_totals.items()}
    stacked.values_by_category_and_stack = {
        str(category): {str(stack): float(value) for stack, value in values.items()}
        for category, values in pivot.to_dict(orient="index").items()
    }
    stacked.highest_combined_category = str(category_totals.idxmax())
    stacked.highest_combined_value = float(category_totals.max())
    stacked.lowest_combined_category = str(category_totals.idxmin())
    stacked.lowest_combined_value = float(category_totals.min())
    stacked.strongest_stack = str(stack_totals.idxmax())
    stacked.strongest_stack_value = float(stack_totals.max())
    stacked.filters_applied = dict(stacked.filters)
    if spec.limit and len(category_totals) >= spec.limit:
        stacked.top_n_applied = spec.limit

    for category, values in pivot.iterrows():
        total = float(values.sum())
        dominant_stack = str(values.idxmax())
        dominant_value = float(values.max())
        category_label = str(category)
        stacked.dominant_stack_by_category[category_label] = dominant_stack
        stacked.dominant_stack_value_by_category[category_label] = dominant_value
        stacked.dominant_stack_share_by_category[category_label] = (
            dominant_value / total * 100 if total else None
        )
    top_category = stacked.highest_combined_category
    if top_category:
        stacked.highest_category_dominant_stack = stacked.dominant_stack_by_category.get(top_category)
        stacked.highest_category_dominant_value = stacked.dominant_stack_value_by_category.get(top_category)
        stacked.highest_category_dominant_share = stacked.dominant_stack_share_by_category.get(top_category)

    stacked.evidence_strength = "high" if stacked.category_count >= 3 and not stacked.excluded_rows else "medium"
    if stacked.category_count < 2 or stacked.stack_count < 2:
        stacked.evidence_strength = "low"
    stacked.calculated_metrics.update({
        "category_count": stacked.category_count,
        "stack_count": stacked.stack_count,
        "category_totals": stacked.category_totals,
        "stack_totals": stacked.stack_totals,
        "highest_combined_category": stacked.highest_combined_category,
        "highest_combined_value": stacked.highest_combined_value,
        "lowest_combined_category": stacked.lowest_combined_category,
        "lowest_combined_value": stacked.lowest_combined_value,
        "strongest_stack": stacked.strongest_stack,
        "strongest_stack_value": stacked.strongest_stack_value,
        "dominant_stack_by_category": stacked.dominant_stack_by_category,
        "highest_category_dominant_stack": stacked.highest_category_dominant_stack,
        "highest_category_dominant_value": stacked.highest_category_dominant_value,
        "highest_category_dominant_share": stacked.highest_category_dominant_share,
    })
    return stacked


def _grouped_bar_evidence(result: ChartResult, evidence: ChartEvidence) -> GroupedBarEvidence:
    rows = _records(result)
    spec = result.spec
    y_col = _metric_column(result)
    grouped = GroupedBarEvidence(**evidence.model_dump(), value_column=y_col)
    if not rows or not spec.x or not spec.color or not y_col:
        grouped.warnings.append("Grouped bar evidence requires a category, group, and numeric value.")
        grouped.evidence_strength = "low"
        return grouped

    frame = pd.DataFrame(rows)
    required = {spec.x, spec.color, y_col}
    if not required.issubset(frame.columns):
        grouped.warnings.append("The displayed grouped-bar data is missing required columns.")
        grouped.evidence_strength = "low"
        return grouped

    frame = frame[[spec.x, spec.color, y_col]].copy()
    frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
    invalid_rows = int(frame[y_col].isna().sum())
    frame = frame.dropna(subset=[y_col])
    grouped.excluded_rows = invalid_rows
    grouped.valid_rows = int(len(frame))
    if frame.empty:
        grouped.warnings.append("No numeric grouped-bar values were available.")
        grouped.evidence_strength = "low"
        return grouped

    frame["_category_label"] = frame[spec.x].map(lambda value: friendly_value(spec.x, value))
    frame["_group_label"] = frame[spec.color].map(lambda value: friendly_value(spec.color, value))
    grouped.category_count = int(frame["_category_label"].nunique(dropna=False))
    grouped.group_count = int(frame["_group_label"].nunique(dropna=False))
    grouped.filters_applied = dict(grouped.filters)

    category_totals = (
        frame.groupby("_category_label", dropna=False)[y_col]
        .sum()
        .sort_values(ascending=False)
    )
    group_totals = (
        frame.groupby("_group_label", dropna=False)[y_col]
        .sum()
        .sort_values(ascending=False)
    )
    pivot = frame.pivot_table(
        index="_category_label",
        columns="_group_label",
        values=y_col,
        aggfunc="sum",
        fill_value=0,
    )
    observed_pairs = set(zip(frame["_category_label"], frame["_group_label"], strict=False))
    missing_pairs = [
        {"category": str(category), "group": str(group)}
        for category in pivot.index
        for group in pivot.columns
        if (category, group) not in observed_pairs
    ]

    grouped.category_totals = {str(key): float(value) for key, value in category_totals.items()}
    grouped.group_totals = {str(key): float(value) for key, value in group_totals.items()}
    grouped.values_by_category_and_group = {
        str(category): {str(group): float(value) for group, value in values.items()}
        for category, values in pivot.to_dict(orient="index").items()
    }
    grouped.missing_group_combinations = missing_pairs
    grouped.highest_combined_category = str(category_totals.idxmax())
    grouped.highest_combined_value = float(category_totals.max())
    grouped.lowest_combined_category = str(category_totals.idxmin())
    grouped.lowest_combined_value = float(category_totals.min())

    individual_idx_max = frame[y_col].idxmax()
    individual_idx_min = frame[y_col].idxmin()
    highest_individual = frame.loc[individual_idx_max]
    lowest_individual = frame.loc[individual_idx_min]
    grouped.highest_individual_category = str(highest_individual["_category_label"])
    grouped.highest_individual_group = str(highest_individual["_group_label"])
    grouped.highest_individual_value = float(highest_individual[y_col])
    grouped.lowest_individual_category = str(lowest_individual["_category_label"])
    grouped.lowest_individual_group = str(lowest_individual["_group_label"])
    grouped.lowest_individual_value = float(lowest_individual[y_col])

    gap_records: list[dict[str, Any]] = []
    for category, values in pivot.iterrows():
        sorted_values = values.sort_values(ascending=False)
        winner_group = str(sorted_values.index[0])
        winner_value = float(sorted_values.iloc[0])
        second_group = str(sorted_values.index[1]) if len(sorted_values) > 1 else ""
        second_value = float(sorted_values.iloc[1]) if len(sorted_values) > 1 else 0.0
        gap = winner_value - second_value
        gap_percent = gap / abs(second_value) * 100 if second_value else None
        category_label = str(category)
        grouped.winner_by_category[category_label] = winner_group
        grouped.winner_value_by_category[category_label] = winner_value
        grouped.winner_gap_by_category[category_label] = float(gap)
        grouped.winner_gap_percent_by_category[category_label] = (
            float(gap_percent) if gap_percent is not None else None
        )
        gap_records.append({
            "category": category_label,
            "winner": winner_group,
            "runner_up": second_group,
            "gap": float(gap),
            "gap_percent": float(gap_percent) if gap_percent is not None else None,
        })
    grouped.group_win_counts = {
        str(group): int(count)
        for group, count in pd.Series(grouped.winner_by_category).value_counts().items()
    }
    if gap_records:
        largest_gap = max(gap_records, key=lambda item: item["gap"])
        smallest_gap = min(gap_records, key=lambda item: abs(item["gap"]))
        grouped.largest_gap_category = largest_gap["category"]
        grouped.largest_gap_groups = [
            item for item in (largest_gap["winner"], largest_gap["runner_up"]) if item
        ]
        grouped.largest_gap_value = float(largest_gap["gap"])
        grouped.largest_gap_percent = largest_gap["gap_percent"]
        grouped.largest_gap_percent_basis = (
            f"{largest_gap['runner_up']} {_short_metric(y_col)}"
            if largest_gap.get("runner_up")
            else None
        )
        grouped.most_balanced_category = smallest_gap["category"]
        grouped.smallest_gap_value = float(smallest_gap["gap"])

    if spec.limit and len(frame) >= spec.limit:
        grouped.top_n_applied = spec.limit
    if missing_pairs:
        grouped.warnings.append(
            f"{len(missing_pairs)} category-group combination(s) are not present in the displayed data."
        )
        grouped.evidence_strength = "medium"
    elif grouped.category_count >= 3 and grouped.group_count >= 2 and grouped.valid_rows >= grouped.category_count:
        grouped.evidence_strength = "high"
    else:
        grouped.evidence_strength = "low"

    grouped.calculated_metrics.update({
        "category_count": grouped.category_count,
        "group_count": grouped.group_count,
        "category_totals": grouped.category_totals,
        "group_totals": grouped.group_totals,
        "highest_combined_category": grouped.highest_combined_category,
        "highest_combined_value": grouped.highest_combined_value,
        "lowest_combined_category": grouped.lowest_combined_category,
        "lowest_combined_value": grouped.lowest_combined_value,
        "highest_individual_category": grouped.highest_individual_category,
        "highest_individual_group": grouped.highest_individual_group,
        "highest_individual_value": grouped.highest_individual_value,
        "lowest_individual_category": grouped.lowest_individual_category,
        "lowest_individual_group": grouped.lowest_individual_group,
        "lowest_individual_value": grouped.lowest_individual_value,
        "winner_by_category": grouped.winner_by_category,
        "group_win_counts": grouped.group_win_counts,
        "largest_gap_category": grouped.largest_gap_category,
        "largest_gap_groups": grouped.largest_gap_groups,
        "largest_gap_value": grouped.largest_gap_value,
        "largest_gap_percent": grouped.largest_gap_percent,
        "largest_gap_percent_basis": grouped.largest_gap_percent_basis,
        "most_balanced_category": grouped.most_balanced_category,
        "missing_group_combinations": grouped.missing_group_combinations,
    })
    return grouped


def _single_line_evidence(result: ChartResult, evidence: ChartEvidence) -> SingleLineEvidence:
    rows = _records(result)
    spec = result.spec
    y_col = _metric_column(result)
    line = SingleLineEvidence(
        **evidence.model_dump(exclude={"x_column", "aggregation", "warnings"}),
        x_column=spec.x or "",
        y_column=y_col or "",
        aggregation=spec.aggregation,
        warnings=list(evidence.warnings),
        filters_applied=dict(evidence.filters),
    )
    if not is_single_series_line(result) or not spec.x or not y_col:
        line.warnings.append("Single-line evidence requires one x-axis, one numeric y-axis, and no grouping.")
        line.evidence_strength = "low"
        return line
    frame = pd.DataFrame(rows)
    if spec.x not in frame or y_col not in frame:
        line.warnings.append("The displayed line data is missing the x-axis or value column.")
        line.evidence_strength = "low"
        return line

    frame = frame[[spec.x, y_col]].copy()
    frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
    line.excluded_rows = int(frame[y_col].isna().sum())
    frame = frame.dropna(subset=[y_col])
    order, inferred_granularity, ordered = _is_ordered_line_axis(frame[spec.x], spec.x)
    if not ordered or order.isna().any():
        line.warnings.append("The x-axis is not an ordered numeric or date-like progression.")
        line.valid_rows = int(len(frame))
        line.evidence_strength = "low"
        return line
    line.time_granularity = spec.time_grain or inferred_granularity
    frame["_order"] = order
    if pd.api.types.is_datetime64_any_dtype(order):
        frame["_period"] = pd.to_datetime(frame[spec.x], errors="coerce")
    else:
        frame["_period"] = frame[spec.x]
    frame = frame.sort_values("_order").reset_index(drop=True)
    line.valid_rows = int(len(frame))
    line.point_count = int(len(frame))
    if len(frame) < 2:
        line.warnings.append("At least two ordered points are required for line-chart movement evidence.")
        line.evidence_strength = "low"
        return line

    values = frame[y_col].astype(float)
    labels = _period_labels(frame["_period"], line.time_granularity)
    first = frame.iloc[0]
    final = frame.iloc[-1]
    start_value = float(values.iloc[0])
    end_value = float(values.iloc[-1])
    endpoint_change = end_value - start_value
    line.start_period = first["_period"]
    line.start_period_label = labels[0]
    line.start_value = start_value
    line.end_period = final["_period"]
    line.end_period_label = labels[-1]
    line.end_value = end_value
    line.endpoint_change = endpoint_change
    line.endpoint_change_percent = endpoint_change / abs(start_value) * 100 if start_value else None
    line.endpoint_change_basis = "starting value"

    peak_index = values.idxmax()
    trough_index = values.idxmin()
    line.peak_period = frame.loc[peak_index, "_period"]
    line.peak_period_label = labels[int(peak_index)]
    line.peak_value = float(values.loc[peak_index])
    line.trough_period = frame.loc[trough_index, "_period"]
    line.trough_period_label = labels[int(trough_index)]
    line.trough_value = float(values.loc[trough_index])
    line.value_range = line.peak_value - line.trough_value

    changes = values.diff().dropna()
    previous_values = values.shift(1)
    percent_changes = [
        _period_change_percent(float(values.iloc[index]), float(previous_values.iloc[index]))
        for index in range(1, len(values))
    ]
    if not changes.empty:
        increase_index = changes.idxmax()
        decline_index = changes.idxmin()
        strongest_increase = float(changes.loc[increase_index])
        strongest_decline = float(changes.loc[decline_index])
        if strongest_increase > 0:
            line.strongest_increase_start = frame.loc[increase_index - 1, "_period"]
            line.strongest_increase_start_label = labels[int(increase_index - 1)]
            line.strongest_increase_end = frame.loc[increase_index, "_period"]
            line.strongest_increase_end_label = labels[int(increase_index)]
            line.strongest_increase_value = strongest_increase
            line.strongest_increase_percent = percent_changes[int(increase_index - 1)]
        if strongest_decline < 0:
            line.strongest_decline_start = frame.loc[decline_index - 1, "_period"]
            line.strongest_decline_start_label = labels[int(decline_index - 1)]
            line.strongest_decline_end = frame.loc[decline_index, "_period"]
            line.strongest_decline_end_label = labels[int(decline_index)]
            line.strongest_decline_value = strongest_decline
            line.strongest_decline_percent = percent_changes[int(decline_index - 1)]
        line.mean_period_change = float(changes.mean())
        line.median_period_change = float(changes.median())
        line.period_change_std = float(changes.std()) if len(changes) > 1 else 0.0
        line.mean_absolute_period_change = float(changes.abs().mean())
        line.median_absolute_period_change = float(changes.abs().median())
        valid_pct_changes = [abs(value) for value in percent_changes if value is not None and isfinite(value)]
        line.mean_absolute_percent_change = float(np.mean(valid_pct_changes)) if valid_pct_changes else None
        line.positive_change_count = int((changes > 0).sum())
        line.negative_change_count = int((changes < 0).sum())
        line.unchanged_count = int((changes == 0).sum())
        line.direction_reversal_count = _direction_reversals(changes)

    mean_value = float(values.mean())
    value_std = float(values.std()) if len(values) > 1 else 0.0
    line.coefficient_of_variation = abs(value_std / mean_value) if mean_value else None
    line.volatility_level = _volatility_level(line.coefficient_of_variation)

    if len(values) >= 2:
        x_numeric = np.arange(len(values), dtype=float)
        slope, intercept = np.polyfit(x_numeric, values.to_numpy(dtype=float), 1)
        predicted = slope * x_numeric + intercept
        ss_res = float(((values.to_numpy(dtype=float) - predicted) ** 2).sum())
        ss_tot = float(((values.to_numpy(dtype=float) - mean_value) ** 2).sum())
        r_squared = 1 - ss_res / ss_tot if ss_tot else 1.0
        line.linear_trend_slope = float(slope)
        line.linear_trend_r_squared = float(r_squared)
        line.trend_direction = "upward" if slope > 0 else "downward" if slope < 0 else "flat"
        line.trend_strength = _trend_strength(line.linear_trend_r_squared)
        if line.direction_reversal_count == 0 and line.trend_strength == "strong":
            line.volatility_level = "low"
        elif line.direction_reversal_count <= 1 and line.volatility_level == "high" and line.trend_strength == "strong":
            line.volatility_level = "moderate"
        line.pattern_classification = _classify_line_pattern(
            line.linear_trend_slope,
            values,
            line.volatility_level,
            line.trend_strength,
            line.endpoint_change,
            line.value_range,
        )

    if pd.api.types.is_datetime64_any_dtype(frame["_order"]):
        missing, irregular = _missing_periods(pd.to_datetime(frame["_order"]), line.time_granularity)
        line.missing_periods = missing
        line.missing_period_labels = _period_labels(pd.Series(missing), line.time_granularity)
        line.irregular_intervals = irregular
    line.seasonality_evidence = _seasonality_evidence(line.point_count, line.time_granularity)

    if line.point_count < 4 or not ordered:
        line.evidence_strength = "low"
    elif line.irregular_intervals or line.missing_periods or line.volatility_level == "high":
        line.evidence_strength = "medium"
    elif line.point_count >= 8 and not line.excluded_rows:
        line.evidence_strength = "high"
    else:
        line.evidence_strength = "medium"

    line.calculated_metrics.update({
        "point_count": line.point_count,
        "start_period": line.start_period_label,
        "start_value": line.start_value,
        "end_period": line.end_period_label,
        "end_value": line.end_value,
        "endpoint_change": line.endpoint_change,
        "endpoint_change_percent": line.endpoint_change_percent,
        "endpoint_change_basis": line.endpoint_change_basis,
        "peak_period": line.peak_period_label,
        "peak_value": line.peak_value,
        "trough_period": line.trough_period_label,
        "trough_value": line.trough_value,
        "value_range": line.value_range,
        "strongest_increase_value": line.strongest_increase_value,
        "strongest_decline_value": line.strongest_decline_value,
        "coefficient_of_variation": line.coefficient_of_variation,
        "volatility_level": line.volatility_level,
        "linear_trend_slope": line.linear_trend_slope,
        "linear_trend_r_squared": line.linear_trend_r_squared,
        "trend_strength": line.trend_strength,
        "pattern_classification": line.pattern_classification,
        "direction_reversal_count": line.direction_reversal_count,
        "missing_periods": line.missing_period_labels,
        "seasonality_evidence": line.seasonality_evidence,
    })
    return line


def _single_area_evidence(result: ChartResult, evidence: ChartEvidence) -> SingleAreaEvidence:
    rows = _records(result)
    spec = result.spec
    y_col = _metric_column(result)
    area = SingleAreaEvidence(
        **evidence.model_dump(exclude={"x_column", "aggregation", "warnings"}),
        x_column=spec.x or "",
        y_column=y_col or "",
        aggregation=spec.aggregation,
        warnings=list(evidence.warnings),
        filters_applied=dict(evidence.filters),
    )
    if not is_single_area(result) or not spec.x or not y_col:
        area.warnings.append("Single-area evidence requires one ordered x-axis, one numeric y-axis, and no stack field.")
        area.evidence_strength = "low"
        return area
    frame = pd.DataFrame(rows)
    if spec.x not in frame or y_col not in frame:
        area.warnings.append("The displayed area data is missing the x-axis or value column.")
        area.evidence_strength = "low"
        return area

    frame = frame[[spec.x, y_col]].copy()
    frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
    area.excluded_rows = int(frame[y_col].isna().sum())
    frame = frame.dropna(subset=[y_col])
    order, inferred_granularity, axis_type, ordered = _area_axis_order(frame[spec.x], spec.x)
    area.x_axis_type = axis_type
    area.time_granularity = spec.time_grain or inferred_granularity
    if not ordered:
        area.warnings.append("The x-axis is unordered, so an area chart may be inappropriate; use a bar chart for categorical comparison.")
        area.valid_rows = int(len(frame))
        area.evidence_strength = "low"
        return area

    frame["_order"] = order
    frame["_period"] = pd.to_datetime(frame[spec.x], errors="coerce") if axis_type == "datetime" else frame[spec.x]
    frame = frame.sort_values("_order").reset_index(drop=True)
    area.valid_rows = int(len(frame))
    area.point_count = int(len(frame))
    if len(frame) < 2:
        area.warnings.append("At least two ordered points are required for area-chart movement evidence.")
        area.evidence_strength = "low"
        return area

    values = frame[y_col].astype(float)
    labels = _period_labels(frame["_period"], area.time_granularity) if axis_type == "datetime" else frame[spec.x].astype(str).tolist()
    area.start_period = frame.loc[0, "_period"]
    area.start_period_label = labels[0]
    area.start_value = float(values.iloc[0])
    area.end_period = frame.loc[len(frame) - 1, "_period"]
    area.end_period_label = labels[-1]
    area.end_value = float(values.iloc[-1])
    area.endpoint_change = area.end_value - area.start_value
    area.endpoint_change_percent = area.endpoint_change / abs(area.start_value) * 100 if area.start_value else None
    area.endpoint_change_basis = "starting value"

    peak_index = values.idxmax()
    trough_index = values.idxmin()
    area.peak_period = frame.loc[peak_index, "_period"]
    area.peak_period_label = labels[int(peak_index)]
    area.peak_value = float(values.loc[peak_index])
    area.trough_period = frame.loc[trough_index, "_period"]
    area.trough_period_label = labels[int(trough_index)]
    area.trough_value = float(values.loc[trough_index])
    area.value_range = area.peak_value - area.trough_value
    area.mean_value = float(values.mean())
    area.median_value = float(values.median())
    area.negative_value_count = int((values < 0).sum())

    high_threshold = float(values.quantile(0.75))
    low_threshold = float(values.quantile(0.25))
    high_mask = values >= high_threshold
    low_mask = values <= low_threshold
    area.high_periods = frame.loc[high_mask, "_period"].tolist()
    area.high_period_labels = [labels[index] for index in frame.index[high_mask]]
    area.low_periods = frame.loc[low_mask, "_period"].tolist()
    area.low_period_labels = [labels[index] for index in frame.index[low_mask]]
    area.longest_above_average_run = _longest_run(values > area.mean_value)
    area.longest_below_average_run = _longest_run(values < area.mean_value)

    changes = values.diff().dropna()
    previous_values = values.shift(1)
    percent_changes = [
        _period_change_percent(float(values.iloc[index]), float(previous_values.iloc[index]))
        for index in range(1, len(values))
    ]
    if not changes.empty:
        increase_index = changes.idxmax()
        decline_index = changes.idxmin()
        strongest_increase = float(changes.loc[increase_index])
        strongest_decline = float(changes.loc[decline_index])
        if strongest_increase > 0:
            area.strongest_increase_start = frame.loc[increase_index - 1, "_period"]
            area.strongest_increase_start_label = labels[int(increase_index - 1)]
            area.strongest_increase_end = frame.loc[increase_index, "_period"]
            area.strongest_increase_end_label = labels[int(increase_index)]
            area.strongest_increase_value = strongest_increase
            area.strongest_increase_percent = percent_changes[int(increase_index - 1)]
        if strongest_decline < 0:
            area.strongest_decline_start = frame.loc[decline_index - 1, "_period"]
            area.strongest_decline_start_label = labels[int(decline_index - 1)]
            area.strongest_decline_end = frame.loc[decline_index, "_period"]
            area.strongest_decline_end_label = labels[int(decline_index)]
            area.strongest_decline_value = strongest_decline
            area.strongest_decline_percent = percent_changes[int(decline_index - 1)]
        area.direction_reversal_count = _direction_reversals(changes)

    value_std = float(values.std()) if len(values) > 1 else 0.0
    area.coefficient_of_variation = abs(value_std / area.mean_value) if area.mean_value else None
    area.volatility_level = _area_volatility_label(area.coefficient_of_variation)
    x_numeric = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x_numeric, values.to_numpy(dtype=float), 1)
    predicted = slope * x_numeric + intercept
    ss_res = float(((values.to_numpy(dtype=float) - predicted) ** 2).sum())
    ss_tot = float(((values.to_numpy(dtype=float) - area.mean_value) ** 2).sum())
    area.linear_trend_slope = float(slope)
    area.linear_trend_r_squared = 1 - ss_res / ss_tot if ss_tot else 1.0
    area.trend_direction = "upward" if slope > 0 else "downward" if slope < 0 else "flat"
    area.trend_strength = _trend_strength(area.linear_trend_r_squared)
    if area.direction_reversal_count == 0 and area.trend_strength == "strong":
        area.volatility_level = "stable"
    area.pattern_classification = _classify_area_pattern(
        area.linear_trend_slope,
        values,
        area.volatility_level,
        area.trend_strength,
    )

    area.area_interpretation_valid = bool(
        area.baseline_is_zero
        and area.negative_value_count == 0
        and axis_type in {"datetime", "numeric", "ordered_period"}
    )
    if (spec.aggregation or "sum") in {"sum", "count"}:
        area.approximate_area_under_curve = float(values.sum())
    elif area.area_interpretation_valid and area.irregular_intervals is False:
        area.approximate_area_under_curve = float(np.trapezoid(values.to_numpy(dtype=float), x_numeric))
    if area.negative_value_count:
        area.warnings.append("The series includes negative values, so filled regions above and below zero require careful interpretation.")
    if axis_type == "datetime":
        missing, irregular = _missing_periods(pd.to_datetime(frame["_order"]), area.time_granularity)
        area.missing_periods = missing
        area.missing_period_labels = _period_labels(pd.Series(missing), area.time_granularity)
        area.irregular_intervals = irregular
        if missing:
            area.warnings.append("Missing periods can make the filled area appear more continuous than the data supports.")

    if area.point_count < 4 or axis_type == "categorical" or area.negative_value_count / max(area.point_count, 1) > 0.3:
        area.evidence_strength = "low"
    elif area.missing_periods or area.irregular_intervals or area.negative_value_count:
        area.evidence_strength = "medium"
    elif area.point_count >= 8 and area.area_interpretation_valid:
        area.evidence_strength = "high"
    else:
        area.evidence_strength = "medium"

    area.calculated_metrics.update({
        "point_count": area.point_count,
        "start_period": area.start_period_label,
        "start_value": area.start_value,
        "end_period": area.end_period_label,
        "end_value": area.end_value,
        "endpoint_change": area.endpoint_change,
        "endpoint_change_percent": area.endpoint_change_percent,
        "peak_period": area.peak_period_label,
        "peak_value": area.peak_value,
        "trough_period": area.trough_period_label,
        "trough_value": area.trough_value,
        "mean_value": area.mean_value,
        "median_value": area.median_value,
        "high_periods": area.high_period_labels,
        "low_periods": area.low_period_labels,
        "longest_above_average_run": area.longest_above_average_run,
        "longest_below_average_run": area.longest_below_average_run,
        "volatility_level": area.volatility_level,
        "trend_strength": area.trend_strength,
        "pattern_classification": area.pattern_classification,
        "missing_periods": area.missing_period_labels,
        "negative_value_count": area.negative_value_count,
    })
    return area


def _stacked_area_evidence(result: ChartResult, evidence: ChartEvidence) -> StackedAreaEvidence:
    rows = _records(result)
    spec = result.spec
    y_col = _metric_column(result)
    stacked = StackedAreaEvidence(
        **evidence.model_dump(exclude={"x_column", "aggregation", "warnings", "chart_type"}),
        x_column=spec.x or "",
        stack_column=spec.color or "",
        y_column=y_col or "",
        aggregation=spec.aggregation,
        warnings=list(evidence.warnings),
        filters_applied=dict(evidence.filters),
    )
    if not is_stacked_area(result) or not spec.x or not spec.color or not y_col:
        stacked.warnings.append("Stacked-area evidence requires an ordered x-axis, one value column, and one stack field.")
        stacked.evidence_strength = "low"
        return stacked
    frame = pd.DataFrame(rows)
    required = {spec.x, spec.color, y_col}
    if not required.issubset(frame.columns):
        stacked.warnings.append("The displayed stacked-area data is missing required columns.")
        stacked.evidence_strength = "low"
        return stacked
    frame = frame[[spec.x, spec.color, y_col]].copy()
    frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
    stacked.excluded_rows = int(frame[y_col].isna().sum())
    frame = frame.dropna(subset=[y_col])
    order, inferred_granularity, axis_type, ordered = _area_axis_order(frame[spec.x], spec.x)
    stacked.time_granularity = spec.time_grain or inferred_granularity
    if not ordered:
        stacked.warnings.append("The x-axis is unordered, so a stacked area chart may be inappropriate.")
        stacked.valid_rows = int(len(frame))
        stacked.evidence_strength = "low"
        return stacked

    frame["_order"] = order
    frame["_period"] = pd.to_datetime(frame[spec.x], errors="coerce") if axis_type == "datetime" else frame[spec.x]
    frame["_period_label"] = (
        _period_labels(frame["_period"], stacked.time_granularity)
        if axis_type == "datetime"
        else frame[spec.x].astype(str).tolist()
    )
    frame["_stack_label"] = frame[spec.color].map(lambda value: friendly_value(spec.color, value))
    frame = frame.sort_values(["_order", "_stack_label"]).reset_index(drop=True)
    stacked.valid_rows = int(len(frame))
    stacked.point_count = int(frame["_period_label"].nunique(dropna=False))
    stacked.stack_count = int(frame["_stack_label"].nunique(dropna=False))
    stacked.negative_value_count = int((frame[y_col] < 0).sum())

    pivot = frame.pivot_table(
        index="_period_label",
        columns="_stack_label",
        values=y_col,
        aggfunc="sum",
        fill_value=0,
    )
    period_order = frame[["_period_label", "_order"]].drop_duplicates().sort_values("_order")
    pivot = pivot.reindex(period_order["_period_label"].tolist())
    totals = pivot.sum(axis=1)
    stacked.total_by_period = {str(period): float(value) for period, value in totals.items()}
    stacked.values_by_period_and_stack = {
        str(period): {str(stack): float(value) for stack, value in values.items()}
        for period, values in pivot.to_dict(orient="index").items()
    }
    if not totals.empty:
        stacked.start_total = float(totals.iloc[0])
        stacked.end_total = float(totals.iloc[-1])
        stacked.total_change = stacked.end_total - stacked.start_total
        stacked.total_change_percent = stacked.total_change / abs(stacked.start_total) * 100 if stacked.start_total else None
        stacked.peak_period_label = str(totals.idxmax())
        stacked.peak_period = stacked.peak_period_label
        stacked.peak_total = float(totals.max())
        stacked.trough_period_label = str(totals.idxmin())
        stacked.trough_period = stacked.trough_period_label
        stacked.trough_total = float(totals.min())

    stack_totals = pivot.sum(axis=0).sort_values(ascending=False)
    total_sum = float(stack_totals.sum())
    stacked.overall_stack_totals = {str(stack): float(value) for stack, value in stack_totals.items()}
    stacked.overall_stack_shares = {
        str(stack): float(value / total_sum * 100) if total_sum else 0.0
        for stack, value in stack_totals.items()
    }
    if not stack_totals.empty:
        stacked.dominant_stack_overall = str(stack_totals.index[0])
        stacked.dominant_stack_share = stacked.overall_stack_shares.get(stacked.dominant_stack_overall)
    share_by_period = pivot.div(totals.replace(0, np.nan), axis=0).fillna(0) * 100
    stacked.stack_share_by_period = {
        str(period): {str(stack): float(value) for stack, value in values.items()}
        for period, values in share_by_period.to_dict(orient="index").items()
    }
    stacked.dominant_stack_by_period = {
        str(period): str(values.idxmax())
        for period, values in pivot.iterrows()
    }
    if len(pivot) >= 2:
        growth = pivot.iloc[-1] - pivot.iloc[0]
        stacked.stack_with_largest_growth = str(growth.idxmax())
        stacked.stack_with_largest_decline = str(growth.idxmin())
        share_shift = (share_by_period.iloc[-1] - share_by_period.iloc[0]).abs()
        if not share_shift.empty and float(share_shift.max()) >= 10:
            stacked.composition_shift_periods = [str(share_by_period.index[-1])]

    all_periods = pivot.index.tolist()
    all_stacks = pivot.columns.tolist()
    observed_pairs = set(zip(frame["_period_label"], frame["_stack_label"], strict=False))
    stacked.missing_combinations = [
        {"period": str(period), "stack": str(stack)}
        for period in all_periods
        for stack in all_stacks
        if (period, stack) not in observed_pairs
    ]
    if stacked.negative_value_count:
        stacked.warnings.append("The stacked area contains negative values, which makes filled component areas harder to compare.")
    if stacked.missing_combinations:
        stacked.warnings.append("Some period-stack combinations are missing from the displayed data.")

    if stacked.point_count < 4 or stacked.stack_count < 2 or stacked.negative_value_count / max(len(frame), 1) > 0.3:
        stacked.evidence_strength = "low"
    elif stacked.missing_combinations or stacked.negative_value_count:
        stacked.evidence_strength = "medium"
    else:
        stacked.evidence_strength = "high" if stacked.point_count >= 6 else "medium"

    stacked.calculated_metrics.update({
        "point_count": stacked.point_count,
        "stack_count": stacked.stack_count,
        "total_by_period": stacked.total_by_period,
        "start_total": stacked.start_total,
        "end_total": stacked.end_total,
        "total_change": stacked.total_change,
        "peak_period": stacked.peak_period_label,
        "peak_total": stacked.peak_total,
        "trough_period": stacked.trough_period_label,
        "trough_total": stacked.trough_total,
        "dominant_stack_overall": stacked.dominant_stack_overall,
        "dominant_stack_share": stacked.dominant_stack_share,
        "stack_with_largest_growth": stacked.stack_with_largest_growth,
        "stack_with_largest_decline": stacked.stack_with_largest_decline,
        "missing_combinations": stacked.missing_combinations,
    })
    return stacked


def _line_like_evidence(result: ChartResult, evidence: ChartEvidence) -> ChartEvidence:
    rows = _records(result)
    spec = result.spec
    y_col = _metric_column(result)
    values = _numeric_series(rows, y_col)
    if values.empty or not y_col or not spec.x:
        evidence.warnings.append("No numeric line values were available.")
        evidence.evidence_strength = "low"
        return evidence
    frame = pd.DataFrame(rows).loc[values.index].copy()
    axis = frame[spec.x]
    axis_name = str(spec.x).lower()
    looks_temporal = (
        pd.api.types.is_datetime64_any_dtype(axis)
        or any(token in axis_name for token in ("date", "time", "year", "month"))
    )
    if looks_temporal:
        frame["_order"] = pd.to_datetime(axis, errors="coerce")
    else:
        frame["_order"] = pd.Series(pd.NA, index=frame.index)
    if frame["_order"].notna().any():
        frame = frame.sort_values("_order")
    first = frame.iloc[0]
    final = frame.iloc[-1]
    change = float(final[y_col]) - float(first[y_col])
    pct_change = change / abs(float(first[y_col])) * 100 if float(first[y_col]) else None
    diffs = pd.to_numeric(frame[y_col], errors="coerce").diff().dropna()
    evidence.calculated_metrics.update({
        "first_label": first[spec.x],
        "first_value": float(first[y_col]),
        "final_label": final[spec.x],
        "final_value": float(final[y_col]),
        "absolute_change": change,
        "percentage_change": pct_change,
        "trend_direction": "increased" if change > 0 else "decreased" if change < 0 else "flat",
        "minimum_label": frame.loc[pd.to_numeric(frame[y_col], errors="coerce").idxmin(), spec.x],
        "maximum_label": frame.loc[pd.to_numeric(frame[y_col], errors="coerce").idxmax(), spec.x],
        "volatility_std": float(diffs.std()) if len(diffs) > 1 else 0.0,
        "average_change": float(diffs.mean()) if not diffs.empty else 0.0,
    })
    return evidence


def _dual_line_evidence(result: ChartResult, evidence: ChartEvidence) -> DualLineEvidence:
    rows = _records(result)
    spec = result.spec
    primary_aggregation = spec.primary_aggregation or spec.aggregation or "sum"
    secondary_aggregation = spec.secondary_aggregation or spec.aggregation or "sum"
    dual = DualLineEvidence(
        **evidence.model_dump(exclude={"chart_type", "x_column", "aggregation", "warnings"}),
        x_column=spec.x or "",
        primary_y_column=spec.y or "",
        primary_aggregation=primary_aggregation,
        secondary_y_column=spec.secondary_y or "",
        secondary_aggregation=secondary_aggregation,
        filters_applied=dict(evidence.filters),
        warnings=list(evidence.warnings),
    )
    if spec.chart_type != "dual_line" or not spec.x or not spec.y or not spec.secondary_y:
        dual.warnings.append("Dual-line evidence requires one x-axis and two metric columns.")
        dual.evidence_strength = "low"
        return dual
    frame = pd.DataFrame(rows)
    required = {spec.x, spec.y, spec.secondary_y}
    if not required.issubset(frame.columns):
        dual.warnings.append("The displayed dual-line data is missing required columns.")
        dual.evidence_strength = "low"
        return dual
    frame = frame[[spec.x, spec.y, spec.secondary_y]].copy()
    frame[spec.y] = pd.to_numeric(frame[spec.y], errors="coerce")
    frame[spec.secondary_y] = pd.to_numeric(frame[spec.secondary_y], errors="coerce")
    order, granularity, axis_type, ordered = _area_axis_order(frame[spec.x], spec.x)
    dual.x_axis_type = axis_type if ordered else "unordered categorical"
    frame["_order"] = order
    if ordered:
        frame = frame.sort_values("_order").reset_index(drop=True)
    dual.category_count = int(len(frame))
    dual.valid_primary_count = int(frame[spec.y].notna().sum())
    dual.valid_secondary_count = int(frame[spec.secondary_y].notna().sum())
    dual.primary_unit = _metric_unit(spec.y)
    dual.secondary_unit = _metric_unit(spec.secondary_y)
    dual.unit_relationship = _unit_relationship(dual.primary_unit, dual.secondary_unit)
    dual.known_metric_relationship, dual.derived_metric_available = _known_metric_relationship(spec.y, spec.secondary_y)
    dual.aggregation_warning = _aggregation_warning(primary_aggregation, secondary_aggregation)
    if dual.aggregation_warning:
        dual.warnings.append(dual.aggregation_warning)

    labels = (
        _period_labels(pd.to_datetime(frame[spec.x], errors="coerce"), granularity)
        if axis_type == "datetime" and ordered
        else frame[spec.x].astype(str).tolist()
    )
    dual.primary_values = {
        str(label): float(value)
        for label, value in zip(labels, frame[spec.y], strict=False)
        if pd.notna(value)
    }
    dual.secondary_values = {
        str(label): float(value)
        for label, value in zip(labels, frame[spec.secondary_y], strict=False)
        if pd.notna(value)
    }

    primary_valid = frame.loc[frame[spec.y].notna(), [spec.x, spec.y]].copy()
    secondary_valid = frame.loc[frame[spec.secondary_y].notna(), [spec.x, spec.secondary_y]].copy()
    if not primary_valid.empty:
        primary_high = primary_valid[spec.y].idxmax()
        primary_low = primary_valid[spec.y].idxmin()
        dual.primary_highest_x = labels[int(primary_high)]
        dual.primary_highest_value = float(frame.loc[primary_high, spec.y])
        dual.primary_lowest_x = labels[int(primary_low)]
        dual.primary_lowest_value = float(frame.loc[primary_low, spec.y])
        dual.primary_range = dual.primary_highest_value - dual.primary_lowest_value
        dual.primary_ranking = [
            labels[int(index)]
            for index in frame[spec.y].sort_values(ascending=False, na_position="last").dropna().index
        ]
    if not secondary_valid.empty:
        secondary_high = secondary_valid[spec.secondary_y].idxmax()
        secondary_low = secondary_valid[spec.secondary_y].idxmin()
        dual.secondary_highest_x = labels[int(secondary_high)]
        dual.secondary_highest_value = float(frame.loc[secondary_high, spec.secondary_y])
        dual.secondary_lowest_x = labels[int(secondary_low)]
        dual.secondary_lowest_value = float(frame.loc[secondary_low, spec.secondary_y])
        dual.secondary_range = dual.secondary_highest_value - dual.secondary_lowest_value
        dual.secondary_ranking = [
            labels[int(index)]
            for index in frame[spec.secondary_y].sort_values(ascending=False, na_position="last").dropna().index
        ]
    dual.same_highest_category = dual.primary_highest_x == dual.secondary_highest_x if dual.primary_highest_x and dual.secondary_highest_x else None
    dual.same_lowest_category = dual.primary_lowest_x == dual.secondary_lowest_x if dual.primary_lowest_x and dual.secondary_lowest_x else None
    dual.rank_agreement_count = sum(
        1 for first, second in zip(dual.primary_ranking, dual.secondary_ranking, strict=False)
        if first == second
    )
    dual.rank_disagreement_categories = [
        item for item in dual.primary_ranking
        if item in dual.secondary_ranking and dual.primary_ranking.index(item) != dual.secondary_ranking.index(item)
    ]

    paired = frame[[spec.y, spec.secondary_y]].dropna()
    if len(paired) >= 3 and paired[spec.y].nunique() > 1 and paired[spec.secondary_y].nunique() > 1:
        dual.pearson_correlation = float(paired[spec.y].corr(paired[spec.secondary_y]))
        dual.spearman_correlation = float(paired[spec.y].corr(paired[spec.secondary_y], method="spearman"))
        dual.relationship_strength = _relationship_strength(dual.pearson_correlation)
        dual.relationship_direction = "positive" if dual.pearson_correlation > 0 else "negative" if dual.pearson_correlation < 0 else "neutral"
    elif len(paired) < 5:
        dual.warnings.append("Fewer than five paired values are displayed, so relationship evidence is limited.")

    if ordered and axis_type in {"datetime", "numeric", "ordered_period"}:
        paired_ordered = frame[[spec.x, spec.y, spec.secondary_y]].dropna(
            subset=[spec.y, spec.secondary_y]
        ).sort_index().reset_index(drop=False)
        paired_labels = [labels[int(index)] for index in paired_ordered["index"]]
        dual.paired_point_count = int(len(paired_ordered))
        if dual.valid_primary_count >= 2:
            primary_series = frame[spec.y].dropna()
            dual.primary_start_x = labels[int(primary_series.index[0])]
            dual.primary_start_value = float(primary_series.iloc[0])
            dual.primary_end_x = labels[int(primary_series.index[-1])]
            dual.primary_end_value = float(primary_series.iloc[-1])
            dual.primary_change = dual.primary_end_value - dual.primary_start_value
            dual.primary_change_percent = dual.primary_change / abs(dual.primary_start_value) * 100 if dual.primary_start_value else None
            dual.primary_endpoint_direction = _endpoint_direction_label(dual.primary_change, dual.primary_start_value)
            primary_peak = frame[spec.y].idxmax()
            primary_trough = frame[spec.y].idxmin()
            dual.primary_peak_period = labels[int(primary_peak)]
            dual.primary_peak_value = float(frame.loc[primary_peak, spec.y])
            dual.primary_trough_period = labels[int(primary_trough)]
            dual.primary_trough_value = float(frame.loc[primary_trough, spec.y])
            primary_change = _strongest_change(frame[spec.y], labels)
            dual.primary_strongest_increase_start = primary_change.get("increase_start")
            dual.primary_strongest_increase_end = primary_change.get("increase_end")
            dual.primary_strongest_increase_value = primary_change.get("increase_value")
            dual.primary_strongest_decline_start = primary_change.get("decline_start")
            dual.primary_strongest_decline_end = primary_change.get("decline_end")
            dual.primary_strongest_decline_value = primary_change.get("decline_value")
            dual.primary_coefficient_of_variation, dual.primary_volatility_level = _series_volatility(primary_series)
        if dual.valid_secondary_count >= 2:
            secondary_series = frame[spec.secondary_y].dropna()
            dual.secondary_start_x = labels[int(secondary_series.index[0])]
            dual.secondary_start_value = float(secondary_series.iloc[0])
            dual.secondary_end_x = labels[int(secondary_series.index[-1])]
            dual.secondary_end_value = float(secondary_series.iloc[-1])
            dual.secondary_change = dual.secondary_end_value - dual.secondary_start_value
            dual.secondary_change_percent = dual.secondary_change / abs(dual.secondary_start_value) * 100 if dual.secondary_start_value else None
            dual.secondary_endpoint_direction = _endpoint_direction_label(dual.secondary_change, dual.secondary_start_value)
            secondary_peak = frame[spec.secondary_y].idxmax()
            secondary_trough = frame[spec.secondary_y].idxmin()
            dual.secondary_peak_period = labels[int(secondary_peak)]
            dual.secondary_peak_value = float(frame.loc[secondary_peak, spec.secondary_y])
            dual.secondary_trough_period = labels[int(secondary_trough)]
            dual.secondary_trough_value = float(frame.loc[secondary_trough, spec.secondary_y])
            secondary_change = _strongest_change(frame[spec.secondary_y], labels)
            dual.secondary_strongest_increase_start = secondary_change.get("increase_start")
            dual.secondary_strongest_increase_end = secondary_change.get("increase_end")
            dual.secondary_strongest_increase_value = secondary_change.get("increase_value")
            dual.secondary_strongest_decline_start = secondary_change.get("decline_start")
            dual.secondary_strongest_decline_end = secondary_change.get("decline_end")
            dual.secondary_strongest_decline_value = secondary_change.get("decline_value")
            dual.secondary_coefficient_of_variation, dual.secondary_volatility_level = _series_volatility(secondary_series)
        dual.peaks_aligned = dual.primary_peak_period == dual.secondary_peak_period if dual.primary_peak_period and dual.secondary_peak_period else None
        dual.troughs_aligned = dual.primary_trough_period == dual.secondary_trough_period if dual.primary_trough_period and dual.secondary_trough_period else None
        if dual.primary_coefficient_of_variation is not None and dual.secondary_coefficient_of_variation is not None:
            if abs(dual.primary_coefficient_of_variation - dual.secondary_coefficient_of_variation) < 0.02:
                dual.more_volatile_metric = "similar"
            elif dual.primary_coefficient_of_variation > dual.secondary_coefficient_of_variation:
                dual.more_volatile_metric = display_name(spec.y)
            else:
                dual.more_volatile_metric = display_name(spec.secondary_y)
        if len(paired_ordered) >= 2:
            primary_diff = paired_ordered[spec.y].diff().dropna()
            secondary_diff = paired_ordered[spec.secondary_y].diff().dropna()
            paired_diffs = pd.DataFrame({"p": primary_diff, "s": secondary_diff}).dropna()
            comparable = int(len(paired_diffs))
            dual.comparable_transition_count = comparable
            if comparable:
                p_sign = np.sign(paired_diffs["p"])
                s_sign = np.sign(paired_diffs["s"])
                unchanged = (p_sign == 0) | (s_sign == 0)
                dual.unchanged_transition_count = int(unchanged.sum())
                comparable_direction = ~unchanged
                dual.aligned_direction_count = int(((p_sign == s_sign) & comparable_direction).sum())
                dual.opposite_direction_count = int(((p_sign != s_sign) & comparable_direction).sum())
                dual.aligned_direction_percent = dual.aligned_direction_count / comparable * 100
                dual.opposite_direction_percent = dual.opposite_direction_count / comparable * 100
                divergence_labels = [
                    paired_labels[int(position)]
                    for position in paired_diffs.index[(p_sign != s_sign) & comparable_direction].tolist()
                    if 0 <= int(position) < len(paired_labels)
                ]
                dual.divergence_periods = divergence_labels
        if len(paired_ordered) >= 2:
            primary_indexed = _index_to_100(paired_ordered[spec.y])
            secondary_indexed = _index_to_100(paired_ordered[spec.secondary_y])
            distance = (primary_indexed - secondary_indexed).abs()
            dual.normalized_primary_values = {
                str(label): float(value)
                for label, value in zip(paired_labels, primary_indexed, strict=False)
                if pd.notna(value)
            }
            dual.normalized_secondary_values = {
                str(label): float(value)
                for label, value in zip(paired_labels, secondary_indexed, strict=False)
                if pd.notna(value)
            }
            if distance.notna().any():
                divergence_index = int(distance.idxmax())
                dual.largest_normalized_divergence_period = paired_labels[divergence_index]
                dual.largest_normalized_divergence_value = float(distance.loc[divergence_index])
        if axis_type == "datetime":
            missing, irregular = _missing_periods(pd.to_datetime(frame["_order"]), granularity)
            dual.missing_periods = _period_labels(pd.Series(missing), granularity)
            dual.irregular_intervals = irregular

    if dual.unit_relationship == "same unit":
        dual.warnings.append("Both metrics use the same unit, so a shared Y-axis would make absolute differences easier to compare.")
    elif dual.unit_relationship == "different unit":
        dual.warnings.append("The metrics use different Y-axis scales, so visual heights and slopes cannot be compared directly.")
    if dual.x_axis_type == "unordered categorical":
        dual.warnings.append("The x-axis contains categories, so connected lines show category ranking rather than a temporal trend.")
    if dual.x_axis_type in {"datetime", "numeric", "ordered_period"}:
        if dual.paired_point_count < 5 or dual.pearson_correlation is None:
            dual.evidence_strength = "low"
        elif dual.paired_point_count < 12 or dual.missing_periods or dual.irregular_intervals:
            dual.evidence_strength = "medium"
        else:
            dual.evidence_strength = "high"
    else:
        dual.evidence_strength = "high" if dual.category_count >= 5 and dual.pearson_correlation is not None else "medium"
        if dual.category_count < 3 or not dual.valid_primary_count or not dual.valid_secondary_count:
            dual.evidence_strength = "low"
    return dual


def _dual_combination_evidence(result: ChartResult, evidence: ChartEvidence) -> DualCombinationEvidence:
    rows = _records(result)
    spec = result.spec
    bar_aggregation = spec.primary_aggregation or spec.aggregation or "sum"
    line_aggregation = spec.secondary_aggregation or spec.aggregation or "sum"
    dual = DualCombinationEvidence(
        **evidence.model_dump(exclude={"chart_type", "x_column", "aggregation", "warnings"}),
        x_column=spec.x or "",
        bar_y_column=spec.y or "",
        bar_aggregation=bar_aggregation,
        line_y_column=spec.secondary_y or "",
        line_aggregation=line_aggregation,
        filters_applied=dict(evidence.filters),
        warnings=list(evidence.warnings),
    )
    if (
        spec.chart_type != "dual_axis"
        or not spec.x
        or not spec.y
        or not spec.secondary_y
        or not bar_aggregation
        or not line_aggregation
    ):
        dual.warnings.append("Dual-combination evidence requires one x-axis, two metric columns, and two aggregations.")
        dual.evidence_strength = "low"
        return dual
    frame = pd.DataFrame(rows)
    required = {spec.x, spec.y, spec.secondary_y}
    if not required.issubset(frame.columns):
        dual.warnings.append("The displayed dual-combination data is missing required columns.")
        dual.evidence_strength = "low"
        return dual
    frame = frame[[spec.x, spec.y, spec.secondary_y]].copy()
    frame[spec.y] = pd.to_numeric(frame[spec.y], errors="coerce")
    frame[spec.secondary_y] = pd.to_numeric(frame[spec.secondary_y], errors="coerce")
    order, granularity, axis_type, ordered = _area_axis_order(frame[spec.x], spec.x)
    dual.x_axis_type = axis_type if ordered else "unordered categorical"
    dual.time_granularity = granularity
    frame["_order"] = order
    if ordered:
        frame = frame.sort_values("_order").reset_index(drop=True)
    labels = (
        _period_labels(pd.to_datetime(frame[spec.x], errors="coerce"), granularity)
        if axis_type == "datetime" and ordered
        else frame[spec.x].astype(str).tolist()
    )

    dual.point_count = int(len(frame))
    dual.valid_rows = int(len(frame))
    dual.valid_bar_count = int(frame[spec.y].notna().sum())
    dual.valid_line_count = int(frame[spec.secondary_y].notna().sum())
    dual.missing_bar_count = int(frame[spec.y].isna().sum())
    dual.missing_line_count = int(frame[spec.secondary_y].isna().sum())
    dual.bar_unit = _metric_unit(spec.y)
    dual.line_unit = _metric_unit(spec.secondary_y)
    dual.unit_relationship = _unit_relationship(dual.bar_unit, dual.line_unit)
    dual.aggregation_relationship = _aggregation_relationship(bar_aggregation, line_aggregation)
    dual.known_metric_relationship, dual.derived_metric_available = _known_metric_relationship(spec.y, spec.secondary_y)
    dual.top_n_applied = spec.limit if spec.limit and len(frame) >= spec.limit else None

    dual.bar_values = {
        str(label): float(value)
        for label, value in zip(labels, frame[spec.y], strict=False)
        if pd.notna(value)
    }
    dual.line_values = {
        str(label): float(value)
        for label, value in zip(labels, frame[spec.secondary_y], strict=False)
        if pd.notna(value)
    }

    if dual.valid_bar_count:
        bar_high = frame[spec.y].idxmax()
        bar_low = frame[spec.y].idxmin()
        dual.bar_highest_x = labels[int(bar_high)]
        dual.bar_highest_value = float(frame.loc[bar_high, spec.y])
        dual.bar_lowest_x = labels[int(bar_low)]
        dual.bar_lowest_value = float(frame.loc[bar_low, spec.y])
        dual.bar_range = dual.bar_highest_value - dual.bar_lowest_value
        dual.bar_ranking = [
            labels[int(index)]
            for index in frame[spec.y].sort_values(ascending=False, na_position="last").dropna().index
        ]
    if dual.valid_line_count:
        line_high = frame[spec.secondary_y].idxmax()
        line_low = frame[spec.secondary_y].idxmin()
        dual.line_highest_x = labels[int(line_high)]
        dual.line_highest_value = float(frame.loc[line_high, spec.secondary_y])
        dual.line_lowest_x = labels[int(line_low)]
        dual.line_lowest_value = float(frame.loc[line_low, spec.secondary_y])
        dual.line_range = dual.line_highest_value - dual.line_lowest_value
        dual.line_ranking = [
            labels[int(index)]
            for index in frame[spec.secondary_y].sort_values(ascending=False, na_position="last").dropna().index
        ]
    dual.same_highest_x = dual.bar_highest_x == dual.line_highest_x if dual.bar_highest_x and dual.line_highest_x else None
    dual.same_lowest_x = dual.bar_lowest_x == dual.line_lowest_x if dual.bar_lowest_x and dual.line_lowest_x else None

    paired = frame[[spec.y, spec.secondary_y]].dropna()
    dual.paired_point_count = int(len(paired))
    if len(paired) >= 3 and paired[spec.y].nunique() > 1 and paired[spec.secondary_y].nunique() > 1:
        dual.pearson_correlation = float(paired[spec.y].corr(paired[spec.secondary_y]))
        dual.spearman_correlation = float(paired[spec.y].corr(paired[spec.secondary_y], method="spearman"))
        dual.relationship_strength = _relationship_strength(dual.pearson_correlation)
        dual.relationship_direction = "positive" if dual.pearson_correlation > 0 else "negative" if dual.pearson_correlation < 0 else "neutral"
    elif len(paired) < 5:
        dual.warnings.append("Fewer than five paired values are displayed, so relationship evidence is limited.")

    if len(paired) >= 2:
        paired_frame = frame[[spec.x, spec.y, spec.secondary_y]].dropna(subset=[spec.y, spec.secondary_y])
        paired_labels = [labels[int(index)] for index in paired_frame.index]
        bar_indexed = _index_to_100(paired_frame[spec.y])
        line_indexed = _index_to_100(paired_frame[spec.secondary_y])
        normalized_gap = bar_indexed - line_indexed
        dual.normalized_bar_values = {
            str(label): float(value)
            for label, value in zip(paired_labels, bar_indexed, strict=False)
            if pd.notna(value)
        }
        dual.normalized_line_values = {
            str(label): float(value)
            for label, value in zip(paired_labels, line_indexed, strict=False)
            if pd.notna(value)
        }
        if normalized_gap.notna().any():
            positive_index = int(normalized_gap.idxmax())
            negative_index = int(normalized_gap.idxmin())
            absolute_index = int(normalized_gap.abs().idxmax())
            dual.largest_positive_divergence_x = labels[positive_index]
            dual.largest_negative_divergence_x = labels[negative_index]
            dual.largest_normalized_divergence_x = labels[absolute_index]

    if ordered and axis_type in {"datetime", "numeric", "ordered_period"}:
        if dual.valid_bar_count >= 2:
            bar_series = frame[spec.y].dropna()
            dual.bar_start_x = labels[int(bar_series.index[0])]
            dual.bar_start_value = float(bar_series.iloc[0])
            dual.bar_end_x = labels[int(bar_series.index[-1])]
            dual.bar_end_value = float(bar_series.iloc[-1])
            dual.bar_change = dual.bar_end_value - dual.bar_start_value
            dual.bar_change_percent = dual.bar_change / abs(dual.bar_start_value) * 100 if dual.bar_start_value else None
            dual.bar_endpoint_direction = _endpoint_direction_label(dual.bar_change, dual.bar_start_value)
            dual.bar_peak_x = dual.bar_highest_x
            dual.bar_peak_value = dual.bar_highest_value
            dual.bar_trough_x = dual.bar_lowest_x
            dual.bar_trough_value = dual.bar_lowest_value
            _, dual.bar_volatility_level = _series_volatility(bar_series)
        if dual.valid_line_count >= 2:
            line_series = frame[spec.secondary_y].dropna()
            dual.line_start_x = labels[int(line_series.index[0])]
            dual.line_start_value = float(line_series.iloc[0])
            dual.line_end_x = labels[int(line_series.index[-1])]
            dual.line_end_value = float(line_series.iloc[-1])
            dual.line_change = dual.line_end_value - dual.line_start_value
            dual.line_change_percent = dual.line_change / abs(dual.line_start_value) * 100 if dual.line_start_value else None
            dual.line_endpoint_direction = _endpoint_direction_label(dual.line_change, dual.line_start_value)
            dual.line_peak_x = dual.line_highest_x
            dual.line_peak_value = dual.line_highest_value
            dual.line_trough_x = dual.line_lowest_x
            dual.line_trough_value = dual.line_lowest_value
            _, dual.line_volatility_level = _series_volatility(line_series)
        dual.peaks_aligned = dual.bar_peak_x == dual.line_peak_x if dual.bar_peak_x and dual.line_peak_x else None
        dual.troughs_aligned = dual.bar_trough_x == dual.line_trough_x if dual.bar_trough_x and dual.line_trough_x else None
        if dual.bar_volatility_level and dual.line_volatility_level:
            bar_cv, _ = _series_volatility(frame[spec.y])
            line_cv, _ = _series_volatility(frame[spec.secondary_y])
            if bar_cv is not None and line_cv is not None:
                if abs(bar_cv - line_cv) < 0.02:
                    dual.more_volatile_metric = "similar"
                elif bar_cv > line_cv:
                    dual.more_volatile_metric = display_name(spec.y)
                else:
                    dual.more_volatile_metric = display_name(spec.secondary_y)
        paired_ordered = frame[[spec.y, spec.secondary_y]].dropna()
        if len(paired_ordered) >= 2:
            diffs = paired_ordered.diff().dropna()
            comparable = int(len(diffs))
            dual.comparable_transition_count = comparable
            if comparable:
                bar_sign = np.sign(diffs[spec.y])
                line_sign = np.sign(diffs[spec.secondary_y])
                comparable_direction = (bar_sign != 0) & (line_sign != 0)
                dual.aligned_direction_count = int(((bar_sign == line_sign) & comparable_direction).sum())
                dual.opposite_direction_count = int(((bar_sign != line_sign) & comparable_direction).sum())
                dual.aligned_direction_percent = dual.aligned_direction_count / comparable * 100
                dual.opposite_direction_percent = dual.opposite_direction_count / comparable * 100
        if axis_type == "datetime":
            missing, irregular = _missing_periods(pd.to_datetime(frame["_order"]), granularity)
            dual.missing_x_values = _period_labels(pd.Series(missing), granularity)
            dual.irregular_intervals = irregular

    if dual.unit_relationship == "same unit":
        dual.warnings.append("Both metrics use the same unit, so a shared-scale comparison can be useful alongside the combination chart.")
    elif dual.unit_relationship == "different unit":
        dual.warnings.append("The bar and line use different Y-axis scales, so compare patterns and paired values rather than visual height.")
    if dual.aggregation_relationship != "same aggregation":
        dual.warnings.append(dual.aggregation_relationship)
    if dual.top_n_applied:
        dual.warnings.append(f"The chart is limited to the top {dual.top_n_applied} displayed x-values.")

    if dual.paired_point_count < 5 or dual.pearson_correlation is None:
        dual.evidence_strength = "low"
    elif dual.paired_point_count < 12 or dual.missing_x_values or dual.irregular_intervals:
        dual.evidence_strength = "medium"
    else:
        dual.evidence_strength = "high"
    return dual


def _dual_evidence(result: ChartResult, evidence: ChartEvidence) -> ChartEvidence:
    rows = _records(result)
    spec = result.spec
    frame = pd.DataFrame(rows)
    metrics = {}
    for column in (spec.y, spec.secondary_y):
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        first, final = float(values.iloc[0]), float(values.iloc[-1])
        metrics[column] = {
            "first_value": first,
            "final_value": final,
            "absolute_change": final - first,
            "percentage_change": (final - first) / abs(first) * 100 if first else None,
            "peak_label": frame.loc[values.idxmax(), spec.x],
            "peak_value": float(values.max()),
        }
    if spec.y and spec.secondary_y and spec.y in frame and spec.secondary_y in frame:
        pair = frame[[spec.y, spec.secondary_y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) >= 2:
            metrics["pearson_correlation"] = float(pair.corr().iloc[0, 1])
        primary_threshold = frame[spec.y].median()
        secondary_threshold = frame[spec.secondary_y].median()
        candidates = frame.loc[
            (frame[spec.y] >= primary_threshold)
            & (frame[spec.secondary_y] < secondary_threshold)
        ]
        metrics["high_primary_low_secondary_count"] = int(len(candidates))
        metrics["primary_threshold"] = float(primary_threshold)
        metrics["secondary_threshold"] = float(secondary_threshold)
        metrics["high_primary_low_secondary_labels"] = (
            candidates[spec.x].astype(str).tolist() if spec.x in candidates else []
        )
    evidence.calculated_metrics.update(metrics)
    if result.spec.chart_type == "dual_axis":
        evidence.warnings.append("Dual-axis charts can visually exaggerate relationships between differently scaled metrics.")
        evidence.warnings.append(
            f"Thresholds use medians: {_fmt(metrics.get('primary_threshold'), spec.y)} "
            f"for {display_name(spec.y)} and {_fmt(metrics.get('secondary_threshold'), spec.secondary_y)} "
            f"for {display_name(spec.secondary_y)}."
        )
    return evidence


def _scatter_evidence(result: ChartResult, evidence: ChartEvidence) -> ScatterEvidence:
    rows = _records(result)
    spec = result.spec
    scatter = ScatterEvidence(
        **evidence.model_dump(exclude={"chart_type", "x_column", "color_column", "size_column", "warnings"}),
        x_column=spec.x or "",
        y_column=spec.y or "",
        color_column=spec.color,
        size_column=None,
        filters_applied=dict(evidence.filters),
        warnings=list(evidence.warnings),
    )
    frame = pd.DataFrame(rows)
    if not spec.x or not spec.y or spec.x not in frame or spec.y not in frame:
        scatter.evidence_strength = "low"
        return scatter
    pair = _finite_pair(frame, spec.x, spec.y)
    scatter.raw_row_count = len(frame)
    scatter.valid_point_count = int(len(pair))
    scatter.displayed_point_count = int(len(pair))
    scatter.excluded_rows = len(frame) - len(pair)
    scatter.valid_rows = int(len(pair))
    scatter.top_n = None
    scatter.x_unit = _metric_unit(spec.x)
    scatter.y_unit = _metric_unit(spec.y)
    if pair.empty:
        scatter.warnings.append("No finite numeric scatter points were available.")
        scatter.evidence_strength = "low"
        return scatter
    scatter.x_min = float(pair[spec.x].min())
    scatter.x_max = float(pair[spec.x].max())
    scatter.x_mean = float(pair[spec.x].mean())
    scatter.x_median = float(pair[spec.x].median())
    scatter.y_min = float(pair[spec.y].min())
    scatter.y_max = float(pair[spec.y].max())
    scatter.y_mean = float(pair[spec.y].mean())
    scatter.y_median = float(pair[spec.y].median())
    scatter.known_metric_relationship, scatter.mathematical_dependency, scatter.derived_metric_available = _scatter_formula_dependency(frame, spec.x, spec.y)
    normalized_columns = {normalize_column_name(column): column for column in frame.columns}
    if {"unitssold", "unitprice", "totalrevenue"}.issubset(normalized_columns):
        expected = (
            pd.to_numeric(frame[normalized_columns["unitssold"]], errors="coerce")
            * pd.to_numeric(frame[normalized_columns["unitprice"]], errors="coerce")
        )
        actual = pd.to_numeric(frame[normalized_columns["totalrevenue"]], errors="coerce")
        matches = np.isclose(actual, expected, rtol=0.001, atol=0.01)
        scatter.calculated_metrics["revenue_formula_match_pct"] = float(np.nanmean(matches) * 100)
    if {"unitssold", "unitcost", "totalcost"}.issubset(normalized_columns):
        expected = (
            pd.to_numeric(frame[normalized_columns["unitssold"]], errors="coerce")
            * pd.to_numeric(frame[normalized_columns["unitcost"]], errors="coerce")
        )
        actual = pd.to_numeric(frame[normalized_columns["totalcost"]], errors="coerce")
        matches = np.isclose(actual, expected, rtol=0.001, atol=0.01)
        scatter.calculated_metrics["cost_formula_match_pct"] = float(np.nanmean(matches) * 100)
    if {"totalrevenue", "totalcost", "totalprofit"}.issubset(normalized_columns):
        expected = (
            pd.to_numeric(frame[normalized_columns["totalrevenue"]], errors="coerce")
            - pd.to_numeric(frame[normalized_columns["totalcost"]], errors="coerce")
        )
        actual = pd.to_numeric(frame[normalized_columns["totalprofit"]], errors="coerce")
        matches = np.isclose(actual, expected, rtol=0.001, atol=0.01)
        scatter.calculated_metrics["profit_formula_match_pct"] = float(np.nanmean(matches) * 100)
    if len(pair) >= 3 and pair[spec.x].nunique() > 1 and pair[spec.y].nunique() > 1:
        corr = float(pair[spec.x].corr(pair[spec.y]))
        spearman = float(pair[spec.x].corr(pair[spec.y], method="spearman"))
        scatter.pearson_correlation = corr if isfinite(corr) else None
        scatter.spearman_correlation = spearman if isfinite(spearman) else None
        if scatter.pearson_correlation is not None:
            scatter.relationship_strength = _relationship_strength(scatter.pearson_correlation)
            scatter.relationship_direction = (
                "positive" if scatter.pearson_correlation > 0
                else "negative" if scatter.pearson_correlation < 0
                else "none or unclear"
            )
            slope, intercept = np.polyfit(pair[spec.x], pair[spec.y], 1)
            predicted = slope * pair[spec.x] + intercept
            ss_res = float(((pair[spec.y] - predicted) ** 2).sum())
            ss_tot = float(((pair[spec.y] - pair[spec.y].mean()) ** 2).sum())
            scatter.linear_slope = float(slope)
            scatter.linear_intercept = float(intercept)
            scatter.r_squared = 1 - ss_res / ss_tot if ss_tot else None
    else:
        scatter.warnings.append("Scatter correlation requires at least three finite points and variation in both variables.")
    scatter.banding_detected, scatter.band_count = _detect_banding(pair, spec.x, spec.y)
    scatter.outlier_count, scatter.influential_point_count, scatter.outlier_indices = _detect_outliers(pair, spec.x, spec.y)
    scatter.heteroscedasticity_detected, scatter.variance_pattern = _detect_heteroscedasticity(pair, spec.x, spec.y)
    scatter.color_group_summary, scatter.group_relationships, scatter.cluster_count = _color_group_evidence(frame, pair, spec.x, spec.y, spec.color)
    scatter.relationship_form = _scatter_relationship_form(
        scatter.pearson_correlation,
        scatter.spearman_correlation,
        scatter.r_squared,
        scatter.banding_detected,
        scatter.heteroscedasticity_detected,
        scatter.cluster_count,
    )
    if scatter.banding_detected:
        scatter.detected_patterns.append("banded relationship")
    if scatter.heteroscedasticity_detected:
        scatter.detected_patterns.append("changing vertical spread")
    if scatter.outlier_count:
        scatter.detected_patterns.append("potential outliers")
    if scatter.mathematical_dependency:
        scatter.warnings.append(scatter.mathematical_dependency)
    if {normalize_column_name(spec.x), normalize_column_name(spec.y)} & {"unitssold", "totalrevenue"}:
        scatter.warnings.append("Revenue may be mathematically derived from units and price; avoid causal wording.")
    if scatter.valid_point_count < 30 or scatter.pearson_correlation is None:
        scatter.evidence_strength = "low"
    elif scatter.valid_point_count < 100 or scatter.influential_point_count:
        scatter.evidence_strength = "medium"
    elif scatter.mathematical_dependency:
        scatter.evidence_strength = "medium"
    else:
        scatter.evidence_strength = "high"
    scatter.calculated_metrics.update({
        "pearson_correlation": scatter.pearson_correlation,
        "spearman_correlation": scatter.spearman_correlation,
        "r_squared": scatter.r_squared,
        "relationship_form": scatter.relationship_form,
        "outlier_count": scatter.outlier_count,
        "banding_detected": scatter.banding_detected,
    })
    return scatter


def _scatter_like_evidence(result: ChartResult, evidence: ChartEvidence) -> ChartEvidence:
    rows = _records(result)
    spec = result.spec
    frame = pd.DataFrame(rows)
    if not spec.x or not spec.y or spec.x not in frame or spec.y not in frame:
        evidence.evidence_strength = "low"
        return evidence
    pair = frame[[spec.x, spec.y]].apply(pd.to_numeric, errors="coerce").dropna()
    evidence.excluded_rows = len(frame) - len(pair)
    evidence.valid_rows = len(pair)
    if len(pair) < 2:
        evidence.warnings.append("Too few valid points for relationship statistics.")
        evidence.evidence_strength = "low"
        return evidence
    corr = float(pair[spec.x].corr(pair[spec.y]))
    spearman = float(pair[spec.x].corr(pair[spec.y], method="spearman"))
    slope, intercept = np.polyfit(pair[spec.x], pair[spec.y], 1)
    predicted = slope * pair[spec.x] + intercept
    ss_res = float(((pair[spec.y] - predicted) ** 2).sum())
    ss_tot = float(((pair[spec.y] - pair[spec.y].mean()) ** 2).sum())
    r_squared = 1 - ss_res / ss_tot if ss_tot else 0.0
    evidence.calculated_metrics.update({
        "pearson_correlation": corr,
        "spearman_correlation": spearman,
        "regression_slope": float(slope),
        "r_squared": float(r_squared),
        "x_min": float(pair[spec.x].min()),
        "x_max": float(pair[spec.x].max()),
        "y_min": float(pair[spec.y].min()),
        "y_max": float(pair[spec.y].max()),
    })
    if abs(corr) >= 0.7:
        evidence.detected_patterns.append("strong association")
    if {normalize_column_name(spec.x), normalize_column_name(spec.y)} & {"unitssold", "totalrevenue"}:
        evidence.warnings.append("Revenue may be mathematically derived from units and price; avoid causal wording.")
    normalized_columns = {
        normalize_column_name(column): column for column in frame.columns
    }
    if {"unitssold", "unitprice", "totalrevenue"}.issubset(normalized_columns):
        expected = (
            pd.to_numeric(frame[normalized_columns["unitssold"]], errors="coerce")
            * pd.to_numeric(frame[normalized_columns["unitprice"]], errors="coerce")
        )
        actual = pd.to_numeric(frame[normalized_columns["totalrevenue"]], errors="coerce")
        matches = np.isclose(actual, expected, rtol=0.001, atol=0.01)
        evidence.calculated_metrics["revenue_formula_match_pct"] = float(np.nanmean(matches) * 100)
        evidence.detected_patterns.append("Total Revenue can be checked as Units Sold × Unit Price")
    if {"unitssold", "unitcost", "totalcost"}.issubset(normalized_columns):
        expected = (
            pd.to_numeric(frame[normalized_columns["unitssold"]], errors="coerce")
            * pd.to_numeric(frame[normalized_columns["unitcost"]], errors="coerce")
        )
        actual = pd.to_numeric(frame[normalized_columns["totalcost"]], errors="coerce")
        matches = np.isclose(actual, expected, rtol=0.001, atol=0.01)
        evidence.calculated_metrics["cost_formula_match_pct"] = float(np.nanmean(matches) * 100)
    if {"totalrevenue", "totalcost", "totalprofit"}.issubset(normalized_columns):
        expected = (
            pd.to_numeric(frame[normalized_columns["totalrevenue"]], errors="coerce")
            - pd.to_numeric(frame[normalized_columns["totalcost"]], errors="coerce")
        )
        actual = pd.to_numeric(frame[normalized_columns["totalprofit"]], errors="coerce")
        matches = np.isclose(actual, expected, rtol=0.001, atol=0.01)
        evidence.calculated_metrics["profit_formula_match_pct"] = float(np.nanmean(matches) * 100)
    return evidence


def _distribution_evidence(result: ChartResult, evidence: ChartEvidence) -> ChartEvidence:
    spec = result.spec
    column = spec.x if spec.chart_type == "histogram" else spec.y
    values = _numeric_series(_records(result), column)
    if values.empty:
        evidence.evidence_strength = "low"
        evidence.warnings.append("No numeric values were available for distribution statistics.")
        return evidence
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    iqr = float(q3 - q1)
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = values[(values < lower) | (values > upper)]
    evidence.calculated_metrics.update({
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(median),
        "standard_deviation": float(values.std()) if len(values) > 1 else 0.0,
        "variance": float(values.var()) if len(values) > 1 else 0.0,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": iqr,
        "skewness": float(values.skew()) if len(values) > 2 else 0.0,
        "zero_count": int((values == 0).sum()),
        "negative_count": int((values < 0).sum()),
        "outlier_count": int(len(outliers)),
    })
    skew = evidence.calculated_metrics["skewness"]
    if skew > 1:
        evidence.detected_patterns.append("right-skewed distribution")
    elif skew < -1:
        evidence.detected_patterns.append("left-skewed distribution")
    return evidence


def _histogram_evidence(result: ChartResult, evidence: ChartEvidence) -> HistogramEvidence:
    spec = result.spec
    rows = _records(result)
    column = spec.x or ""
    hist = HistogramEvidence(
        **evidence.model_dump(exclude={"chart_type", "x_column", "color_column", "warnings"}),
        value_column=column,
        color_column=spec.color,
        unit=_metric_unit(column),
        filters_applied=dict(evidence.filters),
        warnings=list(evidence.warnings),
    )
    frame = pd.DataFrame(rows)
    if spec.chart_type != "histogram" or not column or column not in frame:
        hist.warnings.append("Histogram evidence requires one numeric value column.")
        hist.evidence_strength = "low"
        return hist
    raw = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = raw.dropna()
    hist.raw_row_count = len(frame)
    hist.valid_value_count = int(len(values))
    hist.displayed_value_count = int(len(values))
    hist.missing_value_count = int(raw.isna().sum())
    hist.excluded_value_count = hist.missing_value_count
    hist.valid_rows = int(len(values))
    hist.excluded_rows = hist.excluded_value_count
    hist.top_n = None
    if values.empty:
        hist.warnings.append("No finite numeric values were available for histogram evidence.")
        hist.evidence_strength = "low"
        return hist

    hist.minimum = float(values.min())
    hist.maximum = float(values.max())
    hist.value_range = hist.maximum - hist.minimum
    hist.mean = float(values.mean())
    hist.median = float(values.median())
    hist.standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else None
    hist.q1 = float(values.quantile(0.25))
    hist.q3 = float(values.quantile(0.75))
    hist.iqr = hist.q3 - hist.q1
    hist.p05 = float(values.quantile(0.05))
    hist.p10 = float(values.quantile(0.10))
    hist.p90 = float(values.quantile(0.90))
    hist.p95 = float(values.quantile(0.95))
    hist.skewness = float(values.skew()) if len(values) > 2 and values.nunique() > 1 else 0.0
    hist.skew_direction, hist.skew_strength = _histogram_skew_labels(hist.skewness)
    hist.kurtosis = float(values.kurtosis()) if len(values) > 3 and values.nunique() > 1 else None
    if hist.skew_direction == "right-skewed":
        hist.tail_description = "long upper tail"
    elif hist.skew_direction == "left-skewed":
        hist.tail_description = "long lower tail"
    else:
        hist.tail_description = "balanced tails"
    hist.zero_count = int((values == 0).sum())
    hist.zero_share = hist.zero_count / len(values) * 100
    hist.negative_count = int((values < 0).sum())
    hist.negative_share = hist.negative_count / len(values) * 100
    hist.lower_half_share = float((values <= hist.median).mean() * 100) if hist.median is not None else None
    hist.upper_tail_share = float((values >= hist.p90).mean() * 100) if hist.p90 is not None else None

    hist.bin_edges, hist.bin_counts, hist.bin_method = _histogram_bin_summary(values)
    hist.bin_count = len(hist.bin_counts)
    hist.bin_width = hist.bin_edges[1] - hist.bin_edges[0] if len(hist.bin_edges) >= 2 else None
    if hist.bin_counts:
        modal_index = int(np.argmax(hist.bin_counts))
        hist.modal_bin_start = hist.bin_edges[modal_index]
        hist.modal_bin_end = hist.bin_edges[modal_index + 1]
        hist.modal_bin_count = hist.bin_counts[modal_index]
        hist.modal_bin_share = hist.modal_bin_count / len(values) * 100
    hist.multimodal, hist.mode_count_estimate, hist.multimodal_evidence = _histogram_multimodal(hist.bin_counts, hist.bin_edges)
    if not hist.multimodal and len(values) >= 6 and values.nunique() >= 4:
        unique_values = pd.Series(sorted(values.unique()))
        gaps = unique_values.diff().dropna()
        positive_gaps = gaps[gaps > 0]
        if not positive_gaps.empty:
            median_gap = float(positive_gaps.median())
            largest_gap = float(positive_gaps.max())
            if median_gap and largest_gap >= median_gap * 4:
                hist.multimodal = True
                hist.mode_count_estimate = 2
                hist.multimodal_evidence = "The values separate into at least two distinct ranges with a large gap between them."

    hist.outlier_method = "1.5 x IQR rule"
    hist.lower_outlier_threshold = hist.q1 - 1.5 * hist.iqr
    hist.upper_outlier_threshold = hist.q3 + 1.5 * hist.iqr
    lower_outliers = values < hist.lower_outlier_threshold
    upper_outliers = values > hist.upper_outlier_threshold
    hist.lower_outlier_count = int(lower_outliers.sum())
    hist.upper_outlier_count = int(upper_outliers.sum())
    hist.potential_outlier_count = hist.lower_outlier_count + hist.upper_outlier_count
    hist.potential_outlier_share = hist.potential_outlier_count / len(values) * 100

    value_frame = frame.loc[values.index].copy()
    value_frame[column] = values
    hist.group_summary = _histogram_group_summary(value_frame, column, spec.color)

    hist.calculated_metrics.update({
        "count": hist.displayed_value_count,
        "mean": hist.mean,
        "median": hist.median,
        "standard_deviation": hist.standard_deviation,
        "minimum": hist.minimum,
        "maximum": hist.maximum,
        "q1": hist.q1,
        "q3": hist.q3,
        "iqr": hist.iqr,
        "skewness": hist.skewness,
        "zero_count": hist.zero_count,
        "negative_count": hist.negative_count,
        "outlier_count": hist.potential_outlier_count,
    })
    if hist.skew_direction:
        hist.detected_patterns.append(hist.skew_direction)
    if hist.multimodal:
        hist.detected_patterns.append("possible multimodality")
    if hist.zero_share and hist.zero_share >= 20:
        hist.detected_patterns.append("zero-heavy distribution")
    if hist.negative_count:
        hist.detected_patterns.append("contains negative values")
    hist.warnings.append("Histogram shape depends partly on the selected bin width; different bins may reveal or hide smaller peaks and gaps.")
    missing_share = hist.excluded_value_count / max(hist.raw_row_count, 1) * 100
    if hist.displayed_value_count < 30 or values.nunique() <= 1 or missing_share >= 40:
        hist.evidence_strength = "low"
    elif hist.displayed_value_count < 100 or hist.multimodal or missing_share >= 10:
        hist.evidence_strength = "medium"
    else:
        hist.evidence_strength = "high"
    return hist


def _box_group_evidence(values: pd.Series, x_value: str, breakdown_value: str | None) -> BoxGroupEvidence:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    display_label = f"{x_value} / {breakdown_value}" if breakdown_value is not None else x_value
    if values.empty:
        return BoxGroupEvidence(x_value=x_value, breakdown_value=breakdown_value, display_label=display_label, observation_count=0)
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    lower_threshold = q1 - 1.5 * iqr
    upper_threshold = q3 + 1.5 * iqr
    lower_outliers = values < lower_threshold
    upper_outliers = values > upper_threshold
    in_whisker = values.loc[~(lower_outliers | upper_outliers)]
    skew_direction, _ = _histogram_skew_labels(float(values.skew()) if len(values) > 2 and values.nunique() > 1 else 0.0)
    return BoxGroupEvidence(
        x_value=x_value,
        breakdown_value=breakdown_value,
        display_label=display_label,
        observation_count=int(len(values)),
        minimum=float(values.min()),
        maximum=float(values.max()),
        mean=float(values.mean()),
        median=float(values.median()),
        q1=q1,
        q3=q3,
        iqr=float(iqr),
        lower_whisker=float(in_whisker.min()) if not in_whisker.empty else None,
        upper_whisker=float(in_whisker.max()) if not in_whisker.empty else None,
        lower_outlier_count=int(lower_outliers.sum()),
        upper_outlier_count=int(upper_outliers.sum()),
        potential_outlier_count=int(lower_outliers.sum() + upper_outliers.sum()),
        potential_outlier_share=float((lower_outliers.sum() + upper_outliers.sum()) / len(values) * 100),
        skew_direction=skew_direction,
    )


def _box_plot_evidence(result: ChartResult, evidence: ChartEvidence) -> BoxPlotEvidence:
    spec = result.spec
    rows = _records(result)
    box = BoxPlotEvidence(
        **evidence.model_dump(exclude={"chart_type", "x_column", "color_column", "warnings"}),
        x_column=spec.x or "",
        y_column=spec.y or "",
        breakdown_column=spec.color,
        filters_applied=dict(evidence.filters),
        warnings=list(evidence.warnings),
    )
    frame = pd.DataFrame(rows)
    if spec.chart_type != "box" or not spec.x or not spec.y or spec.x not in frame or spec.y not in frame:
        box.warnings.append("Box Plot evidence requires a categorical x-axis and numeric y-axis.")
        box.evidence_strength = "low"
        return box
    frame = frame.copy()
    frame[spec.y] = pd.to_numeric(frame[spec.y], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[spec.x, spec.y])
    if frame.empty:
        box.warnings.append("No finite numeric values were available for Box Plot evidence.")
        box.evidence_strength = "low"
        return box
    box.valid_rows = int(len(frame))
    box.excluded_rows = len(rows) - len(frame)
    box.x_category_count = int(frame[spec.x].nunique(dropna=False))
    box.breakdown_category_count = int(frame[spec.color].nunique(dropna=False)) if spec.color and spec.color in frame else 0
    group_columns = [spec.x] + ([spec.color] if spec.color and spec.color in frame else [])
    groups: list[BoxGroupEvidence] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        x_value = friendly_value(spec.x, keys[0])
        breakdown_value = friendly_value(spec.color, keys[1]) if len(keys) > 1 and spec.color else None
        groups.append(_box_group_evidence(group[spec.y], x_value, breakdown_value))
    box.groups = groups
    box.box_count = len(groups)
    valid_groups = [group for group in groups if group.median is not None]
    if valid_groups:
        highest = max(valid_groups, key=lambda group: group.median if group.median is not None else -np.inf)
        lowest = min(valid_groups, key=lambda group: group.median if group.median is not None else np.inf)
        widest = max(valid_groups, key=lambda group: group.iqr if group.iqr is not None else -np.inf)
        narrowest = min(valid_groups, key=lambda group: group.iqr if group.iqr is not None else np.inf)
        box.highest_median_combination = highest.display_label
        box.highest_median_value = highest.median
        box.lowest_median_combination = lowest.display_label
        box.lowest_median_value = lowest.median
        box.widest_iqr_combination = widest.display_label
        box.widest_iqr_value = widest.iqr
        box.narrowest_iqr_combination = narrowest.display_label
        box.narrowest_iqr_value = narrowest.iqr
    if spec.color and spec.color in frame:
        leaders = {}
        laggards = {}
        gaps = {}
        lead_counts: dict[str, int] = {}
        ranking_signatures = set()
        for x_value, group_items in pd.DataFrame([group.model_dump() for group in groups]).groupby("x_value", dropna=False):
            ranked = group_items.dropna(subset=["median"]).sort_values("median", ascending=False)
            if ranked.empty:
                continue
            leader = str(ranked.iloc[0]["breakdown_value"])
            laggard = str(ranked.iloc[-1]["breakdown_value"])
            leaders[str(x_value)] = leader
            laggards[str(x_value)] = laggard
            gaps[str(x_value)] = float(ranked.iloc[0]["median"] - ranked.iloc[-1]["median"])
            lead_counts[leader] = lead_counts.get(leader, 0) + 1
            ranking_signatures.add(tuple(ranked["breakdown_value"].astype(str).tolist()))
        box.breakdown_leader_by_x = leaders
        box.breakdown_laggard_by_x = laggards
        box.breakdown_median_gap_by_x = gaps
        box.breakdown_lead_counts = lead_counts
        if len(ranking_signatures) > 1:
            box.x_categories_with_ranking_changes = list(leaders)
    box.total_potential_outlier_count = sum(group.potential_outlier_count for group in groups)
    box.groups_with_outliers = [group.display_label for group in groups if group.potential_outlier_count]
    counts = [group.observation_count for group in groups if group.observation_count]
    box.unequal_sample_sizes = bool(counts and max(counts) / max(min(counts), 1) >= 2)
    if box.unequal_sample_sizes:
        box.warnings.append("Some X-breakdown combinations contain fewer observations, so their medians and quartiles are less stable.")
    if box.total_potential_outlier_count:
        box.warnings.append("Potential outliers are identified by the 1.5 x IQR rule and may still be valid observations.")
    box.calculated_metrics.update({
        "box_count": box.box_count,
        "highest_median_combination": box.highest_median_combination,
        "highest_median_value": box.highest_median_value,
        "widest_iqr_combination": box.widest_iqr_combination,
        "widest_iqr_value": box.widest_iqr_value,
    })
    if box.valid_rows < 30 or box.box_count == 0:
        box.evidence_strength = "low"
    elif box.unequal_sample_sizes or box.valid_rows < 100:
        box.evidence_strength = "medium"
    else:
        box.evidence_strength = "high"
    return box


def _correlation_strength(correlation: float) -> str:
    absolute = abs(float(correlation))
    for threshold, label in CORRELATION_STRENGTH_THRESHOLDS:
        if absolute < threshold:
            return label
    return "near-perfect"


def _correlation_direction(correlation: float) -> str:
    return "positive" if correlation > 0 else "negative" if correlation < 0 else "neutral"


def _is_identifier_like_series(series: pd.Series, column: str, row_count: int) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    normalized = re.sub(r"[^a-z0-9]+", "", column.lower())
    name_hint = (
        bool(ID_NAME_PATTERN.search(column.replace(".", "_")))
        or normalized.endswith(("id", "key", "code", "index", "identifier"))
        or normalized.startswith(("id", "key", "index"))
    )
    unique_ratio = non_null.nunique(dropna=True) / max(len(non_null), 1)
    numeric = pd.to_numeric(non_null, errors="coerce").dropna()
    sequential = False
    if len(numeric) >= 3 and numeric.nunique(dropna=True) == len(numeric):
        unique = np.sort(numeric.unique().astype(float))
        diffs = np.diff(unique)
        sequential = bool(len(diffs) and np.allclose(diffs, diffs[0]) and diffs[0] != 0)
    return bool(name_hint or (row_count >= 10 and unique_ratio >= 0.98 and sequential))


def _known_correlation_relationship(x: str, y: str, columns: list[str]) -> tuple[str | None, bool]:
    normalized = {normalize_column_name(column): column for column in columns}
    pair = {normalize_column_name(x), normalize_column_name(y)}
    has = normalized.__contains__
    if pair == {"totalrevenue", "totalcost"} and has("unitssold"):
        return "Total Revenue and Total Cost are formula-derived totals that share Units Sold as an input.", True
    if pair == {"unitprice", "unitcost"}:
        return "Unit Price and Unit Cost are pricing inputs; their difference can define unit margin.", False
    if pair == {"totalrevenue", "totalprofit"}:
        return "Total Profit is calculated from Total Revenue and Total Cost, so this relationship is partly structural.", True
    if pair == {"totalcost", "totalprofit"}:
        return "Total Profit is calculated from Total Revenue and Total Cost, so this relationship is partly structural.", True
    if pair == {"unitprice", "totalrevenue"} and has("unitssold"):
        return "Total Revenue depends on Units Sold and Unit Price.", True
    if pair == {"unitcost", "totalcost"} and has("unitssold"):
        return "Total Cost depends on Units Sold and Unit Cost.", True
    return None, False


def _correlation_clusters(pairs: list[CorrelationPairEvidence]) -> list[dict[str, Any]]:
    graph: dict[str, set[str]] = {}
    for pair in pairs:
        if pair.absolute_correlation >= CORRELATION_CLUSTER_THRESHOLD:
            graph.setdefault(pair.variable_x, set()).add(pair.variable_y)
            graph.setdefault(pair.variable_y, set()).add(pair.variable_x)
    clusters = []
    seen: set[str] = set()
    for node in graph:
        if node in seen:
            continue
        stack = [node]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(graph.get(current, set()) - component)
        seen.update(component)
        if len(component) >= 3:
            cluster_pairs = [
                pair for pair in pairs
                if pair.variable_x in component and pair.variable_y in component
                and pair.absolute_correlation >= CORRELATION_CLUSTER_THRESHOLD
            ]
            clusters.append({
                "variables": sorted(component),
                "pair_count": len(cluster_pairs),
                "threshold": CORRELATION_CLUSTER_THRESHOLD,
                "strongest_pair": cluster_pairs[0].model_dump() if cluster_pairs else None,
            })
    return clusters


def _heatmap_evidence(result: ChartResult, evidence: ChartEvidence) -> ChartEvidence:
    if result.spec.chart_type != "heatmap" or result.metadata.get("heatmap_type") != "correlation":
        return evidence
    rows = _records(result)
    matrix = pd.DataFrame(rows)
    if matrix.empty or "index" not in matrix:
        evidence.evidence_strength = "low"
        evidence.warnings.append("Correlation Heatmap evidence requires a displayed correlation matrix.")
        return evidence

    selected_columns = [str(column) for column in result.metadata.get("selected_columns") or matrix["index"].astype(str).tolist()]
    corr = matrix.set_index("index")
    corr.index = corr.index.astype(str)
    corr = corr.reindex(index=selected_columns, columns=selected_columns)
    raw_numeric = pd.DataFrame(result.metadata.get("numeric_data") or {})
    if not raw_numeric.empty:
        raw_numeric = raw_numeric.reindex(columns=selected_columns)

    heatmap = CorrelationHeatmapEvidence(
        chart_title=evidence.chart_title,
        total_rows=evidence.total_rows,
        valid_rows=evidence.valid_rows,
        excluded_rows=evidence.excluded_rows,
        filters=evidence.filters,
        correlation_method=str(result.metadata.get("correlation_method") or "pearson"),
        selected_columns=selected_columns,
        displayed_variable_count=len(selected_columns),
        raw_row_count=int(result.metadata.get("raw_row_count") or evidence.total_rows),
        filtered_row_count=int(result.metadata.get("filtered_row_count") or evidence.valid_rows),
        filters_applied=evidence.filters,
    )

    pairs: list[CorrelationPairEvidence] = []
    pair_counts = []
    for i, column_x in enumerate(selected_columns):
        for column_y in selected_columns[i + 1:]:
            value = corr.loc[column_x, column_y] if column_x in corr.index and column_y in corr.columns else np.nan
            if not isinstance(value, (int, float, np.integer, np.floating)) or not isfinite(float(value)):
                continue
            paired_count = None
            missing_count = None
            if not raw_numeric.empty and column_x in raw_numeric and column_y in raw_numeric:
                paired = raw_numeric[[column_x, column_y]].dropna()
                paired_count = int(len(paired))
                missing_count = int(len(raw_numeric) - paired_count)
                pair_counts.append(paired_count)
            relationship, formula_derived = _known_correlation_relationship(column_x, column_y, selected_columns)
            pairs.append(CorrelationPairEvidence(
                variable_x=column_x,
                variable_y=column_y,
                correlation=float(value),
                absolute_correlation=abs(float(value)),
                direction=_correlation_direction(float(value)),
                strength=_correlation_strength(float(value)),
                paired_observation_count=paired_count,
                missing_pair_count=missing_count,
                known_relationship=relationship,
                formula_derived=formula_derived,
            ))

    pairs = sorted(pairs, key=lambda item: item.absolute_correlation, reverse=True)
    heatmap.pairs = pairs
    heatmap.unique_pair_count = len(pairs)
    positives = [pair for pair in pairs if pair.correlation > 0]
    negatives = [pair for pair in pairs if pair.correlation < 0]
    heatmap.strongest_positive_pair = max(positives, key=lambda item: item.correlation) if positives else None
    heatmap.strongest_negative_pair = min(negatives, key=lambda item: item.correlation) if negatives else None
    non_formula = [pair for pair in pairs if not pair.formula_derived]
    strongest = pairs[:4]
    if non_formula and all(pair.formula_derived for pair in strongest):
        strongest = strongest[:3] + [non_formula[0]]
    heatmap.strongest_pairs = strongest[:5]
    heatmap.strong_positive_pairs = [pair for pair in pairs if pair.correlation >= 0.70]
    heatmap.strong_negative_pairs = [pair for pair in pairs if pair.correlation <= -0.70]
    heatmap.moderate_pairs = [pair for pair in pairs if 0.30 <= pair.absolute_correlation < 0.50]
    heatmap.weak_pairs = [pair for pair in pairs if 0.10 <= pair.absolute_correlation < 0.30]
    heatmap.near_zero_pairs = [pair for pair in pairs if pair.absolute_correlation < 0.10]
    heatmap.high_multicollinearity_pairs = [
        pair for pair in pairs if pair.absolute_correlation >= HIGH_MULTICOLLINEARITY_THRESHOLD
    ]
    heatmap.formula_relationships = [
        {
            "variables": [pair.variable_x, pair.variable_y],
            "correlation": pair.correlation,
            "relationship": pair.known_relationship,
        }
        for pair in pairs if pair.formula_derived and pair.known_relationship
    ]
    heatmap.correlation_clusters = _correlation_clusters(pairs)

    if not raw_numeric.empty:
        heatmap.identifier_like_columns = [
            column for column in selected_columns
            if column in raw_numeric and _is_identifier_like_series(raw_numeric[column], column, len(raw_numeric))
        ]
        heatmap.constant_columns = [
            column for column in selected_columns
            if column in raw_numeric and raw_numeric[column].dropna().nunique(dropna=True) <= 1
        ]
        heatmap.near_constant_columns = [
            column for column in selected_columns
            if column not in heatmap.constant_columns
            and column in raw_numeric
            and raw_numeric[column].dropna().nunique(dropna=True) <= max(2, int(len(raw_numeric[column].dropna()) * 0.02))
        ]
    if pair_counts:
        heatmap.minimum_pairwise_count = min(pair_counts)
        heatmap.maximum_pairwise_count = max(pair_counts)
        heatmap.unequal_pairwise_counts = heatmap.minimum_pairwise_count != heatmap.maximum_pairwise_count

    if heatmap.identifier_like_columns:
        heatmap.warnings.append("Identifier-like numeric fields are analytically weak correlation inputs.")
    if heatmap.constant_columns:
        heatmap.warnings.append("One or more selected fields have no meaningful variation, so their correlations are undefined.")
    if heatmap.unequal_pairwise_counts:
        heatmap.warnings.append("Pairwise correlations use different observation counts because missing values vary across fields.")
    if heatmap.formula_relationships:
        heatmap.warnings.append("Formula-derived fields can create high structural correlations.")
    if heatmap.high_multicollinearity_pairs:
        heatmap.warnings.append("Highly correlated predictors may create multicollinearity in statistical or machine-learning models.")

    heatmap.calculated_metrics.update({
        "correlation_method": heatmap.correlation_method,
        "unique_pair_count": heatmap.unique_pair_count,
        "strongest_positive_pair": heatmap.strongest_positive_pair.model_dump() if heatmap.strongest_positive_pair else None,
        "strongest_negative_pair": heatmap.strongest_negative_pair.model_dump() if heatmap.strongest_negative_pair else None,
        "strongest_absolute_pair": heatmap.strongest_pairs[0].model_dump() if heatmap.strongest_pairs else None,
        "high_multicollinearity_pair_count": len(heatmap.high_multicollinearity_pairs),
    })
    if not pairs or heatmap.displayed_variable_count < 2:
        heatmap.evidence_strength = "low"
    elif heatmap.unequal_pairwise_counts or heatmap.constant_columns or heatmap.filtered_row_count < 30:
        heatmap.evidence_strength = "medium"
    else:
        heatmap.evidence_strength = "high"
    return heatmap


def _gantt_evidence(result: ChartResult, evidence: ChartEvidence) -> ChartEvidence:
    spec = result.spec
    frame = pd.DataFrame(_records(result))
    starts = pd.to_datetime(frame.get(spec.x), errors="coerce")
    ends = pd.to_datetime(frame.get(spec.secondary_y), errors="coerce")
    durations = (ends - starts).dt.total_seconds() / 86400
    valid = frame.assign(_start=starts, _end=ends, _duration=durations).dropna(subset=["_start", "_end", "_duration"])
    if valid.empty:
        evidence.evidence_strength = "low"
        return evidence
    longest = valid.loc[valid["_duration"].idxmax()]
    shortest = valid.loc[valid["_duration"].idxmin()]
    overlap_count = 0
    intervals = list(zip(valid["_start"], valid["_end"]))
    for index, (start, end) in enumerate(intervals):
        if any(index != other and start < other_end and end > other_start for other, (other_start, other_end) in enumerate(intervals)):
            overlap_count += 1
    evidence.calculated_metrics.update({
        "task_count": int(len(valid)),
        "earliest_start": valid["_start"].min(),
        "latest_finish": valid["_end"].max(),
        "project_span_days": float((valid["_end"].max() - valid["_start"].min()).days),
        "longest_task": longest[spec.y],
        "longest_task_days": float(longest["_duration"]),
        "shortest_task": shortest[spec.y],
        "shortest_task_days": float(shortest["_duration"]),
        "overlapping_task_count": int(overlap_count),
    })
    return evidence


def _bullet_evidence(result: ChartResult, evidence: ChartEvidence) -> ChartEvidence:
    spec = result.spec
    rows = _records(result)
    gaps = []
    for row in rows:
        actual = float(row.get(spec.y, 0))
        target = float(row.get(spec.secondary_y, 0))
        gap = actual - target
        gaps.append({
            "label": row.get(spec.x),
            "actual": actual,
            "target": target,
            "absolute_variance": gap,
            "percentage_variance": gap / target * 100 if target else None,
            "target_met": gap >= 0,
        })
    evidence.calculated_metrics["target_results"] = gaps
    evidence.calculated_metrics["targets_met"] = sum(1 for item in gaps if item["target_met"])
    evidence.calculated_metrics["target_count"] = len(gaps)
    evidence.calculated_metrics["actual_total"] = sum(item["actual"] for item in gaps)
    evidence.calculated_metrics["target_total"] = sum(item["target"] for item in gaps)
    evidence.calculated_metrics["total_variance"] = (
        evidence.calculated_metrics["actual_total"] - evidence.calculated_metrics["target_total"]
    )
    shortfalls = [item for item in gaps if item["absolute_variance"] < 0]
    surpluses = [item for item in gaps if item["absolute_variance"] >= 0]
    evidence.calculated_metrics["largest_shortfall"] = min(
        shortfalls,
        key=lambda item: item["absolute_variance"],
        default=None,
    )
    evidence.calculated_metrics["largest_surplus"] = max(
        surpluses,
        key=lambda item: item["absolute_variance"],
        default=None,
    )
    return evidence


def _circle_evidence(result: ChartResult, evidence: ChartEvidence) -> CircleViewEvidence:
    spec = result.spec
    rows = _records(result)
    circle = CircleViewEvidence(
        **evidence.model_dump(exclude={"chart_type", "x_column", "color_column", "size_column", "warnings"}),
        x_column=spec.x or "",
        y_column=spec.y or "",
        size_column=spec.secondary_y or "",
        color_column=spec.color,
        filters_applied=dict(evidence.filters),
        warnings=list(evidence.warnings),
    )
    frame = pd.DataFrame(rows)
    if (
        spec.chart_type != "circle_view"
        or not spec.x
        or not spec.y
        or not spec.secondary_y
        or spec.x not in frame
        or spec.y not in frame
        or spec.secondary_y not in frame
    ):
        circle.warnings.append("Circle View evidence requires numeric x, y, and size fields.")
        circle.evidence_strength = "low"
        return circle
    numeric = frame[[spec.x, spec.y, spec.secondary_y]].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric[spec.x]) & np.isfinite(numeric[spec.y]) & np.isfinite(numeric[spec.secondary_y])
    valid = frame.loc[finite].copy()
    valid[spec.x] = numeric.loc[finite, spec.x]
    valid[spec.y] = numeric.loc[finite, spec.y]
    valid[spec.secondary_y] = numeric.loc[finite, spec.secondary_y]
    valid = valid.loc[valid[spec.secondary_y] >= 0]
    circle.raw_row_count = len(frame)
    circle.valid_point_count = int(len(valid))
    circle.displayed_point_count = int(len(valid))
    circle.valid_rows = int(len(valid))
    circle.excluded_rows = len(frame) - len(valid)
    circle.top_n = None
    circle.x_unit = _metric_unit(spec.x)
    circle.y_unit = _metric_unit(spec.y)
    circle.size_unit = _metric_unit(spec.secondary_y)
    if valid.empty:
        circle.warnings.append("No finite non-negative Circle View records were available.")
        circle.evidence_strength = "low"
        return circle

    circle.x_min = float(valid[spec.x].min())
    circle.x_max = float(valid[spec.x].max())
    circle.x_mean = float(valid[spec.x].mean())
    circle.x_median = float(valid[spec.x].median())
    circle.y_min = float(valid[spec.y].min())
    circle.y_max = float(valid[spec.y].max())
    circle.y_mean = float(valid[spec.y].mean())
    circle.y_median = float(valid[spec.y].median())
    circle.size_min = float(valid[spec.secondary_y].min())
    circle.size_max = float(valid[spec.secondary_y].max())
    circle.size_mean = float(valid[spec.secondary_y].mean())
    circle.size_median = float(valid[spec.secondary_y].median())

    largest_index = valid[spec.secondary_y].idxmax()
    largest = valid.loc[largest_index]
    circle.largest_bubble_index = int(largest_index) if isinstance(largest_index, (int, np.integer)) else None
    circle.largest_bubble_x = float(largest[spec.x])
    circle.largest_bubble_y = float(largest[spec.y])
    circle.largest_bubble_size = float(largest[spec.secondary_y])
    circle.largest_bubble_group = str(largest[spec.color]) if spec.color and spec.color in valid else None
    circle.top_size_observations = [
        {
            "x": float(row[spec.x]),
            "y": float(row[spec.y]),
            "size": float(row[spec.secondary_y]),
            **({"group": str(row[spec.color])} if spec.color and spec.color in valid else {}),
        }
        for _, row in valid.sort_values(spec.secondary_y, ascending=False).head(3).iterrows()
    ]

    circle.pearson_xy, circle.spearman_xy, circle.xy_relationship_strength, circle.xy_relationship_direction = _correlation_summary(valid, spec.x, spec.y)
    if circle.pearson_xy is not None and valid[spec.x].nunique() > 1 and valid[spec.y].nunique() > 1:
        slope, intercept = np.polyfit(valid[spec.x], valid[spec.y], 1)
        predicted = slope * valid[spec.x] + intercept
        ss_res = float(((valid[spec.y] - predicted) ** 2).sum())
        ss_tot = float(((valid[spec.y] - valid[spec.y].mean()) ** 2).sum())
        circle.r_squared_xy = 1 - ss_res / ss_tot if ss_tot else None
    circle.pearson_size_x, circle.spearman_size_x, circle.size_x_relationship_strength, circle.size_x_relationship_direction = _correlation_summary(valid, spec.x, spec.secondary_y)
    circle.pearson_size_y, circle.spearman_size_y, circle.size_y_relationship_strength, circle.size_y_relationship_direction = _correlation_summary(valid, spec.y, spec.secondary_y)
    circle.banding_detected, circle.band_count = _detect_banding(valid[[spec.x, spec.y]], spec.x, spec.y)
    circle.outlier_count, circle.influential_point_count, outlier_indices = _detect_outliers(valid[[spec.x, spec.y]], spec.x, spec.y)
    circle.outlier_summary = [
        {"index": int(index), "x": float(valid.loc[index, spec.x]), "y": float(valid.loc[index, spec.y]), "size": float(valid.loc[index, spec.secondary_y])}
        for index in outlier_indices
        if index in valid.index
    ]
    circle.heteroscedasticity_detected, _ = _detect_heteroscedasticity(valid[[spec.x, spec.y]], spec.x, spec.y)
    circle.xy_relationship_form = _scatter_relationship_form(
        circle.pearson_xy,
        circle.spearman_xy,
        circle.r_squared_xy,
        circle.banding_detected,
        circle.heteroscedasticity_detected,
        None,
    )
    circle.bubble_quadrant_summary, circle.largest_bubble_quadrant, circle.large_bubble_concentration = _circle_quadrants(valid, spec.x, spec.y, spec.secondary_y, spec.color)
    circle.color_group_summary, circle.group_with_largest_total_size, circle.group_with_largest_average_size, circle.group_with_largest_single_bubble = _circle_color_groups(valid, spec.x, spec.y, spec.secondary_y, spec.color)
    circle.color_group_count = len(circle.color_group_summary)
    circle.cluster_count = circle.color_group_count if circle.color_group_count >= 2 else None
    circle.overlap_level = _circle_overlap_level(valid, spec.x, spec.y)

    x_mid = float(valid[spec.x].median())
    y_mid = float(valid[spec.y].median())
    size_q1 = float(valid[spec.secondary_y].quantile(0.25))
    size_q3 = float(valid[spec.secondary_y].quantile(0.75))
    circle.high_xy_small_size_count = int(((valid[spec.x] >= x_mid) & (valid[spec.y] >= y_mid) & (valid[spec.secondary_y] <= size_q1)).sum())
    mid_x = valid[spec.x].between(valid[spec.x].quantile(0.35), valid[spec.x].quantile(0.65))
    mid_y = valid[spec.y].between(valid[spec.y].quantile(0.35), valid[spec.y].quantile(0.65))
    circle.moderate_xy_large_size_count = int((mid_x & mid_y & (valid[spec.secondary_y] >= size_q3)).sum())
    binned = valid.copy()
    try:
        binned["_x_bin"] = pd.qcut(binned[spec.x], q=min(4, max(1, binned[spec.x].nunique())), duplicates="drop")
        binned["_y_bin"] = pd.qcut(binned[spec.y], q=min(4, max(1, binned[spec.y].nunique())), duplicates="drop")
        size_spread = binned.groupby(["_x_bin", "_y_bin"], observed=False)[spec.secondary_y].agg(["count", "min", "max"])
        circle.similar_position_different_size_count = int(((size_spread["count"] >= 2) & (size_spread["max"] >= size_spread["min"] * 2)).sum())
    except ValueError:
        circle.similar_position_different_size_count = 0

    circle.known_xy_relationship, circle.mathematical_dependency, circle.derived_metric_available = _scatter_formula_dependency(valid, spec.x, spec.y)
    size_relationship, size_dependency, size_derived = _scatter_formula_dependency(valid, spec.x, spec.secondary_y)
    circle.known_size_relationship = size_relationship
    if not circle.mathematical_dependency:
        circle.mathematical_dependency = size_dependency
    if not circle.derived_metric_available:
        circle.derived_metric_available = size_derived

    circle.calculated_metrics.update({
        "pearson_correlation": circle.pearson_xy,
        "spearman_correlation": circle.spearman_xy,
        "r_squared": circle.r_squared_xy,
        "largest_circle": largest.to_dict(),
        "circle_count": circle.displayed_point_count,
        "size_relationship_with_x": circle.pearson_size_x,
        "size_relationship_with_y": circle.pearson_size_y,
    })
    circle.warnings.append("Bubble area represents the size metric, but humans compare areas less precisely than positions along an axis.")
    if circle.overlap_level == "high":
        circle.warnings.append("Bubble overlap may hide smaller observations in dense regions.")
    if circle.mathematical_dependency:
        circle.warnings.append(circle.mathematical_dependency)
    if circle.displayed_point_count < 30 or circle.pearson_xy is None or valid[spec.secondary_y].nunique() <= 1:
        circle.evidence_strength = "low"
    elif circle.displayed_point_count < 100 or circle.overlap_level == "high" or circle.mathematical_dependency:
        circle.evidence_strength = "medium"
    else:
        circle.evidence_strength = "high"
    return circle


def extract_chart_evidence(result: ChartResult) -> ChartEvidence:
    """Dispatch to chart-specific evidence extraction."""
    evidence = _base_evidence(result)
    chart_type = result.spec.chart_type
    if chart_type == "grouped_bar":
        return _grouped_bar_evidence(result, evidence)
    if chart_type == "stacked_bar":
        return _stacked_bar_evidence(result, evidence)
    if chart_type == "pie":
        return _pie_chart_evidence(result, evidence)
    if chart_type == "treemap":
        return _treemap_evidence(result, evidence)
    if chart_type == "symbol_map":
        return _symbol_map_evidence(result, evidence)
    if chart_type == "sorted_percentage_bar":
        return _sorted_percentage_bar_evidence(result, evidence)
    if chart_type == "period_over_period_change":
        return _period_over_period_evidence(result, evidence)
    if is_single_series_bar(result):
        return _single_bar_evidence(result, evidence)
    if chart_type in {"bar", "pie", "treemap", "symbol_map"}:
        return _bar_like_evidence(result, evidence)
    if is_single_series_line(result):
        return _single_line_evidence(result, evidence)
    if is_stacked_area(result):
        return _stacked_area_evidence(result, evidence)
    if is_single_area(result):
        return _single_area_evidence(result, evidence)
    if chart_type in {"line", "area"}:
        return _line_like_evidence(result, evidence)
    if chart_type == "dual_line":
        return _dual_line_evidence(result, evidence)
    if chart_type == "dual_axis":
        return _dual_combination_evidence(result, evidence)
    if chart_type == "scatter":
        return _scatter_evidence(result, evidence)
    if chart_type == "circle_view":
        return _circle_evidence(result, evidence)
    if chart_type == "histogram":
        return _histogram_evidence(result, evidence)
    if chart_type == "box":
        return _box_plot_evidence(result, evidence)
    if chart_type == "heatmap":
        return _heatmap_evidence(result, evidence)
    if chart_type == "gantt":
        return _gantt_evidence(result, evidence)
    if chart_type == "bullet":
        return _bullet_evidence(result, evidence)
    evidence.warnings.append("No chart-specific extractor was available.")
    evidence.evidence_strength = "low"
    return evidence


def _aggregation_text(evidence: ChartEvidence) -> str:
    return evidence.aggregation or "raw"


def _filter_text(evidence: ChartEvidence) -> str:
    if not evidence.filters:
        return ""
    rendered = ", ".join(f"{display_name(k)} = {v}" for k, v in evidence.filters.items())
    return f" Active filter: {rendered}."


def _restriction_text(evidence: ChartEvidence) -> str:
    parts = []
    if isinstance(evidence, GroupedBarEvidence) and evidence.top_n_applied:
        parts.append(f"the chart is limited to the top {evidence.top_n_applied} displayed categories")
    elif evidence.top_n and evidence.total_rows >= evidence.top_n:
        parts.append(f"only the top {evidence.top_n} displayed categories are included")
    if evidence.excluded_rows:
        parts.append(f"{evidence.excluded_rows:,} row(s) with missing values were excluded")
    return " ".join(parts)


def _sentence(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    value = value[:1].upper() + value[1:]
    return value if value.endswith((".", "?", "!")) else f"{value}."


def _plural(label: str, count: int) -> str:
    return label if count == 1 else f"{label}s"


def _short_metric(metric: str | None) -> str:
    name = display_name(metric).lower() or "value"
    return name.removeprefix("total ") if name.startswith("total ") else name


def _metric_kind(metric_name: str) -> str:
    normalized = normalize_column_name(metric_name)
    if "revenue" in normalized or "sales" in normalized:
        return "revenue"
    if "profit" in normalized:
        return "profit"
    if "unit" in normalized and ("sold" in normalized or "sale" in normalized):
        return "units"
    if "count" in normalized or "order" in normalized or "volume" in normalized:
        return "count"
    return "generic"


def _dedupe_sentences(sentences: list[str], *, max_items: int = 4) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    seen_values: set[str] = set()
    for sentence in sentences:
        rendered = _sentence(sentence)
        normalized = re.sub(r"\s+", " ", rendered.casefold())
        values = set(re.findall(r"\$?[\d,.]+[KMBT]?", rendered))
        if not rendered or normalized in seen:
            continue
        if values and values.issubset(seen_values) and len(selected) >= 3:
            continue
        selected.append(rendered)
        seen.add(normalized)
        seen_values.update(values)
        if len(selected) >= max_items:
            break
    return selected


def build_combined_category_sentence(evidence: GroupedBarEvidence) -> str:
    metric = evidence.value_column or (evidence.y_columns[0] if evidence.y_columns else None)
    metric_name = _short_metric(metric)
    return (
        f"{evidence.highest_combined_category} generates "
        f"{_fmt(evidence.highest_combined_value, metric)} in combined {metric_name}, "
        f"compared with {_fmt(evidence.lowest_combined_value, metric)} in "
        f"{evidence.lowest_combined_category}"
    )


def build_within_category_comparison_sentence(evidence: GroupedBarEvidence) -> str:
    metric = evidence.value_column or (evidence.y_columns[0] if evidence.y_columns else None)
    metric_name = _short_metric(metric)
    category = evidence.highest_combined_category or ""
    winner = evidence.winner_by_category.get(category)
    winner_value = evidence.winner_value_by_category.get(category)
    if not category or not winner or winner_value is None:
        return ""
    return (
        f"Within {category}, {winner} contributes "
        f"{_fmt(winner_value, metric)} in {metric_name}"
    )


def build_overall_group_sentence(evidence: GroupedBarEvidence) -> str:
    metric = evidence.value_column or (evidence.y_columns[0] if evidence.y_columns else None)
    metric_name = _short_metric(metric)
    category_name = display_name(evidence.category_column).lower() or "category"
    strongest_group = next(iter(evidence.group_totals), None)
    if strongest_group is None:
        return ""
    strongest_total = evidence.group_totals[strongest_group]
    win_count = evidence.group_win_counts.get(strongest_group, 0)
    category_label = _plural(category_name, evidence.category_count)
    other_groups = [group for group in evidence.group_totals if group != strongest_group]
    comparison = f" and leads {other_groups[0]} in" if len(other_groups) == 1 else " and leads in"
    return (
        f"Across all displayed {category_label}, {strongest_group} generates "
        f"{_fmt(strongest_total, metric)} in total {metric_name}{comparison} "
        f"{win_count} of the {evidence.category_count} {category_label}"
    )


def build_largest_gap_sentence(evidence: GroupedBarEvidence) -> str:
    metric = evidence.value_column or (evidence.y_columns[0] if evidence.y_columns else None)
    category_name = display_name(evidence.category_column).lower() or "category"
    if not evidence.largest_gap_category or len(evidence.largest_gap_groups) < 2:
        return ""
    winner, runner_up = evidence.largest_gap_groups[:2]
    percent_text = ""
    if evidence.largest_gap_percent is not None:
        basis = evidence.largest_gap_percent_basis or f"{runner_up} value"
        percent_text = f", or {_pct(evidence.largest_gap_percent)} relative to {basis}"
    return (
        f"The largest within-{category_name} gap occurs in {evidence.largest_gap_category}: "
        f"{winner} exceeds {runner_up} by {_fmt(evidence.largest_gap_value, metric)}{percent_text}"
    )


def build_grouped_bar_interpretation(evidence: GroupedBarEvidence) -> str:
    category_name = display_name(evidence.category_column).lower() or "category"
    group_name = display_name(evidence.group_column).lower() or "group"
    strongest_group = next(iter(evidence.group_totals), None)
    winners = set(evidence.winner_by_category.values())
    win_count = evidence.group_win_counts.get(strongest_group, 0) if strongest_group else 0
    if len(winners) > 1:
        if win_count > 1:
            leader_text = (
                f"{strongest_group} is stronger in several displayed "
                f"{_plural(category_name, evidence.category_count)}, while other {group_name} "
                "values lead elsewhere"
            )
        else:
            leader_text = (
                f"{strongest_group} has the highest displayed total, but other {group_name} "
                "values lead in more individual categories"
            )
        return (
            f"{display_name(evidence.group_column)} performance varies by {category_name} "
            f"rather than following one consistent pattern. {leader_text}, suggesting that planning should be evaluated "
            f"{category_name} by {category_name}."
        )
    return (
        f"{display_name(evidence.group_column)} performance is concentrated around "
        f"{strongest_group}, but the size of the advantage differs by {category_name}. "
        f"The pattern supports further analysis of whether the leading {group_name} is also efficient."
    )


def build_metric_specific_caution(metric_name: str) -> str:
    kind = _metric_kind(metric_name)
    if kind == "revenue":
        return (
            "The chart shows total revenue, which can be influenced by order volume, "
            "units sold, and unit prices. It does not show profitability or channel efficiency."
        )
    if kind == "profit":
        return (
            "The chart shows total profit, which may be influenced by market size and order volume. "
            "It does not show profit margin or profit per order."
        )
    if kind == "units":
        return "The chart shows units sold, not revenue, profit, average price, or margin."
    if kind == "count":
        return "The chart shows order volume, not revenue, profit, or average order value."
    return (
        f"The chart shows {metric_name.lower()}, which may be influenced by category size "
        "and record volume. It does not show efficiency or per-record performance."
    )


def build_metric_specific_next_step(
    metric_name: str,
    x_column: str | None,
    group_column: str | None,
) -> str:
    category_name = display_name(x_column).lower() or "category"
    group_name = display_name(group_column).lower() or "group"
    suffix = f" by {category_name} and {group_name}."
    kind = _metric_kind(metric_name)
    if kind == "revenue":
        return "Compare total profit, profit margin, revenue per order, and units sold" + suffix
    if kind == "profit":
        return "Compare profit margin, profit per order, order volume, and total cost" + suffix
    if kind == "units":
        return "Compare revenue per unit, profit per unit, and average unit price" + suffix
    if kind == "count":
        return "Compare average order value, total revenue, and profit per order" + suffix
    return f"Compare {metric_name.lower()} with margin, per-record value, and volume{suffix}"


def _warning_sentence(warning: str) -> str:
    warning = warning.strip()
    if re.match(r"^\d", warning):
        warning = f"There are {warning}"
    return _sentence(warning)


def build_metric_phrase(value_column: str | None, aggregation: str | None) -> str:
    metric = _short_metric(value_column)
    if aggregation == "mean":
        return f"average {metric}"
    if aggregation == "median":
        return f"median {metric}"
    if aggregation == "count":
        if "order" in normalize_column_name(value_column) or value_column == "Count":
            return "number of orders"
        return f"number of {metric}"
    if aggregation == "min":
        return f"minimum {metric}"
    if aggregation == "max":
        return f"maximum {metric}"
    return f"total {metric}"


def _axis_plural(column: str | None) -> str:
    name = display_name(column).lower() or "categories"
    if name.endswith("y"):
        return name[:-1] + "ies"
    if name.endswith("s"):
        return name
    return name + "s"


def build_single_bar_caution(metric_name: str, aggregation: str | None) -> str:
    kind = _metric_kind(metric_name)
    if aggregation == "mean":
        return "An average can be affected by extreme observations and does not show the full distribution."
    if aggregation == "median":
        return "The median represents the central observation but does not show total business contribution."
    if kind == "revenue":
        return (
            "Total revenue can be influenced by order volume, units sold, and unit prices. "
            "It does not show profitability or operational efficiency."
        )
    if kind == "profit":
        return (
            "Total profit can be influenced by market size and order volume. "
            "It does not show profit margin or profit per order."
        )
    if kind == "units":
        return "Units sold measures volume, not revenue, profit, or value per transaction."
    if kind == "count":
        return "Order count measures transaction volume, not average order value, revenue, or profitability."
    return (
        f"{metric_name} may be influenced by category size and record volume. "
        "It does not show efficiency or per-record performance."
    )


def build_single_bar_next_step(metric_name: str, x_column: str | None) -> str:
    axis = _axis_plural(x_column)
    kind = _metric_kind(metric_name)
    if kind == "revenue":
        return (
            "Compare total profit, profit margin, units sold, average order value, "
            f"and revenue per order across {axis}."
        )
    if kind == "profit":
        return (
            "Compare profit margin, total cost, units sold, and profit per order "
            f"across {axis}."
        )
    if kind == "units":
        return (
            "Compare revenue per unit, profit per unit, unit price, and total revenue "
            f"across {axis}."
        )
    if kind == "count":
        return f"Compare average order value, total revenue, and profit per order across {axis}."
    return f"Compare {metric_name.lower()} with margin, per-record value, and volume across {axis}."


def _pie_metric_phrase(evidence: PieChartEvidence) -> str:
    metric = _short_metric(evidence.value_column)
    aggregation = evidence.aggregation or "sum"
    if aggregation == "count":
        return "displayed records"
    if aggregation == "nunique":
        return f"unique {metric}"
    if aggregation == "sum":
        return metric
    if aggregation == "mean":
        return f"average {metric}"
    if aggregation == "median":
        return f"median {metric}"
    return f"{aggregation} {metric}"


def _pie_next_step(evidence: PieChartEvidence) -> str:
    metric_norm = normalize_column_name(evidence.value_column)
    category = display_name(evidence.category_column).lower() or "categories"
    leaders = "the two leading categories" if evidence.second_category else "the leading category"
    if "revenue" in metric_norm or "sales" in metric_norm:
        return f"Compare total profit, profit margin, units sold, and revenue per order for {leaders}."
    if "profit" in metric_norm:
        return f"Compare profit margin and total cost for the largest profit-contributing {category}."
    if "count" in metric_norm or evidence.aggregation == "count":
        return f"Compare average order value and fulfillment performance across the largest {category}."
    if not evidence.part_to_whole_valid:
        return f"Use a sorted bar chart for precise comparison of {display_name(evidence.value_column).lower()} across {category}."
    return f"Use a sorted bar chart for precise comparison and examine a related efficiency or normalized measure for the largest {category}."


def build_pie_chart_fallback(evidence: PieChartEvidence) -> ChartInsight:
    """Create deterministic, part-to-whole insight text for Pie/Donut charts."""
    metric_phrase = _pie_metric_phrase(evidence)
    category_name = display_name(evidence.category_column).lower() or "category"
    if not evidence.part_to_whole_valid:
        key = (
            f"{evidence.largest_category} ranks highest for {metric_phrase}, but this Pie chart should not be read as a true part-to-whole composition."
        )
    elif evidence.second_category and evidence.lead_strength == "narrow":
        key = (
            f"{evidence.largest_category} contributes the largest share of {metric_phrase} at {_pct(evidence.largest_share)}, "
            f"narrowly ahead of {evidence.second_category} at {_pct(evidence.second_share)}."
        )
    elif evidence.second_category:
        key = (
            f"{evidence.largest_category} contributes the largest share of {metric_phrase} at {_pct(evidence.largest_share)}, "
            f"ahead of {evidence.second_category} at {_pct(evidence.second_share)}."
        )
    else:
        key = f"{evidence.largest_category} is the only displayed {category_name} for {metric_phrase}."
    if evidence.part_to_whole_valid and evidence.top_two_share is not None and evidence.lead_strength == "narrow":
        key += f" Together, the two leaders account for {_pct(evidence.top_two_share)} of the displayed total."

    facts = []
    if evidence.largest_category and evidence.largest_value is not None:
        leader = f"{evidence.largest_category} contributes {_fmt(evidence.largest_value, evidence.value_column)}"
        if evidence.largest_share is not None:
            leader += f", or {_pct(evidence.largest_share)} of displayed {metric_phrase}"
        facts.append(leader + ".")
    if evidence.second_category and evidence.second_value is not None:
        runner = f"{evidence.second_category} follows at {_fmt(evidence.second_value, evidence.value_column)}"
        if evidence.second_share is not None:
            runner += f", or {_pct(evidence.second_share)}"
        if evidence.leader_to_second_gap_percent is not None and evidence.lead_strength in {"narrow", "moderate", "clear"}:
            runner += f"; the leader is {_pct(evidence.leader_to_second_gap_percent)} above {evidence.second_category}"
        facts.append(runner + ".")
    if evidence.top_two_share is not None and evidence.second_category:
        facts.append(f"The top two slices account for {_pct(evidence.top_two_share)} of the displayed total.")
    elif evidence.top_three_share is not None:
        facts.append(f"The top three slices account for {_pct(evidence.top_three_share)} of the displayed total.")
    if evidence.smallest_category and evidence.smallest_category != evidence.largest_category and evidence.smallest_value is not None:
        smallest = f"{evidence.smallest_category} contributes the least at {_fmt(evidence.smallest_value, evidence.value_column)}"
        if evidence.smallest_share is not None:
            smallest += f", or {_pct(evidence.smallest_share)}"
        facts.append(smallest + ".")
    if evidence.other_category_present and evidence.other_category_share is not None:
        facts.append(f"{evidence.other_category_label} combines smaller categories and represents {_pct(evidence.other_category_share)} of the displayed total.")
    if evidence.top_n_applied:
        facts.append(f"The chart shows the top {evidence.displayed_category_count} of {evidence.original_category_count} {category_name} values.")
    support = " ".join(_dedupe_sentences(facts, max_items=5))

    if not evidence.part_to_whole_valid:
        interpretation = (
            f"The ranking is still useful, but {evidence.aggregation} values do not form additive slices of one meaningful whole. "
            "A sorted bar chart is clearer for this comparison."
        )
    elif evidence.concentration_level == "balanced":
        interpretation = (
            f"The displayed {metric_phrase} is relatively balanced across {category_name} values; no single slice clearly dominates the total."
        )
    elif evidence.concentration_level == "highly concentrated":
        interpretation = f"The displayed {metric_phrase} is highly concentrated in the leading slice or leading slices."
    else:
        interpretation = (
            f"The Pie chart shows part-to-whole contribution: larger slices represent a larger share of displayed {metric_phrase}."
        )
    if evidence.top_three_share is not None and evidence.remaining_share is not None and evidence.remaining_share > 0:
        interpretation += f" The remaining categories account for {_pct(evidence.remaining_share)} after the top three."

    cautions = []
    if not evidence.part_to_whole_valid:
        cautions.append("The selected aggregation does not form a meaningful additive total, so slice percentages should be interpreted cautiously.")
    if any("negative" in warning.lower() for warning in evidence.warnings):
        cautions.append("Pie charts are not suitable when values include negative contributions.")
    if evidence.small_slice_categories:
        cautions.append("Small slices and similar shares are harder to compare precisely than sorted bar lengths.")
    else:
        cautions.append("Pie slices are less precise to compare than bar lengths, especially when shares are similar.")
    if evidence.filters_applied:
        cautions.append("The shares apply only to the currently filtered data.")
    if evidence.aggregation == "nunique":
        cautions.append("Unique counts may not be additive when the same entity appears in more than one category.")

    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=" ".join(_dedupe_sentences(cautions, max_items=2)),
        recommended_next_step=_pie_next_step(evidence),
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def _treemap_next_step(evidence: TreemapEvidence) -> str:
    metric_norm = normalize_column_name(evidence.value_column)
    category = display_name(evidence.category_column).lower() or "categories"
    if "revenue" in metric_norm or "sales" in metric_norm:
        return f"Compare total profit, profit margin, units sold, and revenue per order for the largest {category}."
    if "profit" in metric_norm:
        return f"Compare profit margin, total cost, and revenue for the largest profit-contributing {category}."
    if evidence.aggregation == "count":
        return f"Compare average order value and fulfillment performance across the largest {category}."
    if not evidence.part_to_whole_valid:
        return f"Use a sorted bar chart to compare {display_name(evidence.value_column).lower()} precisely across {category}."
    return f"Use a sorted bar chart for exact ranking, then compare a normalized efficiency metric for the largest {category}."


def build_treemap_fallback(evidence: TreemapEvidence) -> ChartInsight:
    """Create deterministic area-share insight text for Treemap charts."""
    metric_phrase = _pie_metric_phrase(evidence)  # Same aggregation wording works for area shares.
    category_name = display_name(evidence.category_column).lower() or "category"
    if not evidence.part_to_whole_valid:
        key = (
            f"{evidence.largest_category} has the largest Treemap rectangle for {metric_phrase}, "
            "but the selected values should not be read as additive parts of a whole."
        )
    elif evidence.second_category and evidence.lead_strength == "narrow":
        key = (
            f"{evidence.largest_category} has the largest area share of {metric_phrase} at {_pct(evidence.largest_share)}, "
            f"narrowly ahead of {evidence.second_category} at {_pct(evidence.second_share)}."
        )
    elif evidence.second_category:
        key = (
            f"{evidence.largest_category} has the largest Treemap area for {metric_phrase} at {_pct(evidence.largest_share)}, "
            f"ahead of {evidence.second_category} at {_pct(evidence.second_share)}."
        )
    else:
        key = f"{evidence.largest_category} is the only displayed {category_name} in the Treemap."

    facts = []
    if evidence.largest_category and evidence.largest_value is not None:
        fact = f"{evidence.largest_category} contributes {_fmt(evidence.largest_value, evidence.value_column)}"
        if evidence.largest_share is not None:
            fact += f" and represents {_pct(evidence.largest_share)} of displayed {metric_phrase}"
        facts.append(fact + ".")
    if evidence.second_category and evidence.second_value is not None:
        fact = f"{evidence.second_category} follows at {_fmt(evidence.second_value, evidence.value_column)}"
        if evidence.second_share is not None:
            fact += f", or {_pct(evidence.second_share)}"
        if evidence.leader_to_second_gap_percent is not None:
            fact += f"; the leader is {_pct(evidence.leader_to_second_gap_percent)} above {evidence.second_category}"
        facts.append(fact + ".")
    if evidence.top_two_share is not None:
        facts.append(f"The top two rectangles account for {_pct(evidence.top_two_share)} of the displayed total.")
    if evidence.largest_group and evidence.largest_group_share is not None:
        facts.append(f"Within the hierarchy, {evidence.largest_group} is the largest parent group at {_pct(evidence.largest_group_share)} of the displayed total.")
    if evidence.top_n_applied:
        facts.append(f"The chart shows the top {evidence.displayed_category_count} of {evidence.original_category_count} {category_name} values.")
    if evidence.smallest_category and evidence.smallest_category != evidence.largest_category and evidence.smallest_value is not None:
        fact = f"{evidence.smallest_category} is the smallest rectangle at {_fmt(evidence.smallest_value, evidence.value_column)}"
        if evidence.smallest_share is not None:
            fact += f", or {_pct(evidence.smallest_share)}"
        facts.append(fact + ".")
    support = " ".join(_dedupe_sentences(facts, max_items=5))

    if not evidence.part_to_whole_valid:
        interpretation = (
            f"Treemap area is useful for ranking here, but {evidence.aggregation} values do not form one additive total. "
            "A sorted bar chart is clearer for precise comparison."
        )
    elif evidence.group_column and evidence.largest_group:
        interpretation = (
            f"The Treemap shows both hierarchy and part-to-whole contribution: parent rectangles summarize "
            f"{display_name(evidence.group_column).lower()}, while child rectangles show {category_name} contribution."
        )
    elif evidence.concentration_level == "balanced":
        interpretation = f"The displayed {metric_phrase} is relatively balanced; no single rectangle takes a dominant area."
    elif evidence.concentration_level == "highly concentrated":
        interpretation = f"The displayed {metric_phrase} is highly concentrated in the largest rectangle or leading rectangles."
    else:
        interpretation = f"Rectangle area represents each {category_name}'s share of displayed {metric_phrase}, so larger areas indicate larger contribution."
    if evidence.remaining_share is not None and evidence.top_three_share is not None and evidence.remaining_share > 0:
        interpretation += f" The categories outside the top three account for {_pct(evidence.remaining_share)}."

    cautions = []
    if not evidence.part_to_whole_valid:
        cautions.append("The selected aggregation does not form a meaningful additive total, so area shares should be interpreted cautiously.")
    if evidence.small_rectangle_categories:
        cautions.append("Very small rectangles can be hard to label and compare precisely.")
    else:
        cautions.append("Treemaps are good for part-to-whole structure, but sorted bars are more precise for close comparisons.")
    if evidence.top_n_applied:
        cautions.append("The omitted categories are not represented in the displayed area shares.")
    if evidence.filters_applied:
        cautions.append("The area shares apply only to the currently filtered data.")

    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=" ".join(_dedupe_sentences(cautions, max_items=2)),
        recommended_next_step=_treemap_next_step(evidence),
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def _sorted_percentage_next_step(evidence: SortedPercentageBarEvidence) -> str:
    metric_norm = normalize_column_name(evidence.value_column)
    category = display_name(evidence.category_column).lower() or "categories"
    if "revenue" in metric_norm or "sales" in metric_norm:
        return "Compare total profit, profit margin, units sold, and revenue per order for the two leading categories."
    if "profit" in metric_norm:
        return f"Compare profit margin, total cost, and units sold for the {category} contributing the largest profit shares."
    if "unitssold" in metric_norm or "units" in metric_norm:
        return f"Compare revenue per unit and profit per unit across the leading {category}."
    if evidence.aggregation == "count":
        return f"Compare average order value and profitability across the largest {category}."
    if not evidence.percentage_valid or not evidence.additive_aggregation:
        return f"Use a regular sorted bar chart for {display_name(evidence.value_column).lower()} before interpreting contribution shares."
    return f"Compare the leading {category} with a related normalized or efficiency measure."


def build_sorted_percentage_bar_fallback(evidence: SortedPercentageBarEvidence) -> ChartInsight:
    """Create deterministic insight text for Sorted Percentage Bar charts."""
    metric_phrase = _pie_metric_phrase(evidence)  # same aggregation-aware wording
    if not evidence.percentage_valid or not evidence.additive_aggregation:
        key = (
            f"{evidence.largest_category} ranks highest for {metric_phrase}, but percentage-of-total contribution is not valid for this configuration."
        )
    elif evidence.second_category and evidence.lead_strength == "narrow":
        key = (
            f"{evidence.largest_category} contributes the largest share of {metric_phrase} at {_pct(evidence.largest_share)}, "
            f"narrowly ahead of {evidence.second_category} at {_pct(evidence.second_share)}."
        )
    elif evidence.second_category:
        key = (
            f"{evidence.largest_category} contributes the largest share of {metric_phrase} at {_pct(evidence.largest_share)}, "
            f"ahead of {evidence.second_category} at {_pct(evidence.second_share)}."
        )
    else:
        key = f"{evidence.largest_category} is the only displayed category for {metric_phrase}."
    if evidence.top_two_share is not None and evidence.lead_strength == "narrow":
        key += f" Together, the two leaders account for {_pct(evidence.top_two_share)} of the filtered total."

    facts = []
    if evidence.largest_category and evidence.largest_value is not None:
        fact = f"{evidence.largest_category} contributes {_fmt(evidence.largest_value, evidence.value_column)}"
        if evidence.largest_share is not None:
            fact += f", equal to {_pct(evidence.largest_share)} of filtered {metric_phrase}"
        facts.append(fact + ".")
    if evidence.second_category and evidence.second_value is not None:
        fact = f"{evidence.second_category} follows at {_fmt(evidence.second_value, evidence.value_column)}"
        if evidence.second_share is not None:
            fact += f", or {_pct(evidence.second_share)}"
        if evidence.leader_to_second_gap_percentage_points is not None:
            fact += f", a difference of {evidence.leader_to_second_gap_percentage_points:.1f} percentage points"
        facts.append(fact + ".")
    if evidence.top_three_share is not None:
        facts.append(f"The top three categories account for {_pct(evidence.top_three_share)} of the selected denominator.")
    elif evidence.top_two_share is not None:
        facts.append(f"The top two categories account for {_pct(evidence.top_two_share)} of the selected denominator.")
    if evidence.smallest_category and evidence.smallest_share is not None:
        facts.append(f"{evidence.smallest_category} contributes the smallest share at {_pct(evidence.smallest_share)}.")
    if evidence.top_n_applied:
        facts.append(f"The chart shows {evidence.displayed_category_count} of {evidence.original_category_count} categories using the {evidence.percentage_denominator_mode.replace('_', ' ')} denominator.")
    support = " ".join(_dedupe_sentences(facts, max_items=5))

    if not evidence.percentage_valid or not evidence.additive_aggregation:
        interpretation = "The ranking is useful, but the selected aggregation or denominator does not support true percentage contribution."
    elif evidence.concentration_level == "balanced":
        interpretation = f"The contribution is relatively balanced across categories; no category accounts for a dominant share of {metric_phrase}."
    elif evidence.concentration_level == "highly concentrated":
        interpretation = f"The selected {metric_phrase} is highly concentrated in the leading category or leading categories."
    else:
        interpretation = f"The chart ranks categories by percentage contribution, making the denominator and share differences easier to compare than raw totals alone."
    if evidence.date_column:
        interpretation += f" Shares are calculated after applying the selected date filter on {display_name(evidence.date_column)}."

    cautions = []
    if not evidence.additive_aggregation:
        cautions.append("The selected aggregation does not represent additive contributions, so percentage shares should be interpreted cautiously.")
    if evidence.negative_value_count:
        cautions.append("Percentage-of-total contribution is not meaningful when aggregated values include negative amounts.")
    if evidence.top_n_applied and evidence.percentage_denominator_mode == "full_filtered":
        cautions.append("Top-N is active, so excluded categories remain in the full filtered denominator.")
    if evidence.small_share_categories:
        cautions.append("Several categories contribute very small shares and may need labels for precise comparison.")
    cautions.append("Shares are calculated from the filtered data shown by the current chart configuration.")

    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=" ".join(_dedupe_sentences(cautions, max_items=2)),
        recommended_next_step=_sorted_percentage_next_step(evidence),
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def _period_change_next_step(evidence: PeriodOverPeriodChangeEvidence) -> str:
    normalized = normalize_column_name(evidence.value_column)
    if "revenue" in normalized or "sales" in normalized:
        return "Compare units sold, unit price, and order count for the periods with the largest revenue changes."
    if "profit" in normalized:
        return "Compare revenue, total cost, and profit margin for the periods with the largest profit changes."
    if "units" in normalized:
        return "Compare revenue per unit and profit per unit for the periods with the largest unit-volume changes."
    return "Inspect the periods with the largest positive and negative changes and compare them with related operational drivers."


def build_period_over_period_fallback(evidence: PeriodOverPeriodChangeEvidence) -> ChartInsight:
    """Create deterministic insight text for period-over-period percentage changes."""
    metric_phrase = build_metric_phrase(evidence.value_column, evidence.aggregation)
    basis = "same period last year" if evidence.comparison_basis == "same_period_last_year" else "previous period"
    if evidence.latest_period and evidence.latest_percent_change is not None:
        direction = "increased" if evidence.latest_percent_change > 0 else "decreased" if evidence.latest_percent_change < 0 else "was unchanged"
        key = (
            f"{metric_phrase.capitalize()} {direction} by {_pct(abs(evidence.latest_percent_change))} in {evidence.latest_period} "
            f"versus the {basis}."
        )
    else:
        key = f"The chart has limited comparable {metric_phrase} periods for period-over-period change."

    facts = []
    if evidence.latest_period and evidence.latest_value is not None and evidence.latest_comparison_value is not None:
        facts.append(
            f"{evidence.latest_period} was {_fmt(evidence.latest_value, evidence.value_column)} versus "
            f"{_fmt(evidence.latest_comparison_value, evidence.value_column)} for the comparison period."
        )
    if evidence.largest_increase_period and evidence.largest_increase_percent is not None:
        facts.append(f"The largest increase was {_pct(evidence.largest_increase_percent)} in {evidence.largest_increase_period}.")
    if evidence.largest_decline_period and evidence.largest_decline_percent is not None and evidence.largest_decline_percent < 0:
        facts.append(f"The largest decline was {_pct(abs(evidence.largest_decline_percent))} in {evidence.largest_decline_period}.")
    facts.append(
        f"{evidence.increase_count} period(s) increased and {evidence.decline_count} period(s) declined across {evidence.comparable_period_count} comparable period(s)."
    )
    if evidence.unavailable_period_count:
        facts.append(f"{evidence.unavailable_period_count} displayed period(s) lacked a comparable baseline.")
    support = " ".join(_dedupe_sentences(facts, max_items=5))

    if evidence.increase_count > evidence.decline_count:
        interpretation = f"Most comparable periods improved versus the {basis}, but the magnitude varies by period."
    elif evidence.decline_count > evidence.increase_count:
        interpretation = f"More comparable periods declined than increased versus the {basis}."
    else:
        interpretation = f"The chart compares each displayed period with the {basis}, highlighting direction and magnitude rather than raw level alone."
    if evidence.volatility_level:
        interpretation += f" The period-to-period change volatility is {evidence.volatility_level}."

    cautions = []
    if evidence.unavailable_period_count:
        cautions.append("Some periods have no comparable baseline, so their percentage change is unavailable.")
    if evidence.zero_baseline_count:
        cautions.append("Percentage change is undefined when the comparison baseline is zero.")
    cautions.append("Percentage change can look large when the comparison-period value is small.")

    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=" ".join(_dedupe_sentences(cautions, max_items=2)),
        recommended_next_step=_period_change_next_step(evidence),
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def build_single_bar_key_finding(evidence: SingleBarEvidence) -> str:
    phrase = build_metric_phrase(evidence.value_column, evidence.aggregation)
    high = evidence.highest_category
    second = evidence.second_highest_category
    low = evidence.lowest_category
    if not second:
        return f"{high} records the only displayed {phrase} value."
    if evidence.lead_strength == "narrow":
        lead_text = f"{high} records the highest {phrase}, narrowly ahead of {second}"
    elif evidence.lead_strength == "moderate":
        lead_text = f"{high} leads {second} by a moderate margin for {phrase}"
    elif evidence.lead_strength == "clear":
        lead_text = f"{high} records substantially higher {phrase} than {second}"
    else:
        lead_text = f"{high} records the highest {phrase}"
    if low and low != high:
        lead_text += f", while {low} contributes the least"
    return lead_text + "."


def build_single_bar_support(evidence: SingleBarEvidence) -> str:
    metric = evidence.value_column
    phrase = build_metric_phrase(metric, evidence.aggregation)
    short_metric = _short_metric(metric)
    facts = [
        (
            f"{evidence.highest_category} records {_fmt(evidence.highest_value, metric)} "
            f"for {phrase}"
        )
    ]
    if evidence.highest_share_percent is not None:
        facts[0] += f", representing {_pct(evidence.highest_share_percent)} of displayed {short_metric}"
    if evidence.second_highest_category and evidence.second_highest_value is not None:
        gap_text = ""
        if evidence.leader_to_second_gap is not None:
            gap_text = f", leaving a difference of {_fmt(evidence.leader_to_second_gap, metric)}"
            if evidence.leader_to_second_gap_percent is not None:
                gap_text += (
                    f", or {_pct(evidence.leader_to_second_gap_percent)} relative to "
                    f"{evidence.leader_to_second_gap_basis}"
                )
        facts.append(
            f"{evidence.second_highest_category} follows at "
            f"{_fmt(evidence.second_highest_value, metric)}{gap_text}"
        )
    if evidence.top_two_share_percent is not None and evidence.second_highest_category:
        facts.append(
            f"Together, {evidence.highest_category} and {evidence.second_highest_category} "
            f"account for {_pct(evidence.top_two_share_percent)} of displayed {short_metric}"
        )
    if evidence.lowest_category and evidence.lowest_value is not None and evidence.lowest_category != evidence.highest_category:
        lowest_text = (
            f"{evidence.lowest_category} records the lowest value at "
            f"{_fmt(evidence.lowest_value, metric)}"
        )
        if evidence.highest_to_lowest_gap is not None and evidence.category_count <= 5:
            lowest_text += f", a high-to-low gap of {_fmt(evidence.highest_to_lowest_gap, metric)}"
        facts.append(lowest_text)
    support = " ".join(_dedupe_sentences(facts, max_items=4))
    restriction = _restriction_text(evidence)
    if restriction:
        support += f" {_sentence('Note: ' + restriction)}"
    if evidence.filters_applied:
        support += _filter_text(evidence)
    return support


def build_single_bar_interpretation(evidence: SingleBarEvidence) -> str:
    short_metric = _short_metric(evidence.value_column)
    if evidence.concentration_level == "high" and evidence.top_two_share_percent is not None:
        return (
            f"{display_name(evidence.value_column)} is concentrated in the leading categories. "
            f"The top two account for {_pct(evidence.top_two_share_percent)} of displayed {short_metric}, "
            "so category-level decisions should pay close attention to the leaders."
        )
    if evidence.concentration_level == "moderate" and evidence.top_three_share_percent is not None:
        return (
            f"{display_name(evidence.value_column)} shows moderate concentration. "
            f"The top three account for {_pct(evidence.top_three_share_percent)} of displayed {short_metric}, "
            "while the remaining categories still contribute meaningful volume."
        )
    if evidence.concentration_level == "distributed":
        return (
            f"{display_name(evidence.value_column)} is relatively distributed across the displayed "
            f"{_axis_plural(evidence.category_column)}, with no single category accounting for a dominant share."
        )
    if evidence.lead_strength == "narrow":
        return (
            f"The leading category is close to the runner-up, so the ranking should be interpreted as a narrow lead "
            f"rather than clear dominance."
        )
    return (
        f"The ranking compares displayed {short_metric} across {_axis_plural(evidence.category_column)}. "
        "Use the ordering as a starting point, then check whether volume, pricing, or efficiency explains the difference."
    )


def build_single_bar_fallback(evidence: SingleBarEvidence) -> ChartInsight:
    """Create deterministic, chart-specific insight text for standard bar charts."""
    metric_name = display_name(evidence.value_column) or "Value"
    caution = build_single_bar_caution(metric_name, evidence.aggregation)
    if evidence.warnings:
        caution += " " + " ".join(_warning_sentence(warning) for warning in evidence.warnings)
    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=build_single_bar_key_finding(evidence),
        supporting_evidence=build_single_bar_support(evidence),
        interpretation=build_single_bar_interpretation(evidence),
        caution=caution,
        recommended_next_step=build_single_bar_next_step(metric_name, evidence.category_column),
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def build_stacked_bar_fallback(evidence: StackedBarEvidence) -> ChartInsight:
    """Create deterministic insight text for stacked bars."""
    metric = evidence.value_column or (evidence.y_columns[0] if evidence.y_columns else None)
    metric_name = _short_metric(metric)
    category_name = display_name(evidence.category_column).lower() or "category"
    stack_name = display_name(evidence.group_column).lower() or "stack"
    key = (
        f"{evidence.highest_combined_category} records the highest combined {metric_name}, "
        f"while {evidence.lowest_combined_category} records the lowest."
    )
    support_parts = [
        (
            f"{evidence.highest_combined_category} totals "
            f"{_fmt(evidence.highest_combined_value, metric)} across all {stack_name} segments"
        ),
        (
            f"Within {evidence.highest_combined_category}, "
            f"{evidence.highest_category_dominant_stack} contributes "
            f"{_fmt(evidence.highest_category_dominant_value, metric)}"
            + (
                f", or {_pct(evidence.highest_category_dominant_share)} of that category total"
                if evidence.highest_category_dominant_share is not None
                else ""
            )
        ),
        (
            f"Across all displayed {_plural(category_name, evidence.category_count)}, "
            f"{evidence.strongest_stack} is the largest {stack_name} segment overall at "
            f"{_fmt(evidence.strongest_stack_value, metric)}"
        ),
        (
            f"{evidence.lowest_combined_category} has the lowest combined {metric_name} at "
            f"{_fmt(evidence.lowest_combined_value, metric)}"
        ),
    ]
    support = " ".join(_dedupe_sentences(support_parts, max_items=4))
    restriction = _restriction_text(evidence)
    if restriction:
        support += f" {_sentence('Note: ' + restriction)}"
    if evidence.filters_applied:
        support += _filter_text(evidence)
    interpretation = (
        f"The stacked view separates total {metric_name} by {category_name} from the "
        f"{stack_name} mix inside each bar. Read the bar height as the combined total, "
        f"then compare segment colors to understand composition."
    )
    caution = build_metric_specific_caution(display_name(metric) or "Value")
    if evidence.warnings:
        caution += " " + " ".join(_warning_sentence(warning) for warning in evidence.warnings)
    next_step = (
        f"Compare share of {metric_name}, margin, and volume by {category_name} and "
        f"{stack_name} to see whether the largest segments are also efficient."
    )
    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=caution,
        recommended_next_step=next_step,
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def build_grouped_bar_fallback(evidence: GroupedBarEvidence) -> ChartInsight:
    """Create deterministic, chart-specific grouped-bar insight text."""
    metric = evidence.value_column or (evidence.y_columns[0] if evidence.y_columns else None)
    metric_name = display_name(metric) or "Value"
    short_metric = _short_metric(metric)
    category_name = display_name(evidence.category_column) or "category"
    group_name = display_name(evidence.group_column) or "group"
    strongest_group = next(iter(evidence.group_totals), None)

    key = (
        f"{evidence.highest_combined_category} records the highest combined {short_metric}, "
        f"while {evidence.lowest_combined_category} records the lowest."
    )
    if strongest_group and len(evidence.group_win_counts) > 1:
        key += (
            f" The stronger {group_name.lower()} differs across "
            f"{_plural(category_name.lower(), evidence.category_count)}."
        )

    support = " ".join(_dedupe_sentences([
        build_combined_category_sentence(evidence),
        build_within_category_comparison_sentence(evidence),
        build_overall_group_sentence(evidence),
        build_largest_gap_sentence(evidence),
    ]))
    restriction = _restriction_text(evidence)
    if restriction:
        support += f" {_sentence('Note: ' + restriction)}"
    if evidence.filters_applied:
        support += _filter_text(evidence)
    interpretation = build_grouped_bar_interpretation(evidence)
    caution = build_metric_specific_caution(metric_name)
    if evidence.warnings:
        caution += " " + " ".join(_warning_sentence(warning) for warning in evidence.warnings)
    next_step = build_metric_specific_next_step(
        metric_name,
        evidence.category_column,
        evidence.group_column,
    )
    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=caution,
        recommended_next_step=next_step,
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def _prose_name(value: str | None) -> str:
    return display_name(value).lower()


def build_time_metric_phrase(
    y_column: str,
    aggregation: str | None,
    granularity: str | None,
) -> str:
    metric = _prose_name(y_column)
    time_prefix = {
        "day": "daily",
        "week": "weekly",
        "month": "monthly",
        "quarter": "quarterly",
        "year": "yearly",
    }.get(granularity, "")
    aggregation_word = {
        "sum": "total",
        "mean": "average",
        "median": "median",
        "count": "count",
        "min": "minimum",
        "max": "maximum",
    }.get(aggregation or "", "")
    parts = [time_prefix, aggregation_word, metric]
    return " ".join(part for part in parts if part)


def build_area_metric_phrase(
    y_column: str,
    aggregation: str | None,
    granularity: str | None,
) -> str:
    return build_time_metric_phrase(y_column, aggregation, granularity)


def _endpoint_direction(change: float | None) -> str:
    if change is None:
        return "ended near where it began"
    if change > 0:
        return "increased from its starting level"
    if change < 0:
        return "decreased from its starting level"
    return "ended unchanged"


def _line_metric_caution(evidence: SingleLineEvidence) -> str:
    normalized = normalize_column_name(evidence.y_column)
    aggregation = evidence.aggregation or ""
    if normalized in {"totalrevenue", "revenue", "sales"} or "revenue" in normalized:
        return "Total revenue may change because of order volume, units sold, or unit prices; it does not show profitability."
    if normalized in {"totalprofit", "profit"} or "profit" in normalized:
        return "Total profit may be affected by both revenue and cost changes and does not show profit margin by itself."
    if "ordercount" in normalized or aggregation == "count":
        return "Order count measures transaction volume and does not show order value or profitability."
    if aggregation == "mean":
        return "The chart compares period averages, which may hide changes in sample size and within-period variation."
    return "The series shows aggregated historical values; interpretation depends on the metric definition and consistency of the underlying observations."


def _line_next_step(evidence: SingleLineEvidence) -> str:
    normalized = normalize_column_name(evidence.y_column)
    if normalized in {"totalrevenue", "revenue", "sales"} or "revenue" in normalized:
        return "Compare total profit, units sold, order count, and average order value over the same periods."
    if normalized in {"totalprofit", "profit"} or "profit" in normalized:
        return "Compare revenue, total cost, and profit margin across the same periods."
    if "ordercount" in normalized or evidence.aggregation == "count":
        return "Compare order count with average order value and profit over the same periods."
    return "Add a rolling average, inspect the periods with the largest changes, and compare the trend across a meaningful categorical dimension."


def _area_metric_caution(y_column: str, aggregation: str | None) -> str:
    normalized = normalize_column_name(y_column)
    if normalized in {"totalrevenue", "revenue", "sales"} or "revenue" in normalized or "sales" in normalized:
        return "Monthly totals may change because of order volume, units sold, or unit prices; the area does not show profitability by itself."
    if "profit" in normalized:
        return "Profit can be affected by revenue and cost changes, and the filled area does not show profit margin by itself."
    if "energy" in normalized or "consumption" in normalized:
        return "Energy consumption should often be normalized by building area, occupancy, or operating hours before comparing magnitude."
    if "count" in normalized or aggregation == "count":
        return "Counts show event volume and do not show value, severity, or rate without a denominator."
    if aggregation == "mean":
        return "Averages may hide changes in sample size and within-period variation."
    return "The chart emphasizes the magnitude of the aggregated metric over the ordered range; interpretation depends on the metric definition and aggregation method."


def _area_next_step(y_column: str, aggregation: str | None) -> str:
    normalized = normalize_column_name(y_column)
    if normalized in {"totalrevenue", "revenue", "sales"} or "revenue" in normalized or "sales" in normalized:
        return "Add a rolling average and compare total profit, order count, and units sold over the same periods."
    if "energy" in normalized or "consumption" in normalized:
        return "Normalize consumption by building area or occupancy and compare the pattern across building types."
    if "profit" in normalized:
        return "Compare revenue, total cost, and margin over the same periods."
    if "count" in normalized or aggregation == "count":
        return "Compare counts with a rate or denominator and inspect the longest high- and low-volume periods."
    return "Compare the pattern with a normalized or related measure and examine whether the same trend appears across relevant subgroups."


def build_single_area_fallback(evidence: SingleAreaEvidence) -> ChartInsight:
    """Create a magnitude-aware deterministic insight for a single area chart."""
    metric_label = display_name(evidence.y_column)
    metric_phrase = build_area_metric_phrase(
        evidence.y_column,
        evidence.aggregation,
        evidence.time_granularity,
    )
    if evidence.x_axis_type == "categorical":
        key = f"{metric_label} is shown on an unordered x-axis, so the area chart is weak evidence for movement."
        return ChartInsight(
            chart_title=evidence.chart_title,
            key_finding=key,
            supporting_evidence=f"{evidence.valid_rows:,} displayed point(s) were available, but the x-axis is categorical.",
            interpretation="Area charts work best for ordered sequences because the filled shape implies continuity.",
            caution="The x-axis is unordered, so the filled area may imply a sequence that is not present in the data.",
            recommended_next_step="Use a bar chart for categorical comparison or choose a date, period, or numeric x-axis.",
            evidence_strength="low",
            evidence=evidence,
        )

    endpoint = _endpoint_direction(evidence.endpoint_change)
    volatility = evidence.volatility_level or "variable"
    key = (
        f"{metric_label} {endpoint}, while the filled area shows "
        f"{volatility} magnitude across the displayed range."
    )
    facts = []
    date_summary = _date_summary_sentence(evidence, evidence.y_column)
    if date_summary:
        facts.append(date_summary)
    if evidence.start_period_label and evidence.end_period_label:
        endpoint_sentence = (
            f"{metric_phrase.capitalize()} moved from {_fmt(evidence.start_value, evidence.y_column)} "
            f"in {evidence.start_period_label} to {_fmt(evidence.end_value, evidence.y_column)} "
            f"in {evidence.end_period_label}"
        )
        if evidence.endpoint_change_percent is not None:
            endpoint_sentence += f" ({_pct(evidence.endpoint_change_percent)} relative to the starting value)"
        endpoint_sentence += "."
        facts.append(endpoint_sentence)
    if evidence.peak_period_label and evidence.trough_period_label:
        facts.append(
            f"The highest area occurred at {_fmt(evidence.peak_value, evidence.y_column)} in {evidence.peak_period_label}, "
            f"while the lowest was {_fmt(evidence.trough_value, evidence.y_column)} in {evidence.trough_period_label}."
        )
    if evidence.longest_above_average_run:
        facts.append(
            f"The longest sustained elevated run lasted {evidence.longest_above_average_run} period(s) above the average level."
        )
    if evidence.longest_below_average_run:
        facts.append(
            f"The longest reduced run lasted {evidence.longest_below_average_run} period(s) below the average level."
        )
    if evidence.volatility_level and evidence.trend_strength:
        facts.append(
            f"The fitted trend is {evidence.trend_strength}, and period-to-period movement is {evidence.volatility_level}."
        )
    if evidence.missing_period_labels:
        facts.append(f"Missing periods include {', '.join(evidence.missing_period_labels[:3])}.")
    support = " ".join(dict.fromkeys(facts[:5]))

    interpretation = (
        "The filled area emphasizes sustained magnitude, so the chart is useful for seeing whether high values persist "
        "rather than appearing as a single isolated spike."
    )
    if evidence.volatility_level == "highly volatile":
        interpretation += " The repeated swings mean the endpoint change should not be read as steady growth."

    cautions = [
        "The filled area emphasizes magnitude and can make fluctuations appear more dramatic than a line alone.",
        _area_metric_caution(evidence.y_column, evidence.aggregation),
    ]
    if not evidence.baseline_is_zero:
        cautions.append("The y-axis baseline is not zero, so the filled area may exaggerate differences in magnitude.")
    if evidence.negative_value_count:
        cautions.append("Negative values mean the fill crosses zero, which makes positive and negative regions harder to compare.")
    if evidence.missing_period_labels or evidence.irregular_intervals:
        cautions.append("Missing or irregular periods can make the filled area appear more continuous than the data supports.")
    cautions.append("The available history is not sufficient by itself to confirm seasonality.")

    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=" ".join(cautions),
        recommended_next_step=_area_next_step(evidence.y_column, evidence.aggregation),
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def build_stacked_area_fallback(evidence: StackedAreaEvidence) -> ChartInsight:
    """Create a deterministic insight for stacked area composition."""
    metric_label = display_name(evidence.y_column)
    dominant = evidence.dominant_stack_overall or "the leading component"
    key = (
        f"Total {metric_label.lower()} changed across the displayed range, while "
        f"{dominant} was the largest contributor."
    )
    facts = []
    if evidence.start_total is not None and evidence.end_total is not None:
        sentence = (
            f"The combined total moved from {_fmt(evidence.start_total, evidence.y_column)} "
            f"to {_fmt(evidence.end_total, evidence.y_column)}"
        )
        if evidence.total_change_percent is not None:
            sentence += f" ({_pct(evidence.total_change_percent)} relative to the starting total)"
        sentence += "."
        facts.append(sentence)
    if evidence.peak_period_label and evidence.trough_period_label:
        facts.append(
            f"The total peaked at {_fmt(evidence.peak_total, evidence.y_column)} in {evidence.peak_period_label} "
            f"and was lowest at {_fmt(evidence.trough_total, evidence.y_column)} in {evidence.trough_period_label}."
        )
    if evidence.dominant_stack_overall:
        facts.append(
            f"{evidence.dominant_stack_overall} contributed {_pct(evidence.dominant_stack_share)} of the displayed stacked total."
        )
    if evidence.stack_with_largest_growth:
        facts.append(f"{evidence.stack_with_largest_growth} had the largest absolute growth from start to end.")
    if evidence.missing_combinations:
        facts.append(f"{len(evidence.missing_combinations)} period-component combination(s) are missing.")
    support = " ".join(dict.fromkeys(facts[:5]))
    interpretation = (
        "The stacked area separates combined movement from component contribution, showing whether total change is broad-based "
        "or driven by one component."
    )
    caution = (
        "Only the bottom stacked layer has a constant baseline, so precise comparison of upper components is harder. "
        f"{_area_metric_caution(evidence.y_column, evidence.aggregation)}"
    )
    if evidence.negative_value_count:
        caution += " Negative stacked values make component area comparisons especially difficult."
    if evidence.missing_combinations:
        caution += " Missing component-period combinations can change apparent composition."
    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=caution,
        recommended_next_step="Use small-multiple line charts or normalized shares to compare components more precisely.",
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def _dual_line_endpoint_clause(direction: str | None, start_label: Any | None) -> str:
    reference = f"its {start_label} level" if start_label is not None else "its starting level"
    if direction == "higher":
        return f"ending above {reference}"
    if direction == "lower":
        return f"ending below {reference}"
    if direction == "approximately unchanged":
        return f"ending close to {reference}"
    return f"ending relative to {reference}"


def _dual_line_ordered_next_step(evidence: DualLineEvidence) -> str:
    normalized = {normalize_column_name(evidence.primary_y_column), normalize_column_name(evidence.secondary_y_column)}
    has_revenue = any("revenue" in item or "sales" in item for item in normalized)
    has_profit = any("profit" in item for item in normalized)
    has_cost = any("cost" in item for item in normalized)
    has_units = any("unit" in item or "quantity" in item for item in normalized)
    if has_revenue and has_profit:
        return "Plot profit margin and total cost over the same periods to see whether profit is expanding because revenue rose or costs improved."
    if has_revenue and has_cost:
        return "Add total profit and profit margin over the same periods to separate growth from margin pressure."
    if has_revenue and has_units:
        return "Calculate revenue per unit over the same periods to separate volume changes from price or product mix."
    return "Create a normalized index and inspect the divergence periods to identify where the two metrics separated most."


def _dual_combination_next_step(evidence: DualCombinationEvidence) -> str:
    normalized = {normalize_column_name(evidence.bar_y_column), normalize_column_name(evidence.line_y_column)}
    has_revenue = any("revenue" in item or "sales" in item for item in normalized)
    has_profit = any("profit" in item for item in normalized)
    has_cost = any("cost" in item for item in normalized)
    has_margin = any("margin" in item or "rate" in item or "percent" in item for item in normalized)
    has_units = any("unit" in item or "quantity" in item or "volume" in item for item in normalized)
    has_price = any("price" in item for item in normalized)
    has_employee = any("employee" in item or "headcount" in item or "salary" in item for item in normalized)
    has_energy = any("energy" in item or "consumption" in item or "temperature" in item for item in normalized)
    if has_revenue and (has_profit or has_margin):
        return "Compare profit margin, total cost, and revenue over the same x-values to separate scale effects from profitability."
    if has_revenue and has_cost:
        return "Add total profit and margin over the same x-values to see whether revenue growth translates into profitability."
    if has_units and (has_price or has_revenue):
        return "Calculate revenue per unit or price per unit for the same x-values to separate volume from pricing."
    if has_employee:
        return "Compare headcount, salary distribution, and compensation per employee across the same x-values."
    if has_energy:
        return "Normalize energy consumption by area, operating hours, or occupancy and compare it with temperature over the same periods."
    return "Create a normalized index or ratio view and inspect the x-values where the bar and line metrics diverge most."


def build_dual_combination_fallback(evidence: DualCombinationEvidence) -> ChartInsight:
    """Create a deterministic insight for dual-combination bar-plus-line charts."""
    bar_phrase = build_metric_phrase(evidence.bar_y_column, evidence.bar_aggregation)
    line_phrase = build_metric_phrase(evidence.line_y_column, evidence.line_aggregation)
    bar_name = display_name(evidence.bar_y_column)
    line_name = display_name(evidence.line_y_column)
    x_name = display_name(evidence.x_column)
    is_ordered = evidence.x_axis_type in {"datetime", "numeric", "ordered_period"}

    if evidence.same_highest_x:
        key = f"{evidence.bar_highest_x} has both the highest {bar_phrase} and the highest {line_phrase}."
    elif evidence.relationship_strength and evidence.relationship_direction:
        axis_text = "over the ordered x-axis" if is_ordered else f"across {x_name.lower() or 'categories'}"
        key = (
            f"{bar_name} and {line_name} show a {evidence.relationship_strength} "
            f"{evidence.relationship_direction} relationship {axis_text}."
        )
    else:
        key = f"{bar_name} and {line_name} highlight different x-values in the Dual Combination chart."

    facts = []
    if evidence.bar_highest_x is not None and evidence.bar_lowest_x is not None:
        facts.append(
            f"The bars show {bar_phrase}: highest at {evidence.bar_highest_x} "
            f"({_fmt(evidence.bar_highest_value, evidence.bar_y_column)}) and lowest at {evidence.bar_lowest_x} "
            f"({_fmt(evidence.bar_lowest_value, evidence.bar_y_column)})."
        )
    if evidence.line_highest_x is not None and evidence.line_lowest_x is not None:
        facts.append(
            f"The line shows {line_phrase}: highest at {evidence.line_highest_x} "
            f"({_fmt(evidence.line_highest_value, evidence.line_y_column)}) and lowest at {evidence.line_lowest_x} "
            f"({_fmt(evidence.line_lowest_value, evidence.line_y_column)})."
        )
    if is_ordered and evidence.bar_start_x is not None and evidence.line_start_x is not None:
        endpoint = (
            f"{bar_name} was {_fmt(evidence.bar_start_value, evidence.bar_y_column)} at {evidence.bar_start_x} "
            f"and {_fmt(evidence.bar_end_value, evidence.bar_y_column)} at {evidence.bar_end_x}"
        )
        if evidence.bar_change_percent is not None:
            endpoint += f" ({_pct(evidence.bar_change_percent)} from the start)"
        endpoint += (
            f"; {line_name} was {_fmt(evidence.line_start_value, evidence.line_y_column)} at {evidence.line_start_x} "
            f"and {_fmt(evidence.line_end_value, evidence.line_y_column)} at {evidence.line_end_x}"
        )
        if evidence.line_change_percent is not None:
            endpoint += f" ({_pct(evidence.line_change_percent)} from the start)"
        endpoint += "."
        facts.append(endpoint)
    if is_ordered and evidence.comparable_transition_count:
        facts.append(
            f"The two metrics shared direction in {evidence.aligned_direction_count or 0} of "
            f"{evidence.comparable_transition_count} comparable transitions ({_pct(evidence.aligned_direction_percent)}) "
            f"and differed in {evidence.opposite_direction_count or 0} ({_pct(evidence.opposite_direction_percent)})."
        )
    elif evidence.pearson_correlation is not None and evidence.spearman_correlation is not None:
        facts.append(
            f"Pearson correlation is {evidence.pearson_correlation:.3f} and Spearman correlation is "
            f"{evidence.spearman_correlation:.3f}, indicating a {evidence.relationship_strength or 'measurable'} "
            f"{evidence.relationship_direction or 'neutral'} association."
        )
    if evidence.same_highest_x is True:
        facts.append(f"High bars coincide with the high line value at {evidence.bar_highest_x}.")
    elif evidence.largest_normalized_divergence_x is not None:
        facts.append(f"The largest normalized separation occurs at {evidence.largest_normalized_divergence_x}.")
    if evidence.known_metric_relationship:
        facts.append(evidence.known_metric_relationship)
    if evidence.top_n_applied:
        facts.append(f"The explanation covers the top {evidence.top_n_applied} displayed x-values.")
    if evidence.missing_x_values:
        facts.append(f"Missing periods include {', '.join(str(item) for item in evidence.missing_x_values[:3])}.")
    support = " ".join(dict.fromkeys(facts[:5]))

    if evidence.unit_relationship == "same unit":
        interpretation = (
            "Because the metrics use the same unit, compare both the bar magnitude and the line value at each x-value; "
            "large gaps are analytically meaningful."
        )
    elif evidence.unit_relationship == "different unit":
        interpretation = (
            "Because the metrics use different units, read the relationship through paired highs/lows, correlation, "
            "and normalized divergence rather than visual height."
        )
    else:
        interpretation = (
            "The bars and line should be interpreted from their metric definitions first, then compared through their paired x-values."
        )
    if evidence.aggregation_relationship != "same aggregation":
        interpretation += f" {evidence.aggregation_relationship}"

    cautions = []
    if evidence.unit_relationship == "different unit":
        cautions.append("Separate Y-axis scales can exaggerate or mute the apparent relationship between the bars and line.")
    elif evidence.unit_relationship == "same unit":
        cautions.append("A shared-scale view or difference chart can make absolute gaps easier to judge.")
    if evidence.pearson_correlation is not None:
        cautions.append("Correlation describes association, not causation.")
    if evidence.paired_point_count < 5:
        cautions.append("The small number of paired x-values limits relationship evidence.")
    if evidence.irregular_intervals or evidence.missing_x_values:
        cautions.append("Missing or irregular periods can make ordered comparisons incomplete.")
    caution = " ".join(cautions) or None

    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=caution,
        recommended_next_step=_dual_combination_next_step(evidence),
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def _scatter_next_step(evidence: ScatterEvidence) -> str:
    x_norm = normalize_column_name(evidence.x_column)
    y_norm = normalize_column_name(evidence.y_column)
    if {x_norm, y_norm} == {"unitssold", "totalrevenue"}:
        return "Color the points by item type or unit-price band and compare revenue per unit to separate volume from pricing effects."
    if {x_norm, y_norm} == {"totalrevenue", "totalprofit"}:
        return "Calculate profit margin and color points by item type or sales channel to identify stronger or weaker profit conversion."
    if {x_norm, y_norm} == {"unitcost", "unitprice"}:
        return "Calculate unit margin and compare it by item type to identify unusual pricing or cost patterns."
    if evidence.color_column:
        return f"Compare the relationship within each {display_name(evidence.color_column).lower()} group and inspect influential outliers."
    if evidence.outlier_count:
        return "Inspect the influential outliers and compare the relationship with and without those observations."
    return "Color the points by a meaningful category, inspect influential outliers, and compare the relationship within each subgroup."


def build_scatter_fallback(evidence: ScatterEvidence) -> ChartInsight:
    """Create a pattern-aware deterministic insight for scatter plots."""
    x_name = display_name(evidence.x_column)
    y_name = display_name(evidence.y_column)
    strength = evidence.relationship_strength or "unclear"
    direction = evidence.relationship_direction or "unclear"
    form = evidence.relationship_form or "no clear relationship"
    key = (
        f"{x_name} and {y_name} show a {strength} {direction} association, "
        f"with a {form} point pattern."
    )
    if evidence.banding_detected and evidence.relationship_form != "fan-shaped":
        key = (
            f"{x_name} and {y_name} show a {strength} {direction} association, "
            "split into distinct bands rather than one uniform cloud."
        )
    elif form == "weak or diffuse":
        key = f"{x_name} and {y_name} have a weak, diffuse scatter pattern with no tight relationship."
    elif form == "fan-shaped":
        key = f"{x_name} and {y_name} show a {strength} {direction} association with widening vertical spread."
    elif form == "clustered":
        key = f"{x_name} and {y_name} show a {strength} {direction} association with visible group separation."

    facts = [f"Across {evidence.displayed_point_count:,} displayed observations, {evidence.valid_point_count:,} finite x-y pair(s) were available."]
    if evidence.pearson_correlation is not None and evidence.spearman_correlation is not None:
        facts.append(
            f"Pearson correlation is {evidence.pearson_correlation:.3f} and Spearman correlation is "
            f"{evidence.spearman_correlation:.3f}, indicating a {strength} {direction} association."
        )
    if evidence.r_squared is not None and evidence.relationship_form in {"approximately linear", "approximately linear with influential points"}:
        facts.append(
            f"A linear model explains about {_pct(evidence.r_squared * 100)} of the observed variation in {y_name.lower()}."
        )
    if evidence.relationship_form == "monotonic but non-linear":
        facts.append("Spearman correlation is materially stronger than Pearson, so the relationship is clearer by rank than by a straight line.")
    if evidence.banding_detected and evidence.relationship_form != "fan-shaped":
        band_text = f"The points form {evidence.band_count or 'several'} repeated bands."
        if evidence.known_metric_relationship:
            band_text += f" {evidence.known_metric_relationship}"
        facts.append(band_text)
    elif evidence.heteroscedasticity_detected:
        facts.append(f"The spread changes across the x-axis: {evidence.variance_pattern}.")
    elif evidence.cluster_count:
        facts.append(f"The displayed observations separate into about {evidence.cluster_count} visible group(s).")
    if evidence.outlier_count:
        facts.append(
            f"{evidence.outlier_count} point(s) lie far from the main relationship; "
            f"{evidence.influential_point_count} also have high x-leverage."
        )
    if evidence.color_group_summary:
        groups = ", ".join(list(evidence.color_group_summary)[:3])
        facts.append(f"Color groups are available for {groups}, allowing subgroup comparison.")
    if evidence.sampled:
        facts.append(
            f"The chart displays a sample of {evidence.displayed_point_count:,} from "
            f"{evidence.valid_point_count:,} valid observations."
        )
    support = " ".join(dict.fromkeys(facts[:5]))

    if evidence.mathematical_dependency:
        interpretation = evidence.mathematical_dependency
        if (
            evidence.banding_detected
            and evidence.known_metric_relationship
            and "unit price" in evidence.known_metric_relationship.lower()
        ):
            interpretation += " Observations with similar unit prices fall along similar slopes."
    elif evidence.relationship_form == "monotonic but non-linear":
        interpretation = "The variables tend to change together by rank, but a single straight line does not summarize the pattern well."
    elif evidence.relationship_form == "fan-shaped":
        interpretation = f"The relationship becomes less consistent as {x_name.lower()} changes because vertical spread is not constant."
    elif evidence.relationship_form == "weak or diffuse":
        interpretation = "Substantial spread remains at similar x-values, so additional variables likely explain much of the variation."
    else:
        interpretation = "The chart shows an observed association; compare subgroups and unusual observations before treating it as a stable pattern."

    cautions = []
    if evidence.mathematical_dependency:
        cautions.append("This structural relationship should not be interpreted as independent causal evidence.")
    elif evidence.pearson_correlation is not None:
        cautions.append("Correlation describes association and does not establish causation.")
    if evidence.sampled:
        cautions.append("Because the chart is sampled, rare patterns may not be visible.")
    elif evidence.influential_point_count:
        cautions.append("Influential points may materially affect Pearson correlation and the fitted line.")
    elif evidence.color_group_summary:
        cautions.append("The overall relationship may combine groups with different internal patterns.")
    caution = " ".join(cautions[:2]) or None

    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=caution,
        recommended_next_step=_scatter_next_step(evidence),
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def _circle_view_next_step(evidence: CircleViewEvidence) -> str:
    normalized = {
        normalize_column_name(evidence.x_column),
        normalize_column_name(evidence.y_column),
        normalize_column_name(evidence.size_column),
    }
    has_units = any("unit" in item or "quantity" in item for item in normalized)
    has_revenue = any("revenue" in item or "sales" in item for item in normalized)
    has_profit = any("profit" in item for item in normalized)
    has_price = any("price" in item for item in normalized)
    if has_units and has_revenue and has_profit:
        group_text = f" by {display_name(evidence.color_column).lower()}" if evidence.color_column else ""
        return f"Calculate profit margin and revenue per unit{group_text}, then inspect high-revenue observations with unusually small or large profit bubbles."
    if has_revenue and has_profit:
        return "Compare profit margin and revenue per unit to separate high-volume growth from stronger profit conversion."
    if has_units and has_price and has_profit:
        return "Compare profit per unit across groups to identify whether large profit bubbles are driven by price, volume, or margin."
    if evidence.color_column:
        return f"Compare {display_name(evidence.size_column).lower()} within each {display_name(evidence.color_column).lower()} group and inspect similarly positioned circles with unusual size."
    return "Compare the size metric within meaningful subgroups and inspect circles with similar x-y positions but unusually different bubble sizes."


def build_circle_view_fallback(evidence: CircleViewEvidence) -> ChartInsight:
    """Create a deterministic insight for Circle View charts."""
    x_name = display_name(evidence.x_column)
    y_name = display_name(evidence.y_column)
    size_name = display_name(evidence.size_column)
    strength = evidence.xy_relationship_strength or "unclear"
    direction = evidence.xy_relationship_direction or "unclear"
    size_relation = "is not strongly tied to either axis"
    if evidence.size_y_relationship_strength and evidence.size_y_relationship_strength not in {"very weak", "weak"}:
        size_relation = f"tends to be {evidence.size_y_relationship_direction} with {y_name.lower()}"
    if evidence.size_x_relationship_strength and evidence.size_x_relationship_strength not in {"very weak", "weak"}:
        if evidence.size_y_relationship_strength and evidence.size_y_relationship_strength not in {"very weak", "weak"}:
            size_relation = f"tends to be associated with both {x_name.lower()} and {y_name.lower()}"
        else:
            size_relation = f"tends to be {evidence.size_x_relationship_direction} with {x_name.lower()}"
    key = (
        f"{x_name} and {y_name} show a {strength} {direction} relationship, "
        f"while {size_name.lower()} {size_relation}."
    )
    if evidence.large_bubble_concentration and evidence.large_bubble_concentration != "distributed":
        key += f" The largest bubbles concentrate in the {evidence.large_bubble_concentration} region."
    elif evidence.color_group_summary and evidence.group_with_largest_single_bubble:
        key += f" {evidence.group_with_largest_single_bubble} contains the largest bubble."

    facts = [
        f"Across {evidence.displayed_point_count:,} displayed observations, Circle View encodes {x_name.lower()} on x, {y_name.lower()} on y, and {size_name.lower()} by bubble area."
    ]
    if evidence.pearson_xy is not None and evidence.spearman_xy is not None:
        facts.append(
            f"Pearson correlation is {evidence.pearson_xy:.3f} and Spearman correlation is {evidence.spearman_xy:.3f}, "
            f"indicating a {strength} {direction} x-y relationship."
        )
    if evidence.r_squared_xy is not None and evidence.xy_relationship_form in {"approximately linear", "approximately linear with influential points"}:
        facts.append(f"A linear model explains about {_pct(evidence.r_squared_xy * 100)} of the variation in {y_name.lower()}.")
    if evidence.pearson_size_x is not None and evidence.pearson_size_y is not None:
        facts.append(
            f"Bubble size correlation is {evidence.pearson_size_x:.3f} with {x_name.lower()} and "
            f"{evidence.pearson_size_y:.3f} with {y_name.lower()}."
        )
    if evidence.largest_bubble_size is not None:
        group_text = f" in {evidence.largest_bubble_group}" if evidence.largest_bubble_group else ""
        facts.append(
            f"The largest bubble is {_fmt(evidence.largest_bubble_size, evidence.size_column)}{group_text}, "
            f"at {x_name.lower()} {_fmt(evidence.largest_bubble_x, evidence.x_column)} and "
            f"{y_name.lower()} {_fmt(evidence.largest_bubble_y, evidence.y_column)}."
        )
    if evidence.large_bubble_concentration and evidence.large_bubble_concentration != "distributed":
        facts.append(f"Large bubbles are concentrated in the {evidence.large_bubble_concentration} median-based quadrant.")
    if evidence.color_group_summary:
        facts.append(
            f"{evidence.group_with_largest_total_size} has the largest combined {size_name.lower()}, "
            f"and {evidence.group_with_largest_single_bubble} contains the largest single bubble."
        )
    if evidence.similar_position_different_size_count:
        facts.append(
            f"{evidence.similar_position_different_size_count} x-y bin(s) contain similarly positioned circles with materially different bubble sizes."
        )
    if evidence.banding_detected:
        facts.append(f"The circles form {evidence.band_count or 'several'} repeated bands.")
    if evidence.overlap_level == "high":
        facts.append("Bubble overlap is high in dense regions.")
    support = " ".join(dict.fromkeys(facts[:5]))

    if evidence.mathematical_dependency:
        interpretation = evidence.mathematical_dependency
    elif evidence.similar_position_different_size_count:
        interpretation = (
            f"The {size_name.lower()} measure adds information not fully captured by {x_name.lower()} and {y_name.lower()}, "
            "because similar x-y positions can have different bubble areas."
        )
    elif evidence.size_y_relationship_strength and evidence.size_y_relationship_strength not in {"very weak", "weak"}:
        interpretation = f"The third measure is most aligned with {y_name.lower()}, so vertical position and bubble area reinforce each other."
    else:
        interpretation = (
            f"The chart combines three numeric measures; interpret the x-y relationship together with where {size_name.lower()} bubbles become unusually large or small."
        )
    if evidence.high_xy_small_size_count:
        interpretation += f" {evidence.high_xy_small_size_count} high-x/high-y circle(s) have relatively small {size_name.lower()}."
    if evidence.moderate_xy_large_size_count:
        interpretation += f" {evidence.moderate_xy_large_size_count} mid-position circle(s) have unusually large {size_name.lower()}."

    cautions = ["Bubble area represents the size metric, but area differences are harder to compare precisely than positions along an axis."]
    if evidence.mathematical_dependency:
        cautions.append("The structural relationship should not be interpreted as independent causal evidence.")
    elif evidence.overlap_level == "high":
        cautions.append("Overlapping bubbles may hide smaller observations in dense regions.")
    elif evidence.pearson_xy is not None:
        cautions.append("Correlation describes association and does not establish causation.")

    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=" ".join(cautions[:2]),
        recommended_next_step=_circle_view_next_step(evidence),
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def _histogram_next_step(evidence: HistogramEvidence) -> str:
    normalized = normalize_column_name(evidence.value_column)
    if "profit" in normalized:
        return "Compare profit distributions by item type or sales channel, then calculate profit margin for observations in the upper tail."
    if "revenue" in normalized or "sales" in normalized:
        return "Compare revenue distributions by item type and inspect units sold and unit price for the highest-revenue observations."
    if "unitprice" in normalized or "price" in normalized:
        return "Segment the histogram by item type to determine whether visible peaks correspond to distinct product-price levels."
    if "unit" in normalized or "quantity" in normalized:
        return "Compare units-sold distributions by item type and inspect whether high-volume observations also produce strong revenue per unit."
    if evidence.zero_share and evidence.zero_share >= 20:
        return "Analyze zero and non-zero values separately before comparing the non-zero distribution across groups."
    return "Compare the distribution across a meaningful categorical group and inspect observations beyond the outlier thresholds."


def build_histogram_fallback(evidence: HistogramEvidence) -> ChartInsight:
    """Create a deterministic distribution insight for histograms."""
    metric_name = display_name(evidence.value_column)
    metric_lower = metric_name.lower()
    shape = evidence.skew_strength or "unknown shape"
    direction = evidence.skew_direction or "unknown"
    if evidence.multimodal:
        key = f"{metric_name} shows possible multiple peaks rather than one typical range."
    elif direction == "approximately symmetric":
        key = f"{metric_name} is approximately symmetric, with mean and median close together."
    else:
        key = (
            f"{metric_name} is {shape} {direction}, with a {evidence.tail_description or 'tail pattern'} "
            f"and a median of {_fmt(evidence.median, evidence.value_column)}."
        )

    facts = [
        f"Across {evidence.displayed_value_count:,} valid observations, the median is {_fmt(evidence.median, evidence.value_column)} and the mean is {_fmt(evidence.mean, evidence.value_column)}."
    ]
    if evidence.q1 is not None and evidence.q3 is not None:
        facts.append(
            f"The middle 50% spans {_fmt(evidence.q1, evidence.value_column)} to {_fmt(evidence.q3, evidence.value_column)} "
            f"(IQR {_fmt(evidence.iqr, evidence.value_column)})."
        )
    if evidence.modal_bin_start is not None and evidence.modal_bin_end is not None:
        facts.append(
            f"The most populated bin is {_fmt(evidence.modal_bin_start, evidence.value_column)} to "
            f"{_fmt(evidence.modal_bin_end, evidence.value_column)}, containing {evidence.modal_bin_count} observation(s) "
            f"({_pct(evidence.modal_bin_share)})."
        )
    if evidence.p90 is not None and evidence.p95 is not None:
        facts.append(
            f"90% of observations are at or below {_fmt(evidence.p90, evidence.value_column)}, while the upper 5% starts above {_fmt(evidence.p95, evidence.value_column)}."
        )
    if evidence.potential_outlier_count:
        direction_text = "upper-tail" if evidence.upper_outlier_count >= evidence.lower_outlier_count else "lower-tail"
        facts.append(
            f"The {evidence.outlier_method} flags {evidence.potential_outlier_count} potential {direction_text} outlier(s) "
            f"({_pct(evidence.potential_outlier_share)})."
        )
    if evidence.zero_share and evidence.zero_share >= 20:
        facts.append(f"Zero values account for {_pct(evidence.zero_share)} of observations.")
    if evidence.negative_count:
        facts.append(f"{evidence.negative_count} observation(s) are below zero ({_pct(evidence.negative_share)}).")
    if evidence.multimodal and evidence.multimodal_evidence:
        facts.append(evidence.multimodal_evidence)
    support = " ".join(dict.fromkeys(facts[:5]))

    mean_median_gap = None
    if evidence.mean is not None and evidence.median is not None:
        denominator = max(abs(evidence.median), abs(evidence.mean), 1.0)
        mean_median_gap = (evidence.mean - evidence.median) / denominator
    if direction == "right-skewed" and mean_median_gap is not None and mean_median_gap > 0.05:
        interpretation = f"The mean is above the median because a smaller number of high {metric_lower} values pull the average upward."
    elif direction == "left-skewed" and mean_median_gap is not None and mean_median_gap < -0.05:
        interpretation = f"The mean is below the median because unusually low {metric_lower} values pull the average downward."
    elif direction == "approximately symmetric":
        interpretation = "The mean and median are similar, which is consistent with an approximately symmetric distribution."
    elif evidence.zero_share and evidence.zero_share >= 20:
        interpretation = "The distribution contains a large point mass at zero, so mean and standard deviation alone are not enough to describe a typical value."
    else:
        interpretation = f"The center and spread show how {metric_lower} varies across displayed observations, while the tails show where unusually small or large values occur."
    if "profit" in normalize_column_name(evidence.value_column):
        interpretation += " Total profit combines revenue and cost effects, so the histogram alone does not reveal whether extremes come from volume or margin."

    cautions = ["Histogram shape depends on the selected bin width, so smaller peaks or gaps may change under a different bin setting."]
    if evidence.potential_outlier_count:
        cautions.append("Potential outliers are statistical flags and may be valid observations that need contextual review.")
    elif evidence.sampled:
        cautions.append("The histogram displays a sample of valid observations, so rare values may be underrepresented.")
    elif evidence.negative_count:
        cautions.append("The distribution crosses zero, so positive and negative values should be interpreted separately when the metric semantics matter.")

    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=" ".join(cautions[:2]),
        recommended_next_step=_histogram_next_step(evidence),
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def _box_plot_next_step(evidence: BoxPlotEvidence) -> str:
    normalized = normalize_column_name(evidence.y_column)
    x_name = display_name(evidence.x_column)
    breakdown_name = display_name(evidence.breakdown_column)
    combination_name = f"{x_name}-{breakdown_name} combinations" if breakdown_name else f"{x_name} categories"
    comparison_group = breakdown_name.lower() if breakdown_name else x_name.lower()
    if "revenue" in normalized or "sales" in normalized:
        return f"Compare units sold and unit price for the highest- and lowest-median {combination_name}."
    if "profit" in normalized:
        return f"Compare profit margin and total cost across the {comparison_group} groups with the largest median and IQR differences."
    return f"Inspect the {x_name.lower()} categories with the largest breakdown differences and compare their sample sizes and related numeric measures."


def build_box_plot_fallback(evidence: BoxPlotEvidence) -> ChartInsight:
    """Create a deterministic insight for grouped Box Plot charts."""
    metric_name = display_name(evidence.y_column)
    x_name = display_name(evidence.x_column)
    breakdown_name = display_name(evidence.breakdown_column)
    most_common_leader = None
    most_common_count = 0
    if evidence.breakdown_lead_counts:
        most_common_leader, most_common_count = max(evidence.breakdown_lead_counts.items(), key=lambda item: item[1])
    if evidence.breakdown_column and most_common_leader:
        consistency = (
            f"{most_common_leader} leads in {most_common_count} of {evidence.x_category_count} {x_name.lower()} categories"
            if most_common_count > 1
            else f"leaders vary across {x_name.lower()} categories"
        )
        key = (
            f"{evidence.highest_median_combination} has the highest median {metric_name.lower()}, "
            f"while {consistency}."
        )
    else:
        key = (
            f"{evidence.highest_median_combination} has the highest median {metric_name.lower()}, "
            f"and {evidence.widest_iqr_combination} has the widest middle-50% spread."
        )

    facts = []
    if evidence.highest_median_combination and evidence.lowest_median_combination:
        facts.append(
            f"The highest median box is {evidence.highest_median_combination} at {_fmt(evidence.highest_median_value, evidence.y_column)}, "
            f"while the lowest is {evidence.lowest_median_combination} at {_fmt(evidence.lowest_median_value, evidence.y_column)}."
        )
    if evidence.widest_iqr_combination:
        facts.append(
            f"{evidence.widest_iqr_combination} has the widest IQR at {_fmt(evidence.widest_iqr_value, evidence.y_column)}, "
            f"while {evidence.narrowest_iqr_combination} has the narrowest at {_fmt(evidence.narrowest_iqr_value, evidence.y_column)}."
        )
    if most_common_leader:
        facts.append(
            f"{most_common_leader} is the most frequent breakdown leader, ranking highest in {most_common_count} of {evidence.x_category_count} {x_name.lower()} categories."
        )
    if evidence.x_categories_with_ranking_changes:
        example = evidence.x_categories_with_ranking_changes[0]
        facts.append(
            f"Breakdown rankings change across {x_name.lower()} categories; for example, {example} is led by {evidence.breakdown_leader_by_x.get(example)}."
        )
    if evidence.total_potential_outlier_count:
        facts.append(
            f"The 1.5 x IQR rule flags {evidence.total_potential_outlier_count} potential outlier(s) across {len(evidence.groups_with_outliers)} box(es)."
        )
    if evidence.unequal_sample_sizes:
        facts.append("Displayed boxes have unequal sample sizes, so quartile stability differs across combinations.")
    support = " ".join(dict.fromkeys(facts[:5]))

    if evidence.breakdown_column:
        if evidence.x_category_count == 1:
            only_x = next(iter(evidence.breakdown_leader_by_x), x_name)
            interpretation = (
                f"Within {only_x}, compare breakdown boxes by median for typical {metric_name.lower()} and IQR for variability."
            )
        else:
            interpretation = (
                f"{breakdown_name or 'Breakdown'} effects vary by {x_name.lower()}; a group that leads for one {x_name.lower()} category "
                "does not necessarily lead for another."
            )
    else:
        interpretation = (
            f"Higher medians identify {x_name.lower()} categories with larger typical {metric_name.lower()}, while wider boxes show greater within-category variation."
        )
    if "revenue" in normalize_column_name(evidence.y_column):
        interpretation += " Total revenue reflects both units sold and unit price, so the box plot does not identify which factor drives the differences."
    elif "profit" in normalize_column_name(evidence.y_column):
        interpretation += " Total profit reflects both revenue and cost, so margin analysis is needed to explain the differences."

    cautions = []
    if evidence.unequal_sample_sizes:
        cautions.append("Some X-breakdown combinations contain fewer observations, so their medians and quartiles are less stable.")
    cautions.append("Overlapping boxes indicate shared value ranges but do not prove that the group distributions are statistically equivalent.")
    if evidence.total_potential_outlier_count:
        cautions.append("Potential outliers are statistical flags and may still be valid observations.")

    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=" ".join(cautions[:2]),
        recommended_next_step=_box_plot_next_step(evidence),
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def _symbol_map_location_word(evidence: SymbolMapEvidence, plural: bool = False) -> str:
    location_type = (evidence.location_type or "location").strip().lower()
    if location_type == "country":
        return "countries" if plural else "country"
    if location_type == "city":
        return "cities" if plural else "city"
    if location_type == "postal code":
        return "postal codes" if plural else "postal code"
    if location_type in {"state", "province", "region"}:
        return f"{location_type}s" if plural else location_type
    return "locations" if plural else "location"


def _symbol_map_next_step(evidence: SymbolMapEvidence) -> str:
    location_word = _symbol_map_location_word(evidence)
    location_column = display_name(evidence.location_column).lower()
    normalized_metric = normalize_column_name(evidence.value_column)
    color_name = (
        location_column
        if evidence.color_column == SYMBOL_MAP_COLOR_LOCATION
        else display_name(evidence.color_column).lower()
    )
    breakdown = color_name or location_column or location_word
    if "revenue" in normalized_metric or "sales" in normalized_metric:
        return (
            f"Rank the leading {location_word}s in a sorted bar chart, then compare total profit, "
            f"profit margin, units sold, and revenue per order by {breakdown}."
        )
    if "profit" in normalized_metric:
        return (
            f"Compare profit margin, total cost, and revenue for the leading {location_word}s "
            f"to separate scale from profitability."
        )
    if "unit" in normalized_metric or "quantity" in normalized_metric or "volume" in normalized_metric:
        return (
            f"Compare revenue per unit and product mix for the largest {location_word}s "
            f"to explain the geographic volume pattern."
        )
    if "price" in normalized_metric or "mean" in (evidence.aggregation or "") or "avg" in (evidence.aggregation or ""):
        return (
            f"Compare units sold and revenue per order for the leading {location_word}s "
            f"to distinguish price effects from volume effects."
        )
    return f"Compare the leading {location_word}s with a sorted table and break the map down by {breakdown}."


def build_symbol_map_fallback(evidence: SymbolMapEvidence) -> ChartInsight:
    """Create a deterministic geospatial insight for Symbol Map charts."""
    metric_phrase = build_metric_phrase(evidence.value_column, evidence.aggregation)
    metric_label = display_name(evidence.value_column)
    location_word = _symbol_map_location_word(evidence)
    locations_word = _symbol_map_location_word(evidence, plural=True)
    largest = evidence.largest_location or "The leading location"
    largest_value = _fmt(evidence.largest_value, evidence.value_column)
    largest_share = _pct(evidence.largest_share)
    distribution = evidence.geographic_distribution or "the displayed map shows a geographic ranking"
    concentration = evidence.concentration_level or "unclear concentration"

    if evidence.concentration_level == "widely dispersed" and evidence.largest_share is not None:
        key = (
            f"{largest} has the largest bubble, but its {largest_share} share shows "
            f"{metric_label.lower()} is geographically dispersed."
        )
    elif evidence.concentration_level == "highly concentrated":
        key = (
            f"{metric_label} is geographically concentrated on the map, led by "
            f"{largest} at {largest_value}."
        )
    else:
        key = (
            f"{largest} records the highest {metric_phrase}, while {distribution}."
        )

    facts = []
    if evidence.largest_location:
        leader_fact = f"{largest} ranks first at {largest_value}"
        if evidence.largest_share is not None:
            leader_fact += f" ({largest_share} of the displayed total)"
        if evidence.largest_group:
            leader_fact += f" in {evidence.largest_group}"
        facts.append(leader_fact + ".")
    if evidence.second_location and evidence.third_location:
        second_fact = (
            f"The next largest {locations_word} are {evidence.second_location} at "
            f"{_fmt(evidence.second_value, evidence.value_column)} and {evidence.third_location} at "
            f"{_fmt(evidence.third_value, evidence.value_column)}"
        )
        if evidence.top_three_share is not None:
            second_fact += f"; together the top three contribute {_pct(evidence.top_three_share)}"
        facts.append(second_fact + ".")
    elif evidence.second_location:
        facts.append(
            f"{evidence.second_location} is second at {_fmt(evidence.second_value, evidence.value_column)}."
        )
    if evidence.top_five_share is not None:
        facts.append(f"The top five {locations_word} contribute {_pct(evidence.top_five_share)} of the displayed total.")
    elif evidence.top_two_share is not None:
        facts.append(f"The top two {locations_word} contribute {_pct(evidence.top_two_share)} of the displayed total.")
    if evidence.highest_total_group:
        group_fact = (
            f"{evidence.highest_total_group} has the highest combined {metric_label.lower()} at "
            f"{_fmt(evidence.highest_total_group_value, evidence.value_column)}"
        )
        if evidence.highest_total_group_share is not None:
            group_fact += f" ({_pct(evidence.highest_total_group_share)})"
        if evidence.largest_group and evidence.highest_total_group != evidence.largest_group:
            group_fact += f", even though the largest individual {location_word} is in {evidence.largest_group}"
        facts.append(group_fact + ".")
    if evidence.highest_median_group:
        facts.append(
            f"{evidence.highest_median_group} has the highest median {metric_label.lower()} per {location_word} "
            f"at {_fmt(evidence.highest_median_group_value, evidence.value_column)}."
        )
    if evidence.group_with_most_top_locations:
        facts.append(
            f"{evidence.group_with_most_top_locations} contains "
            f"{evidence.group_with_most_top_locations_count} of the highest-ranked displayed {locations_word}."
        )
    if evidence.top_n_applied and evidence.original_location_count:
        facts.append(
            f"The map displays the top {evidence.displayed_location_count} of "
            f"{evidence.original_location_count} {locations_word} after the active filters."
        )
    else:
        facts.append(f"The map compares {evidence.displayed_location_count} displayed {locations_word}.")
    support = " ".join(dict.fromkeys(facts[:6]))

    interpretation = (
        f"Read this Symbol Map as a geographic distribution of bubble size: {concentration} values mean the largest "
        f"{locations_word} account for more of the displayed total, while the color grouping shows whether the pattern "
        "is regional or spread across groups."
    )
    if evidence.highest_total_group and evidence.group_with_most_top_locations:
        interpretation += (
            f" Here, {evidence.highest_total_group} leads by combined value, and "
            f"{evidence.group_with_most_top_locations} contains the most high-ranked displayed {locations_word}."
        )

    cautions = []
    if any("not additive" in warning for warning in evidence.warnings):
        cautions.append("Because the selected aggregation is not additive, bubble shares should not be read as parts of one total.")
    if evidence.marker_overlap_level in {"moderate", "high"}:
        cautions.append("Markers may overlap in dense areas, so smaller bubbles can be hidden or harder to compare.")
    cautions.append("Bubble area is less precise than a sorted bar chart, so use the map for geographic pattern first and exact ranking second.")
    if evidence.top_n_applied:
        cautions.append("The map is limited to the displayed top locations, so omitted locations are not represented in the pattern.")
    if evidence.date_column:
        cautions.append("The values reflect the active date filter.")
    caution = " ".join(dict.fromkeys(cautions[:3]))

    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=caution,
        recommended_next_step=_symbol_map_next_step(evidence),
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def _pair_label(pair: CorrelationPairEvidence | None) -> str:
    if pair is None:
        return "No pair"
    return f"{display_name(pair.variable_x)} and {display_name(pair.variable_y)}"


def _pair_fact(pair: CorrelationPairEvidence | None, method: str) -> str | None:
    if pair is None:
        return None
    return (
        f"{_pair_label(pair)} have a {method.title()} correlation of {pair.correlation:.2f} "
        f"({pair.strength} {pair.direction})."
    )


def _correlation_method_sentence(method: str) -> str:
    if method == "spearman":
        return "Spearman rank correlation measures monotonic rank association."
    if method == "kendall":
        return "Kendall correlation measures rank concordance."
    return "Pearson correlation measures linear association."


def _correlation_heatmap_next_step(evidence: CorrelationHeatmapEvidence) -> str:
    names = {normalize_column_name(column) for column in evidence.selected_columns}
    if {"unitssold", "unitprice", "unitcost", "totalrevenue", "totalcost", "totalprofit"} & names:
        prefix = "Exclude identifier fields, " if evidence.identifier_like_columns else ""
        return (
            f"{prefix}inspect scatter plots for the strongest non-formula pairs, then compare unit margin "
            "and profit margin to study pricing and profitability beyond accounting formulas."
        )
    if evidence.high_multicollinearity_pairs:
        return "Review variance inflation factors and remove, combine, or regularize highly redundant predictors before fitting a model."
    if evidence.strongest_negative_pair:
        return "Inspect scatter plots for the strongest positive and negative pairs, then test whether the relationships remain after controlling for related variables."
    return "Inspect scatter plots for the strongest nontrivial pairs and check whether nonlinear patterns or outliers affect the correlations."


def build_correlation_heatmap_fallback(evidence: CorrelationHeatmapEvidence) -> ChartInsight:
    """Create a deterministic structural insight for correlation Heatmaps."""
    method = evidence.correlation_method or "pearson"
    strong_count = len(evidence.strong_positive_pairs) + len(evidence.strong_negative_pairs)
    cluster_count = len(evidence.correlation_clusters)
    if evidence.strongest_pairs:
        first = evidence.strongest_pairs[0]
        if cluster_count:
            cluster = evidence.correlation_clusters[0]
            variables = ", ".join(display_name(value) for value in cluster.get("variables", [])[:5])
            key = (
                f"The Heatmap shows {strong_count} strong relationship(s), led by {_pair_label(first)}, "
                f"with a correlated cluster around {variables}."
            )
        elif strong_count:
            key = (
                f"The Heatmap shows {strong_count} strong relationship(s), led by {_pair_label(first)}, "
                f"while weaker pairs form the rest of the matrix."
            )
        else:
            key = "The Heatmap is mostly weak or moderate, with no strong correlation cluster in the displayed variables."
    else:
        key = "The Heatmap has too few valid non-diagonal correlations to identify a relationship structure."

    facts = []
    facts.append(
        f"The matrix uses {method.title()} correlation across {evidence.displayed_variable_count} variable(s), "
        f"creating {evidence.unique_pair_count} unique non-diagonal pair(s)."
    )
    for pair in evidence.strongest_pairs[:3]:
        fact = _pair_fact(pair, method)
        if fact:
            facts.append(fact)
    if evidence.strongest_negative_pair and evidence.strongest_negative_pair.absolute_correlation >= 0.30:
        facts.append(
            f"The strongest negative pair is {_pair_label(evidence.strongest_negative_pair)} at "
            f"{evidence.strongest_negative_pair.correlation:.2f}."
        )
    if evidence.formula_relationships:
        relationship = evidence.formula_relationships[0]["relationship"]
        facts.append(str(relationship))
    if evidence.identifier_like_columns:
        identifiers = ", ".join(display_name(column) for column in evidence.identifier_like_columns[:3])
        facts.append(f"{identifiers} look like identifier fields and should not be treated as substantive numeric measures.")
    if evidence.unequal_pairwise_counts:
        facts.append(
            f"Pairwise observation counts range from {evidence.minimum_pairwise_count:,} to "
            f"{evidence.maximum_pairwise_count:,}."
        )
    support = " ".join(dict.fromkeys(facts[:5]))

    if evidence.formula_relationships:
        interpretation = (
            "The strongest correlations are partly structural: formula-derived totals can move together because "
            "they share inputs or accounting definitions. "
        )
    elif evidence.correlation_clusters:
        interpretation = "The correlated cluster indicates overlapping information among several variables, not a causal structure. "
    else:
        interpretation = "The matrix summarizes pairwise association among the selected numeric variables. "
    interpretation += _correlation_method_sentence(method)

    cautions = []
    if evidence.formula_relationships:
        cautions.append("Formula-derived fields should not be interpreted as independent predictors.")
    if evidence.high_multicollinearity_pairs:
        cautions.append("Highly correlated predictors may create multicollinearity, though pairwise correlation alone does not prove a model is unusable.")
    if evidence.unequal_pairwise_counts:
        cautions.append("Correlations are based on different observation counts because missing values vary by pair.")
    if evidence.constant_columns:
        cautions.append("Constant fields have undefined correlations and should be removed from correlation analysis.")
    cautions.append("Correlation describes association and does not establish causation.")

    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=" ".join(cautions[:2]),
        recommended_next_step=_correlation_heatmap_next_step(evidence),
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def build_dual_line_fallback(evidence: DualLineEvidence) -> ChartInsight:
    """Create a relationship-focused deterministic insight for dual-line charts."""
    primary_phrase = build_metric_phrase(evidence.primary_y_column, evidence.primary_aggregation)
    secondary_phrase = build_metric_phrase(evidence.secondary_y_column, evidence.secondary_aggregation)
    primary_name = display_name(evidence.primary_y_column)
    secondary_name = display_name(evidence.secondary_y_column)
    is_ordered = evidence.x_axis_type in {"datetime", "numeric", "ordered_period"}

    if is_ordered and evidence.primary_change is not None and evidence.secondary_change is not None:
        if evidence.relationship_strength and evidence.relationship_direction:
            relationship = f"showed a {evidence.relationship_strength} {evidence.relationship_direction} association"
        elif (evidence.aligned_direction_percent or 0) >= 60:
            relationship = "generally went in the same direction"
        elif (evidence.opposite_direction_percent or 0) >= 60:
            relationship = "often went in opposite directions"
        else:
            relationship = "had a mixed relationship"
        primary_endpoint = _dual_line_endpoint_clause(evidence.primary_endpoint_direction, evidence.primary_start_x)
        secondary_endpoint = _dual_line_endpoint_clause(evidence.secondary_endpoint_direction, evidence.secondary_start_x)
        volatility = "with similar volatility"
        if evidence.more_volatile_metric and evidence.more_volatile_metric != "similar":
            volatility = f"with {evidence.more_volatile_metric} varying more"
        elif evidence.primary_volatility_level and evidence.secondary_volatility_level:
            volatility = (
                f"with {primary_name} {evidence.primary_volatility_level} and "
                f"{secondary_name} {evidence.secondary_volatility_level}"
            )
        key = (
            f"{primary_name} and {secondary_name} {relationship} over the ordered axis, "
            f"with {primary_name} {primary_endpoint} and {secondary_name} {secondary_endpoint}, {volatility}."
        )
        facts = []
        endpoint_fact = (
            f"{primary_phrase.capitalize()} was {_fmt(evidence.primary_start_value, evidence.primary_y_column)} "
            f"at {evidence.primary_start_x} and {_fmt(evidence.primary_end_value, evidence.primary_y_column)} "
            f"at {evidence.primary_end_x}"
        )
        if evidence.primary_change_percent is not None:
            endpoint_fact += f" ({_pct(evidence.primary_change_percent)} from the start)"
        endpoint_fact += (
            f"; {secondary_phrase} was {_fmt(evidence.secondary_start_value, evidence.secondary_y_column)} "
            f"at {evidence.secondary_start_x} and {_fmt(evidence.secondary_end_value, evidence.secondary_y_column)} "
            f"at {evidence.secondary_end_x}"
        )
        if evidence.secondary_change_percent is not None:
            endpoint_fact += f" ({_pct(evidence.secondary_change_percent)} from the start)"
        endpoint_fact += "."
        facts.append(endpoint_fact)
        if evidence.comparable_transition_count:
            facts.append(
                f"The two metrics had the same period-to-period direction in {evidence.aligned_direction_count or 0} "
                f"of {evidence.comparable_transition_count} comparable transitions "
                f"({_pct(evidence.aligned_direction_percent)}), and opposite directions in "
                f"{evidence.opposite_direction_count or 0} ({_pct(evidence.opposite_direction_percent)})."
            )
        if evidence.pearson_correlation is not None and evidence.spearman_correlation is not None:
            facts.append(
                f"Pearson correlation is {evidence.pearson_correlation:.3f} and Spearman correlation is "
                f"{evidence.spearman_correlation:.3f}, indicating a {evidence.relationship_strength or 'measurable'} "
                f"{evidence.relationship_direction or 'neutral'} association."
            )
        if evidence.known_metric_relationship:
            facts.append(evidence.known_metric_relationship)
        if evidence.peaks_aligned or evidence.troughs_aligned:
            aligned_parts = []
            if evidence.peaks_aligned:
                aligned_parts.append(f"both peaked at {evidence.primary_peak_period}")
            if evidence.troughs_aligned:
                aligned_parts.append(f"both were lowest at {evidence.primary_trough_period}")
            facts.append(f"{' and '.join(aligned_parts).capitalize()}.")
        elif evidence.largest_normalized_divergence_period:
            facts.append(
                f"The largest normalized gap occurred at {evidence.largest_normalized_divergence_period} "
                f"({_fmt(evidence.largest_normalized_divergence_value)} index points apart)."
            )
        if evidence.missing_periods:
            facts.append(f"Missing periods include {', '.join(str(item) for item in evidence.missing_periods[:3])}.")
        elif evidence.irregular_intervals:
            facts.append("The ordered axis has irregular intervals.")
        support = " ".join(dict.fromkeys(facts[:5]))
    else:
        if evidence.same_highest_category:
            key = (
                f"{evidence.primary_highest_x} leads both {primary_phrase} and {secondary_phrase}."
            )
        elif evidence.relationship_strength and evidence.relationship_direction:
            key = (
                f"{primary_phrase.capitalize()} and {secondary_phrase} show a "
                f"{evidence.relationship_strength} {evidence.relationship_direction} association across displayed categories."
            )
        else:
            key = f"{primary_name} and {secondary_name} rank the displayed categories differently."
        support = (
            f"{primary_phrase.capitalize()} ranges from {_fmt(evidence.primary_highest_value, evidence.primary_y_column)} "
            f"in {evidence.primary_highest_x} to {_fmt(evidence.primary_lowest_value, evidence.primary_y_column)} "
            f"in {evidence.primary_lowest_x}. {secondary_phrase.capitalize()} ranges from "
            f"{_fmt(evidence.secondary_highest_value, evidence.secondary_y_column)} in {evidence.secondary_highest_x} "
            f"to {_fmt(evidence.secondary_lowest_value, evidence.secondary_y_column)} in {evidence.secondary_lowest_x}."
        )
    if not is_ordered and evidence.pearson_correlation is not None:
        support += (
            f" Pearson correlation is {evidence.pearson_correlation:.3f} and "
            f"Spearman correlation is {evidence.spearman_correlation:.3f}."
        )
    if not is_ordered and evidence.known_metric_relationship:
        support += f" {evidence.known_metric_relationship}"

    if is_ordered:
        if evidence.unit_relationship == "same unit":
            interpretation = "Because the metrics use the same unit, the relationship can be read through both timing and absolute gaps between the series."
        elif evidence.unit_relationship == "different unit":
            interpretation = "Because the metrics use different units, the relationship is best read through direction, correlation, and normalized separation rather than raw height."
        else:
            interpretation = "Interpretation depends on the metric definitions, so the ordered pattern should be checked against the business meaning of both measures."
    elif evidence.unit_relationship == "same unit":
        interpretation = "Because the metrics use the same unit, absolute gaps between the two series are analytically meaningful."
    elif evidence.unit_relationship == "different unit":
        interpretation = "Because the metrics use different units, the useful comparison is ranking, association, or normalized movement rather than line height."
    else:
        interpretation = "Interpretation depends on the metric definitions because their units are not confidently known."
    if evidence.aggregation_warning:
        interpretation += f" {evidence.aggregation_warning}"

    caution_parts = []
    if evidence.unit_relationship == "same unit":
        caution_parts.append("Both metrics use the same unit, so separate axes can distort the apparent gap; a shared Y-axis is preferable for direct comparison.")
    elif evidence.unit_relationship == "different unit":
        caution_parts.append("The metrics use different Y-axis scales, so line heights, slopes, and visual distances cannot be compared directly.")
    if evidence.x_axis_type == "unordered categorical":
        caution_parts.append("The X-axis contains categories rather than time, so the connected lines show ranking across categories, not a temporal trend.")
    if is_ordered and evidence.pearson_correlation is not None:
        caution_parts.append("Correlation describes association, not causation.")
    if is_ordered and evidence.paired_point_count and evidence.paired_point_count < 12:
        caution_parts.append("The short history limits confidence in persistent patterns.")
    if is_ordered and (evidence.missing_periods or evidence.irregular_intervals):
        caution_parts.append("Missing or irregular periods can overstate continuity.")
    if evidence.aggregation_warning:
        caution_parts.append(evidence.aggregation_warning)
    if not is_ordered and evidence.category_count < 5:
        caution_parts.append("The small number of paired values limits relationship evidence.")
    caution = " ".join(caution_parts[:2] if is_ordered else caution_parts) or None

    if is_ordered:
        next_step = _dual_line_ordered_next_step(evidence)
    elif evidence.derived_metric_available == "profit":
        next_step = f"Compare total profit and profit margin by {display_name(evidence.x_column).lower()} using a shared-axis chart."
    elif evidence.derived_metric_available == "revenue per unit":
        next_step = f"Calculate revenue per unit by {display_name(evidence.x_column).lower()} to separate volume effects from price or mix."
    elif evidence.unit_relationship == "same unit":
        next_step = "Create a shared-axis multi-line chart or difference chart to compare absolute gaps directly."
    else:
        next_step = "Create a normalized index or meaningful ratio to evaluate whether the two measures move proportionally."

    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=caution,
        recommended_next_step=next_step,
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def build_single_line_fallback(evidence: SingleLineEvidence) -> ChartInsight:
    """Create a path-aware deterministic insight for a standard line chart."""
    metric_phrase = build_time_metric_phrase(
        evidence.y_column,
        evidence.aggregation,
        evidence.time_granularity,
    )
    metric_label = display_name(evidence.y_column)
    endpoint = _endpoint_direction(evidence.endpoint_change)
    pattern = evidence.pattern_classification or "mixed movement with no clear trend"
    volatility = evidence.volatility_level or "unknown"
    key = (
        f"{metric_label} {endpoint}, with {pattern} and "
        f"{volatility} period-to-period volatility."
    )

    facts = []
    date_summary = _date_summary_sentence(evidence, evidence.y_column)
    if date_summary:
        facts.append(date_summary)
    if evidence.start_period_label and evidence.end_period_label:
        endpoint_sentence = (
            f"{metric_phrase.capitalize()} moved from {_fmt(evidence.start_value, evidence.y_column)} "
            f"in {evidence.start_period_label} to {_fmt(evidence.end_value, evidence.y_column)} "
            f"in {evidence.end_period_label}, a change of {_fmt(evidence.endpoint_change, evidence.y_column)}"
        )
        if evidence.endpoint_change_percent is not None:
            endpoint_sentence += f" ({_pct(evidence.endpoint_change_percent)} relative to the starting value)"
        endpoint_sentence += "."
        facts.append(endpoint_sentence)
    if evidence.peak_period_label and evidence.trough_period_label:
        facts.append(
            f"It peaked at {_fmt(evidence.peak_value, evidence.y_column)} in {evidence.peak_period_label} "
            f"and reached its low at {_fmt(evidence.trough_value, evidence.y_column)} in {evidence.trough_period_label}."
        )
    if evidence.strongest_increase_value is not None:
        facts.append(
            f"The strongest rise was {_fmt(evidence.strongest_increase_value, evidence.y_column)} "
            f"from {evidence.strongest_increase_start_label} to {evidence.strongest_increase_end_label}."
        )
    if evidence.strongest_decline_value is not None:
        facts.append(
            f"The strongest decline was {_fmt(abs(evidence.strongest_decline_value), evidence.y_column)} "
            f"from {evidence.strongest_decline_start_label} to {evidence.strongest_decline_end_label}."
        )
    if evidence.trend_strength and evidence.linear_trend_r_squared is not None:
        facts.append(
            f"The fitted trend is {evidence.trend_strength} "
            f"(R-squared {evidence.linear_trend_r_squared:.2f}), while volatility is {volatility}."
        )
    if evidence.missing_period_labels:
        facts.append(
            f"Missing periods include {', '.join(evidence.missing_period_labels[:3])}."
        )
    support = " ".join(dict.fromkeys(facts[:5]))

    if volatility == "high":
        interpretation = (
            "The endpoint comparison should not be read as smooth growth or decline; repeated short-term "
            "moves dominate the path between the first and last periods."
        )
    elif evidence.trend_strength == "strong" and pattern.startswith("steady"):
        interpretation = (
            "The fitted trend and endpoint movement point in the same direction, so the displayed path "
            "supports a persistent directional pattern."
        )
    else:
        interpretation = (
            "The series is better read through its peaks, troughs, and period-to-period changes than through "
            "the endpoint comparison alone."
        )

    cautions = [_line_metric_caution(evidence)]
    if evidence.missing_period_labels or evidence.irregular_intervals:
        cautions.append("Missing or irregular periods can make period-to-period comparisons incomplete.")
    if evidence.seasonality_evidence != "no seasonality test performed":
        cautions.append("The available history is not sufficient by itself to confirm seasonality or support forecasting.")
    else:
        cautions.append("No seasonality test was performed, so seasonal behavior is not established.")
    caution = " ".join(cautions)

    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=caution,
        recommended_next_step=_line_next_step(evidence),
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def _fallback_insight(evidence: ChartEvidence) -> ChartInsight:
    """Create a deterministic, business-friendly insight from evidence."""
    if isinstance(evidence, CorrelationHeatmapEvidence):
        return build_correlation_heatmap_fallback(evidence)
    if isinstance(evidence, PieChartEvidence):
        return build_pie_chart_fallback(evidence)
    if isinstance(evidence, TreemapEvidence):
        return build_treemap_fallback(evidence)
    if isinstance(evidence, SymbolMapEvidence):
        return build_symbol_map_fallback(evidence)
    if isinstance(evidence, SortedPercentageBarEvidence):
        return build_sorted_percentage_bar_fallback(evidence)
    if isinstance(evidence, PeriodOverPeriodChangeEvidence):
        return build_period_over_period_fallback(evidence)
    if isinstance(evidence, SingleBarEvidence):
        return build_single_bar_fallback(evidence)
    if isinstance(evidence, StackedBarEvidence):
        return build_stacked_bar_fallback(evidence)
    if isinstance(evidence, GroupedBarEvidence):
        return build_grouped_bar_fallback(evidence)
    if isinstance(evidence, SingleAreaEvidence):
        return build_single_area_fallback(evidence)
    if isinstance(evidence, StackedAreaEvidence):
        return build_stacked_area_fallback(evidence)
    if isinstance(evidence, DualCombinationEvidence):
        return build_dual_combination_fallback(evidence)
    if isinstance(evidence, ScatterEvidence):
        return build_scatter_fallback(evidence)
    if isinstance(evidence, CircleViewEvidence):
        return build_circle_view_fallback(evidence)
    if isinstance(evidence, HistogramEvidence):
        return build_histogram_fallback(evidence)
    if isinstance(evidence, BoxPlotEvidence):
        return build_box_plot_fallback(evidence)
    if isinstance(evidence, DualLineEvidence):
        return build_dual_line_fallback(evidence)
    if isinstance(evidence, SingleLineEvidence):
        return build_single_line_fallback(evidence)

    metric = evidence.y_columns[0] if evidence.y_columns else None
    metric_name = display_name(metric)
    category_name = display_name(evidence.category_column)
    aggregation = _aggregation_text(evidence)
    m = evidence.calculated_metrics
    caution = "; ".join(evidence.warnings) or None

    if evidence.chart_type in {"bar", "stacked_bar", "grouped_bar", "pie", "treemap", "symbol_map"} and "highest_category" in m:
        high = m["highest_category"]
        low = m["lowest_category"]
        top_share = _pct(m.get("top_share_pct"))
        key = f"{high} has the highest {aggregation} {metric_name}."
        support = (
            f"{high} is {_fmt(m['highest_value'], metric)} and represents {top_share} "
            f"of the displayed total. {low} is lowest at {_fmt(m['lowest_value'], metric)}, "
            f"so the gap is {_fmt(m['absolute_gap'], metric)}."
        )
        if evidence.chart_type in {"stacked_bar", "grouped_bar"} and "dominant_segment" in m:
            support += f" The largest displayed segment is {m['dominant_segment']}."
        interpretation = (
            f"The displayed {category_name or 'categories'} are "
            f"{'concentrated' if m.get('top_three_share_pct', 0) and m['top_three_share_pct'] > 70 else 'spread across multiple categories'}."
            f"{_filter_text(evidence)}"
        )
        if evidence.chart_type == "symbol_map":
            interpretation += f" The map compares {evidence.valid_rows} displayed location(s)."
        next_step = f"Compare {metric_name} with a related efficiency or profitability metric."
    elif "multi_metric_leaders" in m:
        leaders = m["multi_metric_leaders"]
        leader_text = "; ".join(
            f"{info['highest_category']} leads {metric} at {_fmt(info['highest_value'], metric)}"
            for metric, info in leaders.items()
        )
        key = f"{display_name(evidence.category_column)} compares multiple displayed metrics."
        support = leader_text + "."
        interpretation = "Each metric should be read independently because scales and business meanings may differ."
        next_step = "Compare the metric leaders against an efficiency ratio or margin."
    elif evidence.chart_type in {"line", "area"} and "trend_direction" in m:
        key = f"{metric_name} {m['trend_direction']} across the displayed range."
        support = (
            f"It moved from {_fmt(m['first_value'], metric)} at {m['first_label']} "
            f"to {_fmt(m['final_value'], metric)} at {m['final_label']}, "
            f"a change of {_fmt(m['absolute_change'], metric)} ({_pct(m.get('percentage_change'))})."
        )
        interpretation = "This describes direction and magnitude only; it does not prove seasonality or causality."
        next_step = f"Break the trend down by {display_name(evidence.color_column) or 'a relevant category'}."
    elif evidence.chart_type in {"dual_axis", "dual_line"} and evidence.y_columns:
        first, second = evidence.y_columns[:2]
        first_metrics = m.get(first, {})
        second_metrics = m.get(second, {})
        if evidence.chart_type == "dual_axis" and m.get("high_primary_low_secondary_count") is not None:
            key = (
                f"{m['high_primary_low_secondary_count']} {display_name(evidence.x_column)} "
                f"value(s) combine high {display_name(first)} with low {display_name(second)}."
            )
        else:
            key = f"{display_name(first)} and {display_name(second)} are compared on separate scales."
        support = (
            f"{display_name(first)} changed by {_fmt(first_metrics.get('absolute_change', 0), first)}, "
            f"while {display_name(second)} changed by {_fmt(second_metrics.get('absolute_change', 0), second)}."
        )
        if m.get("high_primary_low_secondary_labels"):
            examples = ", ".join(m["high_primary_low_secondary_labels"])
            support += (
                f" {examples} combines high {display_name(first)} with low "
                f"{display_name(second)}."
            )
        interpretation = "Use the chart to compare direction and turning points rather than raw heights."
        next_step = "Create a normalized index chart to compare relative movement on one scale."
    elif evidence.chart_type in {"scatter", "circle_view"} and "pearson_correlation" in m:
        corr = m["pearson_correlation"]
        strength = "strong" if abs(corr) >= .7 else "moderate" if abs(corr) >= .4 else "weak"
        direction = "positive" if corr > 0 else "negative" if corr < 0 else "neutral"
        key = f"{display_name(evidence.x_column)} and {metric_name} show a {strength} {direction} association."
        support = (
            f"Pearson correlation is {corr:.3f}, Spearman correlation is {m.get('spearman_correlation', 0):.3f}, "
            f"with R-squared of {m.get('r_squared', 0):.3f} across {evidence.valid_rows:,} valid points."
        )
        interpretation = "This is an observed association and should not be treated as independent causal evidence."
        next_step = f"Color the points by {display_name(evidence.color_column) or 'a category'} or inspect outliers."
        if evidence.chart_type == "circle_view" and "largest_circle" in m:
            largest = m["largest_circle"]
            size_column = evidence.size_column
            support += (
                f" The largest {display_name(size_column)} circle is "
                f"{_fmt(largest.get(size_column), size_column)}."
            )
    elif evidence.chart_type in {"histogram", "box"} and "median" in m:
        key = f"{metric_name or display_name(evidence.x_column)} has a median of {_fmt(m['median'], metric)}."
        support = (
            f"The mean is {_fmt(m['mean'], metric)}, the IQR is {_fmt(m['iqr'], metric)}, "
            f"and {m['outlier_count']} potential outlier(s) were detected."
        )
        interpretation = ", ".join(evidence.detected_patterns) or "The distribution summary is based on verified numeric values."
        next_step = "Review outliers and compare the distribution across a categorical segment."
    elif evidence.chart_type == "heatmap" and "strongest_absolute_pair" in m:
        value, first, second = m["strongest_absolute_pair"]
        key = f"{display_name(first)} and {display_name(second)} have the strongest displayed relationship."
        support = f"Their correlation is {value:.3f}; self-correlations were excluded."
        interpretation = "Correlation describes association and may reflect derived or mathematically related metrics."
        next_step = "Inspect a scatter plot for this pair and check for outliers or formula-derived fields."
    elif evidence.chart_type == "gantt" and "task_count" in m:
        key = f"{m['task_count']} task(s) span {m['project_span_days']:.0f} day(s)."
        support = (
            f"{m['task_count']} task(s) run from {_fmt(m['earliest_start'])} to {_fmt(m['latest_finish'])}. "
            f"The longest task is {m['longest_task']} at {m['longest_task_days']:.1f} day(s)."
        )
        interpretation = f"{m['overlapping_task_count']} task(s) overlap with at least one other task."
        next_step = "Add dependency data before attempting critical-path analysis."
    elif evidence.chart_type == "bullet" and "target_results" in m:
        target_results = m["target_results"]
        first = target_results[0] if target_results else {}
        largest_shortfall = m.get("largest_shortfall")
        largest_surplus = m.get("largest_surplus")
        target_count = m.get("target_count", len(target_results))
        key = f"{m.get('targets_met', 0)} of {target_count} displayed categories meet or exceed target."
        support = (
            f"Total actual is {_fmt(m.get('actual_total', 0), metric)} versus total target "
            f"{_fmt(m.get('target_total', 0), evidence.y_columns[1] if len(evidence.y_columns) > 1 else metric)}, "
            f"a variance of {_fmt(m.get('total_variance', 0), metric)}."
        )
        if largest_shortfall and largest_shortfall["absolute_variance"] < 0:
            support += (
                f" The largest shortfall is {largest_shortfall['label']} at "
                f"{_fmt(abs(largest_shortfall['absolute_variance']), metric)} below target."
            )
        elif largest_surplus:
            support += (
                f" The largest surplus is {largest_surplus['label']} at "
                f"{_fmt(largest_surplus['absolute_variance'], metric)} above target."
            )
        if first:
            support += (
                f" For reference, {first.get('label')} has actual {_fmt(first.get('actual', 0), metric)} "
                f"against target {_fmt(first.get('target', 0), evidence.y_columns[1] if len(evidence.y_columns) > 1 else metric)}."
            )
        interpretation = (
            "Each bullet compares the actual bar with its target marker; positive variance means actual is above target, "
            "and negative variance means it is below target."
        )
        next_step = "Sort by variance or create a variance bar chart to prioritize the largest shortfalls and strongest over-target categories."
    else:
        key = "The chart has limited evidence for a specific analytical pattern."
        support = f"{evidence.valid_rows:,} displayed record(s) were available for analysis."
        interpretation = "The selected chart may need more observations or a clearer metric to support a stronger conclusion."
        next_step = "Try adding a numeric metric, grouping column, or filter."

    restriction = _restriction_text(evidence)
    if restriction:
        support += f" Note: {restriction}."
    return ChartInsight(
        chart_title=evidence.chart_title,
        key_finding=key,
        supporting_evidence=support,
        interpretation=interpretation,
        caution=caution,
        recommended_next_step=next_step,
        evidence_strength=evidence.evidence_strength,
        evidence=evidence,
    )


def generate_chart_insight(result: ChartResult) -> ChartInsight:
    """Generate a validated written insight from chart evidence."""
    evidence = extract_chart_evidence(result)
    return _fallback_insight(evidence)
