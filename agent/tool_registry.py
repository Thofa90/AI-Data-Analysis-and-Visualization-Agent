"""Registry of the only tools the data agent may execute."""

from __future__ import annotations

from collections.abc import Callable
import inspect
import logging
from time import perf_counter
from typing import Any

import pandas as pd

from agent import tools
from agent.schemas import ToolResult

LOGGER = logging.getLogger(__name__)


TOOL_REGISTRY: dict[str, Callable[..., ToolResult]] = {
    "inspect_dataset": tools.inspect_dataset,
    "get_column_information": tools.get_column_information,
    "profile_column": tools.profile_column,
    "calculate_summary_statistics": tools.calculate_summary_statistics,
    "calculate_average_numeric_columns": tools.calculate_average_numeric_columns,
    "group_and_aggregate": tools.group_and_aggregate,
    "analyze_advanced_request": tools.analyze_advanced_request,
    "calculate_grouped_extrema": tools.calculate_grouped_extrema,
    "compare_grouped_to_benchmark": tools.compare_grouped_to_benchmark,
    "compare_category_values": tools.compare_category_values,
    "calculate_filtered_aggregate": tools.calculate_filtered_aggregate,
    "calculate_scalar_aggregate": tools.calculate_scalar_aggregate,
    "calculate_multi_scalar_aggregate": tools.calculate_multi_scalar_aggregate,
    "list_distinct_values": tools.list_distinct_values,
    "count_distinct_values": tools.count_distinct_values,
    "analyze_high_volume_low_outcome": tools.analyze_high_volume_low_outcome,
    "filter_dataset": tools.filter_dataset,
    "sort_and_limit": tools.sort_and_limit,
    "calculate_correlation": tools.calculate_correlation,
    "detect_outliers": tools.detect_outliers,
    "analyze_missing_values": tools.analyze_missing_values,
    "analyze_duplicates": tools.analyze_duplicates,
    "calculate_time_trend": tools.calculate_time_trend,
    "calculate_period_over_period": tools.calculate_period_over_period,
    "calculate_date_aggregate": tools.calculate_date_aggregate,
    "compare_categories": tools.compare_categories,
    "calculate_value_counts": tools.calculate_value_counts,
    "analyze_categorical_value_counts": tools.analyze_categorical_value_counts,
    "create_bar_chart": tools.create_bar_chart,
    "create_line_chart": tools.create_line_chart,
    "create_scatter_plot": tools.create_scatter_plot,
    "create_histogram": tools.create_histogram,
    "create_box_plot": tools.create_box_plot,
    "create_pie_chart": tools.create_pie_chart,
    "create_heatmap": tools.create_heatmap,
    "generate_report": tools.generate_report,
}


def execute_tool(dataframe: pd.DataFrame, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
    """Execute one allowlisted tool with the active DataFrame."""
    validate_tool_arguments(tool_name, arguments)
    started = perf_counter()
    LOGGER.info("Executing tool=%s arguments=%s", tool_name, arguments)
    try:
        result = TOOL_REGISTRY[tool_name](dataframe, **arguments)
        LOGGER.info("Tool completed name=%s seconds=%.4f", tool_name, perf_counter() - started)
        return result
    except Exception:
        LOGGER.exception("Tool failed name=%s", tool_name)
        raise


def validate_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> None:
    """Reject missing or invented tool arguments before execution."""
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f'Unsupported analytical tool "{tool_name}".')
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    signature = inspect.signature(TOOL_REGISTRY[tool_name])
    try:
        signature.bind(object(), **arguments)
    except TypeError as exc:
        raise ValueError(
            f'Invalid arguments for "{tool_name}": {exc}.'
        ) from exc


def active_tool_names() -> list[str]:
    """Return user-facing names for actual implemented capabilities."""
    return [
        "Dataset Inspector",
        "Data Quality Analyzer",
        "Pandas Aggregation Tool",
        "Missing Value Analyzer",
        "Outlier Detector",
        "Correlation Analyzer",
        "Time-Series Analyzer",
        "Chart Generator",
        "Report Generator",
        "Evaluation Engine",
    ]
