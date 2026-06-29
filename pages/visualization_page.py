"""Manual and recommended visualization builder."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from components.chart_panel import render_chart_insight
from services.chart_insight_service import generate_chart_insight
from services.chart_service import (
    MAX_CATEGORY_LIMIT,
    SYMBOL_MAP_COLOR_LOCATION,
    SYMBOL_MAP_COLOR_NONE,
    ChartSpec,
    create_chart,
    figure_to_png,
)
from services.date_aggregation_service import date_bucket_label, date_bucket_start
from utils.formatting import is_currency_column


AGGREGATION_OPTIONS = {
    "Sum": "sum",
    "Mean": "mean",
    "Median": "median",
    "Count": "count",
    "Unique Count": "nunique",
    "Minimum": "min",
    "Maximum": "max",
}

CHART_OPTIONS = {
    "Bar": ("bar", "Compares one metric across categories."),
    "Stacked Bar": (
        "stacked_bar",
        "Shows category totals split into stacked subcategories.",
    ),
    "Side-by-Side Bar": (
        "grouped_bar",
        "Places subcategory bars beside each other for direct comparison.",
    ),
    "Sorted Percentage Bar": (
        "sorted_percentage_bar",
        "Ranks categories by their percentage contribution to a selected metric.",
    ),
    "Line": ("line", "Shows how one metric changes across an ordered axis."),
    "Area": ("area", "Emphasizes the magnitude of change across an ordered axis."),
    "Dual Lines": (
        "dual_line",
        "Compares two metrics as lines, each with its own vertical scale.",
    ),
    "Dual Combination": (
        "dual_axis",
        "Combines bars and a line to compare two differently scaled metrics.",
    ),
    "Period-over-Period % Change": (
        "period_over_period_change",
        "Shows percent change versus a previous comparable date period.",
    ),
    "Scatter": ("scatter", "Shows the relationship between two numeric metrics."),
    "Circle View": (
        "circle_view",
        "Uses bubble position and size to compare up to three numeric measures.",
    ),
    "Histogram": ("histogram", "Shows the distribution of one numeric metric."),
    "Box": ("box", "Summarizes spread, quartiles, and potential outliers."),
    "Heatmap": ("heatmap", "Displays correlations among numeric columns."),
    "Pie": ("pie", "Shows how categories contribute to a whole."),
    "Tree Map": (
        "treemap",
        "Uses nested rectangles to show hierarchical category proportions.",
    ),
    "Symbol Map": (
        "symbol_map",
        "Places sized symbols on countries or standard world regions using a location column.",
    ),
    "Bullet Graph": (
        "bullet",
        "Compares an actual metric against a target metric for each category; choose this when your dataset has a target column.",
    ),
}


def _select(
    label: str,
    options: list[str],
    *,
    key: str,
    allow_none: bool = True,
) -> str | None:
    choices = ["None", *options] if allow_none else options
    value = st.selectbox(label, choices, key=key)
    return None if value == "None" else value


def symbol_map_color_options(categorical_columns: list[str], location_column: str | None) -> list[str]:
    """Return stable Symbol Map color modes without duplicating the location column."""
    return [
        SYMBOL_MAP_COLOR_NONE,
        SYMBOL_MAP_COLOR_LOCATION,
        *[column for column in categorical_columns if column != location_column],
    ]


def symbol_map_color_label(value: str, location_column: str | None) -> str:
    if value == SYMBOL_MAP_COLOR_NONE:
        return "None"
    if value == SYMBOL_MAP_COLOR_LOCATION:
        return f"Color by {friendly_column_name(location_column) or 'Location'}"
    return friendly_column_name(value)


def normalize_symbol_map_color(location_column: str | None, color_column: str | None) -> str | None:
    if not color_column or color_column in {"None", SYMBOL_MAP_COLOR_NONE}:
        return None
    if color_column == SYMBOL_MAP_COLOR_LOCATION or color_column == location_column:
        return SYMBOL_MAP_COLOR_LOCATION
    return color_column


def reset_invalid_symbol_map_color_state(
    state: dict[str, object],
    key: str,
    color_options: list[str],
    location_column: str | None = None,
) -> None:
    if location_column and state.get(key) == location_column:
        state[key] = SYMBOL_MAP_COLOR_LOCATION
        return
    valid_values = set(color_options)
    if state.get(key) not in (None, *valid_values):
        state[key] = SYMBOL_MAP_COLOR_NONE


def date_candidate_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return columns that can be used for date filtering or bucketing."""
    candidates = []
    for column in dataframe.columns:
        lowered = column.lower()
        if (
            pd.api.types.is_datetime64_any_dtype(dataframe[column])
            or any(token in lowered for token in ("date", "time", "timestamp", "start", "end"))
        ):
            candidates.append(column)
            continue
        if (
            pd.api.types.is_bool_dtype(dataframe[column])
            or pd.api.types.is_numeric_dtype(dataframe[column])
        ):
            continue
        parsed = pd.to_datetime(dataframe[column], errors="coerce", format="mixed")
        non_empty = dataframe[column].notna().sum()
        if non_empty and parsed.notna().sum() / non_empty >= 0.8:
            candidates.append(column)
    return candidates


def discrete_date_period_options(
    values: pd.Series,
    grain: str,
) -> list[tuple[str, pd.Timestamp]]:
    """Return display labels and bucket start dates for discrete date filters."""
    dates = pd.to_datetime(values, errors="coerce").dropna()
    periods = date_bucket_start(dates, grain).drop_duplicates().sort_values()
    return [(date_bucket_label(value, grain), pd.Timestamp(value)) for value in periods]


def category_limit_max(chart_type: str, dataframe: pd.DataFrame, x: str | None) -> int:
    """Return the maximum category/observation limit for the selected chart."""
    if chart_type == "scatter":
        return min(MAX_CATEGORY_LIMIT, max(1, len(dataframe)))
    if x and x in dataframe.columns:
        unique_categories = int(dataframe[x].nunique(dropna=True))
        return min(MAX_CATEGORY_LIMIT, max(1, unique_categories))
    return min(MAX_CATEGORY_LIMIT, max(1, len(dataframe)))


def aggregation_display_name(aggregation: str | None) -> str:
    return {
        "sum": "Sum",
        "mean": "Mean",
        "median": "Median",
        "count": "Count",
        "nunique": "Unique Count",
        "min": "Minimum",
        "max": "Maximum",
    }.get(aggregation or "", (aggregation or "").title())


def friendly_column_name(column: str | None) -> str:
    if not column:
        return ""
    spaced = "".join(
        f" {character}" if index and character.isupper() and str(column)[index - 1].islower() else character
        for index, character in enumerate(str(column).replace("_", " "))
    )
    return " ".join(token[:1].upper() + token[1:] for token in spaced.split())


def symbol_map_metric_title(y: str | None, aggregation: str | None) -> str:
    metric = friendly_column_name(y)
    if aggregation == "count":
        return f"Count of {metric}" if metric else "Record Count"
    if aggregation == "mean":
        return f"Average {metric}" if metric else "Average Value"
    if aggregation == "median":
        return f"Median {metric}" if metric else "Median Value"
    if aggregation == "nunique":
        return f"Unique {metric} Count" if metric else "Unique Count"
    if aggregation == "min":
        return f"Minimum {metric}" if metric else "Minimum Value"
    if aggregation == "max":
        return f"Maximum {metric}" if metric else "Maximum Value"
    return metric or "Value"


def _looks_like_identifier(column: str) -> bool:
    normalized = "".join(character for character in column.lower() if character.isalnum())
    return (
        normalized.endswith(("id", "key", "code", "identifier"))
        or normalized.startswith(("id", "key"))
        or any(token in normalized for token in ("orderid", "customerid", "invoice", "transactionkey"))
    )


def _save_visualization_insight(result, insight) -> None:
    metadata = st.session_state.uploaded_file or {}
    saved = {
        "dataset_name": metadata.get("name", "dataset"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": result.spec.title,
        "answer": (
            f"{insight.key_finding}\n\n"
            f"Supporting evidence: {insight.supporting_evidence}\n\n"
            f"Interpretation: {insight.interpretation}\n\n"
            f"Recommended next step: {insight.recommended_next_step}"
        ),
        "result": {
            "tool_name": "manual_visualization",
            "success": True,
            "summary": result.spec.title,
            "data": result.data,
            "warnings": [],
            "execution_seconds": 0,
        },
        "plan": {
            "tool_name": "manual_visualization",
            "arguments": {},
            "chart_spec": result.spec.model_dump(mode="json"),
            "clarification": None,
            "safe_code": "",
        },
        "chart_spec": result.spec.model_dump(mode="json"),
        "chart_data": result.data,
        "suggested_questions": [],
        "assumptions": [],
    }
    st.session_state.saved_insights = [*st.session_state.saved_insights, saved]
    st.toast("Chart insight saved.")


def build_chart_title(
    chart_label: str,
    chart_type: str,
    x: str | None,
    y: str | None,
    secondary_y: str | None,
    color: str | None,
    aggregation: str | None,
    primary_aggregation: str | None = None,
    secondary_aggregation: str | None = None,
) -> str:
    """Build a readable title from the selected chart semantics."""
    aggregation_label = aggregation.title() if aggregation else None
    metric = y or ("Records" if aggregation == "count" else None)
    metric_text = " ".join(
        item for item in (aggregation_label, metric) if item
    )

    if chart_type == "heatmap":
        return "Heatmap: Correlations Between Numeric Columns"
    if chart_type == "period_over_period_change":
        return f"{aggregation_display_name(aggregation)} {y or 'Metric'} % Change by {x or 'Period'}"
    if chart_type == "sorted_percentage_bar":
        metric_name = y or ("Records" if aggregation == "count" else "Selected Metric")
        return f"Share of {metric_name} by {x or 'Category'}"
    if chart_type == "histogram":
        return f"Histogram: Distribution of {x or 'Selected Metric'}"
    if chart_type == "box":
        return f"Box Plot: Distribution of {y or 'Selected Metric'}"
    if chart_type == "scatter":
        return f"Scatter Plot: {y or 'Y'} vs {x or 'X'}"
    if chart_type == "circle_view":
        title = f"Circle View: {y or 'Y'} vs {x or 'X'}"
        if secondary_y:
            title += f", Sized by {secondary_y}"
        if color:
            title += f", Colored by {color}"
        return title
    if chart_type == "symbol_map":
        metric_title = symbol_map_metric_title(y, aggregation)
        location_name = friendly_column_name(x) or "Location"
        effective_color = normalize_symbol_map_color(x, color)
        if effective_color and effective_color != SYMBOL_MAP_COLOR_LOCATION:
            return f"{metric_title} by {location_name} and {friendly_column_name(effective_color)}"
        return f"{metric_title} by {location_name}"
    if chart_type == "gantt":
        return (
            f"Gantt View: {y or 'Tasks'} from {x or 'Start'} "
            f"to {secondary_y or 'End'}"
        )
    if chart_type == "bullet":
        return (
            f"Bullet Graph: {aggregation_label or ''} {y or 'Actual'} vs "
            f"{secondary_y or 'Target'} by {x or 'Category'}"
        ).replace(":  ", ": ")
    if chart_type in {"dual_line", "dual_axis"}:
        if primary_aggregation and secondary_aggregation:
            if primary_aggregation == secondary_aggregation:
                return (
                    f"{chart_label}: {aggregation_display_name(primary_aggregation)} of {y or 'Primary Metric'} "
                    f"and {secondary_y or 'Secondary Metric'} by {x or 'Category'}"
                )
            return (
                f"{chart_label}: {y or 'Primary Metric'} ({aggregation_display_name(primary_aggregation)}) "
                f"and {secondary_y or 'Secondary Metric'} ({aggregation_display_name(secondary_aggregation)}) "
                f"by {x or 'Category'}"
            )
        return (
            f"{chart_label}: {aggregation_label or ''} {y or 'Primary Metric'} "
            f"and {secondary_y or 'Secondary Metric'} by {x or 'Category'}"
        ).replace(":  ", ": ")

    title = f"{chart_label}: {metric_text or 'Values'}"
    if x:
        title += f" by {x}"
    if color:
        connector = " and " if chart_type in {"stacked_bar", "grouped_bar", "treemap"} else ", by "
        title += f"{connector}{color}"
    return title


def render_visualization_page() -> None:
    """Render a validated chart builder."""
    st.title("Smart Visualizations")
    dataframe = st.session_state.active_dataframe
    if dataframe is None:
        st.info("Upload a dataset to build visualizations.")
        return

    columns = list(dataframe.columns)
    numeric = list(dataframe.select_dtypes(include="number").columns)
    categorical = [column for column in columns if column not in numeric]
    date_candidates = date_candidate_columns(dataframe)

    chart_label = st.selectbox("Chart type", list(CHART_OPTIONS))
    chart_type, explanation = CHART_OPTIONS[chart_label]
    st.info(explanation)

    x = y = secondary_y = color = None
    aggregation = None
    primary_aggregation = None
    secondary_aggregation = None
    include_values = []
    time_grain = None
    time_column = None
    date_range_start = None
    date_range_end = None
    date_period_values = []
    percentage_denominator_mode = "full_filtered"
    include_other = False
    show_values = True
    comparison_basis = "previous_period"
    chart_style = "diverging_bar"
    missing_period_policy = "preserve_missing"
    needs_aggregation = chart_type not in {
        "scatter", "circle_view", "histogram", "box", "heatmap", "dual_line", "dual_axis", "sorted_percentage_bar", "period_over_period_change"
    }

    if chart_type == "heatmap":
        st.caption("The heatmap automatically uses all numeric columns.")
    elif chart_type == "sorted_percentage_bar":
        metric_candidates = [column for column in numeric if not _looks_like_identifier(column)]
        fields = st.columns(3)
        with fields[0]:
            x = _select("Category", categorical, key="viz_pct_category", allow_none=False)
        with fields[1]:
            y = _select("Numeric metric", metric_candidates or numeric, key="viz_pct_metric", allow_none=False)
        with fields[2]:
            aggregation_label = st.selectbox(
                "Aggregation",
                list(AGGREGATION_OPTIONS),
                key="viz_pct_aggregation",
            )
            aggregation = AGGREGATION_OPTIONS[aggregation_label]
        pct_options = st.columns(3)
        with pct_options[0]:
            denominator_label = st.selectbox(
                "Percentage denominator",
                ["Full filtered total", "Displayed categories only"],
                key="viz_pct_denominator",
            )
            percentage_denominator_mode = "displayed" if denominator_label == "Displayed categories only" else "full_filtered"
        with pct_options[1]:
            include_other = st.checkbox(
                "Group excluded categories as Other",
                value=False,
                key="viz_pct_include_other",
            )
        with pct_options[2]:
            show_values = st.checkbox("Show values", value=True, key="viz_pct_show_values")
    elif chart_type == "period_over_period_change":
        if not date_candidates:
            st.warning("Period-over-Period % Change needs at least one valid date field.")
            st.stop()
        metric_candidates = [column for column in numeric if not _looks_like_identifier(column)]
        fields = st.columns(3)
        with fields[0]:
            x = _select("Date field", date_candidates, key="viz_pop_date", allow_none=False)
        with fields[1]:
            y = _select("Metric", metric_candidates or numeric, key="viz_pop_metric", allow_none=False)
        with fields[2]:
            aggregation_label = st.selectbox(
                "Aggregation",
                list(AGGREGATION_OPTIONS),
                key="viz_pop_aggregation",
            )
            aggregation = AGGREGATION_OPTIONS[aggregation_label]
        pop_controls = st.columns(4)
        with pop_controls[0]:
            grain_label = st.selectbox(
                "Period granularity",
                ["Day", "Week", "Month", "Quarter", "Year"],
                index=2,
                key="viz_pop_grain",
            )
            time_grain = grain_label.lower()
        with pop_controls[1]:
            basis_label = st.selectbox(
                "Comparison basis",
                ["Previous period", "Same period last year"],
                key="viz_pop_basis",
            )
            comparison_basis = "same_period_last_year" if basis_label == "Same period last year" else "previous_period"
        with pop_controls[2]:
            show_values = st.checkbox("Show % labels", value=True, key="viz_pop_show_values")
        with pop_controls[3]:
            st.caption("Style: diverging bars")
    elif chart_type == "gantt":
        fields = st.columns(4)
        with fields[0]:
            y = _select("Task column", categorical, key="viz_task", allow_none=False)
        with fields[1]:
            x = _select("Start date", date_candidates or columns, key="viz_start", allow_none=False)
        with fields[2]:
            secondary_y = _select("End date", date_candidates or columns, key="viz_end", allow_none=False)
        with fields[3]:
            color = _select("Group / color", categorical, key="viz_gantt_color")
    elif chart_type == "symbol_map":
        fields = st.columns(3)
        with fields[0]:
            x = _select(
                "Country / geographic region",
                categorical,
                key="viz_location",
                allow_none=False,
            )
        with fields[1]:
            y_choice = st.selectbox("Symbol size", ["Count records", *numeric])
            y = None if y_choice == "Count records" else y_choice
            aggregation = "count" if y is None else None
        with fields[2]:
            map_color_options = symbol_map_color_options(categorical, x)
            reset_invalid_symbol_map_color_state(
                st.session_state,
                "viz_map_color",
                map_color_options,
                x,
            )
            color = st.selectbox(
                "Color by",
                map_color_options,
                key="viz_map_color",
                format_func=lambda value: symbol_map_color_label(value, x),
            )
            color = normalize_symbol_map_color(x, color)
    elif chart_type == "circle_view":
        fields = st.columns(4)
        with fields[0]:
            x = _select("X metric", numeric, key="viz_circle_x", allow_none=False)
        with fields[1]:
            y = _select("Y metric", numeric, key="viz_circle_y", allow_none=False)
        with fields[2]:
            secondary_y = _select("Bubble size", numeric, key="viz_circle_size")
        with fields[3]:
            color = _select("Color by", categorical, key="viz_circle_color")
    elif chart_type == "dual_line":
        axis_fields = st.columns(2)
        with axis_fields[0]:
            x = _select("Category / X-axis", categorical or columns, key="viz_dual_x", allow_none=False)
            y = _select(
                "Left Y-axis metric",
                numeric,
                key="dual_line_primary_metric",
                allow_none=False,
            )
            primary_label = st.selectbox(
                "Left Y-axis aggregation",
                list(AGGREGATION_OPTIONS),
                key="dual_line_primary_aggregation",
            )
            primary_aggregation = AGGREGATION_OPTIONS[primary_label]
        with axis_fields[1]:
            secondary_y = _select(
                "Right Y-axis metric",
                [column for column in numeric if column != y],
                key="dual_line_secondary_metric",
                allow_none=False,
            )
            secondary_label = st.selectbox(
                "Right Y-axis aggregation",
                list(AGGREGATION_OPTIONS),
                key="dual_line_secondary_aggregation",
            )
            secondary_aggregation = AGGREGATION_OPTIONS[secondary_label]
    elif chart_type in {"dual_axis", "bullet"}:
        if chart_type == "bullet":
            st.caption(
                "Bullet Graph needs a numeric target column. Select the actual metric first, then choose the target metric to compare against."
            )
        fields = st.columns(3)
        with fields[0]:
            x = _select("Category / X-axis", categorical or columns, key="viz_dual_x", allow_none=False)
        with fields[1]:
            y = _select(
                "Actual / primary metric" if chart_type == "bullet" else "Primary metric",
                numeric,
                key="viz_dual_y",
                allow_none=False,
            )
        with fields[2]:
            secondary_y = _select(
                "Target metric" if chart_type == "bullet" else "Secondary metric",
                [column for column in numeric if column != y],
                key="viz_dual_secondary",
                allow_none=False,
            )
        if chart_type == "dual_axis":
            aggregation_fields = st.columns(2)
            with aggregation_fields[0]:
                primary_label = st.selectbox(
                    "Primary axis aggregation",
                    list(AGGREGATION_OPTIONS),
                    key="dual_axis_primary_aggregation",
                )
                primary_aggregation = AGGREGATION_OPTIONS[primary_label]
            with aggregation_fields[1]:
                secondary_label = st.selectbox(
                    "Secondary axis aggregation",
                    list(AGGREGATION_OPTIONS),
                    key="dual_axis_secondary_aggregation",
                )
                secondary_aggregation = AGGREGATION_OPTIONS[secondary_label]
    else:
        fields = st.columns(3)
        with fields[0]:
            x_options = list(dict.fromkeys([*numeric, *date_candidates])) if chart_type == "histogram" else columns
            x = _select("Category / X-axis", x_options, key="viz_x")
        with fields[1]:
            if chart_type == "box":
                y = _select("Metric / Y-axis", numeric, key="viz_y", allow_none=False)
            elif chart_type not in {"histogram"}:
                metric_options = ["Count records", *numeric]
                selected_metric = st.selectbox("Metric / Y-axis", metric_options)
                y = None if selected_metric == "Count records" else selected_metric
                if y is None:
                    aggregation = "count"
        with fields[2]:
            if chart_type in {"stacked_bar", "grouped_bar"}:
                color = _select(
                    "Stack / group by",
                    [column for column in categorical if column != x],
                    key="viz_bar_color",
                    allow_none=False,
                )
            elif chart_type in {"bar", "line", "area", "scatter", "box"}:
                color = _select(
                    "Break down by",
                    [column for column in categorical if column != x],
                    key="viz_color",
                )
            elif chart_type == "treemap":
                color = _select(
                    "Parent category",
                    [column for column in categorical if column != x],
                    key="viz_parent",
                )

    is_time_series = chart_type in {"line", "area", "period_over_period_change"} and x in date_candidates
    primary_currency_unit = None
    secondary_currency_unit = None
    same_y_axis_scale = False

    if x in categorical and not is_time_series:
        values = sorted(
            dataframe[x].dropna().unique().tolist(),
            key=lambda value: str(value).casefold(),
        )
        include_values = st.multiselect(
            "Category values",
            values,
            help="Leave empty to include every value.",
        )

    selected_metrics = [
        column for column in (x, y, secondary_y) if column
    ]
    money_selected = any(is_currency_column(column) for column in selected_metrics)
    controls = st.columns(4)
    if needs_aggregation and aggregation is None:
        aggregation = controls[0].selectbox(
            "Aggregation", ["sum", "mean", "median", "min", "max"]
        )
    elif needs_aggregation:
        controls[0].text_input("Aggregation", aggregation, disabled=True)
    if date_candidates:
        with st.expander("Date filter and time grouping", expanded=is_time_series):
            apply_time_filter = st.checkbox(
                "Apply date filter",
                value=is_time_series,
                help="Use date controls for this chart.",
            )
            if apply_time_filter:
                time_column = _select(
                    "Date column",
                    date_candidates,
                    key="viz_time_column",
                    allow_none=False,
                )
                dates = pd.to_datetime(dataframe[time_column], errors="coerce").dropna()
                if not dates.empty:
                    date_filter_mode = st.radio(
                        "Date filter option",
                        ["Continuous", "Discrete"],
                        horizontal=True,
                        help="Continuous uses a date range. Discrete uses year, month, week, or day periods.",
                    )
                    if date_filter_mode == "Continuous":
                        minimum_date = dates.min().date()
                        maximum_date = dates.max().date()
                        selected_range = st.slider(
                            "Continuous date range",
                            min_value=minimum_date,
                            max_value=maximum_date,
                            value=(minimum_date, maximum_date),
                            help="Select the exact date window to include in this chart.",
                        )
                        date_range_start, date_range_end = selected_range
                    else:
                        grain_label = st.selectbox(
                            "Discrete date option",
                            ["Year", "Month", "Week", "Day"],
                            key="viz_date_granularity",
                            help="Filter the chart by calendar periods.",
                        )
                        time_grain = grain_label.lower()
                        period_options = discrete_date_period_options(
                            dataframe[time_column],
                            time_grain,
                        )
                        selected_period_labels = st.multiselect(
                            "Date periods",
                            [label for label, _ in period_options],
                            key=f"viz_date_periods_{time_column}_{time_grain}",
                            help="Leave empty to include every period.",
                        )
                        if selected_period_labels:
                            period_lookup = dict(period_options)
                            date_period_values = [
                                period_lookup[label]
                                for label in selected_period_labels
                            ]
                else:
                    st.caption("No valid dates were found for this date column.")
            else:
                st.caption("Date filtering is off for this chart.")

    if is_time_series:
        time_mode = controls[0].selectbox(
            "Timeline mode",
            ["Use date filter settings"],
            disabled=True,
        )
        date_sort = controls[1].selectbox(
            "Date sort",
            ["Oldest to newest", "Newest to oldest"],
            help="Time-series charts use all dates. Choose the date order for the x-axis.",
        )
        sort_descending = date_sort == "Newest to oldest"
        controls[2].caption("All dates in the selected range are included.")
        limit = None
    else:
        sort_descending = controls[1].checkbox(
            "Sort descending",
            value=chart_type == "sorted_percentage_bar",
        )
        max_category_limit = category_limit_max(chart_type, dataframe, x)
        limit = controls[2].number_input(
            "Category limit",
            min_value=1,
            max_value=max_category_limit,
            value=max_category_limit,
            help="Defaults to the maximum allowed for the selected chart.",
        )
    currency_units = {
        "Auto (K/M/B)": "auto",
        "Full currency": "unit",
        "Thousands (K)": "K",
        "Millions (M)": "M",
        "Billions (B)": "B",
    }
    if money_selected:
        if chart_type == "dual_line":
            controls[3].caption("Dual-line axis display is configured below.")
            currency_unit = "auto"
        else:
            currency_unit_label = controls[3].selectbox(
                "Money display",
                list(currency_units),
            )
            currency_unit = currency_units[currency_unit_label]
    else:
        controls[3].caption("Money display applies to monetary metrics.")
        currency_unit = "auto"
    if chart_type == "dual_line":
        axis_controls = st.columns(3)
        primary_money = bool(y and is_currency_column(y))
        secondary_money = bool(secondary_y and is_currency_column(secondary_y))
        if primary_money:
            primary_unit_label = axis_controls[0].selectbox(
                "Primary axis money display",
                list(currency_units),
                key="viz_dual_primary_money",
            )
            primary_currency_unit = currency_units[primary_unit_label]
        else:
            axis_controls[0].caption("Primary metric is not monetary.")
        if secondary_money:
            secondary_unit_label = axis_controls[1].selectbox(
                "Secondary axis money display",
                list(currency_units),
                key="viz_dual_secondary_money",
            )
            secondary_currency_unit = currency_units[secondary_unit_label]
        else:
            axis_controls[1].caption("Secondary metric is not monetary.")
        same_y_axis_scale = axis_controls[2].checkbox(
            "Use same Y-axis scale",
            help="Apply the same numeric range to both dual-line axes after any selected display units.",
        )
    suggested_title = build_chart_title(
        chart_label,
        chart_type,
        x,
        y,
        secondary_y,
        color,
        aggregation,
        primary_aggregation,
        secondary_aggregation,
    )
    title = st.text_input(
        "Chart title",
        value=suggested_title,
        key=(
            f"viz_title_{chart_type}_{x}_{y}_{secondary_y}_"
            f"{color}_{aggregation}_{primary_aggregation}_{secondary_aggregation}"
        ),
        help="This title is generated from the selected axes and aggregation. You can edit it.",
    )

    if st.button("Create visualization", type="primary"):
        try:
            spec = ChartSpec(
                chart_type=chart_type,
                title=title,
                x=x,
                y=y,
                secondary_y=secondary_y,
                color=color,
                aggregation=aggregation,
                primary_aggregation=primary_aggregation,
                secondary_aggregation=secondary_aggregation,
                sort_descending=sort_descending,
                limit=int(limit) if limit is not None else None,
                include_values=include_values,
                currency_symbol=st.session_state.currency_symbol,
                currency_unit=currency_unit,
                primary_currency_unit=primary_currency_unit,
                secondary_currency_unit=secondary_currency_unit,
                same_y_axis_scale=same_y_axis_scale,
                percentage_denominator_mode=percentage_denominator_mode,
                include_other=include_other,
                show_values=show_values,
                comparison_basis=comparison_basis,
                chart_style=chart_style,
                missing_period_policy=missing_period_policy,
                time_grain=time_grain,
                time_column=time_column,
                date_range_start=date_range_start,
                date_range_end=date_range_end,
                date_period_values=date_period_values,
            )
            figure, result = create_chart(dataframe, spec)
            st.session_state.current_chart = result
            st.plotly_chart(figure, width="stretch")
            insight = generate_chart_insight(result)
            render_chart_insight(insight)
            if st.button("Save chart insight", key="save_manual_chart_new"):
                _save_visualization_insight(result, insight)
            try:
                st.download_button(
                    "Download PNG",
                    figure_to_png(figure),
                    file_name="visualization.png",
                    mime="image/png",
                )
            except ValueError as exc:
                st.caption(str(exc))
        except (ValueError, TypeError) as exc:
            if chart_type == "symbol_map":
                st.error(
                    "The Symbol Map could not prepare its location data. "
                    "Please choose a valid location field and a different optional color field."
                )
            else:
                st.error(str(exc))
    elif st.session_state.get("current_chart"):
        result = st.session_state.current_chart
        figure, _ = create_chart(dataframe, result.spec)
        st.plotly_chart(figure, width="stretch")
        insight = generate_chart_insight(result)
        render_chart_insight(insight)
        if st.button("Save chart insight", key="save_manual_chart_current"):
            _save_visualization_insight(result, insight)
