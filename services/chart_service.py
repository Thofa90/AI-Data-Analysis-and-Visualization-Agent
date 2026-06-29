"""Validated Plotly chart specifications, recommendations, and rendering."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field, model_validator
from services.date_aggregation_service import aggregate_metric_by_period, date_bucket_start
from utils.formatting import format_period, is_currency_column


ChartType = Literal[
    "bar", "stacked_bar", "grouped_bar", "line", "area", "dual_line",
    "scatter", "circle_view", "histogram", "box", "heatmap", "pie",
    "treemap", "symbol_map", "dual_axis", "gantt", "bullet",
    "sorted_percentage_bar", "period_over_period_change", "grouped_extrema_bar"
]
ChartAggregation = Literal["sum", "mean", "median", "min", "max", "count", "nunique"]
CurrencyUnit = Literal["auto", "unit", "K", "M", "B"]
TimeGrain = Literal["year", "quarter", "month", "week", "day"]
MAX_CATEGORY_LIMIT = 1000
SYMBOL_MAP_COLOR_NONE = "__none__"
SYMBOL_MAP_COLOR_LOCATION = "__location__"
SYMBOL_MAP_COLOR_GROUP = "_symbol_color_group"
REPORT_SINGLE_BAR_COLOR = "#315eff"
REPORT_CATEGORICAL_COLORS = (
    "#315eff",
    "#25c2a0",
    "#ff9f40",
    "#9b7cff",
    "#ff6b9a",
    "#35b9e8",
    "#7ac943",
    "#f5c542",
    "#6f86d6",
    "#c678dd",
    "#ff7f50",
    "#14b8a6",
)
REPORT_LINE_COLORS = (
    "#315eff",
    "#25c2a0",
    "#ff9f40",
    "#9b7cff",
    "#ff6b9a",
    "#35b9e8",
    "#7ac943",
    "#f5c542",
    "#6f86d6",
    "#c678dd",
)

REGION_CENTROIDS = {
    "asia": (34.0, 100.0),
    "europe": (54.0, 15.0),
    "northamerica": (45.0, -100.0),
    "sub-saharanafrica": (0.0, 20.0),
    "middleeastandnorthafrica": (27.0, 35.0),
    "centralamericaandthecaribbean": (15.0, -75.0),
    "centralamericaandcaribbean": (15.0, -75.0),
    "australiaandoceania": (-25.0, 140.0),
    "oceania": (-25.0, 140.0),
}


def _time_frequency(grain: TimeGrain) -> str:
    return {
        "year": "Y",
        "quarter": "Q",
        "month": "M",
        "week": "W",
        "day": "D",
    }[grain]


def _period_value_key(value: Any, grain: str) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize().isoformat()


def _period_resample_frequency(grain: TimeGrain | None) -> str:
    return {
        "year": "YS",
        "quarter": "QS",
        "month": "MS",
        "week": "W-MON",
        "day": "D",
    }[grain or "month"]


def _period_label_frequency(grain: TimeGrain | None) -> str:
    return {
        "year": "Y",
        "quarter": "Q",
        "month": "M",
        "week": "W-MON",
        "day": "D",
    }[grain or "month"]


def _aggregation_label(aggregation: str | None) -> str:
    return {
        "sum": "Sum",
        "mean": "Mean",
        "median": "Median",
        "count": "Count",
        "nunique": "Unique Count",
        "min": "Minimum",
        "max": "Maximum",
    }.get(aggregation or "", (aggregation or "Value").title())


def _display_name(value: str | None) -> str:
    if not value:
        return ""
    spaced = "".join(
        f" {character}" if index and character.isupper() and str(value)[index - 1].islower() else character
        for index, character in enumerate(str(value).replace("_", " "))
    )
    return " ".join(token[:1].upper() + token[1:] for token in spaced.split())


def _symbol_map_metric_title(column: str | None, aggregation: str | None) -> str:
    metric = _display_name(column)
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


def _symbol_map_title(location_column: str | None, value_column: str | None, aggregation: str | None, color_column: str | None) -> str:
    metric_title = _symbol_map_metric_title(value_column, aggregation)
    location_name = _display_name(location_column) or "Location"
    if color_column and color_column != SYMBOL_MAP_COLOR_LOCATION:
        return f"{metric_title} by {location_name} and {_display_name(color_column)}"
    return f"{metric_title} by {location_name}"


def effective_symbol_map_color(location_column: str | None, color_column: str | None) -> str | None:
    """Return the stable Symbol Map color mode or real color field."""
    if not color_column or color_column in {"None", SYMBOL_MAP_COLOR_NONE}:
        return None
    if color_column == SYMBOL_MAP_COLOR_LOCATION or color_column == location_column:
        return SYMBOL_MAP_COLOR_LOCATION
    return color_column


def symbol_map_plot_color_column(color_mode: str | None) -> str | None:
    if not color_mode:
        return None
    if color_mode == SYMBOL_MAP_COLOR_LOCATION:
        return SYMBOL_MAP_COLOR_GROUP
    return SYMBOL_MAP_COLOR_GROUP


def _single_column_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    values = dataframe.loc[:, column]
    if isinstance(values, pd.DataFrame):
        return values.iloc[:, 0]
    return values


def prepare_symbol_map_data(
    data: pd.DataFrame,
    location_column: str,
    value_column: str,
    color_column: str | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """Prepare Symbol Map rendering data without duplicate location/color columns."""
    if not location_column:
        raise ValueError("A location field is required for a Symbol Map.")
    if location_column not in data.columns:
        raise KeyError(f"Location field '{location_column}' was not found.")
    if value_column not in data.columns:
        raise KeyError(f"Metric field '{value_column}' was not found.")

    effective_color = effective_symbol_map_color(location_column, color_column)
    real_color_column = (
        effective_color
        if effective_color and effective_color != SYMBOL_MAP_COLOR_LOCATION
        else None
    )
    required_columns = list(dict.fromkeys(
        column
        for column in (location_column, value_column, real_color_column)
        if column is not None
    ))
    map_data = data.loc[:, required_columns].copy()
    location_values = _single_column_series(map_data, location_column)
    map_data["_location_label"] = (
        location_values
        .astype("string")
        .fillna("Unknown location")
        .str.strip()
    )
    if effective_color == SYMBOL_MAP_COLOR_LOCATION:
        map_data[SYMBOL_MAP_COLOR_GROUP] = map_data["_location_label"]
    elif real_color_column:
        map_data[SYMBOL_MAP_COLOR_GROUP] = (
            _single_column_series(map_data, real_color_column)
            .astype("string")
            .fillna("Unknown")
            .str.strip()
        )
    return map_data, effective_color


def _aggregate_metric(
    grouped: pd.core.groupby.generic.DataFrameGroupBy,
    column: str,
    aggregation: str,
) -> pd.Series:
    series_group = grouped[column]
    if aggregation == "sum":
        return series_group.sum(min_count=1)
    if aggregation == "mean":
        return series_group.mean()
    if aggregation == "median":
        return series_group.median()
    if aggregation == "count":
        return series_group.count()
    if aggregation == "nunique":
        return series_group.nunique(dropna=True)
    if aggregation == "min":
        return series_group.min()
    if aggregation == "max":
        return series_group.max()
    raise ValueError(f'Unsupported aggregation "{aggregation}".')


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


def _normalized_location(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum() or character == "-")


def _looks_temporal(frame: pd.DataFrame, column: str | None) -> bool:
    if not column or column not in frame:
        return False
    lowered = column.lower()
    return (
        pd.api.types.is_datetime64_any_dtype(frame[column])
        or any(token in lowered for token in ("date", "time", "timestamp", "year", "month"))
    )


def _sort_axis_values(frame: pd.DataFrame, column: str, *, descending: bool = False) -> pd.DataFrame:
    """Sort an axis using datetime order when possible, preserving the original values."""
    if _looks_temporal(frame, column):
        order = pd.to_datetime(frame[column], errors="coerce")
        if order.notna().any():
            return (
                frame.assign(_axis_order=order)
                .sort_values("_axis_order", ascending=not descending)
                .drop(columns="_axis_order")
            )
    return frame.sort_values(column, ascending=not descending)


def _apply_time_options(dataframe: pd.DataFrame, spec: "ChartSpec") -> pd.DataFrame:
    """Apply date-range filtering and optional discrete time bucketing."""
    date_column = spec.time_column or spec.x
    if not date_column or date_column not in dataframe:
        return dataframe
    if not (
        _looks_temporal(dataframe, date_column)
        or spec.date_range_start is not None
        or spec.date_range_end is not None
        or spec.time_grain is not None
    ):
        return dataframe
    dates = pd.to_datetime(dataframe[date_column], errors="coerce", format="mixed")
    working = dataframe.loc[dates.notna()].copy()
    dates = dates.loc[dates.notna()]
    if spec.date_range_start is not None:
        start = pd.to_datetime(spec.date_range_start, errors="coerce")
        if pd.notna(start):
            mask = dates >= start
            working = working.loc[mask]
            dates = dates.loc[mask]
    if spec.date_range_end is not None:
        end = pd.to_datetime(spec.date_range_end, errors="coerce")
        if pd.notna(end):
            if end == end.normalize():
                mask = dates < end + pd.Timedelta(days=1)
            else:
                mask = dates <= end
            working = working.loc[mask]
            dates = dates.loc[mask]
    if working.empty:
        raise ValueError("No rows matched the selected date range.")
    if spec.time_grain:
        bucketed_dates = date_bucket_start(dates, spec.time_grain)
        if spec.date_period_values:
            selected_periods = set()
            for value in spec.date_period_values:
                key = _period_value_key(value, spec.time_grain)
                if key:
                    selected_periods.add(key)
            period_mask = bucketed_dates.map(lambda value: pd.Timestamp(value).normalize().isoformat()).isin(selected_periods)
            working = working.loc[period_mask]
            dates = dates.loc[period_mask]
            bucketed_dates = bucketed_dates.loc[period_mask]
            if working.empty:
                raise ValueError("No rows matched the selected discrete date periods.")
        if spec.x == date_column:
            working[spec.x] = bucketed_dates
    return working


class ChartSpec(BaseModel):
    """Transparent chart metadata used for rendering and evaluation."""

    chart_type: ChartType
    title: str
    x: str | None = None
    y: str | None = None
    secondary_y: str | None = None
    value_columns: list[str] = Field(default_factory=list)
    color: str | None = None
    aggregation: ChartAggregation | None = None
    sort_descending: bool = False
    limit: int | None = Field(default=None, ge=1, le=MAX_CATEGORY_LIMIT)
    filter_column: str | None = None
    filter_value: Any | None = None
    include_values: list[Any] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)
    currency_symbol: str = "$"
    currency_unit: CurrencyUnit = "auto"
    primary_currency_unit: CurrencyUnit | None = None
    secondary_currency_unit: CurrencyUnit | None = None
    same_y_axis_scale: bool = False
    percentage_denominator_mode: Literal["full_filtered", "displayed"] = "full_filtered"
    include_other: bool = False
    show_values: bool = True
    comparison_basis: Literal["previous_period", "same_period_last_year"] = "previous_period"
    chart_style: Literal["diverging_bar"] = "diverging_bar"
    missing_period_policy: Literal["preserve_missing", "zero_fill_additive"] = "preserve_missing"
    primary_aggregation: ChartAggregation | None = None
    secondary_aggregation: ChartAggregation | None = None
    time_grain: TimeGrain | None = None
    time_column: str | None = None
    date_range_start: Any | None = None
    date_range_end: Any | None = None
    date_period_values: list[Any] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_required_axes(self) -> "ChartSpec":
        if self.chart_type == "symbol_map":
            original_color = self.color
            self.color = effective_symbol_map_color(self.x, self.color)
            if original_color and original_color == self.x and self.title.startswith("Symbol Map:"):
                self.title = _symbol_map_title(self.x, self.y, self.aggregation, self.color)
        required = {
            "bar": ("x", "y"),
            "grouped_extrema_bar": ("x", "y", "color"),
            "sorted_percentage_bar": ("x", "y"),
            "period_over_period_change": ("x", "y"),
            "line": ("x", "y"),
            "area": ("x", "y"),
            "dual_line": ("x", "y", "secondary_y"),
            "scatter": ("x", "y"),
            "circle_view": ("x", "y"),
            "histogram": ("x",),
            "box": ("y",),
            "pie": ("x", "y"),
            "stacked_bar": ("x", "y", "color"),
            "grouped_bar": ("x", "y", "color"),
            "treemap": ("x", "y"),
            "symbol_map": ("x", "y"),
            "dual_axis": ("x", "y", "secondary_y"),
            "gantt": ("x", "y", "secondary_y"),
            "bullet": ("x", "y", "secondary_y"),
        }
        for axis in required.get(self.chart_type, ()):
            if getattr(self, axis) is None and not (
                axis == "y"
                and (
                    (self.chart_type == "bar" and self.value_columns)
                    or (
                        self.chart_type in {"bar", "pie", "treemap", "symbol_map"}
                        and self.aggregation == "count"
                    )
                )
            ):
                raise ValueError(f"{self.chart_type} charts require the {axis}-axis.")
        return self


class ChartResult(BaseModel):
    """Structured chart output with the verified plotted values."""

    spec: ChartSpec
    data: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def validate_columns(dataframe: pd.DataFrame, *columns: str | None) -> None:
    """Reject missing columns before any chart calculation."""
    missing = [column for column in columns if column and column not in dataframe.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {', '.join(missing)}.")


def recommend_chart(
    dataframe: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
) -> ChartType:
    """Recommend a readable chart using deterministic type rules."""
    validate_columns(dataframe, x, y)
    if x and y:
        x_is_date = pd.api.types.is_datetime64_any_dtype(dataframe[x]) or "date" in x.lower() or "time" in x.lower()
        x_numeric = pd.api.types.is_numeric_dtype(dataframe[x])
        y_numeric = pd.api.types.is_numeric_dtype(dataframe[y])
        if x_is_date and y_numeric:
            return "line"
        if x_numeric and y_numeric:
            return "scatter"
        if y_numeric:
            return "bar"
    if x and pd.api.types.is_numeric_dtype(dataframe[x]):
        return "histogram"
    if x:
        return "bar"
    numeric_count = len(dataframe.select_dtypes(include="number").columns)
    return "heatmap" if numeric_count >= 2 else "bar"


def _filtered_chart_dataframe(dataframe: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    color_column = (
        None
        if spec.chart_type == "symbol_map" and spec.color == SYMBOL_MAP_COLOR_LOCATION
        else spec.color
    )
    validate_columns(
        dataframe,
        spec.x,
        None
        if spec.aggregation == "count" and spec.y == "Count" and "Count" not in dataframe.columns
        else spec.y if not spec.value_columns else None,
        spec.secondary_y,
        color_column if not spec.value_columns else None,
        spec.filter_column,
        spec.time_column,
        *spec.value_columns,
    )
    working = _apply_time_options(dataframe, spec)
    if spec.include_values:
        if spec.x is None:
            raise ValueError("Selected chart values require an x-axis column.")
        selected = {str(value).casefold() for value in spec.include_values}
        working = working.loc[
            working[spec.x].astype("string").str.casefold().isin(selected)
        ]
        if working.empty:
            raise ValueError("None of the selected chart values were found.")
    if spec.filter_column is not None:
        series = working[spec.filter_column]
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            mask = series.astype("string").str.casefold() == str(spec.filter_value).casefold()
        else:
            mask = series == spec.filter_value
        working = working.loc[mask]
        if working.empty:
            raise ValueError(
                f'No rows matched {spec.filter_column} = "{spec.filter_value}".'
            )
    return working


def _period_lag(grain: TimeGrain | None, comparison_basis: str) -> pd.DateOffset:
    if comparison_basis == "same_period_last_year":
        return pd.DateOffset(years=1)
    return {
        "day": pd.DateOffset(days=1),
        "week": pd.DateOffset(weeks=1),
        "month": pd.DateOffset(months=1),
        "quarter": pd.DateOffset(months=3),
        "year": pd.DateOffset(years=1),
    }[grain or "month"]


def _period_change_data(dataframe: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    validate_columns(dataframe, spec.x, spec.y, spec.filter_column)
    if not spec.x or not spec.y:
        raise ValueError("Period-over-Period % Change requires a date field and numeric metric.")
    if not pd.api.types.is_numeric_dtype(dataframe[spec.y]) and spec.aggregation not in {"count", "nunique"}:
        raise ValueError("Period-over-Period % Change requires a numeric metric for this aggregation.")

    dates = pd.to_datetime(dataframe[spec.x], errors="coerce", format="mixed")
    working = dataframe.loc[dates.notna()].copy()
    working["_period_date"] = dates.loc[dates.notna()]
    if working.empty:
        raise ValueError("No valid dates were available for period-over-period calculation.")
    if spec.filter_column is not None:
        series = working[spec.filter_column]
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            mask = series.astype("string").str.casefold() == str(spec.filter_value).casefold()
        else:
            mask = series == spec.filter_value
        working = working.loc[mask]
        if working.empty:
            raise ValueError(f'No rows matched {spec.filter_column} = "{spec.filter_value}".')

    grain = spec.time_grain or "month"
    frequency = _period_resample_frequency(grain)
    lag = _period_lag(grain, spec.comparison_basis)
    source_start = working["_period_date"].min()
    source_end = working["_period_date"].max()
    display_start = pd.to_datetime(spec.date_range_start, errors="coerce") if spec.date_range_start is not None else source_start
    display_end = pd.to_datetime(spec.date_range_end, errors="coerce") if spec.date_range_end is not None else source_end
    if pd.isna(display_start):
        display_start = source_start
    if pd.isna(display_end):
        display_end = source_end
    if not isinstance(display_start, pd.Timestamp):
        display_start = pd.Timestamp(display_start)
    if not isinstance(display_end, pd.Timestamp):
        display_end = pd.Timestamp(display_end)
    if display_end.time() == pd.Timestamp(display_end.date()).time():
        display_end = display_end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    calculation_start = display_start - lag
    calculation = working.loc[(working["_period_date"] >= calculation_start) & (working["_period_date"] <= display_end)].copy()
    if calculation.empty:
        raise ValueError("No rows matched the selected date range.")

    aggregation = spec.aggregation or "sum"
    calculation = calculation.set_index("_period_date").sort_index()
    if aggregation == "count":
        aggregated = calculation[spec.y].resample(frequency).count()
    elif aggregation == "nunique":
        aggregated = calculation[spec.y].resample(frequency).nunique()
    else:
        aggregated = getattr(calculation[spec.y].resample(frequency), aggregation)()
    if aggregated.empty:
        raise ValueError("No periods were available after aggregation.")
    expected = pd.date_range(aggregated.index.min(), aggregated.index.max(), freq=frequency)
    aggregated = aggregated.reindex(expected)
    if spec.missing_period_policy == "zero_fill_additive" and aggregation in {"sum", "count"}:
        aggregated = aggregated.fillna(0)

    lag_periods = 12 if spec.comparison_basis == "same_period_last_year" and grain == "month" else 1
    if spec.comparison_basis == "same_period_last_year":
        if grain == "day":
            baseline = aggregated.shift(365)
        elif grain == "week":
            baseline = aggregated.shift(52)
        elif grain == "quarter":
            baseline = aggregated.shift(4)
        elif grain == "year":
            baseline = aggregated.shift(1)
        else:
            baseline = aggregated.shift(lag_periods)
    else:
        baseline = aggregated.shift(1)
    absolute_change = aggregated - baseline
    percentage_change = absolute_change / baseline.abs() * 100
    percentage_change = percentage_change.where(baseline != 0)
    result = pd.DataFrame({
        spec.x: aggregated.index,
        spec.y: aggregated.values,
        "comparison_value": baseline.values,
        "absolute_change": absolute_change.values,
        "percentage_change": percentage_change.values,
    })
    result["change_direction"] = result["percentage_change"].map(
        lambda value: "increase" if pd.notna(value) and value > 0 else "decrease" if pd.notna(value) and value < 0 else "no_change" if pd.notna(value) else "unavailable"
    )
    display_period_start = display_start.to_period(_period_label_frequency(grain)).start_time
    display_mask = (result[spec.x] >= display_period_start) & (result[spec.x] <= display_end)
    result = result.loc[display_mask].reset_index(drop=True)
    result.attrs["period_change_metadata"] = {
        "source_row_count": int(len(working)),
        "calculation_row_count": int(len(calculation)),
        "display_start": display_start,
        "display_end": display_end,
        "calculation_start": calculation_start,
        "granularity": grain,
        "frequency": frequency,
        "comparison_basis": spec.comparison_basis,
        "missing_period_count": int(aggregated.isna().sum()),
    }
    return result


def _aggregate_data(dataframe: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    if spec.chart_type == "period_over_period_change":
        return _period_change_data(dataframe, spec)
    working = _filtered_chart_dataframe(dataframe, spec)
    if spec.chart_type == "grouped_extrema_bar":
        if not spec.x or not spec.y or not spec.color:
            raise ValueError("Grouped extrema charts require a primary group, metric, and secondary group.")
        validate_columns(dataframe, spec.x, spec.y, spec.color)
        if not pd.api.types.is_numeric_dtype(dataframe[spec.y]):
            raise ValueError("Grouped extrema charts require a numeric metric.")
        aggregated = (
            working.dropna(subset=[spec.x, spec.color, spec.y])
            .groupby([spec.x, spec.color], dropna=False)[spec.y]
            .agg(spec.aggregation or "sum")
            .reset_index()
        )
        if aggregated.empty:
            raise ValueError("No grouped values were available.")
        group_target = aggregated.groupby(spec.x)[spec.y].transform("max")
        winners = aggregated.loc[
            (aggregated[spec.y].astype(float) - group_target.astype(float)).abs() <= 1e-9
        ].copy()
        winners["Tie"] = winners.groupby(spec.x)[spec.color].transform("count") > 1
        return winners.sort_values([spec.x, spec.color]).reset_index(drop=True)
    if spec.chart_type == "heatmap":
        source = working[spec.value_columns] if spec.value_columns else working
        numeric = source.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            raise ValueError("A correlation heatmap requires at least two numeric columns.")
        return numeric.corr(numeric_only=True)
    if spec.chart_type == "gantt":
        columns = [
            column for column in (spec.x, spec.y, spec.secondary_y, spec.color)
            if column
        ]
        result = working[columns].dropna().copy()
        result[spec.x] = pd.to_datetime(result[spec.x], errors="coerce")
        result[spec.secondary_y] = pd.to_datetime(
            result[spec.secondary_y], errors="coerce"
        )
        result = result.dropna(subset=[spec.x, spec.secondary_y])
        if result.empty:
            raise ValueError("Gantt charts require valid start and end dates.")
        return result
    if spec.chart_type in {"histogram", "scatter", "circle_view", "box"} or (
        not spec.aggregation and spec.chart_type not in {"dual_line", "dual_axis"}
    ):
        columns = [
            column for column in (spec.x, spec.y, spec.secondary_y, spec.color)
            if column
        ]
        columns = list(dict.fromkeys(columns))
        return working[columns].dropna().copy()
    if spec.x is None:
        raise ValueError("Aggregated charts require a grouping column.")
    if spec.chart_type == "dual_line":
        if not spec.y or not spec.secondary_y:
            raise ValueError("This chart requires two numeric metrics.")
        if any(
            not pd.api.types.is_numeric_dtype(dataframe[column])
            for column in (spec.y, spec.secondary_y)
        ):
            raise ValueError("This chart requires numeric metrics.")
        primary_aggregation = spec.primary_aggregation or spec.aggregation or "sum"
        secondary_aggregation = spec.secondary_aggregation or spec.aggregation or "sum"
        grouped = working.groupby(spec.x, dropna=False)
        primary = (
            _aggregate_metric(grouped, spec.y, primary_aggregation)
            .reset_index(name=spec.y)
        )
        secondary = (
            _aggregate_metric(grouped, spec.secondary_y, secondary_aggregation)
            .reset_index(name=spec.secondary_y)
        )
        result = primary.merge(secondary, on=spec.x, how="outer", validate="one_to_one")
        if spec.sort_descending:
            result = result.sort_values(spec.y, ascending=False, na_position="last")
        if spec.limit:
            result = result.head(spec.limit)
        return result
    if spec.chart_type == "dual_axis":
        if not spec.y or not spec.secondary_y:
            raise ValueError("This chart requires two numeric metrics.")
        if any(
            not pd.api.types.is_numeric_dtype(dataframe[column])
            for column in (spec.y, spec.secondary_y)
        ):
            raise ValueError("This chart requires numeric metrics.")
        primary_aggregation = spec.primary_aggregation or spec.aggregation or "sum"
        secondary_aggregation = spec.secondary_aggregation or spec.aggregation or "sum"
        grouped = working.groupby(spec.x, dropna=False)
        primary = (
            _aggregate_metric(grouped, spec.y, primary_aggregation)
            .reset_index(name=spec.y)
        )
        secondary = (
            _aggregate_metric(grouped, spec.secondary_y, secondary_aggregation)
            .reset_index(name=spec.secondary_y)
        )
        result = primary.merge(secondary, on=spec.x, how="outer", validate="one_to_one")
        if spec.sort_descending:
            result = result.sort_values(spec.y, ascending=False)
        if spec.limit:
            result = result.head(spec.limit)
        return result
    if spec.chart_type == "bullet":
        if not spec.y or not spec.secondary_y:
            raise ValueError("This chart requires two numeric metrics.")
        if any(
            not pd.api.types.is_numeric_dtype(dataframe[column])
            for column in (spec.y, spec.secondary_y)
        ):
            raise ValueError("This chart requires numeric metrics.")
        result = (
            working.groupby(spec.x, dropna=False)[[spec.y, spec.secondary_y]]
            .agg(spec.aggregation or "sum")
            .reset_index()
        )
        if spec.sort_descending:
            result = result.sort_values(spec.y, ascending=False)
        if spec.limit:
            result = result.head(spec.limit)
        return result
    if spec.chart_type == "sorted_percentage_bar":
        if not spec.x or not spec.y:
            raise ValueError("Sorted Percentage Bar requires a category and numeric metric.")
        if not pd.api.types.is_numeric_dtype(dataframe[spec.y]):
            raise ValueError("Sorted Percentage Bar requires a numeric metric.")
        aggregation = spec.aggregation or "sum"
        grouped = working.groupby(spec.x, dropna=False)
        all_summary = (
            _aggregate_metric(grouped, spec.y, aggregation)
            .reset_index(name="aggregated_value")
        )
        all_summary["aggregated_value"] = pd.to_numeric(all_summary["aggregated_value"], errors="coerce")
        all_summary = all_summary.dropna(subset=["aggregated_value"])
        if all_summary.empty:
            raise ValueError("No numeric values were available for the percentage chart.")
        all_summary = all_summary.sort_values(
            "aggregated_value",
            ascending=not spec.sort_descending,
            kind="mergesort",
        )
        full_total = float(all_summary["aggregated_value"].sum())
        original_count = int(len(all_summary))
        summary = all_summary.copy()
        other_row: pd.DataFrame | None = None
        if spec.limit and len(summary) > spec.limit:
            retained = summary.head(spec.limit)
            excluded = summary.iloc[spec.limit:]
            if spec.include_other and not excluded.empty:
                other_row = pd.DataFrame({
                    spec.x: ["Other"],
                    "aggregated_value": [float(excluded["aggregated_value"].sum())],
                })
                summary = pd.concat([retained, other_row], ignore_index=True)
            else:
                summary = retained.copy()
        denominator = (
            full_total
            if spec.percentage_denominator_mode == "full_filtered"
            else float(summary["aggregated_value"].sum())
        )
        valid_percentage = (
            aggregation in {"sum", "count"}
            and denominator > 0
            and not (all_summary["aggregated_value"] < 0).any()
        )
        summary["percentage_share"] = (
            summary["aggregated_value"] / denominator * 100
            if valid_percentage
            else pd.NA
        )
        summary["rank"] = range(1, len(summary) + 1)
        summary.attrs["sorted_percentage_bar_metadata"] = {
            "original_category_count": original_count,
            "full_filtered_total": full_total,
            "displayed_total": float(summary["aggregated_value"].sum()),
            "percentage_denominator": denominator,
            "percentage_valid": valid_percentage,
            "negative_value_count": int((all_summary["aggregated_value"] < 0).sum()),
            "other_category_present": bool(other_row is not None),
        }
        return summary
    if spec.value_columns:
        if any(
            not pd.api.types.is_numeric_dtype(dataframe[column])
            for column in spec.value_columns
        ):
            raise ValueError("Multi-metric charts require numeric value columns.")
        result = (
            working.groupby(spec.x, dropna=False)[spec.value_columns]
            .agg(spec.aggregation)
            .reset_index()
        )
        if spec.sort_descending:
            result = result.sort_values(spec.value_columns[0], ascending=False)
        if spec.limit:
            result = result.head(spec.limit)
        return result.melt(
            id_vars=[spec.x],
            value_vars=spec.value_columns,
            var_name="Metric",
            value_name="Value",
        )
    group_columns = [spec.x]
    if spec.color and spec.color not in {spec.x, SYMBOL_MAP_COLOR_LOCATION}:
        group_columns.append(spec.color)
    if spec.aggregation == "count":
        result = (
            working.groupby(group_columns, dropna=False)
            .size()
            .reset_index(name=spec.y or "Count")
        )
    elif spec.aggregation == "nunique":
        if spec.y is None:
            raise ValueError("Unique-count charts require a y-axis column.")
        result = (
            working.groupby(group_columns, dropna=False)[spec.y]
            .nunique(dropna=True)
            .reset_index()
        )
    else:
        if spec.y is None or not pd.api.types.is_numeric_dtype(dataframe[spec.y]):
            raise ValueError("This aggregation requires a numeric y-axis column.")
        result = (
            working.groupby(group_columns, dropna=False)[spec.y]
            .agg(spec.aggregation)
            .reset_index()
        )
    y_column = spec.y or "Count"
    is_time_series = spec.chart_type in {"line", "area"} and _looks_temporal(result, spec.x)
    if is_time_series and spec.x:
        result = _sort_axis_values(result, spec.x, descending=spec.sort_descending)
    elif spec.chart_type == "stacked_bar" and spec.color and spec.color != spec.x:
        category_totals = (
            result.groupby(spec.x, dropna=False)[y_column]
            .sum()
            .sort_values(ascending=not spec.sort_descending)
        )
        if spec.limit:
            category_totals = category_totals.head(spec.limit)
            result = result.loc[result[spec.x].isin(category_totals.index)]
        order_map = {category: index for index, category in enumerate(category_totals.index)}
        result = (
            result.assign(_category_order=result[spec.x].map(order_map))
            .sort_values(["_category_order", spec.color])
            .drop(columns="_category_order")
        )
    elif spec.sort_descending or spec.limit:
        result = result.sort_values(y_column, ascending=not spec.sort_descending)
    if spec.limit and spec.chart_type != "stacked_bar" and not is_time_series:
        result = result.head(spec.limit)
    return result


def create_chart(dataframe: pd.DataFrame, spec: ChartSpec) -> tuple[go.Figure, ChartResult]:
    """Calculate plotted data and build a consistently styled Plotly figure."""
    raw_plotted = _aggregate_data(dataframe, spec)
    plotted = raw_plotted.copy()
    metadata: dict[str, Any] = {}
    if spec.chart_type == "heatmap":
        filtered = _filtered_chart_dataframe(dataframe, spec)
        source = filtered[spec.value_columns] if spec.value_columns else filtered
        numeric = source.select_dtypes(include="number")
        metadata = {
            "heatmap_type": "correlation",
            "correlation_method": "pearson",
            "raw_row_count": int(len(dataframe)),
            "filtered_row_count": int(len(filtered)),
            "selected_columns": list(numeric.columns),
            "numeric_data": numeric.to_dict(orient="list"),
        }
    elif spec.chart_type in {"pie", "treemap", "symbol_map"} and spec.x:
        filtered = _filtered_chart_dataframe(dataframe, spec)
        metadata = {
            "original_category_count": int(filtered[spec.x].nunique(dropna=False)),
            "filtered_row_count": int(len(filtered)),
        }
    elif spec.chart_type == "sorted_percentage_bar":
        filtered = _filtered_chart_dataframe(dataframe, spec)
        metadata = {
            **raw_plotted.attrs.get("sorted_percentage_bar_metadata", {}),
            "filtered_row_count": int(len(filtered)),
            "date_column": spec.time_column,
            "date_range_start": spec.date_range_start,
            "date_range_end": spec.date_range_end,
            "date_period_values": list(spec.date_period_values),
        }
    date_summary_column = spec.y or ("Count" if spec.aggregation == "count" else None)
    date_column = spec.time_column or (spec.x if spec.x and _looks_temporal(dataframe, spec.x) else None)
    if date_column and date_summary_column and (date_summary_column == "Count" or date_summary_column in dataframe.columns):
        try:
            summary_metric = date_summary_column if date_summary_column != "Count" else spec.y or date_summary_column
            if date_summary_column == "Count" and spec.y is None:
                temp = dataframe.copy()
                temp["Count"] = 1
                summary_source = temp
                summary_metric = "Count"
            else:
                summary_source = dataframe
            date_summary = aggregate_metric_by_period(
                summary_source,
                date_column,
                summary_metric,
                aggregation=spec.aggregation or "sum",
                start_date=spec.date_range_start,
                end_date=spec.date_range_end,
                period_type=spec.time_grain,
                period_values=list(spec.date_period_values) if spec.date_period_values else None,
            )
            metadata["date_summary"] = date_summary.as_dict()
        except Exception:
            pass
    elif spec.chart_type == "period_over_period_change":
        metadata = dict(raw_plotted.attrs.get("period_change_metadata", {}))
    labels = dict(spec.labels or {})
    unit_divisors = {
        "unit": 1,
        "K": 1_000,
        "M": 1_000_000,
        "B": 1_000_000_000,
    }
    def unit_for_column(column: str | None) -> CurrencyUnit:
        if spec.chart_type == "dual_line":
            if column == spec.y and spec.primary_currency_unit:
                return spec.primary_currency_unit
            if column == spec.secondary_y and spec.secondary_currency_unit:
                return spec.secondary_currency_unit
        return spec.currency_unit

    def unit_suffix(unit: CurrencyUnit) -> str:
        return "" if unit in {"auto", "unit"} else unit

    divisor = unit_divisors.get(spec.currency_unit, 1)
    default_unit_suffix = unit_suffix(spec.currency_unit)
    if (
        spec.currency_unit != "auto"
        or spec.primary_currency_unit not in {None, "auto"}
        or spec.secondary_currency_unit not in {None, "auto"}
    ):
        if spec.value_columns and {"Metric", "Value"}.issubset(plotted.columns):
            for metric in spec.value_columns:
                if is_currency_column(metric):
                    mask = plotted["Metric"] == metric
                    plotted.loc[mask, "Value"] = plotted.loc[mask, "Value"] / divisor
            labels["Value"] = f"Value ({spec.currency_symbol}{default_unit_suffix})"
        else:
            for column in (spec.x, spec.y, spec.secondary_y):
                column_unit = unit_for_column(column)
                if (
                    column
                    and column in plotted.columns
                    and is_currency_column(column)
                    and pd.api.types.is_numeric_dtype(plotted[column])
                    and column_unit != "auto"
                ):
                    column_divisor = unit_divisors.get(column_unit, 1)
                    plotted[column] = plotted[column] / divisor
                    plotted[column] = plotted[column] * divisor / column_divisor
                    labels[column] = (
                        f"{column} ({spec.currency_symbol}{unit_suffix(column_unit)})"
                    )
    if spec.chart_type == "sorted_percentage_bar":
        labels["percentage_share"] = f"Share of {_display_name(spec.y)} (%)"
        labels["aggregated_value"] = _display_name(spec.y)
        labels[spec.x] = _display_name(spec.x)
    if spec.chart_type == "period_over_period_change":
        labels["percentage_change"] = "% Change"
        labels["absolute_change"] = f"Change in {_display_name(spec.y)}"
        labels["comparison_value"] = "Comparison value"
        labels[spec.y] = _display_name(spec.y)
        labels[spec.x] = _display_name(spec.x)
    if spec.chart_type in {"bar", "stacked_bar", "grouped_bar", "grouped_extrema_bar"}:
        y_column = (
            "Value"
            if spec.value_columns
            else "Count"
            if spec.aggregation == "count" and spec.y is None
            else spec.y
        )
        figure = px.bar(
            plotted,
            x=spec.x,
            y=y_column,
            color="Metric" if spec.value_columns else spec.color,
            barmode=(
                "stack"
                if spec.chart_type == "stacked_bar"
                else "group"
                if spec.chart_type in {"grouped_bar", "grouped_extrema_bar"} or spec.value_columns
                else "relative"
            ),
            title=spec.title,
            labels=labels,
            category_orders=(
                {spec.x: raw_plotted[spec.x].drop_duplicates().tolist()}
                if spec.chart_type == "stacked_bar" and spec.x in raw_plotted
                else None
            ),
        )
    elif spec.chart_type == "line":
        plotted = _sort_axis_values(plotted, spec.x, descending=spec.sort_descending)
        figure = px.line(plotted, x=spec.x, y=spec.y, color=spec.color, markers=True, title=spec.title, labels=labels)
        if spec.x in plotted and _looks_temporal(plotted, spec.x):
            granularity = _infer_time_granularity(plotted[spec.x], spec.time_grain)
            period_labels = plotted[spec.x].map(lambda value: format_period(value, granularity))
            figure.update_traces(
                customdata=period_labels,
                hovertemplate=(
                    f"{spec.x}: %{{customdata}}<br>"
                    f"{spec.y}: %{{y}}<extra></extra>"
                ),
            )
    elif spec.chart_type == "area":
        plotted = _sort_axis_values(plotted, spec.x, descending=spec.sort_descending)
        figure = px.area(
            plotted, x=spec.x, y=spec.y, color=spec.color,
            title=spec.title, labels=labels,
        )
        if spec.x in plotted and _looks_temporal(plotted, spec.x):
            granularity = _infer_time_granularity(plotted[spec.x], spec.time_grain)
            period_labels = plotted[spec.x].map(lambda value: format_period(value, granularity))
            figure.update_traces(
                customdata=period_labels,
                hovertemplate=(
                    f"{spec.x}: %{{customdata}}<br>"
                    f"{spec.y}: %{{y}}<extra></extra>"
                ),
            )
    elif spec.chart_type == "scatter":
        figure = px.scatter(plotted, x=spec.x, y=spec.y, color=spec.color, title=spec.title, labels=labels)
    elif spec.chart_type == "circle_view":
        hover_columns = [
            column for column in (spec.x, spec.y, spec.secondary_y, spec.color)
            if column
        ]
        figure = px.scatter(
            plotted,
            x=spec.x,
            y=spec.y,
            size=spec.secondary_y,
            color=spec.color,
            hover_data={column: True for column in hover_columns},
            title=spec.title,
            labels=labels,
            size_max=55,
        )
    elif spec.chart_type == "dual_line":
        figure = make_subplots(specs=[[{"secondary_y": True}]])
        primary_axis_title = (
            f"{_display_name(spec.y)} ({_aggregation_label(spec.primary_aggregation or spec.aggregation or 'sum')})"
            if spec.y else ""
        )
        secondary_axis_title = (
            f"{_display_name(spec.secondary_y)} ({_aggregation_label(spec.secondary_aggregation or spec.aggregation or 'sum')})"
            if spec.secondary_y else ""
        )
        for metric, secondary, color in (
            (spec.y, False, "#315eff"),
            (spec.secondary_y, True, "#24dc83"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=plotted[spec.x],
                    y=plotted[metric],
                    name=metric,
                    mode="lines+markers",
                    line={"color": color, "width": 3},
                ),
                secondary_y=secondary,
            )
        figure.update_layout(title=spec.title)
        figure.update_yaxes(title_text=primary_axis_title, secondary_y=False)
        figure.update_yaxes(title_text=secondary_axis_title, secondary_y=True)
    elif spec.chart_type == "dual_axis":
        figure = make_subplots(specs=[[{"secondary_y": True}]])
        figure.add_trace(
            go.Bar(
                x=plotted[spec.x],
                y=plotted[spec.y],
                name=spec.y,
                marker_color="#315eff",
                opacity=0.82,
            ),
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=plotted[spec.x],
                y=plotted[spec.secondary_y],
                name=spec.secondary_y,
                mode="lines+markers",
                line={"color": "#24dc83", "width": 3},
                marker={"size": 8},
            ),
            secondary_y=True,
        )
        figure.update_layout(title=spec.title)
        primary_axis_title = (
            f"{_display_name(spec.y)} ({_aggregation_label(spec.primary_aggregation or spec.aggregation or 'sum')})"
            if spec.y else ""
        )
        secondary_axis_title = (
            f"{_display_name(spec.secondary_y)} ({_aggregation_label(spec.secondary_aggregation or spec.aggregation or 'sum')})"
            if spec.secondary_y else ""
        )
        figure.update_yaxes(title_text=primary_axis_title, secondary_y=False)
        figure.update_yaxes(title_text=secondary_axis_title, secondary_y=True)
    elif spec.chart_type == "period_over_period_change":
        colors = plotted["percentage_change"].map(
            lambda value: "#24dc83" if pd.notna(value) and value >= 0 else "#ff5c7a"
        )
        figure = px.bar(
            plotted,
            x=spec.x,
            y="percentage_change",
            color=colors,
            color_discrete_map="identity",
            text="percentage_change",
            title=spec.title,
            labels=labels,
            hover_data={
                spec.y: ":,.2f",
                "comparison_value": ":,.2f",
                "absolute_change": ":,.2f",
                "percentage_change": ":.1f",
            },
        )
        figure.add_hline(y=0, line_color="rgba(255,255,255,0.6)", line_width=1)
        if spec.show_values:
            figure.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        else:
            figure.update_traces(text=None)
        figure.update_yaxes(title_text="% Change")
    elif spec.chart_type == "histogram":
        figure = px.histogram(plotted, x=spec.x, color=spec.color, title=spec.title, labels=labels)
    elif spec.chart_type == "sorted_percentage_bar":
        use_horizontal = len(plotted) >= 6
        if use_horizontal:
            figure = px.bar(
                plotted,
                x="percentage_share",
                y=spec.x,
                orientation="h",
                text="percentage_share" if spec.show_values else None,
                title=spec.title,
                labels=labels,
                hover_data={
                    "aggregated_value": ":,.2f",
                    "percentage_share": ":.1f",
                    "rank": True,
                },
            )
            figure.update_yaxes(categoryorder="array", categoryarray=plotted[spec.x].tolist()[::-1])
        else:
            figure = px.bar(
                plotted,
                x=spec.x,
                y="percentage_share",
                text="percentage_share" if spec.show_values else None,
                title=spec.title,
                labels=labels,
                hover_data={
                    "aggregated_value": ":,.2f",
                    "percentage_share": ":.1f",
                    "rank": True,
                },
            )
            figure.update_xaxes(categoryorder="array", categoryarray=plotted[spec.x].tolist())
        if spec.show_values:
            figure.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        figure.update_layout(xaxis_title=labels["percentage_share"] if use_horizontal else labels[spec.x])
        figure.update_layout(yaxis_title=labels[spec.x] if use_horizontal else labels["percentage_share"])
    elif spec.chart_type == "box":
        figure = px.box(plotted, x=spec.x, y=spec.y, color=spec.color, title=spec.title, labels=labels)
    elif spec.chart_type == "pie":
        value_column = (
            "Count"
            if spec.aggregation == "count" and spec.y is None
            else spec.y
        )
        figure = px.pie(
            plotted,
            names=spec.x,
            values=value_column,
            title=spec.title,
            labels=labels,
            hole=0.38,
        )
    elif spec.chart_type == "treemap":
        value_column = (
            "Count"
            if spec.aggregation == "count" and spec.y is None
            else spec.y
        )
        path = [column for column in (spec.color, spec.x) if column]
        figure = px.treemap(
            plotted,
            path=path,
            values=value_column,
            title=spec.title,
            labels=labels,
        )
    elif spec.chart_type == "symbol_map":
        value_column = (
            "Count"
            if spec.aggregation == "count" and spec.y is None
            else spec.y
        )
        map_data, map_color = prepare_symbol_map_data(
            plotted,
            spec.x,
            value_column,
            spec.color,
        )
        color_argument = symbol_map_plot_color_column(map_color)
        map_labels = dict(labels)
        if color_argument:
            map_labels[color_argument] = _display_name(spec.x if map_color == SYMBOL_MAP_COLOR_LOCATION else map_color)
        normalized_locations = map_data["_location_label"].map(_normalized_location)
        if normalized_locations.isin(REGION_CENTROIDS).all():
            map_data["_latitude"] = normalized_locations.map(
                lambda value: REGION_CENTROIDS[value][0]
            )
            map_data["_longitude"] = normalized_locations.map(
                lambda value: REGION_CENTROIDS[value][1]
            )
            figure = px.scatter_geo(
                map_data,
                lat="_latitude",
                lon="_longitude",
                size=value_column,
                color=color_argument,
                hover_name="_location_label",
                title=spec.title,
                labels=map_labels,
                size_max=45,
            )
        elif "region" in spec.x.lower():
            raise ValueError(
                "The selected region labels are not geographic locations. "
                "Use a Country/ISO column, or standard macro-regions such as "
                "Asia, Europe, North America, or Sub-Saharan Africa."
            )
        else:
            figure = px.scatter_geo(
                map_data,
                locations=spec.x,
                locationmode="country names",
                size=value_column,
                color=color_argument,
                hover_name="_location_label",
                title=spec.title,
                labels=map_labels,
                size_max=45,
            )
        maximum_size = float(pd.to_numeric(
            map_data[value_column], errors="coerce"
        ).max())
        size_reference = maximum_size / 45 if maximum_size > 0 else 1
        value_template = (
            f"{value_column}: {spec.currency_symbol}%{{marker.size:,.2f}}{unit_suffix}"
            if is_currency_column(value_column)
            else f"{value_column}: %{{marker.size:,.2f}}"
        )
        figure.update_traces(
            marker={
                "sizemode": "diameter",
                "sizeref": size_reference,
                "sizemin": 7,
            },
            hovertemplate=(
                f"{spec.x}: %{{hovertext}}<br>"
                f"{value_template}<extra></extra>"
            ),
        )
    elif spec.chart_type == "gantt":
        figure = px.timeline(
            plotted,
            x_start=spec.x,
            x_end=spec.secondary_y,
            y=spec.y,
            color=spec.color,
            title=spec.title,
            labels=labels,
        )
        figure.update_yaxes(autorange="reversed")
    elif spec.chart_type == "bullet":
        figure = go.Figure()
        for index, row in enumerate(plotted.to_dict(orient="records")):
            axis_maximum = max(row[spec.y], row[spec.secondary_y]) * 1.15
            figure.add_trace(go.Indicator(
                mode="number+gauge",
                value=row[spec.y],
                number={"font": {"size": 28}},
                title={"text": str(row[spec.x]), "font": {"size": 13}},
                gauge={
                    "shape": "bullet",
                    "axis": {"range": [0, axis_maximum], "tickfont": {"size": 10}},
                    "bar": {"color": "#315eff"},
                    "threshold": {
                        "line": {"color": "#ffb020", "width": 3},
                        "value": row[spec.secondary_y],
                    },
                },
                domain={
                    "x": [0.14, 0.96],
                    "y": [
                        max(0, 1 - (index + 1) / len(plotted)),
                        1 - index / len(plotted),
                    ],
                },
            ))
        figure.update_layout(title=spec.title, height=max(280, len(plotted) * 74))
    elif spec.chart_type == "heatmap":
        figure = px.imshow(
            plotted,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=spec.title,
        )
    else:
        raise ValueError(f'Unsupported chart type: "{spec.chart_type}".')
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(4,15,34,0.72)",
        font={"color": "#dce7f8", "size": 12},
        legend={"font": {"color": "#ffffff", "size": 12}},
        title={
            "text": spec.title,
            "x": 0.5,
            "xanchor": "center",
            "y": 0.98,
            "yanchor": "top",
            "font": {"size": 20, "color": "#ffffff"},
        },
        margin={"l": 30, "r": 20, "t": 85, "b": 35},
        hoverlabel={"bgcolor": "#081a35"},
    )
    figure.update_xaxes(gridcolor="rgba(100,140,190,0.16)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(100,140,190,0.16)", zeroline=False)
    if spec.chart_type not in {"circle_view", "symbol_map", "heatmap", "gantt"}:
        if spec.x and is_currency_column(spec.x):
            x_unit = unit_for_column(spec.x)
            figure.update_xaxes(
                tickprefix=spec.currency_symbol,
                ticksuffix=unit_suffix(x_unit),
                tickformat="~s" if x_unit == "auto" else ",.2f",
            )
        if spec.y and is_currency_column(spec.y):
            y_unit = unit_for_column(spec.y)
            figure.update_yaxes(
                tickprefix=spec.currency_symbol,
                ticksuffix=unit_suffix(y_unit),
                tickformat="~s" if y_unit == "auto" else ",.2f",
                secondary_y=False if spec.chart_type in {"dual_axis", "dual_line"} else None,
            )
        if spec.secondary_y and is_currency_column(spec.secondary_y):
            secondary_unit = unit_for_column(spec.secondary_y)
            figure.update_yaxes(
                tickprefix=spec.currency_symbol,
                ticksuffix=unit_suffix(secondary_unit),
                tickformat="~s" if secondary_unit == "auto" else ",.2f",
                secondary_y=True,
            )
        if spec.value_columns and all(
            is_currency_column(column) for column in spec.value_columns
        ):
            figure.update_yaxes(
                tickprefix=spec.currency_symbol,
                ticksuffix=unit_suffix(spec.currency_unit),
                tickformat="~s" if spec.currency_unit == "auto" else ",.2f",
            )
    if spec.chart_type == "dual_line" and spec.same_y_axis_scale and spec.y and spec.secondary_y:
        combined = pd.concat(
            [
                pd.to_numeric(plotted[spec.y], errors="coerce"),
                pd.to_numeric(plotted[spec.secondary_y], errors="coerce"),
            ],
            ignore_index=True,
        ).dropna()
        if not combined.empty:
            minimum = float(combined.min())
            maximum = float(combined.max())
            padding = (maximum - minimum) * 0.08 if maximum != minimum else max(abs(maximum) * 0.08, 1)
            shared_range = [minimum - padding, maximum + padding]
            figure.update_yaxes(range=shared_range, secondary_y=False)
            figure.update_yaxes(range=shared_range, secondary_y=True)
    records = (
        raw_plotted.reset_index().to_dict(orient="records")
        if spec.chart_type == "heatmap"
        else raw_plotted.to_dict(orient="records")
    )
    return figure, ChartResult(spec=spec, data=records, metadata=metadata)


def figure_to_png(figure: go.Figure) -> bytes:
    """Export a Plotly figure to PNG with a clear failure mode."""
    try:
        return figure.to_image(format="png", scale=2)
    except Exception as exc:
        raise ValueError("PNG export requires a working Kaleido/Chrome installation.") from exc


def prepare_figure_for_report(figure: go.Figure) -> go.Figure:
    """Return a light, print-friendly copy of an app chart for reports."""
    report_figure = go.Figure(figure)
    bar_traces = [trace for trace in report_figure.data if trace.type == "bar"]
    bar_index = 0
    line_index = 0
    for trace in report_figure.data:
        if trace.type == "bar":
            color = (
                REPORT_SINGLE_BAR_COLOR
                if len(bar_traces) <= 1
                else REPORT_CATEGORICAL_COLORS[bar_index % len(REPORT_CATEGORICAL_COLORS)]
            )
            trace.marker.color = color
            trace.marker.line.color = "#20365f"
            trace.marker.line.width = 0.6
            bar_index += 1
        elif trace.type == "scatter":
            mode = trace.mode or ""
            if "lines" in mode or getattr(trace, "line", None):
                color = REPORT_LINE_COLORS[line_index % len(REPORT_LINE_COLORS)]
                trace.line.color = color
                trace.line.width = 2.6
                trace.marker.color = color
                trace.marker.line.color = "#ffffff"
                trace.marker.line.width = 0.9
                line_index += 1
    report_figure.update_layout(
        template="simple_white",
        width=1100,
        height=650,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#172033", "size": 12},
        title={"font": {"color": "#172033", "size": 17}, "x": 0.02, "xanchor": "left"},
        legend={
            "font": {"color": "#172033", "size": 10},
            "bgcolor": "rgba(255,255,255,0.88)",
            "bordercolor": "#d6deea",
            "borderwidth": 1,
        },
        margin={"l": 76, "r": 58, "t": 72, "b": 92},
        hoverlabel={"font": {"color": "#172033"}, "bgcolor": "#ffffff"},
    )
    axis_style = {
        "showgrid": True,
        "gridcolor": "#e6edf7",
        "zerolinecolor": "#cfd8e6",
        "linecolor": "#9aa8bb",
        "tickfont": {"color": "#334155", "size": 11},
        "title_font": {"color": "#334155", "size": 13},
    }
    report_figure.update_xaxes(**axis_style)
    report_figure.update_yaxes(**axis_style)
    report_figure.update_traces(selector={"type": "bar"}, textfont_color="#172033")
    report_figure.update_traces(selector={"type": "scatter"}, textfont_color="#172033")
    return report_figure
