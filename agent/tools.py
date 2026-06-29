"""Explicit, validated, non-mutating analytical tools."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd

from pandas.api import types as ptypes

from agent.schemas import (
    AnalyticsFilter,
    CategoricalCountRequest,
    CategoricalCountResult,
    CategoricalCountRow,
    ColumnProfileRequest,
    ColumnProfileResult,
    CommonColumnProfile,
    ToolResult,
)
from services.date_aggregation_service import aggregate_metric_by_period
from services.dataset_profiler import is_likely_id_column, profile_dataset

AGGREGATIONS: dict[str, str] = {
    "sum": "sum",
    "mean": "mean",
    "average": "mean",
    "median": "median",
    "min": "min",
    "minimum": "min",
    "max": "max",
    "maximum": "max",
    "count": "count",
    "nunique": "nunique",
    "unique count": "nunique",
}

KNOWN_COLUMN_MEANINGS: dict[str, str] = {
    "region": "a broad geographic area used to classify records.",
    "country": "the country associated with each record.",
    "itemtype": "a product or item category.",
    "saleschannel": "the channel through which the sale was made.",
    "orderpriority": "the urgency or priority assigned to an order.",
    "orderid": "an identifier used to distinguish orders.",
    "unitssold": "the number of units sold.",
    "unitprice": "the selling price per unit.",
    "unitcost": "the cost per unit.",
    "totalrevenue": "the total revenue associated with each record.",
    "totalcost": "the total cost associated with each record.",
    "totalprofit": "the total profit associated with each record.",
    "date": "the date associated with each record.",
}


def _normalize_name(value: str) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def _display_column_name(value: str) -> str:
    import re

    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(value).replace("_", " "))
    return " ".join(spaced.split()).title()


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _example_values(series: pd.Series, limit: int = 5) -> list[Any]:
    values = []
    for value in series.dropna().unique()[:limit]:
        values.append(_json_scalar(value))
    return values


def _looks_boolean(series: pd.Series) -> bool:
    if ptypes.is_bool_dtype(series.dtype):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return False
    normalized = {str(value).strip().casefold() for value in non_null.unique()}
    boolean_sets = (
        {"true", "false"},
        {"yes", "no"},
        {"y", "n"},
        {"1", "0"},
        {"active", "inactive"},
    )
    return any(normalized and normalized <= allowed for allowed in boolean_sets)


def _datetime_values(series: pd.Series, column_name: str) -> pd.Series | None:
    if ptypes.is_datetime64_any_dtype(series.dtype):
        return pd.to_datetime(series, errors="coerce")
    if ptypes.is_numeric_dtype(series.dtype):
        return None
    non_null = series.dropna()
    if non_null.empty:
        return None
    name_hint = any(
        token in _normalize_name(column_name)
        for token in ("date", "time", "created", "updated")
    )
    converted = pd.to_datetime(series, errors="coerce", format="mixed")
    parse_ratio = converted.notna().sum() / max(len(non_null), 1)
    return (
        converted
        if name_hint and parse_ratio >= 0.6 and converted.notna().sum() >= 2
        else None
    )


def _infer_column_semantic_type(dataframe: pd.DataFrame, column_name: str) -> str:
    series = dataframe[column_name]
    non_null = series.dropna()
    unique_count = int(series.nunique(dropna=True))
    unique_ratio = unique_count / max(int(series.notna().sum()), 1)
    normalized_name = _normalize_name(column_name)
    if _looks_boolean(series):
        return "boolean"
    if _datetime_values(series, column_name) is not None:
        return "datetime"
    if is_likely_id_column(series, column_name, len(dataframe)):
        return "identifier"
    if ptypes.is_numeric_dtype(series.dtype):
        if (
            unique_count <= 10
            and unique_ratio <= 0.1
            and not any(
                token in normalized_name
                for token in (
                    "revenue",
                    "profit",
                    "cost",
                    "price",
                    "sales",
                    "amount",
                    "unit",
                )
            )
        ):
            return "categorical"
        return "numerical"
    if (
        ptypes.is_object_dtype(series.dtype)
        or ptypes.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        text = non_null.astype(str)
        average_length = float(text.str.len().mean()) if not text.empty else 0.0
        if unique_ratio >= 0.8 and average_length >= 40:
            return "free_text"
        if unique_ratio >= 0.98 and any(
            token in normalized_name for token in ("id", "key", "code", "number")
        ):
            return "identifier"
        return "categorical"
    return "unknown"


def _formula_meaning_note(dataframe: pd.DataFrame, column_name: str) -> str | None:
    required = {
        "TotalRevenue": ("UnitsSold", "UnitPrice"),
        "TotalCost": ("UnitsSold", "UnitCost"),
        "TotalProfit": ("TotalRevenue", "TotalCost"),
    }
    if column_name not in required or not all(
        column in dataframe.columns for column in required[column_name]
    ):
        return None
    left = pd.to_numeric(dataframe[column_name], errors="coerce")
    if column_name == "TotalProfit":
        right = pd.to_numeric(
            dataframe["TotalRevenue"], errors="coerce"
        ) - pd.to_numeric(dataframe["TotalCost"], errors="coerce")
        formula = "Total Revenue minus Total Cost"
    else:
        first, second = required[column_name]
        right = pd.to_numeric(dataframe[first], errors="coerce") * pd.to_numeric(
            dataframe[second], errors="coerce"
        )
        formula = f"{_display_column_name(first)} multiplied by {_display_column_name(second)}"
    valid = left.notna() & right.notna()
    if valid.any() and bool(
        np.isclose(left.loc[valid], right.loc[valid], rtol=1e-4, atol=1e-2).mean()
        >= 0.95
    ):
        return f" In this dataset, it appears to equal {formula} for nearly all valid rows."
    return None


def _column_meaning(
    dataframe: pd.DataFrame, column_name: str, semantic_type: str
) -> tuple[str, str]:
    normalized = _normalize_name(column_name)
    if normalized in KNOWN_COLUMN_MEANINGS:
        meaning = KNOWN_COLUMN_MEANINGS[normalized]
        formula = _formula_meaning_note(dataframe, column_name)
        return meaning + (formula or ""), "high"
    display = _display_column_name(column_name)
    generic = {
        "categorical": f"{display} appears to be a categorical field containing labels used to classify records.",
        "numerical": f"{display} appears to be a numerical measure, but its exact business meaning is not defined in the dataset metadata.",
        "datetime": f"{display} appears to store dates or timestamps associated with records.",
        "identifier": f"{display} appears to identify records. Arithmetic operations such as mean or sum are usually not meaningful for identifiers.",
        "boolean": f"{display} appears to store true/false style values.",
        "free_text": f"{display} appears to contain free-text descriptions or notes.",
        "unknown": f"The exact meaning of {display} is not defined in the dataset metadata.",
    }
    return generic[semantic_type], "low"


def _validate_columns(dataframe: pd.DataFrame, *columns: str | None) -> None:
    missing = [
        column for column in columns if column and column not in dataframe.columns
    ]
    if missing:
        raise ValueError(
            f"Column(s) not found: {', '.join(missing)}. "
            f"Available columns: {', '.join(map(str, dataframe.columns))}."
        )


def _aggregation(name: str) -> str:
    normalized = name.lower().strip()
    if normalized not in AGGREGATIONS:
        raise ValueError(f'Unsupported aggregation "{name}".')
    return AGGREGATIONS[normalized]


def _timed(
    tool_name: str, operation: Callable[[], tuple[str, Any, list[str]]]
) -> ToolResult:
    started = perf_counter()
    try:
        summary, data, warnings = operation()
        return ToolResult(
            tool_name=tool_name,
            summary=summary,
            data=data,
            warnings=warnings,
            execution_seconds=perf_counter() - started,
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"{tool_name.replace('_', ' ').title()} could not complete safely."
        ) from exc


def inspect_dataset(dataframe: pd.DataFrame) -> ToolResult:
    return _timed(
        "inspect_dataset",
        lambda: (
            f"The dataset has {len(dataframe):,} rows and {len(dataframe.columns):,} columns.",
            {
                "rows": len(dataframe),
                "columns": len(dataframe.columns),
                "column_names": list(dataframe.columns),
            },
            [],
        ),
    )


def get_column_information(
    dataframe: pd.DataFrame, column: str | None = None
) -> ToolResult:
    def operation():
        profile = profile_dataset(dataframe)
        columns = (
            profile.columns
            if column is None
            else [item for item in profile.columns if item.name == column]
        )
        if column and not columns:
            _validate_columns(dataframe, column)
        return (
            f"Profiled {len(columns)} column(s).",
            [item.model_dump(mode="json") for item in columns],
            [],
        )

    return _timed("get_column_information", operation)


def profile_column(
    dataframe: pd.DataFrame,
    column_name: str,
    include_table: bool = True,
    include_chart: bool = True,
    include_examples: bool = True,
    include_semantic_explanation: bool = True,
    original_query: str = "",
) -> ToolResult:
    """Build a deterministic, type-aware profile for one column."""

    def operation():
        _validate_columns(dataframe, column_name)
        request = ColumnProfileRequest(
            column_name=column_name,
            include_table=include_table,
            include_chart=include_chart,
            include_examples=include_examples,
            include_semantic_explanation=include_semantic_explanation,
            original_query=original_query,
        )
        series = dataframe[column_name]
        row_count = int(len(series))
        non_null_count = int(series.notna().sum())
        missing_count = int(series.isna().sum())
        missing_percentage = missing_count / max(row_count, 1) * 100
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = unique_count / max(non_null_count, 1)
        semantic_type = _infer_column_semantic_type(dataframe, column_name)
        meaning, confidence = _column_meaning(dataframe, column_name, semantic_type)
        display_name = _display_column_name(column_name)
        examples = _example_values(series) if include_examples else []
        warnings: list[str] = []
        table_rows: list[dict[str, Any]] = []
        chart_rows: list[dict[str, Any]] = []
        chart_type: str | None = None
        caution: str | None = None
        recommended_next_step = f"Compare {display_name} with related columns to understand how it affects the dataset."

        profile = CommonColumnProfile(
            column_name=column_name,
            display_name=display_name,
            pandas_dtype=str(series.dtype),
            semantic_type=semantic_type,
            row_count=row_count,
            non_null_count=non_null_count,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            meaning=meaning,
            meaning_confidence=confidence,
            example_values=examples,
        )

        if semantic_type == "categorical":
            values = series.dropna().astype("string")
            counts = (
                values.value_counts(dropna=False)
                .rename_axis(column_name)
                .reset_index(name="Count")
            )
            total = int(counts["Count"].sum())
            counts["Percentage"] = counts["Count"] / max(total, 1) * 100
            table_rows = [
                {
                    column_name: str(row[column_name]),
                    "Count": int(row["Count"]),
                    "Percentage": float(row["Percentage"]),
                }
                for _, row in counts.iterrows()
            ]
            chart_rows = table_rows[:20]
            chart_type = "bar" if include_chart and bool(chart_rows) else None
            if len(table_rows) > len(chart_rows):
                warnings.append(
                    f"Chart shows the top {len(chart_rows)} of {len(table_rows)} {display_name} values."
                )
            profile.top_values = table_rows[:5]
            profile.least_frequent_values = table_rows[-5:]
            caution = "Counts represent dataset rows. If one business entity appears in multiple rows, row counts may differ from unique-entity counts."
            recommended_next_step = f"Compare {display_name} with revenue, profit, counts, or another relevant metric."
        elif semantic_type == "numerical":
            numeric = pd.to_numeric(series, errors="coerce")
            valid = numeric.dropna()
            if valid.empty:
                warnings.append(f"{display_name} has no valid numeric values.")
            else:
                q1 = valid.quantile(0.25)
                q3 = valid.quantile(0.75)
                stats = {
                    "Minimum": valid.min(),
                    "Maximum": valid.max(),
                    "Mean": valid.mean(),
                    "Median": valid.median(),
                    "Standard deviation": valid.std(),
                    "Q1": q1,
                    "Q3": q3,
                    "IQR": q3 - q1,
                    "Zero count": int((valid == 0).sum()),
                    "Negative count": int((valid < 0).sum()),
                }
                table_rows = [
                    {"Statistic": key, "Value": _json_scalar(value)}
                    for key, value in stats.items()
                ]
                chart_rows = [
                    {column_name: _json_scalar(value)} for value in valid.head(5000)
                ]
                chart_type = "histogram" if include_chart and len(valid) >= 2 else None
                profile.minimum = _json_scalar(stats["Minimum"])
                profile.maximum = _json_scalar(stats["Maximum"])
                profile.mean = _json_scalar(stats["Mean"])
                profile.median = _json_scalar(stats["Median"])
                profile.standard_deviation = _json_scalar(stats["Standard deviation"])
                profile.q1 = _json_scalar(stats["Q1"])
                profile.q3 = _json_scalar(stats["Q3"])
                profile.iqr = _json_scalar(stats["IQR"])
                profile.zero_count = int(stats["Zero count"])
                profile.negative_count = int(stats["Negative count"])
                caution = "Numerical distributions can be skewed, so compare the mean with the median before interpreting a typical value."
                recommended_next_step = f"Review the histogram and compare {display_name} by a relevant category or date."
        elif semantic_type == "datetime":
            dates = _datetime_values(series, column_name)
            valid_dates = (
                dates.dropna()
                if dates is not None
                else pd.Series(dtype="datetime64[ns]")
            )
            if valid_dates.empty:
                warnings.append(f"{display_name} has no valid date values.")
            else:
                parsed_non_null_count = int(valid_dates.size)
                parsed_missing_count = row_count - parsed_non_null_count
                parsed_unique_count = int(valid_dates.nunique())
                earliest = valid_dates.min()
                latest = valid_dates.max()
                date_span_days = int((latest - earliest).days)
                distinct_years = int(valid_dates.dt.year.nunique())
                distinct_months = int(valid_dates.dt.to_period("M").nunique())
                table_rows = [
                    {
                        "Statistic": "Earliest date",
                        "Value": earliest.date().isoformat(),
                    },
                    {"Statistic": "Latest date", "Value": latest.date().isoformat()},
                    {"Statistic": "Date span days", "Value": date_span_days},
                    {"Statistic": "Distinct years", "Value": distinct_years},
                    {"Statistic": "Distinct months", "Value": distinct_months},
                ]
                grain = "year" if date_span_days > 730 else "month"
                period = valid_dates.dt.to_period(
                    "Y" if grain == "year" else "M"
                ).dt.to_timestamp()
                counts = (
                    period.value_counts()
                    .sort_index()
                    .rename_axis("Period")
                    .reset_index(name="Count")
                )
                chart_rows = [
                    {
                        "Period": row["Period"].date().isoformat(),
                        "Count": int(row["Count"]),
                    }
                    for _, row in counts.iterrows()
                ]
                chart_type = "bar" if include_chart and bool(chart_rows) else None
                profile.earliest_date = earliest.date().isoformat()
                profile.latest_date = latest.date().isoformat()
                profile.date_span_days = date_span_days
                profile.distinct_years = distinct_years
                profile.distinct_months = distinct_months
                profile.non_null_count = parsed_non_null_count
                profile.missing_count = parsed_missing_count
                profile.missing_percentage = (
                    parsed_missing_count / max(row_count, 1) * 100
                )
                profile.unique_count = parsed_unique_count
                profile.unique_ratio = parsed_unique_count / max(
                    parsed_non_null_count, 1
                )
                recommended_next_step = (
                    f"Use {display_name} for monthly or yearly trend analysis."
                )
        elif semantic_type == "identifier":
            duplicate_count = int(series.dropna().duplicated().sum())
            duplicate_percentage = duplicate_count / max(non_null_count, 1) * 100
            table_rows = [
                {"Statistic": "Unique IDs", "Value": unique_count},
                {"Statistic": "Duplicate IDs", "Value": duplicate_count},
                {"Statistic": "Duplicate percentage", "Value": duplicate_percentage},
                {"Statistic": "Missing IDs", "Value": missing_count},
            ]
            profile.duplicate_count = duplicate_count
            profile.duplicate_percentage = duplicate_percentage
            caution = "Although this column may look numeric, arithmetic statistics such as mean or sum are usually not meaningful for identifiers."
            recommended_next_step = f"Check duplicate {display_name} values to confirm whether they represent repeated lines or data-quality issues."
        elif semantic_type == "boolean":
            normalized = series.dropna().map(
                lambda value: str(value).strip().casefold()
            )
            true_values = {"true", "yes", "y", "1", "active"}
            false_values = {"false", "no", "n", "0", "inactive"}
            true_count = int(normalized.isin(true_values).sum())
            false_count = int(normalized.isin(false_values).sum())
            total = true_count + false_count
            table_rows = [
                {
                    "Value": "True",
                    "Count": true_count,
                    "Percentage": true_count / max(total, 1) * 100,
                },
                {
                    "Value": "False",
                    "Count": false_count,
                    "Percentage": false_count / max(total, 1) * 100,
                },
            ]
            chart_rows = table_rows
            chart_type = "bar" if include_chart else None
            profile.true_count = true_count
            profile.false_count = false_count
            profile.true_percentage = true_count / max(total, 1) * 100
            profile.false_percentage = false_count / max(total, 1) * 100
            recommended_next_step = f"Compare the true/false split in {display_name} across important groups."
        elif semantic_type == "free_text":
            text = series.dropna().astype(str)
            lengths = text.str.len()
            average_length = float(lengths.mean()) if not lengths.empty else None
            minimum_length = int(lengths.min()) if not lengths.empty else None
            maximum_length = int(lengths.max()) if not lengths.empty else None
            table_rows = [
                {
                    "Statistic": "Non-empty values",
                    "Value": int(text.str.strip().ne("").sum()),
                },
                {"Statistic": "Unique values", "Value": unique_count},
                {"Statistic": "Average length", "Value": average_length},
                {"Statistic": "Minimum length", "Value": minimum_length},
                {"Statistic": "Maximum length", "Value": maximum_length},
            ]
            profile.average_length = average_length
            profile.minimum_length = minimum_length
            profile.maximum_length = maximum_length
            caution = "Only sample values and length statistics are shown for free-text columns."
            recommended_next_step = f"Inspect sample {display_name} values or search for common text patterns."
        else:
            table_rows = [
                {"Statistic": "Rows", "Value": row_count},
                {"Statistic": "Non-null values", "Value": non_null_count},
                {"Statistic": "Unique values", "Value": unique_count},
            ]
            caution = "The system could not confidently infer a semantic type for this column."

        profile.warnings = warnings
        summary = (
            f"{display_name} is a {semantic_type.replace('_', ' ')} column "
            f"with {profile.row_count:,} row(s), {profile.non_null_count:,} non-null value(s), "
            f"{profile.missing_count:,} missing value(s), and {profile.unique_count:,} unique value(s)."
        )
        result = ColumnProfileResult(
            request=request,
            profile=profile,
            summary=summary,
            table_columns=list(table_rows[0].keys()) if table_rows else [],
            table_rows=table_rows if include_table else [],
            chart_type=chart_type,
            chart_rows=chart_rows if include_chart else [],
            caution=caution,
            recommended_next_step=recommended_next_step,
            warnings=warnings,
        )
        return summary, result.model_dump(mode="json"), warnings

    return _timed("profile_column", operation)


def calculate_summary_statistics(
    dataframe: pd.DataFrame, columns: list[str] | None = None
) -> ToolResult:
    def operation():
        selected = columns or list(dataframe.select_dtypes(include="number").columns)
        if not selected:
            raise ValueError("No numeric columns are available for summary statistics.")
        _validate_columns(dataframe, *selected)
        if any(
            not pd.api.types.is_numeric_dtype(dataframe[column]) for column in selected
        ):
            raise ValueError("Summary statistics require numeric columns.")
        result = dataframe[selected].describe().T.reset_index(names="column")
        return (
            "Calculated numeric summary statistics.",
            result.to_dict(orient="records"),
            [],
        )

    return _timed("calculate_summary_statistics", operation)


def group_and_aggregate(
    dataframe: pd.DataFrame,
    group_by: str,
    value_column: str | None = None,
    aggregation: str = "sum",
    secondary_group_by: str | None = None,
    value_columns: list[str] | None = None,
    limit: int | None = None,
    filter_column: str | None = None,
    filter_value: Any | None = None,
    date_column: str | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
    sort_descending: bool = True,
    include_percentage: bool = False,
    focus_column: str | None = None,
    focus_value: Any | None = None,
) -> ToolResult:
    def operation():
        selected_values = list(
            dict.fromkeys(value_columns or ([value_column] if value_column else []))
        )
        if not selected_values:
            raise ValueError("At least one value column is required.")
        agg = _aggregation(aggregation)
        count_rows = (
            agg == "count"
            and selected_values == ["Count"]
            and "Count" not in dataframe.columns
        )
        validation_values = [] if count_rows else selected_values
        _validate_columns(
            dataframe,
            group_by,
            secondary_group_by,
            *validation_values,
            filter_column,
            date_column,
        )
        if agg not in {"count", "nunique"} and any(
            not pd.api.types.is_numeric_dtype(dataframe[column])
            for column in selected_values
        ):
            raise ValueError(
                "The selected aggregation requires a numeric value column."
            )
        working = dataframe
        filter_description = ""
        if filter_column is not None:
            series = dataframe[filter_column]
            filter_values = (
                filter_value if isinstance(filter_value, list) else [filter_value]
            )
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(
                series
            ):
                normalized_values = {str(value).casefold() for value in filter_values}
                mask = series.astype("string").str.casefold().isin(normalized_values)
            else:
                mask = series.isin(filter_values)
            working = dataframe.loc[mask]
            if working.empty:
                raise ValueError(f'No rows matched {filter_column} = "{filter_value}".')
            filter_description = f' where {filter_column} is "{filter_value}"'
        if date_column is not None and (start_date is not None or end_date is not None):
            dates = pd.to_datetime(working[date_column], errors="coerce")
            working = working.loc[dates.notna()]
            dates = dates.loc[dates.notna()]
            if start_date is not None:
                start = pd.to_datetime(start_date, errors="coerce")
                if pd.notna(start):
                    mask = dates >= start
                    working = working.loc[mask]
                    dates = dates.loc[mask]
            if end_date is not None:
                end = pd.to_datetime(end_date, errors="coerce")
                if pd.notna(end):
                    if end == end.normalize():
                        mask = dates < end + pd.Timedelta(days=1)
                    else:
                        mask = dates <= end
                    working = working.loc[mask]
                    dates = dates.loc[mask]
            if working.empty:
                raise ValueError("No rows matched the selected date range.")
            start_label = (
                pd.to_datetime(start_date).date() if start_date is not None else "start"
            )
            end_label = (
                pd.to_datetime(end_date).date() if end_date is not None else "end"
            )
            filter_description += f" from {start_label} to {end_label}"
        group_columns = [group_by]
        if secondary_group_by and secondary_group_by != group_by:
            group_columns.append(secondary_group_by)
        if count_rows:
            result = (
                working.groupby(group_columns, dropna=False)
                .size()
                .reset_index(name="Count")
                .sort_values("Count", ascending=not sort_descending)
            )
        else:
            result = (
                working.groupby(group_columns, dropna=False)[selected_values]
                .agg(agg)
                .reset_index()
                .sort_values(selected_values[0], ascending=not sort_descending)
            )
        if include_percentage:
            percentage_basis = "Count" if count_rows else selected_values[0]
            denominator = float(pd.to_numeric(result[percentage_basis], errors="coerce").sum())
            result["PercentageOfTotal"] = (
                pd.to_numeric(result[percentage_basis], errors="coerce") / denominator * 100
                if denominator
                else 0.0
            )
        if limit is not None:
            if limit < 1 or limit > 1000:
                raise ValueError("Limit must be between 1 and 1000.")
            result = result.head(limit)
        percentage_text = " with percentage of total" if include_percentage else ""
        return (
            f"Calculated {aggregation} of {', '.join(selected_values)} by "
            f"{' and '.join(group_columns)}{filter_description}{percentage_text}.",
            result.to_dict(orient="records"),
            [],
        )

    return _timed("group_and_aggregate", operation)


def calculate_grouped_extrema(
    dataframe: pd.DataFrame,
    primary_group_column: str,
    secondary_group_column: str,
    metric_column: str,
    aggregation: str = "sum",
    extremum: str = "max",
    filter_column: str | None = None,
    filter_values: list[Any] | None = None,
) -> ToolResult:
    """Find the winning secondary category inside each primary category."""

    def operation():
        _validate_columns(
            dataframe,
            primary_group_column,
            secondary_group_column,
            metric_column,
            filter_column,
        )
        agg = _aggregation(aggregation)
        if agg not in {"sum", "mean", "median", "min", "max", "count", "nunique"}:
            raise ValueError(f'Unsupported aggregation "{aggregation}".')
        if agg not in {"count", "nunique"} and not pd.api.types.is_numeric_dtype(
            dataframe[metric_column]
        ):
            raise ValueError("Grouped extrema requires a numeric metric column.")
        normalized_extremum = extremum.lower().strip()
        if normalized_extremum not in {"max", "min"}:
            raise ValueError('Extremum must be "max" or "min".')

        columns = [primary_group_column, secondary_group_column, metric_column]
        if filter_column and filter_column not in columns:
            columns.append(filter_column)
        working = dataframe[columns].copy()
        if filter_column and filter_values:
            series = working[filter_column]
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(
                series
            ):
                normalized_values = {str(value).casefold() for value in filter_values}
                mask = series.astype("string").str.casefold().isin(normalized_values)
            else:
                mask = series.isin(filter_values)
            working = working.loc[mask]
            if working.empty:
                raise ValueError(
                    f"No rows matched {filter_column} in {', '.join(map(str, filter_values))}."
                )
        if agg not in {"count", "nunique"}:
            working[metric_column] = pd.to_numeric(
                working[metric_column], errors="coerce"
            )
            invalid_metric_count = int(working[metric_column].isna().sum())
            working = working.dropna(
                subset=[primary_group_column, secondary_group_column, metric_column]
            )
        else:
            invalid_metric_count = 0
            working = working.dropna(
                subset=[primary_group_column, secondary_group_column]
            )
        if working.empty:
            raise ValueError(
                "No rows were available after removing missing group or metric values."
            )

        aggregated = (
            working.groupby(
                [primary_group_column, secondary_group_column], dropna=False
            )[metric_column]
            .agg(agg)
            .reset_index()
        )
        if aggregated.empty:
            raise ValueError("No grouped values were available.")
        ascending = normalized_extremum == "min"
        aggregated["Rank"] = (
            aggregated.groupby(primary_group_column)[metric_column]
            .rank(method="dense", ascending=ascending)
            .astype(int)
        )
        group_target = aggregated.groupby(primary_group_column)[
            metric_column
        ].transform(normalized_extremum)
        winners = aggregated.loc[
            np.isclose(
                aggregated[metric_column].astype(float),
                group_target.astype(float),
                rtol=1e-9,
                atol=1e-9,
            )
        ].copy()
        winner_counts = winners.groupby(primary_group_column)[
            secondary_group_column
        ].transform("count")
        winners["Tie"] = winner_counts > 1
        winners["TieCount"] = winner_counts.astype(int)

        regional_totals = aggregated.groupby(primary_group_column)[
            metric_column
        ].transform("sum")
        aggregated["_ShareOfPrimary"] = np.where(
            regional_totals > 0,
            aggregated[metric_column] / regional_totals * 100,
            np.nan,
        )
        second_place = aggregated.loc[aggregated["Rank"] == 2].copy()
        second_lookup = {
            row[primary_group_column]: row
            for _, row in second_place.sort_values(
                [primary_group_column, metric_column],
                ascending=[True, ascending],
            ).iterrows()
        }
        share_lookup = {
            (row[primary_group_column], row[secondary_group_column]): row[
                "_ShareOfPrimary"
            ]
            for _, row in aggregated.iterrows()
        }

        rows = []
        chart_rows = []
        winners = winners.sort_values(
            [primary_group_column, secondary_group_column]
        ).reset_index(drop=True)
        for _, row in winners.iterrows():
            primary = row[primary_group_column]
            secondary = row[secondary_group_column]
            value = float(row[metric_column])
            second = second_lookup.get(primary)
            second_value = float(second[metric_column]) if second is not None else None
            absolute_gap = (
                abs(value - second_value) if second_value is not None else None
            )
            percentage_gap = (
                absolute_gap / abs(second_value) * 100
                if second_value not in (None, 0)
                else None
            )
            share = share_lookup.get((primary, secondary))
            table_row = {
                primary_group_column: primary,
                secondary_group_column: secondary,
                metric_column: value,
                "Rank": int(row["Rank"]),
                "Tie": bool(row["Tie"]),
                "TieCount": int(row["TieCount"]),
                "SecondPlace": (
                    second[secondary_group_column] if second is not None else None
                ),
                "SecondPlaceValue": second_value,
                "AbsoluteGap": absolute_gap,
                "PercentageGap": percentage_gap,
                "WinnerShareOfGroup": float(share) if pd.notna(share) else None,
            }
            rows.append(table_row)
            chart_rows.append(
                {
                    primary_group_column: primary,
                    secondary_group_column: secondary,
                    metric_column: value,
                    "Tie": bool(row["Tie"]),
                }
            )

        warnings = []
        if invalid_metric_count:
            warnings.append(
                f"Excluded {invalid_metric_count:,} row(s) with missing or invalid {metric_column}."
            )
        if agg == "sum" and (working[metric_column] < 0).any():
            warnings.append(
                "Negative metric values were included; winner share may be harder to interpret."
            )
        summary = (
            f"Found {normalized_extremum} {aggregation} {metric_column} "
            f"{secondary_group_column} within each {primary_group_column}."
        )
        return summary, rows, warnings

    return _timed("calculate_grouped_extrema", operation)


def analyze_advanced_request(
    dataframe: pd.DataFrame,
    operation: str,
    group_by: str | list[str] | None = None,
    metric_column: str | None = None,
    aggregation: str = "sum",
    direction: str = "highest",
    limit: int | None = None,
    metrics: list[dict[str, Any]] | None = None,
    filters: list[dict[str, Any]] | None = None,
    numerator_column: str | None = None,
    denominator_column: str | None = None,
    denominator_aggregation: str | None = None,
    result_column: str | None = None,
    entity_id_column: str | None = None,
    entity_label_column: str | None = None,
    rank_by: str | None = None,
) -> ToolResult:
    """Execute schema-resolved advanced analytics without arbitrary expressions."""

    def operation_fn():
        normalized_operation = operation.strip().lower()
        warnings: list[str] = []

        def groups() -> list[str]:
            if group_by is None:
                return []
            return [group_by] if isinstance(group_by, str) else list(group_by)

        def apply_filters(frame: pd.DataFrame) -> pd.DataFrame:
            working = frame
            for item in filters or []:
                column = item.get("column")
                operator = item.get("operator", "equals")
                value = item.get("value")
                _validate_columns(working, column)
                series = working[column]
                if operator == "equals":
                    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                        mask = series.astype("string").str.casefold() == str(value).casefold()
                    else:
                        mask = series == value
                elif operator == "is_negative":
                    mask = pd.to_numeric(series, errors="coerce") < 0
                elif operator == "is_positive":
                    mask = pd.to_numeric(series, errors="coerce") > 0
                elif operator == "less_than":
                    mask = pd.to_numeric(series, errors="coerce") < float(value)
                elif operator == "greater_than":
                    mask = pd.to_numeric(series, errors="coerce") > float(value)
                else:
                    raise ValueError(f'Unsupported filter operator "{operator}".')
                working = working.loc[mask.fillna(False)]
            return working

        def aggregate(frame: pd.DataFrame, group_columns: list[str], metric: str, agg: str, name: str) -> pd.DataFrame:
            _validate_columns(frame, *group_columns, metric)
            resolved_agg = _aggregation(agg)
            if resolved_agg not in {"count", "nunique"} and not pd.api.types.is_numeric_dtype(frame[metric]):
                raise ValueError(f"{agg} requires a numeric metric column.")
            if group_columns:
                return (
                    frame.groupby(group_columns, dropna=False)[metric]
                    .agg(resolved_agg)
                    .reset_index(name=name)
                )
            return pd.DataFrame([{name: frame[metric].agg(resolved_agg)}])

        working = apply_filters(dataframe)

        if normalized_operation == "ranking":
            group_columns = groups()
            if not group_columns or not metric_column:
                raise ValueError("Ranking requires a group and metric.")
            value_name = result_column or f"{aggregation.title()} {metric_column}"
            result = aggregate(working, group_columns, metric_column, aggregation, value_name)
            ascending = direction in {"lowest", "bottom", "min", "minimum"}
            result = result.sort_values(value_name, ascending=ascending, kind="mergesort")
            if limit:
                result = result.head(limit)
            return (
                f"Ranked {' and '.join(group_columns)} by {aggregation} {metric_column} ({direction}).",
                result.to_dict(orient="records"),
                warnings,
            )

        if normalized_operation == "grouped_aggregation":
            group_columns = groups()
            if not group_columns:
                raise ValueError("Grouped aggregation requires at least one group.")
            metric_specs = metrics or [{"column": metric_column, "aggregation": aggregation}]
            result: pd.DataFrame | None = None
            for spec in metric_specs:
                column = spec["column"]
                agg = spec.get("aggregation", aggregation)
                name = spec.get("alias") or f"{agg.title()} {column}"
                part = aggregate(working, group_columns, column, agg, name)
                result = part if result is None else result.merge(part, on=group_columns, how="outer")
            assert result is not None
            if rank_by and rank_by in result.columns:
                result = result.sort_values(rank_by, ascending=False)
                if limit:
                    result = result.head(limit)
            return (
                f"Calculated grouped metrics by {' and '.join(group_columns)}.",
                result.to_dict(orient="records"),
                warnings,
            )

        if normalized_operation == "multi_metric_extrema":
            group_columns = groups()
            if not group_columns or not metrics:
                raise ValueError("Multi-metric extrema requires a group and metrics.")
            table: pd.DataFrame | None = None
            winners = []
            for spec in metrics:
                column = spec["column"]
                agg = spec.get("aggregation", "sum")
                metric_direction = spec.get("direction", "highest")
                name = spec.get("alias") or f"{agg.title()} {column}"
                part = aggregate(working, group_columns, column, agg, name)
                table = part if table is None else table.merge(part, on=group_columns, how="outer")
                ascending = metric_direction in {"lowest", "bottom", "min", "minimum"}
                target = part.sort_values(name, ascending=ascending, kind="mergesort").iloc[0]
                winners.append({
                    "Metric": column,
                    "Aggregation": agg,
                    "Objective": metric_direction,
                    **{group_columns[0]: target[group_columns[0]]},
                    "Value": float(target[name]),
                })
            assert table is not None
            return (
                "Calculated independent extrema for each requested metric.",
                {"winners": winners, "table_rows": table.to_dict(orient="records")},
                warnings,
            )

        if normalized_operation == "share_unique":
            group_columns = groups()
            if not group_columns or not metric_column:
                raise ValueError("Share of unique entities requires a group and identifier.")
            count_name = result_column or f"Unique {metric_column} Count"
            result = aggregate(working, group_columns, metric_column, "nunique", count_name)
            denominator = float(result[count_name].sum())
            result["PercentageOfTotal"] = result[count_name] / denominator * 100 if denominator else 0.0
            if len(group_columns) == 1:
                entity_groups = working.groupby(metric_column)[group_columns[0]].nunique(dropna=True)
                if (entity_groups > 1).any():
                    warnings.append(
                        f"Some {metric_column} values occur in multiple {group_columns[0]} values, so shares can overlap."
                    )
            return (
                f"Calculated each {' and '.join(group_columns)} share of unique {metric_column}.",
                result.sort_values("PercentageOfTotal", ascending=False).to_dict(orient="records"),
                warnings,
            )

        if normalized_operation == "negative_groups":
            group_columns = groups()
            if not group_columns or not metric_column:
                raise ValueError("Negative group analysis requires a group and metric.")
            value_name = result_column or f"Total {metric_column}"
            result = aggregate(working, group_columns, metric_column, aggregation, value_name)
            result = result.loc[pd.to_numeric(result[value_name], errors="coerce") < 0]
            result = result.sort_values(value_name, ascending=True, kind="mergesort")
            return (
                f"Found {len(result)} group(s) with negative {aggregation} {metric_column}.",
                result.to_dict(orient="records"),
                warnings,
            )

        if normalized_operation == "negative_record_percentage":
            if not metric_column:
                raise ValueError("Negative record percentage requires a metric.")
            _validate_columns(working, metric_column)
            values = pd.to_numeric(working[metric_column], errors="coerce")
            valid_count = int(values.notna().sum())
            negative_count = int((values < 0).sum())
            percentage = negative_count / valid_count * 100 if valid_count else 0.0
            return (
                f"Calculated percentage of valid records where {metric_column} is below zero.",
                {
                    "metric_column": metric_column,
                    "negative_count": negative_count,
                    "valid_count": valid_count,
                    "percentage": percentage,
                },
                warnings,
            )

        if normalized_operation == "loss_by_group":
            group_columns = groups()
            if not group_columns or not metric_column:
                raise ValueError("Loss analysis requires a group and metric.")
            _validate_columns(working, *group_columns, metric_column)
            negative = working.loc[pd.to_numeric(working[metric_column], errors="coerce") < 0].copy()
            if negative.empty:
                result = pd.DataFrame(columns=[*group_columns, "Loss-making records", "Total loss"])
            else:
                result = (
                    negative.groupby(group_columns, dropna=False)
                    .agg(**{
                        "Loss-making records": (metric_column, "count"),
                        "Total loss": (metric_column, "sum"),
                    })
                    .reset_index()
                    .sort_values("Total loss", ascending=True)
                )
                if "Order ID" in negative.columns:
                    unique_orders = (
                        negative.groupby(group_columns, dropna=False)["Order ID"]
                        .nunique()
                        .reset_index(name="Unique loss-making orders")
                    )
                    result = result.merge(unique_orders, on=group_columns, how="left")
            return (
                f"Grouped records where {metric_column} is below zero.",
                result.to_dict(orient="records"),
                warnings,
            )

        if normalized_operation == "derived_ratio":
            group_columns = groups()
            if not group_columns or not numerator_column:
                raise ValueError("Derived ratio requires a group and numerator.")
            if denominator_column is None:
                raise ValueError("Derived ratio requires a denominator.")
            _validate_columns(working, *group_columns, numerator_column, denominator_column)
            numerator = aggregate(working, group_columns, numerator_column, "sum", "_numerator")
            denom_agg = denominator_aggregation or "sum"
            denominator = aggregate(working, group_columns, denominator_column, denom_agg, "_denominator")
            result = numerator.merge(denominator, on=group_columns, how="outer")
            alias = result_column or f"{numerator_column} per {denominator_column}"
            denominator_values = pd.to_numeric(result["_denominator"], errors="coerce")
            result[alias] = np.where(denominator_values != 0, result["_numerator"] / denominator_values, np.nan)
            result = result.drop(columns=["_numerator", "_denominator"])
            if rank_by == alias:
                result = result.sort_values(alias, ascending=direction not in {"highest", "top", "max"}, kind="mergesort")
                if limit:
                    result = result.head(limit)
            return (
                f"Calculated {alias} as sum({numerator_column}) divided by {denom_agg}({denominator_column}).",
                result.to_dict(orient="records"),
                warnings,
            )

        if normalized_operation == "derived_difference":
            group_columns = groups()
            if not group_columns or not numerator_column or not denominator_column:
                raise ValueError("Derived difference requires a group, minuend, and subtrahend.")
            _validate_columns(working, *group_columns, numerator_column, denominator_column)
            numerator = aggregate(working, group_columns, numerator_column, "sum", "_numerator")
            denominator = aggregate(working, group_columns, denominator_column, "sum", "_denominator")
            result = numerator.merge(denominator, on=group_columns, how="outer")
            alias = result_column or f"{numerator_column} minus {denominator_column}"
            result[alias] = (
                pd.to_numeric(result["_numerator"], errors="coerce")
                - pd.to_numeric(result["_denominator"], errors="coerce")
            )
            result = result.drop(columns=["_numerator", "_denominator"])
            if rank_by == alias:
                result = result.sort_values(alias, ascending=direction not in {"highest", "top", "max"}, kind="mergesort")
                if limit:
                    result = result.head(limit)
            return (
                f"Calculated {alias} as sum({numerator_column}) minus sum({denominator_column}).",
                result.to_dict(orient="records"),
                warnings,
            )

        if normalized_operation == "distribution":
            if not metric_column:
                raise ValueError("Distribution analysis requires a metric.")
            _validate_columns(working, metric_column)
            values = pd.to_numeric(working[metric_column], errors="coerce")
            data = {
                "metric_column": metric_column,
                "count": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "min": float(values.min()) if values.notna().any() else None,
                "q1": float(values.quantile(0.25)) if values.notna().any() else None,
                "mean": float(values.mean()) if values.notna().any() else None,
                "median": float(values.median()) if values.notna().any() else None,
                "q3": float(values.quantile(0.75)) if values.notna().any() else None,
                "max": float(values.max()) if values.notna().any() else None,
            }
            return f"Calculated distribution statistics for {metric_column}.", data, warnings

        if normalized_operation == "relationship":
            metric_specs = metrics or []
            if len(metric_specs) < 2:
                raise ValueError("Relationship analysis requires two numeric metrics.")
            first, second = metric_specs[0]["column"], metric_specs[1]["column"]
            _validate_columns(working, first, second)
            subset = working[[first, second]].apply(pd.to_numeric, errors="coerce").dropna()
            if subset.empty:
                raise ValueError("No paired numeric values were available.")
            data = {
                "first_column": first,
                "second_column": second,
                "pearson": float(subset[first].corr(subset[second], method="pearson")),
                "spearman": float(subset[first].corr(subset[second], method="spearman")),
                "rows": subset.to_dict(orient="records"),
            }
            return f"Calculated association between {first} and {second}.", data, warnings

        raise ValueError(f'Unsupported advanced operation "{operation}".')

    return _timed("analyze_advanced_request", operation_fn)


def compare_grouped_to_benchmark(
    dataframe: pd.DataFrame,
    category_column: str,
    value_column: str,
    aggregation: str = "sum",
    benchmark: str = "mean",
    comparison: str = "below",
    benchmark_group_by: str | None = None,
    filter_column: str | None = None,
    filter_value: Any | None = None,
) -> ToolResult:
    """Compare category aggregates with a global or parent-group benchmark."""

    def operation():
        _validate_columns(
            dataframe,
            category_column,
            value_column,
            benchmark_group_by,
            filter_column,
        )
        if benchmark_group_by == category_column:
            raise ValueError("The category and benchmark group must be different.")
        if not pd.api.types.is_numeric_dtype(dataframe[value_column]):
            raise ValueError("Benchmark comparisons require a numeric value column.")

        agg = _aggregation(aggregation)
        benchmark_agg = _aggregation(benchmark)
        if benchmark_agg not in {"mean", "median"}:
            raise ValueError("The benchmark must be mean or median.")
        normalized_comparison = comparison.lower().strip()
        if normalized_comparison not in {"below", "above"}:
            raise ValueError('Comparison must be "below" or "above".')

        working = dataframe
        filter_description = ""
        if filter_column:
            series = dataframe[filter_column]
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(
                series
            ):
                mask = (
                    series.astype("string").str.casefold()
                    == str(filter_value).casefold()
                )
            else:
                mask = series == filter_value
            working = dataframe.loc[mask]
            if working.empty:
                raise ValueError(f'No rows matched {filter_column} = "{filter_value}".')
            filter_description = f' where {filter_column} is "{filter_value}"'

        group_columns = [
            column for column in (benchmark_group_by, category_column) if column
        ]
        grouped = (
            working.groupby(group_columns, dropna=False)[value_column]
            .agg(agg)
            .reset_index()
        )
        if benchmark_group_by:
            grouped["Benchmark"] = grouped.groupby(benchmark_group_by, dropna=False)[
                value_column
            ].transform(benchmark_agg)
        else:
            grouped["Benchmark"] = float(grouped[value_column].agg(benchmark_agg))
        grouped["DifferenceFromBenchmark"] = (
            grouped[value_column] - grouped["Benchmark"]
        )
        mask = (
            grouped[value_column] < grouped["Benchmark"]
            if normalized_comparison == "below"
            else grouped[value_column] > grouped["Benchmark"]
        )
        matches = grouped.loc[mask].sort_values(
            ["DifferenceFromBenchmark", value_column],
            ascending=normalized_comparison == "below",
        )
        scope = (
            f" within each {benchmark_group_by}"
            if benchmark_group_by
            else " across all categories"
        )
        return (
            f"Found {len(matches)} {category_column} value(s) {normalized_comparison} "
            f"the {benchmark_agg} of their {agg} {value_column}{scope}"
            f"{filter_description}.",
            matches.to_dict(orient="records"),
            [],
        )

    return _timed("compare_grouped_to_benchmark", operation)


def compare_category_values(
    dataframe: pd.DataFrame,
    category_column: str,
    value_column: str,
    first_value: Any,
    second_value: Any,
    aggregation: str = "sum",
) -> ToolResult:
    """Compare one metric between two values from the same category column."""

    def operation():
        _validate_columns(dataframe, category_column, value_column)
        agg = _aggregation(aggregation)
        if not pd.api.types.is_numeric_dtype(dataframe[value_column]):
            raise ValueError("Category comparisons require a numeric value column.")
        categories = dataframe[category_column].astype("string")

        def aggregate_value(requested: Any) -> float:
            mask = categories.str.casefold() == str(requested).casefold()
            if not mask.any():
                raise ValueError(f'No rows matched {category_column} = "{requested}".')
            return float(dataframe.loc[mask, value_column].agg(agg))

        first_total = aggregate_value(first_value)
        second_total = aggregate_value(second_value)
        signed_difference = first_total - second_total
        absolute_difference = abs(signed_difference)
        percentage_difference = (
            absolute_difference / abs(second_total) * 100 if second_total != 0 else None
        )
        higher_value = (
            first_value
            if signed_difference > 0
            else second_value if signed_difference < 0 else None
        )
        return (
            f"Compared {aggregation} of {value_column} between "
            f"{first_value} and {second_value}.",
            {
                "category_column": category_column,
                "value_column": value_column,
                "aggregation": aggregation,
                "first_value": first_value,
                "first_total": first_total,
                "second_value": second_value,
                "second_total": second_total,
                "signed_difference": signed_difference,
                "absolute_difference": absolute_difference,
                "percentage_difference": percentage_difference,
                "higher_value": higher_value,
            },
            [],
        )

    return _timed("compare_category_values", operation)


def calculate_filtered_aggregate(
    dataframe: pd.DataFrame,
    category_column: str,
    category_value: Any,
    value_column: str | None = None,
    value_columns: list[str] | None = None,
    aggregation: str = "sum",
    filters: list[dict[str, Any]] | None = None,
) -> ToolResult:
    """Calculate scalar metric(s) for one explicit category value."""

    def operation():
        selected_values = list(
            dict.fromkeys(value_columns or ([value_column] if value_column else []))
        )
        if not selected_values:
            raise ValueError("At least one value column is required.")
        parsed_filters = [
            AnalyticsFilter.model_validate(item) for item in (filters or [])
        ]
        if not parsed_filters:
            parsed_filters = [
                AnalyticsFilter(column=category_column, operator="equals", value=category_value)
            ]
        _validate_columns(
            dataframe,
            category_column,
            *selected_values,
            *(item.column for item in parsed_filters),
        )
        agg = _aggregation(aggregation)
        if any(
            not pd.api.types.is_numeric_dtype(dataframe[column])
            for column in selected_values
        ):
            raise ValueError("The selected aggregation requires numeric value columns.")
        mask = pd.Series(True, index=dataframe.index)
        for item in parsed_filters:
            series = dataframe[item.column]
            if item.operator != "equals":
                raise ValueError("Filtered aggregate currently supports equality filters.")
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                item_mask = series.astype("string").str.casefold() == str(item.value).casefold()
            else:
                item_mask = series == item.value
            mask &= item_mask.fillna(False)
        if not mask.any():
            filter_text = ", ".join(
                f'{item.column} = "{item.value}"' for item in parsed_filters
            )
            raise ValueError(f"No rows matched {filter_text}.")
        records = []
        for column in selected_values:
            value = dataframe.loc[mask, column].agg(agg)
            if pd.isna(value):
                continue
            records.append(
                {
                    "value_column": column,
                    "aggregation": aggregation,
                    "value": float(value),
                }
            )
        if not records:
            raise ValueError(
                "No selected columns had values available for aggregation."
            )
        first = records[0]
        data = {
            "category_column": category_column,
            "category_value": category_value,
            "filters": [
                {"column": item.column, "operator": item.operator, "value": item.value}
                for item in parsed_filters
            ],
            "value_column": first["value_column"],
            "aggregation": aggregation,
            "value": first["value"],
        }
        if len(records) > 1:
            data["value_columns"] = selected_values
            data["values"] = records
        filter_description = ", ".join(
            f"{item.column} = {item.value}" for item in parsed_filters
        )
        return (
            f"Calculated {aggregation} of {', '.join(selected_values)} for "
            f"{filter_description}.",
            data,
            [],
        )

    return _timed("calculate_filtered_aggregate", operation)


def calculate_scalar_aggregate(
    dataframe: pd.DataFrame,
    value_column: str,
    aggregation: str = "sum",
) -> ToolResult:
    """Calculate one dataset-wide scalar aggregate."""

    def operation():
        _validate_columns(dataframe, value_column)
        agg = _aggregation(aggregation)
        if not pd.api.types.is_numeric_dtype(dataframe[value_column]):
            raise ValueError(
                "The selected aggregation requires a numeric value column."
            )
        value = dataframe[value_column].agg(agg)
        if pd.isna(value):
            raise ValueError(
                f'"{value_column}" has no values available for aggregation.'
            )
        return (
            f"Calculated {aggregation} of {value_column}.",
            {
                "value_column": value_column,
                "aggregation": aggregation,
                "value": float(value),
            },
            [],
        )

    return _timed("calculate_scalar_aggregate", operation)


def calculate_multi_scalar_aggregate(
    dataframe: pd.DataFrame,
    value_columns: list[str],
    aggregation: str = "sum",
) -> ToolResult:
    """Calculate one scalar aggregate for each selected numeric column."""

    def operation():
        selected = list(dict.fromkeys(value_columns or []))
        if not selected:
            raise ValueError("At least one value column is required.")
        _validate_columns(dataframe, *selected)
        agg = _aggregation(aggregation)
        if any(
            not pd.api.types.is_numeric_dtype(dataframe[column]) for column in selected
        ):
            raise ValueError("The selected aggregation requires numeric value columns.")
        records = []
        for column in selected:
            value = dataframe[column].agg(agg)
            if pd.isna(value):
                continue
            records.append(
                {
                    "value_column": column,
                    "aggregation": aggregation,
                    "value": float(value),
                }
            )
        if not records:
            raise ValueError(
                "No selected columns had values available for aggregation."
            )
        return (
            f"Calculated {aggregation} for {len(records)} numeric column(s).",
            records,
            [],
        )

    return _timed("calculate_multi_scalar_aggregate", operation)


def list_distinct_values(
    dataframe: pd.DataFrame,
    target_column: str,
    filter_column: str | None = None,
    filter_value: Any | None = None,
) -> ToolResult:
    """List distinct target values, optionally filtered by another column."""

    def operation():
        _validate_columns(dataframe, target_column, filter_column)
        working = dataframe
        filter_description = ""
        if filter_column is not None:
            series = dataframe[filter_column]
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(
                series
            ):
                mask = (
                    series.astype("string").str.casefold()
                    == str(filter_value).casefold()
                )
            else:
                mask = series == filter_value
            working = dataframe.loc[mask]
            if working.empty:
                raise ValueError(f'No rows matched {filter_column} = "{filter_value}".')
            filter_description = f' where {filter_column} is "{filter_value}"'
        values = sorted(
            working[target_column].dropna().unique().tolist(),
            key=lambda value: str(value).casefold(),
        )
        return (
            f"Found {len(values)} distinct {target_column} value(s)"
            f"{filter_description}.",
            {
                "target_column": target_column,
                "filter_column": filter_column,
                "filter_value": filter_value,
                "count": len(values),
                "values": values,
            },
            [],
        )

    return _timed("list_distinct_values", operation)


def count_distinct_values(dataframe: pd.DataFrame, column: str) -> ToolResult:
    """Count distinct non-null values in one column."""

    def operation():
        _validate_columns(dataframe, column)
        count = int(dataframe[column].nunique(dropna=True))
        return (
            f"Counted {count} distinct value(s) in {column}.",
            {"column": column, "count": count},
            [],
        )

    return _timed("count_distinct_values", operation)


def analyze_high_volume_low_outcome(
    dataframe: pd.DataFrame,
    category_column: str,
    volume_column: str,
    outcome_column: str,
    aggregation: str = "sum",
) -> ToolResult:
    """Find categories with high volume and low outcome using median thresholds."""

    def operation():
        _validate_columns(dataframe, category_column, volume_column, outcome_column)
        if any(
            not pd.api.types.is_numeric_dtype(dataframe[column])
            for column in (volume_column, outcome_column)
        ):
            raise ValueError("Volume/outcome analysis requires numeric metrics.")
        agg = _aggregation(aggregation)
        grouped = (
            dataframe.groupby(category_column, dropna=False)[
                [volume_column, outcome_column]
            ]
            .agg(agg)
            .reset_index()
        )
        volume_threshold = float(grouped[volume_column].median())
        outcome_threshold = float(grouped[outcome_column].median())
        candidates = grouped.loc[
            (grouped[volume_column] >= volume_threshold)
            & (grouped[outcome_column] < outcome_threshold)
        ].sort_values(
            [volume_column, outcome_column],
            ascending=[False, True],
        )
        records = candidates.to_dict(orient="records")
        return (
            f"Found {len(records)} {category_column} value(s) with "
            f"{volume_column} at or above the median and "
            f"{outcome_column} below the median.",
            {
                "category_column": category_column,
                "volume_column": volume_column,
                "outcome_column": outcome_column,
                "aggregation": aggregation,
                "volume_threshold": volume_threshold,
                "outcome_threshold": outcome_threshold,
                "candidates": records,
            },
            [],
        )

    return _timed("analyze_high_volume_low_outcome", operation)


def filter_dataset(
    dataframe: pd.DataFrame,
    column: str,
    operator: str,
    value: Any,
) -> ToolResult:
    def operation():
        _validate_columns(dataframe, column)
        operations = {
            "equals": lambda series: series == value,
            "not equals": lambda series: series != value,
            "greater than": lambda series: series > value,
            "less than": lambda series: series < value,
            "contains": lambda series: series.astype(str).str.contains(
                str(value), case=False, na=False
            ),
        }
        if operator not in operations:
            raise ValueError(f'Unsupported filter operator "{operator}".')
        result = dataframe.loc[operations[operator](dataframe[column])].copy()
        return (
            f"Filtered to {len(result):,} matching rows.",
            result.to_dict(orient="records"),
            [],
        )

    return _timed("filter_dataset", operation)


def sort_and_limit(
    dataframe: pd.DataFrame,
    sort_by: str,
    limit: int = 5,
    ascending: bool = False,
    columns: list[str] | None = None,
) -> ToolResult:
    def operation():
        _validate_columns(dataframe, sort_by, *(columns or []))
        if limit < 1 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000.")
        selected = columns or list(dataframe.columns)
        result = dataframe.sort_values(sort_by, ascending=ascending).head(limit)[
            selected
        ]
        return (
            f"Returned {len(result)} rows sorted by {sort_by}.",
            result.to_dict(orient="records"),
            [],
        )

    return _timed("sort_and_limit", operation)


def calculate_correlation(
    dataframe: pd.DataFrame,
    first_column: str | None = None,
    second_column: str | None = None,
) -> ToolResult:
    def operation():
        if first_column and second_column:
            _validate_columns(dataframe, first_column, second_column)
            if not all(
                pd.api.types.is_numeric_dtype(dataframe[column])
                for column in (first_column, second_column)
            ):
                raise ValueError("Correlation requires numeric columns.")
            value = dataframe[[first_column, second_column]].corr().iloc[0, 1]
            return (
                f"The Pearson correlation between {first_column} and {second_column} is {value:.4f}.",
                {
                    "first_column": first_column,
                    "second_column": second_column,
                    "correlation": float(value),
                },
                [],
            )
        numeric = dataframe.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            raise ValueError(
                "At least two numeric columns are required for correlation."
            )
        return (
            "Calculated the numeric correlation matrix.",
            numeric.corr().reset_index().to_dict(orient="records"),
            [],
        )

    return _timed("calculate_correlation", operation)


def detect_outliers(dataframe: pd.DataFrame, column: str | None = None) -> ToolResult:
    def operation():
        selected = (
            [column]
            if column
            else list(dataframe.select_dtypes(include="number").columns)
        )
        _validate_columns(dataframe, *selected)
        records = []
        for name in selected:
            values = pd.to_numeric(dataframe[name], errors="coerce")
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                count = 0
                lower = upper = None
            else:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                count = int(((values < lower) | (values > upper)).sum())
            records.append(
                {
                    "column": name,
                    "outlier_count": count,
                    "lower_bound": lower,
                    "upper_bound": upper,
                }
            )
        total = sum(item["outlier_count"] for item in records)
        return (f"Detected {total:,} potential IQR outlier(s).", records, [])

    return _timed("detect_outliers", operation)


def analyze_missing_values(dataframe: pd.DataFrame) -> ToolResult:
    def operation():
        missing = dataframe.isna().sum()
        result = [
            {
                "column": column,
                "missing_count": int(count),
                "missing_percentage": float(count / max(len(dataframe), 1) * 100),
            }
            for column, count in missing.items()
            if count
        ]
        return (f"{len(result)} column(s) contain missing values.", result, [])

    return _timed("analyze_missing_values", operation)


def analyze_duplicates(dataframe: pd.DataFrame) -> ToolResult:
    return _timed(
        "analyze_duplicates",
        lambda: (
            f"The dataset contains {int(dataframe.duplicated().sum()):,} duplicate row(s).",
            {
                "duplicate_rows": int(dataframe.duplicated().sum()),
                "duplicate_percentage": float(dataframe.duplicated().mean() * 100),
            },
            [],
        ),
    )


def calculate_time_trend(
    dataframe: pd.DataFrame,
    date_column: str,
    value_column: str,
    value_columns: list[str] | None = None,
    breakdown_column: str | None = None,
    aggregation: str = "sum",
    frequency: str = "month",
    start_date: Any | None = None,
    end_date: Any | None = None,
    filter_column: str | None = None,
    filter_value: Any | None = None,
) -> ToolResult:
    def operation():
        selected_values = list(dict.fromkeys(value_columns or [value_column]))
        _validate_columns(
            dataframe, date_column, *selected_values, filter_column, breakdown_column
        )
        dates = pd.to_datetime(dataframe[date_column], errors="coerce")
        if dates.notna().sum() == 0:
            raise ValueError(f'"{date_column}" does not contain valid dates.')
        agg = _aggregation(aggregation)
        if agg not in {"count", "nunique"} and any(
            not pd.api.types.is_numeric_dtype(dataframe[column])
            for column in selected_values
        ):
            raise ValueError("Time trends require a numeric value column.")
        frequency_map = {
            "day": "D",
            "week": "W",
            "month": "M",
            "quarter": "Q",
            "year": "Y",
        }
        if frequency not in frequency_map:
            raise ValueError(f'Unsupported time frequency "{frequency}".')
        frame = dataframe.loc[dates.notna()].copy()
        dates = dates.loc[dates.notna()]
        if filter_column is not None:
            series = frame[filter_column]
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(
                series
            ):
                mask = (
                    series.astype("string").str.casefold()
                    == str(filter_value).casefold()
                )
            else:
                mask = series == filter_value
            frame = frame.loc[mask]
            dates = dates.loc[mask]
            if frame.empty:
                raise ValueError(f'No rows matched {filter_column} = "{filter_value}".')
        if start_date is not None:
            start = pd.to_datetime(start_date, errors="coerce")
            if pd.notna(start):
                mask = dates >= start
                frame = frame.loc[mask]
                dates = dates.loc[mask]
        if end_date is not None:
            end = pd.to_datetime(end_date, errors="coerce")
            if pd.notna(end):
                if end == end.normalize():
                    mask = dates < end + pd.Timedelta(days=1)
                else:
                    mask = dates <= end
                frame = frame.loc[mask]
                dates = dates.loc[mask]
        if frame.empty:
            raise ValueError("No rows matched the selected date range.")
        frame = frame.assign(
            _period=dates.dt.to_period(frequency_map[frequency]).astype(str)
        )
        group_columns = ["_period"]
        if breakdown_column and breakdown_column not in group_columns:
            group_columns.append(breakdown_column)
        result = (
            frame.dropna(subset=["_period"])
            .groupby(group_columns, dropna=False)[selected_values]
            .agg(agg)
            .reset_index()
            .rename(columns={"_period": date_column})
        )
        if breakdown_column:
            result = result.sort_values([date_column, breakdown_column])
        return (
            f"Calculated the {frequency}ly {aggregation} trend for {', '.join(selected_values)}.",
            result.to_dict(orient="records"),
            [],
        )

    return _timed("calculate_time_trend", operation)


def calculate_period_over_period(
    dataframe: pd.DataFrame,
    date_column: str,
    value_column: str,
    aggregation: str = "sum",
    frequency: str = "month",
    comparison_basis: str = "previous_period",
    start_date: Any | None = None,
    end_date: Any | None = None,
    filter_column: str | None = None,
    filter_value: Any | None = None,
) -> ToolResult:
    """Calculate current, previous-period, absolute-change, and percent-change values."""

    def operation():
        _validate_columns(dataframe, date_column, value_column, filter_column)
        agg = _aggregation(aggregation)
        frequency_map = {"day": "D", "week": "W-MON", "month": "MS", "quarter": "QS", "year": "YS"}
        period_format = {"day": "%Y-%m-%d", "week": "%Y-%m-%d", "month": "%Y-%m", "quarter": None, "year": "%Y"}
        period_alias = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}
        if frequency not in frequency_map:
            raise ValueError(f'Unsupported time frequency "{frequency}".')
        if comparison_basis not in {"previous_period", "previous_year", "same_period_last_year"}:
            raise ValueError(f'Unsupported comparison basis "{comparison_basis}".')
        working = dataframe.copy()
        if filter_column is not None:
            series = working[filter_column]
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                mask = series.astype("string").str.casefold() == str(filter_value).casefold()
            else:
                mask = series == filter_value
            working = working.loc[mask]
            if working.empty:
                raise ValueError(f'No rows matched {filter_column} = "{filter_value}".')
        dates = pd.to_datetime(working[date_column], errors="coerce", format="mixed")
        working = working.loc[dates.notna()].copy()
        working["_period_date"] = dates.loc[dates.notna()]
        if working.empty:
            raise ValueError(f'"{date_column}" does not contain valid dates.')
        if agg not in {"count", "nunique"}:
            if not pd.api.types.is_numeric_dtype(working[value_column]):
                raise ValueError("Period-over-period analysis requires a numeric value column.")
            working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
            working = working.dropna(subset=[value_column])
        if working.empty:
            raise ValueError(f"No valid {value_column} values were available.")
        display_start = pd.to_datetime(start_date, errors="coerce") if start_date is not None else working["_period_date"].min()
        display_end = pd.to_datetime(end_date, errors="coerce") if end_date is not None else working["_period_date"].max()
        if pd.isna(display_start):
            display_start = working["_period_date"].min()
        if pd.isna(display_end):
            display_end = working["_period_date"].max()
        display_start = pd.Timestamp(display_start)
        display_end = pd.Timestamp(display_end)
        lag = pd.DateOffset(years=1) if comparison_basis in {"previous_year", "same_period_last_year"} else {
            "day": pd.DateOffset(days=1),
            "week": pd.DateOffset(weeks=1),
            "month": pd.DateOffset(months=1),
            "quarter": pd.DateOffset(months=3),
            "year": pd.DateOffset(years=1),
        }[frequency]
        calculation_start = display_start - lag
        calculation = working.loc[
            (working["_period_date"] >= calculation_start)
            & (working["_period_date"] <= display_end)
        ].copy()
        if calculation.empty:
            raise ValueError("No rows matched the selected date range.")
        calculation = calculation.set_index("_period_date").sort_index()
        resampler = calculation[value_column].resample(frequency_map[frequency])
        if agg == "count":
            aggregated = resampler.count()
        elif agg == "nunique":
            aggregated = resampler.nunique()
        else:
            aggregated = getattr(resampler, agg)()
        expected = pd.date_range(aggregated.index.min(), aggregated.index.max(), freq=frequency_map[frequency])
        aggregated = aggregated.reindex(expected)
        baseline = aggregated.shift(1)
        absolute_change = aggregated - baseline
        percentage_change = (absolute_change / baseline.abs() * 100).where(baseline != 0)
        frame = pd.DataFrame({
            date_column: aggregated.index,
            value_column: aggregated.values,
            "PreviousPeriodValue": baseline.values,
            "AbsoluteChange": absolute_change.values,
            "PercentageChange": percentage_change.values,
        })
        display_period_start = display_start.to_period(period_alias[frequency]).start_time
        frame = frame.loc[(frame[date_column] >= display_period_start) & (frame[date_column] <= display_end)]
        if frame.empty:
            raise ValueError("No display periods were available after applying the date range.")
        if frequency == "quarter":
            frame[date_column] = frame[date_column].dt.to_period("Q").astype(str)
        else:
            frame[date_column] = frame[date_column].dt.strftime(period_format[frequency])
        rows = frame.replace({np.nan: None}).to_dict(orient="records")
        return (
            f"Calculated {frequency}ly {agg} {value_column} change compared with the previous period.",
            rows,
            [],
        )

    return _timed("calculate_period_over_period", operation)


def calculate_date_aggregate(
    dataframe: pd.DataFrame,
    date_column: str,
    value_column: str,
    aggregation: str = "sum",
    start_date: Any | None = None,
    end_date: Any | None = None,
    period_type: str | None = None,
    period_value: Any | None = None,
    filter_column: str | None = None,
    filter_value: Any | None = None,
) -> ToolResult:
    def operation():
        _validate_columns(dataframe, date_column, value_column, filter_column)
        category_filters = {filter_column: filter_value} if filter_column else None
        result = aggregate_metric_by_period(
            dataframe,
            date_column,
            value_column,
            aggregation=aggregation,
            period_type=period_type,
            period_value=period_value,
            start_date=start_date,
            end_date=end_date,
            category_filters=category_filters,
            current_date=pd.Timestamp.now(),
        )
        label = result.period_label
        summary = (
            f"Calculated {aggregation} {value_column} for {label} "
            f"using {date_column}."
        )
        return (summary, result.as_dict(), list(result.warnings))

    return _timed("calculate_date_aggregate", operation)


def compare_categories(
    dataframe: pd.DataFrame,
    category_column: str,
    value_column: str,
    aggregation: str = "mean",
) -> ToolResult:
    return group_and_aggregate(
        dataframe, category_column, value_column, aggregation
    ).model_copy(update={"tool_name": "compare_categories"})


def calculate_value_counts(
    dataframe: pd.DataFrame, column: str, limit: int = 20
) -> ToolResult:
    def operation():
        _validate_columns(dataframe, column)
        result = (
            dataframe[column]
            .value_counts(dropna=False)
            .head(limit)
            .rename_axis(column)
            .reset_index(name="count")
        )
        return (
            f"Calculated the most frequent values in {column}.",
            result.to_dict(orient="records"),
            [],
        )

    return _timed("calculate_value_counts", operation)


def _filter_mask(series: pd.Series, operator: str, value: Any) -> pd.Series:
    values = value if isinstance(value, list) else [value]
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        normalized = series.astype("string").str.strip().str.casefold()
        normalized_values = [str(item).strip().casefold() for item in values]
        if operator == "equals":
            return normalized == normalized_values[0]
        if operator == "not_equals":
            return normalized != normalized_values[0]
        if operator == "in":
            return normalized.isin(normalized_values)
        if operator == "not_in":
            return ~normalized.isin(normalized_values)
    if operator == "equals":
        return series == value
    if operator == "not_equals":
        return series != value
    if operator == "in":
        return series.isin(values)
    if operator == "not_in":
        return ~series.isin(values)
    raise ValueError(f'Unsupported categorical filter operator "{operator}".')


def analyze_categorical_value_counts(
    dataframe: pd.DataFrame,
    counted_column: str,
    primary_group_column: str | None = None,
    filters: list[dict[str, Any]] | None = None,
    include_missing: bool = False,
    normalization: str = "none",
    chart_type: str = "bar",
    sort_mode: str = "count_descending",
    measure_type: str = "row_count",
    distinct_column: str | None = None,
    original_query: str = "",
) -> ToolResult:
    """Count categorical values, optionally grouped and filtered."""

    def operation():
        parsed_filters = [
            AnalyticsFilter.model_validate(item) for item in (filters or [])
        ]
        request = CategoricalCountRequest(
            counted_column=counted_column,
            primary_group_column=primary_group_column,
            filters=parsed_filters,
            include_missing=include_missing,
            normalization=normalization,
            chart_type=chart_type,
            sort_mode=sort_mode,
            measure_type=measure_type,
            distinct_column=distinct_column,
            original_query=original_query,
        )
        _validate_columns(
            dataframe,
            counted_column,
            primary_group_column,
            distinct_column if measure_type == "distinct_count" else None,
            *(item.column for item in parsed_filters),
        )
        working = dataframe.copy()
        for item in parsed_filters:
            mask = _filter_mask(working[item.column], item.operator, item.value)
            working = working.loc[mask]
        if working.empty:
            filter_text = ", ".join(
                f"{item.column} {item.operator} {item.value}" for item in parsed_filters
            )
            raise ValueError(f"No records were found for {filter_text}.")

        counted_label = counted_column
        group_label = primary_group_column
        value_series = working[counted_column]
        if include_missing:
            working = working.copy()
            working[counted_column] = value_series.astype("string").fillna("Missing")
            if primary_group_column:
                working[primary_group_column] = (
                    working[primary_group_column].astype("string").fillna("Missing")
                )
        else:
            drop_columns = [counted_column]
            if primary_group_column:
                drop_columns.append(primary_group_column)
            if measure_type == "distinct_count" and distinct_column:
                drop_columns.append(distinct_column)
            working = working.dropna(subset=drop_columns)
        if working.empty:
            raise ValueError(
                "No records were available after excluding missing categorical values."
            )

        count_name = (
            "Count"
            if measure_type == "row_count"
            else f"Unique {distinct_column} Count"
        )
        if primary_group_column:
            group_columns = [primary_group_column, counted_column]
            if measure_type == "distinct_count":
                counts = (
                    working.groupby(group_columns, dropna=False)[distinct_column]
                    .nunique(dropna=True)
                    .reset_index(name=count_name)
                )
            else:
                counts = (
                    working.groupby(group_columns, dropna=False)
                    .size()
                    .reset_index(name=count_name)
                )
            counts[count_name] = counts[count_name].astype(int)
            counts["GroupTotal"] = counts.groupby(primary_group_column)[
                count_name
            ].transform("sum")
            counts["Percentage"] = counts[count_name] / counts["GroupTotal"] * 100
            global_order = (
                counts.groupby(counted_column)[count_name]
                .sum()
                .sort_values(ascending=False, kind="mergesort")
                .index.tolist()
            )
            category_order = {value: index for index, value in enumerate(global_order)}
            if sort_mode == "category":
                counts = counts.sort_values(
                    [primary_group_column, counted_column], kind="mergesort"
                )
            elif sort_mode == "count_ascending":
                counts = counts.sort_values(
                    [primary_group_column, count_name],
                    ascending=[True, True],
                    kind="mergesort",
                )
            else:
                counts = (
                    counts.assign(
                        _category_order=counts[counted_column].map(category_order)
                    )
                    .sort_values(
                        [primary_group_column, "_category_order"], kind="mergesort"
                    )
                    .drop(columns="_category_order")
                )
            table_rows = [
                {
                    primary_group_column: row[primary_group_column],
                    counted_column: row[counted_column],
                    count_name: int(row[count_name]),
                    "Percentage": float(row["Percentage"]),
                }
                for _, row in counts.iterrows()
            ]
            rows = [
                CategoricalCountRow(
                    primary_group=str(row[primary_group_column]),
                    category_value=str(row[counted_column]),
                    count=int(row[count_name]),
                    percentage=float(row["Percentage"]),
                )
                for _, row in counts.iterrows()
            ]
            primary_group_count = int(
                counts[primary_group_column].nunique(dropna=False)
            )
        else:
            if measure_type == "distinct_count":
                counts = (
                    working.groupby(counted_column, dropna=False)[distinct_column]
                    .nunique(dropna=True)
                    .reset_index(name=count_name)
                )
            else:
                counts = (
                    working[counted_column].value_counts(dropna=False).reset_index()
                )
                counts.columns = [counted_column, count_name]
            counts[count_name] = counts[count_name].astype(int)
            total = int(counts[count_name].sum())
            counts["Percentage"] = counts[count_name] / max(total, 1) * 100
            if sort_mode == "category":
                counts = counts.sort_values(counted_column, kind="mergesort")
            elif sort_mode == "count_ascending":
                counts = counts.sort_values(
                    count_name, ascending=True, kind="mergesort"
                )
            else:
                counts = counts.sort_values(
                    count_name, ascending=False, kind="mergesort"
                )
            table_rows = [
                {
                    counted_column: row[counted_column],
                    count_name: int(row[count_name]),
                    "Percentage": float(row["Percentage"]),
                }
                for _, row in counts.iterrows()
            ]
            rows = [
                CategoricalCountRow(
                    category_value=str(row[counted_column]),
                    count=int(row[count_name]),
                    percentage=float(row["Percentage"]),
                )
                for _, row in counts.iterrows()
            ]
            primary_group_count = 0

        category_value_count = int(len({row.category_value for row in rows}))
        warnings: list[str] = []
        if category_value_count > 50:
            warnings.append(
                f"{counted_column} has {category_value_count} displayed values; the table is complete but the chart may be dense."
            )
        missing_count = int(dataframe[counted_column].isna().sum())
        if missing_count and not include_missing:
            warnings.append(
                f"Excluded {missing_count:,} missing {counted_column} value(s)."
            )
        if "OrderID" in dataframe.columns and measure_type == "row_count":
            duplicate_orders = int(dataframe["OrderID"].duplicated().sum())
            if duplicate_orders:
                warnings.append(
                    "Counts are dataset rows, not unique OrderID values; duplicate OrderID rows exist."
                )

        result = CategoricalCountResult(
            request=request,
            total_matching_rows=int(len(working)),
            primary_group_count=primary_group_count,
            category_value_count=category_value_count,
            rows=rows,
            table_columns=list(table_rows[0].keys()) if table_rows else [],
            table_rows=table_rows,
            chart_type=chart_type,
            chart_rows=table_rows,
            warnings=warnings,
        )
        filter_text = " after filters" if parsed_filters else ""
        group_text = f" by {primary_group_column}" if primary_group_column else ""
        return (
            f"Calculated {count_name.lower()} for {counted_label}{group_text}{filter_text}.",
            result.model_dump(mode="json"),
            warnings,
        )

    return _timed("analyze_categorical_value_counts", operation)


def calculate_average_numeric_columns(dataframe: pd.DataFrame) -> ToolResult:
    def operation():
        numeric = dataframe.select_dtypes(include="number")
        if numeric.empty:
            raise ValueError("No numeric columns are available.")
        result = numeric.mean().rename("mean").rename_axis("column").reset_index()
        return (
            "Calculated the average of each numeric column.",
            result.to_dict(orient="records"),
            [],
        )

    return _timed("calculate_average_numeric_columns", operation)


def create_bar_chart(dataframe: pd.DataFrame, **_: Any) -> ToolResult:
    return ToolResult(
        tool_name="create_bar_chart",
        summary="Use chart_service with a validated bar ChartSpec.",
    )


def create_line_chart(dataframe: pd.DataFrame, **_: Any) -> ToolResult:
    return ToolResult(
        tool_name="create_line_chart",
        summary="Use chart_service with a validated line ChartSpec.",
    )


def create_scatter_plot(dataframe: pd.DataFrame, **_: Any) -> ToolResult:
    return ToolResult(
        tool_name="create_scatter_plot",
        summary="Use chart_service with a validated scatter ChartSpec.",
    )


def create_histogram(dataframe: pd.DataFrame, **_: Any) -> ToolResult:
    return ToolResult(
        tool_name="create_histogram",
        summary="Use chart_service with a validated histogram ChartSpec.",
    )


def create_box_plot(dataframe: pd.DataFrame, **_: Any) -> ToolResult:
    return ToolResult(
        tool_name="create_box_plot",
        summary="Use chart_service with a validated box ChartSpec.",
    )


def create_pie_chart(dataframe: pd.DataFrame, **_: Any) -> ToolResult:
    return ToolResult(
        tool_name="create_pie_chart",
        summary="Use chart_service with a validated pie ChartSpec.",
    )


def create_heatmap(dataframe: pd.DataFrame, **_: Any) -> ToolResult:
    return ToolResult(
        tool_name="create_heatmap",
        summary="Use chart_service with a validated heatmap ChartSpec.",
    )


def generate_report(dataframe: pd.DataFrame, **_: Any) -> ToolResult:
    profile = profile_dataset(dataframe)
    return ToolResult(
        tool_name="generate_report",
        summary="Prepared verified dataset facts for report generation.",
        data={
            "rows": profile.row_count,
            "columns": profile.column_count,
            "quality_score": profile.quality.score,
            "missing_percentage": profile.missing_percentage,
        },
    )
