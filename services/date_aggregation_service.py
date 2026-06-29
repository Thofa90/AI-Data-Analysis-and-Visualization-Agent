"""Shared date filtering and aggregation helpers for charts and chat."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
import re
from typing import Any

import pandas as pd


PERIOD_FREQUENCIES = {
    "year": "Y",
    "month": "M",
    "week": "W-MON",
    "day": "D",
}


@dataclass(frozen=True)
class DateAggregateResult:
    metric: str
    aggregation: str
    date_column: str
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None
    value: float | int | None
    row_count: int
    period_label: str
    period_type: str | None = None
    period_value: Any | None = None
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "aggregation": self.aggregation,
            "date_column": self.date_column,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "value": self.value,
            "row_count": self.row_count,
            "period_label": self.period_label,
            "period_type": self.period_type,
            "period_value": self.period_value,
            "warnings": list(self.warnings),
        }


def date_bucket_start(dates: pd.Series, granularity: str) -> pd.Series:
    """Return comparable bucket start timestamps for a datetime series."""
    if granularity not in PERIOD_FREQUENCIES:
        raise ValueError(f'Unsupported date granularity "{granularity}".')
    if granularity == "week":
        normalized = dates.dt.normalize()
        return normalized - pd.to_timedelta(normalized.dt.weekday, unit="D")
    return dates.dt.to_period(PERIOD_FREQUENCIES[granularity]).dt.start_time


def date_bucket_label(value: pd.Timestamp, granularity: str) -> str:
    timestamp = pd.Timestamp(value)
    if granularity == "year":
        return timestamp.strftime("%Y")
    if granularity == "month":
        return timestamp.strftime("%B %Y")
    if granularity == "week":
        iso = timestamp.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return timestamp.strftime("%d %B %Y")


def period_bounds_from_text(
    text: str,
    *,
    current_date: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str | None, str | None]:
    """Parse common relative and explicit date periods from user text."""
    now = pd.Timestamp(current_date or pd.Timestamp.now()).normalize()
    lowered = text.lower()
    if "last year" in lowered:
        year = now.year - 1
        return pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31), "year", str(year)
    if "this year" in lowered:
        return pd.Timestamp(now.year, 1, 1), now, "year", str(now.year)
    if "last month" in lowered:
        first_this_month = pd.Timestamp(now.year, now.month, 1)
        end = first_this_month - pd.Timedelta(days=1)
        start = pd.Timestamp(end.year, end.month, 1)
        return start, end, "month", start.strftime("%Y-%m")
    if "this month" in lowered:
        return pd.Timestamp(now.year, now.month, 1), now, "month", now.strftime("%Y-%m")
    if "last week" in lowered:
        this_monday = now - pd.Timedelta(days=now.weekday())
        start = this_monday - pd.Timedelta(days=7)
        end = this_monday - pd.Timedelta(days=1)
        iso = start.isocalendar()
        return start, end, "week", f"{iso.year}-W{iso.week:02d}"
    if "this week" in lowered:
        start = now - pd.Timedelta(days=now.weekday())
        iso = start.isocalendar()
        return start, now, "week", f"{iso.year}-W{iso.week:02d}"
    if "yesterday" in lowered:
        day = now - pd.Timedelta(days=1)
        return day, day, "day", day.strftime("%Y-%m-%d")
    if "today" in lowered:
        return now, now, "day", now.strftime("%Y-%m-%d")

    month_names = "|".join(calendar.month_name[1:])
    month_match = pd.Series([lowered]).str.extract(rf"\b({month_names})\s+(\d{{4}})\b", flags=re.IGNORECASE, expand=True)
    if month_match.notna().all(axis=None):
        month_name = str(month_match.iloc[0, 0]).title()
        year = int(month_match.iloc[0, 1])
        month = list(calendar.month_name).index(month_name)
        start = pd.Timestamp(year, month, 1)
        end = start + pd.offsets.MonthEnd(0)
        return start, end, "month", start.strftime("%Y-%m")

    year_match = pd.Series([lowered]).str.extract(r"\b(20\d{2}|19\d{2})\b", expand=False).iloc[0]
    if pd.notna(year_match):
        year = int(year_match)
        return pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31), "year", str(year)
    return None, None, None, None


def aggregate_metric_by_period(
    df: pd.DataFrame,
    date_column: str,
    metric_column: str,
    aggregation: str = "sum",
    period_type: str | None = None,
    period_value: Any | None = None,
    period_values: list[Any] | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
    category_filters: dict[str, Any] | None = None,
    current_date: pd.Timestamp | None = None,
) -> DateAggregateResult:
    """Filter by date/category and aggregate one metric consistently."""
    if date_column not in df.columns:
        raise ValueError(f'Date column "{date_column}" was not found.')
    if metric_column not in df.columns:
        raise ValueError(f'Metric column "{metric_column}" was not found.')
    dates = pd.to_datetime(df[date_column], errors="coerce", format="mixed")
    working = df.loc[dates.notna()].copy()
    dates = dates.loc[dates.notna()]
    warnings: list[str] = []
    if len(working) < len(df):
        warnings.append("Rows with invalid or missing dates were excluded.")

    for column, value in (category_filters or {}).items():
        if column not in working.columns:
            raise ValueError(f'Filter column "{column}" was not found.')
        series = working[column]
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            mask = series.astype("string").str.casefold() == str(value).casefold()
        else:
            mask = series == value
        working = working.loc[mask]
        dates = dates.loc[mask]

    if start_date is not None:
        start = pd.to_datetime(start_date, errors="coerce")
        if pd.notna(start):
            mask = dates >= start
            working = working.loc[mask]
            dates = dates.loc[mask]
    else:
        start = None
    if end_date is not None:
        end = pd.to_datetime(end_date, errors="coerce")
        if pd.notna(end):
            if end == end.normalize():
                mask = dates < end + pd.Timedelta(days=1)
            else:
                mask = dates <= end
            working = working.loc[mask]
            dates = dates.loc[mask]
    else:
        end = None

    if current_date is not None and any(term is not None for term in (start_date, end_date)):
        current = pd.Timestamp(current_date).normalize()
        if end is not None and end > current:
            mask = dates < current + pd.Timedelta(days=1)
            working = working.loc[mask]
            dates = dates.loc[mask]
            end = current

    selected_values = period_values if period_values is not None else ([period_value] if period_value is not None else [])
    if period_type and selected_values and not working.empty:
        buckets = date_bucket_start(dates, period_type)
        selected_periods = {
            pd.Timestamp(parsed).normalize()
            for value in selected_values
            if pd.notna(parsed := pd.to_datetime(value, errors="coerce"))
        }
        if selected_periods:
            mask = buckets.map(lambda value: pd.Timestamp(value).normalize()).isin(selected_periods)
            working = working.loc[mask]
            dates = dates.loc[mask]

    agg = aggregation.lower()
    values = pd.to_numeric(working[metric_column], errors="coerce") if metric_column in working else pd.Series(dtype="float64")
    if agg == "count":
        value: float | int | None = int(len(working))
    elif agg in {"nunique", "unique count", "distinct_count"}:
        value = int(working[metric_column].nunique(dropna=True))
    elif agg == "mean":
        value = float(values.mean()) if not values.dropna().empty else None
    elif agg == "median":
        value = float(values.median()) if not values.dropna().empty else None
    elif agg == "min":
        value = float(values.min()) if not values.dropna().empty else None
    elif agg == "max":
        value = float(values.max()) if not values.dropna().empty else None
    else:
        value = float(values.sum()) if not values.dropna().empty else 0.0

    if dates.empty:
        effective_start = pd.to_datetime(start_date, errors="coerce") if start_date is not None else None
        effective_end = pd.to_datetime(end_date, errors="coerce") if end_date is not None else None
    else:
        effective_start = pd.Timestamp(dates.min()).normalize()
        effective_end = pd.Timestamp(dates.max()).normalize()
    if period_type and selected_values:
        labels = []
        for value_item in selected_values:
            label_ts = pd.to_datetime(value_item, errors="coerce")
            labels.append(date_bucket_label(label_ts, period_type) if pd.notna(label_ts) else str(value_item))
        period_label = ", ".join(labels[:3]) + (f" and {len(labels) - 3} more" if len(labels) > 3 else "")
    elif effective_start is not None and effective_end is not None and pd.notna(effective_start) and pd.notna(effective_end):
        period_label = f"{pd.Timestamp(effective_start).date()} to {pd.Timestamp(effective_end).date()}"
    else:
        period_label = "selected period"

    return DateAggregateResult(
        metric=metric_column,
        aggregation=agg,
        date_column=date_column,
        start_date=pd.Timestamp(effective_start) if effective_start is not None and pd.notna(effective_start) else None,
        end_date=pd.Timestamp(effective_end) if effective_end is not None and pd.notna(effective_end) else None,
        value=value,
        row_count=int(len(working)),
        period_label=period_label,
        period_type=period_type,
        period_value=period_value,
        warnings=tuple(warnings),
    )
