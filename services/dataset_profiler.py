"""Deterministic dataset profiling and explainable quality scoring."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from services.profile_models import (
    ColumnProfile,
    DataQualityScore,
    DatasetProfile,
    QualityIssue,
)


ID_NAME_PATTERN = re.compile(
    r"(^|[\s_\-])(id|key|code|index|identifier|number|no)([\s_\-]|$)",
    re.IGNORECASE,
)
DATE_NAME_PATTERN = re.compile(r"date|time|timestamp|created|updated|year|month|day", re.IGNORECASE)
MEASURE_NAME_PATTERN = re.compile(
    r"revenue|sales|profit|salary|cost|price|amount|value|quantity|score|rate|age|duration|output",
    re.IGNORECASE,
)


def _safe_scalar(value: Any) -> Any | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _is_sequential_numeric(series: pd.Series) -> bool:
    values = series.dropna()
    if len(values) < 3 or not ptypes.is_numeric_dtype(values):
        return False
    unique = np.sort(values.unique())
    if len(unique) != len(values):
        return False
    differences = np.diff(unique.astype(float))
    return bool(len(differences) and np.allclose(differences, differences[0]) and differences[0] != 0)


def is_likely_id_column(series: pd.Series, name: str, row_count: int) -> bool:
    """Detect identifiers using names, uniqueness, and sequential patterns."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_ratio = non_null.nunique(dropna=True) / len(non_null)
    normalized_name = re.sub(r"[^a-z0-9]+", "", name.lower())
    name_hint = (
        bool(ID_NAME_PATTERN.search(name.replace(".", "_")))
        or normalized_name.endswith(("id", "key", "code", "index", "identifier"))
        or normalized_name.startswith(("id", "key", "index"))
    )
    sequential = _is_sequential_numeric(non_null)
    near_one_to_one = row_count >= 5 and unique_ratio >= 0.98
    string_tokens = (
        ptypes.is_object_dtype(series.dtype)
        and unique_ratio >= 0.95
        and non_null.astype(str).str.match(r"^[A-Za-z]{0,5}[-_]?\d+$").mean() >= 0.8
    )
    sequence_identifier = (
        row_count >= 10
        and near_one_to_one
        and sequential
        and not MEASURE_NAME_PATTERN.search(name)
    )
    return bool(name_hint or string_tokens or sequence_identifier)


def _datetime_candidate(series: pd.Series, name: str) -> pd.Series | None:
    if ptypes.is_datetime64_any_dtype(series.dtype):
        return pd.to_datetime(series, errors="coerce")
    if not (ptypes.is_object_dtype(series.dtype) or ptypes.is_string_dtype(series.dtype)):
        return None
    non_null = series.dropna()
    if non_null.empty or not DATE_NAME_PATTERN.search(name):
        return None
    converted = pd.to_datetime(series, errors="coerce")
    return converted if converted.notna().sum() / len(non_null) >= 0.8 else None


def _potential_type_issue(series: pd.Series, name: str) -> str | None:
    if not (ptypes.is_object_dtype(series.dtype) or ptypes.is_string_dtype(series.dtype)):
        return None
    raw = series.dropna().astype(str)
    non_null = raw.str.strip()
    if non_null.empty:
        return None
    numeric_ratio = pd.to_numeric(non_null, errors="coerce").notna().mean()
    if 0.2 <= numeric_ratio < 0.95:
        return f'"{name}" mixes numeric-looking and text values.'
    if non_null.str.len().gt(0).any() and (raw != non_null).any():
        return f'"{name}" contains leading or trailing whitespace.'
    return None


def _outlier_count(series: pd.Series) -> int:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 4:
        return 0
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0 or pd.isna(iqr):
        return 0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((values < lower) | (values > upper)).sum())


def _profile_column(series: pd.Series, row_count: int) -> ColumnProfile:
    name = str(series.name)
    non_null = series.dropna()
    missing_count = int(series.isna().sum())
    unique_count = int(non_null.nunique(dropna=True))
    unique_percentage = unique_count / max(len(non_null), 1) * 100
    likely_id = is_likely_id_column(series, name, row_count)
    parsed_datetime = _datetime_candidate(series, name)
    type_issue = _potential_type_issue(series, name)

    if likely_id:
        kind = "identifier"
    elif ptypes.is_bool_dtype(series.dtype):
        kind = "boolean"
    elif parsed_datetime is not None:
        kind = "datetime"
    elif ptypes.is_numeric_dtype(series.dtype):
        kind = "numeric"
    elif (
        ptypes.is_object_dtype(series.dtype)
        or ptypes.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        kind = "categorical"
    else:
        kind = "other"

    high_cardinality = (
        kind == "categorical"
        and unique_count > 20
        and unique_count / max(len(non_null), 1) > 0.5
    )
    top_values = {
        str(key): int(value)
        for key, value in non_null.astype(str).value_counts().head(5).items()
    }
    profile = ColumnProfile(
        name=name,
        pandas_dtype=str(series.dtype),
        kind=kind,
        row_count=row_count,
        non_null_count=int(series.notna().sum()),
        missing_count=missing_count,
        missing_percentage=missing_count / max(row_count, 1) * 100,
        unique_count=unique_count,
        unique_percentage=unique_percentage,
        is_constant=unique_count <= 1,
        is_high_cardinality=high_cardinality,
        is_likely_id=likely_id,
        potential_type_issue=type_issue,
        top_values=top_values,
    )

    if kind == "numeric":
        numeric = pd.to_numeric(series, errors="coerce")
        profile.outlier_count = _outlier_count(numeric)
        profile.minimum = _safe_scalar(numeric.min())
        profile.maximum = _safe_scalar(numeric.max())
        profile.mean = _safe_scalar(numeric.mean())
        profile.median = _safe_scalar(numeric.median())
        profile.standard_deviation = _safe_scalar(numeric.std())
        skew = numeric.skew()
        profile.skewness = None if pd.isna(skew) or math.isinf(float(skew)) else float(skew)
    elif kind == "datetime" and parsed_datetime is not None:
        profile.minimum = _safe_scalar(parsed_datetime.min())
        profile.maximum = _safe_scalar(parsed_datetime.max())
    return profile


def calculate_quality_score(
    dataframe: pd.DataFrame,
    columns: list[ColumnProfile],
    duplicate_rows: int,
) -> DataQualityScore:
    """Calculate a capped, deterministic 0-100 quality score."""
    rows, column_count = dataframe.shape
    total_cells = max(rows * column_count, 1)
    issues: list[QualityIssue] = []

    missing_percentage = dataframe.isna().sum().sum() / total_cells * 100
    if missing_percentage:
        deduction = min(30.0, missing_percentage * 0.6)
        issues.append(QualityIssue(
            category="Missing values",
            deduction=deduction,
            detail=f"{missing_percentage:.2f}% of all cells are missing.",
            recommendation="Review affected columns and choose an appropriate fill or removal strategy.",
        ))

    duplicate_percentage = duplicate_rows / max(rows, 1) * 100
    if duplicate_rows:
        issues.append(QualityIssue(
            category="Duplicate rows",
            deduction=min(20.0, duplicate_percentage * 0.8),
            detail=f"{duplicate_rows:,} duplicate rows ({duplicate_percentage:.2f}%).",
            recommendation="Verify whether repeated records are valid, then remove confirmed duplicates.",
        ))

    checks = [
        ("Constant columns", [c for c in columns if c.is_constant], 3.0, 12.0,
         "Remove columns that contain only one distinct value."),
        ("Mixed data types", [c for c in columns if c.potential_type_issue], 4.0, 16.0,
         "Standardize values and convert each affected column to one intended type."),
        ("High cardinality", [c for c in columns if c.is_high_cardinality], 1.5, 8.0,
         "Review high-cardinality categories before grouping or visualization."),
        ("Potential outliers", [c for c in columns if c.outlier_count], 1.0, 8.0,
         "Inspect outliers in context before removing or capping them."),
    ]
    for category, affected, per_column, cap, recommendation in checks:
        if affected:
            names = ", ".join(column.name for column in affected[:5])
            issues.append(QualityIssue(
                category=category,
                deduction=min(cap, len(affected) * per_column),
                detail=f"{len(affected)} affected column(s): {names}.",
                recommendation=recommendation,
            ))

    score = max(0.0, 100.0 - sum(issue.deduction for issue in issues))
    rating = "Excellent" if score >= 90 else "Good" if score >= 75 else "Fair" if score >= 50 else "Poor"
    return DataQualityScore(score=round(score, 1), rating=rating, issues=issues)


def profile_dataset(dataframe: pd.DataFrame) -> DatasetProfile:
    """Build a complete structured profile without mutating the source."""
    columns = [_profile_column(dataframe[column], len(dataframe)) for column in dataframe.columns]
    duplicate_rows = int(dataframe.duplicated().sum())
    quality = calculate_quality_score(dataframe, columns, duplicate_rows)
    total_cells = max(dataframe.shape[0] * dataframe.shape[1], 1)
    total_missing = int(dataframe.isna().sum().sum())

    def names_for(kind: str) -> list[str]:
        return [column.name for column in columns if column.kind == kind]

    return DatasetProfile(
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        memory_bytes=int(dataframe.memory_usage(deep=True).sum()),
        total_missing=total_missing,
        missing_percentage=total_missing / total_cells * 100,
        duplicate_rows=duplicate_rows,
        duplicate_percentage=duplicate_rows / max(len(dataframe), 1) * 100,
        numeric_columns=names_for("numeric"),
        categorical_columns=names_for("categorical"),
        boolean_columns=names_for("boolean"),
        datetime_columns=names_for("datetime"),
        id_columns=names_for("identifier"),
        constant_columns=[c.name for c in columns if c.is_constant],
        high_cardinality_columns=[c.name for c in columns if c.is_high_cardinality],
        columns_with_missing=[c.name for c in columns if c.missing_count],
        potential_type_problems=[c.name for c in columns if c.potential_type_issue],
        outlier_columns=[c.name for c in columns if c.outlier_count],
        columns=columns,
        quality=quality,
    )
