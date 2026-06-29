"""Previewable, reversible data-cleaning operations."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field


CleaningAction = Literal[
    "remove_duplicates",
    "fill_numeric_mean",
    "fill_numeric_median",
    "fill_categorical_mode",
    "fill_custom",
    "remove_columns",
    "rename_column",
    "convert_type",
    "trim_whitespace",
    "standardize_category_text",
    "remove_outliers",
]


class CleaningPlan(BaseModel):
    """Validated cleaning request and impact preview."""

    action: CleaningAction
    parameters: dict[str, Any] = Field(default_factory=dict)
    affected_rows: int
    affected_columns: list[str] = Field(default_factory=list)
    description: str


def _column(dataframe: pd.DataFrame, name: str) -> pd.Series:
    if name not in dataframe.columns:
        raise ValueError(f'Column "{name}" was not found.')
    return dataframe[name]


def preview_cleaning(
    dataframe: pd.DataFrame,
    action: CleaningAction,
    parameters: dict[str, Any] | None = None,
) -> CleaningPlan:
    """Calculate the expected impact without mutating data."""
    params = parameters or {}
    if action == "remove_duplicates":
        count = int(dataframe.duplicated().sum())
        return CleaningPlan(action=action, affected_rows=count, affected_columns=list(dataframe.columns),
                            description=f"Remove {count:,} duplicate row(s).")
    if action in {"fill_numeric_mean", "fill_numeric_median", "fill_categorical_mode", "fill_custom"}:
        name = params.get("column")
        series = _column(dataframe, name)
        if action.startswith("fill_numeric") and not pd.api.types.is_numeric_dtype(series):
            raise ValueError("Mean and median filling require a numeric column.")
        count = int(series.isna().sum())
        return CleaningPlan(action=action, parameters=params, affected_rows=count, affected_columns=[name],
                            description=f"Fill {count:,} missing value(s) in {name}.")
    if action == "remove_columns":
        columns = params.get("columns", [])
        for name in columns:
            _column(dataframe, name)
        if not columns or len(columns) >= len(dataframe.columns):
            raise ValueError("Select at least one column while leaving one usable column.")
        return CleaningPlan(action=action, parameters=params, affected_rows=len(dataframe),
                            affected_columns=columns, description=f"Remove {len(columns)} selected column(s).")
    if action == "rename_column":
        old, new = params.get("old"), str(params.get("new", "")).strip()
        _column(dataframe, old)
        if not new or (new in dataframe.columns and new != old):
            raise ValueError("Enter a unique non-empty column name.")
        return CleaningPlan(action=action, parameters={"old": old, "new": new}, affected_rows=len(dataframe),
                            affected_columns=[old], description=f'Rename "{old}" to "{new}".')
    if action == "convert_type":
        name, target = params.get("column"), params.get("target")
        _column(dataframe, name)
        if target not in {"string", "integer", "float", "boolean", "datetime"}:
            raise ValueError("Unsupported target data type.")
        return CleaningPlan(action=action, parameters=params, affected_rows=len(dataframe),
                            affected_columns=[name], description=f"Convert {name} to {target}.")
    if action in {"trim_whitespace", "standardize_category_text"}:
        name = params.get("column")
        series = _column(dataframe, name)
        normalized = series.astype("string").str.strip()
        if action == "standardize_category_text":
            case = params.get("case", "title")
            normalized = getattr(normalized.str, case)()
        count = int((series.astype("string") != normalized).fillna(False).sum())
        return CleaningPlan(action=action, parameters=params, affected_rows=count, affected_columns=[name],
                            description=f"Standardize {count:,} value(s) in {name}.")
    if action == "remove_outliers":
        name = params.get("column")
        series = pd.to_numeric(_column(dataframe, name), errors="coerce")
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = pd.Series(False, index=dataframe.index) if pd.isna(iqr) or iqr == 0 else (
            (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
        )
        count = int(mask.sum())
        return CleaningPlan(action=action, parameters=params, affected_rows=count, affected_columns=[name],
                            description=f"Remove {count:,} potential IQR outlier row(s) from {name}.")
    raise ValueError(f'Unsupported cleaning action "{action}".')


def apply_cleaning(dataframe: pd.DataFrame, plan: CleaningPlan) -> pd.DataFrame:
    """Apply a confirmed plan to a copy and return the cleaned data."""
    result = dataframe.copy(deep=True)
    params = plan.parameters
    if plan.action == "remove_duplicates":
        return result.drop_duplicates().reset_index(drop=True)
    if plan.action in {"fill_numeric_mean", "fill_numeric_median"}:
        name = params["column"]
        fill = result[name].mean() if plan.action.endswith("mean") else result[name].median()
        result[name] = result[name].fillna(fill)
    elif plan.action == "fill_categorical_mode":
        name = params["column"]
        mode = result[name].mode(dropna=True)
        if mode.empty:
            raise ValueError("The selected column has no mode to use as a fill value.")
        result[name] = result[name].fillna(mode.iloc[0])
    elif plan.action == "fill_custom":
        result[params["column"]] = result[params["column"]].fillna(params.get("value"))
    elif plan.action == "remove_columns":
        result = result.drop(columns=params["columns"])
    elif plan.action == "rename_column":
        result = result.rename(columns={params["old"]: params["new"]})
    elif plan.action == "convert_type":
        name, target = params["column"], params["target"]
        converters = {
            "string": lambda value: value.astype("string"),
            "integer": lambda value: pd.to_numeric(value, errors="raise").astype("Int64"),
            "float": lambda value: pd.to_numeric(value, errors="raise").astype(float),
            "boolean": lambda value: value.astype("boolean"),
            "datetime": lambda value: pd.to_datetime(value, errors="raise"),
        }
        try:
            result[name] = converters[target](result[name])
        except Exception as exc:
            raise ValueError(f'"{name}" could not be converted to {target}.') from exc
    elif plan.action == "trim_whitespace":
        name = params["column"]
        result[name] = result[name].astype("string").str.strip()
    elif plan.action == "standardize_category_text":
        name, case = params["column"], params.get("case", "title")
        result[name] = getattr(result[name].astype("string").str.strip().str, case)()
    elif plan.action == "remove_outliers":
        name = params["column"]
        series = pd.to_numeric(result[name], errors="coerce")
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if not pd.isna(iqr) and iqr != 0:
            result = result.loc[(series >= q1 - 1.5 * iqr) & (series <= q3 + 1.5 * iqr)].copy()
    return result.reset_index(drop=True)
