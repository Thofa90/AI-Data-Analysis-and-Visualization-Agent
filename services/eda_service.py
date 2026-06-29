"""Automated deterministic exploratory data analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field
import pandas as pd

from services.chart_service import ChartSpec
from services.profile_models import DatasetProfile


class EDASummary(BaseModel):
    """Structured EDA observations and suggested charts."""

    observations: list[str] = Field(default_factory=list)
    chart_specs: list[ChartSpec] = Field(default_factory=list)


def generate_eda_summary(dataframe: pd.DataFrame, profile: DatasetProfile) -> EDASummary:
    """Generate verified observations and useful chart specifications."""
    observations = [
        f"The dataset contains {profile.row_count:,} rows and {profile.column_count:,} columns.",
        (
            f"Missing values affect {len(profile.columns_with_missing)} column(s) "
            f"and {profile.missing_percentage:.2f}% of all cells."
            if profile.total_missing
            else "No missing values were detected."
        ),
        (
            f"{profile.duplicate_rows:,} duplicate row(s) were detected."
            if profile.duplicate_rows
            else "No duplicate rows were detected."
        ),
        f"The deterministic data quality score is {profile.quality.score:.1f}/100 ({profile.quality.rating}).",
    ]
    for column in profile.columns:
        if column.kind == "numeric" and column.skewness is not None and abs(column.skewness) >= 1:
            direction = "right" if column.skewness > 0 else "left"
            observations.append(f'"{column.name}" is strongly {direction}-skewed (skewness {column.skewness:.2f}).')
        if column.outlier_count:
            observations.append(f'"{column.name}" contains {column.outlier_count:,} potential IQR outlier(s).')
    for column in profile.columns:
        if column.kind == "datetime":
            observations.append(f'"{column.name}" ranges from {column.minimum} to {column.maximum}.')

    chart_specs: list[ChartSpec] = []
    for column in profile.numeric_columns[:3]:
        chart_specs.append(ChartSpec(
            chart_type="histogram",
            x=column,
            title=f"Distribution of {column}",
        ))
        chart_specs.append(ChartSpec(
            chart_type="box",
            y=column,
            title=f"Potential Outliers in {column}",
        ))
    if len(profile.numeric_columns) >= 2:
        chart_specs.append(ChartSpec(chart_type="heatmap", title="Numeric Correlation Matrix"))
    if profile.categorical_columns and profile.numeric_columns:
        category = profile.categorical_columns[0]
        metric = profile.numeric_columns[0]
        chart_specs.append(ChartSpec(
            chart_type="bar",
            x=category,
            y=metric,
            aggregation="mean",
            sort_descending=True,
            limit=15,
            title=f"Average {metric} by {category}",
        ))
    if profile.datetime_columns and profile.numeric_columns:
        chart_specs.append(ChartSpec(
            chart_type="line",
            x=profile.datetime_columns[0],
            y=profile.numeric_columns[0],
            aggregation="sum",
            title=f"{profile.numeric_columns[0]} over Time",
        ))
    return EDASummary(observations=observations[:12], chart_specs=chart_specs[:8])
