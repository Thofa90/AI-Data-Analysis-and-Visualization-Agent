"""Structured models for deterministic dataset profiling."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ColumnKind = Literal["numeric", "categorical", "boolean", "datetime", "identifier", "other"]


class ColumnProfile(BaseModel):
    """Computed facts for one dataset column."""

    name: str
    pandas_dtype: str
    kind: ColumnKind
    row_count: int
    non_null_count: int
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    is_constant: bool = False
    is_high_cardinality: bool = False
    is_likely_id: bool = False
    potential_type_issue: str | None = None
    outlier_count: int = 0
    minimum: Any | None = None
    maximum: Any | None = None
    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None
    skewness: float | None = None
    top_values: dict[str, int] = Field(default_factory=dict)


class QualityIssue(BaseModel):
    """One explainable contribution to the data quality score."""

    category: str
    deduction: float
    detail: str
    recommendation: str


class DataQualityScore(BaseModel):
    """Deterministic data quality result."""

    score: float
    rating: Literal["Excellent", "Good", "Fair", "Poor"]
    issues: list[QualityIssue] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    """Complete structured dataset profile."""

    row_count: int
    column_count: int
    memory_bytes: int
    total_missing: int
    missing_percentage: float
    duplicate_rows: int
    duplicate_percentage: float
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    boolean_columns: list[str] = Field(default_factory=list)
    datetime_columns: list[str] = Field(default_factory=list)
    id_columns: list[str] = Field(default_factory=list)
    constant_columns: list[str] = Field(default_factory=list)
    high_cardinality_columns: list[str] = Field(default_factory=list)
    columns_with_missing: list[str] = Field(default_factory=list)
    potential_type_problems: list[str] = Field(default_factory=list)
    outlier_columns: list[str] = Field(default_factory=list)
    columns: list[ColumnProfile] = Field(default_factory=list)
    quality: DataQualityScore
