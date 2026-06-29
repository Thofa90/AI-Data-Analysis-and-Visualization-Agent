"""Structured contracts for planning, tool execution, and answers."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from services.chart_service import ChartSpec


class AnalyticsFilter(BaseModel):
    """Structured filter extracted from a natural-language analytics request."""

    column: str
    operator: Literal[
        "equals",
        "not_equals",
        "in",
        "not_in",
        "greater_than",
        "greater_than_or_equal",
        "less_than",
        "less_than_or_equal",
        "between",
        "contains",
    ] = "equals"
    value: Any


class TimeRangeFilter(BaseModel):
    """Optional date window extracted from a natural-language analytics request."""

    date_column: Optional[str] = None
    start_date: Optional[Any] = None
    end_date: Optional[Any] = None
    relative_period: Optional[str] = None
    year: Optional[int] = None
    month: Optional[int] = None
    quarter: Optional[int] = None


class AnalyticsQueryRequest(BaseModel):
    """Normalized analytics intent used before deterministic execution."""

    intent: Literal[
        "scalar_aggregation",
        "time_series_breakdown",
        "time_series",
        "grouped_extrema",
        "categorical_breakdown",
        "comparison",
        "unknown",
    ] = "unknown"
    metric_column: str | None = None
    aggregation: Literal[
        "sum",
        "mean",
        "median",
        "count",
        "nunique",
        "min",
        "max",
    ] = "sum"
    filters: list[AnalyticsFilter] = Field(default_factory=list)
    date_column: Optional[str] = None
    time_granularity: Optional[
        Literal[
            "day",
            "week",
            "month",
            "quarter",
            "year",
        ]
    ] = None
    time_range: Optional[TimeRangeFilter] = None
    breakdown_column: Optional[str] = None
    include_overall_summary: bool = True
    include_table: bool = False
    include_chart: bool = False
    include_explanation: bool = True
    chart_type: Optional[Literal["line", "bar"]] = None
    sort_direction: Literal[
        "ascending",
        "descending",
        "chronological",
    ] = "chronological"


class CategoricalCountRequest(BaseModel):
    """Normalized request for categorical value-count/frequency analysis."""

    intent: Literal["categorical_value_counts"] = "categorical_value_counts"
    counted_column: str
    primary_group_column: str | None = None
    filters: list[AnalyticsFilter] = Field(default_factory=list)
    include_missing: bool = False
    normalization: Literal["none", "within_primary_group", "overall"] = "none"
    chart_type: Literal[
        "bar",
        "grouped_bar",
        "stacked_bar",
        "percentage_stacked_bar",
    ] = "bar"
    sort_mode: Literal[
        "count_descending",
        "count_ascending",
        "category",
        "primary_group",
    ] = "count_descending"
    include_table: bool = True
    include_chart: bool = True
    include_explanation: bool = True
    measure_type: Literal["row_count", "distinct_count"] = "row_count"
    distinct_column: str | None = None
    original_query: str = ""


class CategoricalCountRow(BaseModel):
    """One categorical count row."""

    primary_group: str | None = None
    category_value: str
    count: int
    percentage: float | None = None


class CategoricalCountResult(BaseModel):
    """Structured categorical count result."""

    request: CategoricalCountRequest
    total_matching_rows: int
    primary_group_count: int = 0
    category_value_count: int = 0
    rows: list[CategoricalCountRow] = Field(default_factory=list)
    table_columns: list[str] = Field(default_factory=list)
    table_rows: list[dict[str, Any]] = Field(default_factory=list)
    chart_type: str
    chart_rows: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


ColumnSemanticType = Literal[
    "categorical",
    "numerical",
    "datetime",
    "identifier",
    "boolean",
    "free_text",
    "unknown",
]


class ColumnProfileRequest(BaseModel):
    """Normalized request for a single-column data profile."""

    intent: Literal["column_profile"] = "column_profile"
    column_name: str
    include_table: bool = True
    include_chart: bool = True
    include_examples: bool = True
    include_semantic_explanation: bool = True
    original_query: str = ""


class CommonColumnProfile(BaseModel):
    """Type-aware facts for one dataset column."""

    column_name: str
    display_name: str
    pandas_dtype: str
    semantic_type: ColumnSemanticType
    row_count: int
    non_null_count: int
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_ratio: float
    meaning: str
    meaning_confidence: Literal["high", "medium", "low"] = "low"
    example_values: list[Any] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    minimum: Any | None = None
    maximum: Any | None = None
    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None
    q1: float | None = None
    q3: float | None = None
    iqr: float | None = None
    zero_count: int = 0
    negative_count: int = 0
    earliest_date: Any | None = None
    latest_date: Any | None = None
    date_span_days: int | None = None
    distinct_years: int = 0
    distinct_months: int = 0
    duplicate_count: int = 0
    duplicate_percentage: float = 0.0
    true_count: int = 0
    false_count: int = 0
    true_percentage: float = 0.0
    false_percentage: float = 0.0
    average_length: float | None = None
    minimum_length: int | None = None
    maximum_length: int | None = None
    top_values: list[dict[str, Any]] = Field(default_factory=list)
    least_frequent_values: list[dict[str, Any]] = Field(default_factory=list)


class ColumnProfileResult(BaseModel):
    """Structured output for a column-profile request."""

    request: ColumnProfileRequest
    profile: CommonColumnProfile
    summary: str
    table_columns: list[str] = Field(default_factory=list)
    table_rows: list[dict[str, Any]] = Field(default_factory=list)
    chart_type: str | None = None
    chart_rows: list[dict[str, Any]] = Field(default_factory=list)
    caution: str | None = None
    recommended_next_step: str | None = None
    warnings: list[str] = Field(default_factory=list)


class GroupedExtremaWinner(BaseModel):
    """One winner row for a grouped-extrema calculation."""

    primary_group: str
    secondary_group: str
    value: float
    rank: int = 1
    is_tie: bool = False
    tie_count: int = 1


class GroupedExtremaResult(BaseModel):
    """Structured grouped-extrema result with table and chart rows."""

    primary_group_column: str
    secondary_group_column: str
    metric_column: str
    aggregation: str
    extremum: Literal["max", "min"]
    winners: list[GroupedExtremaWinner] = Field(default_factory=list)
    table_rows: list[dict[str, Any]] = Field(default_factory=list)
    chart_rows: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AgentPlan(BaseModel):
    """Validated decision selecting one approved analytical tool."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    chart_spec: ChartSpec | None = None
    clarification: str | None = None
    safe_code: str | None = None
    response_mode: str = "full"


class ToolResult(BaseModel):
    """Structured verified output from an approved tool."""

    tool_name: str
    success: bool = True
    summary: str
    data: Any = None
    warnings: list[str] = Field(default_factory=list)
    execution_seconds: float = 0.0


class MetricSummary(BaseModel):
    """One safely rendered metric summary block for chat narratives."""

    metric_label: str
    total_value: str | None = None
    highest_period: str | None = None
    highest_value: str | None = None
    lowest_period: str | None = None
    lowest_value: str | None = None
    first_period: str | None = None
    first_value: str | None = None
    latest_period: str | None = None
    latest_value: str | None = None
    trend_text: str | None = None


class ChatNarrativeResponse(BaseModel):
    """Structured prose rendered deterministically in the chat UI."""

    summary: str
    key_findings: list[str] = Field(default_factory=list)
    metric_summaries: list[MetricSummary] = Field(default_factory=list)
    caution: str | None = None
    recommended_next_step: str | None = None


class AgentResponse(BaseModel):
    """Complete analysis response stored in history."""

    question: str
    answer: str
    narrative: ChatNarrativeResponse | None = None
    plan: AgentPlan
    result: ToolResult | None = None
    chart_spec: ChartSpec | None = None
    chart_data: list[dict[str, Any]] = Field(default_factory=list)
    suggested_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    total_seconds: float = 0.0
    interpretation_seconds: float = 0.0
    tool_seconds: float = 0.0
    explanation_seconds: float = 0.0
