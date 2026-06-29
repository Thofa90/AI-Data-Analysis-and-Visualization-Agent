"""Structured evaluation results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationScores(BaseModel):
    """Normalized 0-1 answer quality scores."""

    correctness: float
    faithfulness: float
    relevancy: float
    completeness: float
    tool_accuracy: float
    chart_accuracy: float
    execution_success: float
    error_handling: float
    overall_score: float
    response_time_seconds: float = 0.0
    judge_mode: str = "deterministic"
    notes: list[str] = Field(default_factory=list)


class BenchmarkCaseResult(BaseModel):
    """One benchmark case with expected-versus-actual metadata."""

    case_id: str
    category: str
    passed: bool
    question: str
    expected_tool: str
    actual_tool: str
    tool_score: float
    result_score: float
    elapsed_seconds: float
    failure_reason: str | None = None
