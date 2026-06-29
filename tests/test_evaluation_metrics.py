"""Tests for required evaluation formulas and evidence scoring."""

from __future__ import annotations

from agent.schemas import AgentPlan, AgentResponse, ToolResult
from evaluation.deterministic_metrics import (
    evaluate_chart_correctness,
    evaluate_response,
    numeric_close,
    overall_evaluation_score,
)
from services.chart_service import ChartSpec


def test_overall_evaluation_formula() -> None:
    score = overall_evaluation_score(1, 1, 1, 1, 1, 1)
    assert score == 1
    assert numeric_close(100.0, 100.5)


def test_response_evaluation_uses_execution_evidence() -> None:
    response = AgentResponse(
        question="How many rows are there?",
        answer="The dataset has 10 rows.",
        plan=AgentPlan(tool_name="inspect_dataset", safe_code="df.shape"),
        result=ToolResult(
            tool_name="inspect_dataset",
            summary="The dataset has 10 rows.",
            data={"rows": 10, "columns": 2},
        ),
        suggested_questions=["Show missing values"],
        total_seconds=0.2,
    )
    scores = evaluate_response(response)
    assert scores.correctness == 1
    assert scores.tool_accuracy == 1
    assert 0 <= scores.overall_score <= 1


def test_chart_correctness_uses_spec_and_verified_data() -> None:
    spec = ChartSpec(
        chart_type="bar", x="Region", y="Sales", aggregation="sum", title="Sales by Region"
    )
    assert evaluate_chart_correctness(spec, [{"Region": "West", "Sales": 10}]) == 1
