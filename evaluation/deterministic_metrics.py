"""Deterministic per-answer and benchmark metrics."""

from __future__ import annotations

import math
import re
from typing import Any

from agent.schemas import AgentResponse
from agent.tool_registry import TOOL_REGISTRY
from evaluation.evaluation_models import EvaluationScores


def numeric_close(actual: float, expected: float) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.01, abs_tol=1e-9)


def overall_evaluation_score(
    correctness: float,
    faithfulness: float,
    relevancy: float,
    completeness: float,
    tool_chart_accuracy: float,
    execution_error_handling: float,
) -> float:
    """Apply the required normalized weighted formula."""
    return (
        0.35 * correctness
        + 0.20 * faithfulness
        + 0.15 * relevancy
        + 0.10 * completeness
        + 0.10 * tool_chart_accuracy
        + 0.10 * execution_error_handling
    )


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _result_values(data: Any) -> list[str]:
    if isinstance(data, dict):
        return [str(value) for value in data.values()]
    if isinstance(data, list):
        return [str(value) for row in data[:10] for value in (row.values() if isinstance(row, dict) else [row])]
    return [str(data)] if data is not None else []


def _negative_intent_unmet(response: AgentResponse) -> bool:
    """Detect when a negative/loss-making request returned unfiltered positive rows."""
    question = response.question.lower()
    if not re.search(r"\b(?:negative|loss-making|loss making)\b", question):
        return False
    if re.search(r"\b(?:include|including|contains?|with)\s+negative\b", question):
        return False
    if not response.result or not isinstance(response.result.data, list):
        return False
    operation = response.plan.arguments.get("operation")
    if operation in {"negative_groups", "negative_record_percentage", "loss_by_group"}:
        return False
    metric = (
        response.plan.arguments.get("metric_column")
        or response.plan.arguments.get("value_column")
    )
    if not metric:
        return False
    numeric_values = []
    for row in response.result.data:
        if isinstance(row, dict) and metric in row:
            try:
                numeric_values.append(float(row[metric]))
            except (TypeError, ValueError):
                continue
    return bool(numeric_values) and any(value > 0 for value in numeric_values)


def evaluate_response(response: AgentResponse) -> EvaluationScores:
    """Evaluate evidence contained in a completed response without inventing a gold answer."""
    result = response.result
    notes = ["Scores use verified execution evidence; no external gold answer was supplied."]
    execution_success = 1.0 if result and result.success else 0.0
    correctness = execution_success
    tool_accuracy = 1.0 if response.plan.tool_name in TOOL_REGISTRY else 0.0
    if response.plan.clarification:
        correctness = 1.0
        execution_success = 1.0
        tool_accuracy = 1.0
    chart_accuracy = evaluate_chart_correctness(response.chart_spec, response.chart_data)
    question_tokens = _tokens(response.question)
    answer_tokens = _tokens(response.answer)
    relevancy = len(question_tokens & answer_tokens) / max(len(question_tokens), 1)
    relevancy = min(1.0, 0.5 + relevancy)
    result_values = _result_values(result.data if result else None)
    represented = sum(value in response.answer for value in result_values)
    faithfulness = 1.0 if not result_values else min(1.0, 0.7 + 0.3 * represented / len(result_values))
    completeness_parts = [
        bool(response.answer),
        result is not None or response.plan.clarification is not None,
        bool(response.plan.safe_code) or response.plan.clarification is not None,
        bool(response.suggested_questions) or response.plan.clarification is not None,
    ]
    completeness = sum(completeness_parts) / len(completeness_parts)
    error_handling = 1.0 if response.answer and (result is not None or response.plan.clarification) else 0.0
    if _negative_intent_unmet(response):
        correctness = min(correctness, 0.35)
        relevancy = min(relevancy, 0.60)
        tool_accuracy = min(tool_accuracy, 0.50)
        notes.append(
            "The question requested negative values, but the executed result included positive metric rows."
        )
    tool_chart = (tool_accuracy + chart_accuracy) / 2
    execution_error = (execution_success + error_handling) / 2
    overall = overall_evaluation_score(
        correctness, faithfulness, relevancy, completeness, tool_chart, execution_error
    )
    return EvaluationScores(
        correctness=correctness,
        faithfulness=faithfulness,
        relevancy=relevancy,
        completeness=completeness,
        tool_accuracy=tool_accuracy,
        chart_accuracy=chart_accuracy,
        execution_success=execution_success,
        error_handling=error_handling,
        overall_score=overall,
        response_time_seconds=response.total_seconds,
        notes=notes,
    )


def evaluate_chart_correctness(chart_spec, chart_data: list[dict[str, Any]]) -> float:
    """Score chart metadata and verified plotted data using required components."""
    if chart_spec is None:
        return 1.0
    chart_type = 1.0 if chart_spec.chart_type in {"bar", "line", "scatter", "histogram", "box", "heatmap", "pie"} else 0.0
    required_axes = {
        "bar": bool(chart_spec.x and chart_spec.y),
        "line": bool(chart_spec.x and chart_spec.y),
        "scatter": bool(chart_spec.x and chart_spec.y),
        "histogram": bool(chart_spec.x),
        "box": bool(chart_spec.y),
        "heatmap": True,
        "pie": bool(chart_spec.x),
    }
    column_selection = 1.0 if required_axes.get(chart_spec.chart_type, False) else 0.0
    aggregation = 1.0 if (
        chart_spec.chart_type in {"scatter", "histogram", "box", "heatmap"}
        or chart_spec.aggregation is not None
    ) else 0.75
    data_values = 1.0 if chart_data else 0.0
    labels = 1.0 if chart_spec.title and (
        chart_spec.chart_type == "heatmap" or chart_spec.x or chart_spec.y
    ) else 0.0
    score = 0.20 * chart_type + 0.25 * column_selection + 0.25 * aggregation + 0.20 * data_values + 0.10 * labels
    return max(0.0, min(1.0, round(score, 10)))


def compare_expected_result(actual: Any, expected: Any) -> float:
    """Compare nested expected data using exact values and numeric tolerance."""
    if expected is None:
        return 1.0
    if isinstance(expected, dict) and isinstance(actual, dict):
        if not expected:
            return 1.0
        matches = []
        for key, expected_value in expected.items():
            if key not in actual:
                matches.append(False)
            elif isinstance(expected_value, (int, float)) and isinstance(actual[key], (int, float)):
                matches.append(numeric_close(actual[key], expected_value))
            else:
                matches.append(actual[key] == expected_value)
        return sum(matches) / len(matches)
    return 1.0 if actual == expected else 0.0
