"""Deterministic benchmark execution over JSON test cases."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from agent.data_agent import deterministic_plan
from agent.tool_registry import execute_tool
from evaluation.deterministic_metrics import compare_expected_result
from evaluation.evaluation_models import BenchmarkCaseResult
from services.dataset_profiler import profile_dataset


def load_test_cases(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def run_benchmark_cases(
    cases: list[dict],
    sample_data_directory: str | Path,
) -> list[BenchmarkCaseResult]:
    """Run each case with deterministic planning and approved tools."""
    directory = Path(sample_data_directory)
    cache: dict[str, tuple[pd.DataFrame, object]] = {}
    results = []
    for case in cases:
        started = perf_counter()
        dataset_name = case["dataset"]
        if dataset_name not in cache:
            dataframe = pd.read_csv(directory / dataset_name)
            cache[dataset_name] = (dataframe, profile_dataset(dataframe))
        dataframe, profile = cache[dataset_name]
        plan = deterministic_plan(case["question"], profile)
        expected_tool = case["expected_tool"]
        actual_tool = plan.tool_name or "clarification"
        tool_score = 1.0 if actual_tool == expected_tool else 0.0
        result_score = 0.0
        failure = None
        if plan.clarification:
            result_score = 1.0 if expected_tool == "clarification" else 0.0
        elif plan.tool_name:
            try:
                output = execute_tool(dataframe, plan.tool_name, plan.arguments)
                result_score = compare_expected_result(output.data, case.get("expected_result"))
            except ValueError as exc:
                failure = str(exc)
        passed = tool_score == 1.0 and result_score >= 0.99
        if not passed and failure is None:
            failure = f"Expected {expected_tool}, received {plan.tool_name or 'clarification'}."
        results.append(BenchmarkCaseResult(
            case_id=case["id"],
            category=case["category"],
            passed=passed,
            question=case["question"],
            expected_tool=expected_tool,
            actual_tool=actual_tool,
            tool_score=tool_score,
            result_score=result_score,
            elapsed_seconds=perf_counter() - started,
            failure_reason=failure,
        ))
    return results
