"""Per-answer and benchmark evaluation dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from evaluation.benchmark_runner import load_test_cases, run_benchmark_cases
from evaluation.evaluation_models import EvaluationScores
from services.export_service import history_to_json


def _average(history: list[dict], field: str) -> float:
    return sum(item.get(field, 0) for item in history) / max(len(history), 1)


def render_evaluation_page() -> None:
    st.title("Evaluation Dashboard")
    history = st.session_state.evaluation_history
    benchmark = st.session_state.benchmark_results
    if not history and not benchmark:
        st.info("No evaluations have been run yet. Evaluate an answer or run the benchmark suite.")
    if history:
        cards = st.columns(4)
        cards[0].metric("Overall Evaluation Score", f"{_average(history, 'overall_score'):.0%}")
        cards[1].metric("Correctness", f"{_average(history, 'correctness'):.0%}")
        cards[2].metric("Faithfulness", f"{_average(history, 'faithfulness'):.0%}")
        cards[3].metric("Relevancy", f"{_average(history, 'relevancy'):.0%}")
        cards = st.columns(4)
        cards[0].metric("Completeness", f"{_average(history, 'completeness'):.0%}")
        cards[1].metric("Tool Accuracy", f"{_average(history, 'tool_accuracy'):.0%}")
        cards[2].metric("Chart Accuracy", f"{_average(history, 'chart_accuracy'):.0%}")
        cards[3].metric("Avg. Response Time", f"{_average(history, 'response_time_seconds'):.2f}s")
        st.markdown("### Recent Answer Evaluations")
        st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)
    st.markdown("### Benchmark Suite")
    if st.button("Run 30-case deterministic benchmark", type="primary"):
        cases = load_test_cases(Path("evaluation/test_cases.json"))
        with st.spinner("Running benchmark cases..."):
            results = run_benchmark_cases(cases, Path("sample_data"))
        st.session_state.benchmark_results = [result.model_dump(mode="json") for result in results]
        st.rerun()
    if benchmark:
        benchmark_frame = pd.DataFrame(benchmark)
        passed = int(benchmark_frame["passed"].sum())
        metrics = st.columns(3)
        metrics[0].metric("Passed Test Cases", f"{passed}/{len(benchmark_frame)}")
        metrics[1].metric("Execution Success Rate", f"{passed / len(benchmark_frame):.0%}")
        metrics[2].metric("Average Response Time", f"{benchmark_frame['elapsed_seconds'].mean():.3f}s")
        st.markdown("### Score by Query Category")
        category = benchmark_frame.groupby("category")["passed"].mean().reset_index(name="pass_rate")
        st.dataframe(category, width="stretch", hide_index=True)
        st.markdown("### Expected vs. Actual")
        st.dataframe(benchmark_frame, width="stretch", hide_index=True)
        st.download_button(
            "Download evaluation report",
            history_to_json([*history, *benchmark]),
            file_name="evaluation_report.json",
            mime="application/json",
        )
