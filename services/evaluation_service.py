"""Per-answer evaluation orchestration with optional Ollama rubric judging."""

from __future__ import annotations

from agent.schemas import AgentResponse
from config.settings import Settings
from evaluation.deterministic_metrics import evaluate_response, overall_evaluation_score
from evaluation.llm_judge import judge_answer
from evaluation.evaluation_models import EvaluationScores


def evaluate_answer(
    response: AgentResponse,
    settings: Settings,
    model_name: str,
    use_llm_judge: bool = False,
) -> EvaluationScores:
    """Calculate deterministic evidence scores and optionally run the LLM judge."""
    scores = evaluate_response(response)
    if not use_llm_judge or response.result is None:
        return scores
    try:
        judged = judge_answer(
            response.question,
            response.result.data,
            response.result.tool_name,
            response.chart_spec.model_dump(mode="json") if response.chart_spec else None,
            response.answer,
            settings,
            model_name,
        )
        normalized = {
            key: max(0.0, min(1.0, float(judged[key]) / 5))
            for key in ("relevancy", "faithfulness", "completeness")
        }
        scores.relevancy = normalized["relevancy"]
        scores.faithfulness = normalized["faithfulness"]
        scores.completeness = normalized["completeness"]
        scores.judge_mode = "ollama"
        scores.notes = [str(note) for note in judged.get("notes", [])]
        scores.overall_score = overall_evaluation_score(
            scores.correctness,
            scores.faithfulness,
            scores.relevancy,
            scores.completeness,
            (scores.tool_accuracy + scores.chart_accuracy) / 2,
            (scores.execution_success + scores.error_handling) / 2,
        )
    except Exception:
        scores.notes.append("Ollama judge was unavailable; deterministic evidence scores were retained.")
    return scores
