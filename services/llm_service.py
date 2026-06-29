"""Bounded Ollama integration for planning and explaining verified results."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from langchain_ollama import ChatOllama

from agent.prompts import (
    AREA_INSIGHT_SYSTEM_PROMPT,
    BOX_PLOT_INSIGHT_SYSTEM_PROMPT,
    CHART_INSIGHT_SYSTEM_PROMPT,
    CIRCLE_VIEW_INSIGHT_SYSTEM_PROMPT,
    CORRELATION_HEATMAP_INSIGHT_SYSTEM_PROMPT,
    DUAL_COMBINATION_INSIGHT_SYSTEM_PROMPT,
    DUAL_LINE_INSIGHT_SYSTEM_PROMPT,
    EXPLANATION_SYSTEM_PROMPT,
    GROUPED_BAR_INSIGHT_SYSTEM_PROMPT,
    HISTOGRAM_INSIGHT_SYSTEM_PROMPT,
    PERIOD_OVER_PERIOD_CHANGE_INSIGHT_SYSTEM_PROMPT,
    PIE_CHART_INSIGHT_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SCATTER_INSIGHT_SYSTEM_PROMPT,
    SINGLE_BAR_INSIGHT_SYSTEM_PROMPT,
    SINGLE_LINE_INSIGHT_SYSTEM_PROMPT,
    SORTED_PERCENTAGE_BAR_INSIGHT_SYSTEM_PROMPT,
    SYMBOL_MAP_INSIGHT_SYSTEM_PROMPT,
    TREEMAP_INSIGHT_SYSTEM_PROMPT,
)
from agent.schemas import AgentPlan, ToolResult
from config.settings import Settings
from services.profile_models import DatasetProfile
from services.chart_insight_service import (
    ChartEvidence,
    ChartInsight,
    BoxPlotEvidence,
    CircleViewEvidence,
    CorrelationHeatmapEvidence,
    GroupedBarEvidence,
    HistogramEvidence,
    PeriodOverPeriodChangeEvidence,
    PieChartEvidence,
    SingleBarEvidence,
    SingleLineEvidence,
    SingleAreaEvidence,
    SortedPercentageBarEvidence,
    ScatterEvidence,
    StackedAreaEvidence,
    SymbolMapEvidence,
    TreemapEvidence,
    DualCombinationEvidence,
    DualLineEvidence,
)


def _model(settings: Settings, model_name: str) -> ChatOllama:
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=model_name,
        temperature=settings.ollama_temperature,
        num_ctx=settings.ollama_num_ctx,
        client_kwargs={"timeout": settings.ollama_timeout_seconds},
    )


def structured_dataset_context(
    dataset_name: str,
    profile: DatasetProfile,
    sample_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build limited context that never contains the complete dataset."""
    return {
        "dataset_name": dataset_name,
        "columns": [
            {"name": column.name, "type": column.kind, "dtype": column.pandas_dtype}
            for column in profile.columns
        ],
        "row_count": profile.row_count,
        "missing_percentage": profile.missing_percentage,
        "sample_rows": sample_rows,
    }


def plan_with_ollama(
    question: str,
    context: dict[str, Any],
    tool_names: list[str],
    settings: Settings,
    model_name: str,
) -> tuple[AgentPlan, float]:
    """Ask Ollama for a JSON plan; callers must retain a deterministic fallback."""
    started = perf_counter()
    model = _model(settings, model_name)
    prompt = (
        f"{PLANNER_SYSTEM_PROMPT}\nApproved tools: {tool_names}\n"
        f"Dataset context: {json.dumps(context, default=str)}\nQuestion: {question}"
    )
    response = model.invoke(prompt)
    content = str(response.content).strip().removeprefix("```json").removesuffix("```").strip()
    return AgentPlan.model_validate_json(content), perf_counter() - started


def explain_with_ollama(
    question: str,
    result: ToolResult,
    settings: Settings,
    model_name: str,
) -> tuple[str, float]:
    """Explain only the verified tool output."""
    started = perf_counter()
    model = _model(settings, model_name)
    prompt = (
        f"{EXPLANATION_SYSTEM_PROMPT}\nQuestion: {question}\n"
        f"Tool: {result.tool_name}\nVerified result: {result.model_dump_json()}"
    )
    response = model.invoke(prompt)
    return str(response.content).strip(), perf_counter() - started


def explain_chart_insight_with_ollama(
    evidence: ChartEvidence,
    fallback: ChartInsight,
    settings: Settings,
    model_name: str,
) -> tuple[ChartInsight, float]:
    """Ask Ollama to explain calculated chart evidence with validated JSON."""
    started = perf_counter()
    model = _model(settings, model_name)
    if isinstance(evidence, CircleViewEvidence):
        system_prompt = CIRCLE_VIEW_INSIGHT_SYSTEM_PROMPT
    elif isinstance(evidence, CorrelationHeatmapEvidence):
        system_prompt = CORRELATION_HEATMAP_INSIGHT_SYSTEM_PROMPT
    elif isinstance(evidence, PieChartEvidence):
        system_prompt = PIE_CHART_INSIGHT_SYSTEM_PROMPT
    elif isinstance(evidence, TreemapEvidence):
        system_prompt = TREEMAP_INSIGHT_SYSTEM_PROMPT
    elif isinstance(evidence, SymbolMapEvidence):
        system_prompt = SYMBOL_MAP_INSIGHT_SYSTEM_PROMPT
    elif isinstance(evidence, SortedPercentageBarEvidence):
        system_prompt = SORTED_PERCENTAGE_BAR_INSIGHT_SYSTEM_PROMPT
    elif isinstance(evidence, PeriodOverPeriodChangeEvidence):
        system_prompt = PERIOD_OVER_PERIOD_CHANGE_INSIGHT_SYSTEM_PROMPT
    elif isinstance(evidence, HistogramEvidence):
        system_prompt = HISTOGRAM_INSIGHT_SYSTEM_PROMPT
    elif isinstance(evidence, BoxPlotEvidence):
        system_prompt = BOX_PLOT_INSIGHT_SYSTEM_PROMPT
    elif isinstance(evidence, DualCombinationEvidence):
        system_prompt = DUAL_COMBINATION_INSIGHT_SYSTEM_PROMPT
    elif isinstance(evidence, ScatterEvidence):
        system_prompt = SCATTER_INSIGHT_SYSTEM_PROMPT
    elif isinstance(evidence, DualLineEvidence):
        system_prompt = DUAL_LINE_INSIGHT_SYSTEM_PROMPT
    elif isinstance(evidence, (SingleAreaEvidence, StackedAreaEvidence)):
        system_prompt = AREA_INSIGHT_SYSTEM_PROMPT
    elif isinstance(evidence, SingleLineEvidence):
        system_prompt = SINGLE_LINE_INSIGHT_SYSTEM_PROMPT
    elif isinstance(evidence, GroupedBarEvidence) or evidence.chart_type == "grouped_bar":
        system_prompt = GROUPED_BAR_INSIGHT_SYSTEM_PROMPT
    elif isinstance(evidence, SingleBarEvidence) or (
        evidence.chart_type == "bar" and not evidence.group_column and not evidence.color_column
    ):
        system_prompt = SINGLE_BAR_INSIGHT_SYSTEM_PROMPT
    else:
        system_prompt = CHART_INSIGHT_SYSTEM_PROMPT
    prompt = (
        f"{system_prompt}\n"
        f"Evidence: {evidence.model_dump_json()}"
    )
    for attempt in range(2):
        response = model.invoke(prompt)
        content = str(response.content).strip().removeprefix("```json").removesuffix("```").strip()
        try:
            insight = ChartInsight.model_validate_json(content)
            return insight.model_copy(update={"evidence": evidence}), perf_counter() - started
        except Exception:
            prompt = (
                f"{system_prompt}\n"
                "The previous response was invalid. Return only valid JSON "
                "matching the schema, using only this evidence.\n"
                f"Evidence: {evidence.model_dump_json()}"
            )
            if attempt == 1:
                break
    return fallback, perf_counter() - started
