"""Optional on-demand qualitative Ollama judge."""

from __future__ import annotations

import json
from typing import Any

from langchain_ollama import ChatOllama

from config.settings import Settings
from evaluation.judge_prompts import JUDGE_PROMPT


def judge_answer(
    question: str,
    verified_result: Any,
    tool_name: str,
    chart_metadata: Any,
    answer: str,
    settings: Settings,
    model_name: str,
) -> dict[str, Any]:
    """Run the qualitative judge only when explicitly requested."""
    model = ChatOllama(
        base_url=settings.ollama_base_url,
        model=model_name,
        temperature=0,
        num_ctx=settings.ollama_num_ctx,
        format="json",
        sync_client_kwargs={"timeout": settings.ollama_timeout_seconds},
    )
    payload = {
        "question": question,
        "verified_result": verified_result,
        "tool": tool_name,
        "chart_metadata": chart_metadata,
        "answer": answer,
    }
    response = model.invoke(f"{JUDGE_PROMPT}\nInput: {json.dumps(payload, default=str)}")
    return json.loads(str(response.content))
