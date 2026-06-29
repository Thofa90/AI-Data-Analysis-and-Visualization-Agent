"""Rubric for the optional Ollama qualitative judge."""

JUDGE_PROMPT = """Score relevancy, faithfulness, completeness, clarity, and helpfulness from 1 to 5.
Use the verified result as the sole factual reference. Return JSON only with those five keys and a notes list.
Faithfulness must be judged against the verified result, never against general intuition."""
