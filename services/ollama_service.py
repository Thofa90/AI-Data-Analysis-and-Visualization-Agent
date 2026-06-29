"""Lightweight Ollama availability and model discovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class OllamaStatus:
    """Current local Ollama status."""

    online: bool
    models: tuple[str, ...] = ()
    message: str = "Ollama is unavailable."


def check_ollama(base_url: str, timeout_seconds: float = 1.5) -> OllamaStatus:
    """Check Ollama and return installed model names without raising."""
    request = Request(f"{base_url.rstrip('/')}/api/tags", headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = tuple(
            item.get("name", "")
            for item in payload.get("models", [])
            if isinstance(item, dict) and item.get("name")
        )
        return OllamaStatus(True, models, "Ollama is online.")
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
        return OllamaStatus(False, (), "Start Ollama to enable AI features.")
