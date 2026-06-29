"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


@dataclass(frozen=True)
class Settings:
    """Runtime settings with conservative local defaults."""

    app_name: str = "Data Insight Agent"
    app_subtitle: str = "AI-Powered Data Analyst"
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "qwen2.5:1.5b"
    fallback_model: str = "llama3.2:1b"
    ollama_temperature: float = 0.1
    ollama_num_ctx: int = 4096
    ollama_timeout_seconds: int = 120
    max_upload_size_mb: int = 100
    max_sample_rows_for_llm: int = 5
    app_env: str = "development"
    log_level: str = "INFO"

    @property
    def supported_models(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.default_model, self.fallback_model)))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process."""
    _load_dotenv()
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        default_model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
        fallback_model=os.getenv("OLLAMA_FALLBACK_MODEL", "llama3.2:1b"),
        ollama_temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
        ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "4096")),
        ollama_timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
        max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "100")),
        max_sample_rows_for_llm=int(os.getenv("MAX_SAMPLE_ROWS_FOR_LLM", "5")),
        app_env=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
