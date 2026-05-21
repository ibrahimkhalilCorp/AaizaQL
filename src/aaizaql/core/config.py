"""
AAIZAQL.core.config
─────────────────
All configuration for the library via pydantic-settings.
Values are read from environment variables or a .env file.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    CLAUDE = "claude"
    OPENAI = "openai"
    OLLAMA = "ollama"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    PERPLEXITY = "perplexity"
    GEMINI = "gemini"
    MISTRAL = "mistral"


class VectorStoreBackend(StrEnum):
    CHROMA = "chroma"
    QDRANT = "qdrant"


class Settings(BaseSettings):
    """
    Master settings for AAIZAQL.

    All fields can be overridden via environment variables prefixed with AAIZAQL_.
    Example: AAIZAQL_LLM_PROVIDER=openai

    Or by passing kwargs directly to QueryEngine(...).
    """

    model_config = SettingsConfigDict(
        env_prefix="AAIZAQL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    llm_provider: LLMProvider = Field(
        default=LLMProvider.CLAUDE,
        description="Which LLM provider to use for SQL generation.",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key (required when llm_provider=claude).",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key (required when llm_provider=openai).",
    )
    groq_api_key: SecretStr | None = Field(
        default=None,
        description="Groq API key (required when llm_provider=groq). Free at console.groq.com",
    )
    # AFTER
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model string. Options: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768, gemma2-9b-it",  # noqa: E501
    )
    deepseek_api_key: SecretStr | None = Field(
        default=None,
        description="DeepSeek API key (required when llm_provider=deepseek). Get at platform.deepseek.com",
    )
    deepseek_model: str = Field(
        default="deepseek-chat",
        description="DeepSeek model string. Options: deepseek-chat, deepseek-reasoner",
    )
    perplexity_api_key: SecretStr | None = Field(
        default=None,
        description="Perplexity API key (required when llm_provider=perplexity). Get at perplexity.ai/settings/api",
    )
    perplexity_model: str = Field(
        default="sonar",
        description="Perplexity model string. Options: sonar, sonar-pro, sonar-reasoning, sonar-reasoning-pro",
    )
    gemini_api_key: SecretStr | None = Field(
        default=None,
        description="Google Gemini API key (required when llm_provider=gemini). Get at aistudio.google.com",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model string. Options: gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash",
    )
    mistral_api_key: SecretStr | None = Field(
        default=None,
        description="Mistral API key (required when llm_provider=mistral). Get at console.mistral.ai",
    )
    mistral_model: str = Field(
        default="mistral-large-latest",
        description="Mistral model string. Options: mistral-large-latest, mistral-small-latest, codestral-latest",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama server.",
    )
    ollama_model: str = Field(
        default="llama3",
        description="Model name to use when llm_provider=ollama.",
    )
    claude_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Claude model string.",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model string.",
    )
    llm_max_tokens: int = Field(default=1024, ge=256, le=8192)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Seconds before an LLM API call is cancelled. Applies to all providers.",
    )

    # ── Vector Store ─────────────────────────────────────────────────────────
    vector_store: VectorStoreBackend = Field(
        default=VectorStoreBackend.CHROMA,
        description="Vector store backend for RAG.",
    )
    chroma_persist_dir: str = Field(
        default=".aaizaql_chroma",
        description="Local directory for ChromaDB persistence.",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL (used when vector_store=qdrant).",
    )
    qdrant_api_key: SecretStr | None = Field(default=None)
    vector_store_namespace: str = Field(
        default="default",
        description="Namespace / collection prefix in the vector store.",
    )
    schema_top_k: int = Field(
        default=5,
        description="Number of schema chunks to retrieve per query.",
    )
    examples_top_k: int = Field(
        default=3,
        description="Number of Q→SQL pair examples to retrieve per query.",
    )

    # ── Self-Correction ──────────────────────────────────────────────────────
    max_self_correction_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="How many times to retry SQL generation after a DB error.",
    )

    # ── Security ─────────────────────────────────────────────────────────────
    allowed_sql_operations: list[str] = Field(
        default=["SELECT", "WITH"],
        description="Whitelist of SQL statement types allowed to execute.",
    )
    enable_injection_detection: bool = Field(
        default=True,
        description="Scan user questions for prompt injection patterns.",
    )

    # ── Context Memory ───────────────────────────────────────────────────────
    session_history_limit: int = Field(
        default=10,
        description="Max number of previous Q&A turns to include in context.",
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    audit_log_file: str | None = Field(
        default=None,
        description="Write structured audit log to this file path.",
    )

    # ── Development ──────────────────────────────────────────────────────────
    debug: bool = Field(default=False)

    @field_validator("allowed_sql_operations", mode="before")
    @classmethod
    def uppercase_ops(cls, v: list[str]) -> list[str]:
        return [op.upper() for op in v]


def make_settings(**overrides: object) -> Settings:
    """
    Create a fresh Settings instance from environment variables, then apply
    ``overrides`` on top.

    Use this instead of the module-level ``settings`` singleton whenever you
    need per-engine configuration — particularly in tests or multi-tenant
    servers where different engines need different settings in the same process.

    Example
    -------
    ::

        s = make_settings(llm_provider="groq", groq_api_key="gsk_...", llm_timeout_seconds=60)
        engine = QueryEngine(settings=s)
    """
    return Settings(**overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Backwards-compatible module-level singleton.
#
# Importing this directly is DEPRECATED for library code.  It is read once at
# import time, so a second QueryEngine with different env vars in the same
# process will see stale values.
#
# Use ``make_settings(**overrides)`` or pass ``settings=`` to QueryEngine
# instead.
# ---------------------------------------------------------------------------
settings = Settings()
