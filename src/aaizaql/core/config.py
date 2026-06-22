"""
aaizaql.core.config
───────────────────
Library-wide configuration backed by `pydantic-settings
<https://docs.pydantic.dev/latest/concepts/pydantic_settings/>`_.

Values are resolved in this order (highest priority first):

1. Keyword arguments passed directly to :class:`Settings` or
   :func:`make_settings`.
2. Environment variables prefixed with ``AAIZAQL_`` (case-insensitive).
3. A ``.env`` file in the current working directory.
4. Field defaults defined below.

Quick start
───────────
**.env file**::

    AAIZAQL_LLM_PROVIDER=openai
    AAIZAQL_OPENAI_API_KEY=sk-...
    AAIZAQL_LLM_TIMEOUT_SECONDS=60

**In code**::

    from aaizaql.core.config import make_settings

    settings = make_settings(llm_provider="groq", groq_api_key="gsk_...")

For multi-tenant or per-engine configuration, always use :func:`make_settings`
rather than the module-level :data:`settings` singleton.
"""

from enum import StrEnum
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "LLMProvider",
    "VectorStoreBackend",
    "Settings",
    "make_settings",
    "settings",
]


class LLMProvider(StrEnum):
    """Supported LLM providers for SQL generation."""

    CLAUDE = "claude"
    OPENAI = "openai"
    OLLAMA = "ollama"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    PERPLEXITY = "perplexity"
    GEMINI = "gemini"
    MISTRAL = "mistral"


class VectorStoreBackend(StrEnum):
    """Supported vector store backends for RAG schema retrieval."""

    CHROMA = "chroma"
    QDRANT = "qdrant"


class Settings(BaseSettings):
    """Master configuration for AAIZAQL.

    All fields may be set via environment variables prefixed with
    ``AAIZAQL_`` (e.g. ``AAIZAQL_LLM_PROVIDER=openai``) or by passing
    keyword arguments to :func:`make_settings`.

    Prefer :func:`make_settings` over direct instantiation — it makes
    per-engine overrides explicit and keeps the module-level singleton
    unmodified.
    """

    model_config = SettingsConfigDict(
        env_prefix="AAIZAQL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────

    llm_provider: LLMProvider = Field(
        default=LLMProvider.CLAUDE,
        description="Which LLM provider to use for SQL generation.",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key. Required when llm_provider=claude.",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key. Required when llm_provider=openai.",
    )
    groq_api_key: SecretStr | None = Field(
        default=None,
        description="Groq API key. Required when llm_provider=groq.",
    )
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description=(
            "Groq model string. "
            "Options: llama-3.3-70b-versatile, llama-3.1-8b-instant, "
            "mixtral-8x7b-32768, gemma2-9b-it."
        ),
    )
    deepseek_api_key: SecretStr | None = Field(
        default=None,
        description="DeepSeek API key. Required when llm_provider=deepseek.",
    )
    deepseek_model: str = Field(
        default="deepseek-chat",
        description="DeepSeek model string. Options: deepseek-chat, deepseek-reasoner.",
    )
    perplexity_api_key: SecretStr | None = Field(
        default=None,
        description="Perplexity API key. Required when llm_provider=perplexity.",
    )
    perplexity_model: str = Field(
        default="sonar",
        description=(
            "Perplexity model string. "
            "Options: sonar, sonar-pro, sonar-reasoning, sonar-reasoning-pro."
        ),
    )
    gemini_api_key: SecretStr | None = Field(
        default=None,
        description="Google Gemini API key. Required when llm_provider=gemini.",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description=(
            "Gemini model string. " "Options: gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash."
        ),
    )
    mistral_api_key: SecretStr | None = Field(
        default=None,
        description="Mistral API key. Required when llm_provider=mistral.",
    )
    mistral_model: str = Field(
        default="mistral-large-latest",
        description=(
            "Mistral model string. "
            "Options: mistral-large-latest, mistral-small-latest, codestral-latest."
        ),
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the local Ollama server.",
    )
    ollama_model: str = Field(
        default="llama3",
        description="Ollama model name. Used when llm_provider=ollama.",
    )
    claude_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Claude model string. Used when llm_provider=claude.",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model string. Used when llm_provider=openai.",
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=256,
        le=8192,
        description="Maximum tokens the LLM may generate per response.",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Sampling temperature for LLM responses. "
            "0.0 = deterministic; higher values increase creativity."
        ),
    )
    llm_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Seconds before an LLM API call is cancelled. Applies to all providers.",
    )

    # ── Vector Store ──────────────────────────────────────────────────────────

    vector_store: VectorStoreBackend = Field(
        default=VectorStoreBackend.CHROMA,
        description="Vector store backend used for RAG schema retrieval.",
    )
    chroma_persist_dir: str = Field(
        default=".aaizaql_chroma",
        description="Local directory for ChromaDB persistence.",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL. Used when vector_store=qdrant.",
    )
    qdrant_api_key: SecretStr | None = Field(
        default=None,
        description="Qdrant API key. Required for cloud-hosted Qdrant clusters.",
    )
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
        description="Number of Q→SQL example pairs to retrieve per query.",
    )

    # ── Connection Pool ───────────────────────────────────────────────────────

    db_pool_size: int = Field(
        default=5,
        description="Maximum DB connections per pool (PostgreSQL, MySQL).",
    )

    # ── Query Safety ──────────────────────────────────────────────────────────

    max_result_rows: int = Field(
        default=10000,
        description="Maximum rows returned per query. Guards against RAM exhaustion.",
    )

    # ── Self-Correction ───────────────────────────────────────────────────────

    max_self_correction_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="How many times to retry SQL generation after a database error.",
    )

    # ── Security ──────────────────────────────────────────────────────────────

    allowed_sql_operations: list[str] = Field(
        default=["SELECT", "WITH"],
        description=(
            "Allowlist of SQL statement types permitted to execute. "
            "Values are normalised to uppercase automatically."
        ),
    )
    enable_injection_detection: bool = Field(
        default=True,
        description="Scan user questions for prompt injection patterns before processing.",
    )

    # ── Context Memory ────────────────────────────────────────────────────────

    session_history_limit: int = Field(
        default=10,
        description="Maximum number of previous Q&A turns included in the LLM context.",
    )

    # ── Rate Limiting ─────────────────────────────────────────────────────────

    rate_limit_qpm: int = Field(
        default=60,
        description="Maximum queries per minute per tenant_id. Set to 0 to disable.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Minimum log level emitted by the library logger.",
    )
    audit_log_file: str | None = Field(
        default=None,
        description="Path to write structured JSON audit log entries. Disabled when None.",
    )

    # ── Development ───────────────────────────────────────────────────────────

    debug: bool = Field(
        default=False,
        description="Enable verbose debug output. Never set to True in production.",
    )

    @field_validator("allowed_sql_operations", mode="before")
    @classmethod
    def _uppercase_ops(cls, v: list[str]) -> list[str]:
        return [op.upper() for op in v]


def make_settings(**overrides: object) -> Settings:
    """Create a :class:`Settings` instance from the environment, then apply *overrides*.

    Use this instead of the module-level :data:`settings` singleton whenever
    different engines in the same process need different configuration (e.g.
    different LLM providers, or different tenants in tests).

    Args:
        **overrides: Any :class:`Settings` field name as a keyword argument.

    Returns:
        A fully validated :class:`Settings` instance.

    Example::

        s = make_settings(
            llm_provider="groq",
            groq_api_key="gsk_...",
            llm_timeout_seconds=60,
        )
        engine = QueryEngine(settings=s)
    """
    return Settings(**overrides)  # type: ignore[arg-type]


# ── Module-level singleton ────────────────────────────────────────────────────
#
# Provided for convenience in simple single-engine scripts.
#
# WARNING: ``settings`` is evaluated once at import time.  A second
# QueryEngine constructed with different environment variables in the same
# process will see the values from this singleton, not its own environment.
# Use ``make_settings(**overrides)`` or pass ``settings=`` to QueryEngine
# to avoid this.
# ─────────────────────────────────────────────────────────────────────────────
settings = Settings()
