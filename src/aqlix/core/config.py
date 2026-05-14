"""
aqlix.core.config
─────────────────
All configuration for the library via pydantic-settings.
Values are read from environment variables or a .env file.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    OLLAMA = "ollama"
    GROQ = "groq"


class VectorStoreBackend(str, Enum):
    CHROMA = "chroma"
    QDRANT = "qdrant"


class Settings(BaseSettings):
    """
    Master settings for aqlix.

    All fields can be overridden via environment variables prefixed with aqlix_.
    Example: aqlix_LLM_PROVIDER=openai

    Or by passing kwargs directly to QueryEngine(...).
    """

    model_config = SettingsConfigDict(
        env_prefix="aqlix_",
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
    groq_model: str = Field(
        default="llama3-70b-8192",
        description="Groq model string. Options: llama3-70b-8192,"
                    "llama3-8b-8192, mixtral-8x7b-32768",
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

    # ── Vector Store ─────────────────────────────────────────────────────────
    vector_store: VectorStoreBackend = Field(
        default=VectorStoreBackend.CHROMA,
        description="Vector store backend for RAG.",
    )
    chroma_persist_dir: str = Field(
        default=".aqlix_chroma",
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


# Module-level singleton — import this wherever settings are needed
settings = Settings()
