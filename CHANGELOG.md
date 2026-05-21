# Changelog

All notable changes to AaizaQL are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)  
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [0.2.0] — 2026-05-21

### Added

- **8 LLM providers** — Claude (Anthropic), OpenAI, Groq, DeepSeek, Perplexity, Gemini, Mistral, Ollama. Each installs via its own optional extra (`pip install aaizaql[groq]`).
- **9 database connectors** — SQLite, PostgreSQL, MySQL, MSSQL, DuckDB, MongoDB, Snowflake, BigQuery, Oracle. All registered in the plugin `REGISTRY`; no core changes needed to add more.
- **`QueryResult` dataclass** — every `engine.query()` call returns `{sql, data, summary, chart, execution_time_ms, session_id, was_corrected, correction_attempts}`.
- **`SelfCorrector`** — configurable retry loop that sends failed SQL and the database error back to the LLM. Exposes `last_sql` for logging. Raises `MaxRetriesExceeded` after the configured attempt limit.
- **Three-layer `SQLValidator`** — keyword blocklist → operation whitelist → `sqlglot` AST parse. Multi-statement detection via `len(statements) > 1`. Comment stripping before keyword scan prevents `SELECT 1 /* DROP TABLE users */` bypass class.
- **Prompt injection scanning** — the raw NL question is scanned for injection patterns before the LLM call.
- **`SemanticStore`** — RAG-backed schema memory with `train(documentation=...)`, `train(question=..., sql=...)`, and `define_enum(table, column, mapping)`. Enum mappings are always injected into the prompt (no RAG miss possible).
- **`ContextManager`** — per-session multi-turn history with configurable window limit.
- **`DuckDBConnector.register_dataframe()`** — federation helper; registers a `DataFrame` as a virtual DuckDB table for in-process joining.
- **`make_settings()` factory** — replaces the module-level `Settings()` singleton. Each `QueryEngine` instance gets an isolated settings copy via `model_copy(update=overrides)`.
- **Structured exception hierarchy** — `AAIZAQLError` root with typed subclasses: `SecurityException`, `PromptInjectionDetected`, `SQLGenerationError`, `MaxRetriesExceeded`, `DatabaseError`, `ConnectionError`, `ConnectorNotFound`, `LLMError`, `LLMTimeoutError`, `LLMProviderNotFound`, `FederationError`, `CredentialError`, `UnsupportedQueryError`.
- **`LLMTimeoutError`** — raised when a provider call exceeds `settings.llm_timeout_seconds` (default: 30 s).
- **NL summarizer** — calls the LLM a second time to produce a plain-English summary of query results, attached to `QueryResult.summary`.
- **Plotly chart renderer** — auto-selects chart type from result shape; returns a `plotly.graph_objects.Figure` in `QueryResult.chart`. Optional (`pip install aaizaql[charts]`).
- **CLI** — `aaizaql query --dsn ... --llm groq "Show top 10 customers"` entry point.
- **`pyproject.toml` optional dependency groups** — `rag`, `charts`, `qdrant`, and per-provider/per-connector groups. Core install has zero heavy dependencies.
- **GitHub Actions CI** — lint (Ruff + Black auto-fix), unit tests (Python 3.11/3.12/3.13 matrix), provider matrix, security gate, coverage upload to Codecov, wheel build + `twine check`.
- **Connector integration test matrix** — Docker Compose services for Postgres 16, MySQL 8, MSSQL 2022, MongoDB 7 in CI. SQLite and DuckDB run without containers. Snowflake, BigQuery, Oracle gated behind `SNOWFLAKE_DSN` / `BIGQUERY_DSN` / `ORACLE_DSN` env vars.
- **`MongoDBConnector`** — JSON query descriptor API (`{"collection": "...", "filter": {...}}`); `get_schema()` samples up to 100 docs per collection to infer field types.

### Changed

- `chromadb` and `sentence-transformers` moved from hard core dependencies to the `[rag]` optional group. Cold install no longer pulls gigabytes of ML tooling.
- `LLMProviderNotFound` error message now references the live provider registry instead of a hardcoded list.
- `ConnectorNotFound` no longer lazy-imports `REGISTRY` inside the exception constructor; available keys are captured at the raise site.
- Corrected SQL from `SelfCorrector` now passes through `SQLValidator` before execution (previously skipped on retry passes).
- `settings = Settings()` module-level singleton replaced with `make_settings()` factory; `QueryEngine.__init__` applies per-instance overrides via `model_copy`.

### Fixed

- `.env` removed from repository and zip archive. `.gitignore` updated. `detect-secrets` pre-commit hook added.
- `mongodb.py` connector file previously contained a copy of the Oracle connector; replaced with the correct MongoDB implementation.

### Security

- Corrected SQL on self-correction retry now re-validated through `SQLValidator` — closes a path where a hallucinated destructive statement could execute on a correction pass.
- `.env` with API keys removed from version control.

### Deprecated

- Nothing deprecated in this release.

### Removed

- Nothing removed in this release.

---

## [0.1.2] — 2026-05-20

### Added

- Initial public alpha release on PyPI.
- `QueryEngine` with SQLite and PostgreSQL connectors.
- Groq, Claude, and OpenAI LLM providers.
- Basic `SQLValidator` (keyword blocklist only).
- `SelfCorrector` (no re-validation on retry — fixed in 0.2.0).
- `SchemaIngester` + ChromaDB vector store (hard dependency — relaxed in 0.2.0).

---

[0.2.0]: https://github.com/ibrahimkhalil/aaizaql/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/ibrahimkhalil/aaizaql/releases/tag/v0.1.2