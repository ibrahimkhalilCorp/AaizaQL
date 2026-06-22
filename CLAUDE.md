# Developer Preferences — AaizaQL

## Identity
- Name: Ibrahim
- Role: Senior full-stack engineer + open-source library author
- Primary project: AaizaQL — NL-to-SQL open-source Python library (PyPI: aaizaql)
- Stack: Python, FastAPI, async/await, SQLAlchemy, ChromaDB/Qdrant, sqlglot

## Communication
- Concise and actionable — no verbose prose, no filler phrases
- Code first, explanation after (only if needed)
- Bengali/Banglish for deep concept explanations when helpful

---

## Project: AaizaQL

### What it does
Natural language to SQL library. User asks a question in plain English →
library generates SQL → executes it → returns DataFrame + chart + summary.
Key differentiators: SQL security layer, self-correction loop, multi-turn memory,
enum injection (never missed by RAG), plugin architecture, per-user credential delegation.

### Codebase layout
```
src/aaizaql/
├── __init__.py
├── cli.py                          # CLI entry point (aaizaql query ...)
├── api/
│   ├── __init__.py
│   └── health.py                   # FastAPI health endpoint
├── core/
│   ├── engine.py                   # QueryEngine — main public interface
│   ├── config.py                   # All settings (env vars + direct params)
│   ├── exceptions.py               # Custom exception hierarchy
│   └── rate_limiter.py             # Rate limiting logic
├── llm/
│   ├── base.py                     # BaseLLMProvider (abstract)
│   ├── groq_provider.py
│   ├── claude_provider.py
│   ├── openai_provider.py
│   ├── deepseek_provider.py
│   ├── gemini_provider.py
│   ├── mistral_provider.py
│   ├── perplexity_provider.py
│   └── ollama_provider.py
├── connectors/
│   ├── base.py                     # DatabaseConnector (abstract)
│   ├── _limit.py                   # Row limit enforcement
│   ├── sqlite.py
│   ├── postgres.py
│   ├── mysql.py
│   ├── mssql.py
│   ├── oracle.py
│   ├── duckdb.py
│   ├── snowflake.py
│   ├── bigquery.py
│   └── mongodb.py
├── nlp/
│   ├── generator.py                # NL → SQL generation (calls LLM)
│   ├── corrector.py                # Self-correction loop (retry on error)
│   ├── decomposer.py               # Complex query decomposition
│   ├── graph_retriever.py          # Graph-based schema retrieval
│   ├── prompts.py                  # All prompt templates
│   └── utils.py                    # NLP helpers
├── schema/
│   ├── ingestion.py                # ingest_schema() — reads DB schema
│   ├── embedder.py                 # Embeds schema for vector search
│   ├── semantic_store.py           # ChromaDB/Qdrant vector store wrapper
│   ├── graph_ingestion.py          # Schema → graph (nodes/edges)
│   └── graph_store.py              # Graph store interface
├── memory/
│   ├── context.py                  # Multi-turn conversation memory
│   └── vector_store.py             # RAG store for train() knowledge
├── security/
│   ├── validator.py                # SQL whitelist + injection detection
│   └── audit.py                    # Audit logging
├── visualization/
│   ├── renderer.py                 # Plotly chart generation
│   └── summarizer.py              # LLM-based result summarization
└── federation/
    └── __init__.py                 # Cross-database federation (Phase 4, WIP)
```

### Architecture rules — never violate these
- `core/engine.py` is the ONLY public interface. All external calls go through `QueryEngine`.
- `llm/base.py` and `connectors/base.py` are abstract base classes. Every provider/connector
  inherits from them. Never call a provider directly from engine — always go through the base.
- `nlp/` layer never imports from `connectors/` directly. It receives schema context as data.
- `security/validator.py` runs BEFORE every SQL execution — never bypass it.
- `memory/` is stateful — treat it carefully in async contexts.
- `federation/` is Phase 4 (WIP) — do not add business logic there yet.

### Plugin contract
New LLM provider → extend `llm/base.py`, register in `llm/__init__.py`
New DB connector → extend `connectors/base.py`, register in `connectors/__init__.py`
Zero changes to `core/engine.py` for new providers/connectors.

### Roadmap (current state)
- [x] Phase 1: Core (RAG, self-correction, security, memory, connectors)
- [x] Phase 2: Extended LLMs (DeepSeek, Perplexity, Gemini, Mistral) + DBs (MSSQL, Oracle, MongoDB, BigQuery)
- [ ] Phase 3: SaaS web UI (FastAPI + Next.js)
- [ ] Phase 4: Federated cross-database queries (DuckDB workspace)
- [ ] Phase 5: Enterprise (SSO, RBAC, audit log, SOC2)

### Env var prefix
All config via `AAIZAQL_*` env vars or direct `QueryEngine(...)` params.

---

## Code standards

### Style
- Python >=3.11, async/await preferred for all I/O
- Max 20 lines per function, single responsibility
- Meaningful names — no single-letter vars except loop indices
- Constants over magic numbers, never hardcode config values
- Max 3 levels of nesting — extract deeper logic into named helpers

### Documentation
- Every file: module-level docstring (purpose, author, date, version)
- Every class/function/method: docstring with params (type + purpose), return, exceptions
- Inline comments explain WHY, not WHAT
- Simple enough for a junior dev (3 months exp) to understand

### Error handling
- Every external call (DB, API, LLM, file I/O): explicit try/except with specific exception types
- Use custom exceptions from `core/exceptions.py` — never raise bare Exception
- Log errors with: function name + input + error message
- Never silently swallow exceptions

### Testing
- Pure functions where possible — no hidden global state
- No side effects in functions that compute values
- Tests live in `tests/` mirroring `src/aaizaql/` structure

---

## What NOT to do
- Do not suggest changes outside the scope of what I asked
- Do not add placeholder comments like `# add logic here`
- Do not use `print()` for debugging — use `logging`
- Do not touch `security/validator.py` logic without explicit instruction
- Do not add dependencies to `core/` — it must stay lightweight

## Output format
When writing or refactoring code, always end with:
1. What was changed and why (one line each)
2. What still needs attention (if anything)

---

## Codebase Graph Analysis (via Graphify)
> Source: graphify-out/GRAPH_REPORT.md — 1982 nodes, 4030 edges, 144 communities

### God Nodes — touch with extreme caution
These classes have the highest edge count in the entire graph.
A change here can break dozens of files. Always run full test suite after modifying.

| Class | Edges | Risk |
|-------|-------|------|
| `DatabaseError` | 97 | CRITICAL — bridges 30+ communities |
| `Settings` | 74 | CRITICAL — wires all LLM providers + connectors |
| `LLMError` | 71 | HIGH |
| `SelfCorrector` | 64 | HIGH |
| `LLMTimeoutError` | 58 | HIGH |
| `ConnectionError` | 51 | HIGH |
| `SQLValidator` | 49 | HIGH |
| `SQLGenerator` | 47 | HIGH |

### Cross-community bridges — never refactor in isolation
- `DatabaseError` → bridges exceptions ↔ all connectors ↔ corrector ↔ tests
- `Settings` → bridges config ↔ all LLM providers ↔ engine ↔ connectors
- `QueryEngine` → bridges engine ↔ schema ↔ NLP ↔ connectors ↔ CLI

### Refactor order (graphify cohesion-based, low risk → high risk)
1. `core/exceptions.py` ← done ✅
2. `core/rate_limiter.py` (cohesion 0.15, isolated)
3. `security/audit.py` (cohesion 0.31, isolated)
4. `schema/graph_ingestion.py` (cohesion 0.17, isolated)
5. `schema/graph_store.py` (cohesion 0.11)
6. `memory/context.py` (cohesion 0.07)
7. `memory/vector_store.py` (cohesion 0.22)
8. `schema/embedder.py` (cohesion 0.32, singleton — careful)
9. `schema/semantic_store.py` (cohesion 0.11)
10. `schema/ingestion.py` (cohesion 0.06, complex)
11. `nlp/decomposer.py` (cohesion 0.18)
12. `nlp/graph_retriever.py` (cohesion 0.14)
13. `nlp/prompts.py`
14. `nlp/utils.py`
15. `nlp/generator.py` (cohesion 0.08, God Node dependency)
16. `nlp/corrector.py` (cohesion 0.17, SelfCorrector = God Node)
17. `connectors/_limit.py`
18. `connectors/base.py`
19. `connectors/sqlite.py`
20. `connectors/duckdb.py`
21. `connectors/mysql.py`
22. `connectors/mssql.py`
23. `connectors/oracle.py`
24. `connectors/snowflake.py`
25. `connectors/bigquery.py`
26. `connectors/mongodb.py`
27. `connectors/postgres.py`
28. `security/validator.py` (SQLValidator = God Node, 49 edges)
29. `core/config.py` (Settings = God Node, 74 edges)
30. `core/engine.py` (QueryEngine = God Node, LAST)
31. `visualization/summarizer.py`
32. `visualization/renderer.py`
33. `api/health.py`
34. `cli.py`
35. `__init__.py`

### Known high-risk inferred relationships (verify after refactor)
- `DatabaseError` has 93 INFERRED edges — verify connectors still raise it correctly
- `Settings` has 70 INFERRED edges — verify all providers still boot from Settings
- `LLMError` has 66 INFERRED edges — verify all providers raise LLMError on failure
