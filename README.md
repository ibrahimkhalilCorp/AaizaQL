# aqlix 🔍

**Query any database in plain English — federated, secure, context-aware.**

[![CI](https://github.com/your-org/aqlix/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/aqlix/actions)
[![PyPI](https://img.shields.io/pypi/v/aqlix)](https://pypi.org/project/aqlix/)
[![Python](https://img.shields.io/pypi/pyversions/aqlix)](https://pypi.org/project/aqlix/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

```python
from aqlix import QueryEngine

engine = QueryEngine(llm="claude", database="sqlite", dsn="sqlite:///sales.db")
engine.ingest_schema()

result = engine.query("What were the top 5 products by revenue last month?")
print(result.sql)  # Generated SQL
print(result.data)  # pandas DataFrame
result.chart.show()  # Interactive Plotly chart
print(result.summary)  # "The top product was Widget A with $42,300 in revenue..."
```

---

## Why aqlix?

| Feature | aqlix | Vanna AI | Text2SQL.ai |
|---|---|---|---|
| Open-source (MIT) | ✅ | ✅ | ❌ |
| Self-correction loop | ✅ | Partial | ❌ |
| Multi-turn context memory | ✅ | Limited | ❌ |
| Per-user credential delegation | ✅ | ❌ (CVE-2024-5565) | ❌ |
| Federated cross-DB queries | ✅ | ❌ | ❌ |
| Semantic layer | ✅ | ❌ | ❌ |
| Local LLM (Ollama) | ✅ | ✅ | ❌ |

---

## Installation

```bash
pip install aqlix
```

For local LLM support (no API key required):
```bash
pip install aqlix
ollama pull llama3   # https://ollama.ai
```

---

## Quick Start

### 1. Connect to SQLite (zero config)

```python
from aqlix import QueryEngine

engine = QueryEngine(
    llm="ollama",  # free, local — no API key
    database="sqlite",
    dsn="sqlite:///my.db",
)
engine.ingest_schema()  # auto-reads your tables

result = engine.query("How many orders were placed last month?")
print(result.sql)
print(result.data)
```

### 2. Connect to PostgreSQL with Claude

```python
import os
from aqlix import QueryEngine

engine = QueryEngine(
    llm="claude",
    database="postgresql",
    dsn="postgresql://user:pass@localhost:5432/mydb",
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
)
engine.ingest_schema()
result = engine.query("Show me monthly revenue for Q2 2025")
result.chart.show()
```

### 3. Multi-turn conversations

```python
engine.query("Show me the top 10 customers by revenue")
engine.query("Now filter those to only US customers")      # remembers context
engine.query("What was their average order value?")        # still remembers
```

### 4. Teach the engine (improve accuracy)

```python
# After confirming a query was correct:
engine.teach(
    question="Total sales by region for 2024",
    sql="SELECT region, SUM(amount) FROM sales WHERE year=2024 GROUP BY region",
)
# This Q→SQL pair is now stored and retrieved as a few-shot example
```

---

## Architecture

```
User Question
     │
     ▼
ContextManager (prepend conversation history)
     │
     ▼
SQLGenerator ──► VectorStore (RAG: schema + examples)
     │                         └─► LLMProvider (Claude / OpenAI / Ollama)
     │
     ▼
SQLValidator (whitelist + injection detection + sqlglot parse)
     │
     ▼
SelfCorrector ──► DatabaseConnector.execute()
     │                   ↑ retry on error (up to 3x)
     ▼
ResultRenderer (Plotly chart auto-detection)
     │
NLSummarizer (plain-English insight)
     │
     ▼
QueryResult { .sql, .data, .chart, .summary }
```

Every query passes through every layer — no shortcuts. Security is non-negotiable.

---

## Supported Databases

| Database | Status |
|---|---|
| SQLite | ✅ |
| PostgreSQL | ✅ |
| MySQL | ✅ |
| Snowflake | ✅ |
| DuckDB | ✅ |
| BigQuery | 🔜 Phase 2 |
| Databricks | 🔜 Phase 3 |

---

## Supported LLM Providers

| Provider | Model | Requires API Key |
|---|---|---|
| Anthropic Claude | claude-sonnet-4-20250514 | Yes |
| OpenAI | gpt-4o | Yes |
| Ollama (local) | llama3, mistral, etc. | No |

---

## Configuration

All settings can be passed to `QueryEngine(...)` or set as environment variables:

```bash
export aqlix_LLM_PROVIDER=claude
export aqlix_ANTHROPIC_API_KEY=sk-ant-...
export aqlix_MAX_SELF_CORRECTION_RETRIES=3
export aqlix_VECTOR_STORE=chroma
export aqlix_LOG_LEVEL=INFO
```

Or via `.env` file (auto-loaded).

---

## Security

aqlix is built with security as a first principle:

- **SQL whitelist**: only `SELECT` and `WITH` are ever executed
- **Prompt injection detection**: regex patterns catch common injection attacks
- **Dialect-aware parsing**: `sqlglot` validates SQL structure before execution
- **Self-correction never retries unsafe SQL**: validation runs on each correction attempt
- **Per-user credential delegation**: coming in Phase 3 (SaaS)

---

## Development

```bash
git clone https://github.com/your-org/aqlix
cd aqlix
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest tests/unit/
pytest tests/integration/

# Lint
black src/ tests/
flake8 src/ tests/
mypy src/aqlix
```

---

## Roadmap

| Phase | Timeline | Focus |
|---|---|---|
| **Phase 1** | Month 1–4 | ✅ Core library (you are here) |
| **Phase 2** | Month 4–7 | Web UI + Hosted SaaS |
| **Phase 3** | Month 7–10 | Federation + Credential Delegation |
| **Phase 4** | Month 10–12 | Enterprise Launch |

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

- Bug fixes: open a PR
- New database connectors: subclass `DatabaseConnector` and register in `connectors/__init__.py`
- New LLM providers: subclass `LLMProvider` and add to `llm/__init__.py`

---

## License

MIT — see [LICENSE](LICENSE).
