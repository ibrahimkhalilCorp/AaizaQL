# AAIZAQL — Natural Language to SQL

**Query any database in plain English.**
AAIZAQL is an open-source Python library that converts natural language questions into SQL, executes them, and returns results with charts and insights. It fixes the key limitations of Vanna AI: better security, context memory, self-correction, and a plugin architecture.

```python
from AAIZAQL import QueryEngine

engine = QueryEngine(llm="groq", database="sqlite", dsn="sqlite:///sales.db")
engine.ingest_schema()

result = engine.query("Show top 5 customers by revenue last quarter")
print(result.sql)      # Generated SQL
print(result.data)     # pandas DataFrame
result.chart.show()    # Interactive Plotly chart
print(result.summary)  # "The top customer was Acme Corp with $1.2M revenue..."
```

---

## Why AAIZAQL over Vanna AI?

| Feature | AAIZAQL | Vanna AI |
|---|---|---|
| SQL security layer (whitelist + injection detection) | ✅ | ⚠️ Partial |
| Self-correction loop (auto-fix broken SQL) | ✅ | ⚠️ Partial |
| Context memory (multi-turn conversations) | ✅ | ⚠️ Limited |
| Enum/code mapping (always injected, no miss) | ✅ | ❌ |
| Per-user credential delegation | ✅ | ❌ (CVE-2024-5565) |
| Plugin architecture (zero core changes) | ✅ | ❌ |
| Groq support (free, fast LLM) | ✅ | ❌ |
| Local LLM via Ollama | ✅ | ✅ |

---

## Installation

```bash
pip install AAIZAQL
```

Install with your LLM provider and database driver:

```bash
# Groq (free, fast — recommended for getting started)
pip install "AAIZAQL[groq]"

# Anthropic Claude
pip install "AAIZAQL[claude]"

# OpenAI
pip install "AAIZAQL[openai]"

# PostgreSQL
pip install "AAIZAQL[postgres]"

# Everything
pip install "AAIZAQL[all]"
```

---

## Quick Start

### 1. Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com) — it is free.

```bash
export AAIZAQL_GROQ_API_KEY="gsk_your_key_here"
```

### 2. Query your database

```python
from AAIZAQL import QueryEngine

engine = QueryEngine(
    llm="groq",
    database="sqlite",
    dsn="sqlite:///mydata.db",
)
engine.ingest_schema()

result = engine.query("How many orders were placed last month?")
print(result.sql)
print(result.data)
```

### 3. CLI usage

```bash
# Interactive REPL
AAIZAQL query --db sqlite:///mydata.db --llm groq

# Single question
AAIZAQL query --db sqlite:///mydata.db --llm groq -q "Total revenue by region"
```

---

## Core Features

### Multi-turn memory
```python
engine.query("Show me the top 10 customers by revenue")
engine.query("Now filter those to only US customers")   # remembers context
engine.query("Which of those signed up in 2024?")       # still remembers
```

### Train with business knowledge
```python
# Free-text business context (retrieved via RAG)
engine.train(documentation="""
    employees.status: 1=Active, 2=On Leave, 3=Resigned, 4=Terminated
    Use strftime('%Y-%m', created_at) for SQLite month grouping.
    business_unit_id: 4=ACCL, 8=APFIL, 12=IBOS
""")

# Enum mappings — ALWAYS injected, never missed by RAG
engine.define_enum("employees", "status", {
    1: "Active", 2: "On Leave", 3: "Resigned", 4: "Terminated"
})

# Sample Q→SQL pairs for few-shot learning
engine.train(
    question="Top 5 employees by total sales",
    sql="SELECT e.name, SUM(s.total) FROM employees e JOIN sales s ON e.id = s.emp_id GROUP BY e.name ORDER BY 2 DESC LIMIT 5",
)
```

### Self-correction loop
When the generated SQL fails, AAIZAQL automatically sends the error back to the LLM and retries (up to 3 times by default):

```
attempt 1: SELECT * FROM employes   → DatabaseError: no such table
attempt 2: SELECT * FROM employees  → ✅ success
```

### SQL security layer
Every SQL passes through a security gate before execution:
- **Whitelist enforcement** — only `SELECT` and `WITH` are allowed
- **Prompt injection detection** — scans user questions for manipulation attempts
- **Structural parsing** — uses `sqlglot` to catch disguised dangerous statements
- **Multi-statement blocking** — `SELECT 1; DROP TABLE x` is rejected

### Add a new database connector
```python
from AAIZAQL.connectors.base import DatabaseConnector
from AAIZAQL.connectors import REGISTRY

class BigQueryConnector(DatabaseConnector):
    name = "bigquery"

    def connect(self, dsn): ...
    def execute(self, sql): ...
    def get_schema(self): ...

REGISTRY["bigquery"] = BigQueryConnector
```

---

## Configuration

All settings can be set via environment variables (prefixed `AAIZAQL_`) or passed directly to `QueryEngine`:

| Setting | Env var | Default | Description |
|---|---|---|---|
| LLM provider | `AAIZAQL_LLM_PROVIDER` | `groq` | `groq`, `claude`, `openai`, `ollama` |
| Groq API key | `AAIZAQL_GROQ_API_KEY` | — | Get free key at console.groq.com |
| Groq model | `AAIZAQL_GROQ_MODEL` | `llama3-70b-8192` | Any Groq-supported model |
| Anthropic key | `AAIZAQL_ANTHROPIC_API_KEY` | — | For `llm="claude"` |
| OpenAI key | `AAIZAQL_OPENAI_API_KEY` | — | For `llm="openai"` |
| Ollama URL | `AAIZAQL_OLLAMA_BASE_URL` | `http://localhost:11434` | For local models |
| Vector store | `AAIZAQL_VECTOR_STORE` | `chroma` | `chroma` or `qdrant` |
| Max retries | `AAIZAQL_MAX_SELF_CORRECTION_RETRIES` | `3` | Self-correction attempts |
| Session history | `AAIZAQL_SESSION_HISTORY_LIMIT` | `10` | Turns kept in context |

---

## Supported Databases

| Database | Connector name | Install |
|---|---|---|
| SQLite | `sqlite` | Built-in |
| PostgreSQL | `postgresql` / `postgres` | `pip install "AAIZAQL[postgres]"` |
| MySQL | `mysql` | `pip install pymysql` |
| Snowflake | `snowflake` | `pip install "AAIZAQL[snowflake]"` |
| DuckDB | `duckdb` | `pip install "AAIZAQL[duckdb]"` |

---

## Supported LLM Providers

| Provider | Key | Notes |
|---|---|---|
| Groq | `groq` | Free tier available. Fastest inference. Recommended. |
| Anthropic Claude | `claude` | Best accuracy on complex schemas. |
| OpenAI | `openai` | GPT-4o and others. |
| Ollama | `ollama` | Local, private, no API key. |

---

## Roadmap

- [x] Phase 1: Core library (RAG, self-correction, security, memory, connectors)
- [ ] Phase 2: SaaS web UI (FastAPI + Next.js)
- [ ] Phase 3: Federated cross-database queries (DuckDB workspace)
- [ ] Phase 4: Enterprise (SSO, RBAC, audit log, SOC2)

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/ibrahimkhalilCorp/AAIZAQL
cd AAIZAQL
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT — see [LICENSE](LICENSE).