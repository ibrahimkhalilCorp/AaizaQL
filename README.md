# AaizaQL — Natural Language to SQL

**Query any database in plain English.**
AaizaQL is an open-source Python library that converts natural language questions into SQL, executes them, and returns results with charts and insights. It fixes the key limitations of Vanna AI: better security, context memory, self-correction, and a plugin architecture.

```python
from aaizaql import QueryEngine

engine = QueryEngine(llm="groq", database="sqlite", dsn="sqlite:///sales.db")
engine.ingest_schema()

result = engine.query("Show top 5 customers by revenue last quarter")
print(result.sql)      # Generated SQL
print(result.data)     # pandas DataFrame
result.chart.show()    # Interactive Plotly chart
print(result.summary)  # "The top customer was Acme Corp with $1.2M revenue..."
```

---

## Why AaizaQL over Vanna AI?

| Feature | AaizaQL | Vanna AI |
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
pip install aaizaql
```

Install with your LLM provider and database driver:

```bash
# Groq (free, fast — recommended for getting started)
pip install "aaizaql[groq]"

# Anthropic Claude
pip install "aaizaql[claude]"

# OpenAI
pip install "aaizaql[openai]"

# DeepSeek
pip install "aaizaql[deepseek]"

# Google Gemini
pip install "aaizaql[gemini]"

# Mistral
pip install "aaizaql[mistral]"

# Perplexity
pip install "aaizaql[perplexity]"

# PostgreSQL
pip install "aaizaql[postgres]"

# Microsoft SQL Server
pip install "aaizaql[mssql]"

# Oracle
pip install "aaizaql[oracle]"

# MongoDB
pip install "aaizaql[mongodb]"

# BigQuery
pip install "aaizaql[bigquery]"

# Everything
pip install "aaizaql[all]"
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
from aaizaql import QueryEngine

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
aaizaql query --db sqlite:///mydata.db --llm groq

# Single question
aaizaql query --db sqlite:///mydata.db --llm groq -q "Total revenue by region"
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
When the generated SQL fails, AaizaQL automatically sends the error back to the LLM and retries (up to 3 times by default):

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

---

## Configuration

All settings can be set via environment variables (prefixed `AAIZAQL_`) or passed directly to `QueryEngine`:

| Setting | Env var | Default | Description |
|---|---|---|---|
| LLM provider | `AAIZAQL_LLM_PROVIDER` | `groq` | See supported providers below |
| Groq API key | `AAIZAQL_GROQ_API_KEY` | — | Free at console.groq.com |
| Groq model | `AAIZAQL_GROQ_MODEL` | `llama-3.3-70b-versatile` | Any Groq-supported model |
| Anthropic key | `AAIZAQL_ANTHROPIC_API_KEY` | — | For `llm="claude"` |
| OpenAI key | `AAIZAQL_OPENAI_API_KEY` | — | For `llm="openai"` |
| DeepSeek key | `AAIZAQL_DEEPSEEK_API_KEY` | — | For `llm="deepseek"` |
| DeepSeek model | `AAIZAQL_DEEPSEEK_MODEL` | `deepseek-chat` | `deepseek-chat`, `deepseek-reasoner` |
| Perplexity key | `AAIZAQL_PERPLEXITY_API_KEY` | — | For `llm="perplexity"` |
| Perplexity model | `AAIZAQL_PERPLEXITY_MODEL` | `sonar` | `sonar`, `sonar-pro`, `sonar-reasoning` |
| Gemini key | `AAIZAQL_GEMINI_API_KEY` | — | For `llm="gemini"` |
| Gemini model | `AAIZAQL_GEMINI_MODEL` | `gemini-2.5-flash` | `gemini-2.5-flash`, `gemini-2.5-pro` |
| Mistral key | `AAIZAQL_MISTRAL_API_KEY` | — | For `llm="mistral"` |
| Mistral model | `AAIZAQL_MISTRAL_MODEL` | `mistral-large-latest` | `mistral-large-latest`, `codestral-latest` |
| Ollama URL | `AAIZAQL_OLLAMA_BASE_URL` | `http://localhost:11434` | For local models |
| Vector store | `AAIZAQL_VECTOR_STORE` | `chroma` | `chroma` or `qdrant` |
| Max retries | `AAIZAQL_MAX_SELF_CORRECTION_RETRIES` | `3` | Self-correction attempts |
| Session history | `AAIZAQL_SESSION_HISTORY_LIMIT` | `10` | Turns kept in context |

---

## Supported LLM Providers

| Provider | Key | Default Model | Install | Notes |
|---|---|---|---|---|
| Groq | `groq` | `llama-3.3-70b-versatile` | `pip install "aaizaql[groq]"` | Free tier. Fastest inference. **Recommended.** |
| Anthropic Claude | `claude` | `claude-sonnet-4-20250514` | `pip install "aaizaql[claude]"` | Best accuracy on complex schemas. |
| OpenAI | `openai` | `gpt-4o` | `pip install "aaizaql[openai]"` | GPT-4o and others. |
| DeepSeek | `deepseek` | `deepseek-chat` | `pip install "aaizaql[deepseek]"` | High quality at very low cost. |
| Perplexity | `perplexity` | `sonar` | `pip install "aaizaql[perplexity]"` | Fast Sonar models. |
| Google Gemini | `gemini` | `gemini-2.5-flash` | `pip install "aaizaql[gemini]"` | 1M token context window. |
| Mistral | `mistral` | `mistral-large-latest` | `pip install "aaizaql[mistral]"` | `codestral-latest` great for SQL. |
| Ollama | `ollama` | `llama3` | Built-in | Local, private, no API key. |

### Provider usage examples

```python
# DeepSeek — high quality, very affordable
engine = QueryEngine(
    llm="deepseek",
    database="sqlite",
    dsn="sqlite:///mydata.db",
    deepseek_api_key="sk-...",
)

# Google Gemini
engine = QueryEngine(
    llm="gemini",
    database="postgresql",
    dsn="postgresql://user:pass@localhost:5432/mydb",
    gemini_api_key="AIza...",
)

# Mistral — codestral is specialized for code/SQL
engine = QueryEngine(
    llm="mistral",
    database="mysql",
    dsn="mysql+pymysql://user:pass@localhost/mydb",
    mistral_api_key="...",
    mistral_model="codestral-latest",
)

# Perplexity
engine = QueryEngine(
    llm="perplexity",
    database="sqlite",
    dsn="sqlite:///mydata.db",
    perplexity_api_key="pplx-...",
)
```

---

## Supported Databases

| Database | Connector name | Install |
|---|---|---|
| SQLite | `sqlite` | Built-in (no install needed) |
| PostgreSQL | `postgresql` / `postgres` | `pip install "aaizaql[postgres]"` |
| MySQL | `mysql` | `pip install pymysql` |
| Snowflake | `snowflake` | `pip install "aaizaql[snowflake]"` |
| DuckDB | `duckdb` | `pip install "aaizaql[duckdb]"` |
| Microsoft SQL Server | `mssql` / `sqlserver` | `pip install "aaizaql[mssql]"` |
| Oracle | `oracle` | `pip install "aaizaql[oracle]"` |
| MongoDB | `mongodb` / `mongo` | `pip install "aaizaql[mongodb]"` |
| Google BigQuery | `bigquery` | `pip install "aaizaql[bigquery]"` |

### Database usage examples

```python
# Microsoft SQL Server
engine = QueryEngine(
    llm="groq",
    database="mssql",
    dsn="mssql://user:password@localhost:1433/mydb",
)

# Oracle (thin mode — no Oracle Client needed)
engine = QueryEngine(
    llm="groq",
    database="oracle",
    dsn="oracle://hr:password@localhost:1521/XEPDB1",
)

# MongoDB
engine = QueryEngine(
    llm="groq",
    database="mongodb",
    dsn="mongodb://user:password@localhost:27017/mydb",
)

# Google BigQuery
engine = QueryEngine(
    llm="gemini",
    database="bigquery",
    dsn="bigquery://my-project/my_dataset",
)
```

---

## Roadmap

- [x] Phase 1: Core library (RAG, self-correction, security, memory, connectors)
- [x] Phase 2: Extended LLM providers (DeepSeek, Perplexity, Gemini, Mistral)
- [x] Phase 2: Extended DB connectors (MSSQL, Oracle, MongoDB, BigQuery)
- [ ] Phase 3: SaaS web UI (FastAPI + Next.js)
- [ ] Phase 4: Federated cross-database queries (DuckDB workspace)
- [ ] Phase 5: Enterprise (SSO, RBAC, audit log, SOC2)

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/ibrahimkhalilCorp/aaizaql
cd aaizaql
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT — see [LICENSE](LICENSE).