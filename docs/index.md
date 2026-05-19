# aqlix — Natural Language to SQL

**Query any database in plain English.**

aqlix is an open-source Python library that converts natural language questions into SQL,
executes them securely, and returns results with charts and plain-English insights.

```python
from aqlix import QueryEngine

engine = QueryEngine(llm="groq", database="sqlite", dsn="sqlite:///sales.db")
engine.ingest_schema()

result = engine.query("Show top 5 customers by revenue last quarter")
print(result.sql)      # Generated SQL
print(result.data)     # pandas DataFrame
result.chart.show()    # Interactive Plotly chart
print(result.summary)  # "The top customer was Acme Corp with $1.2M revenue..."
```

## Why aqlix?

| Feature | aqlix | Vanna AI |
|---|---|---|
| SQL security layer (whitelist + injection detection) | ✅ | ⚠️ Partial |
| Self-correction loop (auto-fix broken SQL) | ✅ | ⚠️ Partial |
| Context memory (multi-turn conversations) | ✅ | ⚠️ Limited |
| Per-user credential delegation | ✅ | ❌ (CVE-2024-5565) |
| Plugin architecture (zero core changes) | ✅ | ❌ |
| Groq support (free, fast LLM) | ✅ | ❌ |
| Local LLM via Ollama | ✅ | ✅ |

## Quick links

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [Contributing](contributing.md)
- [GitHub](https://github.com/ibrahimkhalil/aqlix)