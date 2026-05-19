# Quick Start

This guide gets you from zero to your first natural language query in under 5 minutes.

## 1. Get a free API key

The easiest way to start is with [Groq](https://console.groq.com) — it is free and
the fastest inference available.

```bash
export AQLIX_GROQ_API_KEY="gsk_your_key_here"
```

## 2. Query a SQLite database

```python
from aqlix import QueryEngine

engine = QueryEngine(
    llm="groq",
    database="sqlite",
    dsn="sqlite:///mydata.db",
)

# Ingest the schema so the LLM knows your tables
engine.ingest_schema()

# Ask a question
result = engine.query("How many orders were placed last month?")

print(result.sql)      # SELECT COUNT(*) FROM orders WHERE ...
print(result.data)     # pandas DataFrame
print(result.summary)  # "There were 1,243 orders placed in June 2025."
```

## 3. Multi-turn memory

aqlix remembers previous questions in the same session:

```python
engine.query("Show me the top 10 customers by revenue")
engine.query("Now filter those to only US customers")   # remembers context
engine.query("Which of those signed up in 2024?")       # still remembers
```

## 4. Train with business knowledge

Teach aqlix domain-specific terms so it generates better SQL:

```python
# Free-text documentation (retrieved via RAG)
engine.train(documentation="""
    employees.status: 1=Active, 2=On Leave, 3=Resigned, 4=Terminated
    Use strftime('%Y-%m', created_at) for SQLite month grouping.
""")

# Verified Q→SQL pairs for few-shot learning
engine.train(
    question="Top 5 employees by total sales",
    sql="SELECT e.name, SUM(s.total) FROM employees e "
        "JOIN sales s ON e.id = s.emp_id "
        "GROUP BY e.name ORDER BY 2 DESC LIMIT 5",
)
```

## 5. CLI usage

```bash
# Interactive REPL
aqlix query --db sqlite:///mydata.db --llm groq

# Single question
aqlix query --db sqlite:///mydata.db --llm groq -q "Total revenue by region"

# Ingest schema only
aqlix ingest --db sqlite:///mydata.db --llm groq
```

## Next steps

- [Configuration](configuration.md) — all environment variables and settings
- [Databases](../guide/databases.md) — connect to PostgreSQL, MySQL, Snowflake, DuckDB
- [LLM Providers](../guide/llm-providers.md) — Claude, OpenAI, Ollama