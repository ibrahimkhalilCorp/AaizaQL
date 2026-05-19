# QueryEngine

The `QueryEngine` is the single public entry point for the aqlix library.
All other modules (NLP, security, connectors, memory) are internal implementation details.

## Import

```python
from aqlix import QueryEngine
```

## Constructor

```python
QueryEngine(
    llm: str = "groq",
    database: str = "sqlite",
    dsn: str = "",
    **settings_overrides,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `llm` | `str` | `"groq"` | LLM provider: `"groq"`, `"claude"`, `"openai"`, `"ollama"` |
| `database` | `str` | `"sqlite"` | Connector name: `"sqlite"`, `"postgresql"`, `"mysql"`, `"snowflake"`, `"duckdb"` |
| `dsn` | `str` | `""` | Database connection string |
| `**settings_overrides` | | | Any `Settings` field overridden at runtime |

## Methods

### `query(question)`

Convert a natural language question to SQL, execute it, and return a result.

```python
result = engine.query("How many users signed up last month?")

result.sql            # str — the generated SQL
result.data           # pd.DataFrame — query results
result.chart          # plotly.Figure — auto-detected chart
result.summary        # str — plain-English insight
result.execution_time_ms  # int — query duration in ms
```

### `ingest_schema()`

Introspect the connected database and store its schema in the vector store.
Call this once after connecting, and again whenever the schema changes.

```python
count = engine.ingest_schema()
print(f"{count} table chunks stored")
```

### `train(question=None, sql=None, documentation=None)`

Store knowledge in the vector store for future retrieval.

```python
# Verified Q→SQL pair
engine.train(
    question="Total revenue by region",
    sql="SELECT region, SUM(amount) FROM orders GROUP BY region",
)

# Free-text business documentation
engine.train(documentation="status column: 1=Active, 2=Inactive, 3=Deleted")
```

### `define_enum(table, column, mapping)`

Register an enum mapping that is always injected into the prompt — never missed by RAG.

```python
engine.define_enum("orders", "status", {
    1: "Pending", 2: "Shipped", 3: "Delivered", 4: "Cancelled"
})
```

### `reset_context()`

Clear the current session's conversation history.

```python
engine.reset_context()
```

### `close()`

Close the database connection and release resources.

```python
engine.close()
```

## QueryResult

The object returned by `engine.query()`.

| Attribute | Type | Description |
|---|---|---|
| `sql` | `str` | Generated SQL statement |
| `data` | `pd.DataFrame` | Query results |
| `chart` | `plotly.Figure` | Auto-detected visualisation |
| `summary` | `str` | Plain-English insight from LLM |
| `execution_time_ms` | `int` | Total execution time in milliseconds |
| `session_id` | `str` | Session identifier for multi-turn tracking |