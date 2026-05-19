# Supported Databases

## Overview

| Database | Connector name | Install |
|---|---|---|
| SQLite | `sqlite` | Built-in — no extra install |
| PostgreSQL | `postgresql` / `postgres` | `pip install "aqlix[postgres]"` |
| MySQL | `mysql` | `pip install "aqlix[mysql]"` |
| Snowflake | `snowflake` | `pip install "aqlix[snowflake]"` |
| DuckDB | `duckdb` | `pip install "aqlix[duckdb]"` |

---

## SQLite

Best for local development and testing. No server required.

```python
engine = QueryEngine(llm="groq", database="sqlite", dsn="sqlite:///mydata.db")

# In-memory (wiped when engine closes)
engine = QueryEngine(llm="groq", database="sqlite", dsn="sqlite:///:memory:")
```

---

## PostgreSQL

```bash
pip install "aqlix[postgres]"
```

```python
engine = QueryEngine(
    llm="groq",
    database="postgresql",
    dsn="postgresql://user:password@localhost:5432/mydb",
)
```

Both `postgresql` and `postgres` are accepted as the connector name.

---

## MySQL

```bash
pip install "aqlix[mysql]"
```

```python
engine = QueryEngine(
    llm="groq",
    database="mysql",
    dsn="mysql://user:password@localhost:3306/mydb",
)
```

---

## Snowflake

```bash
pip install "aqlix[snowflake]"
```

```python
engine = QueryEngine(
    llm="claude",
    database="snowflake",
    dsn="snowflake://user:password@myaccount/mydb/myschema?warehouse=COMPUTE_WH&role=ANALYST",
)
```

DSN parameters:

| Parameter | Required | Description |
|---|---|---|
| `user` | Yes | Snowflake username |
| `password` | Yes | Snowflake password |
| `account` | Yes | Account identifier (e.g. `myorg-myaccount`) |
| `database` | No | Default database |
| `schema` | No | Default schema (defaults to `PUBLIC`) |
| `warehouse` | No | Compute warehouse |
| `role` | No | Snowflake role |

---

## DuckDB

```bash
pip install "aqlix[duckdb]"
```

```python
# File-based (persistent)
engine = QueryEngine(llm="groq", database="duckdb", dsn="duckdb:///analytics.db")

# In-memory (ephemeral)
engine = QueryEngine(llm="groq", database="duckdb", dsn="duckdb:///:memory:")
```

DuckDB is also used internally as the ephemeral workspace for federated queries
(Phase 3) — you do not need to configure anything for that.

---

## Adding a custom connector

Any database can be added without modifying aqlix core:

```python
from aqlix.connectors.base import DatabaseConnector
from aqlix.connectors import REGISTRY
import pandas as pd

class BigQueryConnector(DatabaseConnector):
    name = "bigquery"

    def connect(self, dsn: str) -> None:
        from google.cloud import bigquery
        self.client = bigquery.Client()

    def execute(self, sql: str) -> pd.DataFrame:
        return self.client.query(sql).to_dataframe()

    def get_schema(self) -> str:
        # Return DDL string for all tables
        return ""

REGISTRY["bigquery"] = BigQueryConnector

# Now use it
engine = QueryEngine(llm="groq", database="bigquery", dsn="...")
```