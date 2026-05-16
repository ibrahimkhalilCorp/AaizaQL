"""
examples/quickstart_sqlite.py
──────────────────────────────
Getting started with aqlix — querying a local SQLite database.

Run this script after installing aqlix and setting your Groq API key:

    pip install "aqlix[groq]"
    export AQLIX_GROQ_API_KEY="gsk_your_key_here"
    python examples/quickstart_sqlite.py

No database setup required — this script creates a sample SQLite database.
"""

import sqlite3
import tempfile
from pathlib import Path


def create_sample_db(path: str) -> None:
    """Create a small sample database for demonstration."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS employees (
            id          INTEGER PRIMARY KEY,
            name        TEXT    NOT NULL,
            department  TEXT    NOT NULL,
            salary      REAL    NOT NULL,
            status      INTEGER NOT NULL DEFAULT 1,
            hire_date   TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS departments (
            id   INTEGER PRIMARY KEY,
            name TEXT    NOT NULL,
            head TEXT
        );

        INSERT OR IGNORE INTO employees VALUES
            (1, 'Alice',   'Engineering', 95000, 1, '2021-03-15'),
            (2, 'Bob',     'Marketing',   72000, 2, '2020-07-01'),
            (3, 'Charlie', 'Engineering', 88000, 1, '2022-01-10'),
            (4, 'Diana',   'HR',          65000, 1, '2019-11-20'),
            (5, 'Eve',     'Marketing',   78000, 1, '2023-04-05'),
            (6, 'Frank',   'Engineering', 102000, 1, '2018-06-12');

        INSERT OR IGNORE INTO departments VALUES
            (1, 'Engineering', 'Alice'),
            (2, 'Marketing',   'Bob'),
            (3, 'HR',          'Diana');
    """)
    conn.commit()
    conn.close()


def main() -> None:
    # ── Setup ────────────────────────────────────────────────────────────────
    tmp = tempfile.mkdtemp()
    db_path = str(Path(tmp) / "company.db")
    create_sample_db(db_path)
    print(f"Sample database created at: {db_path}\n")

    # ── Engine ───────────────────────────────────────────────────────────────
    from aqlix import QueryEngine

    engine = QueryEngine(
        llm="groq",          # free API key from console.groq.com
        database="sqlite",
        dsn=f"sqlite:///{db_path}",
    )

    # Ingest the schema so the LLM knows what tables exist
    n = engine.ingest_schema()
    print(f"Schema ingested: {n} table chunk(s)\n")

    # ── Train with business knowledge ─────────────────────────────────────────
    engine.define_enum("employees", "status", {
        1: "Active",
        2: "On Leave",
        3: "Resigned",
        4: "Terminated",
    })

    engine.train(documentation="""
        Use strftime('%Y', hire_date) for year grouping in SQLite.
        The salary column is in USD (annual).
        status=1 means Active employees only.
    """)

    # ── Queries ───────────────────────────────────────────────────────────────
    questions = [
        "How many employees are there in total?",
        "What is the average salary by department?",
        "Show me all active employees in Engineering ordered by salary",
        "Who earns more than $90,000?",
    ]

    for question in questions:
        print(f"{'─' * 60}")
        print(f"Q: {question}")
        try:
            result = engine.query(question)
            print(f"\nSQL: {result.sql}\n")
            print(result.data.to_string(index=False))
            if result.summary:
                print(f"\n→ {result.summary}")
            print(f"\n  ({len(result.data)} rows, {result.execution_time_ms}ms)\n")
        except Exception as exc:
            print(f"  [error] {exc}\n")

    # ── Multi-turn memory ────────────────────────────────────────────────────
    print(f"{'─' * 60}")
    print("Multi-turn conversation (context memory):\n")
    session = "demo-session"

    r1 = engine.query("Show me all employees", session_id=session)
    print(f"Q1: Show me all employees")
    print(f"    → {len(r1.data)} rows returned\n")

    r2 = engine.query("Now filter to only Engineering", session_id=session)
    print(f"Q2: Now filter to only Engineering  (remembers previous context)")
    print(f"    SQL: {r2.sql}")
    print(f"    → {len(r2.data)} rows returned\n")

    engine.close()
    print("Done! ✓")


if __name__ == "__main__":
    main()
