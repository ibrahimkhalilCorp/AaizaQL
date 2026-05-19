"""
examples/01_quickstart_sqlite.py
──────────────────────────────────
Getting started with AAIZAQL using a local SQLite database.
No API key needed — uses Ollama with llama3 (free, local).

Prerequisites:
  pip install AAIZAQL
  ollama pull llama3    # https://ollama.ai
"""

import sqlite3
import os
from AAIZAQL import QueryEngine

# ── Step 1: Create a sample SQLite database ──────────────────────────────────

print("Creating sample database...")
conn = sqlite3.connect("sample_store.db")
conn.executescript("""
    DROP TABLE IF EXISTS orders;
    DROP TABLE IF EXISTS customers;
    DROP TABLE IF EXISTS products;

    CREATE TABLE customers (
        id      INTEGER PRIMARY KEY,
        name    TEXT NOT NULL,
        email   TEXT UNIQUE,
        country TEXT NOT NULL,
        joined  TEXT NOT NULL
    );
    CREATE TABLE products (
        id       INTEGER PRIMARY KEY,
        name     TEXT NOT NULL,
        category TEXT NOT NULL,
        price    REAL NOT NULL
    );
    CREATE TABLE orders (
        id          INTEGER PRIMARY KEY,
        customer_id INTEGER REFERENCES customers(id),
        product_id  INTEGER REFERENCES products(id),
        quantity    INTEGER NOT NULL,
        amount      REAL NOT NULL,
        placed_at   TEXT NOT NULL
    );

    -- Sample data
    INSERT INTO customers VALUES
        (1, 'Alice Johnson', 'alice@example.com', 'US', '2024-01-15'),
        (2, 'Bob Mueller',   'bob@example.com',   'DE', '2024-02-20'),
        (3, 'Carol Lee',     'carol@example.com', 'US', '2024-03-01'),
        (4, 'David Park',    'david@example.com', 'KR', '2024-04-10'),
        (5, 'Eva Garcia',    'eva@example.com',   'MX', '2024-05-05');

    INSERT INTO products VALUES
        (1, 'Widget Pro',    'Electronics', 299.99),
        (2, 'Gadget Plus',   'Electronics', 149.99),
        (3, 'Office Chair',  'Furniture',   599.00),
        (4, 'Standing Desk', 'Furniture',   899.00),
        (5, 'Coffee Mug',    'Kitchen',      19.99);

    INSERT INTO orders VALUES
        (1, 1, 1, 2, 599.98, '2025-06-01'),
        (2, 1, 3, 1, 599.00, '2025-06-05'),
        (3, 2, 2, 3, 449.97, '2025-06-10'),
        (4, 3, 4, 1, 899.00, '2025-06-15'),
        (5, 3, 1, 1, 299.99, '2025-06-20'),
        (6, 4, 5, 5,  99.95, '2025-07-01'),
        (7, 5, 2, 2, 299.98, '2025-07-05'),
        (8, 1, 5, 3,  59.97, '2025-07-10');
""")
conn.close()
print("✅ Database created: sample_store.db")

# ── Step 2: Create the QueryEngine ───────────────────────────────────────────

# Using Ollama (local, free) — switch to "claude" or "openai" with an API key
LLM = os.environ.get("AAIZAQL_LLM", "ollama")

engine = QueryEngine(
    llm=LLM,
    database="sqlite",
    dsn="sqlite:///sample_store.db",
)

# ── Step 3: Ingest schema (must run once, or after schema changes) ────────────

print("\nIngesting schema into vector store...")
count = engine.ingest_schema()
print(f"✅ {count} schema chunks ingested")

# ── Step 4: Ask questions ─────────────────────────────────────────────────────

questions = [
    "How many customers do we have in total?",
    "What are the top 3 products by total revenue?",
    "Show me all orders placed in July 2025",
    "What is the average order value per country?",
    "Which customer has spent the most overall?",
]

session_id = "demo-session"

for question in questions:
    print(f"\n{'─' * 60}")
    print(f"❓ {question}")
    try:
        result = engine.query(question, session_id=session_id)
        print(f"🔧 SQL: {result.sql}")
        print(f"📊 Rows returned: {len(result.data)}")
        print(result.data.to_string(index=False))
        print(f"💬 Summary: {result.summary}")
        print(f"⏱  Execution: {result.execution_time_ms}ms")
    except Exception as e:
        print(f"❌ Error: {e}")

# ── Step 5: Multi-turn demo ───────────────────────────────────────────────────

print(f"\n{'═' * 60}")
print("MULTI-TURN CONVERSATION DEMO")
print(f"{'═' * 60}")

multi_session = "multi-turn-demo"
turns = [
    "Show me the top 5 customers by total spending",
    "Now filter those to only US customers",       # remembers context
    "What products did they buy?",                  # still remembers
]

for q in turns:
    print(f"\n❓ {q}")
    try:
        r = engine.query(q, session_id=multi_session)
        print(f"🔧 SQL: {r.sql[:100]}...")
        print(f"📊 {len(r.data)} rows")
    except Exception as e:
        print(f"❌ {e}")

engine.close()
print("\n✅ Done! Check sample_store.db for the database.")
