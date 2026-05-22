"""
examples/02_groq_end_to_end.py
────────────────────────────────
aaizaql — End-to-End Test with Groq + SQLite
==========================================

এই script টি:
1. It will create a sample SQLite database (e-commerce data)
2. Groq LLM দিয়ে schema ingest করবে
3. বাংলায় এবং English-এ প্রশ্ন করে SQL generate করবে
4. Results দেখাবে

Prerequisites:
    pip install groq chromadb sentence-transformers

Setup (PowerShell):
    $env:aaizaql_GROQ_API_KEY = "gsk_your_key_here"

Then run:
    python examples/02_groq_end_to_end.py
"""

import os
import sqlite3
import sys
import time

# ── Check API key early ───────────────────────────────────────────────────────
api_key = os.environ.get("AAIZAQL_GROQ_API_KEY") or os.environ.get("AAIZAQL_GROQ_API_KEY", "YOUR_GROQ_API_KEY")
if not api_key:
    print("❌ Groq API key not found!")
    print()
    print("Please set it first in PowerShell:")
    print('   $env:aaizaql_GROQ_API_KEY = "YOUR_GROQ_API_KEY"')
    print()
    print("Get a FREE key at: https://console.groq.com")
    sys.exit(1)

# Set the env var with the prefix the library expects
os.environ["AAIZAQL_GROQ_API_KEY"] = api_key

# ── Import aaizaql ──────────────────────────────────────────────────────────────
try:
    from aaizaql import QueryEngine
except ImportError:
    # fallback if running from project root without install
    sys.path.insert(0, "src")
    from aaizaql import QueryEngine

print("=" * 60)
print("  aaizaql — End-to-End Test  |  Groq + SQLite")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# STEP 1: Create sample e-commerce database
# ─────────────────────────────────────────────────────────────────
print("\n📦 Step 1: Creating sample e-commerce database...")

DB_PATH = "aaizaql_test.db"

conn = sqlite3.connect(DB_PATH)
conn.executescript("""
    DROP TABLE IF EXISTS order_items;
    DROP TABLE IF EXISTS orders;
    DROP TABLE IF EXISTS products;
    DROP TABLE IF EXISTS customers;

    CREATE TABLE customers (
        id          INTEGER PRIMARY KEY,
        name        TEXT    NOT NULL,
        email       TEXT    UNIQUE,
        country     TEXT    NOT NULL,
        joined_date TEXT    NOT NULL
    );

    CREATE TABLE products (
        id          INTEGER PRIMARY KEY,
        name        TEXT    NOT NULL,
        category    TEXT    NOT NULL,
        price       REAL    NOT NULL,
        stock       INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE orders (
        id          INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL REFERENCES customers(id),
        status      TEXT    NOT NULL DEFAULT 'completed',
        placed_at   TEXT    NOT NULL
    );

    CREATE TABLE order_items (
        id          INTEGER PRIMARY KEY,
        order_id    INTEGER NOT NULL REFERENCES orders(id),
        product_id  INTEGER NOT NULL REFERENCES products(id),
        quantity    INTEGER NOT NULL,
        unit_price  REAL    NOT NULL
    );

    -- Customers
    INSERT INTO customers VALUES
        (1,  'Alice Rahman',   'alice@example.com',  'Bangladesh', '2024-01-10'),
        (2,  'Bob Müller',     'bob@example.com',    'Germany',    '2024-02-15'),
        (3,  'Carol Chen',     'carol@example.com',  'Singapore',  '2024-03-01'),
        (4,  'David Kim',      'david@example.com',  'South Korea','2024-03-20'),
        (5,  'Eva Garcia',     'eva@example.com',    'Spain',      '2024-04-05'),
        (6,  'Farhan Ahmed',   'farhan@example.com', 'Bangladesh', '2024-04-15'),
        (7,  'Grace Liu',      'grace@example.com',  'Singapore',  '2024-05-01'),
        (8,  'Hassan Ali',     'hassan@example.com', 'UAE',        '2024-05-20');

    -- Products
    INSERT INTO products VALUES
        (1, 'Laptop Pro 15',     'Electronics', 1299.99, 50),
        (2, 'Wireless Mouse',    'Electronics',   29.99, 200),
        (3, 'Mechanical Keyboard','Electronics',   89.99, 150),
        (4, 'Standing Desk',     'Furniture',    499.00,  30),
        (5, 'Ergonomic Chair',   'Furniture',    349.00,  25),
        (6, 'USB-C Hub',         'Electronics',   49.99, 300),
        (7, 'Monitor 27"',       'Electronics',  399.99,  40),
        (8, 'Desk Lamp LED',     'Furniture',     39.99, 100);

    -- Orders
    INSERT INTO orders VALUES
        (1,  1, 'completed', '2025-01-05'),
        (2,  2, 'completed', '2025-01-10'),
        (3,  3, 'completed', '2025-01-15'),
        (4,  1, 'completed', '2025-02-01'),
        (5,  4, 'completed', '2025-02-10'),
        (6,  5, 'completed', '2025-02-20'),
        (7,  6, 'completed', '2025-03-01'),
        (8,  7, 'completed', '2025-03-10'),
        (9,  2, 'completed', '2025-03-15'),
        (10, 8, 'completed', '2025-03-20'),
        (11, 1, 'completed', '2025-04-01'),
        (12, 3, 'completed', '2025-04-10');

    -- Order Items
    INSERT INTO order_items VALUES
        (1,  1,  1, 1, 1299.99),
        (2,  1,  2, 2,   29.99),
        (3,  2,  3, 1,   89.99),
        (4,  2,  6, 2,   49.99),
        (5,  3,  7, 1,  399.99),
        (6,  3,  2, 1,   29.99),
        (7,  4,  4, 1,  499.00),
        (8,  5,  5, 1,  349.00),
        (9,  5,  8, 2,   39.99),
        (10, 6,  1, 1, 1299.99),
        (11, 7,  3, 2,   89.99),
        (12, 8,  6, 3,   49.99),
        (13, 9,  7, 1,  399.99),
        (14, 10, 2, 5,   29.99),
        (15, 11, 4, 1,  499.00),
        (16, 12, 5, 1,  349.00),
        (17, 12, 8, 1,   39.99);
""")
conn.close()
print(f"✅ Database created: {DB_PATH}")
print("   Tables: customers, products, orders, order_items")

# ─────────────────────────────────────────────────────────────────
# STEP 2: Initialize QueryEngine with Groq
# ─────────────────────────────────────────────────────────────────
print("\n🔧 Step 2: Initializing aaizaql with Groq...")

try:
    engine = QueryEngine(
        llm="groq",
        database="sqlite",
        dsn=f"sqlite:///{DB_PATH}",
        groq_model="llama-3.3-70b-versatile",
    )

    print("✅ QueryEngine ready  |  Model: groq/llama-3.3-70b-versatile")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────
# STEP 3: Ingest schema into vector store
# ─────────────────────────────────────────────────────────────────
print("\n📚 Step 3: Ingesting schema into vector store (ChromaDB)...")

try:
    count = engine.ingest_schema()
    print(f"✅ {count} schema chunks ingested")
except Exception as e:
    print(f"❌ Schema ingestion failed: {e}")
    print("   Make sure chromadb and sentence-transformers are installed:")
    print("   pip install chromadb sentence-transformers")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────
# STEP 4: Run queries
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4: Running Natural Language Queries")
print("=" * 60)

SESSION = "aaizaql-e2e-test"

test_queries = [
    {
        "label": "Simple COUNT",
        "question": "How many customers do we have in total?",
    },
    {
        "label": "Aggregation + ORDER",
        "question": "What are the top 3 best-selling products by total quantity sold?",
    },
    {
        "label": "JOIN query",
        "question": "Show me total revenue per customer, sorted highest first",
    },
    {
        "label": "Filter + GROUP BY",
        "question": "How many orders were placed each month in 2025?",
    },
    {
        "label": "Category analysis",
        "question": "What is the total revenue broken down by product category?",
    },
    {
        "label": "Multi-turn: follow-up",
        "question": "Which of those categories has the highest average order value?",
    },
]

passed = 0
failed = 0

for i, test in enumerate(test_queries, 1):
    print(f"\n[{i}/{len(test_queries)}] {test['label']}")
    print(f"  ❓ {test['question']}")
    print(f"  {'─' * 50}")

    t0 = time.time()
    try:
        result = engine.query(test["question"], session_id=SESSION)
        elapsed = int((time.time() - t0) * 1000)

        print(f"  🔧 SQL: {result.sql}")
        print(f"  📊 Rows: {len(result.data)}")

        if not result.data.empty:
            # Show up to 5 rows, nicely formatted
            print()
            display = result.data.head(5).to_string(index=False)
            for line in display.splitlines():
                print(f"     {line}")

        if result.summary:
            print(f"\n  💬 {result.summary}")

        print(f"  ⏱  {elapsed}ms", end="")
        if result.was_corrected:
            print(f"  (self-corrected after {result.correction_attempts} attempt(s))", end="")
        print()

        passed += 1

    except Exception as e:
        elapsed = int((time.time() - t0) * 1000)
        print(f"  ❌ Error ({elapsed}ms): {e}")
        failed += 1

# ─────────────────────────────────────────────────────────────────
# STEP 5: Multi-turn memory demo
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5: Multi-Turn Conversation Memory Demo")
print("=" * 60)

multi_session = "aaizaql-multiturn"
conversation = [
    "Show me all customers from Bangladesh",
    "What did they order?",           # must remember Bangladesh context
    "What was their total spending?",  # still in context
]

for q in conversation:
    print(f"\n  ❓ {q}")
    try:
        r = engine.query(q, session_id=multi_session)
        print(f"  🔧 SQL: {r.sql}")
        print(f"  📊 {len(r.data)} rows")
        if not r.data.empty:
            print(f"     {r.data.head(3).to_string(index=False)}")
    except Exception as e:
        print(f"  ❌ {e}")

# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TEST SUMMARY")
print("=" * 60)
print(f"  ✅ Passed : {passed}")
print(f"  ❌ Failed : {failed}")
print(f"  📦 DB     : {DB_PATH}")
print("  🤖 LLM    : groq/llama-3.3-70b-versatile")
print("=" * 60)

engine.close()

if failed == 0:
    print("\n🎉 All tests passed! aaizaql is working end-to-end with Groq.")
else:
    print(f"\n⚠️  {failed} test(s) failed. Check errors above.")
