"""
examples/03_train_and_query.py
────────────────────────────────
aaizaql — train() + define_enum() demo
Mirrors the vanna_demo.py but uses aaizaql's cleaner API.

Setup:
    $env:aaizaql_GROQ_API_KEY = "gsk_your_key_here"

Run:
    python examples/03_train_and_query.py
"""

import os
import sqlite3
import sys
import time

# ── API key check ─────────────────────────────────────────────────────────────
if not os.environ.get("AAIZAQL_GROQ_API_KEY","YOUR_GROQ_API_KEY"):
    print("❌  Set your key first:")
    print('    $env:aaizaql_GROQ_API_KEY = "gsk_your_key_here"')
    sys.exit(1)

from aaizaql import QueryEngine

print("=" * 60)
print("  aaizaql — train() + define_enum() demo")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# STEP 1: Build company.db  (same as vanna_demo.py)
# ─────────────────────────────────────────────────────────────────
print("\n📦  Building company.db ...")

conn = sqlite3.connect("company.db")
conn.executescript("""
    DROP TABLE IF EXISTS sales_orders;
    DROP TABLE IF EXISTS employees;
    DROP TABLE IF EXISTS departments;

    CREATE TABLE departments (
        id       INTEGER PRIMARY KEY,
        name     TEXT    NOT NULL,
        budget   REAL,
        location TEXT
    );

    CREATE TABLE employees (
        id               INTEGER PRIMARY KEY,
        name             TEXT    NOT NULL,
        department_id    INTEGER REFERENCES departments(id),
        salary           REAL,
        job_grade        INTEGER,   -- 1=Junior 2=Mid 3=Senior 4=Manager 5=Director
        status           INTEGER,   -- 1=Active 2=OnLeave 3=Resigned 4=Terminated
        business_unit_id INTEGER,   -- 4=ACCL 8=APFIL 12=IBOS
        join_date        TEXT
    );

    CREATE TABLE sales_orders (
        id             INTEGER PRIMARY KEY,
        employee_id    INTEGER REFERENCES employees(id),
        product        TEXT,
        category       TEXT,
        quantity       INTEGER,
        unit_price     REAL,
        total_amount   REAL,
        order_date     TEXT,
        order_status   INTEGER,   -- 1=Pending 2=Processing 3=Delivered 4=Cancelled 5=Returned
        payment_status INTEGER,   -- 1=Unpaid 2=Partial 3=Paid 4=Refunded
        region         TEXT
    );

    -- Departments
    INSERT INTO departments VALUES
        (1,'Engineering', 5000000,'Dhaka'),
        (2,'Sales',       3000000,'Chittagong'),
        (3,'HR',          1500000,'Dhaka'),
        (4,'IT',          2000000,'Sylhet');

    -- Employees
    INSERT INTO employees VALUES
        (1,'Rahim Uddin',    1, 85000, 3, 1,  4, '2019-03-15'),
        (2,'Karim Hassan',   2, 60000, 2, 1,  4, '2020-07-01'),
        (3,'Nadia Islam',    1, 95000, 4, 1,  8, '2018-01-10'),
        (4,'Faruk Ahmed',    3, 45000, 1, 2,  8, '2022-05-20'),
        (5,'Sumaiya Begum',  2, 72000, 3, 1, 12, '2021-09-01'),
        (6,'Tanvir Khan',    4, 55000, 2, 3,  4, '2019-11-15'),
        (7,'Mitu Akter',     1,120000, 5, 1,  8, '2017-06-01'),
        (8,'Sohel Rana',     2, 48000, 1, 4,  4, '2023-01-10'),
        (9,'Ripa Khatun',    3, 65000, 3, 1, 12, '2020-04-22'),
        (10,'Omar Faruq',    4, 78000, 4, 1, 12, '2018-08-30');

    -- Sales Orders
    INSERT INTO sales_orders VALUES
        (1,  2,'Laptop Pro',    'Electronics', 2, 1299.99, 2599.98,'2025-01-05', 3, 3,'Dhaka'),
        (2,  5,'Office Chair',  'Furniture',   5,  349.00, 1745.00,'2025-01-12', 3, 3,'Chittagong'),
        (3,  2,'USB-C Hub',     'Electronics', 10,  49.99,  499.90,'2025-02-01', 2, 2,'Dhaka'),
        (4,  9,'Stapler Set',   'Stationery',  20,  12.50,  250.00,'2025-02-14', 3, 3,'Sylhet'),
        (5,  5,'Monitor 27"',   'Electronics',  3, 399.99, 1199.97,'2025-02-20', 1, 1,'Rajshahi'),
        (6,  2,'Standing Desk', 'Furniture',    2, 499.00,  998.00,'2025-03-01', 3, 3,'Dhaka'),
        (7,  9,'Notebook Pack', 'Stationery',  50,   5.99,  299.50,'2025-03-10', 4, 4,'Khulna'),
        (8,  5,'Laptop Pro',    'Electronics',  1,1299.99, 1299.99,'2025-03-15', 3, 3,'Chittagong'),
        (9,  2,'Keyboard',      'Electronics',  8,  89.99,  719.92,'2025-04-01', 3, 3,'Dhaka'),
        (10, 9,'Pen Set',       'Stationery',  30,   8.99,  269.70,'2025-04-08', 2, 1,'Sylhet'),
        (11, 5,'Ergonomic Chair','Furniture',   3, 349.00, 1047.00,'2025-04-20', 3, 3,'Rajshahi'),
        (12, 2,'Wireless Mouse','Electronics', 15,  29.99,  449.85,'2025-05-01', 3, 3,'Dhaka');
""")
conn.close()
print("✅  company.db ready")

# ─────────────────────────────────────────────────────────────────
# STEP 2: Initialize engine
# ─────────────────────────────────────────────────────────────────
print("\n🔧  Initializing engine ...")
engine = QueryEngine(
    llm="groq",
    database="sqlite",
    dsn="sqlite:///company.db",
    groq_model="llama-3.3-70b-versatile",
)
print("✅  Engine ready")

# ─────────────────────────────────────────────────────────────────
# STEP 3: Ingest schema
# ─────────────────────────────────────────────────────────────────
print("\n📚  Ingesting schema ...")
count = engine.ingest_schema()
print(f"✅  {count} schema chunks ingested")

# ─────────────────────────────────────────────────────────────────
# STEP 4: Define enums  (always injected — no retrieval miss)
# ─────────────────────────────────────────────────────────────────
print("\n🏷️   Defining enum mappings ...")

engine.define_enum("employees", "status", {
    1: "Active", 2: "On Leave", 3: "Resigned", 4: "Terminated"
})
engine.define_enum("employees", "job_grade", {
    1: "Junior", 2: "Mid", 3: "Senior", 4: "Manager", 5: "Director"
})
engine.define_enum("employees", "business_unit_id", {
    4: "ACCL", 8: "APFIL", 12: "IBOS"
})
engine.define_enum("sales_orders", "order_status", {
    1: "Pending", 2: "Processing", 3: "Delivered", 4: "Cancelled", 5: "Returned"
})
engine.define_enum("sales_orders", "payment_status", {
    1: "Unpaid", 2: "Partial", 3: "Paid", 4: "Refunded"
})

info = engine.training_info()
print(f"✅  {info['enum_count']} enum mappings registered")
for e in info['enums']:
    print(f"     {e['table']}.{e['column']}: {e['mapping']}")

# ─────────────────────────────────────────────────────────────────
# STEP 5: Train documentation  (business context via RAG)
# ─────────────────────────────────────────────────────────────────
print("\n📝  Training documentation ...")

engine.train(documentation="""
Database: company.db
Tables: departments, employees, sales_orders

departments: id, name, budget, location
Locations: Dhaka, Chittagong, Sylhet

employees: id, name, department_id (FK to departments.id), salary, job_grade, status, business_unit_id, join_date

sales_orders: id, employee_id (FK to employees.id), product, category, quantity, unit_price, total_amount,
order_date (TEXT format: YYYY-MM-DD), order_status, payment_status, region

categories: Electronics, Furniture, Stationery
regions: Dhaka, Chittagong, Sylhet, Rajshahi, Khulna
""")

engine.train(documentation="""
SQLite date filtering conventions:
- For month/year grouping: strftime('%Y-%m', order_date)
- For current month: strftime('%Y-%m', order_date) = strftime('%Y-%m', 'now')
- For last month: strftime('%Y-%m', order_date) = strftime('%Y-%m', 'now', '-1 month')
- For last 30 days: order_date >= date('now', '-30 days')
- For this year: strftime('%Y', order_date) = strftime('%Y', 'now')
""")

print("✅  Documentation stored")

# ─────────────────────────────────────────────────────────────────
# STEP 6: Train Q→SQL pairs  (same as vanna_demo.py)
# ─────────────────────────────────────────────────────────────────
print("\n🎓  Training Q→SQL pairs ...")

engine.train(
    question="Total sales amount by category",
    sql="""
    SELECT category, SUM(total_amount) AS TotalSales
    FROM sales_orders
    GROUP BY category ORDER BY TotalSales DESC
    """
)
engine.train(
    question="Monthly sales trend",
    sql="""
    SELECT strftime('%Y-%m', order_date) AS Month,
           SUM(total_amount) AS TotalSales,
           COUNT(*) AS OrderCount
    FROM sales_orders
    GROUP BY strftime('%Y-%m', order_date)
    ORDER BY Month
    """
)
engine.train(
    question="Top 5 employees by sales amount",
    sql="""
    SELECT e.name, SUM(s.total_amount) AS TotalSales
    FROM sales_orders s
    JOIN employees e ON s.employee_id = e.id
    GROUP BY e.name ORDER BY TotalSales DESC LIMIT 5
    """
)
engine.train(
    question="Show all active ACCL employees",
    sql="""
    SELECT id, name, salary, job_grade, join_date
    FROM employees
    WHERE status = 1 AND business_unit_id = 4
    """
)
engine.train(
    question="Delivered orders with paid payment status",
    sql="""
    SELECT * FROM sales_orders
    WHERE order_status = 3 AND payment_status = 3
    ORDER BY order_date DESC
    """
)
engine.train(
    question="Average salary by department",
    sql="""
    SELECT d.name AS Department,
           AVG(e.salary) AS AvgSalary,
           COUNT(e.id) AS EmployeeCount
    FROM employees e
    JOIN departments d ON e.department_id = d.id
    GROUP BY d.name ORDER BY AvgSalary DESC
    """
)

print("✅  6 Q→SQL pairs trained")

# ─────────────────────────────────────────────────────────────────
# STEP 7: Test queries — same questions as vanna_demo.py
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 7: Testing Queries (same as vanna_demo.py)")
print("=" * 60)

test_queries = [
    # Aggregations
    ("Aggregation",    "Which product has the highest total sales?"),
    ("Aggregation",    "Total quantity sold by region"),
    ("Aggregation",    "Average order value by category"),
    ("Aggregation",    "How many cancelled orders are there?"),
    ("Aggregation",    "Total unpaid orders amount"),
    # Date/Time
    ("Date/Time",      "Sales in the last 7 days"),
    ("Date/Time",      "Which month had the highest sales?"),
    ("Date/Time",      "How many orders were placed this year?"),
    # Business Logic (enum-heavy)
    ("Business Logic", "How many employees joined after 2022?"),
    ("Business Logic", "List all terminated employees"),
    ("Business Logic", "Which business unit has the highest total salary cost?"),
    ("Business Logic", "How many directors are in ACCL?"),
    # Cross-table
    ("Cross-table",    "Which department generated the most sales?"),
    ("Cross-table",    "Show IBOS employees with their total sales amount"),
]

passed = 0
failed = 0
SESSION = "train-test"

for category, question in test_queries:
    print(f"\n[{category}]  ❓ {question}")
    t0 = time.time()
    try:
        result = engine.query(question, session_id=SESSION)
        ms = int((time.time() - t0) * 1000)
        print(f"  🔧 {result.sql[:100]}...")
        print(f"  📊 {len(result.data)} rows  ⏱ {ms}ms", end="")
        if result.was_corrected:
            print("  (corrected)", end="")
        print()
        if not result.data.empty:
            print(f"     {result.data.head(3).to_string(index=False)}")
        passed += 1
    except Exception as e:
        ms = int((time.time() - t0) * 1000)
        print(f"  ❌ {e}  ({ms}ms)")
        failed += 1

# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  ✅ Passed: {passed}   ❌ Failed: {failed}")
print(f"  Accuracy: {passed}/{passed+failed} = {100*passed//(passed+failed)}%")
print("=" * 60)

engine.close()
