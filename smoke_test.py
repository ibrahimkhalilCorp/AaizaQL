# smoke_test.py  — put it in E:\my_work\aaizaql\
from aaizaql.connectors.sqlite import SQLiteConnector
from aaizaql.connectors._limit import inject_limit
from aaizaql.nlp.utils import parse_sql_response
from aaizaql.core.rate_limiter import RateLimiter
from aaizaql.schema.graph_store import GraphStore

# T1.3 — fence stripping
sql = parse_sql_response("```sql\nSELECT 1\n```")
assert sql == "SELECT 1", f"Got: {sql}"
print("✅ T1.3 parse_sql_response")

# T1.4 — LIMIT injection
limited, was_truncated = inject_limit("SELECT * FROM employees", 100)
assert "100" in limited
print("✅ T1.4 inject_limit")

# T1.2 / T1.4 — SQLite connector + LIMIT
conn = SQLiteConnector()
conn.connect("sqlite:///:memory:")
df = conn.execute("SELECT 1 AS val")
assert df["val"][0] == 1
print("✅ T1.2 SQLite connector")

# T5.3 — rate limiter
rl = RateLimiter(qpm=60)
rl.check("user_123")
print("✅ T5.3 RateLimiter")

# T4.1 — GraphStore (needs networkx)
try:
    gs = GraphStore(persist_path=".test_graph.json")
    gs.add_table("employees", tenant_id="test")
    gs.add_column("id", "employees", tenant_id="test", is_pk=True)
    print("✅ T4.1 GraphStore")
except ImportError:
    print("⚠️  T4.1 GraphStore skipped — run: pip install networkx")

print("\n🎉 All smoke tests passed!")
