"""
tests/unit/test_graph_ingestion.py
───────────────────────────────────
Unit tests for aaizaql.schema.graph_ingestion.GraphIngestion.
GraphStore is used directly (no mocking needed — it's fast in-memory).
"""

from __future__ import annotations

from pathlib import Path

from aaizaql.schema.graph_ingestion import GraphIngestion
from aaizaql.schema.graph_store import GraphStore

TENANT = "t1"
CONNECTOR = "sqlite"


def make_ingestion(tmp_path: Path) -> GraphIngestion:
    store = GraphStore(persist_path=str(tmp_path / "graph.json"))
    return GraphIngestion(store)


# ── _extract_table_name ───────────────────────────────────────────────────────


class TestExtractTableName:
    def test_simple_create_table(self, tmp_path: Path) -> None:
        ing = make_ingestion(tmp_path)
        assert ing._extract_table_name("CREATE TABLE users (id INTEGER);") == "users"

    def test_create_table_if_not_exists(self, tmp_path: Path) -> None:
        ing = make_ingestion(tmp_path)
        block = "CREATE TABLE IF NOT EXISTS orders (id INTEGER);"
        assert ing._extract_table_name(block) == "orders"

    def test_backtick_quoted_name(self, tmp_path: Path) -> None:
        ing = make_ingestion(tmp_path)
        block = "CREATE TABLE `products` (id INTEGER);"
        assert ing._extract_table_name(block) == "products"

    def test_double_quote_name(self, tmp_path: Path) -> None:
        ing = make_ingestion(tmp_path)
        block = 'CREATE TABLE "invoices" (id INTEGER);'
        assert ing._extract_table_name(block) == "invoices"

    def test_no_match_returns_empty(self, tmp_path: Path) -> None:
        ing = make_ingestion(tmp_path)
        assert ing._extract_table_name("-- just a comment") == ""


# ── _ingest_columns ───────────────────────────────────────────────────────────


class TestIngestColumns:
    _DDL = """
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    salary FLOAT,
    FOREIGN KEY (dept_id) REFERENCES departments(id)
);
"""

    def test_columns_added_to_store(self, tmp_path: Path) -> None:
        store = GraphStore(persist_path=str(tmp_path / "g.json"))
        store.add_table("employees", TENANT)
        ing = GraphIngestion(store)
        ing._ingest_columns(self._DDL, "employees", TENANT)
        col_id = f"{TENANT}::COLUMN::employees.id"
        assert store._G.has_node(col_id)

    def test_pk_flag_set(self, tmp_path: Path) -> None:
        store = GraphStore(persist_path=str(tmp_path / "g.json"))
        store.add_table("employees", TENANT)
        ing = GraphIngestion(store)
        ing._ingest_columns(self._DDL, "employees", TENANT)
        col_id = f"{TENANT}::COLUMN::employees.id"
        assert store._G.nodes[col_id]["is_pk"] is True

    def test_not_null_flag_set(self, tmp_path: Path) -> None:
        store = GraphStore(persist_path=str(tmp_path / "g.json"))
        store.add_table("employees", TENANT)
        ing = GraphIngestion(store)
        ing._ingest_columns(self._DDL, "employees", TENANT)
        col_id = f"{TENANT}::COLUMN::employees.name"
        assert store._G.nodes[col_id]["not_null"] is True

    def test_reserved_keywords_skipped(self, tmp_path: Path) -> None:
        store = GraphStore(persist_path=str(tmp_path / "g.json"))
        store.add_table("employees", TENANT)
        ing = GraphIngestion(store)
        ing._ingest_columns(self._DDL, "employees", TENANT)
        # FOREIGN KEY line should not produce a column node named "FOREIGN"
        assert not store._G.has_node(f"{TENANT}::COLUMN::employees.FOREIGN")


# ── _ingest_fks ───────────────────────────────────────────────────────────────


class TestIngestFks:
    _DDL = """
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
"""

    def test_fk_edge_created(self, tmp_path: Path) -> None:
        store = GraphStore(persist_path=str(tmp_path / "g.json"))
        store.add_table("orders", TENANT)
        store.add_column("customer_id", "orders", TENANT, dtype="INTEGER")
        store.add_table("customers", TENANT)
        store.add_column("id", "customers", TENANT, dtype="INTEGER")
        ing = GraphIngestion(store)
        ing._ingest_fks(self._DDL, "orders", TENANT)
        src = f"{TENANT}::COLUMN::orders.customer_id"
        dst = f"{TENANT}::COLUMN::customers.id"
        assert store._G.has_edge(src, dst)

    def test_no_fk_no_crash(self, tmp_path: Path) -> None:
        store = GraphStore(persist_path=str(tmp_path / "g.json"))
        store.add_table("simple", TENANT)
        ing = GraphIngestion(store)
        ing._ingest_fks("CREATE TABLE simple (id INTEGER);", "simple", TENANT)
        # No edges beyond HAS_COLUMN expected — no crash
        assert store._G.number_of_nodes() >= 1


# ── ingest (full pipeline) ────────────────────────────────────────────────────


class TestIngest:
    _FULL_DDL = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255)
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    total FLOAT,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
"""

    def test_tables_ingested(self, tmp_path: Path) -> None:
        store = GraphStore(persist_path=str(tmp_path / "g.json"))
        ing = GraphIngestion(store)
        ing.ingest(self._FULL_DDL, CONNECTOR, TENANT)
        assert store._G.has_node(f"{TENANT}::TABLE::customers")
        assert store._G.has_node(f"{TENANT}::TABLE::orders")

    def test_columns_ingested(self, tmp_path: Path) -> None:
        store = GraphStore(persist_path=str(tmp_path / "g.json"))
        ing = GraphIngestion(store)
        ing.ingest(self._FULL_DDL, CONNECTOR, TENANT)
        assert store._G.has_node(f"{TENANT}::COLUMN::customers.id")
        assert store._G.has_node(f"{TENANT}::COLUMN::orders.total")

    def test_fk_edges_ingested(self, tmp_path: Path) -> None:
        store = GraphStore(persist_path=str(tmp_path / "g.json"))
        ing = GraphIngestion(store)
        ing.ingest(self._FULL_DDL, CONNECTOR, TENANT)
        src = f"{TENANT}::COLUMN::orders.customer_id"
        dst = f"{TENANT}::COLUMN::customers.id"
        assert store._G.has_edge(src, dst)

    def test_graph_saved_to_disk(self, tmp_path: Path) -> None:
        store = GraphStore(persist_path=str(tmp_path / "g.json"))
        ing = GraphIngestion(store)
        ing.ingest(self._FULL_DDL, CONNECTOR, TENANT)
        assert (tmp_path / "g.json").exists()

    def test_empty_ddl_no_crash(self, tmp_path: Path) -> None:
        store = GraphStore(persist_path=str(tmp_path / "g.json"))
        ing = GraphIngestion(store)
        ing.ingest("", CONNECTOR, TENANT)
        assert store._G.number_of_nodes() == 0

    def test_connector_name_stored_on_table(self, tmp_path: Path) -> None:
        store = GraphStore(persist_path=str(tmp_path / "g.json"))
        ing = GraphIngestion(store)
        ing.ingest("CREATE TABLE foo (id INTEGER);", "postgres", TENANT)
        nid = f"{TENANT}::TABLE::foo"
        assert store._G.nodes[nid]["connector"] == "postgres"
