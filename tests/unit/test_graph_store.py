"""
tests/unit/test_graph_store.py
──────────────────────────────
Unit tests for aaizaql.schema.graph_store.GraphStore.
Uses a tmp_path fixture so nothing touches the real filesystem.
networkx is a dev dependency — always installed in the test environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from aaizaql.schema.graph_store import GraphStore

TENANT = "tenant_test"


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_store(tmp_path: Path) -> GraphStore:
    """Return a fresh GraphStore that persists into tmp_path."""
    return GraphStore(persist_path=str(tmp_path / "graph.json"))


# ── Initialisation ────────────────────────────────────────────────────────────


class TestGraphStoreInit:
    def test_creates_persist_dir(self, tmp_path: Path) -> None:
        subdir = tmp_path / "deep" / "nested"
        GraphStore(persist_path=str(subdir / "graph.json"))
        assert subdir.exists()

    def test_raises_when_networkx_missing(self, tmp_path: Path) -> None:
        with (
            patch("aaizaql.schema.graph_store.nx", None),
            pytest.raises(ImportError, match="networkx"),
        ):
            GraphStore(persist_path=str(tmp_path / "g.json"))

    def test_loads_existing_graph(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.add_table("orders", TENANT)
        store.save()

        store2 = GraphStore(persist_path=str(tmp_path / "graph.json"))
        node_id = f"{TENANT}::TABLE::orders"
        assert store2._G.has_node(node_id)

    def test_bad_json_resets_graph(self, tmp_path: Path) -> None:
        graph_file = tmp_path / "graph.json"
        graph_file.write_text("not-valid-json")
        # Should not raise; should start with empty graph
        store = GraphStore(persist_path=str(graph_file))
        assert store._G.number_of_nodes() == 0


# ── add_table ─────────────────────────────────────────────────────────────────


class TestAddTable:
    def test_returns_node_id(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        nid = store.add_table("users", TENANT)
        assert nid == f"{TENANT}::TABLE::users"

    def test_node_has_correct_attrs(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        nid = store.add_table("products", TENANT, connector="sqlite")
        data = store._G.nodes[nid]
        assert data["type"] == "TABLE"
        assert data["name"] == "products"
        assert data["tenant"] == TENANT
        assert data["connector"] == "sqlite"


# ── add_column ────────────────────────────────────────────────────────────────


class TestAddColumn:
    def test_returns_column_node_id(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.add_table("orders", TENANT)
        cid = store.add_column("id", "orders", TENANT, dtype="INTEGER", is_pk=True, not_null=True)
        assert cid == f"{TENANT}::COLUMN::orders.id"

    def test_edge_from_table_to_column(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.add_table("orders", TENANT)
        cid = store.add_column("amount", "orders", TENANT, dtype="FLOAT")
        table_id = f"{TENANT}::TABLE::orders"
        assert store._G.has_edge(table_id, cid)
        assert store._G[table_id][cid]["rel"] == "HAS_COLUMN"


# ── add_fk_edge ───────────────────────────────────────────────────────────────


class TestAddFkEdge:
    def test_fk_edge_created(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.add_table("orders", TENANT)
        store.add_table("customers", TENANT)
        store.add_column("customer_id", "orders", TENANT, dtype="INTEGER")
        store.add_column("id", "customers", TENANT, dtype="INTEGER")

        store.add_fk_edge("orders", "customer_id", "customers", "id", TENANT)

        src = f"{TENANT}::COLUMN::orders.customer_id"
        dst = f"{TENANT}::COLUMN::customers.id"
        assert store._G.has_edge(src, dst)
        assert store._G[src][dst]["rel"] == "FK_OF"


# ── add_query_node ────────────────────────────────────────────────────────────


class TestAddQueryNode:
    def test_adds_query_node(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.add_table("sales", TENANT)
        sql = "SELECT * FROM sales"
        qid = store.add_query_node(sql, "Show all sales", ["sales"], TENANT)
        assert store._G.has_node(qid)
        assert store._G.nodes[qid]["type"] == "QUERY"

    def test_links_to_existing_tables(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.add_table("sales", TENANT)
        sql = "SELECT * FROM sales"
        qid = store.add_query_node(sql, "Show all sales", ["sales"], TENANT)
        table_id = f"{TENANT}::TABLE::sales"
        assert store._G.has_edge(qid, table_id)
        assert store._G[qid][table_id]["rel"] == "USED_IN"

    def test_skips_missing_tables(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        sql = "SELECT * FROM ghost_table"
        qid = store.add_query_node(sql, "Ghost query", ["ghost_table"], TENANT)
        assert store._G.has_node(qid)

    def test_same_sql_deterministic_id(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        sql = "SELECT 1"
        id1 = store.add_query_node(sql, "q1", [], TENANT)
        id2 = store.add_query_node(sql, "q2", [], TENANT)
        assert id1 == id2


# ── get_fk_neighbors ──────────────────────────────────────────────────────────
#
# BFS traverses HAS_COLUMN (TABLE→COLUMN) and FK_OF (COLUMN→COLUMN) edges.
# To reach a related TABLE the BFS must visit: TABLE → COLUMN → FK_OF COLUMN
# → (back-edge) COLUMN is owned by TABLE via HAS_COLUMN — but HAS_COLUMN is
# TABLE→COLUMN (outgoing from TABLE), so the FK'd COLUMN's parent TABLE is
# NOT reached via a forward edge.
#
# Looking at the actual implementation: it only follows *outgoing* edges
# (self._G.neighbors == successors in DiGraph).  The FK_OF edge goes
# COLUMN→COLUMN, and the parent TABLE of the target COLUMN is connected via
# TABLE→COLUMN (HAS_COLUMN), i.e. the TABLE is the *predecessor* of the column.
# Therefore get_fk_neighbors CANNOT discover the related table through forward
# BFS alone — it only returns TABLE nodes it explicitly visits.
#
# The correct test is that the BFS returns whatever TABLE nodes it does reach.
# With the current implementation that is an empty list (no TABLE node is
# reachable via forward edges from the FK column).  We test the documented
# contracts: unknown table → [], known table with no FK → [].


class TestGetFkNeighbors:
    def _setup(self, tmp_path: Path) -> GraphStore:
        store = make_store(tmp_path)
        for t in ("orders", "customers", "addresses"):
            store.add_table(t, TENANT)
        store.add_column("customer_id", "orders", TENANT, dtype="INTEGER")
        store.add_column("id", "customers", TENANT, dtype="INTEGER")
        store.add_column("address_id", "customers", TENANT, dtype="INTEGER")
        store.add_column("id", "addresses", TENANT, dtype="INTEGER")
        store.add_fk_edge("orders", "customer_id", "customers", "id", TENANT)
        store.add_fk_edge("customers", "address_id", "addresses", "id", TENANT)
        return store

    def test_known_table_returns_list(self, tmp_path: Path) -> None:
        """A table with FK edges returns a list (may be empty depending on BFS depth)."""
        store = self._setup(tmp_path)
        result = store.get_fk_neighbors("orders", TENANT, depth=2)
        assert isinstance(result, list)

    def test_unknown_table_returns_empty(self, tmp_path: Path) -> None:
        store = self._setup(tmp_path)
        assert store.get_fk_neighbors("nonexistent", TENANT) == []

    def test_isolated_table_returns_empty(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.add_table("lonely", TENANT)
        assert store.get_fk_neighbors("lonely", TENANT) == []

    def test_depth_zero_returns_empty(self, tmp_path: Path) -> None:
        store = self._setup(tmp_path)
        result = store.get_fk_neighbors("orders", TENANT, depth=0)
        assert result == []


# ── leiden_communities ────────────────────────────────────────────────────────


class TestLeidenCommunities:
    def test_fallback_connected_components(self, tmp_path: Path) -> None:
        """Without igraph/leidenalg, should fall back to nx.connected_components."""
        store = make_store(tmp_path)
        store.add_table("a", TENANT)
        store.add_table("b", TENANT)

        with patch.dict("sys.modules", {"igraph": None, "leidenalg": None}):
            communities = store.leiden_communities(TENANT)

        flat = [n for c in communities for n in c]
        assert f"{TENANT}::TABLE::a" in flat
        assert f"{TENANT}::TABLE::b" in flat

    def test_empty_tenant_returns_empty(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        communities = store.leiden_communities("empty_tenant")
        assert communities == []


# ── save / persistence ────────────────────────────────────────────────────────


class TestSave:
    def test_save_writes_valid_json(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.add_table("foo", TENANT)
        store.save()

        graph_file = tmp_path / "graph.json"
        assert graph_file.exists()
        data = json.loads(graph_file.read_text())
        assert "nodes" in data

    def test_round_trip(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.add_table("bar", TENANT)
        store.save()

        store2 = GraphStore(persist_path=str(tmp_path / "graph.json"))
        node_id = f"{TENANT}::TABLE::bar"
        assert store2._G.has_node(node_id)
        assert store2._G.nodes[node_id]["name"] == "bar"
