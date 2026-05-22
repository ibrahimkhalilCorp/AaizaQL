"""
aaizaql.schema.graph_store
──────────────────────────
T4.1 — GraphStore: NetworkX-backed schema graph with Leiden clustering.

Node types : TABLE, COLUMN, INDEX, QUERY
Edge types : HAS_COLUMN, FK_OF, INDEXED_BY, USED_IN, USED_FN
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore[assignment]


class GraphStore:
    """
    In-memory schema graph persisted to graph.json.
    Each tenant gets an isolated subgraph via the tenant_id prefix.
    """

    def __init__(self, persist_path: str = ".aaizaql_graph/graph.json") -> None:
        if nx is None:
            raise ImportError(
                "networkx is required for GraphRAG. Run: pip install networkx"
            )
        self._path = Path(persist_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._G: Any = nx.DiGraph()
        self._load()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._G = nx.node_link_graph(data)
                logger.info("graph_store.loaded", nodes=self._G.number_of_nodes())
            except Exception as exc:
                logger.warning("graph_store.load_failed", detail=str(exc))
                self._G = nx.DiGraph()

    def save(self) -> None:
        data = nx.node_link_data(self._G)
        self._path.write_text(json.dumps(data))
        logger.debug("graph_store.saved", nodes=self._G.number_of_nodes())

    # ── Node/edge API ─────────────────────────────────────────────────

    def add_table(self, name: str, tenant_id: str, **attrs: Any) -> str:
        node_id = f"{tenant_id}::TABLE::{name}"
        self._G.add_node(node_id, type="TABLE", name=name, tenant=tenant_id, **attrs)
        return node_id

    def add_column(
        self, name: str, table_name: str, tenant_id: str, **attrs: Any
    ) -> str:
        table_id = f"{tenant_id}::TABLE::{table_name}"
        col_id = f"{tenant_id}::COLUMN::{table_name}.{name}"
        self._G.add_node(col_id, type="COLUMN", name=name, table=table_name,
                         tenant=tenant_id, **attrs)
        self._G.add_edge(table_id, col_id, rel="HAS_COLUMN")
        return col_id

    def add_fk_edge(
        self,
        from_table: str,
        from_col: str,
        to_table: str,
        to_col: str,
        tenant_id: str,
    ) -> None:
        src = f"{tenant_id}::COLUMN::{from_table}.{from_col}"
        dst = f"{tenant_id}::COLUMN::{to_table}.{to_col}"
        self._G.add_edge(src, dst, rel="FK_OF")

    def add_query_node(
        self, sql: str, question: str, tables: list[str], tenant_id: str
    ) -> str:
        import hashlib
        q_id = f"{tenant_id}::QUERY::{hashlib.sha256(sql.encode()).hexdigest()[:12]}"
        self._G.add_node(q_id, type="QUERY", sql=sql, question=question, tenant=tenant_id)
        for table in tables:
            table_id = f"{tenant_id}::TABLE::{table}"
            if self._G.has_node(table_id):
                self._G.add_edge(q_id, table_id, rel="USED_IN")
        return q_id

    def get_fk_neighbors(self, table_name: str, tenant_id: str, depth: int = 2) -> list[str]:
        """BFS from a table node over FK_OF edges; returns related table names."""
        start = f"{tenant_id}::TABLE::{table_name}"
        if not self._G.has_node(start):
            return []
        visited: set[str] = {start}
        frontier = {start}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                for neighbor in self._G.neighbors(node):
                    edge_rel = self._G[node][neighbor].get("rel", "")
                    if edge_rel in ("FK_OF", "HAS_COLUMN") and neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = next_frontier
        table_names = [
            self._G.nodes[n]["name"]
            for n in visited
            if self._G.nodes[n].get("type") == "TABLE" and n != start
        ]
        return table_names

    def leiden_communities(self, tenant_id: str) -> list[list[str]]:
        """
        Return schema communities for the tenant subgraph.
        Falls back to connected components if python-leidenalg is unavailable.
        """
        subgraph_nodes = [
            n for n, d in self._G.nodes(data=True)
            if d.get("tenant") == tenant_id and d.get("type") == "TABLE"
        ]
        subgraph = self._G.subgraph(subgraph_nodes).to_undirected()
        try:
            import leidenalg  # type: ignore[import]
            import igraph as ig  # type: ignore[import]
            ig_graph = ig.Graph.from_networkx(subgraph)
            partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
            return [
                [subgraph_nodes[i] for i in community]
                for community in partition
            ]
        except ImportError:
            # Fallback: connected components
            return [
                list(c) for c in nx.connected_components(subgraph)
            ]
