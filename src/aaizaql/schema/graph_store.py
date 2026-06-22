"""
aaizaql.schema.graph_store
──────────────────────────
GraphStore: NetworkX-backed schema graph with Leiden community detection.

Stores the database schema as a directed property graph where tables and
columns are nodes and foreign-key relationships are typed edges. Persisted
to a JSON file so the graph survives process restarts.

Node types : TABLE, COLUMN, INDEX, QUERY
Edge types : HAS_COLUMN, FK_OF, INDEXED_BY, USED_IN, USED_FN

Author: Ibrahim
Date: 2026-06-16
Version: 1.0.0
"""

import hashlib
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
    """In-memory schema graph persisted to a JSON file.

    Each tenant gets an isolated subgraph via a ``tenant_id`` prefix on all
    node IDs, so a single GraphStore instance safely serves multiple engines
    in the same process.

    Args:
        persist_path: File path for JSON persistence. The parent directory is
            created automatically if it does not exist.

    Raises:
        ImportError: If ``networkx`` is not installed.
    """

    def __init__(self, persist_path: str = ".aaizaql_graph/graph.json") -> None:
        if nx is None:
            raise ImportError("networkx is required for GraphRAG. Run: pip install networkx")
        self._path = Path(persist_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._G: Any = nx.DiGraph()
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load the graph from the JSON persistence file if it exists.

        Silently resets to an empty DiGraph on any parse error so a corrupt
        file does not prevent startup.
        """
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._G = nx.node_link_graph(data)
                logger.info("graph_store.loaded", nodes=self._G.number_of_nodes())
            except Exception as exc:
                logger.warning("graph_store.load_failed", detail=str(exc))
                self._G = nx.DiGraph()

    def save(self) -> None:
        """Persist the current graph to the JSON file."""
        data = nx.node_link_data(self._G)
        self._path.write_text(json.dumps(data))
        logger.debug("graph_store.saved", nodes=self._G.number_of_nodes())

    # ── Node / Edge API ───────────────────────────────────────────────────────

    def add_table(self, name: str, tenant_id: str, **attrs: Any) -> str:
        """Add a TABLE node to the graph.

        Args:
            name: Table name.
            tenant_id: Tenant namespace prefix.
            **attrs: Extra attributes stored on the node (e.g. ``connector``).

        Returns:
            The node ID string used in the graph.
        """
        node_id = f"{tenant_id}::TABLE::{name}"
        self._G.add_node(node_id, type="TABLE", name=name, tenant=tenant_id, **attrs)
        return node_id

    def add_column(self, name: str, table_name: str, tenant_id: str, **attrs: Any) -> str:
        """Add a COLUMN node and a HAS_COLUMN edge from its parent table.

        Args:
            name: Column name.
            table_name: Name of the table this column belongs to.
            tenant_id: Tenant namespace prefix.
            **attrs: Extra attributes (e.g. ``dtype``, ``is_pk``, ``not_null``).

        Returns:
            The column node ID string.
        """
        table_id = f"{tenant_id}::TABLE::{table_name}"
        col_id = f"{tenant_id}::COLUMN::{table_name}.{name}"
        self._G.add_node(
            col_id, type="COLUMN", name=name, table=table_name, tenant=tenant_id, **attrs
        )
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
        """Add a FK_OF directed edge between two COLUMN nodes.

        Args:
            from_table: Source table name.
            from_col: Source column name (the foreign key column).
            to_table: Referenced table name.
            to_col: Referenced column name (typically the primary key).
            tenant_id: Tenant namespace prefix.
        """
        src = f"{tenant_id}::COLUMN::{from_table}.{from_col}"
        dst = f"{tenant_id}::COLUMN::{to_table}.{to_col}"
        self._G.add_edge(src, dst, rel="FK_OF")

    def add_query_node(self, sql: str, question: str, tables: list[str], tenant_id: str) -> str:
        """Add a QUERY node and USED_IN edges to the tables it references.

        Args:
            sql: Executed SQL statement. Used to derive a stable node ID.
            question: Original natural language question.
            tables: List of table names referenced by the query.
            tenant_id: Tenant namespace prefix.

        Returns:
            The query node ID string.
        """
        q_id = f"{tenant_id}::QUERY::{hashlib.sha256(sql.encode()).hexdigest()[:12]}"
        self._G.add_node(q_id, type="QUERY", sql=sql, question=question, tenant=tenant_id)
        for table in tables:
            table_id = f"{tenant_id}::TABLE::{table}"
            if self._G.has_node(table_id):
                self._G.add_edge(q_id, table_id, rel="USED_IN")
        return q_id

    def get_fk_neighbors(self, table_name: str, tenant_id: str, depth: int = 2) -> list[str]:
        """BFS from a TABLE node over FK_OF and HAS_COLUMN edges.

        Returns the names of related tables reachable within *depth* hops,
        used by the GraphRetriever to expand schema context beyond direct
        vector-search hits.

        Args:
            table_name: Starting table name.
            tenant_id: Tenant namespace prefix.
            depth: Maximum BFS depth (number of edge hops).

        Returns:
            List of related table name strings, excluding the starting table.
            Returns an empty list if the table node does not exist in the graph.
        """
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

        return [
            self._G.nodes[n]["name"]
            for n in visited
            if self._G.nodes[n].get("type") == "TABLE" and n != start
        ]

    def leiden_communities(self, tenant_id: str) -> list[list[str]]:
        """Return schema communities for the tenant subgraph.

        Uses the Leiden algorithm for high-quality community detection when
        ``python-leidenalg`` and ``igraph`` are installed. Falls back to
        NetworkX connected components otherwise.

        Args:
            tenant_id: Tenant namespace prefix used to filter the subgraph.

        Returns:
            List of communities, where each community is a list of node ID
            strings belonging to that cluster.
        """
        subgraph_nodes = [
            n
            for n, data in self._G.nodes(data=True)
            if data.get("tenant") == tenant_id and data.get("type") == "TABLE"
        ]
        subgraph = self._G.subgraph(subgraph_nodes).to_undirected()
        try:
            import igraph as ig  # type: ignore[import]
            import leidenalg  # type: ignore[import]

            ig_graph = ig.Graph.from_networkx(subgraph)
            partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
            return [[subgraph_nodes[i] for i in community] for community in partition]
        except ImportError:
            return [list(c) for c in nx.connected_components(subgraph)]
