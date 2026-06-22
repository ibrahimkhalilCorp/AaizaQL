"""
aaizaql.schema.graph_ingestion
──────────────────────────────
GraphIngestion: parse DDL strings into GraphStore nodes and edges.

Converts CREATE TABLE DDL into a typed property graph used by
:class:`~aaizaql.nlp.graph_retriever.GraphRetriever` and
:class:`~aaizaql.nlp.decomposer.QueryDecomposer` for FK-aware schema
traversal and Leiden community detection.

Node types created: ``TABLE``, ``COLUMN``
Edge types created: ``HAS_COLUMN``, ``FK_OF``

Author: Ibrahim
Date: 2026-06-16
Version: 1.0.0
"""

import re

import structlog

from aaizaql.schema.graph_store import GraphStore

logger = structlog.get_logger(__name__)

_DDL_SPLIT_PATTERN = re.compile(r"(?=CREATE\s+TABLE\b)", re.IGNORECASE)
_TABLE_NAME_PATTERN = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?(\w+)[`\"\]]?",
    re.IGNORECASE,
)
_COLUMN_PATTERN = re.compile(r"^[`\"\[]?(\w+)[`\"\]]?\s+(\w[\w()]+)(.*)")
_FK_PATTERN = re.compile(
    r"FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+(\w+)\s*\(([^)]+)\)",
    re.IGNORECASE,
)

_DDL_KEYWORDS = frozenset(
    {"CREATE", "TABLE", "PRIMARY", "FOREIGN", "UNIQUE", "INDEX", "KEY", "CONSTRAINT"}
)


class GraphIngestion:
    """Parses DDL and populates a GraphStore with table/column nodes and FK edges.

    Args:
        graph_store: GraphStore instance to write nodes and edges into.
    """

    def __init__(self, graph_store: GraphStore) -> None:
        self._gs = graph_store

    def ingest(self, ddl: str, connector_name: str, tenant_id: str) -> None:
        """Parse *ddl* and populate the graph with tables, columns, and FK edges.

        Splits the DDL on ``CREATE TABLE`` boundaries, then for each block
        adds a TABLE node, COLUMN nodes, and FK_OF edges.  Calls
        :meth:`~aaizaql.schema.graph_store.GraphStore.save` when done.

        Args:
            ddl: Full DDL string containing one or more CREATE TABLE statements.
            connector_name: Connector identifier (e.g. ``"sqlite"``) stored as
                a node attribute for multi-source graphs.
            tenant_id: Tenant identifier used to namespace all nodes and edges.
        """
        for block in _DDL_SPLIT_PATTERN.split(ddl):
            if not block.strip():
                continue
            table_name = self._extract_table_name(block)
            if not table_name:
                continue
            self._gs.add_table(table_name, tenant_id, connector=connector_name)
            self._ingest_columns(block, table_name, tenant_id)
            self._ingest_fks(block, table_name, tenant_id)

        self._gs.save()
        logger.info("graph_ingestion.done", tenant=tenant_id)

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_table_name(block: str) -> str:
        """Extract the table name from a CREATE TABLE DDL block.

        Args:
            block: A single CREATE TABLE DDL string.

        Returns:
            Table name string, or an empty string if no match is found.
        """
        match = _TABLE_NAME_PATTERN.search(block)
        return match.group(1) if match else ""

    def _ingest_columns(self, block: str, table_name: str, tenant_id: str) -> None:
        """Parse column definitions from a CREATE TABLE block and add them to the graph.

        Skips DDL keyword lines (``PRIMARY KEY``, ``FOREIGN KEY``, etc.) and
        only processes lines that look like column definitions.

        Args:
            block: Single CREATE TABLE DDL string.
            table_name: Name of the table being processed.
            tenant_id: Tenant namespace for graph nodes.
        """
        for line in block.splitlines():
            line = line.strip().rstrip(",")
            match = _COLUMN_PATTERN.match(line)
            if not match or match.group(1).upper() in _DDL_KEYWORDS:
                continue
            col_name, dtype, rest = match.group(1), match.group(2), match.group(3)
            is_pk = "PRIMARY KEY" in rest.upper()
            not_null = "NOT NULL" in rest.upper()
            self._gs.add_column(
                col_name,
                table_name,
                tenant_id,
                dtype=dtype,
                is_pk=is_pk,
                not_null=not_null,
            )

    def _ingest_fks(self, block: str, table_name: str, tenant_id: str) -> None:
        """Parse FOREIGN KEY constraints and add FK_OF edges to the graph.

        Args:
            block: Single CREATE TABLE DDL string.
            table_name: Name of the source table.
            tenant_id: Tenant namespace for graph edges.
        """
        for match in _FK_PATTERN.finditer(block):
            from_col = match.group(1).strip().strip('`"')
            to_table = match.group(2).strip()
            to_col = match.group(3).strip().strip('`"')
            self._gs.add_fk_edge(table_name, from_col, to_table, to_col, tenant_id)
