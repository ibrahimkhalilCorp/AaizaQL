"""
aaizaql.schema.graph_ingestion
─────────────────────────────
T4.2 — GraphIngestion: parse DDL into GraphStore nodes and edges.
"""
from __future__ import annotations

import re

import structlog

from aaizaql.schema.graph_store import GraphStore

logger = structlog.get_logger(__name__)

class GraphIngestion:
    def __init__(self, graph_store: GraphStore) -> None:
        self._gs = graph_store

    def ingest(self, ddl: str, connector_name: str, tenant_id: str) -> None:
        """Parse DDL and populate the graph with tables, columns, and FK edges."""
        table_blocks = re.split(r"(?=CREATE\s+TABLE\b)", ddl, flags=re.IGNORECASE)
        for block in table_blocks:
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

    def _extract_table_name(self, block: str) -> str:
        m = re.search(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?(\w+)[`\"\]]?",
            block, re.IGNORECASE,
        )
        return m.group(1) if m else ""

    def _ingest_columns(self, block: str, table_name: str, tenant_id: str) -> None:
        for line in block.splitlines():
            line = line.strip().rstrip(",")
            m = re.match(r"^[`\"\[]?(\w+)[`\"\]]?\s+(\w[\w()]+)(.*)", line)
            if not m or m.group(1).upper() in (
                "CREATE", "TABLE", "PRIMARY", "FOREIGN", "UNIQUE", "INDEX", "KEY", "CONSTRAINT"
            ):
                continue
            col_name, dtype, rest = m.group(1), m.group(2), m.group(3)
            is_pk = "PRIMARY KEY" in rest.upper()
            not_null = "NOT NULL" in rest.upper()
            self._gs.add_column(col_name, table_name, tenant_id,
                                dtype=dtype, is_pk=is_pk, not_null=not_null)

    def _ingest_fks(self, block: str, table_name: str, tenant_id: str) -> None:
        pattern = re.compile(
            r"FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+(\w+)\s*\(([^)]+)\)",
            re.IGNORECASE,
        )
        for m in pattern.finditer(block):
            from_col = m.group(1).strip().strip("`\"")
            to_table = m.group(2).strip()
            to_col = m.group(3).strip().strip("`\"")
            self._gs.add_fk_edge(table_name, from_col, to_table, to_col, tenant_id)
