"""
aaizaql.nlp.graph_retriever
───────────────────────────
GraphRetriever: vector seed → FK traversal → enriched DDL context.

Improves schema retrieval accuracy for multi-table queries by following
foreign key edges in the schema graph. A plain vector search returns the most
semantically similar DDL chunks, but misses related tables that are joined via
FK relationships. This retriever expands the seed set via BFS before
assembling the final context string.

Retrieval pipeline per :meth:`get_context` call:

1. Vector search → top-k DDL chunks → extract seed table names.
2. BFS from seed tables over FK edges (depth = 2) → related table names.
3. Fetch DDL chunks for FK-related tables and merge with seed chunks.
4. Return a single newline-separated DDL context string.

Author: Ibrahim
Date: 2026-06-16
Version: 1.0.0
"""

import re
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class GraphRetriever:
    """Schema context retriever that augments vector search with FK graph traversal.

    Args:
        vector_store: Vector store adapter used for DDL chunk retrieval.
        graph_store: Schema graph store used for FK neighbor lookup.
        top_k: Number of seed DDL chunks to retrieve in the initial vector
            search.
    """

    def __init__(self, vector_store: Any, graph_store: Any, top_k: int = 5) -> None:
        self._vs = vector_store
        self._gs = graph_store
        self._top_k = top_k

    def get_context(self, query: str, tenant_id: str) -> str:
        """Return an enriched schema context string for *query*.

        Performs a vector search to find seed tables, then expands to FK
        neighbors via BFS so that joined tables appear in the LLM prompt
        even when they are not semantically similar to the question text.

        Args:
            query: Natural language question or search term.
            tenant_id: Tenant identifier for multi-tenant graph store lookups.

        Returns:
            Newline-separated DDL context string combining seed chunks and
            FK-expanded chunks. Returns ``"(no schema ingested yet)"`` when
            the vector search returns no results.
        """
        hits = self._vs.search(query=query, filter_type="ddl", top_k=self._top_k)
        if not hits:
            return "(no schema ingested yet)"

        seed_tables = [t for t in (self._extract_table_name(h.text) for h in hits) if t]

        related: set[str] = set()
        for table in seed_tables:
            related.update(self._gs.get_fk_neighbors(table, tenant_id, depth=2))

        context_chunks: list[str] = [h.text for h in hits]

        for table in related:
            for extra_hit in self._vs.search(query=table, filter_type="ddl", top_k=2):
                if extra_hit.text not in context_chunks:
                    context_chunks.append(extra_hit.text)

        logger.debug(
            "graph_retriever.context_built",
            seed_tables=seed_tables,
            fk_expanded=list(related),
            chunks=len(context_chunks),
        )
        return "\n\n".join(context_chunks)

    @staticmethod
    def _extract_table_name(ddl_chunk: str) -> str:
        """Extract the table name from a CREATE TABLE DDL string.

        Args:
            ddl_chunk: A DDL snippet starting with ``CREATE TABLE``.

        Returns:
            Table name string, or an empty string if no match is found.
        """
        match = re.search(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?(\w+)[`\"\]]?",
            ddl_chunk,
            re.IGNORECASE,
        )
        return match.group(1) if match else ""
