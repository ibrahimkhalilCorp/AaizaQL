"""

aaizaql.nlp.graph_retriever

────────────────────────────

T4.3 — GraphRAG Retriever: vector seed → FK traversal → enriched DDL context.

T4.4 — Used by both SQLGenerator (initial context) and SelfCorrector (retry).

"""

from __future__ import annotations

import re

import structlog

logger = structlog.get_logger(__name__)

class GraphRetriever:

    """

    1. Vector search → top-k DDL chunks → extract seed table names

    2. BFS from seed tables over FK edges (depth=2) → related tables

    3. Format as enriched DDL context for the LLM prompt

    """

    def __init__(self, vector_store: object, graph_store: object, top_k: int = 5) -> None:

        self._vs = vector_store

        self._gs = graph_store

        self._top_k = top_k

    def get_context(self, query: str, tenant_id: str) -> str:

        """Return enriched schema context string for the given query."""

        # Step 1 — vector seed

        hits = self._vs.search(query=query, filter_type="ddl", top_k=self._top_k)

        if not hits:

            return "(no schema ingested yet)"

        seed_tables = [self._extract_table_name(h.text) for h in hits]

        seed_tables = [t for t in seed_tables if t]

        # Step 2 — FK expansion

        related: set[str] = set()

        for table in seed_tables:

            related.update(self._gs.get_fk_neighbors(table, tenant_id, depth=2))

        # Step 3 — collect all relevant DDL chunks

        context_chunks: list[str] = []

        # Always include seed chunks first

        for hit in hits:

            context_chunks.append(hit.text)

        # Fetch FK-related chunks from vector store

        if related:

            for table in related:

                extra = self._vs.search(

                    query=table, filter_type="ddl", top_k=2

                )

                for h in extra:

                    if h.text not in context_chunks:

                        context_chunks.append(h.text)

        logger.debug(

            "graph_retriever.context_built",

            seed_tables=seed_tables,

            fk_expanded=list(related),

            chunks=len(context_chunks),

        )

        return "\n\n".join(context_chunks)

    @staticmethod

    def _extract_table_name(ddl_chunk: str) -> str:

        m = re.search(

            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?(\w+)[`\"\]]?",

            ddl_chunk, re.IGNORECASE,

        )

        return m.group(1) if m else ""
