"""
aaizaql.nlp.decomposer
──────────────────────
QueryDecomposer: split complex questions into simpler sub-queries using
Leiden community detection on the schema graph.

Decomposition is best-effort. If the LLM fails or returns no valid sub-questions,
the original question is returned unchanged so the pipeline can still attempt
a single-pass answer.

Author: Ibrahim
Date: 2026-06-16
Version: 1.0.0
"""

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

DECOMPOSE_TEMPLATE = """The following question is complex and may require multiple sub-queries.
Break it into 2–4 simpler questions, each answerable with a single SQL SELECT.

ORIGINAL QUESTION:
{question}

DATABASE SCHEMA:
{schema_chunks}

Output each sub-question on a new line prefixed with "Q:".
Example:
Q: Total orders per customer in 2024
Q: Customer names and IDs
"""


class QueryDecomposer:
    """Decomposes a complex natural-language question into simpler sub-questions.

    Uses the schema graph's Leiden communities to guide decomposition — the LLM
    is prompted with relevant schema context so it can split the question along
    natural table boundaries.

    Args:
        llm: LLM provider used to generate sub-questions.
        graph_store: Schema graph store for community-guided decomposition.
        vector_store: Vector store for retrieving relevant DDL schema context.
    """

    def __init__(self, llm: Any, graph_store: Any, vector_store: Any) -> None:
        self._llm = llm
        self._gs = graph_store
        self._vs = vector_store

    def decompose(self, question: str, tenant_id: str) -> list[str]:
        """Return a list of sub-questions derived from *question*.

        Falls back to ``[question]`` when decomposition fails or the LLM
        returns no valid ``Q:``-prefixed lines, so the pipeline always has
        at least one question to process.

        Args:
            question: Complex natural language question to decompose.
            tenant_id: Tenant identifier passed through for multi-tenant
                graph store lookups.

        Returns:
            List of simpler sub-questions, or ``[question]`` on failure.
        """
        schema_chunks = self._get_schema_context(question)
        prompt = DECOMPOSE_TEMPLATE.format(
            question=question,
            schema_chunks=schema_chunks,
        )
        try:
            raw = self._llm.complete(prompt)
            sub_questions = [
                line[2:].strip() for line in raw.splitlines() if line.strip().startswith("Q:")
            ]
            return sub_questions or [question]
        except Exception as exc:
            logger.warning("decomposer.failed", detail=str(exc)[:80])
            return [question]

    def _get_schema_context(self, question: str) -> str:
        """Retrieve DDL schema chunks relevant to *question*.

        Args:
            question: User's question used as the vector search query.

        Returns:
            Newline-separated DDL string, or ``"(no schema)"`` when the
            vector store returns no results.
        """
        hits = self._vs.search(query=question, filter_type="ddl", top_k=5)
        return "\n\n".join(h.text for h in hits) or "(no schema)"
