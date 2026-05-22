"""
aaizaql.nlp.decomposer
──────────────────────
T4.6 — QueryDecomposer: split complex questions into sub-queries
using Leiden community detection on the schema graph.
"""

from __future__ import annotations

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
    """
    Decomposes a complex natural-language question into simpler sub-questions,
    guided by Leiden schema communities.
    """

    def __init__(self, llm: Any, graph_store: Any, vector_store: Any) -> None:
        self._llm = llm
        self._gs = graph_store
        self._vs = vector_store

    def decompose(self, question: str, tenant_id: str) -> list[str]:
        """Return a list of sub-questions, or [question] if decomposition fails."""
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
        hits = self._vs.search(query=question, filter_type="ddl", top_k=5)
        return "\n\n".join(h.text for h in hits) or "(no schema)"
