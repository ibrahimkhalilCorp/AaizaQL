"""
aqlix.visualization.summarizer
────────────────────────────────
NLSummarizer: generates a plain-English insight from query results.
"""

from __future__ import annotations

import pandas as pd

from aqlix.llm.base import LLMProvider
from aqlix.nlp.prompts import SUMMARY_TEMPLATE
import structlog

logger = structlog.get_logger(__name__)

_MAX_SAMPLE_ROWS = 10


class NLSummarizer:
    """Produces a 1-2 sentence insight from a query result DataFrame."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def summarize(self, question: str, data: pd.DataFrame) -> str:
        """
        Parameters
        ----------
        question : str           The original natural language question.
        data     : pd.DataFrame  The query result.

        Returns
        -------
        str  A plain-English summary sentence.
        """
        if data.empty:
            return "The query returned no results."

        sample = data.head(_MAX_SAMPLE_ROWS).to_string(index=False, max_cols=8)
        prompt = SUMMARY_TEMPLATE.format(
            question=question,
            row_count=len(data),
            data_sample=sample,
        )

        try:
            summary = self._llm.complete(prompt).strip()
            logger.debug("summarizer.done", chars=len(summary))
            return summary
        except Exception as exc:
            logger.warning("summarizer.failed", detail=str(exc)[:80])
            return f"Query returned {len(data)} rows."
