"""
aaizaql.visualization.summarizer
────────────────────────────────
NLSummarizer: sends a data sample to the LLM and returns a 1-2 sentence
plain-English insight describing what the result means.

If the LLM call fails for any reason the summarizer returns an empty string
— it is a nice-to-have, not a blocker.
"""

from __future__ import annotations

import pandas as pd
import structlog

from aaizaql.nlp.prompts import SUMMARY_TEMPLATE

logger = structlog.get_logger(__name__)

_SAMPLE_ROWS = 5  # max rows sent to LLM for summarisation
_SAMPLE_COLS = 6  # max columns included in sample


class NLSummarizer:
    """
    Generate a plain-English summary from a query result DataFrame.

    Uses the configured LLM provider — same object as the SQL generator.
    """

    def __init__(self, llm: object) -> None:  # LLMProvider (Any avoids circular)
        self._llm = llm  # type: ignore[assignment]

    def summarize(self, question: str, data: pd.DataFrame) -> str:
        """
        Return a 1-2 sentence summary of the query result.
        Returns "" on any error so the caller is never blocked.
        """
        if data.empty:
            return "The query returned no results."

        try:
            sample = self._build_sample(data)
            prompt = SUMMARY_TEMPLATE.format(
                question=question,
                row_count=len(data),
                data_sample=sample,
            )
            summary = self._llm.complete(prompt)  # type: ignore[attr-defined]
            cleaned = summary.strip()
            logger.debug("summarizer.done", chars=len(cleaned))
            return cleaned
        except Exception as exc:
            logger.warning("summarizer.failed", detail=str(exc)[:80])
            return ""

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_sample(data: pd.DataFrame) -> str:
        """Return a compact CSV-like text sample for the LLM."""
        subset = data.iloc[:_SAMPLE_ROWS, :_SAMPLE_COLS]
        return subset.to_string(index=False, max_colwidth=40)
