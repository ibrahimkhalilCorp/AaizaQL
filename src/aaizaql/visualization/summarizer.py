"""
aaizaql.visualization.summarizer
──────────────────────────────────
NLSummarizer: sends a data sample to the LLM and returns a 1-2 sentence
plain-English insight describing what the result means.

If the LLM call fails for any reason the summarizer returns an empty string
— it is a nice-to-have feature, not a blocker for the query pipeline.

Author : Ibrahim
Date   : 2024-01-01
Version: 1.0.0
"""

import pandas as pd
import structlog

from aaizaql.nlp.prompts import SUMMARY_TEMPLATE

logger = structlog.get_logger(__name__)

_SAMPLE_ROWS = 5  # max rows sent to LLM for summarisation
_SAMPLE_COLS = 6  # max columns included in sample


class NLSummarizer:
    """
    Generate a plain-English summary from a query result DataFrame.

    Uses the configured LLM provider — the same object used by the SQL
    generator — so no additional API credentials are required.
    """

    def __init__(self, llm: object) -> None:
        """
        Initialise the summarizer with an LLM provider.

        Parameters
        ----------
        llm : BaseLLMProvider
            An LLM provider instance that exposes a ``complete(prompt)``
            method.  Typed as ``object`` to avoid a circular import with
            ``llm/base.py``.
        """
        self._llm = llm  # type: ignore[assignment]

    def summarize(self, question: str, data: pd.DataFrame) -> str:
        """
        Return a 1-2 sentence natural-language summary of the query result.

        Sends a compact data sample to the LLM via ``SUMMARY_TEMPLATE``.
        Returns an empty string on any error so the caller is never blocked.

        Parameters
        ----------
        question : str
            The original natural-language question that produced *data*.
        data     : pd.DataFrame
            The query result to summarise.

        Returns
        -------
        str
            A plain-English insight, or ``""`` when the LLM call fails or
            the DataFrame is empty.
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
        """
        Build a compact text representation of the first few rows and columns.

        Keeps the sample small so it fits comfortably in the LLM context window
        without wasting tokens on large result sets.

        Parameters
        ----------
        data : pd.DataFrame
            Full query result DataFrame.

        Returns
        -------
        str
            A fixed-width string table of up to ``_SAMPLE_ROWS`` rows and
            ``_SAMPLE_COLS`` columns, without the row index.
        """
        subset = data.iloc[:_SAMPLE_ROWS, :_SAMPLE_COLS]
        return subset.to_string(index=False, max_colwidth=40)
