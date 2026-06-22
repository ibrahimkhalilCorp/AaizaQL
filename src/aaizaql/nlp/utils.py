"""
aaizaql.nlp.utils
─────────────────
Shared NLP utility functions used by both generator.py and corrector.py.

Author: Ibrahim
Date: 2026-06-16
Version: 1.0.0
"""

import re


def parse_sql_response(raw: str, use_cot: bool = False) -> str:
    """Strip markdown fences and extract SQL from a raw LLM response.

    Handles the following response formats:

    - Triple-backtick fences: `` ```sql\\nSELECT ...\\n``` ``
    - Plain backtick fences: `` ``` ``
    - Single-backtick wrapping: `` `SELECT ...` ``
    - Chain-of-thought ``[SQL]`` markers (when ``use_cot=True``)

    Args:
        raw: Raw text response from the LLM.
        use_cot: When ``True``, splits on the ``[SQL]`` marker and discards
            the reasoning section before extracting SQL.

    Returns:
        Clean SQL string with fences, markers, and surrounding whitespace
        removed.
    """
    if use_cot and "[SQL]" in raw:
        raw = raw.split("[SQL]", 1)[1]
    raw = re.sub(r"^\s*```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.IGNORECASE)
    return raw.strip().strip("`").strip()
