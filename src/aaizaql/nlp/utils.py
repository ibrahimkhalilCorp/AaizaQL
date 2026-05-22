"""
aaizaql.nlp.utils
────────────────
Shared NLP utility functions used by both generator.py and corrector.py.
"""

from __future__ import annotations

import re


def parse_sql_response(raw: str, use_cot: bool = False) -> str:
    """
    T1.3 — Strip markdown fences and extract SQL from an LLM response.

    Handles:
      - Triple-backtick fences:  ```sql\nSELECT ...\n```
      - Single-backtick wrapping: `SELECT ...`
      - Chain-of-thought [SQL] markers
    """
    if use_cot and "[SQL]" in raw:
        raw = raw.split("[SQL]", 1)[1]
    # Remove opening fence:  ```sql  or  ```
    raw = re.sub(r"^\s*```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    # Remove closing fence
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.IGNORECASE)
    # Remove single-backtick wrapping
    raw = raw.strip().strip("`").strip()
    return raw
