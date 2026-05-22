"""
aaizaql.connectors._limit
─────────────────────────
T1.4 — Shared LIMIT injection utility used by all SQL connectors.
"""
from __future__ import annotations


def inject_limit(sql: str, max_rows: int, dialect: str = "") -> tuple[str, bool]:
    """
    Append a LIMIT clause to *sql* if none is present.

    Only applies to SELECT statements — DDL (CREATE, DROP, ALTER) and DML
    (INSERT, UPDATE, DELETE) are returned unchanged.

    Returns (modified_sql, was_truncated_flag).
    Uses sqlglot for safe AST-level injection; falls back to string
    append if sqlglot is not available.
    """
    try:
        import sqlglot
        import sqlglot.expressions as exp

        tree = sqlglot.parse_one(sql, dialect=dialect or None)
        # Only inject LIMIT on SELECT queries — never on DDL/DML.
        if not isinstance(tree, exp.Select):
            return sql, False
        if tree.find(exp.Limit):
            return sql, False  # already limited
        limited = tree.limit(max_rows)
        return limited.sql(dialect=dialect or None), True
    except Exception:
        # Fallback: naive string check + append
        stripped = sql.strip().rstrip(";")
        # Only apply to SELECT statements
        first_word = stripped.split()[0].upper() if stripped.split() else ""
        if first_word != "SELECT":
            return sql, False
        if "limit" in stripped.lower().split()[-5:]:
            return sql, False
        return f"{stripped} LIMIT {max_rows}", True
