"""
aaizaql.connectors._limit
─────────────────────────
Shared LIMIT injection utility used by SQL connectors to cap result rows.

Prevents runaway queries from exhausting memory by appending a LIMIT clause
when none is present. Uses sqlglot for AST-level injection when available,
with a string-based fallback for environments without sqlglot.

Author: Ibrahim
Date: 2026-06-15
Version: 1.0.0
"""


def inject_limit(sql: str, max_rows: int, dialect: str = "") -> tuple[str, bool]:
    """Append a LIMIT clause to *sql* if none is already present.

    Only applies to SELECT statements — DDL (CREATE, DROP, ALTER) and DML
    (INSERT, UPDATE, DELETE) are returned unchanged and are never truncated.

    Prefers sqlglot for safe AST-level injection. Falls back to a naive string
    append when sqlglot is not installed.

    Args:
        sql: The SQL statement to inspect and potentially modify.
        max_rows: Maximum number of rows to allow. Added as the LIMIT value.
        dialect: sqlglot dialect string (e.g. ``"sqlite"``, ``"postgres"``).
            Empty string lets sqlglot auto-detect.

    Returns:
        A tuple of ``(modified_sql, was_truncated)`` where ``was_truncated``
        is ``True`` if a LIMIT clause was injected, ``False`` if the statement
        already had one or is not a SELECT.
    """
    try:
        import sqlglot
        import sqlglot.expressions as exp

        tree = sqlglot.parse_one(sql, dialect=dialect or None)
        if not isinstance(tree, exp.Select):
            return sql, False
        if tree.find(exp.Limit):
            return sql, False
        limited = tree.limit(max_rows)
        return limited.sql(dialect=dialect or None), True
    except Exception:
        # Fallback: naive string check + append when sqlglot unavailable.
        stripped = sql.strip().rstrip(";")
        tokens = stripped.split()
        first_word = tokens[0].upper() if tokens else ""
        if first_word != "SELECT":
            return sql, False
        if "limit" in stripped.lower().split()[-5:]:
            return sql, False
        return f"{stripped} LIMIT {max_rows}", True
