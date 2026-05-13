"""
aqlix.nlp.prompts
──────────────────
All prompt templates used in SQL generation.

Rule: Never put prompts in generator.py — keep them here so they can be
reviewed, versioned, and overridden without touching logic.
"""

from __future__ import annotations

# ── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert SQL generator. Your ONLY job is to produce a single valid SQL SELECT statement.

STRICT RULES — never violate these:
1. Output ONLY the raw SQL statement — no explanation, no markdown, no code fences, no comments.
2. You MUST NOT produce INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, CREATE, or any DDL/DML.
3. If the question cannot be answered with SELECT, output exactly: UNSUPPORTED
4. Use only tables and columns defined in the DATABASE SCHEMA section below.
5. Always qualify column names with table names (table.column).
6. Use CTEs (WITH ... AS) for complex multi-step logic — never nested subqueries deeper than 2 levels.
7. For date/time filters, use the dialect-appropriate syntax shown in the schema dialect hint.
"""

# ── Context Template ─────────────────────────────────────────────────────────

CONTEXT_TEMPLATE = """\
--- DATABASE SCHEMA (dialect: {dialect}) ---
{schema_chunks}

{enum_block}\
{doc_block}\
--- SIMILAR PAST QUERIES (few-shot reference) ---
{example_pairs}

--- CONVERSATION HISTORY ---
{history}

--- CURRENT QUESTION ---
{question}

SQL:"""

# ── Enum block wrapper (injected only when enums exist) ───────────────────────

ENUM_BLOCK_TEMPLATE = """\
--- COLUMN CODE MAPPINGS (ALWAYS use these exact numbers, never guess) ---
{enums}

"""

# ── Documentation block wrapper ───────────────────────────────────────────────

DOC_BLOCK_TEMPLATE = """\
--- BUSINESS CONTEXT ---
{docs}

"""

# ── Chain-of-Thought Prompt ───────────────────────────────────────────────────

COT_PROMPT_PREFIX = """\
Before writing the SQL, reason through these steps:
1. Which tables are needed to answer this question?
2. What are the exact join conditions between those tables?
3. What aggregation is needed (GROUP BY, COUNT, SUM, AVG)?
4. Are there date/time filters? What format does the dialect use?
5. Are there HAVING clauses or window functions needed?

Think step by step. Then write the final SQL after the [SQL] marker.

[REASONING]
"""

COT_PROMPT_SUFFIX = "\n\n[SQL]\n"

# ── Self-Correction Template ─────────────────────────────────────────────────

SELF_CORRECTION_TEMPLATE = """\
The following SQL query failed with an error. Fix it.

ORIGINAL SQL:
{sql}

ERROR MESSAGE:
{error}

DATABASE SCHEMA (for reference):
{schema_chunks}

Output ONLY the corrected SQL — no explanation, no markdown.\
"""

# ── Query Decomposition ───────────────────────────────────────────────────────

DECOMPOSE_TEMPLATE = """\
The following question is complex and may require multiple sub-queries.
Break it into 2–4 simpler questions, each of which can be answered with a single SQL SELECT.

ORIGINAL QUESTION:
{question}

DATABASE SCHEMA:
{schema_chunks}

Output each sub-question on a new line, prefixed with "Q:".
Example:
Q: Total orders per customer in 2024
Q: Customer names and IDs
"""

# ── NL Summary Template ───────────────────────────────────────────────────────

SUMMARY_TEMPLATE = """\
A user asked: "{question}"

The query returned {row_count} rows. Here is a sample of the data:
{data_sample}

Write 1–2 sentences summarising the key insight from this result.
Be direct and specific. Use numbers. Do not say "the data shows" or "based on the results".\
"""
