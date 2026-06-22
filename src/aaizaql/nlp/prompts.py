"""
aaizaql.nlp.prompts
───────────────────
All prompt templates used in SQL generation and correction.

Rule: prompts never live in generator.py or corrector.py — keeping them here
allows review, versioning, and override without touching pipeline logic.

Author: Ibrahim
Date: 2026-06-16
Version: 1.0.0
"""

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert SQL generator. Your ONLY job is to produce a single valid SQL SELECT statement.

STRICT RULES — never violate these:
1. Output ONLY raw SQL — no explanations, markdown, comments, or code fences.
2. Only generate SELECT queries. Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or TRUNCATE.
3. If the request cannot be solved with SELECT, output exactly: UNSUPPORTED
4. Use only tables and columns from the DATABASE SCHEMA section.
5. Always qualify columns with table names (table.column).
6. Prefer CTEs (WITH ... AS) for complex logic; avoid deeply nested subqueries.
7. Use dialect-appropriate date/time syntax from the schema hint.
"""

# ── Context Template ──────────────────────────────────────────────────────────

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

# ── Enum Block Wrapper ────────────────────────────────────────────────────────

ENUM_BLOCK_TEMPLATE = """\
--- COLUMN CODE MAPPINGS (ALWAYS use these exact numbers, never guess) ---
{enums}

"""

# ── Documentation Block Wrapper ───────────────────────────────────────────────

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

# ── Self-Correction Template ──────────────────────────────────────────────────

SELF_CORRECTION_TEMPLATE = """\
The following SQL query failed with an error. Fix it and return ONLY the corrected SQL.

DIALECT: {dialect}

ORIGINAL SQL:
{sql}

ERROR MESSAGE:
{error}

DATABASE SCHEMA (for reference):
{schema_chunks}
{enum_block}{doc_block}
Output ONLY the corrected SQL — no explanation, no markdown.\
"""

# ── NL Summary Template ───────────────────────────────────────────────────────

SUMMARY_TEMPLATE = """\
A user asked: "{question}"

The query returned {row_count} rows. Here is a sample of the data:
{data_sample}

Write 1–2 sentences summarising the key insight from this result.
Be direct and specific. Use numbers. Do not say "the data shows" or "based on the results".\
"""
