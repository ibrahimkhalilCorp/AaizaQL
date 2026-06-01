# Contributing to AAIZAQL

Contributions are welcome — bug fixes, new connectors, LLM providers, docs, and tests all help.
This guide covers everything you need to go from zero to a merged pull request.

---

## Table of contents

- [Getting started](#getting-started)
- [Project structure](#project-structure)
- [Running tests](#running-tests)
- [Code style](#code-style)
- [Types of contributions](#types-of-contributions)
- [Pull request process](#pull-request-process)
- [Commit message format](#commit-message-format)
- [Reporting bugs](#reporting-bugs)

---

## Getting started

**Requirements:** Python 3.11+, Git.

```bash
git clone https://github.com/ibrahimkhalilCorp/AAIZAQL
cd AAIZAQL
pip install -e ".[dev]"
pre-commit install
```

That installs the library in editable mode with all dev dependencies (pytest, black, ruff, mypy)
and registers the pre-commit hooks so formatting runs automatically before every commit.

Verify the setup works:

```bash
pytest tests/unit/          # fast, no API key needed
```

---

## Project structure

```
src/AAIZAQL/
├── core/          # QueryEngine, Settings, exceptions — the public API
├── nlp/           # SQLGenerator, SelfCorrector, prompts
├── memory/        # ContextManager, VectorStoreAdapter
├── security/      # SQLValidator, AuditLogger
├── connectors/    # One file per database (sqlite, postgres, mysql, …)
├── llm/           # One file per LLM provider (claude, openai, groq, ollama)
├── federation/    # FederationCoordinator, QueryPlanner (Phase 3)
├── schema/        # SchemaIngester, SemanticLayer
├── visualization/ # ResultRenderer, NLSummarizer
└── cli.py         # `AAIZAQL` command-line entry point
tests/
├── unit/          # Pure unit tests — no DB, no API key, always fast
├── integration/   # Tests against a real SQLite database
└── conftest.py    # Shared fixtures (sqlite_db, mock_llm, vector_store, …)
```

The most important rule: **every layer has one job.**
The NLP layer produces SQL. The security layer validates it. The execution layer runs it.
Never mix concerns across layers.

---

## Running tests

```bash
# Unit tests only (fast, no external deps)
pytest tests/unit/

# Full test suite (requires chromadb and sentence-transformers)
pytest tests/

# With coverage report
pytest tests/ --cov=src/AAIZAQL --cov-report=term-missing

# Run a specific file
pytest tests/unit/test_validator.py -v
```

Tests marked `@pytest.mark.requires_api_key` are automatically skipped when no
`ANTHROPIC_API_KEY` or `OPENAI_API_KEY` environment variable is set. To run them locally:

```bash
export AAIZAQL_GROQ_API_KEY="gsk_..."
pytest tests/ -m requires_api_key
```

---

## Code style

The CI pipeline enforces all of these automatically. Running `pre-commit install` means
they also run locally before every commit.

| Tool | What it does | Config |
|---|---|---|
| `black` | Formats code | `pyproject.toml` → `[tool.black]` |
| `ruff` | Lints and auto-fixes | `pyproject.toml` → `[tool.ruff]` |
| `mypy` | Type-checks (non-strict) | `pyproject.toml` → `[tool.mypy]` |

Line length is **100 characters**. Use type hints everywhere. Avoid bare `Exception` —
raise the specific exception from `AAIZAQL.core.exceptions` that matches the failure.

To run checks manually:

```bash
black src/ tests/
ruff check src/ tests/ --fix
mypy src/
```

---

## Types of contributions

### Adding a new database connector

1. Create `src/AAIZAQL/connectors/<name>.py` — subclass `DatabaseConnector` and implement
   `connect()`, `execute()`, `get_schema()`, and `close()`. Follow the pattern in
   `connectors/sqlite.py` or `connectors/postgres.py`.

2. Add a lazy-import block in `src/AAIZAQL/connectors/__init__.py`:

   ```python
   try:
       from AAIZAQL.connectors.bigquery import BigQueryConnector
       REGISTRY["bigquery"] = BigQueryConnector
   except ImportError:
       pass
   ```

3. Add the driver to `pyproject.toml` as an optional dependency:

   ```toml
   [project.optional-dependencies]
   bigquery = ["google-cloud-bigquery>=3.0"]
   ```

4. Write tests in `tests/test_connectors.py` following the existing pattern.

### Adding a new LLM provider

1. Create `src/AAIZAQL/llm/<name>_provider.py` — subclass `LLMProvider` and implement
   `complete()`. Follow the pattern in `llm/groq_provider.py`.

2. Register it in `src/AAIZAQL/llm/__init__.py`.

3. Add the SDK to `pyproject.toml` as an optional dependency.

4. Update the `Supported LLM Providers` table in `README.md`.

### Fixing a bug

- Open an issue first if the fix is non-trivial, so we can agree on the approach.
- Include a failing test that reproduces the bug in your PR.
- The fix and the test should be in the same commit.

### Improving documentation

- Docs changes (README, docstrings, examples) are always welcome without an issue first.
- Example notebooks live in `examples/` — follow the naming convention
  `NN_short_description.py` (e.g. `04_snowflake_quickstart.py`).

---

## Pull request process

1. **Fork** the repo and create a branch from `main`:

   ```bash
   git checkout -b feat/bigquery-connector
   ```

2. **Write tests** for your change. PRs without tests for new behaviour will be asked
   to add them before merging.

3. **Make sure CI passes** locally before pushing:

   ```bash
   pytest tests/unit/
   ruff check src/ tests/
   black --check src/ tests/
   ```

4. **Open a PR** against `main`. Fill in the PR template:
   - What does this change?
   - Why is it needed?
   - How was it tested?

5. **One approval** from a maintainer is required to merge. Small fixes (typos, docs)
   can be merged by a maintainer directly.

6. Squash-merge is preferred for feature branches to keep the `main` history clean.

---

## Commit message format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short summary in sentence case>

<optional body explaining why, not what>
```

Common types:

| Type | When to use |
|---|---|
| `feat` | New feature or connector |
| `fix` | Bug fix |
| `test` | Adding or fixing tests |
| `docs` | Documentation only |
| `style` | Formatting, no logic change |
| `refactor` | Code change with no behaviour change |
| `chore` | Build scripts, CI, dependencies |

Examples:

```
feat: add BigQuery connector
fix: handle empty result set in NLSummarizer
docs: add Snowflake quickstart example
test: add unit tests for MySQLConnector DSN parser
```

---

## Reporting bugs

Open a [GitHub issue](https://github.com/ibrahimkhalilCorp/AAIZAQL/issues) and include:

- AAIZAQL version (`pip show AAIZAQL`)
- Python version (`python --version`)
- Database and LLM provider being used
- The question you asked and the SQL that was generated (if any)
- The full error traceback

For security vulnerabilities, do **not** open a public issue.
Email the maintainer directly (see the `authors` field in `pyproject.toml`).

---

MIT License — by contributing you agree your work will be released under the same license.
---

## Checklist: removing or renaming a class/function

> This checklist exists because of a real incident in v0.2.2: `_SentenceEmbedder`
> was removed from `ingestion.py` but `test_rag_optional.py` still imported it.
> The tests passed in CI only because that specific test file was not being exercised
> in the failing path. Follow these steps every time you remove or rename any symbol.

### When you remove a symbol from source

1. **Search tests immediately:**
   ```bash
   grep -r "SymbolName" tests/
   ```
   Update or remove every match before opening a PR.

2. **Run the import validator:**
   ```bash
   python scripts/check_test_imports.py
   ```
   This must exit 0 before you commit. The pre-commit hook runs it automatically,
   but run it manually after any rename/removal to catch issues early.

3. **Check `__init__.py`:**
   If the symbol was exported from `aaizaql.__init__` or any sub-package
   `__init__.py`, remove it from there too (and from `__all__`).

4. **Run the public API test:**
   ```bash
   pytest tests/unit/test_public_api.py -v
   ```
   This catches any `__all__` entries that no longer exist in source.

5. **Update `CHANGELOG.md`:**
   Add a `### Removed` entry so downstream users know what broke and why.

6. **Add a migration note** if the symbol was part of the public API (`__all__`):
   ```markdown
   ### Removed
   - `OldName` — replaced by `NewName`. Update: `from aaizaql import NewName`.
   ```

### When you rename a symbol

Follow all the steps above, and additionally:

- Add a deprecation alias in the **old location** for one minor version before removal:
  ```python
  # Deprecated in 0.3.0 — remove in 0.4.0
  OldName = NewName
  ```
- Add a `DeprecationWarning` so users see it at runtime:
  ```python
  import warnings
  def OldName(*args, **kwargs):
      warnings.warn("OldName is deprecated, use NewName", DeprecationWarning, stacklevel=2)
      return NewName(*args, **kwargs)
  ```

### Pre-commit hook

The `check-test-imports` pre-commit hook runs `scripts/check_test_imports.py`
automatically before every commit. Install it once with:

```bash
pre-commit install
```

If it fails, fix the broken imports before the commit goes through.
