"""
AAIZAQL.cli
──────────
Command-line interface for AAIZAQL.

Usage examples::

    # Interactive REPL against a local SQLite file
    AAIZAQL query --db sqlite:///mydata.db --llm groq

    # Single question (non-interactive)
    AAIZAQL query --db sqlite:///mydata.db --llm groq -q "How many users signed up last month?"

    # Ingest schema only (useful for CI / pre-caching)
    AAIZAQL ingest --db sqlite:///mydata.db

    # Print version
    AAIZAQL version
"""

from __future__ import annotations

import sys


def _require(pkg: str, install: str) -> None:
    """Print a friendly error if a package is missing."""
    try:
        __import__(pkg)
    except ImportError:
        print(f"[AAIZAQL] Missing dependency: {pkg}")
        print(f"        Install with:  pip install {install}")
        sys.exit(1)


def cmd_version(_args: object) -> None:
    from AAIZAQL import __version__

    print(f"AAIZAQL {__version__}")


def cmd_ingest(args: object) -> None:
    """Ingest the schema of the connected database into the vector store."""
    from AAIZAQL import QueryEngine

    print(f"[AAIZAQL] Connecting to {args.db} …")  # type: ignore[attr-defined]
    engine = QueryEngine(
        llm=args.llm,  # type: ignore[attr-defined]
        database=args.database,  # type: ignore[attr-defined]
        dsn=args.db,  # type: ignore[attr-defined]
    )
    count = engine.ingest_schema()
    print(f"[AAIZAQL] Schema ingested: {count} table chunk(s) stored.")
    engine.close()


def cmd_query(args: object) -> None:
    """Run a natural language query (interactive REPL or single question)."""
    from AAIZAQL import QueryEngine

    _require("chromadb", "chromadb")

    print(f"[AAIZAQL] Connecting with LLM={args.llm} DB={args.db} …")  # type: ignore[attr-defined]
    try:
        engine = QueryEngine(
            llm=args.llm,  # type: ignore[attr-defined]
            database=args.database,  # type: ignore[attr-defined]
            dsn=args.db,  # type: ignore[attr-defined]
        )
    except Exception as exc:
        print(f"[AAIZAQL] Connection failed: {exc}")
        sys.exit(1)

    engine.ingest_schema()
    print("[AAIZAQL] Schema ready.\n")

    question: str | None = getattr(args, "question", None)

    if question:
        # Non-interactive single question
        _run_once(engine, question)
    else:
        # Interactive REPL
        print("Type your question and press Enter.  Type 'exit' or Ctrl-C to quit.\n")
        while True:
            try:
                q = input("AAIZAQL> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n[AAIZAQL] Bye!")
                break
            if not q:
                continue
            if q.lower() in {"exit", "quit", "q"}:
                print("[AAIZAQL] Bye!")
                break
            _run_once(engine, q)
            print()

    engine.close()


def _run_once(engine: object, question: str) -> None:
    from AAIZAQL import AAIZAQLError, QueryEngine

    assert isinstance(engine, QueryEngine)
    try:
        result = engine.query(question)
        print(f"\n  SQL: {result.sql}\n")
        if result.summary:
            print(f"  {result.summary}\n")
        print(result.data.to_string(index=False))
        print(f"\n  ({len(result.data)} rows, {result.execution_time_ms} ms)")
    except AAIZAQLError as exc:
        print(f"\n  [error] {exc}")


def main() -> None:
    """Entry point registered in pyproject.toml."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="AAIZAQL",
        description="AAIZAQL — Natural Language to SQL",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── version ──────────────────────────────────────────────────────────────
    sub.add_parser("version", help="Print version and exit.")

    # ── ingest ───────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Ingest database schema into vector store.")
    p_ingest.add_argument("--db", required=True, help="DSN, e.g. sqlite:///my.db")
    p_ingest.add_argument("--llm", default="groq", help="LLM provider (groq/claude/openai/ollama)")
    p_ingest.add_argument("--database", default="sqlite", help="DB connector (sqlite/postgresql/…)")

    # ── query ─────────────────────────────────────────────────────────────────
    p_query = sub.add_parser("query", help="Run natural language queries.")
    p_query.add_argument("--db", required=True, help="DSN, e.g. sqlite:///my.db")
    p_query.add_argument("--llm", default="groq", help="LLM provider (groq/claude/openai/ollama)")
    p_query.add_argument("--database", default="sqlite", help="DB connector (sqlite/postgresql/…)")
    p_query.add_argument(
        "-q", "--question", default=None, help="Single question (omit for interactive REPL)"
    )

    args = parser.parse_args()

    if args.command == "version":
        cmd_version(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
