"""
aaizaql.core.engine
───────────────────
QueryEngine — the single public entry point for the entire library.

Quick start::

    from aaizaql import QueryEngine

    engine = QueryEngine(llm="groq", database="sqlite", dsn="sqlite:///my.db")
    engine.ingest_schema()

    # Optional training (improves accuracy)
    engine.train(documentation="employees.status: 1=Active, 2=Resigned...")
    engine.train(question="Top 5 employees by sales", sql="SELECT ...")
    engine.define_enum("employees", "status", {1: "Active", 2: "Resigned"})

    result = engine.query("Show all active employees")

Author: Ibrahim
Date: 2026-06-15
Version: 1.0.0
"""

import time
import uuid
from dataclasses import dataclass
from typing import Any

import pandas as pd
import structlog

from aaizaql.connectors import REGISTRY
from aaizaql.core.config import Settings, make_settings
from aaizaql.core.exceptions import ConnectorNotFound, SQLGenerationError, UnsupportedQueryError
from aaizaql.core.rate_limiter import RateLimiter
from aaizaql.llm import build_llm_provider
from aaizaql.memory.context import ContextManager
from aaizaql.memory.vector_store import VectorStoreAdapter
from aaizaql.nlp.corrector import SelfCorrector
from aaizaql.nlp.generator import SQLGenerator
from aaizaql.schema.ingestion import SchemaIngester
from aaizaql.schema.semantic_store import SemanticStore
from aaizaql.security.validator import SQLValidator
from aaizaql.visualization.renderer import ResultRenderer
from aaizaql.visualization.summarizer import NLSummarizer

logger = structlog.get_logger(__name__)


@dataclass
class QueryResult:
    """Everything returned from a single ``engine.query()`` call.

    Attributes:
        question: The original natural language question.
        sql: The final SQL that was executed (may differ from the first generated
            SQL if self-correction triggered a retry).
        data: Query result as a DataFrame.
        summary: LLM-generated plain-English summary of the results.
        chart: Plotly figure object, or None if no chart was generated.
        execution_time_ms: Wall-clock time from question to result, in milliseconds.
        session_id: UUID identifying the conversation turn.
        was_corrected: True if at least one self-correction retry occurred.
        correction_attempts: Number of correction retries that were made.
        truncated: True if the result was capped at ``max_result_rows``.
    """

    question: str
    sql: str
    data: pd.DataFrame
    summary: str = ""
    chart: Any = None
    execution_time_ms: int = 0
    session_id: str = ""
    was_corrected: bool = False
    correction_attempts: int = 0
    truncated: bool = False


class QueryEngine:
    """Main entry point for the AaizaQL library.

    Accepts a natural language question, generates SQL via an LLM, executes it
    against a connected database, and returns a :class:`QueryResult` with the
    data, a chart, and a plain-English summary.

    Args:
        llm: LLM provider name. One of: ``"claude"``, ``"openai"``, ``"groq"``,
            ``"ollama"``, ``"deepseek"``, ``"perplexity"``, ``"gemini"``,
            ``"mistral"``.
        database: Database connector name. One of: ``"sqlite"``, ``"postgresql"``,
            ``"mysql"``, ``"mssql"``, ``"oracle"``, ``"duckdb"``, ``"snowflake"``,
            ``"bigquery"``, ``"mongodb"``.
        dsn: SQLAlchemy-compatible connection string for the database.
        settings: Pre-built :class:`~aaizaql.core.config.Settings` instance.
            When provided, ``llm`` and ``**kwargs`` are ignored.
        **kwargs: Any :class:`~aaizaql.core.config.Settings` field name, e.g.
            ``groq_model="llama-3.1-8b-instant"``, ``llm_timeout_seconds=60``.

    Example::

        engine = QueryEngine(
            llm="groq",
            database="postgresql",
            dsn="postgresql+psycopg2://user:pass@localhost/mydb",
            groq_model="llama-3.3-70b-versatile",
        )
    """

    def __init__(
        self,
        llm: str = "groq",
        database: str = "sqlite",
        dsn: str = "",
        settings: Settings | None = None,
        **kwargs: Any,
    ) -> None:
        # Never mutate the module-level singleton: doing so causes cross-engine
        # bleed in multi-engine / multi-threaded use.
        self._settings = (
            settings if settings is not None else make_settings(llm_provider=llm, **kwargs)
        )

        if database not in REGISTRY:
            raise ConnectorNotFound(database, available=sorted(REGISTRY.keys()))

        self._connector = REGISTRY[database]()
        if dsn:
            self._connector.connect(dsn)
            logger.info("database.connected", connector=database, dsn_hint=dsn[:40])

        self._llm = build_llm_provider(llm, self._settings)

        # Deferred: importing chromadb is expensive. Components that need the
        # vector store are wired up via _ensure_vector_store() on first use.
        self._vector_store: VectorStoreAdapter | None = None

        # SemanticStore and SchemaIngester are cheap to construct; their
        # internal _vs reference is patched to the real VectorStoreAdapter
        # on first use, so define_enum() works before ingest_schema().
        self._semantic = SemanticStore(None)  # type: ignore[arg-type]
        self._ingester = SchemaIngester(None)  # type: ignore[arg-type]

        self._context = ContextManager(limit=self._settings.session_history_limit)
        self._generator = SQLGenerator(
            self._llm,
            None,  # type: ignore[arg-type]  # patched on first use
            self._settings,
            self._semantic,
            connector=self._connector,
        )
        self._validator = SQLValidator(self._settings)
        self._corrector = SelfCorrector(
            self._llm,
            self._settings,
            validator=self._validator,
            vector_store=None,  # type: ignore[arg-type]  # patched on first use
            semantic_store=self._semantic,
            dialect=self._connector.name,
        )
        self._renderer = ResultRenderer()
        self._summarizer = NLSummarizer(self._llm)
        self._rate_limiter = RateLimiter(qpm=self._settings.rate_limit_qpm)

        logger.info("engine.ready", llm=llm, database=database)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_vector_store(self) -> VectorStoreAdapter:
        """Return the VectorStoreAdapter, initialising it on first call.

        Deferred construction means :meth:`__init__` does not import chromadb at
        startup. The first call to :meth:`ingest_schema`, :meth:`train`, or
        :meth:`query` triggers the import; if chromadb is missing the user gets a
        clear ``VectorStoreError`` pointing to ``pip install "aaizaql[rag]"``
        rather than an obscure ``ImportError`` during engine construction.

        Returns:
            The initialised :class:`~aaizaql.memory.vector_store.VectorStoreAdapter`.
        """
        if self._vector_store is None:
            self._vector_store = VectorStoreAdapter(self._settings)
            self._semantic._vs = self._vector_store  # type: ignore[attr-defined]
            self._ingester._vs = self._vector_store  # type: ignore[attr-defined]
            self._generator._vs = self._vector_store  # type: ignore[attr-defined]
            self._corrector._vector_store = self._vector_store
        return self._vector_store

    # ── Schema ────────────────────────────────────────────────────────────────

    def ingest_schema(self) -> int:
        """Read the connected database schema and store it in the vector store.

        Call once before the first :meth:`query`, and again after schema changes.

        Returns:
            Number of schema chunks indexed.
        """
        self._ensure_vector_store()
        logger.info("schema.ingesting")
        count = self._ingester.ingest_from_database(self._connector)
        logger.info("schema.ingested", chunks=count)
        return count

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        documentation: str | None = None,
        question: str | None = None,
        sql: str | None = None,
    ) -> None:
        """Train the engine with business knowledge.

        Three modes — use any combination:

        1. **Documentation** (free-text business context)::

                engine.train(documentation='''
                    employees.status: 1=Active, 2=On Leave, 3=Resigned, 4=Terminated
                    Use strftime('%Y-%m', order_date) for SQLite month grouping.
                    business_unit_id: 4=ACCL, 8=APFIL, 12=IBOS
                ''')

        2. **Q→SQL pair** (sample question with correct SQL)::

                engine.train(
                    question="Top 5 employees by total sales",
                    sql="SELECT e.name, SUM(s.total_amount) FROM employees e ...",
                )

        3. **Both at once**::

                engine.train(documentation="...", question="...", sql="...")

        Args:
            documentation: Free-text business rules, column explanations, or
                domain knowledge to embed in the vector store.
            question: Example natural language question (must be paired with ``sql``).
            sql: Correct SQL for the paired ``question``.
        """
        self._ensure_vector_store()

        if documentation is not None:
            self._semantic.train_documentation(documentation)

        if question is not None and sql is not None:
            self._semantic.train_sql_pair(question, sql)
        elif question is not None or sql is not None:
            logger.warning(
                "engine.train.incomplete_pair",
                detail="Both 'question' and 'sql' are required together. Skipping pair.",
            )

    def define_enum(
        self,
        table: str,
        column: str,
        mapping: dict[int | str, str],
    ) -> None:
        """Register a numeric code → label mapping for a column.

        Unlike documentation stored in the vector store, enum mappings are
        injected into every prompt — no RAG retrieval miss is possible.

        Args:
            table: Table name the column belongs to.
            column: Column name that holds the numeric codes.
            mapping: Dict of ``{code: label}`` pairs, e.g.
                ``{1: "Active", 2: "Resigned"}``.

        Example::

            engine.define_enum("employees", "status", {
                1: "Active",
                2: "On Leave",
                3: "Resigned",
                4: "Terminated",
            })
        """
        self._semantic.define_enum(table, column, mapping)

    def teach(self, question: str, sql: str) -> None:
        """Shortcut for ``engine.train(question=..., sql=...)``.

        Args:
            question: Natural language question.
            sql: Correct SQL for the question.
        """
        self._semantic.train_sql_pair(question, sql)
        logger.info("engine.taught", question=question[:60])

    def training_info(self) -> dict[str, object]:
        """Return a summary of all training data currently loaded.

        Returns:
            Dict with keys ``"enums"`` (mapping of ``"table.column"`` → codes)
            and ``"enum_count"`` (total number of registered enum columns).
        """
        enums_list = self._semantic.list_enums()
        enums_dict = {f"{e['table']}.{e['column']}": e["mapping"] for e in enums_list}
        return {
            "enums": enums_dict,
            "enum_count": self._semantic.enum_count(),
        }

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        session_id: str | None = None,
    ) -> QueryResult:
        """Convert a natural language question to SQL and execute it.

        Args:
            question: Natural language question to answer.
            session_id: Pass the same ID across turns to enable multi-turn memory.
                A new UUID is generated when omitted.

        Returns:
            :class:`QueryResult` with ``.sql``, ``.data``, ``.chart``,
            and ``.summary``.

        Raises:
            UnsupportedQueryError: When the LLM determines the question cannot
                be answered with SQL.
            SQLGenerationError: When the LLM returns an empty response.
            RateLimitError: When the query rate for this session exceeds
                ``rate_limit_qpm``.
        """
        sid = session_id or str(uuid.uuid4())
        t_start = time.monotonic()
        logger.info("query.start", session_id=sid, question=question[:80])

        self._ensure_vector_store()
        self._rate_limiter.check(sid)
        self._validator.check_question(question)

        history = self._context.get_history(sid)
        sql = self._generator.generate(question, history)

        if sql.strip().upper() == "UNSUPPORTED":
            raise UnsupportedQueryError(question)
        if not sql:
            raise SQLGenerationError(question, "LLM returned an empty response.")

        if self._connector.requires_sql_validation:
            self._validator.validate(sql)

        data, was_corrected, attempts = self._corrector.execute_with_correction(
            sql=sql,
            executor=self._connector,
            question=question,
        )
        sql = self._corrector.last_sql

        chart = self._renderer.render(data, question)
        summary = self._summarizer.summarize(question, data)
        self._context.add_turn(sid, question=question, sql=sql, row_count=len(data))

        execution_ms = int((time.monotonic() - t_start) * 1000)
        logger.info(
            "query.complete",
            session_id=sid,
            rows=len(data),
            execution_ms=execution_ms,
            was_corrected=was_corrected,
        )

        return QueryResult(
            question=question,
            sql=sql,
            data=data,
            summary=summary,
            chart=chart,
            execution_time_ms=execution_ms,
            session_id=sid,
            was_corrected=was_corrected,
            correction_attempts=attempts,
        )

    # ── Utility ───────────────────────────────────────────────────────────────

    def health_check(self) -> dict:
        """Run health checks on all subsystems.

        Returns:
            Dict with a status entry for each subsystem (LLM, DB, vector store).
        """
        from aaizaql.api.health import run_health_check

        return run_health_check(self)

    def reset_session(self, session_id: str) -> None:
        """Clear conversation memory for a session.

        Args:
            session_id: The session whose history should be erased.
        """
        self._context.clear(session_id)

    def close(self) -> None:
        """Close the database connection and release resources."""
        self._connector.close()
        logger.info("engine.closed")

    def __enter__(self) -> "QueryEngine":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
