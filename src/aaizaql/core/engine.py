"""
aaizaql.core.engine
─────────────────
QueryEngine — the single public entry point for the entire library.

    from aaizaql import QueryEngine

    engine = QueryEngine(llm="groq", database="sqlite", dsn="sqlite:///my.db")
    engine.ingest_schema()

    # Optional training (improves accuracy)
    engine.train(documentation="employees.status: 1=Active, 2=Resigned...")
    engine.train(question="Top 5 employees by sales", sql="SELECT ...")
    engine.define_enum("employees", "status", {1:"Active", 2:"Resigned"})

    result = engine.query("Show all active employees")
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

import pandas as pd
import structlog

from aaizaql.connectors import REGISTRY
from aaizaql.core.config import Settings
from aaizaql.core.config import settings as _default_settings
from aaizaql.core.exceptions import ConnectorNotFound, SQLGenerationError, UnsupportedQueryError
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
    """Everything returned from a single engine.query() call."""

    question: str
    sql: str
    data: pd.DataFrame
    summary: str = ""
    chart: Any = None
    execution_time_ms: int = 0
    session_id: str = ""
    was_corrected: bool = False
    correction_attempts: int = 0


class QueryEngine:
    """
    Main entry point for the AAIZAQL library.

    Parameters
    ----------
    llm      : str   Provider: "groq" | "claude" | "openai" | "ollama" | "deepseek" | "perplexity" | "gemini" | "mistral"
    database : str   Connector: "sqlite" | "postgresql" | "mysql" | "snowflake" | "duckdb" | "mssql" | "oracle" | "mongodb" | "bigquery"
    dsn      : str   Connection string.
    **kwargs         Any Settings field (e.g. groq_model="llama-3.1-8b-instant")
    """

    def __init__(
        self,
        llm: str = "groq",
        database: str = "sqlite",
        dsn: str = "",
        settings: Settings | None = None,
        **kwargs: Any,
    ) -> None:
        # Settings
        if settings is None:
            overrides = {"llm_provider": llm}
            overrides.update(kwargs)
            self._settings = _default_settings.model_copy(update=overrides)
        else:
            self._settings = settings

        # Database connector
        if database not in REGISTRY:
            raise ConnectorNotFound(database)
        self._connector = REGISTRY[database]()
        if dsn:
            self._connector.connect(dsn)
            logger.info("database.connected", connector=database, dsn_hint=dsn[:40])

        # LLM
        self._llm = build_llm_provider(llm, self._settings)

        # Vector store
        self._vector_store = VectorStoreAdapter(self._settings)

        # Semantic store — holds documentation, enums, Q→SQL pairs
        self._semantic = SemanticStore(self._vector_store)

        # Schema ingester
        self._ingester = SchemaIngester(self._vector_store)

        # Pipeline components
        self._context = ContextManager(limit=self._settings.session_history_limit)
        self._generator = SQLGenerator(
            self._llm, self._vector_store, self._settings, self._semantic
        )
        self._validator = SQLValidator(self._settings)
        self._corrector = SelfCorrector(self._llm, self._settings, validator=self._validator)
        self._renderer = ResultRenderer()
        self._summarizer = NLSummarizer(self._llm)

        logger.info("engine.ready", llm=llm, database=database)

    # ─────────────────────────────────────────────────────────────────────────
    # Schema
    # ─────────────────────────────────────────────────────────────────────────

    def ingest_schema(self) -> int:
        """
        Auto-read the connected database schema and store it in the vector store.
        Call once before the first query, and again after schema changes.
        """
        logger.info("schema.ingesting")
        count = self._ingester.ingest_from_database(self._connector)
        logger.info("schema.ingested", chunks=count)
        return count

    # ─────────────────────────────────────────────────────────────────────────
    # Training — three methods, one clean API
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        documentation: str | None = None,
        question: str | None = None,
        sql: str | None = None,
    ) -> None:
        """
        Train the engine with business knowledge.

        Three modes — use any combination:

        1. Documentation (free-text business context):
            engine.train(documentation='''
                employees.status: 1=Active, 2=On Leave, 3=Resigned, 4=Terminated
                Use strftime('%Y-%m', order_date) for SQLite month grouping.
                business_unit_id: 4=ACCL, 8=APFIL, 12=IBOS
            ''')

        2. Q→SQL pair (sample question with correct SQL):
            engine.train(
                question="Top 5 employees by total sales",
                sql="SELECT e.name, SUM(s.total_amount) FROM employees e ..."
            )

        3. Both at once:
            engine.train(documentation="...", question="...", sql="...")
        """
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
        """
        Register a numeric code → label mapping for a column.

        Unlike documentation, enum mappings are ALWAYS injected into every
        prompt — no RAG retrieval miss is possible.

        Use for any integer-coded column:
            engine.define_enum("employees", "status", {
                1: "Active",
                2: "On Leave",
                3: "Resigned",
                4: "Terminated",
            })
            engine.define_enum("employees", "job_grade", {
                1: "Junior", 2: "Mid", 3: "Senior", 4: "Manager", 5: "Director"
            })
            engine.define_enum("employees", "business_unit_id", {
                4: "ACCL", 8: "APFIL", 12: "IBOS"
            })
        """
        self._semantic.define_enum(table, column, mapping)

    def teach(self, question: str, sql: str) -> None:
        """
        Shortcut for engine.train(question=..., sql=...).
        Kept for backwards compatibility.
        """
        self._semantic.train_sql_pair(question, sql)
        logger.info("engine.taught", question=question[:60])

    def training_info(self) -> dict[str, object]:
        """Return a summary of all training data currently loaded."""
        enums_list = self._semantic.list_enums()
        enums_dict = {f"{e['table']}.{e['column']}": e["mapping"] for e in enums_list}
        return {
            "enums": enums_dict,
            "enum_count": self._semantic.enum_count(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        session_id: str | None = None,
    ) -> QueryResult:
        """
        Convert a natural language question to SQL and execute it.

        Parameters
        ----------
        question   : str        Natural language question.
        session_id : str | None Pass the same ID across turns for memory.

        Returns
        -------
        QueryResult  (.sql, .data, .chart, .summary)
        """
        sid = session_id or str(uuid.uuid4())
        t_start = time.monotonic()

        logger.info("query.start", session_id=sid, question=question[:80])

        # Security: scan for prompt injection BEFORE any LLM call
        self._validator.check_question(question)

        history = self._context.get_history(sid)
        sql = self._generator.generate(question, history)

        if sql.strip().upper() == "UNSUPPORTED":
            raise UnsupportedQueryError(question)
        if not sql:
            raise SQLGenerationError(question, "LLM returned an empty response.")

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

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def reset_session(self, session_id: str) -> None:
        """Clear conversation memory for a session."""
        self._context.clear(session_id)

    def close(self) -> None:
        self._connector.close()
        logger.info("engine.closed")

    def __enter__(self) -> QueryEngine:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()