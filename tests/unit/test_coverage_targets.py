"""
tests/unit/test_coverage_targets.py
────────────────────────────────────
T1.2 — Targeted unit tests for the top-10 modules identified as having
low or zero coverage in the unit suite.

Modules covered here (in priority order):
 1. aaizaql.core.exceptions          — full exception hierarchy
 2. aaizaql.core.rate_limiter        — token bucket, exhaustion, disabled mode
 3. aaizaql.nlp.utils                — parse_sql_response (fence stripping)
 4. aaizaql.nlp.prompts              — template constants present and formattable
 5. aaizaql.nlp.decomposer           — QueryDecomposer with stub LLM
 6. aaizaql.nlp.graph_retriever      — GraphRetriever with stub store
 7. aaizaql.connectors._limit        — inject_limit (sqlglot + fallback paths)
 8. aaizaql.connectors.base          — DatabaseConnector abstract + test_connection
 9. aaizaql.memory.context           — ContextManager multi-session, eviction
10. aaizaql.api.health               — run_health_check happy / error paths

No external services, no LLM calls, no disk I/O required.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ══════════════════════════════════════════════════════════════════════════════
# 1. aaizaql.core.exceptions
# ══════════════════════════════════════════════════════════════════════════════


class TestExceptions:
    def test_base_exception(self):
        from aaizaql.core.exceptions import AAIZAQLError

        exc = AAIZAQLError("boom")
        assert "boom" in str(exc)

    def test_security_exception_no_sql(self):
        from aaizaql.core.exceptions import SecurityException

        exc = SecurityException("DROP detected")
        assert "DROP detected" in str(exc)
        assert exc.sql == ""

    def test_security_exception_with_sql(self):
        from aaizaql.core.exceptions import SecurityException

        exc = SecurityException("DROP detected", sql="DROP TABLE users")
        assert "DROP TABLE users" in str(exc)
        assert exc.sql == "DROP TABLE users"

    def test_prompt_injection_is_security(self):
        from aaizaql.core.exceptions import PromptInjectionDetected, SecurityException

        exc = PromptInjectionDetected("injected")
        assert isinstance(exc, SecurityException)

    def test_sql_generation_error(self):
        from aaizaql.core.exceptions import SQLGenerationError

        exc = SQLGenerationError("who won?", detail="model returned gibberish")
        assert "who won?" in str(exc)
        assert exc.question == "who won?"

    def test_rate_limit_error(self):
        from aaizaql.core.exceptions import RateLimitError

        exc = RateLimitError("tenant-42", retry_after_seconds=30)
        assert "tenant-42" in str(exc)
        assert exc.retry_after_seconds == 30

    def test_unsupported_query_error(self):
        from aaizaql.core.exceptions import UnsupportedQueryError

        exc = UnsupportedQueryError("delete all users")
        assert "delete all users" in str(exc)
        assert exc.question == "delete all users"

    def test_max_retries_exceeded(self):
        from aaizaql.core.exceptions import MaxRetriesExceeded

        exc = MaxRetriesExceeded(sql="SELECT x", last_error="syntax error", attempts=3)
        assert "3" in str(exc)
        assert "syntax error" in str(exc)

    def test_database_error_full(self):
        from aaizaql.core.exceptions import DatabaseError

        exc = DatabaseError("query failed", sql="SELECT 1", connector="sqlite")
        msg = str(exc)
        assert "sqlite" in msg
        assert "SELECT 1" in msg

    def test_database_error_minimal(self):
        from aaizaql.core.exceptions import DatabaseError

        exc = DatabaseError("query failed")
        assert exc.sql == ""
        assert exc.connector == ""

    def test_connection_error(self):
        from aaizaql.core.exceptions import ConnectionError

        exc = ConnectionError("postgres", dsn_hint="localhost:5432", detail="refused")
        assert "postgres" in str(exc)
        assert "refused" in str(exc)

    def test_connection_error_minimal(self):
        from aaizaql.core.exceptions import ConnectionError

        exc = ConnectionError("sqlite")
        assert "sqlite" in str(exc)

    def test_connector_not_found(self):
        from aaizaql.core.exceptions import ConnectorNotFound

        exc = ConnectorNotFound("oracle", available=["sqlite", "duckdb"])
        assert "oracle" in str(exc)
        assert sorted(["sqlite", "duckdb"]) == sorted(exc.available)

    def test_llm_error(self):
        from aaizaql.core.exceptions import LLMError

        exc = LLMError("openai", detail="429 rate limit")
        assert "openai" in str(exc)

    def test_llm_timeout_error_is_llm_error(self):
        from aaizaql.core.exceptions import LLMError, LLMTimeoutError

        exc = LLMTimeoutError("groq", timeout=30)
        assert isinstance(exc, LLMError)
        assert exc.timeout == 30
        assert "30s" in str(exc)

    def test_llm_provider_not_found(self):
        from aaizaql.core.exceptions import LLMProviderNotFound

        exc = LLMProviderNotFound("gpt-99", available=["openai", "claude"])
        assert "gpt-99" in str(exc)

    def test_vector_store_error(self):
        from aaizaql.core.exceptions import VectorStoreError

        exc = VectorStoreError("chroma down")
        assert "chroma down" in str(exc)

    def test_schema_ingestion_error(self):
        from aaizaql.core.exceptions import SchemaIngestionError

        exc = SchemaIngestionError("parse failed")
        assert isinstance(exc, Exception)

    def test_federation_error_with_sources(self):
        from aaizaql.core.exceptions import FederationError

        exc = FederationError("timeout", sources=["pg", "mysql"])
        assert "pg" in str(exc)
        assert exc.sources == ["pg", "mysql"]

    def test_federation_error_no_sources(self):
        from aaizaql.core.exceptions import FederationError

        exc = FederationError("timeout")
        assert exc.sources == []

    def test_credential_error(self):
        from aaizaql.core.exceptions import CredentialError

        exc = CredentialError("bad key")
        assert isinstance(exc, Exception)


# ══════════════════════════════════════════════════════════════════════════════
# 2. aaizaql.core.rate_limiter
# ══════════════════════════════════════════════════════════════════════════════


class TestRateLimiter:
    def test_first_call_passes(self):
        from aaizaql.core.rate_limiter import RateLimiter

        rl = RateLimiter(qpm=60)
        rl.check("tenant-1")  # should not raise

    def test_multiple_calls_within_quota(self):
        from aaizaql.core.rate_limiter import RateLimiter

        rl = RateLimiter(qpm=10)
        for _ in range(10):
            rl.check("t1")  # 10 tokens available at start

    def test_exhausted_raises_rate_limit_error(self):
        from aaizaql.core.exceptions import RateLimitError
        from aaizaql.core.rate_limiter import RateLimiter

        rl = RateLimiter(qpm=2)
        rl.check("t1")
        rl.check("t1")
        with pytest.raises(RateLimitError) as exc_info:
            rl.check("t1")
        assert exc_info.value.tenant_id == "t1"
        assert exc_info.value.retry_after_seconds >= 1

    def test_disabled_when_qpm_zero(self):
        from aaizaql.core.rate_limiter import RateLimiter

        rl = RateLimiter(qpm=0)
        for _ in range(100):
            rl.check("t1")  # never raises

    def test_different_tenants_isolated(self):
        from aaizaql.core.exceptions import RateLimitError
        from aaizaql.core.rate_limiter import RateLimiter

        rl = RateLimiter(qpm=1)
        rl.check("tenant-A")
        rl.check("tenant-B")  # separate bucket — should not raise
        with pytest.raises(RateLimitError):
            rl.check("tenant-A")

    def test_tokens_refill_over_time(self):
        from aaizaql.core.rate_limiter import RateLimiter

        rl = RateLimiter(qpm=60)  # 1 token/s
        rl.check("t1")  # consume 1 token from full bucket (59 remain)
        # Drain almost all remaining tokens
        for _ in range(58):
            rl.check("t1")
        # Now 1 token left — wait slightly > 1 second to refill 1
        time.sleep(1.1)
        rl.check("t1")  # should not raise after refill

    def test_get_bucket_creates_on_first_access(self):
        from aaizaql.core.rate_limiter import RateLimiter

        rl = RateLimiter(qpm=5)
        assert "new-tenant" not in rl._buckets
        rl._get_bucket("new-tenant")
        assert "new-tenant" in rl._buckets


# ══════════════════════════════════════════════════════════════════════════════
# 3. aaizaql.nlp.utils — parse_sql_response
# ══════════════════════════════════════════════════════════════════════════════


class TestParseSqlResponse:
    def setup_method(self):
        from aaizaql.nlp.utils import parse_sql_response

        self.parse = parse_sql_response

    def test_plain_sql_unchanged(self):
        assert self.parse("SELECT 1") == "SELECT 1"

    def test_strips_sql_fence(self):
        raw = "```sql\nSELECT 1\n```"
        assert self.parse(raw) == "SELECT 1"

    def test_strips_generic_fence(self):
        raw = "```\nSELECT 1\n```"
        assert self.parse(raw) == "SELECT 1"

    def test_strips_single_backtick(self):
        raw = "`SELECT 1`"
        assert self.parse(raw) == "SELECT 1"

    def test_cot_extracts_after_marker(self):
        raw = "[REASONING]\nsome thoughts\n[SQL]\nSELECT 2"
        result = self.parse(raw, use_cot=True)
        assert "SELECT 2" in result
        assert "[REASONING]" not in result

    def test_cot_flag_false_ignores_marker(self):
        raw = "[SQL]\nSELECT 2"
        result = self.parse(raw, use_cot=False)
        assert "[SQL]" in result  # marker not stripped when cot=False

    def test_case_insensitive_fence(self):
        raw = "```SQL\nSELECT 1\n```"
        assert self.parse(raw) == "SELECT 1"

    def test_multiline_sql_preserved(self):
        sql = "SELECT a,\n       b\nFROM t\nWHERE x = 1"
        raw = f"```sql\n{sql}\n```"
        assert self.parse(raw) == sql


# ══════════════════════════════════════════════════════════════════════════════
# 4. aaizaql.nlp.prompts — template constants
# ══════════════════════════════════════════════════════════════════════════════


class TestPrompts:
    def test_system_prompt_exists(self):
        from aaizaql.nlp.prompts import SYSTEM_PROMPT

        assert "SELECT" in SYSTEM_PROMPT
        assert "UNSUPPORTED" in SYSTEM_PROMPT

    def test_context_template_formattable(self):
        from aaizaql.nlp.prompts import CONTEXT_TEMPLATE

        rendered = CONTEXT_TEMPLATE.format(
            dialect="sqlite",
            schema_chunks="CREATE TABLE t (id INT)",
            enum_block="",
            doc_block="",
            example_pairs="",
            history="",
            question="How many rows?",
        )
        assert "sqlite" in rendered
        assert "How many rows?" in rendered

    def test_self_correction_template_formattable(self):
        from aaizaql.nlp.prompts import SELF_CORRECTION_TEMPLATE

        rendered = SELF_CORRECTION_TEMPLATE.format(
            dialect="postgres",
            sql="SELECT x FORM t",
            error="syntax error near FORM",
            schema_chunks="CREATE TABLE t (x INT)",
            enum_block="",
            doc_block="",
        )
        assert "syntax error near FORM" in rendered

    def test_summary_template_formattable(self):
        from aaizaql.nlp.prompts import SUMMARY_TEMPLATE

        rendered = SUMMARY_TEMPLATE.format(
            question="top 5 customers?",
            row_count=5,
            data_sample="id name\n1  Alice",
        )
        assert "top 5 customers?" in rendered
        assert "5" in rendered

    def test_cot_prefix_and_suffix_present(self):
        from aaizaql.nlp.prompts import COT_PROMPT_PREFIX, COT_PROMPT_SUFFIX

        assert "[REASONING]" in COT_PROMPT_PREFIX
        assert "[SQL]" in COT_PROMPT_SUFFIX

    def test_enum_block_template_formattable(self):
        from aaizaql.nlp.prompts import ENUM_BLOCK_TEMPLATE

        rendered = ENUM_BLOCK_TEMPLATE.format(enums="status: 1=active, 2=inactive")
        assert "active" in rendered

    def test_doc_block_template_formattable(self):
        from aaizaql.nlp.prompts import DOC_BLOCK_TEMPLATE

        rendered = DOC_BLOCK_TEMPLATE.format(docs="Orders are placed by customers.")
        assert "Orders" in rendered


# ══════════════════════════════════════════════════════════════════════════════
# 5. aaizaql.nlp.decomposer — QueryDecomposer
# ══════════════════════════════════════════════════════════════════════════════


def _make_stub_vector_store(chunks: list[str]):
    """Return a minimal vector-store stub."""
    Hit = SimpleNamespace
    vs = MagicMock()
    vs.search.return_value = [Hit(text=c) for c in chunks]
    return vs


class TestQueryDecomposer:
    def _make(self, llm_response: str):
        from aaizaql.nlp.decomposer import QueryDecomposer

        llm = MagicMock()
        llm.complete.return_value = llm_response
        vs = _make_stub_vector_store(["CREATE TABLE orders (id INT)"])
        gs = MagicMock()
        return QueryDecomposer(llm=llm, graph_store=gs, vector_store=vs)

    def test_returns_sub_questions(self):
        d = self._make("Q: Total orders\nQ: Customer count")
        result = d.decompose("complex question", tenant_id="t1")
        assert result == ["Total orders", "Customer count"]

    def test_fallback_on_no_q_lines(self):
        d = self._make("I don't know how to break this down.")
        result = d.decompose("question", tenant_id="t1")
        assert result == ["question"]

    def test_fallback_on_llm_exception(self):
        from aaizaql.nlp.decomposer import QueryDecomposer

        llm = MagicMock()
        llm.complete.side_effect = RuntimeError("LLM down")
        vs = _make_stub_vector_store([])
        gs = MagicMock()
        d = QueryDecomposer(llm=llm, graph_store=gs, vector_store=vs)
        result = d.decompose("q", tenant_id="t1")
        assert result == ["q"]

    def test_empty_vector_store_returns_no_schema(self):
        from aaizaql.nlp.decomposer import QueryDecomposer

        llm = MagicMock()
        llm.complete.return_value = "Q: sub"
        vs = _make_stub_vector_store([])
        gs = MagicMock()
        d = QueryDecomposer(llm=llm, graph_store=gs, vector_store=vs)
        result = d.decompose("q", tenant_id="t1")
        assert result == ["sub"]
        # Confirm the prompt received "(no schema)" placeholder
        prompt_arg = llm.complete.call_args[0][0]
        assert "(no schema)" in prompt_arg


# ══════════════════════════════════════════════════════════════════════════════
# 6. aaizaql.nlp.graph_retriever — GraphRetriever
# ══════════════════════════════════════════════════════════════════════════════


class TestGraphRetriever:
    def _make(self, ddl_hits: list[str], fk_neighbors: list[str] | None = None):
        from aaizaql.nlp.graph_retriever import GraphRetriever

        Hit = SimpleNamespace
        vs = MagicMock()
        vs.search.return_value = [Hit(text=h) for h in ddl_hits]
        gs = MagicMock()
        gs.get_fk_neighbors.return_value = fk_neighbors or []
        return GraphRetriever(vector_store=vs, graph_store=gs, top_k=3)

    def test_empty_vector_store(self):
        gr = self._make([])
        result = gr.get_context("how many?", tenant_id="t1")
        assert result == "(no schema ingested yet)"

    def test_returns_seed_chunks(self):
        ddl = "CREATE TABLE orders (id INT)"
        gr = self._make([ddl])
        result = gr.get_context("orders", tenant_id="t1")
        assert ddl in result

    def test_fk_expansion_fetches_extra(self):
        from aaizaql.nlp.graph_retriever import GraphRetriever

        Hit = SimpleNamespace
        vs = MagicMock()
        seed_ddl = "CREATE TABLE orders (id INT)"

        def search_side_effect(query, filter_type, top_k):
            if query == "orders":
                return [Hit(text=seed_ddl)]
            return [Hit(text=f"CREATE TABLE {query} (id INT)")]

        vs.search.side_effect = search_side_effect
        gs = MagicMock()
        gs.get_fk_neighbors.return_value = ["customers"]
        gr = GraphRetriever(vector_store=vs, graph_store=gs, top_k=3)
        result = gr.get_context("orders", tenant_id="t1")
        assert "orders" in result
        assert "customers" in result

    def test_extract_table_name_standard(self):
        from aaizaql.nlp.graph_retriever import GraphRetriever

        name = GraphRetriever._extract_table_name("CREATE TABLE orders (id INT PRIMARY KEY)")
        assert name == "orders"

    def test_extract_table_name_if_not_exists(self):
        from aaizaql.nlp.graph_retriever import GraphRetriever

        name = GraphRetriever._extract_table_name(
            "CREATE TABLE IF NOT EXISTS `line_items` (id INT)"
        )
        assert name == "line_items"

    def test_extract_table_name_no_match(self):
        from aaizaql.nlp.graph_retriever import GraphRetriever

        name = GraphRetriever._extract_table_name("SELECT 1")
        assert name == ""


# ══════════════════════════════════════════════════════════════════════════════
# 7. aaizaql.connectors._limit — inject_limit
# ══════════════════════════════════════════════════════════════════════════════


class TestInjectLimit:
    def setup_method(self):
        from aaizaql.connectors._limit import inject_limit

        self.inject = inject_limit

    def test_adds_limit_to_select(self):
        sql, truncated = self.inject("SELECT * FROM t", max_rows=100)
        assert "100" in sql
        assert truncated is True

    def test_no_limit_on_already_limited(self):
        sql, truncated = self.inject("SELECT * FROM t LIMIT 10", max_rows=1000)
        assert truncated is False

    def test_dml_not_limited_insert(self):
        raw = "INSERT INTO t VALUES (1)"
        sql, truncated = self.inject(raw, max_rows=100)
        assert truncated is False
        assert sql == raw

    def test_dml_not_limited_update(self):
        raw = "UPDATE t SET x=1"
        sql, truncated = self.inject(raw, max_rows=100)
        assert truncated is False

    def test_dml_not_limited_delete(self):
        raw = "DELETE FROM t"
        sql, truncated = self.inject(raw, max_rows=100)
        assert truncated is False

    def test_ddl_not_limited(self):
        raw = "CREATE TABLE t (id INT)"
        sql, truncated = self.inject(raw, max_rows=100)
        assert truncated is False

    def test_fallback_path_used_when_sqlglot_unavailable(self):
        """Force the except branch by making sqlglot raise on import."""
        with patch.dict("sys.modules", {"sqlglot": None}):
            from importlib import reload

            import aaizaql.connectors._limit as lim_mod

            reload(lim_mod)
            sql, truncated = lim_mod.inject_limit("SELECT * FROM t", max_rows=50)
            assert "50" in sql
            assert truncated is True

    def test_dialect_passed_through(self):
        sql, _ = self.inject("SELECT * FROM t", max_rows=5, dialect="mysql")
        assert "5" in sql


# ══════════════════════════════════════════════════════════════════════════════
# 8. aaizaql.connectors.base — DatabaseConnector
# ══════════════════════════════════════════════════════════════════════════════


class TestDatabaseConnectorBase:
    def _make_connector(self, execute_result=None, execute_raises=None):
        """Build a minimal concrete subclass for testing the base."""
        from aaizaql.connectors.base import DatabaseConnector

        class _Stub(DatabaseConnector):
            name = "stub"

            def connect(self, dsn: str) -> None:
                pass

            def execute(self, sql: str) -> pd.DataFrame:
                if execute_raises:
                    raise execute_raises
                return execute_result if execute_result is not None else pd.DataFrame()

            def get_schema(self) -> str:
                return "CREATE TABLE t (id INT)"

        return _Stub()

    def test_close_is_noop(self):
        c = self._make_connector()
        c.close()  # should not raise

    def test_test_connection_true_on_success(self):
        c = self._make_connector(execute_result=pd.DataFrame({"x": [1]}))
        assert c.test_connection() is True

    def test_test_connection_false_on_error(self):
        c = self._make_connector(execute_raises=RuntimeError("refused"))
        assert c.test_connection() is False

    def test_requires_sql_validation_default_true(self):
        c = self._make_connector()
        assert c.requires_sql_validation is True

    def test_get_schema_returns_ddl(self):
        c = self._make_connector()
        assert "CREATE TABLE" in c.get_schema()


# ══════════════════════════════════════════════════════════════════════════════
# 9. aaizaql.memory.context — ContextManager
# ══════════════════════════════════════════════════════════════════════════════


class TestContextManager:
    def setup_method(self):
        from aaizaql.memory.context import ContextManager

        self.cm = ContextManager(limit=3)

    def test_empty_session_returns_empty_list(self):
        assert self.cm.get_history("new-session") == []

    def test_add_and_retrieve_turn(self):
        self.cm.add_turn("s1", question="how many?", sql="SELECT COUNT(*) FROM t", row_count=42)
        history = self.cm.get_history("s1")
        assert len(history) == 1
        assert history[0]["question"] == "how many?"
        assert history[0]["sql"] == "SELECT COUNT(*) FROM t"
        assert history[0]["row_count"] == 42

    def test_limit_evicts_oldest_turn(self):
        for i in range(4):
            self.cm.add_turn("s1", question=f"q{i}", sql=f"SELECT {i}", row_count=i)
        history = self.cm.get_history("s1")
        assert len(history) == 3  # limit=3, so oldest (q0) evicted
        assert history[0]["question"] == "q1"

    def test_clear_removes_session(self):
        self.cm.add_turn("s1", question="q", sql="SELECT 1", row_count=1)
        self.cm.clear("s1")
        assert self.cm.get_history("s1") == []

    def test_clear_nonexistent_session_safe(self):
        self.cm.clear("does-not-exist")  # should not raise

    def test_session_count(self):
        assert self.cm.session_count() == 0
        self.cm.add_turn("a", "q", "SELECT 1", 0)
        self.cm.add_turn("b", "q", "SELECT 1", 0)
        assert self.cm.session_count() == 2

    def test_sessions_are_independent(self):
        self.cm.add_turn("alice", question="alice's q", sql="SELECT a", row_count=1)
        self.cm.add_turn("bob", question="bob's q", sql="SELECT b", row_count=2)
        assert self.cm.get_history("alice")[0]["question"] == "alice's q"
        assert self.cm.get_history("bob")[0]["question"] == "bob's q"

    def test_history_oldest_first(self):
        for i in range(3):
            self.cm.add_turn("s", f"q{i}", f"SELECT {i}", i)
        history = self.cm.get_history("s")
        questions = [t["question"] for t in history]
        assert questions == ["q0", "q1", "q2"]

    def test_row_count_defaults_to_zero(self):
        self.cm.add_turn("s", "q", "SELECT 1")
        assert self.cm.get_history("s")[0]["row_count"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# 10. aaizaql.api.health — run_health_check
# ══════════════════════════════════════════════════════════════════════════════


class TestRunHealthCheck:
    def _make_engine(self, db_ok=True, vs_ok=True, llm_ok=True):
        engine = MagicMock()

        if db_ok:
            engine._connector.test_connection.return_value = True
        else:
            engine._connector.test_connection.side_effect = RuntimeError("db down")

        if vs_ok:
            engine._vector_store.count.return_value = 42
        else:
            engine._vector_store.count.side_effect = RuntimeError("vs down")

        if llm_ok:
            engine._llm.complete.return_value = "pong"
        else:
            engine._llm.complete.side_effect = RuntimeError("llm down")

        return engine

    def test_all_ok(self):
        from aaizaql.api.health import run_health_check

        result = run_health_check(self._make_engine())
        assert result["db"] == "ok"
        assert result["vector_store"] == "ok"
        assert result["llm"] == "ok"
        assert "latency_ms" in result

    def test_db_error_reported(self):
        from aaizaql.api.health import run_health_check

        result = run_health_check(self._make_engine(db_ok=False))
        assert result["db"].startswith("error:")
        assert result["vector_store"] == "ok"

    def test_vector_store_error_reported(self):
        from aaizaql.api.health import run_health_check

        result = run_health_check(self._make_engine(vs_ok=False))
        assert result["vector_store"].startswith("error:")

    def test_llm_error_reported(self):
        from aaizaql.api.health import run_health_check

        result = run_health_check(self._make_engine(llm_ok=False))
        assert result["llm"].startswith("error:")

    def test_all_errors_still_returns_dict(self):
        from aaizaql.api.health import run_health_check

        result = run_health_check(self._make_engine(db_ok=False, vs_ok=False, llm_ok=False))
        assert all(k in result for k in ("db", "vector_store", "llm", "latency_ms"))

    def test_latency_is_non_negative_int(self):
        from aaizaql.api.health import run_health_check

        result = run_health_check(self._make_engine())
        assert isinstance(result["latency_ms"], int)
        assert result["latency_ms"] >= 0
