"""
tests/test_schema.py
─────────────────────
Unit tests for SchemaIngester and SemanticStore.
Uses a mock vector store to avoid requiring ChromaDB in basic CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aaizaql.schema.ingestion import SchemaIngester
from aaizaql.schema.semantic_store import SemanticStore

# ── SchemaIngester ────────────────────────────────────────────────────────────


class TestSchemaIngester:
    @pytest.fixture(autouse=True)
    def patch_embedder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Prevent sentence-transformers from downloading models during tests."""
        from aaizaql.schema import ingestion as ing_mod

        monkeypatch.setattr(ing_mod._SentenceEmbedder, "_load", lambda self: None)
        monkeypatch.setattr(
            ing_mod._SentenceEmbedder,
            "embed",
            lambda self, text: [0.0] * 384,
        )

    @pytest.fixture()
    def mock_vs(self) -> MagicMock:
        vs = MagicMock()
        return vs

    @pytest.fixture()
    def ingester(self, mock_vs: MagicMock) -> SchemaIngester:
        return SchemaIngester(mock_vs)

    def test_ingest_ddl_two_tables(self, ingester: SchemaIngester, mock_vs: MagicMock) -> None:
        ddl = """
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
        """
        count = ingester.ingest_ddl(ddl)
        assert count == 2
        # 2 DDL table upserts + 1 schema_version sentinel upsert (T5.5)
        assert mock_vs.upsert.call_count == 3

    def test_ingest_ddl_extracts_table_name(
        self, ingester: SchemaIngester, mock_vs: MagicMock
    ) -> None:
        ddl = "CREATE TABLE orders (id INTEGER, amount REAL);"
        ingester.ingest_ddl(ddl)
        # call_args_list[0] is the DDL table upsert; [-1] would be the schema_version sentinel
        call_kwargs = mock_vs.upsert.call_args_list[0][1]
        assert call_kwargs["metadata"]["table"] == "orders"
        assert call_kwargs["metadata"]["type"] == "ddl"

    def test_ingest_empty_ddl_returns_zero(
        self, ingester: SchemaIngester, mock_vs: MagicMock
    ) -> None:
        count = ingester.ingest_ddl("   ")
        assert count == 0
        mock_vs.upsert.assert_not_called()

    def test_ingest_from_database(
        self,
        ingester: SchemaIngester,
        mock_vs: MagicMock,
        sqlite_connector: object,
    ) -> None:
        count = ingester.ingest_from_database(sqlite_connector)  # type: ignore[arg-type]
        assert count >= 2  # employees + departments

    @pytest.mark.skip(
        reason="T2.5: ingest_sql_pair() removed from SchemaIngester — use SemanticStore.train_sql_pair() instead"
    )
    def test_ingest_sql_pair(self, ingester: SchemaIngester, mock_vs: MagicMock) -> None:
        ingester.ingest_sql_pair(
            question="How many employees are there?",
            sql="SELECT COUNT(*) FROM employees",
        )
        mock_vs.upsert.assert_called_once()
        call_kwargs = mock_vs.upsert.call_args[1]
        assert call_kwargs["metadata"]["type"] == "qa_pair"
        assert "How many employees" in call_kwargs["text"]

    def test_fingerprint_is_deterministic(self, ingester: SchemaIngester) -> None:
        fp1 = ingester._fingerprint("hello world")
        fp2 = ingester._fingerprint("hello world")
        assert fp1 == fp2

    def test_extract_table_name_variants(self, ingester: SchemaIngester) -> None:
        assert ingester._extract_table_name("CREATE TABLE employees (id INT)") == "employees"
        assert (
            ingester._extract_table_name("CREATE TABLE IF NOT EXISTS orders (id INT)") == "orders"
        )
        assert ingester._extract_table_name('CREATE TABLE "my_table" (x TEXT)') == "my_table"


# ── SemanticStore ─────────────────────────────────────────────────────────────


class TestSemanticStore:
    @pytest.fixture(autouse=True)
    def patch_embedder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aaizaql.schema import ingestion as ing_mod

        monkeypatch.setattr(ing_mod._SentenceEmbedder, "_load", lambda self: None)
        monkeypatch.setattr(
            ing_mod._SentenceEmbedder,
            "embed",
            lambda self, text: [0.0] * 384,
        )

    @pytest.fixture()
    def mock_vs(self) -> MagicMock:
        vs = MagicMock()
        vs.search.return_value = []
        return vs

    @pytest.fixture()
    def store(self, mock_vs: MagicMock) -> SemanticStore:
        return SemanticStore(mock_vs)

    def test_define_enum_and_retrieve(self, store: SemanticStore) -> None:
        store.define_enum("employees", "status", {1: "Active", 2: "On Leave", 3: "Resigned"})
        assert store.has_enums()
        assert store.enum_count() == 1

    def test_get_enum_block_format(self, store: SemanticStore) -> None:
        store.define_enum("employees", "status", {1: "Active", 2: "Resigned"})
        block = store.get_enum_block()
        assert "employees.status" in block
        assert "1=Active" in block
        assert "2=Resigned" in block

    def test_multiple_enums(self, store: SemanticStore) -> None:
        store.define_enum("employees", "status", {1: "Active", 2: "Resigned"})
        store.define_enum("employees", "grade", {1: "Junior", 2: "Senior"})
        assert store.enum_count() == 2
        block = store.get_enum_block()
        assert "employees.status" in block
        assert "employees.grade" in block

    def test_no_enums_returns_empty_block(self, store: SemanticStore) -> None:
        assert not store.has_enums()
        assert store.get_enum_block() == ""

    def test_train_documentation_calls_upsert(
        self, store: SemanticStore, mock_vs: MagicMock
    ) -> None:
        store.train_documentation(
            "employees.status: 1=Active, 2=Resigned\n\nUse strftime for SQLite dates."
        )
        # Two paragraphs → two upsert calls
        assert mock_vs.upsert.call_count == 2

    def test_train_sql_pair_calls_upsert(self, store: SemanticStore, mock_vs: MagicMock) -> None:
        store.train_sql_pair(
            question="Total employees",
            sql="SELECT COUNT(*) FROM employees",
        )
        mock_vs.upsert.assert_called_once()

    def test_list_enums(self, store: SemanticStore) -> None:
        store.define_enum("t", "col", {0: "No", 1: "Yes"})
        enums = store.list_enums()
        assert len(enums) == 1
        assert enums[0]["table"] == "t"
        assert enums[0]["column"] == "col"
        assert enums[0]["mapping"][0] == "No"
