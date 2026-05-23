"""
tests/unit/test_semantic_store.py
──────────────────────────────────
Unit tests for SemanticStore — documentation, enums, Q→SQL pairs.
No LLM, DB, or sentence-transformers needed (all mocked).
"""

from unittest.mock import MagicMock, patch

import pytest

from aaizaql.schema.semantic_store import EnumMapping, SemanticStore

# Patch _embed globally for all tests in this module
pytestmark = pytest.mark.usefixtures("mock_embed")


@pytest.fixture(autouse=True)
def mock_embed():
    """Patch _embed so sentence-transformers is never loaded."""
    with patch.object(SemanticStore, "_embed", return_value=[0.1] * 384):
        yield


@pytest.fixture
def mock_vs():
    vs = MagicMock()
    vs.search.return_value = []
    return vs


@pytest.fixture
def store(mock_vs):
    return SemanticStore(mock_vs)


# ── EnumMapping ───────────────────────────────────────────────────────────────


class TestEnumMapping:
    def test_to_prompt_text(self):
        e = EnumMapping("employees", "status", {1: "Active", 2: "Resigned"})
        text = e.to_prompt_text()
        assert "employees.status" in text
        assert "1=Active" in text
        assert "2=Resigned" in text

    def test_to_prompt_text_string_keys(self):
        e = EnumMapping("orders", "region", {"DH": "Dhaka", "CTG": "Chittagong"})
        text = e.to_prompt_text()
        assert "DH=Dhaka" in text


# ── define_enum ───────────────────────────────────────────────────────────────


class TestDefineEnum:
    def test_enum_registered(self, store):
        store.define_enum("employees", "status", {1: "Active", 2: "Resigned"})
        assert store.has_enums()
        assert store.enum_count() == 1

    def test_multiple_enums(self, store):
        store.define_enum("employees", "status", {1: "Active", 2: "Resigned"})
        store.define_enum("employees", "job_grade", {1: "Junior", 4: "Manager"})
        store.define_enum("orders", "order_status", {3: "Delivered"})
        assert store.enum_count() == 3

    def test_enum_overwrite_same_column(self, store):
        """Redefining same table.column replaces, not duplicates."""
        store.define_enum("employees", "status", {1: "Active"})
        store.define_enum("employees", "status", {1: "Active", 2: "Inactive"})
        assert store.enum_count() == 1
        enums = store.list_enums()
        assert len(enums[0]["mapping"]) == 2

    def test_get_enum_block_empty(self, store):
        assert store.get_enum_block() == ""

    def test_get_enum_block_content(self, store):
        store.define_enum("employees", "status", {1: "Active", 2: "On Leave"})
        block = store.get_enum_block()
        assert "employees.status" in block
        assert "1=Active" in block
        assert "2=On Leave" in block

    def test_get_enum_block_multiple(self, store):
        store.define_enum("employees", "status", {1: "Active"})
        store.define_enum("employees", "job_grade", {3: "Senior"})
        block = store.get_enum_block()
        assert "status" in block
        assert "job_grade" in block

    def test_list_enums(self, store):
        store.define_enum("employees", "status", {1: "Active", 2: "Resigned"})
        enums = store.list_enums()
        assert len(enums) == 1
        assert enums[0]["table"] == "employees"
        assert enums[0]["column"] == "status"
        assert enums[0]["mapping"][1] == "Active"

    def test_upserted_to_vector_store(self, store, mock_vs):
        store.define_enum("employees", "status", {1: "Active"})
        mock_vs.upsert.assert_called_once()
        call_kwargs = mock_vs.upsert.call_args.kwargs
        assert call_kwargs["metadata"]["type"] == "enum"
        assert call_kwargs["metadata"]["table"] == "employees"

    def test_idempotent(self, store):
        for _ in range(3):
            store.define_enum("t", "c", {1: "A"})
        assert store.enum_count() == 1


# ── train_documentation ───────────────────────────────────────────────────────


class TestDocumentation:
    def test_empty_documentation_ignored(self, store, mock_vs):
        store.train_documentation("   ")
        mock_vs.upsert.assert_not_called()

    def test_single_chunk_stored(self, store, mock_vs):
        store.train_documentation("employees.status: 1=Active, 2=Resigned")
        mock_vs.upsert.assert_called_once()

    def test_multi_paragraph_split(self, store, mock_vs):
        doc = "Paragraph one about status.\n\nParagraph two about job_grade."
        store.train_documentation(doc)
        assert mock_vs.upsert.call_count == 2

    def test_metadata_type_is_documentation(self, store, mock_vs):
        store.train_documentation("some context")
        call_kwargs = mock_vs.upsert.call_args.kwargs
        assert call_kwargs["metadata"]["type"] == "documentation"

    def test_search_documentation_called(self, store, mock_vs):
        mock_vs.search.return_value = [MagicMock(text="employees.status: 1=Active")]
        result = store.search_documentation("active employees")
        assert "Active" in result
        mock_vs.search.assert_called_once_with(
            query="active employees", filter_type="documentation", top_k=3
        )

    def test_search_documentation_empty(self, store, mock_vs):
        mock_vs.search.return_value = []
        result = store.search_documentation("nothing here")
        assert result == ""


# ── train_sql_pair ────────────────────────────────────────────────────────────


class TestSQLPair:
    def test_pair_stored_in_vector_store(self, store, mock_vs):
        store.train_sql_pair("Top 5 employees by sales", "SELECT e.name FROM employees e LIMIT 5")
        mock_vs.upsert.assert_called_once()
        call_kwargs = mock_vs.upsert.call_args.kwargs
        assert call_kwargs["metadata"]["type"] == "qa_pair"
        assert "Top 5 employees" in call_kwargs["text"]

    def test_pair_text_contains_sql(self, store, mock_vs):
        store.train_sql_pair("Count users", "SELECT COUNT(*) FROM users")
        text = mock_vs.upsert.call_args.kwargs["text"]
        assert "SELECT COUNT(*)" in text

    def test_pair_embeds_on_question(self, store, mock_vs):
        """Embedding should be on question, not full text (for better retrieval)."""
        store.train_sql_pair("How many orders?", "SELECT COUNT(*) FROM orders")
        # embedding kwarg should exist
        call_kwargs = mock_vs.upsert.call_args.kwargs
        assert "embedding" in call_kwargs


# ── has_enums / enum_count ────────────────────────────────────────────────────


class TestHelpers:
    def test_has_enums_false_initially(self, store):
        assert store.has_enums() is False

    def test_has_enums_true_after_define(self, store):
        store.define_enum("t", "c", {1: "A"})
        assert store.has_enums() is True

    def test_enum_count_zero_initially(self, store):
        assert store.enum_count() == 0

    def test_enum_count_increments(self, store):
        store.define_enum("t1", "c1", {1: "A"})
        store.define_enum("t2", "c2", {1: "B"})
        assert store.enum_count() == 2
