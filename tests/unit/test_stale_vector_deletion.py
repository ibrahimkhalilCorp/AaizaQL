"""
tests/unit/test_stale_vector_deletion.py
─────────────────────────────────────────
Unit tests for stale DDL vector cleanup in SchemaIngester.ingest_ddl().

Scenario coverage:
  - First ingest into empty store → no deletions
  - Re-ingest same schema → no deletions (idempotent)
  - Re-ingest with one table removed → that table's vector deleted
  - Re-ingest with multiple tables removed → all stale vectors deleted
  - Re-ingest with table added → only new table upserted, nothing deleted
  - Re-ingest with table renamed → old name deleted, new name upserted
  - schema_version sentinel never counted as a stale DDL vector
  - Empty DDL → no deletions, no upserts, returns 0
  - delete() called exactly once per stale ID (no double-delete)

No database, LLM, or sentence-transformers required — all mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aaizaql.schema.ingestion import SchemaIngester

# ── Fixtures ──────────────────────────────────────────────────────────────────

DDL_USERS = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);"
DDL_ORDERS = "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total REAL);"
DDL_PRODUCTS = "CREATE TABLE products (id INTEGER PRIMARY KEY, sku TEXT, price REAL);"

DDL_TWO_TABLES = f"{DDL_USERS}\n\n{DDL_ORDERS}"
DDL_THREE_TABLES = f"{DDL_USERS}\n\n{DDL_ORDERS}\n\n{DDL_PRODUCTS}"
DDL_USERS_ONLY = DDL_USERS
DDL_ORDERS_ONLY = DDL_ORDERS


def _make_ingester(existing_ddl_ids: set[str] | None = None) -> tuple[SchemaIngester, MagicMock]:
    """
    Build a SchemaIngester with a mocked VectorStoreAdapter.

    Parameters
    ----------
    existing_ddl_ids
        The set of DDL doc IDs already in the vector store before this ingest.
        Simulates what list_ids(filter_type="ddl") would return.
    """
    mock_vs = MagicMock()
    mock_vs.list_ids.return_value = existing_ddl_ids or set()

    # Patch the EmbeddingService so no model is loaded
    with patch("aaizaql.schema.ingestion.EmbeddingService") as mock_emb_cls:
        mock_emb_cls.get_instance.return_value.embed.return_value = [0.1] * 384
        ingester = SchemaIngester(mock_vs)

    # Keep embed fast for the whole test
    ingester._embedder = MagicMock()
    ingester._embedder.embed.return_value = [0.1] * 384

    return ingester, mock_vs


def _ids_for(ddl: str) -> set[str]:
    """Compute the doc IDs that ingest_ddl would generate for a given DDL string."""
    ingester, _ = _make_ingester()
    chunks = ingester._chunk_by_table(ddl)
    return {f"ddl_{ingester._fingerprint(c)}" for c in chunks}


# ── First ingest (empty store) ────────────────────────────────────────────────


class TestFirstIngest:
    def test_no_deletions_on_empty_store(self) -> None:
        """First ingest — nothing to delete."""
        ingester, mock_vs = _make_ingester(existing_ddl_ids=set())
        ingester.ingest_ddl(DDL_TWO_TABLES)
        mock_vs.delete.assert_not_called()

    def test_upserts_each_table_chunk(self) -> None:
        ingester, mock_vs = _make_ingester(existing_ddl_ids=set())
        count = ingester.ingest_ddl(DDL_TWO_TABLES)
        # 2 tables + 1 schema_version sentinel
        assert mock_vs.upsert.call_count == 3
        assert count == 2

    def test_returns_zero_for_empty_ddl(self) -> None:
        ingester, mock_vs = _make_ingester(existing_ddl_ids=set())
        result = ingester.ingest_ddl("   ")
        assert result == 0
        mock_vs.delete.assert_not_called()
        mock_vs.upsert.assert_not_called()


# ── Re-ingest identical schema ────────────────────────────────────────────────


class TestIdempotentReingest:
    def test_no_deletions_when_schema_unchanged(self) -> None:
        """Re-ingesting the same DDL must not delete anything."""
        existing = _ids_for(DDL_TWO_TABLES)
        ingester, mock_vs = _make_ingester(existing_ddl_ids=existing)
        ingester.ingest_ddl(DDL_TWO_TABLES)
        mock_vs.delete.assert_not_called()

    def test_upsert_still_called_for_all_chunks(self) -> None:
        """Even on identical re-ingest, upsert is called (ChromaDB deduplicates)."""
        existing = _ids_for(DDL_TWO_TABLES)
        ingester, mock_vs = _make_ingester(existing_ddl_ids=existing)
        ingester.ingest_ddl(DDL_TWO_TABLES)
        # 2 tables + schema_version sentinel
        assert mock_vs.upsert.call_count == 3


# ── Table removed ─────────────────────────────────────────────────────────────


class TestTableRemoved:
    def test_single_table_removed_triggers_delete(self) -> None:
        """If orders is removed from the schema, its vector must be deleted."""
        old_ids = _ids_for(DDL_TWO_TABLES)  # users + orders
        new_ids = _ids_for(DDL_USERS_ONLY)  # users only
        stale_ids = old_ids - new_ids  # just the orders ID

        ingester, mock_vs = _make_ingester(existing_ddl_ids=old_ids)
        ingester.ingest_ddl(DDL_USERS_ONLY)

        deleted = {c.args[0] for c in mock_vs.delete.call_args_list}
        assert deleted == stale_ids, f"Expected stale IDs {stale_ids} to be deleted, got {deleted}"

    def test_delete_count_equals_removed_table_count(self) -> None:
        """Removing 2 tables → exactly 2 delete() calls."""
        old_ids = _ids_for(DDL_THREE_TABLES)
        ingester, mock_vs = _make_ingester(existing_ddl_ids=old_ids)
        ingester.ingest_ddl(DDL_USERS_ONLY)

        assert mock_vs.delete.call_count == 2

    def test_surviving_table_not_deleted(self) -> None:
        """The table that remains must never be passed to delete()."""
        old_ids = _ids_for(DDL_TWO_TABLES)
        surviving_ids = _ids_for(DDL_USERS_ONLY)

        ingester, mock_vs = _make_ingester(existing_ddl_ids=old_ids)
        ingester.ingest_ddl(DDL_USERS_ONLY)

        deleted = {c.args[0] for c in mock_vs.delete.call_args_list}
        assert not (
            deleted & surviving_ids
        ), f"Surviving table IDs {surviving_ids} were incorrectly deleted"

    def test_all_tables_removed_deletes_all(self) -> None:
        """Switching to a completely different schema deletes all old vectors."""
        old_ids = _ids_for(DDL_TWO_TABLES)
        new_ddl = DDL_PRODUCTS  # completely different table
        ingester, mock_vs = _make_ingester(existing_ddl_ids=old_ids)
        ingester.ingest_ddl(new_ddl)

        deleted = {c.args[0] for c in mock_vs.delete.call_args_list}
        assert deleted == old_ids


# ── Table added ───────────────────────────────────────────────────────────────


class TestTableAdded:
    def test_no_deletions_when_table_added(self) -> None:
        """Adding a new table to the schema must not delete anything."""
        old_ids = _ids_for(DDL_TWO_TABLES)
        ingester, mock_vs = _make_ingester(existing_ddl_ids=old_ids)
        ingester.ingest_ddl(DDL_THREE_TABLES)  # adds products
        mock_vs.delete.assert_not_called()

    def test_new_table_is_upserted(self) -> None:
        """The new table must appear in the upsert calls."""
        old_ids = _ids_for(DDL_TWO_TABLES)
        ingester, mock_vs = _make_ingester(existing_ddl_ids=old_ids)
        ingester.ingest_ddl(DDL_THREE_TABLES)
        # 3 tables + schema_version sentinel
        assert mock_vs.upsert.call_count == 4


# ── Table renamed (= delete old + upsert new) ─────────────────────────────────


class TestTableRenamed:
    def test_renamed_table_old_id_deleted(self) -> None:
        """Renaming a table changes its fingerprint — old ID must be deleted."""
        ddl_old = "CREATE TABLE employees (id INTEGER, name TEXT);"
        ddl_new = "CREATE TABLE staff (id INTEGER, name TEXT);"  # renamed

        old_ids = _ids_for(ddl_old)
        ingester, mock_vs = _make_ingester(existing_ddl_ids=old_ids)
        ingester.ingest_ddl(ddl_new)

        deleted = {c.args[0] for c in mock_vs.delete.call_args_list}
        assert deleted == old_ids

    def test_renamed_table_new_id_upserted(self) -> None:
        """The new name must be upserted."""
        ddl_old = "CREATE TABLE employees (id INTEGER, name TEXT);"
        ddl_new = "CREATE TABLE staff (id INTEGER, name TEXT);"

        old_ids = _ids_for(ddl_old)
        ingester, mock_vs = _make_ingester(existing_ddl_ids=old_ids)
        ingester.ingest_ddl(ddl_new)

        upserted_ids = {
            c.kwargs["doc_id"]
            for c in mock_vs.upsert.call_args_list
            if c.kwargs.get("doc_id", "").startswith("ddl_")
        }
        new_ids = _ids_for(ddl_new)
        assert upserted_ids == new_ids


# ── schema_version sentinel ───────────────────────────────────────────────────


class TestSchemaVersionSentinel:
    def test_schema_version_always_upserted(self) -> None:
        """A schema_version sentinel must be upserted on every ingest."""
        ingester, mock_vs = _make_ingester(existing_ddl_ids=set())
        ingester.ingest_ddl(DDL_USERS_ONLY)

        upserted_ids = [c.kwargs["doc_id"] for c in mock_vs.upsert.call_args_list]
        assert "schema_version" in upserted_ids

    def test_schema_version_not_treated_as_stale_ddl(self) -> None:
        """schema_version ID in old_ids must not trigger a delete,
        because list_ids(filter_type='ddl') should never return it
        (it is stored as type='schema_version'). Simulate the correct
        filter by not including it in existing_ddl_ids."""
        # Correct behaviour: list_ids filters by type="ddl", so schema_version
        # is never in old_ids. Verify delete() is not called for it.
        old_ids = _ids_for(DDL_USERS_ONLY)  # only real DDL IDs
        ingester, mock_vs = _make_ingester(existing_ddl_ids=old_ids)
        ingester.ingest_ddl(DDL_USERS_ONLY)

        deleted = {c.args[0] for c in mock_vs.delete.call_args_list}
        assert "schema_version" not in deleted

    def test_schema_version_metadata_type(self) -> None:
        """The sentinel must be stored with type='schema_version', not 'ddl'."""
        ingester, mock_vs = _make_ingester(existing_ddl_ids=set())
        ingester.ingest_ddl(DDL_USERS_ONLY)

        sentinel_call = next(
            c for c in mock_vs.upsert.call_args_list if c.kwargs.get("doc_id") == "schema_version"
        )
        assert sentinel_call.kwargs["metadata"]["type"] == "schema_version"


# ── No double-delete ──────────────────────────────────────────────────────────


class TestNoDoubleDelete:
    def test_each_stale_id_deleted_exactly_once(self) -> None:
        """delete() must be called exactly once per stale ID, never twice."""
        old_ids = _ids_for(DDL_THREE_TABLES)
        ingester, mock_vs = _make_ingester(existing_ddl_ids=old_ids)
        ingester.ingest_ddl(DDL_USERS_ONLY)  # removes orders + products

        deleted_ids = [c.args[0] for c in mock_vs.delete.call_args_list]
        assert len(deleted_ids) == len(
            set(deleted_ids)
        ), f"Duplicate delete() calls detected: {deleted_ids}"
