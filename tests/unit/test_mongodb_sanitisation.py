"""
tests/unit/test_mongodb_sanitisation.py
────────────────────────────────────────
Unit tests for the MongoDB query sanitiser in MongoDBConnector.

All tests exercise _sanitise_doc (and the backward-compat _sanitise_filter
shim) without a live MongoDB instance.  The connector is imported directly;
execute() is *not* called so pymongo is never needed.

Covered surface:
  - Blocked operators: $where, $function, $accumulator, $expr
  - Safe comparison operators: $gt, $lt, $gte, $lte, $ne, $in, $nin, $regex
  - Nesting: operators inside $or / $and arrays must still be caught
  - Projection: _sanitise_doc called on projection document
  - Collection name validation: system.* and names containing $ are rejected
"""

from __future__ import annotations

import pytest

from aaizaql.connectors.mongodb import MongoDBConnector
from aaizaql.core.exceptions import DatabaseError

# ── Helpers ───────────────────────────────────────────────────────────────────

RAW = '{"collection": "test", "filter": {}}'


def sanitise(doc: object, context: str = "filter") -> None:
    """Thin wrapper so tests don't repeat MongoDBConnector everywhere."""
    MongoDBConnector._sanitise_doc(doc, RAW, context=context)


# ── Blocked operators ─────────────────────────────────────────────────────────


class TestBlockedOperators:
    def test_where_at_top_level(self) -> None:
        with pytest.raises(DatabaseError, match=r"\$where"):
            sanitise({"$where": "this.salary > 100000"})

    def test_function_at_top_level(self) -> None:
        with pytest.raises(DatabaseError, match=r"\$function"):
            sanitise({"$function": {"body": "function() {}", "args": [], "lang": "js"}})

    def test_accumulator_at_top_level(self) -> None:
        with pytest.raises(DatabaseError, match=r"\$accumulator"):
            sanitise({"$accumulator": {}})

    def test_expr_at_top_level(self) -> None:
        """`$expr` can embed `$function` and must be blocked."""
        with pytest.raises(DatabaseError, match=r"\$expr"):
            sanitise({"$expr": {"$gt": ["$salary", 50000]}})

    def test_where_nested_under_and(self) -> None:
        """Dangerous operators inside $and arrays must be caught."""
        with pytest.raises(DatabaseError, match=r"\$where"):
            sanitise(
                {
                    "$and": [
                        {"department": "Engineering"},
                        {"$where": "this.salary > 100000"},
                    ]
                }
            )

    def test_function_nested_under_or(self) -> None:
        with pytest.raises(DatabaseError, match=r"\$function"):
            sanitise(
                {
                    "$or": [
                        {"status": 1},
                        {"$function": {"body": "function() { return true; }"}},
                    ]
                }
            )

    def test_expr_deeply_nested(self) -> None:
        """$expr buried two levels deep must still be caught."""
        with pytest.raises(DatabaseError, match=r"\$expr"):
            sanitise({"$and": [{"$or": [{"$expr": {"$gt": ["$x", 0]}}]}]})

    def test_error_message_names_context(self) -> None:
        """DatabaseError message should mention the context (filter/projection)."""
        with pytest.raises(DatabaseError) as exc_info:
            sanitise({"$where": "1"}, context="projection")
        assert "projection" in str(exc_info.value).lower()


# ── Allowed operators ─────────────────────────────────────────────────────────


class TestAllowedOperators:
    """Legitimate MongoDB query operators must pass through without error."""

    def test_gt_operator_allowed(self) -> None:
        sanitise({"salary": {"$gt": 50000}})

    def test_lt_operator_allowed(self) -> None:
        sanitise({"salary": {"$lt": 100000}})

    def test_gte_and_lte_allowed(self) -> None:
        sanitise({"salary": {"$gte": 30000, "$lte": 90000}})

    def test_ne_operator_allowed(self) -> None:
        sanitise({"status": {"$ne": 3}})

    def test_in_operator_allowed(self) -> None:
        sanitise({"status": {"$in": [1, 2]}})

    def test_nin_operator_allowed(self) -> None:
        sanitise({"status": {"$nin": [3, 4]}})

    def test_regex_operator_allowed(self) -> None:
        sanitise({"name": {"$regex": "^John", "$options": "i"}})

    def test_exists_operator_allowed(self) -> None:
        sanitise({"email": {"$exists": True}})

    def test_and_with_safe_predicates_allowed(self) -> None:
        sanitise(
            {
                "$and": [
                    {"status": {"$in": [1, 2]}},
                    {"salary": {"$gt": 50000}},
                ]
            }
        )

    def test_or_with_safe_predicates_allowed(self) -> None:
        sanitise(
            {
                "$or": [
                    {"department": "Engineering"},
                    {"department": "Product"},
                ]
            }
        )

    def test_empty_filter_allowed(self) -> None:
        sanitise({})

    def test_plain_equality_filter_allowed(self) -> None:
        sanitise({"status": 1, "department": "HR"})


# ── Projection sanitisation ───────────────────────────────────────────────────


class TestProjectionSanitisation:
    def test_safe_projection_allowed(self) -> None:
        sanitise({"name": 1, "salary": 1, "_id": 0}, context="projection")

    def test_dangerous_operator_in_projection_blocked(self) -> None:
        with pytest.raises(DatabaseError):
            sanitise({"$where": "1"}, context="projection")

    def test_expr_in_projection_blocked(self) -> None:
        with pytest.raises(DatabaseError, match=r"\$expr"):
            sanitise({"computed": {"$expr": {"$multiply": ["$x", 2]}}}, context="projection")


# ── Backward-compat shim ──────────────────────────────────────────────────────


class TestSanitiseFilterShim:
    """_sanitise_filter must still work for callers that use it directly."""

    def test_shim_blocks_where(self) -> None:
        with pytest.raises(DatabaseError):
            MongoDBConnector._sanitise_filter({"$where": "1"}, RAW)

    def test_shim_blocks_function(self) -> None:
        with pytest.raises(DatabaseError):
            MongoDBConnector._sanitise_filter({"$function": {}}, RAW)

    def test_shim_blocks_expr(self) -> None:
        with pytest.raises(DatabaseError):
            MongoDBConnector._sanitise_filter({"$expr": {}}, RAW)

    def test_shim_allows_gt(self) -> None:
        MongoDBConnector._sanitise_filter({"salary": {"$gt": 50000}}, RAW)


# ── Collection name validation ────────────────────────────────────────────────


class TestCollectionNameValidation:
    """
    MongoDBConnector.execute() must reject collection names that reference
    internal MongoDB namespaces.  We test this without a live connection by
    calling execute() with a disconnected connector and confirming the error
    is raised *before* the connection check would matter.

    Actually: the connection guard fires first ("Not connected"), so we test
    the name check via the execute() path with a patched _db.
    """

    def _make_connected_connector(self, mock_db) -> MongoDBConnector:
        c = MongoDBConnector()
        c._db = mock_db
        return c

    def test_system_collection_blocked(self) -> None:
        import json
        from unittest.mock import MagicMock

        connector = self._make_connected_connector(MagicMock())
        query = json.dumps({"collection": "system.users", "filter": {}})

        with pytest.raises(DatabaseError, match="not allowed"):
            connector.execute(query)

    def test_dollar_in_collection_name_blocked(self) -> None:
        import json
        from unittest.mock import MagicMock

        connector = self._make_connected_connector(MagicMock())
        query = json.dumps({"collection": "emp$loyees", "filter": {}})

        with pytest.raises(DatabaseError, match="not allowed"):
            connector.execute(query)

    def test_normal_collection_name_allowed(self) -> None:
        import json
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.find.return_value.__iter__ = MagicMock(return_value=iter([]))
        mock_collection.find.return_value.sort = MagicMock(
            return_value=mock_collection.find.return_value
        )
        mock_collection.find.return_value.limit = MagicMock(
            return_value=mock_collection.find.return_value
        )
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        connector = self._make_connected_connector(mock_db)
        query = json.dumps({"collection": "employees", "filter": {}})

        # Should not raise — empty result is fine.
        result = connector.execute(query)
        assert result is not None
