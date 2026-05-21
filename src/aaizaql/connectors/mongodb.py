"""
aaizaql.connectors.mongodb
──────────────────────────
MongoDB connector using pymongo.

Unlike SQL connectors, MongoDB operates on collections rather than SQL tables.
The execute() method accepts a JSON-encoded query descriptor:

  {"collection": "users", "filter": {"active": true}}
  {"collection": "orders", "filter": {}, "limit": 100}
  {"collection": "products", "filter": {"price": {"$gt": 50}}, "projection": {"name": 1}}

DSN format:
  mongodb://user:password@host:27017/dbname
  mongodb://user:password@host:27017/dbname?authSource=admin
  mongodb://localhost:27017/dbname

Install:
  pip install aaizaql[mongodb]   # or: pip install pymongo
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import structlog

from aaizaql.connectors.base import DatabaseConnector
from aaizaql.core.exceptions import ConnectionError, DatabaseError

logger = structlog.get_logger(__name__)


class MongoDBConnector(DatabaseConnector):
    """
    MongoDB adapter via pymongo.

    MongoDB does not use SQL; the execute() method accepts a JSON query
    descriptor string instead of a SQL statement.

    Usage::

        engine = QueryEngine(
            llm="groq",
            database="mongodb",
            dsn="mongodb://user:password@localhost:27017/mydb?authSource=admin",
        )

    Query format passed to execute()::

        '{"collection": "employees", "filter": {"department": "Engineering"}}'

    Optional keys: ``limit`` (int), ``projection`` (dict), ``sort`` (list of [field, direction]).
    """

    name = "mongodb"

    def __init__(self) -> None:
        self._client: Any = None
        self._db: Any = None
        self._db_name: str = ""
        self._dsn: str = ""

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self, dsn: str) -> None:
        """
        Connect to MongoDB.

        DSN examples::

          mongodb://user:password@localhost:27017/mydb?authSource=admin
          mongodb://localhost:27017/mydb
          mongodb+srv://user:password@cluster.mongodb.net/mydb
        """
        try:
            import pymongo
        except ImportError as exc:
            raise ConnectionError(
                "mongodb",
                dsn[:40],
                "pymongo is not installed. Run: pip install pymongo",
            ) from exc

        self._dsn = dsn
        try:
            self._client = pymongo.MongoClient(dsn, serverSelectionTimeoutMS=5000)
            # Force connection check
            self._client.admin.command("ping")
            # Derive database name from DSN path
            parsed = urlparse(dsn)
            db_name = parsed.path.lstrip("/").split("?")[0] or "test"
            self._db_name = db_name
            self._db = self._client[db_name]
            logger.info("mongodb.connected", db=db_name, dsn_hint=dsn[:40])
        except Exception as exc:
            raise ConnectionError("mongodb", dsn[:40], str(exc)) from exc

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute(self, query_json: str) -> pd.DataFrame:
        """
        Execute a MongoDB query described by a JSON string.

        Parameters
        ----------
        query_json:
            JSON string with keys:
            - ``collection`` (required): collection name
            - ``filter`` (optional, default ``{}``): MongoDB filter document
            - ``projection`` (optional): fields to include/exclude
            - ``limit`` (optional, default 0 = no limit): max documents
            - ``sort`` (optional): list of ``[field, direction]`` pairs
            - ``raise_if_empty`` (optional, bool): raise DatabaseError if no docs found

        Returns
        -------
        pd.DataFrame
            Query results. Empty DataFrame if no documents match.

        Raises
        ------
        aaizaql.core.exceptions.DatabaseError
            On execution failure or when ``raise_if_empty=True`` and result is empty.
        """
        if self._db is None:
            raise DatabaseError(
                "Not connected. Call connect() first.",
                sql=query_json,
                connector="mongodb",
            )
        try:
            spec = json.loads(query_json)
        except json.JSONDecodeError as exc:
            raise DatabaseError(
                f"Invalid JSON query: {exc}",
                sql=query_json,
                connector="mongodb",
            ) from exc

        collection_name: str = spec.get("collection", "")
        if not collection_name:
            raise DatabaseError(
                "Query descriptor must include a 'collection' key.",
                sql=query_json,
                connector="mongodb",
            )

        filter_doc: dict = spec.get("filter", {})
        projection: dict | None = spec.get("projection")
        limit: int = int(spec.get("limit", 0))
        sort: list | None = spec.get("sort")
        raise_if_empty: bool = bool(spec.get("raise_if_empty", False))

        try:
            collection = self._db[collection_name]
            cursor = collection.find(filter_doc, projection)
            if sort:
                cursor = cursor.sort(sort)
            if limit:
                cursor = cursor.limit(limit)
            docs = list(cursor)
        except Exception as exc:
            raise DatabaseError(str(exc), sql=query_json, connector="mongodb") from exc

        if raise_if_empty and not docs:
            raise DatabaseError(
                f"No documents found in collection '{collection_name}' matching filter.",
                sql=query_json,
                connector="mongodb",
            )

        if not docs:
            return pd.DataFrame()

        df = pd.DataFrame(docs)
        # Convert ObjectId to str so pandas doesn't choke
        if "_id" in df.columns:
            df["_id"] = df["_id"].astype(str)
        return df

    # ── Schema ────────────────────────────────────────────────────────────────

    def get_schema(self) -> str:
        """
        Return a human-readable schema summary for all collections.

        Samples up to 100 documents per collection to infer field names and types.
        Returns a string of pseudo-DDL suitable for LLM context injection.
        """
        if self._db is None:
            return ""
        try:
            parts: list[str] = []
            for coll_name in sorted(self._db.list_collection_names()):
                sample = list(self._db[coll_name].find().limit(100))
                if not sample:
                    parts.append(f"COLLECTION {coll_name} (empty)")
                    continue
                # Collect unique field → type mappings
                fields: dict[str, str] = {}
                for doc in sample:
                    for k, v in doc.items():
                        if k not in fields:
                            fields[k] = type(v).__name__
                field_lines = ", ".join(f"{k} {t}" for k, t in fields.items())
                parts.append(f"COLLECTION {coll_name} ({field_lines})")
            return "\n\n".join(parts)
        except Exception:
            return ""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def test_connection(self) -> bool:
        try:
            self._client.admin.command("ping")
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("mongodb.closed")
