"""
aaizaql.connectors.bigquery
───────────────────────────
Google BigQuery connector using google-cloud-bigquery.

Authentication uses Application Default Credentials (ADC) by default.
Run ``gcloud auth application-default login`` or set the
``GOOGLE_APPLICATION_CREDENTIALS`` environment variable to a service account
key JSON path for non-interactive environments.

DSN format::

    bigquery://project_id/dataset_id
    bigquery://project_id/dataset_id?credentials_path=/path/to/key.json

Install::

    pip install "aaizaql[bigquery]"   # or: pip install google-cloud-bigquery

Author: Ibrahim
Date: 2026-06-15
Version: 1.0.0
"""

import re
from typing import Any

import pandas as pd
import structlog

from aaizaql.connectors.base import DatabaseConnector
from aaizaql.core.exceptions import ConnectionError, DatabaseError

logger = structlog.get_logger(__name__)

_DSN_PATTERN = re.compile(r"bigquery://([^/]+)/([^?]+)(?:\?credentials_path=(.+))?")


class BigQueryConnector(DatabaseConnector):
    """Google BigQuery adapter via google-cloud-bigquery.

    Supports both ADC (Application Default Credentials) and explicit service
    account key files via the ``credentials_path`` DSN query parameter.

    Args (set at construction, no direct params):
        Call :meth:`connect` with a DSN string after instantiation.

    Example::

        engine = QueryEngine(
            llm="gemini",
            database="bigquery",
            dsn="bigquery://my-project/my_dataset",
        )
    """

    name = "bigquery"

    def __init__(self) -> None:
        self._client: Any = None
        self._project: str = ""
        self._dataset: str = ""

    def connect(self, dsn: str) -> None:
        """Establish a connection to BigQuery.

        Args:
            dsn: BigQuery connection string. Optionally includes
                ``?credentials_path=/path/to/key.json`` for service account
                authentication.

        Raises:
            ConnectionError: If google-cloud-bigquery is not installed, or
                ADC/service account authentication fails.
        """
        try:
            from google.cloud import bigquery
        except ImportError as exc:
            raise ConnectionError(
                "bigquery",
                dsn[:40],
                "google-cloud-bigquery is not installed. " "Run: pip install google-cloud-bigquery",
            ) from exc

        project, dataset, credentials_path = self._parse_dsn(dsn)
        self._project = project
        self._dataset = dataset

        try:
            if credentials_path:
                from google.oauth2 import service_account

                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/bigquery"],
                )
                self._client = bigquery.Client(project=project, credentials=credentials)
            else:
                self._client = bigquery.Client(project=project)

            logger.info("bigquery.connected", project=project, dataset=dataset)
        except Exception as exc:
            raise ConnectionError("bigquery", dsn[:40], str(exc)) from exc

    def execute(self, sql: str) -> pd.DataFrame:
        """Execute a SQL statement and return results as a DataFrame.

        Args:
            sql: A validated SQL statement to execute (BigQuery SQL dialect).

        Returns:
            Query results as a DataFrame. Returns an empty DataFrame for
            statements that produce no rows.

        Raises:
            DatabaseError: On any BigQuery execution error.
        """
        if self._client is None:
            raise DatabaseError(
                "Not connected. Call connect() first.", sql=sql, connector="bigquery"
            )
        try:
            query_job = self._client.query(sql)
            return query_job.to_dataframe()
        except Exception as exc:
            raise DatabaseError(str(exc), sql=sql, connector="bigquery") from exc

    def get_schema(self) -> str:
        """Return CREATE TABLE DDL for all tables in the connected dataset.

        Fetches table schemas from the BigQuery client API and reconstructs
        DDL from field metadata, since BigQuery does not expose a standard
        DDL export endpoint via the client library.

        Returns:
            DDL string with one CREATE TABLE block per table. Returns an
            empty string if not connected or on any error.
        """
        if self._client is None:
            return ""
        try:
            dataset_ref = self._client.dataset(self._dataset)
            tables = list(self._client.list_tables(dataset_ref))
            schema_parts = []

            for table_ref in tables:
                table = self._client.get_table(table_ref)
                fields = []
                for field in table.schema:
                    not_null = " NOT NULL" if field.mode == "REQUIRED" else ""
                    fields.append(f"  {field.name} {field.field_type}{not_null}")
                ddl = (
                    f"CREATE TABLE `{self._project}.{self._dataset}.{table.table_id}` (\n"
                    + ",\n".join(fields)
                    + "\n);"
                )
                schema_parts.append(ddl)

            return "\n\n".join(schema_parts)
        except Exception:
            return ""

    def close(self) -> None:
        """Close the BigQuery client and release resources."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("bigquery.closed")

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_dsn(dsn: str) -> tuple[str, str, str]:
        """Parse a BigQuery DSN URL into connection components.

        Args:
            dsn: BigQuery connection string in the form
                ``"bigquery://project/dataset[?credentials_path=...]"``.

        Returns:
            Tuple of ``(project, dataset, credentials_path)``.
            ``credentials_path`` is an empty string when not specified.

        Raises:
            ValueError: If the DSN does not match the expected format.
        """
        match = _DSN_PATTERN.match(dsn)
        if not match:
            raise ValueError(
                f"Cannot parse BigQuery DSN: {dsn!r}\n"
                "Expected format: bigquery://project_id/dataset_id"
            )
        project, dataset, credentials_path = match.groups()
        return project, dataset, credentials_path or ""
