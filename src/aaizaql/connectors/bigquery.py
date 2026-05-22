"""
aaizaql.connectors.bigquery
────────────────────────────
Google BigQuery connector using google-cloud-bigquery.

DSN format:
  bigquery://project_id/dataset_id
  bigquery://project_id/dataset_id?credentials_path=/path/to/key.json

Install:
  pip install aaizaql[bigquery]   # or: pip install google-cloud-bigquery
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import structlog

from aaizaql.connectors.base import DatabaseConnector
from aaizaql.core.exceptions import ConnectionError, DatabaseError

logger = structlog.get_logger(__name__)


class BigQueryConnector(DatabaseConnector):
    """
    Google BigQuery adapter via google-cloud-bigquery.

    Authentication uses Application Default Credentials (ADC) by default.
    Run `gcloud auth application-default login` or set GOOGLE_APPLICATION_CREDENTIALS
    env var to a service account key JSON path.

    Usage:
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
        """
        DSN examples:
          bigquery://my-project/my_dataset
          bigquery://my-project/my_dataset?credentials_path=/path/key.json
        """
        try:
            from google.cloud import bigquery
        except ImportError as exc:
            raise ConnectionError(
                "bigquery",
                dsn[:40],
                "google-cloud-bigquery is not installed. Run: pip install google-cloud-bigquery",
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

    def _parse_dsn(self, dsn: str) -> tuple[str, str, str]:
        """Parse bigquery://project/dataset[?credentials_path=...] DSN."""
        import re

        pattern = r"bigquery://([^/]+)/([^?]+)(?:\?credentials_path=(.+))?"
        m = re.match(pattern, dsn)
        if not m:
            raise ValueError(
                f"Cannot parse BigQuery DSN: {dsn!r}\n"
                "Expected format: bigquery://project_id/dataset_id"
            )
        project, dataset, credentials_path = m.groups()
        return project, dataset, credentials_path or ""

    def execute(self, sql: str) -> pd.DataFrame:
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
        """Return DDL-style schema for all tables in the connected dataset."""
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
                    # T3.2 — was: both branches returned empty string (bug)
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
        if self._client:
            self._client.close()
            self._client = None
            logger.info("bigquery.closed")
