"""Unit tests for aaizaql.eval.spider_loader (B1 — dataset loader)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from aaizaql.eval.spider_loader import EvalItem, SpiderLoader


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_spider_dir(tmp_path: Path, items: list[dict]) -> Path:
    """Scaffold a minimal Spider-layout directory with real SQLite files."""
    for item in items:
        db_id = item["db_id"]
        db_dir = tmp_path / "database" / db_id
        db_dir.mkdir(parents=True, exist_ok=True)
        db_file = db_dir / f"{db_id}.sqlite"
        conn = sqlite3.connect(db_file)
        conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

    dev_json = tmp_path / "dev.json"
    dev_json.write_text(json.dumps(items), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def spider_dir(tmp_path: Path) -> Path:
    items = [
        {"db_id": "concert_singer", "question": "How many singers?", "query": "SELECT COUNT(*) FROM singer"},
        {"db_id": "concert_singer", "question": "List all singers", "query": "SELECT * FROM singer"},
        {"db_id": "pets_1", "question": "How many pets?", "query": "SELECT COUNT(*) FROM pets"},
    ]
    return _make_spider_dir(tmp_path, items)


# ── SpiderLoader construction ──────────────────────────────────────────────────


def test_loader_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        SpiderLoader(tmp_path / "nonexistent")


def test_loader_rejects_unknown_split(spider_dir: Path) -> None:
    loader = SpiderLoader(spider_dir)
    with pytest.raises(ValueError, match="Unknown split"):
        list(loader.items(split="bogus"))


def test_loader_rejects_missing_split_file(spider_dir: Path) -> None:
    loader = SpiderLoader(spider_dir)
    with pytest.raises(FileNotFoundError):
        list(loader.items(split="train"))


# ── items() happy path ────────────────────────────────────────────────────────


def test_items_returns_all_rows(spider_dir: Path) -> None:
    loader = SpiderLoader(spider_dir)
    items = list(loader.items())
    assert len(items) == 3


def test_items_respect_max_items(spider_dir: Path) -> None:
    loader = SpiderLoader(spider_dir)
    items = list(loader.items(max_items=1))
    assert len(items) == 1


def test_items_fields_populated(spider_dir: Path) -> None:
    loader = SpiderLoader(spider_dir)
    item = next(loader.items())
    assert item.db_id == "concert_singer"
    assert item.question == "How many singers?"
    assert item.gold_sql == "SELECT COUNT(*) FROM singer"
    assert item.evidence == ""
    assert item.db_path.exists()


def test_item_dsn_format(spider_dir: Path) -> None:
    loader = SpiderLoader(spider_dir)
    item = next(loader.items())
    assert item.dsn.startswith("sqlite:///")
    assert "concert_singer.sqlite" in item.dsn


def test_item_build_connector(spider_dir: Path) -> None:
    loader = SpiderLoader(spider_dir)
    item = next(loader.items())
    connector = item.build_connector()
    # Must be disconnected until connect() is called
    assert connector._conn is None
    # Connect and verify schema round-trip
    connector.connect(item.dsn)
    schema = connector.get_schema()
    assert "CREATE TABLE" in schema.upper()
    connector.close()


# ── db_ids() ──────────────────────────────────────────────────────────────────


def test_db_ids_unique_ordered(spider_dir: Path) -> None:
    loader = SpiderLoader(spider_dir)
    ids = loader.db_ids()
    assert ids == ["concert_singer", "pets_1"]


# ── Missing / incomplete rows ─────────────────────────────────────────────────


def test_skips_incomplete_rows(tmp_path: Path) -> None:
    items = [
        {"db_id": "ok_db", "question": "Q?", "query": "SELECT 1"},
        {"db_id": "ok_db"},  # missing question + query — should be skipped
    ]
    root = _make_spider_dir(tmp_path, [items[0]])
    (root / "dev.json").write_text(json.dumps(items), encoding="utf-8")
    loader = SpiderLoader(root, strict=False)
    result = list(loader.items())
    assert len(result) == 1


def test_strict_raises_on_missing_sqlite(tmp_path: Path) -> None:
    items = [{"db_id": "ghost_db", "question": "Q?", "query": "SELECT 1"}]
    root = tmp_path
    (root / "dev.json").write_text(json.dumps(items), encoding="utf-8")
    # No database/ghost_db/ghost_db.sqlite created
    loader = SpiderLoader(root, strict=True)
    with pytest.raises(FileNotFoundError, match="ghost_db"):
        list(loader.items())


def test_non_strict_skips_missing_sqlite(tmp_path: Path) -> None:
    items = [{"db_id": "ghost_db", "question": "Q?", "query": "SELECT 1"}]
    root = tmp_path
    (root / "dev.json").write_text(json.dumps(items), encoding="utf-8")
    loader = SpiderLoader(root, strict=False)
    result = list(loader.items())
    assert result == []


# ── EvalItem properties ───────────────────────────────────────────────────────


def test_eval_item_frozen(spider_dir: Path) -> None:
    loader = SpiderLoader(spider_dir)
    item = next(loader.items())
    with pytest.raises((AttributeError, TypeError)):
        item.db_id = "changed"  # type: ignore[misc]


def test_eval_item_extra_fields_preserved(tmp_path: Path) -> None:
    items = [{"db_id": "d", "question": "Q?", "query": "SELECT 1", "hardness": "easy"}]
    root = _make_spider_dir(tmp_path, items)
    loader = SpiderLoader(root)
    item = next(loader.items())
    assert item.extra.get("hardness") == "easy"
