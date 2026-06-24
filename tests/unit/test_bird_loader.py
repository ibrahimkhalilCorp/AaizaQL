"""Unit tests for aaizaql.eval.bird_loader (B4 — BIRD dataset loader)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aaizaql.eval.bird_loader import BirdLoader
from aaizaql.eval.spider_loader import EvalItem


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_bird_dir(
    tmp_path: Path,
    items: list[dict],
    split: str = "dev",
) -> Path:
    """Scaffold a minimal BIRD-layout directory with real SQLite files."""
    db_subdir = f"{split}_databases"
    for item in items:
        db_id = item["db_id"]
        db_dir = tmp_path / db_subdir / db_id
        db_dir.mkdir(parents=True, exist_ok=True)
        db_file = db_dir / f"{db_id}.sqlite"
        conn = sqlite3.connect(db_file)
        conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

    json_filename = f"{split}.json"
    (tmp_path / json_filename).write_text(json.dumps(items), encoding="utf-8")
    return tmp_path


_SAMPLE_ITEMS = [
    {
        "question_id": 0,
        "db_id": "debit_card_specializing",
        "question": "How many customers have a card?",
        "evidence": "A customer holds exactly one card per account.",
        "SQL": "SELECT COUNT(*) FROM customers",
        "difficulty": "simple",
    },
    {
        "question_id": 1,
        "db_id": "debit_card_specializing",
        "question": "List all card types.",
        "evidence": "",
        "SQL": "SELECT DISTINCT type FROM cards",
        "difficulty": "moderate",
    },
    {
        "question_id": 2,
        "db_id": "european_football_2",
        "question": "How many leagues are there?",
        "evidence": "Each league belongs to one country.",
        "SQL": "SELECT COUNT(*) FROM league",
        "difficulty": "simple",
    },
]


@pytest.fixture()
def bird_dir(tmp_path: Path) -> Path:
    return _make_bird_dir(tmp_path, _SAMPLE_ITEMS)


# ── BirdLoader construction ───────────────────────────────────────────────────


def test_loader_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        BirdLoader(tmp_path / "nonexistent")


def test_loader_rejects_unknown_split(bird_dir: Path) -> None:
    loader = BirdLoader(bird_dir)
    with pytest.raises(ValueError, match="Unknown split"):
        list(loader.items(split="bogus"))


def test_loader_rejects_missing_split_file(bird_dir: Path) -> None:
    loader = BirdLoader(bird_dir)
    with pytest.raises(FileNotFoundError):
        list(loader.items(split="train"))


# ── items() happy path ────────────────────────────────────────────────────────


def test_items_returns_all_rows(bird_dir: Path) -> None:
    loader = BirdLoader(bird_dir)
    items = list(loader.items())
    assert len(items) == 3


def test_items_respect_max_items(bird_dir: Path) -> None:
    loader = BirdLoader(bird_dir)
    items = list(loader.items(max_items=1))
    assert len(items) == 1


def test_items_fields_populated(bird_dir: Path) -> None:
    loader = BirdLoader(bird_dir)
    item = next(loader.items())
    assert item.db_id == "debit_card_specializing"
    assert item.question == "How many customers have a card?"
    assert item.gold_sql == "SELECT COUNT(*) FROM customers"
    assert item.evidence == "A customer holds exactly one card per account."
    assert item.db_path.exists()


def test_items_empty_evidence_preserved(bird_dir: Path) -> None:
    loader = BirdLoader(bird_dir)
    items = list(loader.items())
    no_evidence = [i for i in items if i.db_id == "debit_card_specializing" and "List all" in i.question]
    assert len(no_evidence) == 1
    assert no_evidence[0].evidence == ""


def test_item_dsn_format(bird_dir: Path) -> None:
    loader = BirdLoader(bird_dir)
    item = next(loader.items())
    assert item.dsn.startswith("sqlite:///")
    assert "debit_card_specializing.sqlite" in item.dsn


def test_item_build_connector(bird_dir: Path) -> None:
    loader = BirdLoader(bird_dir)
    item = next(loader.items())
    connector = item.build_connector()
    assert connector._conn is None
    connector.connect(item.dsn)
    schema = connector.get_schema()
    assert "CREATE TABLE" in schema.upper()
    connector.close()


# ── db_ids() ─────────────────────────────────────────────────────────────────


def test_db_ids_unique_ordered(bird_dir: Path) -> None:
    loader = BirdLoader(bird_dir)
    ids = loader.db_ids()
    assert ids == ["debit_card_specializing", "european_football_2"]


# ── BIRD-specific: SQL key (uppercase) ───────────────────────────────────────


def test_bird_uppercase_sql_key(tmp_path: Path) -> None:
    """BIRD uses 'SQL' (uppercase), not 'query' like Spider."""
    items = [
        {
            "question_id": 0,
            "db_id": "test_db",
            "question": "Count rows",
            "evidence": "hint",
            "SQL": "SELECT COUNT(*) FROM t",
            "difficulty": "simple",
        }
    ]
    root = _make_bird_dir(tmp_path, items)
    loader = BirdLoader(root)
    item = next(loader.items())
    assert item.gold_sql == "SELECT COUNT(*) FROM t"


def test_bird_fallback_to_lowercase_query_key(tmp_path: Path) -> None:
    """Gracefully handle items that use Spider-style 'query' field."""
    items = [
        {
            "db_id": "test_db",
            "question": "Count rows",
            "query": "SELECT COUNT(*) FROM t",
        }
    ]
    root = _make_bird_dir(tmp_path, items)
    loader = BirdLoader(root)
    item = next(loader.items())
    assert item.gold_sql == "SELECT COUNT(*) FROM t"


# ── difficulty stored in extra ────────────────────────────────────────────────


def test_difficulty_in_extra(bird_dir: Path) -> None:
    loader = BirdLoader(bird_dir)
    item = next(loader.items())
    assert item.extra.get("difficulty") == "simple"


# ── Missing / incomplete rows ─────────────────────────────────────────────────


def test_skips_incomplete_rows(tmp_path: Path) -> None:
    items = [
        {"db_id": "ok_db", "question": "Q?", "SQL": "SELECT 1", "evidence": "e"},
        {"db_id": "ok_db"},  # missing question + SQL — should be skipped
    ]
    root = _make_bird_dir(tmp_path, [items[0]])
    (root / "dev.json").write_text(json.dumps(items), encoding="utf-8")
    loader = BirdLoader(root, strict=False)
    result = list(loader.items())
    assert len(result) == 1


def test_strict_raises_on_missing_sqlite(tmp_path: Path) -> None:
    items = [{"db_id": "ghost_db", "question": "Q?", "SQL": "SELECT 1"}]
    (tmp_path / "dev.json").write_text(json.dumps(items), encoding="utf-8")
    loader = BirdLoader(tmp_path, strict=True)
    with pytest.raises(FileNotFoundError, match="ghost_db"):
        list(loader.items())


def test_non_strict_skips_missing_sqlite(tmp_path: Path) -> None:
    items = [{"db_id": "ghost_db", "question": "Q?", "SQL": "SELECT 1"}]
    (tmp_path / "dev.json").write_text(json.dumps(items), encoding="utf-8")
    loader = BirdLoader(tmp_path, strict=False)
    result = list(loader.items())
    assert result == []


# ── EvalItem reuse ────────────────────────────────────────────────────────────


def test_returns_eval_item_instances(bird_dir: Path) -> None:
    loader = BirdLoader(bird_dir)
    for item in loader.items():
        assert isinstance(item, EvalItem)


def test_eval_item_frozen(bird_dir: Path) -> None:
    loader = BirdLoader(bird_dir)
    item = next(loader.items())
    with pytest.raises((AttributeError, TypeError)):
        item.db_id = "changed"  # type: ignore[misc]


# ── inject_evidence() ─────────────────────────────────────────────────────────


def test_inject_evidence_calls_train_documentation(bird_dir: Path) -> None:
    loader = BirdLoader(bird_dir)
    item = next(loader.items())
    assert item.evidence  # sanity: this item has evidence

    mock_store = MagicMock()
    BirdLoader.inject_evidence(item, mock_store)

    mock_store.train_documentation.assert_called_once_with(item.evidence)


def test_inject_evidence_skips_empty_evidence(bird_dir: Path) -> None:
    items = list(BirdLoader(bird_dir).items())
    empty_evidence_item = next(i for i in items if i.evidence == "")

    mock_store = MagicMock()
    BirdLoader.inject_evidence(empty_evidence_item, mock_store)

    mock_store.train_documentation.assert_not_called()


def test_inject_evidence_is_classmethod_compatible(bird_dir: Path) -> None:
    """inject_evidence works without instantiating a loader."""
    items = list(BirdLoader(bird_dir).items())
    item_with_evidence = next(i for i in items if i.evidence)
    mock_store = MagicMock()
    BirdLoader.inject_evidence(item_with_evidence, mock_store)
    mock_store.train_documentation.assert_called_once()


# ── dev_databases layout ──────────────────────────────────────────────────────


def test_uses_dev_databases_subdir(bird_dir: Path) -> None:
    """BIRD uses dev_databases/, not database/ like Spider."""
    loader = BirdLoader(bird_dir)
    item = next(loader.items())
    assert "dev_databases" in str(item.db_path)


def test_train_split_uses_train_databases_subdir(tmp_path: Path) -> None:
    items = [
        {
            "db_id": "train_db",
            "question": "Count rows",
            "SQL": "SELECT COUNT(*) FROM t",
        }
    ]
    root = _make_bird_dir(tmp_path, items, split="train")
    loader = BirdLoader(root)
    item = next(loader.items(split="train"))
    assert "train_databases" in str(item.db_path)
