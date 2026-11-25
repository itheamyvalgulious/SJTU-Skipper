from __future__ import annotations

import sqlite3
from pathlib import Path

from plugins.learning.store import LearningStore, VALUE_LABELS


def test_learning_store_migrates_old_description_schema(tmp_path: Path) -> None:
    path = tmp_path / "learning.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE entries (id INTEGER PRIMARY KEY AUTOINCREMENT, keywords TEXT NOT NULL, "
        "description TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL)"
    )
    conn.execute(
        "INSERT INTO entries (keywords, description, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ('["伽罗瓦理论"]', "没学过", 1.0, 1.0),
    )
    conn.execute(
        "INSERT INTO entries (keywords, description, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ('["有限域"]', "学过一些，还不熟", 2.0, 2.0),
    )
    conn.commit()
    conn.close()

    store = LearningStore(path)
    rows = store.items()
    assert [row["value"] for row in rows] == ["impression", "unknown"]
    assert rows[0]["label"] == VALUE_LABELS["impression"]


def test_learning_store_upsert_and_update_value(tmp_path: Path) -> None:
    store = LearningStore(tmp_path / "learning.db")
    created = store.upsert(["仿射簇", "affine variety"], "unknown")
    assert created["ok"]
    found = store.search(["仿射簇"])
    assert found[0]["value"] == "unknown"
    assert store.update_value(found[0]["id"], "understand")
    assert store.search(["仿射簇"])[0]["label"] == "肯定能看懂"
    replaced = store.upsert(["different"], "proficient", mode="replace", entry_id=found[0]["id"])
    assert replaced["ok"]
    rows = store.search(["仿射簇"])
    assert len(rows) == 1
    assert rows[0]["value"] == "proficient"
    assert store.upsert(["x"], "unknown", mode="replace")["ok"] is False
