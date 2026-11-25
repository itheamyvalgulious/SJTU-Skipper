"""SQLite-backed learning-state entries: knowledge point aliases + level."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

MAX_KEYWORDS = 8

# Ordered from least to most mastery.  ``unknown`` is a stored value meaning
# "asked and the student does not know"; a missing row means "not assessed".
VALUES = ("unknown", "impression", "understand", "proficient", "preferred")
VALUE_LABELS = {
    "unknown": "不了解",
    "impression": "有大概印象",
    "understand": "肯定能看懂",
    "proficient": "比较熟练",
    "preferred": "喜欢用这个",
}
VALUE_DESCRIPTIONS = {
    "unknown": "没学过/完全不了解",
    "impression": "大概知道思路、流程或用途，但细节不清楚",
    "understand": "看到别人使用它时能够完全看懂，但自己可能无法主动想到要用",
    "proficient": "能够熟练使用",
    "preferred": "能够熟练使用，并且相比其它方法/知识点更倾向于或习惯使用这个",
}


def value_label(value: str) -> str:
    return VALUE_LABELS.get(str(value), "不了解")


def infer_value(description: str) -> str:
    """Best-effort migration for rows created by the old free-text schema."""

    text = str(description or "")
    if any(token in text for token in ("偏好", "习惯")):
        return "preferred"
    if any(token in text for token in ("不熟", "学过一些", "需要回顾", "听过", "见过", "印象")):
        return "impression"
    if any(token in text for token in ("擅长", "熟练", "很会", "拿手")):
        return "proficient"
    if any(token in text for token in ("能看懂", "看得懂", "能理解", "能跟上")):
        return "understand"
    if any(token in text for token in ("没学过", "没听", "缺席", "没来", "不会", "不了解", "不知道")):
        return "unknown"
    if "学过" in text:
        return "understand"
    return "unknown"


class LearningStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._migrate(self._conn)
        return self._conn

    @staticmethod
    def _create_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE TABLE entries ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "keywords TEXT NOT NULL,"
            "value TEXT NOT NULL,"
            "created_at REAL NOT NULL,"
            "updated_at REAL NOT NULL)"
        )

    def _migrate(self, conn: sqlite3.Connection) -> None:
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(entries)").fetchall()
        }
        if not columns:
            self._create_table(conn)
            conn.commit()
            return
        if "value" in columns:
            return
        # Old schema: keywords + free-text description.  Preserve rows by
        # mapping descriptions onto the new fixed levels.
        legacy = conn.execute(
            "SELECT id, keywords, description, created_at, updated_at FROM entries ORDER BY id"
        ).fetchall()
        conn.execute("ALTER TABLE entries RENAME TO entries_legacy")
        self._create_table(conn)
        for row in legacy:
            try:
                keywords = json.loads(row[1])
            except (TypeError, json.JSONDecodeError):
                keywords = [str(row[1])]
            conn.execute(
                "INSERT INTO entries (id, keywords, value, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (
                    row[0],
                    json.dumps(keywords, ensure_ascii=False),
                    infer_value(str(row[2] or "")),
                    float(row[3] or time.time()),
                    float(row[4] or time.time()),
                ),
            )
        conn.execute("DROP TABLE entries_legacy")
        conn.commit()

    @staticmethod
    def _clean_keywords(keywords: list[str] | str) -> list[str]:
        items = keywords if isinstance(keywords, list) else str(keywords).split(",")
        cleaned: list[str] = []
        for item in items:
            text = str(item).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned[:MAX_KEYWORDS]

    def search(self, keywords: list[str] | str, *, limit: int = 20) -> list[dict[str, Any]]:
        cleaned = self._clean_keywords(keywords)
        if not cleaned:
            return []
        with self._lock:
            conn = self._connection()
            clauses: list[str] = []
            params: list[Any] = []
            for keyword in cleaned:
                clauses.append("LOWER(keywords) LIKE ?")
                params.append(f"%{keyword.lower()}%")
            sql = (
                "SELECT id, keywords, value, updated_at FROM entries WHERE "
                + " OR ".join(clauses)
                + " ORDER BY updated_at DESC LIMIT ?"
            )
            params.append(max(1, min(50, int(limit))))
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "id": row[0],
                "keywords": json.loads(row[1]),
                "value": row[2],
                "label": value_label(row[2]),
                "updated_at": row[3],
            }
            for row in rows
        ]

    def upsert(
        self,
        keywords: list[str] | str,
        value: str,
        *,
        mode: str = "insert",
        entry_id: int | str | None = None,
    ) -> dict[str, Any]:
        level = str(value or "").strip()
        if level not in VALUES:
            return {"ok": False, "message": "value 必须是五级状态之一"}
        now = time.time()
        with self._lock:
            conn = self._connection()
            if str(mode) == "replace":
                try:
                    target = int(entry_id)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return {
                        "ok": False,
                        "message": "replace 必须提供 learning_search 返回的 id；只允许按 id 更新 value",
                    }
                cursor = conn.execute(
                    "UPDATE entries SET value=?, updated_at=? WHERE id=?",
                    (level, now, target),
                )
                conn.commit()
                if not cursor.rowcount:
                    return {"ok": False, "message": f"没有找到学情条目 #{target}"}
                return {"ok": True, "id": target, "message": f"已更新学情条目 #{target}"}
        cleaned = self._clean_keywords(keywords)
        if not cleaned:
            return {"ok": False, "message": "insert 时 keywords 不能为空"}
        with self._lock:
            conn = self._connection()
            cursor = conn.execute(
                "INSERT INTO entries (keywords, value, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (json.dumps(cleaned, ensure_ascii=False), level, now, now),
            )
            conn.commit()
            return {"ok": True, "id": cursor.lastrowid, "message": "已新增学情条目"}

    def update_value(self, entry_id: int, value: str) -> bool:
        level = str(value or "").strip()
        if int(entry_id) <= 0 or level not in VALUES:
            return False
        with self._lock:
            conn = self._connection()
            cursor = conn.execute(
                "UPDATE entries SET value=?, updated_at=? WHERE id=?",
                (level, time.time(), int(entry_id)),
            )
            conn.commit()
            return cursor.rowcount > 0

    def remove(self, entry_id: int) -> bool:
        with self._lock:
            conn = self._connection()
            cursor = conn.execute("DELETE FROM entries WHERE id=?", (int(entry_id),))
            conn.commit()
            return cursor.rowcount > 0

    def items(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection().execute(
                "SELECT id, keywords, value, updated_at FROM entries ORDER BY updated_at DESC"
            ).fetchall()
        return [
            {
                "id": row[0],
                "keywords": json.loads(row[1]),
                "value": row[2],
                "label": value_label(row[2]),
                "updated_at": row[3],
            }
            for row in rows
        ]


__all__ = [
    "LearningStore",
    "MAX_KEYWORDS",
    "VALUES",
    "VALUE_LABELS",
    "VALUE_DESCRIPTIONS",
    "value_label",
    "infer_value",
]
