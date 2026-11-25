"""Shared teaching-preference profile for the tutor plugin.

Design notes
------------
Preferences are a *profile*, not a collection: a small, always-relevant list
where new information updates existing entries instead of accumulating.  The
agent never writes here directly; it may only *propose* changes, which are
stored as pending proposals until the user accepts them in the settings page.

The file lives at ``<data_dir>/tutor/preferences.json`` and is written
atomically.  ``defaults.json`` next to this module seeds a fresh install and
is also the export target of "把我的偏好发布成初始设置".
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

CATEGORIES: tuple[str, ...] = (
    "style",    # 表达风格
    "depth",    # 讲解深度
    "content",  # 内容偏好
    "format",   # 排版/公式
    "dislike",  # 不喜欢
    "goal",     # 学习目标
)
CATEGORY_LABELS: dict[str, str] = {
    "style": "表达风格",
    "depth": "讲解深度",
    "content": "内容偏好",
    "format": "格式偏好",
    "dislike": "不喜欢",
    "goal": "学习目标",
}
SCOPES: tuple[str, ...] = ("both", "ask", "notes")
SCOPE_LABELS: dict[str, str] = {"both": "问答+笔记", "ask": "仅问答", "notes": "仅笔记"}

MAX_ITEMS = 40
MAX_TEXT = 120
MAX_PENDING = 8
MAX_OPS = 6

_WS = re.compile(r"\s+")


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def normalize_text(text: str) -> str:
    return _WS.sub(" ", str(text or "")).strip()


def _dedupe_key(text: str) -> str:
    return re.sub(r"[\s，。、；：！？,.;:!?\"'“”‘’（）()\[\]【】…—-]+", "", normalize_text(text)).lower()


@dataclass(slots=True)
class Preference:
    id: str
    text: str
    category: str = "style"
    scope: str = "both"
    source: str = "manual"
    updated_at: float = field(default_factory=_now)
    evidence: str = ""

    def label(self) -> str:
        return CATEGORY_LABELS.get(self.category, self.category)

    def scope_label(self) -> str:
        return SCOPE_LABELS.get(self.scope, self.scope)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "category": self.category,
            "scope": self.scope,
            "source": self.source,
            "updated_at": self.updated_at,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Preference":
        category = str(raw.get("category") or "style")
        if category not in CATEGORIES:
            category = "style"
        scope = str(raw.get("scope") or "both")
        if scope not in SCOPES:
            scope = "both"
        return cls(
            id=str(raw.get("id") or _new_id()),
            text=normalize_text(str(raw.get("text") or "")),
            category=category,
            scope=scope,
            source=str(raw.get("source") or "manual"),
            updated_at=float(raw.get("updated_at") or _now()),
            evidence=str(raw.get("evidence") or ""),
        )


@dataclass(slots=True)
class Proposal:
    id: str
    operations: list[dict[str, Any]]
    source: str = "ask"
    reason: str = ""
    status: str = "pending"
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "operations": [dict(op) for op in self.operations],
            "source": self.source,
            "reason": self.reason,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Proposal":
        operations = [
            dict(op) for op in raw.get("operations") or [] if isinstance(op, dict)
        ]
        return cls(
            id=str(raw.get("id") or _new_id()),
            operations=operations,
            source=str(raw.get("source") or "ask"),
            reason=str(raw.get("reason") or ""),
            status=str(raw.get("status") or "pending"),
            created_at=float(raw.get("created_at") or _now()),
        )


def normalize_operations(raw: Iterable[Any]) -> list[dict[str, Any]]:
    """Validate model-proposed operations; keep only well-formed ones."""

    operations: list[dict[str, Any]] = []
    for item in list(raw)[:MAX_OPS]:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op") or item.get("operation") or "").strip().lower()
        op = {"delete": "remove", "del": "remove", "add_item": "add"}.get(op, op)
        if op not in ("add", "update", "remove"):
            continue
        entry: dict[str, Any] = {"op": op}
        if op in ("add", "update"):
            text = normalize_text(str(item.get("text") or ""))
            if not text or len(text) > MAX_TEXT:
                continue
            entry["text"] = text
            category = str(item.get("category") or "style")
            entry["category"] = category if category in CATEGORIES else "style"
            scope = str(item.get("scope") or "both")
            entry["scope"] = scope if scope in SCOPES else "both"
        if op in ("update", "remove"):
            target = str(item.get("id") or "").strip()
            if not target:
                continue
            entry["id"] = target
        operations.append(entry)
    return operations


class PreferenceStore:
    """Thread-safe preference profile + pending agent proposals.

    Every mutation is persisted atomically.  ``propose`` never touches the
    profile itself; only ``accept`` applies a proposal.
    """

    def __init__(self, path: Path, *, defaults_path: Path | None = None) -> None:
        self.path = Path(path)
        self.defaults_path = Path(defaults_path) if defaults_path else None
        self._lock = threading.RLock()
        self._items: list[Preference] = []
        self._pending: list[Proposal] = []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        with self._lock:
            raw = self._read(self.path)
            if raw is None:
                self._items = self._read_defaults()
                self._pending = []
                self._save()
                return
            self._items = [Preference.from_dict(item) for item in raw.get("items") or [] if isinstance(item, dict)]
            self._items = [item for item in self._items if item.text]
            self._pending = [Proposal.from_dict(item) for item in raw.get("pending") or [] if isinstance(item, dict)]
            self._pending = [p for p in self._pending if p.status == "pending"]

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _read_defaults(self) -> list[Preference]:
        if self.defaults_path is None:
            return []
        raw = self._read(self.defaults_path) or {}
        items: list[Preference] = []
        for entry in raw.get("items") or []:
            if not isinstance(entry, dict):
                continue
            item = Preference.from_dict(entry)
            if item.text:
                item.source = "default"
                items.append(item)
        return items

    def _save(self) -> None:
        payload = {
            "version": 1,
            "items": [item.to_dict() for item in self._items],
            "pending": [proposal.to_dict() for proposal in self._pending],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=f".{self.path.name}.", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------
    # Profile queries
    # ------------------------------------------------------------------
    def items(self) -> list[Preference]:
        with self._lock:
            return list(self._items)

    def get(self, item_id: str) -> Preference | None:
        with self._lock:
            return next((item for item in self._items if item.id == item_id), None)

    def pending(self) -> list[Proposal]:
        with self._lock:
            return list(self._pending)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "items": [item.to_dict() for item in self._items],
                "pending": [proposal.to_dict() for proposal in self._pending],
            }

    # ------------------------------------------------------------------
    # Profile mutations
    # ------------------------------------------------------------------
    def add(
        self,
        text: str,
        *,
        category: str = "style",
        scope: str = "both",
        source: str = "manual",
        evidence: str = "",
    ) -> tuple[bool, str]:
        cleaned = normalize_text(text)
        if not cleaned:
            return False, "偏好内容为空。"
        if len(cleaned) > MAX_TEXT:
            return False, f"偏好太长（>{MAX_TEXT} 字）。"
        if category not in CATEGORIES:
            category = "style"
        if scope not in SCOPES:
            scope = "both"
        key = _dedupe_key(cleaned)
        with self._lock:
            for item in self._items:
                if _dedupe_key(item.text) == key:
                    return False, f"已经有一条相同偏好：{item.text}"
            if len(self._items) >= MAX_ITEMS:
                return False, f"偏好数量已达上限（{MAX_ITEMS} 条），请先删掉不再适用的。"
            item = Preference(
                id=_new_id(),
                text=cleaned,
                category=category,
                scope=scope,
                source=source,
                evidence=evidence,
            )
            self._items.append(item)
            self._save()
            return True, f"已添加（{item.label()}）：{item.text}"

    def remove(self, item_id: str) -> bool:
        with self._lock:
            before = len(self._items)
            self._items = [item for item in self._items if item.id != item_id]
            if len(self._items) == before:
                return False
            self._save()
            return True

    def clear(self) -> int:
        with self._lock:
            count = len(self._items)
            self._items = []
            self._pending = []
            self._save()
            return count

    def reset_to_defaults(self) -> int:
        with self._lock:
            self._items = self._read_defaults()
            self._pending = []
            self._save()
            return len(self._items)

    def export_defaults(self, target: Path | None = None) -> Path:
        destination = Path(target) if target else self.defaults_path
        if destination is None:
            raise ValueError("no defaults path configured")
        payload = {
            "version": 1,
            "items": [
                {
                    "text": item.text,
                    "category": item.category,
                    "scope": item.scope,
                }
                for item in self.items()
            ],
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8"
        )
        return destination

    # ------------------------------------------------------------------
    # Proposals (agent may only propose; the user accepts)
    # ------------------------------------------------------------------
    def propose(
        self,
        operations: Iterable[Any],
        *,
        source: str = "ask",
        reason: str = "",
    ) -> Proposal | None:
        clean = normalize_operations(operations)
        if not clean:
            return None
        with self._lock:
            if len(self._pending) >= MAX_PENDING:
                return None
            proposal = Proposal(
                id=_new_id(),
                operations=clean,
                source=source,
                reason=normalize_text(reason)[:200],
            )
            self._pending.append(proposal)
            self._save()
            return proposal

    def accept(self, proposal_id: str) -> tuple[bool, str]:
        with self._lock:
            proposal = next(
                (item for item in self._pending if item.id == proposal_id), None
            )
            if proposal is None:
                return False, "没有找到待确认的偏好建议。"
            applied = 0
            messages: list[str] = []
            for op in proposal.operations:
                kind = op.get("op")
                if kind == "add":
                    ok, message = self._apply_add(dict(op))
                    messages.append(message)
                    applied += 1 if ok else 0
                elif kind == "update":
                    ok, message = self._apply_update(dict(op))
                    messages.append(message)
                    applied += 1 if ok else 0
                elif kind == "remove":
                    ok, message = self._apply_remove(str(op.get("id") or ""))
                    messages.append(message)
                    applied += 1 if ok else 0
            self._pending = [item for item in self._pending if item.id != proposal_id]
            self._save()
            detail = "；".join(message for message in messages if message)
            return applied > 0, f"已应用 {applied} 条偏好" + (f"（{detail}）" if detail else "")

    def reject(self, proposal_id: str) -> bool:
        with self._lock:
            before = len(self._pending)
            self._pending = [item for item in self._pending if item.id != proposal_id]
            if len(self._pending) == before:
                return False
            self._save()
            return True

    # -- internal apply helpers (must be called with lock held) ---------
    def _apply_add(self, op: dict[str, Any]) -> tuple[bool, str]:
        text = normalize_text(str(op.get("text") or ""))
        key = _dedupe_key(text)
        if any(_dedupe_key(item.text) == key for item in self._items):
            return False, f"已存在：{text}"
        if len(self._items) >= MAX_ITEMS:
            return False, "偏好数量已达上限"
        item = Preference(
            id=_new_id(),
            text=text,
            category=str(op.get("category") or "style"),
            scope=str(op.get("scope") or "both"),
            source="ask",
        )
        if item.category not in CATEGORIES:
            item.category = "style"
        if item.scope not in SCOPES:
            item.scope = "both"
        self._items.append(item)
        return True, f"新增「{item.text}」"

    def _apply_update(self, op: dict[str, Any]) -> tuple[bool, str]:
        target = self.get(str(op.get("id") or ""))
        if target is None:
            return False, "建议更新的偏好已不存在"
        text = normalize_text(str(op.get("text") or ""))
        if text:
            target.text = text
        category = str(op.get("category") or "")
        if category in CATEGORIES:
            target.category = category
        scope = str(op.get("scope") or "")
        if scope in SCOPES:
            target.scope = scope
        target.updated_at = _now()
        return True, f"更新「{target.text}」"

    def _apply_remove(self, item_id: str) -> tuple[bool, str]:
        target = self.get(item_id)
        if target is None:
            return False, "建议删除的偏好已不存在"
        self._items = [item for item in self._items if item.id != item_id]
        return True, f"删除「{target.text}」"

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------
    def block(self, scopes: Iterable[str] = ("both", "ask", "notes")) -> str:
        allowed = set(scopes)
        items = [item for item in self.items() if item.scope in allowed]
        header = (
            "## 学生长期偏好（跨对话记忆，学生已确认）\n"
            "下面每条都由学生本人确认过，优先遵守；与当轮明确要求冲突时以当轮为准，"
            "不要因为一次临时要求修改它们。格式：#id 是条目编号，供你提议修改时引用。\n"
        )
        if not items:
            return header + "- （暂无偏好）\n"
        lines = [f"- [{item.label()}] #{item.id} {item.text}" for item in items]
        return header + "\n".join(lines) + "\n"


__all__ = [
    "CATEGORIES",
    "CATEGORY_LABELS",
    "MAX_ITEMS",
    "MAX_TEXT",
    "Preference",
    "PreferenceStore",
    "Proposal",
    "SCOPES",
    "SCOPE_LABELS",
    "normalize_operations",
    "normalize_text",
]
