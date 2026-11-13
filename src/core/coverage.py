"""Coverage bookkeeping for the STT / VLM pipelines.

A session's media timeline can have holes: the app may start after the class
began, be paused in the middle, or crash and restart.  Each pipeline records
which ``[start, end)`` ranges it actually processed, persisted next to the
session, so a later pass can skip what is already done and fill only the gaps.

Times are seconds on the *session's own* media timeline.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping

Interval = tuple[float, float]
KINDS = ("audio", "video")

_EPS = 1e-6


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Return the sorted, merged union of *intervals*."""

    ordered = sorted(
        (max(0.0, float(a)), max(0.0, float(b)))
        for a, b in intervals
        if float(b) > float(a)
    )
    merged: list[Interval] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + _EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


class Coverage:
    """Union of processed time ranges (values are seconds)."""

    __slots__ = ("_items",)

    def __init__(self, intervals: Iterable[Interval] = ()) -> None:
        self._items: list[Interval] = merge_intervals(intervals)

    @property
    def intervals(self) -> list[Interval]:
        return list(self._items)

    def to_list(self) -> list[list[float]]:
        return [[start, end] for start, end in self._items]

    @classmethod
    def from_any(cls, raw: Any) -> "Coverage":
        if isinstance(raw, Coverage):
            return cls(raw.intervals)
        items: list[Interval] = []
        if isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    try:
                        items.append((float(entry[0]), float(entry[1])))
                    except (TypeError, ValueError):
                        continue
        return cls(items)

    def add(self, start: float, end: float) -> None:
        if float(end) <= float(start):
            return
        self._items = merge_intervals([*self._items, (start, end)])

    def total(self) -> float:
        return sum(end - start for start, end in self._items)

    def overlaps(self, start: float, end: float) -> float:
        """Seconds of ``[start, end)`` that are already covered."""

        if end <= start:
            return 0.0
        covered = 0.0
        for low, high in self._items:
            if high <= start:
                continue
            if low >= end:
                break
            covered += max(0.0, min(high, end) - max(low, start))
        return covered

    def covers(self, start: float, end: float) -> bool:
        """True when *all* of ``[start, end)`` is covered."""

        if end <= start:
            return True
        return self.overlaps(start, end) >= (end - start) - 1e-3

    def gaps(self, start: float, end: float) -> list[Interval]:
        """Parts of ``[start, end)`` that are *not* covered."""

        if end <= start:
            return []
        holes: list[Interval] = []
        cursor = float(start)
        for low, high in self._items:
            if high <= cursor:
                continue
            if low >= end:
                break
            if low > cursor:
                holes.append((cursor, min(low, end)))
            cursor = max(cursor, min(high, end))
            if cursor >= end:
                break
        if cursor < end:
            holes.append((cursor, float(end)))
        return holes


class CoverageStore:
    """Per-session coverage, persisted as ``sessions/<id>/coverage.json``.

    Writes are atomic (tmp + replace) and cheap: STT marks once per chunk and
    VLM once per window, so the file stays tiny.
    """

    def __init__(self, data_dir: Path | str, session_id: str) -> None:
        self.session_id = str(session_id)
        self.path = Path(data_dir) / "sessions" / self.session_id / "coverage.json"
        self._lock = threading.RLock()
        self._data: dict[str, Coverage] = {}
        self._load()

    # -- io ------------------------------------------------------------
    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text("utf-8"))
        except (OSError, ValueError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        self._data = {kind: Coverage.from_any(raw.get(kind)) for kind in KINDS}

    def _save(self) -> None:
        payload = {kind: self._data[kind].to_list() for kind in KINDS}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload), "utf-8")
            tmp.replace(self.path)
        except OSError:
            # Coverage is an optimisation; never break the pipeline over it.
            pass

    # -- api -----------------------------------------------------------
    def mark(self, kind: str, start: float, end: float) -> None:
        """Record that ``[start, end)`` of *kind* has been processed."""

        if kind not in self._data:
            return
        if float(end) <= float(start):
            return
        with self._lock:
            before = self._data[kind].total()
            self._data[kind].add(start, end)
            if self._data[kind].total() > before + _EPS:
                self._save()

    def coverage(self, kind: str) -> Coverage:
        with self._lock:
            return Coverage(self._data[kind].intervals)

    def covered_seconds(self, kind: str) -> float:
        with self._lock:
            return self._data[kind].total()

    def gaps(self, kind: str, start: float, end: float) -> list[Interval]:
        with self._lock:
            return self._data[kind].gaps(start, end)

    def covers(self, kind: str, start: float, end: float) -> bool:
        with self._lock:
            return self._data[kind].covers(start, end)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                kind: {
                    "intervals": self._data[kind].to_list(),
                    "seconds": self._data[kind].total(),
                }
                for kind in KINDS
            }


__all__ = [
    "Coverage",
    "CoverageStore",
    "Interval",
    "KINDS",
    "merge_intervals",
]


def skip_coverage(meta: Mapping[str, Any] | None, kind: str) -> Coverage:
    """Ranges a *fill* session must not process again.

    The input plugin puts them on the session meta when it starts a
    gap-filling pass over the class recording; they are expressed in that
    recording's media seconds.
    """

    raw = (meta or {}).get("skip_ranges")
    if not isinstance(raw, Mapping):
        return Coverage()
    return Coverage.from_any(raw.get(kind))
