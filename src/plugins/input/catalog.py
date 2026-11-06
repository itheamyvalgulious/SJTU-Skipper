"""Catalog abstraction: resolve courses → recordings → stream URLs.

Wraps ``SJTUCatalog`` with settings-driven configuration and caching.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

from integrations.sjtu_canvas import SJTUCatalog

_log = logging.getLogger("lecture_agent.plugins.input.catalog")
T = TypeVar("T")


class Catalog:
    """Lazy catalog that reads token/base-url from a settings service."""

    def __init__(
        self,
        *,
        settings: Any,  # settings service (FakeSettings or real)
    ) -> None:
        self._settings = settings
        self._sjt_cat: SJTUCatalog | None = None
        self._cache: dict[tuple[Any, ...], tuple[float, Any]] = {}

    def _cached(self, key: tuple[Any, ...], ttl: float, loader: Callable[[], T]) -> T:
        """Small TTL cache; UI state() re-renders toolbox items every poll."""

        now = time.monotonic()
        hit = self._cache.get(key)
        if hit is not None and now - hit[0] < ttl:
            return hit[1]
        value = loader()
        self._cache[key] = (now, value)
        return value

    def _cat(self) -> SJTUCatalog:
        if self._sjt_cat is None:
            token = str(self._settings.get("input.canvas_token", "") or "")
            base_url = str(
                self._settings.get(
                    "input.canvas_base_url", "https://oc.sjtu.edu.cn"
                )
                or "https://oc.sjtu.edu.cn"
            )
            # The settings service is the single source of truth; do not fall
            # back to a hidden environment read here.
            self._sjt_cat = SJTUCatalog(token=token, base_url=base_url)
        return self._sjt_cat

    def has_token(self) -> bool:
        return bool(self._settings.get("input.canvas_token", ""))

    # -- Courses ----------------------------------------------------------

    def list_courses(self) -> list[dict[str, Any]]:
        return self._cached(("courses",), 60.0, lambda: self._cat().list_courses())

    # -- Recordings -------------------------------------------------------

    def list_recordings(
        self, course_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        return self._cached(
            ("recordings", str(course_id), int(limit)),
            30.0,
            lambda: self._cat().list_recordings(course_id, limit=limit),
        )

    def get_recording_url(self, course_id: str, video_id: str) -> str:
        return self._cached(
            ("recording_url", str(course_id), str(video_id)),
            60.0,
            lambda: self._cat().get_recording_url(course_id, video_id),
        )

    def get_recording_streams(
        self, course_id: str, video_id: str
    ) -> dict[str, str]:
        """Per-view URLs (``"1"``=课堂, ``"2"``=屏幕)."""

        return self._cached(
            ("recording_streams", str(course_id), str(video_id)),
            60.0,
            lambda: self._cat().get_recording_streams(course_id, video_id),
        )

    # -- Live -------------------------------------------------------------

    def list_lives(self, start: str, end: str) -> list[dict[str, Any]]:
        return self._cached(
            ("lives", start, end),
            15.0,
            lambda: self._cat().list_lives(start, end),
        )

    def get_live_devices(self, live_id: str) -> list[dict[str, Any]]:
        return self._cached(
            ("live_devices", str(live_id)),
            30.0,
            lambda: self._cat().get_live_devices(live_id),
        )

    # -- Lifecycle --------------------------------------------------------

    def close(self) -> None:
        self._cache.clear()
        if self._sjt_cat is not None:
            self._sjt_cat.close()
            self._sjt_cat = None

    def reset(self) -> None:
        """Discard cached catalog (e.g. after token change)."""
        self.close()