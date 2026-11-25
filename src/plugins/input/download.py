from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx2

logger = logging.getLogger("lecture_agent.plugins.input.download")

ProgressFn = Callable[[int, int], Any]

_PARALLEL_CONNECTIONS = 8


@dataclass(slots=True)
class DownloadResult:
    path: Path | None
    total: int
    ok: bool
    error: str = ""


class _Progress:
    """Thread-safe byte counter that reports at most every ``interval``."""

    def __init__(
        self, total: int, callback: ProgressFn | None, interval: float = 0.5
    ) -> None:
        self.total = int(total)
        self._callback = callback
        self._interval = float(interval)
        self._received = 0
        self._last = 0.0
        self._lock = threading.Lock()

    def add(self, size: int) -> None:
        with self._lock:
            self._received += int(size)
            now = time.monotonic()
            if self._callback is None:
                return
            if now - self._last < self._interval and self._received < self.total:
                return
            self._last = now
            received = self._received
        self._callback(received, self.total)

    @property
    def received(self) -> int:
        with self._lock:
            return self._received

    def finish(self) -> None:
        if self._callback is not None:
            self._callback(self.received, self.total or self.received)


class MediaDownloader:
    """Stream one media URL to ``dest`` (``.part`` → atomic rename)."""

    def __init__(
        self,
        *,
        chunk_size: int = 1 << 20,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
        connections: int = _PARALLEL_CONNECTIONS,
    ) -> None:
        self._chunk_size = max(64 * 1024, int(chunk_size))
        self._timeout = max(5.0, float(timeout))
        self._connections = max(1, int(connections))
        self._connection_condition = threading.Condition()
        self._connection_active = [0]
        self._connection_waiters = []
        self._headers = {"User-Agent": "sjtu-skipper/0.1"}
        if headers:
            self._headers.update(headers)
        self._headers["Accept-Encoding"] = "identity"

    def for_headers(self, headers: dict[str, str]) -> MediaDownloader:
        import copy

        clone = copy.copy(self)
        clone._headers = {**self._headers, **headers, "Accept-Encoding": "identity"}
        return clone

    def acquire_connection(self, cache, demand: bool) -> object:
        ticket = object()
        with self._connection_condition:
            self._connection_waiters.append((ticket, cache, demand))
            try:
                while True:
                    cache.check()
                    best = min(
                        self._connection_waiters,
                        key=lambda item: (not item[2], item[1].prefetch),
                    )
                    if (
                        best[0] is ticket
                        and self._connection_active[0] < self._connections
                    ):
                        self._connection_waiters.remove(best)
                        self._connection_active[0] += 1
                        return ticket
                    self._connection_condition.wait(0.1)
            except Exception:
                self._connection_waiters[:] = [
                    item for item in self._connection_waiters if item[0] is not ticket
                ]
                self._connection_condition.notify_all()
                raise

    def release_connection(self) -> None:
        with self._connection_condition:
            self._connection_active[0] -= 1
            self._connection_condition.notify_all()

    def start(
        self,
        url: str,
        dest: Path,
        *,
        stop_event: Any | None = None,
        on_progress: ProgressFn | None = None,
        progress_interval: float = 0.5,
    ) -> DownloadHandle:
        from .media_cache import MediaCache

        dest = Path(dest)
        # A recording is renamed from ``.part`` to ``dest`` only after every
        # byte arrived.  Reuse it instead of downloading the same class again
        # after a restart or a failed recognition pass.
        if dest.is_file() and dest.stat().st_size > 0:
            size = dest.stat().st_size
            if on_progress is not None:
                on_progress(size, size)
            return DownloadHandle(None, str(dest))

        progress = _Progress(0, on_progress, progress_interval)
        cache = MediaCache(self, url, dest, stop_event, progress)
        input_url = cache.start()
        return DownloadHandle(cache, input_url)

    def download(
        self,
        url: str,
        dest: Path,
        *,
        stop_event: Any | None = None,
        on_progress: ProgressFn | None = None,
        progress_interval: float = 0.5,
    ) -> DownloadResult:
        try:
            handle = self.start(
                url,
                dest,
                stop_event=stop_event,
                on_progress=on_progress,
                progress_interval=progress_interval,
            )
        except Exception as exc:
            return DownloadResult(None, 0, False, str(exc))
        try:
            return handle.wait()
        finally:
            handle.close()

    def _make_client(self, timeout: float | None = None) -> httpx2.Client:
        value = self._timeout if timeout is None else timeout
        return httpx2.Client(
            follow_redirects=True,
            timeout=httpx2.Timeout(value, read=value),
            headers=self._headers,
        )


class DownloadHandle:
    def __init__(self, cache, input_url: str) -> None:
        self._cache = cache
        self.input_url = input_url
        if cache is None:
            self.path = Path(input_url)
            self.total = self.path.stat().st_size
        else:
            self.path = cache.path
            self.total = cache.total

    def set_prefetch(self, value: bool) -> None:
        if self._cache is None:
            return
        self._cache.prefetch = bool(value)
        with self._cache.downloader._connection_condition:
            self._cache.downloader._connection_condition.notify_all()

    def wait(self) -> DownloadResult:
        if self._cache is None:
            return DownloadResult(self.path, self.total, True)
        self._cache.done.wait()
        if self._cache.error:
            return DownloadResult(
                None, self._cache.progress.received, False, self._cache.error
            )
        if not self.path.exists():
            return DownloadResult(None, 0, False, "cancelled")
        return DownloadResult(self.path, self.path.stat().st_size, True)

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()


__all__ = ["DownloadHandle", "DownloadResult", "MediaDownloader"]
