"""Download a recording to local disk before feeding it to ffmpeg.

Recording mode reads the lecture as fast as possible, so pulling it once is
cheaper than letting two ffmpeg processes (audio + video) stream it from the
campus portal at the same time.  The local copy is also what the UI replays.

The campus CDN throttles (and sometimes stalls) long single connections, so
large files are fetched as independent byte ranges over several connections;
each range resumes from its last byte and is retried a few times.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger("lecture_agent.plugins.input.download")

ProgressFn = Callable[[int, int], Any]

# Files above this size use ranged, parallel downloads.
_PARALLEL_MIN = 24 * 1024 * 1024
_CHUNK_SIZE = 8 * 1024 * 1024
_PARALLEL_CONNECTIONS = 4
_RANGE_ATTEMPTS = 4


@dataclass(slots=True)
class DownloadResult:
    path: Path | None
    total: int
    ok: bool
    error: str = ""


class _Cancelled(Exception):
    pass


class _Progress:
    """Thread-safe byte counter that reports at most every ``interval``."""

    def __init__(self, total: int, callback: ProgressFn | None, interval: float = 0.5) -> None:
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
        self._headers = {"User-Agent": "sjtu-skipper/0.1"}
        if headers:
            self._headers.update(headers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def download(
        self,
        url: str,
        dest: Path,
        *,
        stop_event: Any | None = None,
        on_progress: ProgressFn | None = None,
        progress_interval: float = 0.5,
    ) -> DownloadResult:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        progress: _Progress | None = None
        try:
            total, supports_range = self._probe(url)
            progress = _Progress(total, on_progress, progress_interval)
            if total > 0 and total >= _PARALLEL_MIN and supports_range:
                ok = self._download_ranges(url, part, total, stop_event, progress)
            else:
                ok = self._download_stream(url, part, stop_event, progress)
            if not ok:
                raise IOError("download did not complete")
            if total and part.stat().st_size != total:
                raise IOError(
                    f"incomplete download: {part.stat().st_size}/{total} bytes"
                )
            os.replace(part, dest)
        except _Cancelled:
            self._cleanup(part)
            return DownloadResult(path=None, total=0, ok=False, error="cancelled")
        except Exception as exc:  # noqa: BLE001 - falls back to remote ingest
            logger.warning("media download failed for %s: %s", url, exc)
            self._cleanup(part)
            return DownloadResult(
                path=None,
                total=progress.received if progress else 0,
                ok=False,
                error=str(exc),
            )
        if progress is not None:
            progress.finish()
        size = dest.stat().st_size
        return DownloadResult(path=dest, total=size, ok=True)

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------
    def _make_client(self, timeout: float | None = None) -> httpx.Client:
        value = self._timeout if timeout is None else timeout
        return httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(value, read=value),
            headers=self._headers,
        )

    def _probe(self, url: str) -> tuple[int, bool]:
        """Return ``(size, supports_range)``; ``size=0`` when unknown."""

        try:
            with self._make_client(20.0) as client:
                response = client.head(url)
                if response.status_code < 400:
                    size = int(response.headers.get("content-length") or 0)
                    accepts = "bytes" in (
                        response.headers.get("accept-ranges") or ""
                    ).lower()
                    if size > 0:
                        return size, accepts
        except httpx.HTTPError:
            pass
        try:
            with self._make_client(20.0) as client:
                response = client.get(url, headers={"Range": "bytes=0-0"})
                if response.status_code == 206:
                    match = re.search(
                        r"/(\d+)$", response.headers.get("content-range", "")
                    )
                    if match:
                        return int(match.group(1)), True
                size = int(response.headers.get("content-length") or 0)
                return size, False
        except httpx.HTTPError:
            return 0, False

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------
    def _download_stream(
        self,
        url: str,
        part: Path,
        stop_event: Any | None,
        progress: _Progress,
    ) -> bool:
        with self._make_client() as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(part, "wb") as handle:
                    for chunk in response.iter_bytes(self._chunk_size):
                        if stop_event is not None and stop_event.is_set():
                            raise _Cancelled
                        if not chunk:
                            continue
                        handle.write(chunk)
                        progress.add(len(chunk))
        return True

    def _download_ranges(
        self,
        url: str,
        part: Path,
        total: int,
        stop_event: Any | None,
        progress: _Progress,
    ) -> bool:
        chunks: list[tuple[int, int, Path]] = []
        index = 0
        start = 0
        while start < total:
            end = min(start + _CHUNK_SIZE - 1, total - 1)
            chunk_path = part.with_suffix(part.suffix + f".chunk{index:04d}")
            chunk_path.unlink(missing_ok=True)
            chunks.append((start, end, chunk_path))
            start = end + 1
            index += 1

        failures: list[str] = []
        queue = list(chunks)
        queue_lock = threading.Lock()
        abort = threading.Event()

        def worker() -> None:
            with self._make_client() as client:
                while not abort.is_set():
                    with queue_lock:
                        if not queue:
                            return
                        begin, end, path = queue.pop(0)
                    try:
                        self._fetch_range(
                            client, url, begin, end, path, stop_event, progress
                        )
                    except _Cancelled:
                        abort.set()
                        return
                    except Exception as exc:  # noqa: BLE001 - retried per chunk
                        failures.append(f"{begin}-{end}: {exc}")
                        return

        threads = [
            threading.Thread(target=worker, name=f"download-{i}", daemon=True)
            for i in range(min(self._connections, len(chunks)))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if abort.is_set():
            raise _Cancelled

        for _begin, _end, path in chunks:
            if not path.exists():
                self._cleanup_chunks(chunks)
                raise IOError(f"missing range {failures[:2]}")
        # Concatenate in offset order.
        with open(part, "wb") as out:
            for _begin, _end, path in chunks:
                with open(path, "rb") as handle:
                    while True:
                        block = handle.read(1 << 20)
                        if not block:
                            break
                        out.write(block)
        self._cleanup_chunks(chunks)
        return True

    def _fetch_range(
        self,
        client: httpx.Client,
        url: str,
        start: int,
        end: int,
        path: Path,
        stop_event: Any | None,
        progress: _Progress,
    ) -> None:
        expected = end - start + 1
        last_error = ""
        for attempt in range(_RANGE_ATTEMPTS):
            have = path.stat().st_size if path.exists() else 0
            if have >= expected:
                return
            headers = {"Range": f"bytes={start + have}-{end}"}
            try:
                with client.stream("GET", url, headers=headers) as response:
                    if response.status_code == 200:
                        # Server ignored the range: cannot slice this way.
                        raise IOError("server ignored Range request")
                    response.raise_for_status()
                    with open(path, "ab") as handle:
                        for block in response.iter_bytes(64 * 1024):
                            if stop_event is not None and stop_event.is_set():
                                raise _Cancelled
                            if not block:
                                continue
                            handle.write(block)
                            progress.add(len(block))
            except _Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - retry from last byte
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < _RANGE_ATTEMPTS:
                    time.sleep(min(8.0, 1.0 * (2**attempt)))
                continue
            have = path.stat().st_size if path.exists() else 0
            if have >= expected:
                return
        raise IOError(
            f"range {start}-{end} incomplete after {_RANGE_ATTEMPTS} attempts: "
            f"{path.stat().st_size if path.exists() else 0}/{expected} ({last_error})"
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    @staticmethod
    def _cleanup_chunks(chunks: list[tuple[int, int, Path]]) -> None:
        for _begin, _end, path in chunks:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _cleanup(self, part: Path) -> None:
        try:
            part.unlink(missing_ok=True)
        except OSError:
            pass
        for candidate in part.parent.glob(part.name + ".chunk*"):
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = ["DownloadResult", "MediaDownloader"]
