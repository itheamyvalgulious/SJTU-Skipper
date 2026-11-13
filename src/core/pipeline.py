"""Concurrent window processing with in-order commits.

Recording mode is not CPU bound: every window spends most of its life waiting
for Google STT or the vision model.  :class:`OrderedWindowPool` therefore runs
several windows at once on a thread pool (network waits release the GIL) while
still committing results strictly in window order, so ``transcript`` and
``visual`` keep a monotonically increasing timeline.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger("lecture_agent.pipeline")

ProcessFn = Callable[[float, float], Any]
CommitFn = Callable[[float, float, Any], None]


class WindowError:
    """Marker result for a window whose processing raised."""

    __slots__ = ("error",)

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"WindowError({self.error!r})"


class OrderedWindowPool:
    """Submit windows concurrently, commit results in submission order.

    Parameters
    ----------
    name:
        Thread name prefix and log tag.
    window:
        Nominal window length in media seconds.
    concurrency:
        Number of simultaneously in-flight windows.
    process:
        ``(start, end) -> result``, executed on the pool threads.
    commit:
        ``(start, end, result)``, always called from the caller thread and in
        ascending window order.  ``result`` may be a :class:`WindowError`.
    max_backlog:
        Extra completed-but-not-yet-committed windows allowed before
        :meth:`submit` refuses new work (bounds pinning a slow head window).
    """

    def __init__(
        self,
        *,
        name: str,
        window: float,
        concurrency: int,
        process: ProcessFn,
        commit: CommitFn,
        max_backlog: int = 3,
        start: float = 0.0,
    ) -> None:
        self.name = name
        self.window = max(0.1, float(window))
        self.concurrency = max(1, int(concurrency))
        self._process = process
        self._commit = commit
        self._max_backlog = max(1, int(max_backlog))
        self._executor = ThreadPoolExecutor(
            max_workers=self.concurrency,
            thread_name_prefix=name,
        )
        self._lock = threading.Lock()
        self._next_index = 0
        self._next_start = max(0.0, float(start))
        self._submitted_end = self._next_start
        self._commit_index = 0
        self._committed_end = self._next_start
        self._inflight: dict[int, tuple[float, float, Future[Any]]] = {}
        self._finished: dict[int, tuple[float, float, Any]] = {}
        self._closed = False

    # ------------------------------------------------------------------
    # Sizing / state
    # ------------------------------------------------------------------
    @property
    def next_start(self) -> float:
        with self._lock:
            return self._next_start

    @property
    def submitted_end(self) -> float:
        with self._lock:
            return self._submitted_end

    @property
    def committed_end(self) -> float:
        with self._lock:
            return self._committed_end

    def pending(self) -> int:
        with self._lock:
            return len(self._inflight) + len(self._finished)

    def busy(self) -> bool:
        return self.pending() > 0

    def can_submit(self) -> bool:
        with self._lock:
            return (
                not self._closed
                and len(self._inflight) < self.concurrency
                and len(self._finished) < self._max_backlog
            )

    # ------------------------------------------------------------------
    # Work
    # ------------------------------------------------------------------
    def submit(self, start: float, end: float) -> bool:
        """Queue one window; returns False when the pool is saturated."""

        if end - start <= 1e-6:
            return False
        with self._lock:
            if self._closed:
                return False
            if len(self._inflight) >= self.concurrency:
                return False
            if len(self._finished) >= self._max_backlog:
                return False
            index = self._next_index
            self._next_index += 1
            self._next_start = float(end)
            self._submitted_end = max(self._submitted_end, float(end))
        future = self._executor.submit(self._run, start, end)
        with self._lock:
            self._inflight[index] = (float(start), float(end), future)
        return True

    def _run(self, start: float, end: float) -> Any:
        try:
            return self._process(start, end)
        except Exception as exc:  # noqa: BLE001 - reported through commit
            logger.warning(
                "%s window %.1f-%.1f failed: %s", self.name, start, end, exc
            )
            return WindowError(exc)

    def poll(self, timeout: float | None = None) -> int:
        """Commit every consecutive finished window; returns how many."""

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            ready: list[tuple[float, float, Any]] = []
            with self._lock:
                for index, (start, end, future) in list(self._inflight.items()):
                    if future.done():
                        self._finished[index] = (start, end, future.result())
                        del self._inflight[index]
                while self._commit_index in self._finished:
                    ready.append(self._finished.pop(self._commit_index))
                    self._commit_index += 1
            for start, end, result in ready:
                self._commit(start, end, result)
                with self._lock:
                    self._committed_end = max(self._committed_end, float(end))
            if ready:
                return len(ready)
            if deadline is not None and time.monotonic() >= deadline:
                return 0
            time.sleep(0.01)

    def drain(self, timeout: float | None = None) -> int:
        """Wait for every submitted window and commit them in order."""

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        committed = 0
        while True:
            if self.pending() == 0:
                return committed
            if deadline is not None and time.monotonic() >= deadline:
                logger.warning(
                    "%s drain timed out with %d windows pending",
                    self.name,
                    self.pending(),
                )
                return committed
            committed += self.poll(timeout=0.1)

    def advance(self, start: float) -> bool:
        """Move the next window start forward, skipping media.

        Used by the gap-filling pass: a range that was already processed must
        be dropped from memory instead of being sent to the model again.
        Refuses while any window is in flight, so commit order stays intact.
        """

        with self._lock:
            if self._closed or self._inflight or self._finished:
                return False
            new_start = max(self._next_start, float(start))
            self._next_start = new_start
            self._submitted_end = max(self._submitted_end, new_start)
            self._committed_end = max(self._committed_end, new_start)
            return True

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True)


__all__ = ["OrderedWindowPool", "WindowError"]
