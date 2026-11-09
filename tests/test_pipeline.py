"""Tests for concurrent window processing and stream backlog gating."""

from __future__ import annotations

import threading
import time

from core.pipeline import OrderedWindowPool, WindowError
from core.streams import BacklogGate


# ---------------------------------------------------------------------------
# BacklogGate
# ---------------------------------------------------------------------------

class TestBacklogGate:
    def test_disabled_gate_returns_immediately(self) -> None:
        gate = BacklogGate("audio", limit=0.0)
        assert gate.wait(10_000.0) is True

    def test_blocks_until_released(self) -> None:
        gate = BacklogGate("audio", limit=5.0)
        gate.configure(5.0)
        assert gate.wait(5.0) is True
        result: list[bool] = []
        worker = threading.Thread(target=lambda: result.append(gate.wait(10.0)))
        worker.start()
        time.sleep(0.2)
        assert result == []
        gate.release(10.0)
        worker.join(timeout=2.0)
        assert result == [True]

    def test_close_unblocks_waiter(self) -> None:
        gate = BacklogGate("video", limit=1.0)
        gate.configure(1.0)
        result: list[bool] = []
        worker = threading.Thread(target=lambda: result.append(gate.wait(60.0)))
        worker.start()
        time.sleep(0.2)
        gate.close()
        worker.join(timeout=2.0)
        assert result == [False]

    def test_configure_resets_released(self) -> None:
        gate = BacklogGate("audio", limit=5.0)
        gate.configure(5.0)
        gate.release(100.0)
        gate.configure(5.0)
        assert gate.released == 0.0


# ---------------------------------------------------------------------------
# OrderedWindowPool
# ---------------------------------------------------------------------------

def submit_all(pool: OrderedWindowPool, windows: list[tuple[float, float]]) -> None:
    """Submit every window, polling while the pool is saturated."""

    for start, end in windows:
        while not pool.submit(start, end):
            pool.poll(0.01)


def _delayed_process(delays: dict[float, float]):
    def process(start: float, end: float):
        time.sleep(delays.get(start, 0.0))
        return f"{start:.0f}-{end:.0f}"

    return process


class TestOrderedWindowPool:
    def test_commits_in_window_order_despite_completion_order(self) -> None:
        commits: list[str] = []

        def process(start: float, end: float) -> str:
            # later windows finish first
            time.sleep(0.05 if start == 0 else 0.0)
            return f"{start:.0f}-{end:.0f}"

        pool = OrderedWindowPool(
            name="test",
            window=1.0,
            concurrency=3,
            process=process,
            commit=lambda start, end, result: commits.append(result),
        )
        try:
            submit_all(pool, [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])
            pool.drain(timeout=5.0)
        finally:
            pool.close()

        assert commits == ["0-1", "1-2", "2-3"]
        assert pool.committed_end == 3.0

    def test_respects_concurrency_limit(self) -> None:
        active = 0
        peak = 0
        lock = threading.Lock()

        def process(start: float, end: float) -> str:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return "ok"

        pool = OrderedWindowPool(
            name="test",
            window=1.0,
            concurrency=2,
            process=process,
            commit=lambda *_: None,
        )
        try:
            submit_all(pool, [(float(i), float(i + 1)) for i in range(6)])
            pool.drain(timeout=5.0)
        finally:
            pool.close()
        assert peak <= 2

    def test_errors_are_wrapped_and_reported(self) -> None:
        results: list[object] = []
        errors: list[Exception] = []

        def process(start: float, end: float) -> str:
            if start == 1.0:
                raise RuntimeError("boom")
            return "ok"

        def commit(start: float, end: float, result: object) -> None:
            results.append(result)
            if isinstance(result, WindowError):
                errors.append(result.error)

        pool = OrderedWindowPool(
            name="test",
            window=1.0,
            concurrency=2,
            process=process,
            commit=commit,
        )
        try:
            submit_all(pool, [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])
            pool.drain(timeout=5.0)
        finally:
            pool.close()

        assert results[0] == "ok"
        assert isinstance(results[1], WindowError)
        assert results[2] == "ok"
        assert len(errors) == 1 and "boom" in str(errors[0])

    def test_can_start_from_a_cursor(self) -> None:
        commits: list[float] = []
        pool = OrderedWindowPool(
            name="test",
            window=1.0,
            concurrency=1,
            process=lambda start, end: start,
            commit=lambda start, end, result: commits.append(start),
            start=30.0,
        )
        try:
            pool.submit(30.0, 31.0)
            pool.drain(timeout=5.0)
        finally:
            pool.close()
        assert commits == [30.0]

    def test_saturated_pool_refuses_submissions(self) -> None:
        release = threading.Event()

        def process(start: float, end: float) -> str:
            release.wait(timeout=5.0)
            return "ok"

        pool = OrderedWindowPool(
            name="test",
            window=1.0,
            concurrency=1,
            process=process,
            commit=lambda *_: None,
        )
        try:
            assert pool.submit(0.0, 1.0) is True
            assert pool.submit(1.0, 2.0) is False
            release.set()
            pool.drain(timeout=5.0)
            assert pool.submit(2.0, 3.0) is True
        finally:
            pool.close()
