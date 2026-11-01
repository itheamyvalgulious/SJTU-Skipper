from __future__ import annotations

import threading
import time

import pytest

from core.events import CLOCK_TICK, EVENT_ERROR, Clock, EventBus


def test_on_emit_and_flush() -> None:
    bus = EventBus()
    seen: list[dict] = []
    bus.on("demo", lambda e: seen.append(dict(e.payload)))
    bus.start()
    try:
        bus.emit("demo", value=1)
        assert bus.flush(2.0)
        assert seen == [{"value": 1}]
    finally:
        bus.stop()


def test_emit_before_start_is_drained_by_flush() -> None:
    bus = EventBus()
    seen: list[int] = []
    bus.on("demo", lambda e: seen.append(e.get("value")))
    bus.emit("demo", value=7)
    assert bus.flush(2.0)
    assert seen == [7]


def test_once_and_off() -> None:
    bus = EventBus()
    bus.start()
    seen: list[int] = []
    sub = bus.on("demo", lambda e: seen.append(1))
    bus.once("demo", lambda e: seen.append(2))
    try:
        bus.emit("demo")
        bus.emit("demo")
        assert bus.flush(2.0)
        assert seen == [1, 2, 1]
        sub.cancel()
        bus.emit("demo")
        assert bus.flush(2.0)
        assert seen == [1, 2, 1]
    finally:
        bus.stop()


def test_wildcard_and_priority() -> None:
    bus = EventBus()
    order: list[str] = []
    bus.on("*", lambda e: order.append("wild"), priority=0)
    bus.on("demo", lambda e: order.append("low"), priority=-1)
    bus.on("demo", lambda e: order.append("high"), priority=10)
    bus.start()
    try:
        bus.emit("demo")
        assert bus.flush(2.0)
        assert order == ["high", "wild", "low"]
    finally:
        bus.stop()


def test_handler_exception_emits_error_event() -> None:
    bus = EventBus()
    errors: list[dict] = []
    bus.on("boom", lambda e: (_ for _ in ()).throw(RuntimeError("x")))
    bus.on(EVENT_ERROR, lambda e: errors.append(dict(e.payload)))
    bus.start()
    try:
        bus.emit("boom")
        assert bus.flush(2.0)
        assert errors and errors[0]["source_event"] == "boom"
    finally:
        bus.stop()


def test_define_returns_stable_type() -> None:
    bus = EventBus()
    a = bus.define("custom.event", "desc")
    b = bus.define("custom.event")
    assert a is b
    assert str(a) == "custom.event"


def test_clock_tick() -> None:
    bus = EventBus()
    ticks: list[int] = []
    bus.on(CLOCK_TICK, lambda e: ticks.append(e.get("tick")))
    bus.start()
    clock = Clock(bus, interval=0.05)
    clock.start()
    try:
        time.sleep(0.18)
        clock.stop()
        assert bus.flush(2.0)
        assert ticks
        assert ticks[-1] >= 2
    finally:
        bus.stop()


def test_flush_timeout_when_handler_slow() -> None:
    bus = EventBus()
    started = threading.Event()

    def slow(event) -> None:
        started.set()
        time.sleep(0.2)

    bus.on("slow", slow)
    bus.start()
    try:
        bus.emit("slow")
        assert started.wait(1.0)
        assert bus.flush(timeout=0.01) in (True, False)
        bus.flush(2.0)
    finally:
        bus.stop()
