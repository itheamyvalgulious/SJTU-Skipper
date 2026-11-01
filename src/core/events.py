"""Append-only stream event bus and wall-clock ticks.

The bus dispatches events on one dedicated thread, preserving order.  Producers
(ffmpeg readers, workers, HTTP requests) only enqueue and never block on
handlers.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

logger = logging.getLogger("lecture_agent.events")

CLOCK_TICK = "clock.tick"
SESSION_STARTED = "session.started"
SESSION_STOPPED = "session.stopped"
STREAM_APPENDED = "stream.appended"
EVENT_ERROR = "event.error"


@dataclass(frozen=True, slots=True)
class EventType:
    """A named event type plugins may define for documentation/validation."""

    name: str
    description: str = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


@dataclass(frozen=True, slots=True)
class Event:
    """An immutable event delivered to handlers."""

    type: str
    payload: Mapping[str, Any]
    ts: float
    event_id: int

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


Handler = Callable[[Event], None]


@dataclass(slots=True)
class Subscription:
    """Handle returned by :meth:`EventBus.on`."""

    bus: "EventBus"
    type: str
    handler: Handler
    plugin_id: str = ""
    priority: int = 0
    once: bool = False
    seq: int = 0
    active: bool = True

    def cancel(self) -> None:
        self.bus.off(self)


class EventBus:
    """Thread-safe, ordered event dispatcher."""

    def __init__(self, *, name: str = "events") -> None:
        self.name = name
        self._types: dict[str, EventType] = {}
        self._handlers: dict[str, list[Subscription]] = {}
        self._any: list[Subscription] = []
        self._lock = threading.RLock()
        self._queue: queue.Queue[Event | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._pending = 0
        self._pending_cv = threading.Condition(self._lock)
        self._seq = 0
        self._event_id = 0
        self._logger = logger

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def define(self, name: str, description: str = "") -> EventType:
        if not name or not isinstance(name, str):
            raise ValueError("event type name must be a non-empty string")
        with self._lock:
            existing = self._types.get(name)
            if existing is not None:
                return existing
            event_type = EventType(name=name, description=description)
            self._types[name] = event_type
            return event_type

    #: Alias matching the "create event" wording in the plugin API.
    create = define

    def on(
        self,
        event_type: str | EventType,
        handler: Handler,
        *,
        plugin_id: str = "",
        priority: int = 0,
        once: bool = False,
    ) -> Subscription:
        if not callable(handler):
            raise TypeError("handler must be callable")
        name = str(event_type)
        with self._lock:
            self.define(name)
            self._seq += 1
            sub = Subscription(
                bus=self,
                type=name,
                handler=handler,
                plugin_id=plugin_id,
                priority=priority,
                once=once,
                seq=self._seq,
            )
            if name == "*":
                self._any.append(sub)
                self._any.sort(key=lambda s: (-s.priority, s.seq))
            else:
                self._handlers.setdefault(name, []).append(sub)
                self._handlers[name].sort(key=lambda s: (-s.priority, s.seq))
            return sub

    def once(self, event_type: str | EventType, handler: Handler, **kwargs: Any) -> Subscription:
        kwargs["once"] = True
        return self.on(event_type, handler, **kwargs)

    def off(self, subscription: Subscription | None) -> None:
        if subscription is None:
            return
        with self._lock:
            subscription.active = False

    def remove_plugin(self, plugin_id: str) -> None:
        """Cancel every subscription owned by a plugin."""

        with self._lock:
            for subs in self._handlers.values():
                for sub in subs:
                    if sub.plugin_id == plugin_id:
                        sub.active = False
            for sub in self._any:
                if sub.plugin_id == plugin_id:
                    sub.active = False

    def handler_count(self, event_type: str | None = None) -> int:
        with self._lock:
            if event_type is None:
                return sum(len(s) for s in self._handlers.values()) + len(self._any)
            return len(self._handlers.get(str(event_type), ())) + len(self._any)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------
    def emit(self, event_type: str | EventType, **payload: Any) -> Event:
        name = str(event_type)
        with self._lock:
            self._event_id += 1
            event = Event(
                type=name,
                payload=MappingProxyType(dict(payload)),
                ts=time.time(),
                event_id=self._event_id,
            )
            self._pending += 1
            if not self._running:
                # Not started yet: keep the event queued for start(), but make
                # synchronous flush possible for tests.
                self._queue.put(event)
                return event
            self._queue.put(event)
            return event

    def emit_event(self, event: Event) -> None:
        with self._lock:
            self._pending += 1
            self._queue.put(event)

    def flush(self, timeout: float | None = 10.0) -> bool:
        """Wait until every emitted event has been dispatched."""

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._pending_cv:
            while self._pending > 0:
                if not self._running:
                    # Drain synchronously when the bus was never started.
                    self._pending_cv.release()
                    try:
                        self._drain_sync()
                    finally:
                        self._pending_cv.acquire()
                    continue
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._pending_cv.wait(0.05 if remaining is None else min(0.05, remaining))
            return True

    def _drain_sync(self) -> None:
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                return
            if event is None:
                continue
            self._dispatch(event)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._dispatch_loop, name=f"{self.name}-dispatcher", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._queue.put(None)
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
        self.flush(timeout=0.5)

    @property
    def running(self) -> bool:
        return self._running

    def _dispatch_loop(self) -> None:
        while True:
            event = self._queue.get()
            if event is None:
                break
            self._dispatch(event)
        # Drain anything left behind after the sentinel.
        self._drain_sync()

    def _dispatch(self, event: Event) -> None:
        try:
            with self._lock:
                specific = list(self._handlers.get(event.type, ()))
                wildcard = list(self._any)
            merged = sorted(specific + wildcard, key=lambda s: (-s.priority, s.seq))
            seen: set[int] = set()
            for sub in merged:
                if id(sub) in seen:
                    continue
                seen.add(id(sub))
                if not sub.active:
                    continue
                if sub.once:
                    sub.active = False
                try:
                    sub.handler(event)
                except Exception:  # noqa: BLE001 - handlers are untrusted
                    self._logger.exception(
                        "event handler failed: type=%s plugin=%s",
                        event.type,
                        sub.plugin_id or "<anonymous>",
                    )
                    if event.type != EVENT_ERROR:
                        self.emit(
                            EVENT_ERROR,
                            source_event=event.type,
                            plugin_id=sub.plugin_id,
                            error="handler failed",
                            event_id=event.event_id,
                        )
        finally:
            with self._pending_cv:
                self._pending = max(0, self._pending - 1)
                self._pending_cv.notify_all()


class Clock(threading.Thread):
    """Emit ``clock.tick`` every ``interval`` seconds."""

    def __init__(
        self,
        bus: EventBus,
        interval: float = 1.0,
        *,
        name: str = "clock",
        autostart: bool = False,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.bus = bus
        self.interval = max(0.05, float(interval))
        self._stop_event = threading.Event()
        self._tick = 0
        self.bus.define(CLOCK_TICK, "wall-clock tick")
        if autostart:
            self.start()

    def run(self) -> None:  # pragma: no cover - timing loop
        while not self._stop_event.wait(self.interval):
            self.tick_once()

    def tick_once(self) -> None:
        self._tick += 1
        self.bus.emit(
            CLOCK_TICK,
            ts=time.time(),
            tick=self._tick,
            interval=self.interval,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=2.0)
