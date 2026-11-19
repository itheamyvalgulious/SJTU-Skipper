from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

import httpx2
import pytest

from core.events import EventBus
from core.media import MediaKind, ProcessingMode
from core.meta import MetaRegistry
from core.plugin import PluginContext
from core.services import ServiceRegistry
from core.streams import StreamHub, StreamKind
from core.ui import UIManager
from plugins.stt import STTPlugin
from plugins.stt.client import STTClient, SttEventType
from plugins.stt.errors import SttError
from plugins.stt.plugin import PAGE_ID
from tests.support import FakeSettings


def make_ctx(tmp_path: Path, values: dict[str, Any] | None = None) -> PluginContext:
    bus = EventBus()
    bus.start()
    services = ServiceRegistry()
    services.register(
        "settings",
        FakeSettings({"stt.enabled": False, **(values or {})}),
        plugin_id="test",
    )
    return PluginContext(
        plugin_id="stt",
        data_dir=tmp_path,
        events=bus,
        streams=StreamHub(emit=bus.emit, audio_emit_interval=100),
        llm=None,
        ui=UIManager(),
        services=services,
        meta=MetaRegistry(),
        logger=logging.getLogger("test_stt"),
    )


@pytest.fixture
def app(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    plugin = STTPlugin()
    plugin.setup(ctx)
    try:
        yield plugin, ctx
    finally:
        plugin.teardown()
        ctx.events.stop()


def wait_until(predicate: Any) -> None:
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, "The consumer did not finish."
        time.sleep(0.01)


class Stream:
    def __init__(self, *, fail: bool = False) -> None:
        self.events: queue.Queue[dict[str, Any] | BaseException] = queue.Queue()
        self.offset = 0
        self.samples = 0
        self.frames: list[bytes] = []
        self.fail = fail
        self.metadata = {
            "type": SttEventType.READY,
            "model": "test-model",
            "language": "Chinese",
        }
        self.finished = False

    def reset(self, offset_ms: int) -> None:
        self.offset = offset_ms

    def send(self, pcm: bytes) -> None:
        self.frames.append(pcm)
        self.samples += len(pcm) // 2
        end = self.offset + self.samples // 16
        self.events.put({"type": SttEventType.ACCEPTED, "received_ms": end})
        self.events.put(
            {
                "type": SttEventType.PARTIAL,
                "segment_id": 0,
                "revision": len(self.frames),
                "start_ms": self.offset,
                "end_ms": end,
                "text": "临时",
            }
        )
        if self.fail:
            self.events.put(SttError("Connection lost.", connection_failed=True))
            self.fail = False

    def receive(self, timeout: float | None = None) -> dict[str, Any]:
        try:
            event = self.events.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError from None
        if isinstance(event, BaseException):
            raise event
        return event

    def finish(self) -> None:
        self.finished = True
        end = self.offset + self.samples // 16
        self.events.put(
            {
                "type": SttEventType.ANOMALY,
                "segment_id": 0,
                "start_ms": self.offset,
                "end_ms": end,
                "code": "repetition_collapsed",
            }
        )
        self.events.put(
            {
                "type": SttEventType.FINAL,
                "segment_id": 0,
                "start_ms": self.offset,
                "end_ms": end,
                "text": "定稿",
            }
        )
        self.events.put({"type": SttEventType.PROGRESS, "processed_ms": end})
        self.events.put({"type": SttEventType.DONE, "processed_ms": end})

    def close(self) -> None:
        pass


class Client:
    supports_stream = True
    max_window = 120.0

    def __init__(
        self,
        streams: list[Stream] | None = None,
        *,
        language: str = "Chinese",
        context: str = "",
    ) -> None:
        self.streams: list[Stream] = []
        self.prepared = streams or []
        self.calls: list[dict[str, Any]] = []
        self._language = language
        self._context = context

    def metadata(self, context: str = "") -> dict[str, str]:
        return {"language": self._language, "context": context}

    def stream(self, **kwargs: Any) -> Stream:
        self.calls.append(kwargs)
        stream = self.prepared.pop(0) if self.prepared else Stream()
        self.streams.append(stream)
        return stream

    def close_stream(self, stream: Stream) -> None:
        stream.close()

    def close(self) -> None:
        pass


def test_settings_service_and_debug_page(tmp_path: Path) -> None:
    ctx = make_ctx(
        tmp_path,
        {
            "stt.credentials_json": "secret",
            "stt.chunk_seconds": 20,
            "stt.context": "手动提示",
            "stt.timeout": 7,
            "stt.concurrency": 3,
        },
    )
    plugin = STTPlugin()
    plugin.setup(ctx)
    try:
        settings = ctx.service("settings")
        assert settings.get("stt.provider") == "compatible"
        assert "stt.credentials_json" not in settings.values
        assert "stt.chunk_seconds" not in settings.values
        assert "stt.context" not in settings.values
        assert settings.get("stt.timeout") == 7
        assert settings.get("stt.concurrency") == 3
        assert settings.get("stt.window_seconds") == 20
        assert settings.get("stt.google.credentials_json") == ""
        assert PAGE_ID in ctx.ui.snapshot()["pages"]
        assert "stt.debug" in ctx.ui.snapshot()["commands"]
        assert ctx.service("stt").flushing() is False
    finally:
        plugin.teardown()
        ctx.events.stop()


def test_http_metadata_precedes_binary_audio() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(
            200,
            json={
                "duration_ms": 1,
                "segments": [],
                "model": "m",
                "language": "English",
            },
        )

    client = STTClient("http://test", transport=httpx2.MockTransport(handler))
    try:
        client.transcribe(b"RIFF\x00\xff", language="English", context="calculus")
    finally:
        client.close()
    body = seen[0].content
    assert body.index(b'name="metadata"') < body.index(b'name="audio"')
    assert b"RIFF\x00\xff" in body
    assert b'"context": "calculus"' in body
    assert b'"language": "English"' in body


@pytest.mark.parametrize(
    "status,attempts",
    [(400, 1), (403, 1), (500, 1), (429, 3), (502, 3), (503, 3), (504, 3)],
)
def test_bounded_structured_retry(status: int, attempts: int) -> None:
    calls = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request.content)
        return httpx2.Response(status, json={"error": {"message": "Rejected."}})

    client = STTClient("http://test", transport=httpx2.MockTransport(handler))
    try:
        with pytest.raises(SttError) as error:
            client.transcribe(b"wav", language="Chinese", context="")
        assert error.value.status_code == status
        assert len(calls) == attempts
        assert all(b"wav" in body for body in calls)
    finally:
        client.close()


@pytest.mark.parametrize(
    "mode,offline",
    [
        (ProcessingMode.LIVE, False),
        (ProcessingMode.REALTIME, False),
        (ProcessingMode.FAST, True),
    ],
)
def test_continuous_stream_and_subsecond_tail(
    app: Any, monkeypatch: pytest.MonkeyPatch, mode: ProcessingMode, offline: bool
) -> None:
    plugin, ctx = app
    stream = Stream()
    stream.metadata["max_audio_ms"] = 960000
    client = Client([stream])
    monkeypatch.setattr(plugin, "_new_client", lambda: client)
    session = ctx.streams.current
    session.meta["mode"] = mode
    session.meta["course_name"] = "微积分"
    session.audio.append(b"\x01\x00" * 7200)
    session.input_done = True
    ctx.service("settings").set_many({"stt.language": "Chinese", "stt.enabled": True})
    wait_until(lambda: len(session.transcript) == 1 and not plugin.busy())
    assert client.calls[0]["offline"] is offline
    assert client.calls[0]["context"] == "这是一堂微积分课程。"
    assert len(client.streams) == 1
    assert all(len(frame) <= 6400 for frame in client.streams[0].frames)
    assert b"".join(client.streams[0].frames) == b"\x01\x00" * 7200
    assert client.streams[0].finished
    assert plugin._gate_limit() >= 1930
    assert plugin.status()["cursor"] == pytest.approx(0.45)
    assert plugin.status()["partial"] is None
    assert plugin.status()["anomalies"][0]["code"] == "repetition_collapsed"
    log = ctx.data_dir / "sessions" / session.session_id / "transcript.jsonl"
    assert [json.loads(line)["text"] for line in log.read_text().splitlines()] == [
        "定稿"
    ]
    assert "confidence" not in session.transcript[0].meta


def test_stream_window_releases_producer_gate(app, monkeypatch):
    plugin, ctx = app
    stream = Stream()
    stream.metadata["max_audio_ms"] = 1000
    monkeypatch.setattr(plugin, "_new_client", lambda: Client([stream]))
    session = ctx.streams.current
    session.meta["mode"] = ProcessingMode.FAST
    session.audio_gate.configure(0.2, fail_closed=True)
    ctx.service("settings").set_many({"stt.buffer_seconds": 0.2, "stt.enabled": True})

    def produce():
        for _ in range(10):
            if not session.audio_gate.wait(session.audio.duration + 0.2):
                return
            session.audio.append(b"\x01\x00" * 3200)
        session.audio_input_done = True

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    try:
        wait_until(lambda: session.audio_input_done and len(session.transcript) == 1)
        assert session.audio_gate.limit >= 12
        assert plugin.status()["cursor"] == pytest.approx(2.0)
        assert b"".join(stream.frames) == b"\x01\x00" * 32000
    finally:
        session.audio_gate.close()
        producer.join(timeout=1)


def test_accepted_and_partial_do_not_commit_or_trim(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin, ctx = app
    session = ctx.streams.current
    session.audio.append(b"\x00\x00" * 64000)
    monkeypatch.setattr(
        plugin, "_cancelled", lambda current: current is not ctx.streams.current
    )
    connection = {"sent": 4.0, "received": 0.0, "processed": 0.0}
    plugin._live_event(
        session, {"type": SttEventType.ACCEPTED, "received_ms": 4000}, 0, connection
    )
    plugin._live_event(
        session,
        {
            "type": SttEventType.PARTIAL,
            "segment_id": 1,
            "revision": 1,
            "text": "临时",
            "start_ms": 0,
            "end_ms": 4000,
        },
        0,
        connection,
    )
    assert plugin._coverage(session).covered_seconds(MediaKind.AUDIO) == 0
    assert session.audio.earliest_time == 0
    assert session.audio_gate.released == 0
    assert len(session.transcript) == 0
    assert plugin.status()["cursor"] == 0
    assert any("临时" in b.get("text", "") for b in plugin._render_page())
    ctx.service("settings").set("stt.enabled", False)


def test_exact_coverage_gaps_and_discarded_audio(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin, ctx = app
    client = Client()
    monkeypatch.setattr(plugin, "_new_client", lambda: client)
    session = ctx.streams.current
    session.meta["mode"] = ProcessingMode.FAST
    session.audio.append(b"\x01\x00" * 16000)
    session.audio.discard(1)
    session.audio.append(b"\x02\x00" * 16000)
    plugin._coverage(session).mark(MediaKind.AUDIO, 0, 0.25)
    session.meta["skip_ranges"] = {"audio": [[0.5, 0.75]]}
    session.input_done = True
    ctx.service("settings").set("stt.enabled", True)
    wait_until(lambda: not plugin.busy())
    assert [s.offset for s in client.streams] == [250, 750, 2000]
    assert plugin._coverage(session).coverage(MediaKind.AUDIO).intervals == [
        (0.0, 0.5),
        (0.75, 1.0),
        (2.0, 3.0),
    ]
    assert plugin.status()["cursor"] == 1.0
    assert plugin.status()["uncovered_seconds"] == 1.0
    assert session.audio_gate.released == 3


def test_connection_failure_replays_only_uncommitted_audio(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin, ctx = app
    client = Client([Stream(fail=True), Stream()])
    monkeypatch.setattr(plugin, "_new_client", lambda: client)
    session = ctx.streams.current
    session.audio.append(b"\x01\x00" * 8000)
    session.input_done = True
    ctx.service("settings").set("stt.enabled", True)
    wait_until(lambda: len(session.transcript) == 1 and not plugin.busy())
    assert len(client.streams) == 2
    assert client.streams[1].offset == 0
    assert plugin.status()["cursor"] == 0.5
    assert plugin.status()["failed_ranges"] == []


def test_old_session_event_does_not_write_new_session(app: Any) -> None:
    plugin, ctx = app
    old = ctx.streams.current
    old.audio.append(b"\x00\x00" * 16000)
    ctx.streams.start_session()
    ctx.events.flush(timeout=1)
    plugin._cancel.clear()
    plugin._live_event(
        old,
        {
            "type": SttEventType.FINAL,
            "segment_id": 0,
            "start_ms": 0,
            "end_ms": 1000,
            "text": "旧",
        },
        0,
        {"sent": 1.0, "received": 1.0, "processed": 0.0},
    )
    assert len(ctx.streams.current.transcript) == 0
    assert len(old.transcript) == 0


def test_failed_range_is_not_successful_coverage(app: Any) -> None:
    plugin, ctx = app
    session = ctx.streams.current
    plugin._fail(session, 1, 2, SttError("Rejected.", status_code=400))
    assert plugin.status()["failed_ranges"] == [
        {"start": 1, "end": 2, "error": "Rejected."}
    ]
    assert plugin.status()["cursor"] == 0
    assert plugin._coverage(session).covered_seconds(MediaKind.AUDIO) == 0


def test_manual_http_preserves_log_and_metadata(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin, ctx = app
    seen = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request.content)
        return httpx2.Response(
            200,
            json={
                "duration_ms": 500,
                "model": "model",
                "language": "Chinese",
                "segments": [
                    {
                        "start_ms": 10,
                        "end_ms": 490,
                        "text": "定稿",
                        "anomaly": "repetition_collapsed",
                    }
                ],
            },
        )

    monkeypatch.setattr(
        plugin,
        "_new_client",
        lambda: STTClient("http://test", transport=httpx2.MockTransport(handler)),
    )
    session = ctx.streams.current
    session.meta["course_name"] = "复分析"
    session.audio.append(b"\x00\x00" * 8000)
    result = ctx.service("stt").transcribe(0, 0.5)
    assert result["segments"] == 1
    item = session.transcript[0]
    assert item.start == 0.01 and item.end == 0.49
    assert dict(item.meta) == {
        "model": "model",
        "language": "Chinese",
        "anomaly": "repetition_collapsed",
    }
    assert plugin.status()["cursor"] == 0.5
    assert b"RIFF" in seen[0]
    assert json.dumps("这是一堂复分析课程。").encode() in seen[0]


@pytest.mark.parametrize("name", [None, "", " \n ", 93330, "课" * 81])
def test_no_topic(app: Any, name: Any) -> None:
    plugin, ctx = app
    session = ctx.streams.current
    session.meta["course_name"] = name
    assert plugin._context_hint(session) == ""


def test_session_topic(app: Any) -> None:
    plugin, ctx = app
    old = ctx.streams.current
    old.meta["course_name"] = " 数学分析\n（荣誉）III "
    ctx.streams.start_session(meta={"course_name": "复分析"})
    assert plugin._context_hint(old) == "这是一堂数学分析 （荣誉）III课程。"
    assert plugin._context_hint(ctx.streams.current) == "这是一堂复分析课程。"


def test_service_reports_tail_as_busy_and_flushing(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin, ctx = app
    session = ctx.streams.current
    state = plugin._state(session)
    monkeypatch.setattr(plugin, "_new_client", lambda: Client())
    ctx.service("settings").set("stt.enabled", True)
    session.audio.append(b"\x00\x00" * 100)
    state.flushing = True
    assert ctx.service("stt").busy()
    assert ctx.service("stt").flushing()
    ctx.service("settings").set("stt.enabled", False)


@pytest.mark.parametrize("offline", [False, True])
def test_stream_mode_is_explicit_in_query_and_start(
    monkeypatch: pytest.MonkeyPatch, offline: bool
) -> None:
    sent = []
    urls = []

    class Socket:
        def send(self, message: str) -> None:
            sent.append(json.loads(message))

        def recv(self, timeout: float | None = None) -> str:
            return json.dumps({"type": SttEventType.READY})

        def close(self) -> None:
            pass

    def connect(url: str, **kwargs: Any) -> Socket:
        urls.append(url)
        return Socket()

    monkeypatch.setattr("plugins.stt.client.connect", connect)
    client = STTClient("http://test")
    try:
        client.stream(language="Chinese", context="", concurrency=2, offline=offline)
        assert urls == [
            "ws://test/v1/stream?mode=" + ("offline" if offline else "live")
        ]
        assert sent[0]["mode"] == ("offline" if offline else "live")
    finally:
        client.close()


def test_live_eof_rounding_does_not_reopen_stream(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin, ctx = app
    client = Client()
    monkeypatch.setattr(plugin, "_new_client", lambda: client)
    session = ctx.streams.current
    session.meta["mode"] = ProcessingMode.REALTIME
    for _ in range(80):
        session.audio.append(b"\x01\x00" * 3200)
    session.input_done = True
    ctx.service("settings").set("stt.enabled", True)
    wait_until(lambda: len(session.transcript) == 1 and not plugin.busy())
    assert len(client.streams) == 1
    assert plugin.flushing() is False
    assert plugin.status()["cursor"] == pytest.approx(16.0)
    assert plugin.status()["scheduled_cursor"] == session.audio.duration


def test_only_processed_audio_seals(app, monkeypatch):
    plugin, ctx = app
    session = ctx.streams.current
    monkeypatch.setattr(plugin, "_cancelled", lambda session: False)
    connection = {"sent": 10.0, "received": 0.0, "processed": 0.0}
    plugin._live_event(
        session, {"type": SttEventType.ACCEPTED, "received_ms": 10000}, 0, connection
    )
    plugin._live_event(
        session, {"type": SttEventType.PARTIAL, "text": "draft"}, 0, connection
    )
    assert (
        ctx.streams.text_progress(session.session_id, StreamKind.TRANSCRIPT).before == 0
    )
    plugin._live_event(
        session, {"type": SttEventType.PROGRESS, "processed_ms": 10000}, 0, connection
    )
    assert (
        ctx.streams.text_progress(session.session_id, StreamKind.TRANSCRIPT).before
        == 10
    )


def test_stop_callback_does_not_wait_for_tail(app):
    import threading

    plugin, ctx = app
    session = ctx.streams.current
    state = plugin._state(session)
    state.active = True
    state.done.clear()
    returned = threading.Event()

    def stop():
        plugin._on_session_stopped({"session_id": session.session_id})
        returned.set()

    thread = threading.Thread(target=stop)
    thread.start()
    try:
        assert returned.wait(1)
        assert not state.done.is_set()
    finally:
        state.active = False
        state.done.set()
        thread.join()
