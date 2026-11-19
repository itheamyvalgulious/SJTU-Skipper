from __future__ import annotations

import io
import threading
import time
import wave

import pytest

from core.media import MediaKind
from core.streams import StreamHub, StreamKind, TextSource, TextStream


def test_request_navigation_is_independent_of_processing(navigation):
    hub = StreamHub()
    navigation.class_id = "viewed"
    navigation.class_meta = {"name": "Selected lecture"}
    hub.start_session("background")
    assert navigation.class_id == "viewed"
    assert navigation.class_meta["name"] == "Selected lecture"


def test_navigation_scopes_are_isolated(navigation):
    from core.navigation import Navigation, current_navigation, navigation_scope

    navigation.class_id = "first"
    navigation.pages.append("first-page")
    with navigation_scope(Navigation(class_id="second")) as other:
        other.pages.append("second-page")
        assert current_navigation().class_id == "second"
    assert current_navigation() is navigation
    assert navigation.pages == ["first-page"]
    observed = []
    worker = threading.Thread(
        target=lambda: observed.append(current_navigation().class_id)
    )
    worker.start()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert observed == [""]


def test_audio_append_read_and_len() -> None:
    hub = StreamHub()
    audio = hub.current.audio
    pcm = b"\x01\x02" * 16000  # 1 s @ 16k mono s16le
    audio.append(pcm)
    audio.append(pcm[:16000], duration=0.5)
    assert len(audio) == 1  # int seconds for the len() protocol
    assert audio.duration == pytest.approx(1.5)
    assert audio.read(0, 1) == pcm
    assert len(audio.read(0.25, 0.75)) == 16000
    assert len(audio.read(1.0, 2.0)) == 16000


def test_audio_prefix_is_immutable() -> None:
    hub = StreamHub()
    audio = hub.current.audio
    audio.append(b"\x00\x01" * 8000)  # 0.5 s
    before = audio.read(0, 0.5)
    audio.append(b"\x02\x03" * 8000)
    assert audio.read(0, 0.5) == before
    assert audio.read(0.5, 1.0) == b"\x02\x03" * 8000


def test_audio_read_as_wav() -> None:
    hub = StreamHub()
    audio = hub.current.audio
    pcm = b"\x01\x02" * audio.sample_rate
    audio.append(pcm)
    with wave.open(io.BytesIO(audio.read_as_wav(0.25, 0.75)), "rb") as wav:
        assert wav.getnchannels() == audio.channels
        assert wav.getsampwidth() == audio.sample_width
        assert wav.getframerate() == audio.sample_rate
        assert (
            wav.readframes(wav.getnframes()) == pcm[len(pcm) // 4 : 3 * len(pcm) // 4]
        )


def test_audio_wait_for() -> None:
    hub = StreamHub()
    audio = hub.current.audio

    def producer() -> None:
        time.sleep(0.05)
        audio.append(b"\x00\x00" * 16000)

    thread = threading.Thread(target=producer)
    thread.start()
    assert audio.wait_for(1.0, timeout=2.0)
    thread.join()
    assert audio.duration == pytest.approx(1.0)


def test_video_frames() -> None:
    hub = StreamHub(video_fps=1.0)
    video = hub.current.video
    video.append_frame(b"jpeg-0")
    video.append_frame(b"jpeg-1")
    assert len(video) == 2
    assert video.duration == pytest.approx(2.0)
    assert video.read(0, 2) == b"jpeg-0jpeg-1"
    frames = video.read_frames(0.5, 1.5)
    assert [f.data for f in frames] == [b"jpeg-0", b"jpeg-1"]
    assert [f.start for f in frames] == [0.0, 1.0]


def test_text_stream_is_append_only_sequence() -> None:
    hub = StreamHub()
    text = hub.current.transcript
    item0 = text.append("hello ", start=0, end=1, source=TextSource.STT)
    item1 = text.append("world", start=1, end=2, source=TextSource.STT)
    assert item0 is not None and item1 is not None
    assert len(text) == 2
    assert text[0].text == "hello "
    assert text[1].text == "world"
    assert text.text_since(0) == "hello world"
    assert text.text_since(1) == "world"
    assert text.char_count == 11
    assert text.char_count_since(1) == 5
    assert text.snapshot_index() == 2
    assert text.append("") is None
    assert list(text)[0] == item0


def test_text_metadata_is_immutable() -> None:
    stream = TextStream("s", StreamKind.TRANSCRIPT)
    item = stream.append("x", meta={"a": 1})
    assert item is not None
    with pytest.raises(TypeError):
        item.meta["a"] = 2  # type: ignore[index]


def test_stream_hub_events_are_coalesced() -> None:
    events: list[dict] = []
    hub = StreamHub(
        emit=lambda name, **payload: events.append({"name": name, **payload})
    )
    session = hub.start_session("math", meta={"course": "math"})
    events.clear()
    audio = session.audio
    for _ in range(5):
        audio.append(b"\x00\x00" * 1600)  # 0.1 s each
    new = [
        e
        for e in events
        if e["name"] == "stream.appended" and e["stream"] is StreamKind.AUDIO
    ]
    assert len(new) == 1
    assert new[0]["start"] == 0.0
    assert new[0]["end"] == pytest.approx(0.5)

    session.transcript.append("hi", start=0, end=1)
    text_events = [
        e
        for e in events
        if e["name"] == "stream.appended" and e["stream"] is StreamKind.TRANSCRIPT
    ]
    assert len(text_events) == 1
    assert text_events[0]["chars"] == 2


def test_session_switch_keeps_old_stream_readable() -> None:
    hub = StreamHub()
    first = hub.start_session("first", meta={"n": 1})
    first.audio.append(b"\x00\x00" * 16000)
    first.transcript.append("one")
    second = hub.start_session("second", meta={"n": 2})
    assert second.session_id != first.session_id
    assert second.audio.duration == 0
    assert first.audio.duration == pytest.approx(1.0)
    assert first.transcript[0].text == "one"
    assert len(hub.list_sessions()) >= 3


@pytest.mark.parametrize("with_content", [False, True])
def test_same_id_restart_replaces_stream(with_content) -> None:
    events = []
    resolved = []

    def emit(name, **payload):
        events.append((name, payload))
        if name in ("session.started", "session.stopped"):
            resolved.append((name, hub.get_session(payload["session_id"])))

    hub = StreamHub()
    hub._emit = emit
    first = hub.start_session("lecture", meta={"run": 1})
    if with_content:
        first.audio.append(b"\x00\x00" * 16000)
    hub.seal_text("lecture", StreamKind.TRANSCRIPT, 1, final=True)
    events.clear()
    resolved.clear()

    second = hub.start_session("lecture", meta={"run": 2})

    lifecycle = [(name, data) for name, data in events if name.startswith("session.")]
    assert [name for name, _ in lifecycle] == ["session.stopped", "session.started"]
    assert resolved == [("session.stopped", second), ("session.started", second)]
    assert lifecycle[0][1]["meta"] == {"run": 1}
    assert lifecycle[0][1]["started_at"] == first.started_at
    assert lifecycle[0][1]["ended_at"] == first.ended_at
    assert first.ended_at is not None
    assert not first.audio_gate.wait(0)
    assert not first.video_gate.wait(0)
    assert second is hub.current is hub.get_session("lecture")
    assert second is not first and second.audio.duration == 0
    assert [s.session_id for s in hub.list_sessions()].count("lecture") == 1
    assert not hub.text_progress("lecture", StreamKind.TRANSCRIPT).final
    events.clear()
    second.audio.append(b"\x00\x00" * 8000)
    audio_events = [data for name, data in events if name == "stream.appended"]
    assert audio_events[0]["old_len"] == 0


def test_restart_historical_id_replaces_lookup() -> None:
    hub = StreamHub()
    original = hub.start_session("first")
    other = hub.start_session("second")
    replacement = hub.start_session("first")
    assert hub.get_session("first") is replacement is not original
    assert hub.get_session("second") is other
    assert other.ended_at is not None
    assert [session.session_id for session in hub.list_sessions()] == [
        "boot",
        "second",
        "first",
    ]


def test_queued_stop_keeps_old_metadata() -> None:
    from core.events import EventBus

    bus = EventBus()
    events = []
    bus.on("session.stopped", events.append)
    hub = StreamHub(emit=bus.emit)
    first = hub.start_session("lecture", meta={"run": 1})
    second = hub.start_session("lecture", meta={"run": 2})
    assert bus.flush()
    assert len(events) == 1
    assert events[0].get("meta") == {"run": 1}
    assert events[0].get("started_at") == first.started_at
    assert events[0].get("ended_at") == first.ended_at
    assert hub.get_session(events[0].get("session_id")) is second


def test_session_callbacks_release_hub_lock() -> None:
    hub = StreamHub()
    callbacks = []

    def emit(name, **payload):
        if not name.startswith("session."):
            return
        completed = threading.Event()

        def read_session():
            hub.get_session(payload["session_id"])
            completed.set()

        reader = threading.Thread(target=read_session, daemon=True)
        reader.start()
        assert completed.wait(1), "session callback holds the hub lock"
        reader.join()
        callbacks.append(name)

    hub._emit = emit
    hub.start_session("lecture")
    hub.start_session("lecture")
    assert callbacks == ["session.started", "session.stopped", "session.started"]


@pytest.mark.parametrize(
    "session_id", ["", ".", "..", "../x", "/x", "x/y", "x\\y", "x\x00"]
)
def test_session_id_rejects_unsafe_path(session_id) -> None:
    hub = StreamHub()
    with pytest.raises(ValueError, match="safe path component"):
        hub.start_session(session_id)
    assert hub.current.session_id == "boot"


def test_session_id_is_required() -> None:
    with pytest.raises(TypeError):
        StreamHub().start_session()  # type: ignore[call-arg]


def test_stream_set_status() -> None:
    hub = StreamHub()
    session = hub.start_session("topology", meta={"course": "topology"})
    session.audio.append(b"\x00\x00" * 16000)
    session.transcript.append("abc")
    session.visual.append("defg")
    status = session.status()
    assert status["audio_seconds"] == pytest.approx(1.0)
    assert status["transcript_chars"] == 3
    assert status["visual_chars"] == 4
    assert status["meta"]["course"] == "topology"


class TestTrim:
    def test_audio_trim_drops_consumed_chunks(self) -> None:
        from core.streams import AudioStream

        stream = AudioStream("s1", sample_rate=16000)
        for _ in range(4):
            stream.append(b"\x00\x00" * 16000)  # 1 s each
        assert stream.duration == pytest.approx(4.0)

        dropped = stream.trim(2.0)
        assert dropped == 2
        assert stream.duration == pytest.approx(4.0)
        assert stream.kept_seconds == pytest.approx(2.0)
        assert len(stream.read(2.0, 4.0)) == 64000
        assert stream.read(0.0, 1.0) == b""

    def test_video_trim_keeps_timeline(self) -> None:
        from core.streams import VideoStream

        stream = VideoStream("s1", fps=1.0)
        for _ in range(5):
            stream.append_frame(b"jpg", duration=1.0)
        assert stream.duration == pytest.approx(5.0)
        assert stream.trim(3.0) == 3
        assert stream.duration == pytest.approx(5.0)
        assert stream.kept_seconds == pytest.approx(2.0)
        assert [f.start for f in stream.read_frames(3.0, 5.0)] == [3.0, 4.0]


def test_discard_advances_timeline_without_pcm() -> None:
    from core.streams import AudioStream

    notifications = []
    audio = AudioStream("s", notify=lambda **event: notifications.append(event))
    audio.discard(2.0)
    assert audio.wait_for(2.0, timeout=0)
    assert audio.duration == 2.0
    assert audio.chunk_count == 0
    assert audio.read(0, 2) == b""
    assert audio.earliest_time == 2.0
    audio.append(b"\x01\x02" * 1600)
    audio.discard(1.0)
    audio.append(b"\x03\x04" * 1600)
    assert audio.earliest_time == 2.0
    assert audio.buffered_ranges() == [(2.0, 2.1), (3.1, 3.2)]
    assert audio.read(0, 2) == b""
    assert notifications[0]["new_count"] == 0
    assert notifications[0]["new_len"] == 2.0
    audio.trim(2.1)
    assert audio.earliest_time == pytest.approx(3.1)


def test_audio_discard_notifies_waiting_consumer() -> None:
    audio = StreamHub().current.audio
    ready = threading.Event()
    result = []

    def consume() -> None:
        ready.set()
        result.append(audio.wait_for(1.0, timeout=1.0))

    consumer = threading.Thread(target=consume)
    consumer.start()
    ready.wait()
    audio.discard(1.0)
    consumer.join(timeout=2.0)
    assert result == [True]
    assert audio.chunk_count == 0


def test_gate_continuation_preserves_release_and_audio_timeout_stops(
    monkeypatch,
) -> None:
    from core.streams import BacklogGate

    monkeypatch.setattr("core.streams.time.monotonic", lambda: 0.0)
    gate = BacklogGate(MediaKind.AUDIO, stall_timeout=1.0)
    gate.configure(10.0, released=120.0, fail_closed=True)
    assert gate.wait(130.0)
    monkeypatch.setattr("core.streams.time.monotonic", lambda: 2.0)
    assert not gate.wait(130.2)
    assert gate.limit == 10.0
    assert gate.released == 120.0


def test_video_gate_keeps_default_timeout_behavior(monkeypatch) -> None:
    from core.streams import BacklogGate

    monkeypatch.setattr("core.streams.time.monotonic", lambda: 0.0)
    gate = BacklogGate(MediaKind.VIDEO, stall_timeout=1.0)
    gate.configure(10.0, released=120.0)
    monkeypatch.setattr("core.streams.time.monotonic", lambda: 2.0)
    assert gate.wait(131.0)
    assert gate.limit == 0.0
    assert gate.released == 120.0


def test_text_progress_is_session_scoped() -> None:
    events = []
    hub = StreamHub(emit=lambda name, **data: events.append((name, data)))
    first = hub.start_session("first")
    hub.seal_text(
        first.session_id, StreamKind.TRANSCRIPT, 600, error={"start": 10, "end": 20}
    )
    second = hub.start_session("second")
    hub.seal_text(first.session_id, StreamKind.TRANSCRIPT, 100, final=True)
    snapshot = hub.text_progress(first.session_id, StreamKind.TRANSCRIPT)
    assert snapshot.before == 600 and snapshot.final
    snapshot.errors[0]["start"] = 99
    assert (
        hub.text_progress(first.session_id, StreamKind.TRANSCRIPT).errors[0]["start"]
        == 10
    )
    assert hub.text_progress(second.session_id, StreamKind.TRANSCRIPT).before == 0
    hub.set_text_enabled(first.session_id, StreamKind.VISUAL, False)
    state = hub.text_progress(first.session_id, StreamKind.VISUAL)
    assert not state.enabled and not state.final
    assert events[-1][0] == "stream.sealed"
    assert events[-1][1]["session_id"] == first.session_id
    with pytest.raises(ValueError):
        hub.seal_text(first.session_id, StreamKind.AUDIO, 1)
    with pytest.raises(KeyError):
        hub.text_progress("missing", StreamKind.VISUAL)


def test_stream_lookup_requires_enum() -> None:
    session = StreamHub().current
    assert session.stream(StreamKind.AUDIO) is session.audio
    assert session.stream(StreamKind.VIDEO) is session.video
    assert session.stream(StreamKind.TRANSCRIPT) is session.transcript
    assert session.stream(StreamKind.VISUAL) is session.visual
    with pytest.raises(TypeError, match="StreamKind"):
        session.stream("audio")  # type: ignore[arg-type]


def test_text_progress_requires_enum() -> None:
    hub = StreamHub()
    session_id = hub.current.session_id
    with pytest.raises(TypeError, match="StreamKind"):
        hub.text_progress(session_id, "transcript")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="StreamKind"):
        hub.seal_text(session_id, "transcript", 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="StreamKind"):
        hub.set_text_enabled(session_id, "visual", False)  # type: ignore[arg-type]


def test_text_identifiers_are_extensible() -> None:
    stream = TextStream("s")
    assert stream.name == "text"
    item = stream.append("custom", source="custom-producer")
    assert item is not None
    assert item.source == "custom-producer"


def test_consumer_media_kinds() -> None:
    hub = StreamHub()
    hub.register_consumer(
        "stt", MediaKind.AUDIO, cursor_fn=lambda: 12, gate_limit_fn=lambda: 30
    )
    hub.register_consumer("vlm", MediaKind.VIDEO, cursor_fn=lambda: 5)
    assert hub.consumer_cursor(MediaKind.AUDIO) == 12
    assert hub.consumer_cursor(MediaKind.VIDEO) == 5
    assert hub.consumer_gate_limit(MediaKind.AUDIO) == 30
    assert hub.consumers()[0]["kind"] is MediaKind.AUDIO
    hub.unregister_consumer("stt", MediaKind.AUDIO)
    assert hub.consumer_cursor(MediaKind.AUDIO) == 0
    hub.remove_plugin("vlm")
    assert hub.consumers() == []


def test_consumer_kind_requires_enum() -> None:
    hub = StreamHub()
    with pytest.raises(TypeError, match="MediaKind"):
        hub.register_consumer("stt", "audio")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MediaKind"):
        hub.unregister_consumer("stt", "audio")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MediaKind"):
        hub.consumer_cursor("audio")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MediaKind"):
        hub.consumer_gate_limit("audio")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MediaKind"):
        hub.register_consumer("stt", StreamKind.AUDIO)  # type: ignore[arg-type]
