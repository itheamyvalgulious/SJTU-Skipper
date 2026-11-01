from __future__ import annotations

import threading
import time

import pytest

from core.streams import StreamHub, TextStream


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
    audio.append(b"\x00\x00" * 16000)
    wav = audio.read_as_wav(0, 1)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"


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
    item0 = text.append("hello ", start=0, end=1, source="stt")
    item1 = text.append("world", start=1, end=2, source="stt")
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
    stream = TextStream("s", "transcript")
    item = stream.append("x", meta={"a": 1})
    assert item is not None
    with pytest.raises(TypeError):
        item.meta["a"] = 2  # type: ignore[index]


def test_stream_hub_events_are_coalesced() -> None:
    events: list[dict] = []
    hub = StreamHub(emit=lambda name, **payload: events.append({"name": name, **payload}))
    session = hub.start_session(meta={"course": "math"})
    events.clear()  # drop boot/session event
    audio = session.audio
    for _ in range(5):
        audio.append(b"\x00\x00" * 1600)  # 0.1 s each
    new = [e for e in events if e["name"] == "stream.appended" and e["stream"] == "audio"]
    assert len(new) == 1
    assert new[0]["start"] == 0.0
    assert new[0]["end"] == pytest.approx(0.5)

    session.transcript.append("hi", start=0, end=1)
    text_events = [
        e for e in events if e["name"] == "stream.appended" and e["stream"] == "transcript"
    ]
    assert len(text_events) == 1
    assert text_events[0]["chars"] == 2


def test_session_switch_keeps_old_stream_readable() -> None:
    hub = StreamHub()
    first = hub.start_session(meta={"n": 1})
    first.audio.append(b"\x00\x00" * 16000)
    first.transcript.append("one")
    second = hub.start_session(meta={"n": 2})
    assert second.session_id != first.session_id
    assert second.audio.duration == 0
    assert first.audio.duration == pytest.approx(1.0)
    assert first.transcript[0].text == "one"
    assert len(hub.list_sessions()) >= 3  # boot + first + second


def test_stream_set_status() -> None:
    hub = StreamHub()
    session = hub.start_session(meta={"course": "topology"})
    session.audio.append(b"\x00\x00" * 16000)
    session.transcript.append("abc")
    session.visual.append("defg")
    status = session.status()
    assert status["audio_seconds"] == pytest.approx(1.0)
    assert status["transcript_chars"] == 3
    assert status["visual_chars"] == 4
    assert status["meta"]["course"] == "topology"
