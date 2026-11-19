from __future__ import annotations

from core.media import MediaKind, ProcessingMode, SourceKind
from core.settings_spec import SettingType
from core.ui import BlockType
from core.streams import StreamKind
from plugins.input.tree import TreeBadge
from plugins.input.plugin import JobState, StreamView
from plugins.input.sources import DownloadState, SourceState

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Callable
from unittest import mock

import pytest

from types import SimpleNamespace

from core.events import EventBus
from core.plugin import PluginContext
from core.services import ServiceRegistry
from core.streams import StreamHub
from core.ui import UIManager
from plugins.input.ffmpeg import (
    build_audio_cmd,
    build_reconnect_args,
    build_video_cmd,
    has_complete_frame,
    split_jpeg_frames,
)
from plugins.input.plugin import (
    InputPlugin,
    _map_live_streams,
    _map_view_streams,
)
from plugins.input.sources import (
    SourceManager,
    SourceSpec,
)
from plugins.input.tree import next_boundary
from tests.support import FakeSettings

_log = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _disable_media_probe(monkeypatch):
    # Unit tests use synthetic files and subprocesses, not decodable media.
    monkeypatch.setattr("plugins.input.sources.probe_duration", lambda path: None)


class FakeSource:
    """A source that appends fake audio/video data without ffmpeg."""

    def __init__(self, spec: SourceSpec, *, hub: Any, emit: Callable[..., Any]):
        self.spec = spec
        self._hub = hub
        self._emit = emit
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()
        self._running = False
        self._started_at: float = 0.0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def paused(self) -> bool:
        return not self._pause.is_set()

    @property
    def started_at(self) -> float:
        return self._started_at

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._started_at = time.time()
        self._pause.set()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"fake-input-{self.spec.title}",
            daemon=True,
        )
        self._thread.start()
        self._emit("input.status", status=SourceState.PLAYING, source=self.spec.title)

    def stop(self) -> None:
        self._running = False
        self._stop.set()
        self._pause.set()
        self._emit("input.status", status=SourceState.STOPPED, source=self.spec.title)

    def pause(self) -> None:
        self._pause.clear()
        self._emit("input.status", status=SourceState.PAUSED, source=self.spec.title)

    def resume(self) -> None:
        self._pause.set()
        self._emit("input.status", status=SourceState.PLAYING, source=self.spec.title)

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "paused": self.paused,
            "kind": self.spec.kind,
            "title": self.spec.title,
        }

    def _run(self) -> None:
        sample_rate = 16000
        fake_audio_chunk = b"\x00" * (sample_rate * 2)  # 1 second of silence
        fake_jpeg = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00\x43\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07"
            b"\x09\x09\x08\x0a\x0c\x14\x0d\x0c\x0b\x0b\x0c\x19\x12\x13\x0f"
            b"\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c\x20\x24\x2e\x27\x20\x22"
            b"\x2c\x23\x1c\x1c\x28\x37\x29\x2c\x30\x31\x34\x34\x34\x1f\x27"
            b"\x39\x3d\x38\x32\x3c\x2e\x33\x34\x32\xff\xc0\x00\x0b\x08\x00"
            b"\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05"
            b"\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01"
            b"\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\xff\xc4\x00\xb5\x10"
            b"\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01"
            b"\x7d\x01\x02\x03\x00\x04\x11\x05\x12\x21\x31\x41\x06\x13\x51"
            b"\x61\x07\x22\x71\x14\x32\x81\x91\xa1\x08\x23\x42\xb1\xc1\x15"
            b"\x52\xd1\xf0\x24\x33\x62\x72\x82\x09\x0a\x16\x17\x18\x19\x1a"
            b"\x25\x26\x27\x28\x29\x2a\x34\x35\x36\x37\x38\x39\x3a\x43\x44"
            b"\x45\x46\x47\x48\x49\x4a\x53\x54\x55\x56\x57\x58\x59\x5a\x63"
            b"\x64\x65\x66\x67\x68\x69\x6a\x73\x74\x75\x76\x77\x78\x79\x7a"
            b"\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98"
            b"\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5"
            b"\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2"
            b"\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7"
            b"\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda"
            b"\x00\x08\x01\x01\x00\x00\x3f\x00\xd2\xff\xd9"
        )
        while not self._stop.is_set():
            self._pause.wait()
            if self._stop.is_set():
                break
            try:
                current = self._hub.current
                current.audio.append(fake_audio_chunk)
                current.video.append_frame(fake_jpeg)
            except Exception:
                _log.exception("fake source append")
            time.sleep(0.05)


def _make_context(data_dir: Any = None) -> PluginContext:
    bus = EventBus()
    hub = StreamHub(emit=bus.emit)
    services = ServiceRegistry()
    settings = FakeSettings()
    services.register("settings", settings)
    settings.set("instant-summarize.enabled", False)
    ui = UIManager()
    import logging

    return PluginContext(
        plugin_id="input",
        data_dir=data_dir if data_dir is not None else mock.MagicMock(),
        events=bus,
        streams=hub,
        llm=mock.MagicMock(),
        ui=ui,
        services=services,
        meta=mock.MagicMock(),
        logger=logging.getLogger("test.input"),
    )


def _start_recording_job(ctx, plugin, monkeypatch, *, videos=("v1", "v2")):
    """Start a queued recording without external media processes."""
    started = Queue()
    monkeypatch.setattr(plugin, "_resolve_class_streams", lambda: None)
    monkeypatch.setattr(plugin, "_probe_job_durations", lambda job: None)
    monkeypatch.setattr(
        plugin._catalog,
        "get_recording_streams",
        lambda course_id, video_id: {"1": f"https://example.com/{video_id}.mp4"},
    )

    def start_source(spec):
        meta = {**spec.meta, **spec.course, "kind": spec.kind}
        if spec.continue_session:
            ctx.streams.current.meta.update(meta)
        else:
            ctx.streams.start_session(meta=meta)
        ctx.events.emit(
            "input.status",
            status=SourceState.PLAYING,
            session_id=ctx.streams.current.session_id,
        )
        started.put(spec)

    monkeypatch.setattr(plugin, "_start_source", start_source)
    opened = ctx.ui.dispatch_action(
        "input.open_class",
        {
            "id": "class:1:recording",
            "kind": SourceKind.RECORDING,
            "course": "1",
            "name": "高等数学",
            "label": "11-05 08:00-09:35",
            "videos": list(videos),
        },
    )
    assert opened.ok
    result = ctx.ui.dispatch_action("input.start_processing", {})
    assert result.ok
    first = started.get(timeout=5.0)
    assert ctx.events.flush()
    job = plugin._active_job
    assert job is not None
    assert job.state is JobState.RUNNING
    assert job.session_id == ctx.streams.current.session_id
    return job, ctx.streams.current, first, started


def _fake_jpeg() -> bytes:
    """A minimal valid JPEG (1x1 gray)."""
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00\x43\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07"
        b"\x09\x09\x08\x0a\x0c\x14\x0d\x0c\x0b\x0b\x0c\x19\x12\x13\x0f"
        b"\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c\x20\x24\x2e\x27\x20\x22"
        b"\x2c\x23\x1c\x1c\x28\x37\x29\x2c\x30\x31\x34\x34\x34\x1f\x27"
        b"\x39\x3d\x38\x32\x3c\x2e\x33\x34\x32\xff\xc0\x00\x0b\x08\x00"
        b"\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05"
        b"\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01"
        b"\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\xff\xc4\x00\xb5\x10"
        b"\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01"
        b"\x7d\x01\x02\x03\x00\x04\x11\x05\x12\x21\x31\x41\x06\x13\x51"
        b"\x61\x07\x22\x71\x14\x81\x91\xa1\x08\x23\x42\xb1\xc1\x15\x52"
        b"\xd1\xf0\x24\x33\x62\x72\x82\x09\x0a\x16\x17\x18\x19\x1a"
        b"\x25\x26\x27\x28\x29\x2a\x34\x35\x36\x37\x38\x39\x3a\x43\x44"
        b"\x45\x46\x47\x48\x49\x4a\x53\x54\x55\x56\x57\x58\x59\x5a\x63"
        b"\x64\x65\x66\x67\x68\x69\x6a\x73\x74\x75\x76\x77\x78\x79\x7a"
        b"\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98"
        b"\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5"
        b"\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2"
        b"\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7"
        b"\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda"
        b"\x00\x08\x01\x01\x00\x00\x3f\x00\xd2\xff\xd9"
    )


class TestJPEGSplitter:
    def test_single_frame(self):
        jpeg = _fake_jpeg()
        frames = split_jpeg_frames(jpeg)
        assert len(frames) == 1
        assert frames[0] == jpeg

    def test_two_frames(self):
        jpeg = _fake_jpeg()
        data = jpeg + jpeg
        frames = split_jpeg_frames(data)
        assert len(frames) == 2
        assert frames[0] == jpeg
        assert frames[1] == jpeg

    def test_garbage_before_frame(self):
        jpeg = _fake_jpeg()
        data = b"\x00\x00\xff\x00" + jpeg + jpeg
        frames = split_jpeg_frames(data)
        assert len(frames) == 2
        assert frames[0] == jpeg

    def test_garbage_between_frames(self):
        jpeg = _fake_jpeg()
        data = jpeg + b"\xaa\xbb\xcc" + jpeg
        frames = split_jpeg_frames(data)
        assert len(frames) == 2

    def test_sticky_half_frame(self):
        jpeg = _fake_jpeg()
        half = len(jpeg) // 2
        chunk1 = jpeg + jpeg[:half]
        frames = split_jpeg_frames(chunk1)
        assert len(frames) == 1
        # The second (partial) is dropped because EOI is missing.

        chunk2 = jpeg[half:] + jpeg
        frames2 = split_jpeg_frames(chunk2)
        # chunk2 starts mid-frame, but parser finds the EOI first.
        # The remaining data (mid-frame start to EOI) + another jpeg.
        assert len(frames2) >= 1

    def test_no_frame(self):
        assert split_jpeg_frames(b"hello world") == []
        assert split_jpeg_frames(b"") == []

    def test_soi_no_eoi(self):
        jpeg = _fake_jpeg()
        soi = jpeg[:10]
        frames = split_jpeg_frames(soi)
        assert frames == []

    def test_has_complete_frame(self):
        assert has_complete_frame(_fake_jpeg())
        assert not has_complete_frame(b"no jpeg here")
        assert not has_complete_frame(_fake_jpeg()[:10])

    def test_concat_many_frames(self):
        jpeg = _fake_jpeg()
        data = jpeg * 100
        frames = split_jpeg_frames(data)
        assert len(frames) == 100

    def test_embedded_fake_marker(self):
        jpeg = _fake_jpeg()
        # Insert a fake EOI-like sequence inside the data portion.
        # The parser only looks for exact \xff\xd8 and \xff\xd9 byte sequences.
        modified = jpeg[:50] + b"\xff\xd9" + jpeg[52:]
        frames = split_jpeg_frames(modified)
        assert len(frames) == 1
        assert len(frames[0]) < len(modified)


class TestBuildCommands:
    def test_audio_cmd_basic(self):
        cmd = build_audio_cmd("http://example.com/video.mp4")
        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert "http://example.com/video.mp4" in cmd
        assert "-acodec" in cmd
        assert "pcm_s16le" in cmd
        assert "-ar" in cmd
        assert "16000" in cmd
        assert "-ac" in cmd
        assert "1" in cmd
        assert "-f" in cmd
        assert "s16le" in cmd
        assert cmd[-1] == "pipe:1"

    def test_audio_cmd_custom_params(self):
        cmd = build_audio_cmd(
            "file.mp4",
            sample_rate=8000,
            channels=2,
        )
        assert "8000" in cmd
        assert "2" in cmd

    def test_video_cmd_basic(self):
        cmd = build_video_cmd("http://example.com/stream.flv")
        assert cmd[0] == "ffmpeg"
        assert "mjpeg" in cmd
        assert "image2pipe" in cmd
        assert cmd[-1] == "pipe:1"

    def test_video_cmd_live(self):
        cmd = build_video_cmd("http://example.com/live.flv", live=True)
        assert "nobuffer" in cmd
        assert "low_delay" in cmd

    def test_video_cmd_custom_fps_width(self):
        cmd = build_video_cmd("test.mp4", fps=2.0, width=640)
        assert any("fps=2.0" in s for s in cmd)
        assert any("scale=640:-1" in s for s in cmd)

    def test_headers_passed(self):
        cmd = build_audio_cmd(
            "url",
            headers={"Referer": "https://v.sjtu.edu.cn/", "User-Agent": "test/1.0"},
        )
        assert "-headers" in cmd
        # Multiple headers are joined into "-headers <k1>: <v1>\\r\\n<k2>: <v2>\\r\\n"
        header_arg = cmd[cmd.index("-headers") + 1]
        assert "Referer: https://v.sjtu.edu.cn/" in header_arg
        assert "User-Agent: test/1.0" in header_arg
        assert "-user_agent" in cmd

    def test_reconnect_args(self):
        args = build_reconnect_args(max_retries=5)
        assert "-reconnect" in args
        assert "1" in args
        assert "-reconnect_streamed" in args
        assert "-reconnect_delay_max" in args
        assert "5" in args

    def test_extra_args(self):
        cmd = build_audio_cmd("url", extra_args=["-ss", "10"])
        assert "-ss" in cmd
        assert "10" in cmd


class TestSourceManager:
    def test_failed_start_resets_state_and_reports_error(self, monkeypatch):
        hub = StreamHub(emit=lambda *a, **kw: None)
        events: list[dict] = []

        def emit(event_type: str, **kw: Any) -> None:
            events.append({"type": event_type, **kw})

        def fail_popen(*_args: Any, **_kwargs: Any) -> None:
            raise FileNotFoundError("ffmpeg")

        monkeypatch.setattr("plugins.input.sources.subprocess.Popen", fail_popen)
        mgr = SourceManager(hub=hub, emit=emit)
        spec = SourceSpec(
            kind=SourceKind.RECORDING, url="broken://source", title="Broken"
        )

        mgr.start(spec)

        assert mgr.state is SourceState.IDLE
        assert mgr.status()["source"] is None
        assert any(e["type"] == "input.error" for e in events)
        assert hub.current.ended_at is not None

    def test_start_stop_idle(self):
        mgr = SourceManager(
            hub=mock.MagicMock(),
            emit=mock.MagicMock(),
        )
        assert mgr.state is SourceState.IDLE
        spec = SourceSpec(kind=SourceKind.URL, url="test://source", title="Test")

        src1 = FakeSource(spec, hub=mock.MagicMock(), emit=mock.MagicMock())
        src2 = FakeSource(
            SourceSpec(kind=SourceKind.URL, url="test://other", title="Other"),
            hub=mock.MagicMock(),
            emit=mock.MagicMock(),
        )

        src1.start()
        assert src1.running
        src1.stop()
        assert not src1.running

        src2.start()
        assert src2.running
        src2.stop()
        assert not src2.running

    def test_fake_source_lifecycle(self):
        hub = StreamHub(emit=lambda *a, **kw: None)
        events: list[dict] = []

        def track_emit(event_type: str, **kw: Any) -> None:
            events.append({"type": event_type, **kw})

        spec = SourceSpec(kind=SourceKind.FILE, url="fake://input", title="Fake")
        src = FakeSource(spec, hub=hub, emit=track_emit)
        src.start()
        assert src.running
        import time

        time.sleep(0.2)
        src.pause()
        assert src.paused
        src.resume()
        assert not src.paused
        time.sleep(0.1)
        src.stop()
        assert not src.running

        statuses = [e["status"] for e in events if e.get("type") == "input.status"]
        assert SourceState.PLAYING in statuses
        assert SourceState.PAUSED in statuses
        assert SourceState.STOPPED in statuses

        assert hub.current.audio.duration > 0
        assert hub.current.video.duration > 0

    def test_source_manager_session_start(self):
        hub = StreamHub(emit=lambda *a, **kw: None)
        old_session = hub.current.session_id
        mgr = SourceManager(hub=hub, emit=lambda *a, **kw: None)

        assert mgr.state is SourceState.IDLE

    def test_continue_session(self, monkeypatch):
        import plugins.input.sources as sources_mod

        class DummySource:
            running = True
            finished = True
            _prepare_thread = None
            _download_thread = None
            _audio_thread = None
            _video_thread = None

            def __init__(self, spec, **kwargs):
                self.spec = spec
                self.audio_enabled = kwargs.get("audio_enabled")

            def start(self):
                pass

            def stop(self):
                pass

        monkeypatch.setattr(sources_mod, "FFmpegSource", DummySource)

        hub = StreamHub(emit=lambda *a, **kw: None)
        hub.start_session(meta={"course_name": "高等数学"})
        sid = hub.current.session_id

        hub.current.audio_gate.configure(1930.0, fail_closed=True)
        hub.current.audio_gate.release(90.0)
        hub.current.video_gate.release(75.0)
        enabled = lambda: False
        mgr = SourceManager(hub=hub, emit=lambda *a, **kw: None, audio_enabled=enabled)
        mgr.start(
            SourceSpec(
                kind=SourceKind.RECORDING,
                url="x",
                continue_session=True,
                audio_gate_limit=362.0,
                video_gate_limit=120.0,
            )
        )
        assert hub.current.audio_gate.released == 90.0
        assert hub.current.audio_gate.limit == 1930.0
        # A continuous STT window must fit after the next part starts.
        assert hub.current.audio_gate.wait(90.0 + 960.0)
        assert hub.current.video_gate.released == 75.0
        assert mgr._current.audio_enabled is enabled
        assert hub.current.session_id == sid, "next period must reuse the session"

        mgr.start(SourceSpec(kind=SourceKind.RECORDING, url="y"))
        assert hub.current.session_id != sid, "a new class starts a new session"
        assert hub.current.audio_gate.released == 0.0
        assert hub.current.audio_gate.limit == 0.0
        assert hub.current.video_gate.released == 0.0
        mgr.stop()


class TestCatalogNoToken:
    def test_no_token_message(self):
        from integrations.sjtu_canvas import InputError, SJTUCatalog

        cat = SJTUCatalog(token="", base_url="https://oc.sjtu.edu.cn")
        with pytest.raises(InputError, match="token"):
            cat.list_courses()

    def test_has_token_method(self):
        from plugins.input.catalog import Catalog

        settings = FakeSettings()
        settings.register(
            __import__("core.settings_spec", fromlist=["SettingsSpec"]).SettingsSpec(
                key="input.canvas_token",
                label="Token",
                type=SettingType.SECRET,
                default="",
            )
        )
        cat = Catalog(settings=settings)
        assert not cat.has_token()
        settings.set("input.canvas_token", "fake123")
        assert cat.has_token()


class TestCatalogMapping:
    def test_catalog_uses_fake_settings(self):
        from plugins.input.catalog import Catalog

        settings = FakeSettings(
            {"input.canvas_token": "fake", "input.canvas_base_url": "http://localhost"}
        )
        cat = Catalog(settings=settings)
        assert cat.has_token()
        cat.close()
        cat.reset()

    def test_has_token_ignores_env(self, monkeypatch):
        from plugins.input.catalog import Catalog

        monkeypatch.setenv("CANVAS_TOKEN", "env-token-123")
        settings = FakeSettings({"input.canvas_token": ""})
        cat = Catalog(settings=settings)
        assert not cat.has_token()

        settings.set("input.canvas_token", "settings-token")
        assert cat.has_token()

    def test_list_courses_ttl_cache_hit(self):
        from plugins.input.catalog import Catalog

        counter = [0]
        fake_courses = [{"id": "1", "name": "Test"}]

        class FakeSJTUCatalog:
            def list_courses(self):
                counter[0] += 1
                return fake_courses

            def close(self):
                pass

        settings = FakeSettings(
            {"input.canvas_token": "fake", "input.canvas_base_url": "http://localhost"}
        )
        cat = Catalog(settings=settings)
        cat._sjt_cat = FakeSJTUCatalog()

        result1 = cat.list_courses()
        result2 = cat.list_courses()
        assert result1 == fake_courses
        assert result2 == fake_courses
        assert counter[0] == 1, f"Expected 1 call, got {counter[0]}"

    def test_ttl_cache_expires(self):
        from plugins.input.catalog import Catalog

        counter = [0]

        class FakeSJTUCatalog:
            def list_courses(self):
                counter[0] += 1
                return [{"id": str(counter[0])}]

            def close(self):
                pass

        settings = FakeSettings({"input.canvas_token": "fake"})
        cat = Catalog(settings=settings)
        cat._sjt_cat = FakeSJTUCatalog()

        result1 = cat.list_courses()
        assert counter[0] == 1
        result2 = cat.list_courses()
        assert counter[0] == 1

        cat._cache[("courses",)] = (0.0, [])
        result3 = cat.list_courses()
        assert counter[0] == 2

    def test_reset_clears_cache(self):
        from plugins.input.catalog import Catalog

        counter = [0]

        class FakeSJTUCatalog:
            def list_courses(self):
                counter[0] += 1
                return [{"id": str(counter[0])}]

            def close(self):
                pass

        settings = FakeSettings({"input.canvas_token": "fake"})
        cat = Catalog(settings=settings)
        cat._sjt_cat = FakeSJTUCatalog()

        result1 = cat.list_courses()
        assert counter[0] == 1
        result2 = cat.list_courses()
        assert counter[0] == 1

        cat.reset()
        assert cat._sjt_cat is None
        assert len(cat._cache) == 0

        cat._sjt_cat = FakeSJTUCatalog()
        result3 = cat.list_courses()
        assert counter[0] == 2

    def test_recordings_cache_per_course(self):
        from plugins.input.catalog import Catalog

        calls = []

        class FakeSJTUCatalog:
            def list_recordings(self, course_id, limit=50):
                calls.append(("list_recordings", course_id, limit))
                return [{"video_id": f"{course_id}-1"}]

            def close(self):
                pass

        settings = FakeSettings({"input.canvas_token": "fake"})
        cat = Catalog(settings=settings)
        cat._sjt_cat = FakeSJTUCatalog()

        r1a = cat.list_recordings("42")
        r1b = cat.list_recordings("42")
        r2 = cat.list_recordings("99")
        assert len(calls) == 2
        assert r1a[0]["video_id"] == "42-1"
        assert r2[0]["video_id"] == "99-1"


class TestInputPluginUI:
    def test_live_progress(self, tmp_path, monkeypatch):
        from core.streams import StreamKind

        ctx = _make_context(tmp_path)
        session = ctx.streams.current
        session.started_at = 1000.0
        session.audio.discard(220)
        ctx.streams.seal_text(session.session_id, StreamKind.TRANSCRIPT, 100)
        ctx.streams.seal_text(session.session_id, StreamKind.VISUAL, 180)
        status = {"session_id": session.session_id, "partial": {"end_ms": 218000}}
        ctx.services.register(
            "stt", SimpleNamespace(enabled=lambda: True, status=lambda: status)
        )
        ctx.services.register("vlm", SimpleNamespace(enabled=lambda: True))
        notes = {
            "chunk_enabled": True,
            "completed_chunks": 0,
            "expected_chunks": None,
            "published_end": 0,
            "error": "",
            "errors": [],
        }
        ctx.services.register(
            "notes-pipeline", SimpleNamespace(status=lambda sid: notes)
        )
        plugin = InputPlugin()
        plugin._ctx = ctx
        monkeypatch.setattr("plugins.input.plugin.time.time", lambda: 1240.0)
        blocks = plugin._render_recording_page(
            True, False, {}, session.session_id, active=True, progress_only=True
        )
        assert blocks[0]["value"] == pytest.approx(180 / 240)
        assert "落后 01:00" in blocks[0]["label"]
        assert blocks[1]["value"] == pytest.approx(218 / 240)
        assert "落后 00:22" in blocks[1]["label"]
        assert len(blocks) == 4
        assert session.transcript_progress.before == 100
        status["session_id"] = "another-course"
        blocks = plugin._render_recording_page(
            True, False, {}, session.session_id, active=True, progress_only=True
        )
        assert blocks[1]["value"] == pytest.approx(100 / 240)
        recorded = plugin._render_recording_page(
            True, False, {}, session.session_id, active=True, total=600
        )
        assert recorded[2]["value"] == pytest.approx(100 / 600)
        session.input_done_at = 1240.0
        monkeypatch.setattr("plugins.input.plugin.time.time", lambda: 2000.0)
        finished = plugin._render_recording_page(
            False, True, {}, session.session_id, progress_only=True
        )
        assert finished[0]["value"] == pytest.approx(100 / 240)

    def test_class_errors(self, tmp_path):
        ctx = _make_context(tmp_path)
        sid = ctx.streams.current.session_id
        notes = {
            "chunk_enabled": False,
            "completed_chunks": 0,
            "error": "",
            "errors": [],
        }
        ctx.services.register(
            "notes-pipeline", SimpleNamespace(status=lambda sid: notes)
        )
        plugin = InputPlugin()
        plugin._ctx = ctx
        assert not any(
            b["type"] == BlockType.HTML
            for b in plugin._render_recording_page(False, False, {}, sid)
        )
        error = {
            "source": "visual",
            "start": 10.0,
            "end": 20.0,
            "message": "<script>failed</script>",
        }
        notes["errors"] = [error, error, {"chunk": 0, "error": "writer failed"}]
        blocks = plugin._render_recording_page(False, False, {}, sid)
        errors = [b for b in blocks if b["type"] == BlockType.HTML]
        assert len(errors) == 1
        assert errors[0]["html"].count("visual 10.0-20.0s") == 1
        assert "&lt;script&gt;" in errors[0]["html"]
        assert "writer failed" in errors[0]["html"]

    def test_course_download_progress(self, tmp_path):
        from core.events import Event
        from plugins.input.plugin import _Job

        ctx = _make_context(tmp_path)
        plugin = InputPlugin()
        plugin.setup(ctx)
        try:
            cls = {"id": "course", "name": "概率论", "videos": ["v1", "v2"]}
            job = _Job(id="course", cls=cls, state=JobState.RUNNING)
            plugin._active_job = job
            plugin._jobs[job.id] = job
            plugin._current_class = cls
            for video, status, received in [
                ("v2", DownloadState.PROGRESS, 25),
                ("v1", DownloadState.DONE, 100),
            ]:
                plugin._on_input_event(
                    Event(
                        type="input.download",
                        ts=0.0,
                        event_id="download-progress",
                        payload={
                            "video_id": video,
                            "status": status,
                            "received": received,
                            "total": 100,
                        },
                    )
                )
            progress = plugin._download_progress(job)
            assert progress == {
                "status": DownloadState.PROGRESS,
                "received": 125,
                "total": 200,
            }
            blocks = plugin._render_session()
            assert [b for b in blocks if b["type"] == BlockType.PROGRESS][1][
                "value"
            ] == 0.625
            plugin._on_input_event(
                Event(
                    type="input.download",
                    ts=0.0,
                    event_id="download-done",
                    payload={
                        "video_id": "v2",
                        "status": DownloadState.DONE,
                        "received": 100,
                        "total": 100,
                    },
                )
            )
            blocks = plugin._render_session()
            assert [b for b in blocks if b["type"] == BlockType.PROGRESS][1][
                "value"
            ] == 1.0
        finally:
            plugin.teardown()

    def test_action_stop_stops_source(self):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        result = ctx.ui.dispatch_action("input.stop", {})
        assert result.ok

        plugin.teardown()

    def test_action_pause_resume_without_source(self):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        result_pause = ctx.ui.dispatch_action("input.pause", {})
        assert result_pause.ok
        result_resume = ctx.ui.dispatch_action("input.resume", {})
        assert result_resume.ok

        plugin.teardown()

    def test_action_play_url_empty(self):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        result = ctx.ui.dispatch_action("input.play_url", {"url": ""})
        assert not result.ok

        plugin.teardown()

    def test_action_play_url_with_url(self):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        # This will try to start ffmpeg which won't work, but the action
        # returns ok immediately (start is sync).
        result = ctx.ui.dispatch_action(
            "input.play_url",
            {"url": "fake://test", "title": "Fake Source", "kind": SourceKind.URL},
        )
        assert result.ok
        plugin.teardown()

    def test_manual_form_payload_has_url_title_kind(self):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        result = ctx.ui.dispatch_action(
            "input.play_url",
            {
                "url": "rtsp://example.com/live",
                "title": "Test Live",
                "kind": SourceKind.LIVE,
            },
        )
        assert result.ok
        assert result.open_page == "input:session"

        result2 = ctx.ui.dispatch_action("input.play_url", {"title": "No URL"})
        assert not result2.ok

        plugin.teardown()

    def test_action_toggle_course(self):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        result = ctx.ui.dispatch_action("input.toggle_course", {"course": "12345"})
        assert result.ok

        plugin.teardown()

    def test_action_open_class_missing_videos(self):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        result = ctx.ui.dispatch_action(
            "input.open_class", {"kind": SourceKind.RECORDING}
        )
        assert not result.ok

        plugin.teardown()

    def test_action_open_class_live_missing_id(self):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        result = ctx.ui.dispatch_action("input.open_class", {"kind": SourceKind.LIVE})
        assert not result.ok

        plugin.teardown()

    def test_refresh_courses(self):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        result = ctx.ui.dispatch_action("input.refresh_courses", {})
        assert result.ok
        plugin.teardown()


class TestFFmpegSourceInternals:
    def test_read1_is_used_over_read(self):
        from plugins.input.sources import FFmpegSource

        spec = SourceSpec(kind=SourceKind.FILE, url="/tmp/test.mp4", title="test")
        src = FFmpegSource(
            spec,
            hub=mock.MagicMock(),
            emit=mock.MagicMock(),
            reconnect=False,
        )

        class HasRead1:
            read1_called = False
            read_called = False

            def read1(self, size: int = -1) -> bytes:
                HasRead1.read1_called = True
                return b""

            def read(self, size: int = -1) -> bytes:
                HasRead1.read_called = True
                return b""

        stdout = HasRead1()
        read = getattr(stdout, "read1", stdout.read)
        read(4096)
        assert HasRead1.read1_called, "read1 should be preferred over read"
        assert not HasRead1.read_called, "read should not be called when read1 exists"

    def test_read_fallback_when_no_read1(self):

        class NoRead1:
            read_called = False

            def read(self, size: int = -1) -> bytes:
                NoRead1.read_called = True
                return b""

        stdout = NoRead1()
        read = getattr(stdout, "read1", stdout.read)
        read(4096)
        assert NoRead1.read_called, "read fallback should work when read1 is absent"

    def test_restart_local_file_no_http_headers(self, monkeypatch):
        from plugins.input.ffmpeg import build_audio_cmd, build_video_cmd
        from plugins.input.sources import FFmpegSource

        spec = SourceSpec(
            kind=SourceKind.FILE,
            url="/tmp/test.mp4",
            title="test",
            headers={"Referer": "https://should-be-dropped/"},
        )
        src = FFmpegSource(
            spec,
            hub=mock.MagicMock(),
            emit=mock.MagicMock(),
            reconnect=False,
        )
        with mock.patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value.stdout = None
            mock_popen.return_value.poll.return_value = -1
            src._restart_audio_process()
            audio_cmd = mock_popen.call_args[0][0]
            assert "-headers" not in audio_cmd, (
                f"Local file should not get -headers: {audio_cmd}"
            )

    def test_restart_http_url_passes_headers(self, monkeypatch):
        from plugins.input.sources import FFmpegSource

        spec = SourceSpec(
            kind=SourceKind.RECORDING,
            url="https://example.com/video.mp4",
            title="test",
            headers={"Referer": "https://v.sjtu.edu.cn/"},
        )
        src = FFmpegSource(
            spec,
            hub=mock.MagicMock(),
            emit=mock.MagicMock(),
            reconnect=False,
        )
        with mock.patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value.stdout = None
            mock_popen.return_value.poll.return_value = -1
            src._restart_audio_process()
            audio_cmd = mock_popen.call_args[0][0]
            assert "-headers" in audio_cmd, f"HTTP URL should pass headers: {audio_cmd}"


class TestRecordingIngest:
    def _source(self, hub=None, **kwargs):
        from plugins.input.sources import FFmpegSource

        hub = hub or StreamHub(emit=lambda *a, **kw: None)
        events: list[dict] = []

        def emit(event_type: str, **kw: Any) -> None:
            events.append({"type": event_type, **kw})

        spec = SourceSpec(
            kind=SourceKind.RECORDING, url="/tmp/x.mp4", title="t", **kwargs
        )
        src = FFmpegSource(spec, hub=hub, emit=emit, reconnect=False)
        return src, hub, events

    def test_ended_waits_for_both_readers(self) -> None:

        src, hub, events = self._source()
        hub.start_session(meta={"mode": ProcessingMode.FAST})
        src._mark_reader_done(MediaKind.AUDIO)
        assert not [e for e in events if e["type"] == "input.ended"]
        assert hub.current.input_done is False

        src._mark_reader_done(MediaKind.VIDEO)
        ended = [e for e in events if e["type"] == "input.ended"]
        assert len(ended) == 1
        assert hub.current.input_done is True

        src._mark_reader_done(MediaKind.VIDEO)
        assert len([e for e in events if e["type"] == "input.ended"]) == 1

    def test_gate_limits_come_from_the_spec(self) -> None:
        hub = StreamHub(emit=lambda *a, **kw: None)
        mgr = SourceManager(hub=hub, emit=lambda *a, **kw: None)
        spec = SourceSpec(
            kind=SourceKind.RECORDING,
            url="x",
            title="t",
            realtime=False,
            audio_gate_limit=120.0,
            video_gate_limit=60.0,
        )
        with mock.patch("plugins.input.sources.subprocess.Popen") as popen:
            popen.return_value.stdout = None
            popen.return_value.poll.return_value = 1
            mgr.start(spec)
            mgr.stop()
        assert hub.current.audio_gate.limit == 120.0
        assert hub.current.video_gate.limit == 60.0

    def test_session_meta_records_mode(self) -> None:
        hub = StreamHub(emit=lambda *a, **kw: None)
        mgr = SourceManager(hub=hub, emit=lambda *a, **kw: None)
        with mock.patch("plugins.input.sources.subprocess.Popen") as popen:
            popen.return_value.stdout = None
            popen.return_value.poll.return_value = 1
            mgr.start(
                SourceSpec(
                    kind=SourceKind.RECORDING, url="x", title="t", realtime=False
                )
            )
            mgr.stop()
        assert hub.current.meta["mode"] == ProcessingMode.FAST

    def test_download_is_used_for_remote_recordings(self, tmp_path) -> None:
        from plugins.input.sources import FFmpegSource

        hub = StreamHub(emit=lambda *a, **kw: None)
        hub.start_session(meta={})
        downloaded: list[str] = []

        class FakeDownloader:
            def start(self, url, dest, *, stop_event=None, on_progress=None):
                from plugins.input.download import DownloadResult

                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"media")
                downloaded.append(str(dest))
                if on_progress:
                    on_progress(5, 5)
                return mock.Mock(
                    input_url=str(dest),
                    wait=lambda: DownloadResult(path=dest, total=5, ok=True),
                )

        spec = SourceSpec(
            kind=SourceKind.RECORDING,
            url="https://live.sjtu.edu.cn/x/rec.mp4?key=1",
            title="t",
            download=True,
            realtime=False,
        )
        src = FFmpegSource(
            spec,
            hub=hub,
            emit=lambda *a, **kw: None,
            downloader=FakeDownloader(),
            media_dir=tmp_path,
        )
        with (
            mock.patch("plugins.input.sources.probe_duration", return_value=1.0),
            mock.patch("plugins.input.sources.subprocess.Popen") as popen,
        ):
            popen.return_value.stdout = None
            popen.return_value.poll.return_value = 1
            src.start()
            src._download_done.wait(2)
        assert downloaded, "recording should be downloaded first"
        assert hub.current.meta["media_name"] == "rec.mp4"
        assert src._input_url.startswith(str(tmp_path))
        assert hub.current.meta["media_path"].startswith(str(tmp_path))

    def test_download_failure_stops_input(self, tmp_path) -> None:
        from plugins.input.sources import FFmpegSource

        hub = StreamHub(emit=lambda *a, **kw: None)
        hub.start_session(meta={})
        events: list[dict] = []

        class BrokenDownloader:
            def start(self, url, dest, *, stop_event=None, on_progress=None):
                raise OSError("nope")

        spec = SourceSpec(
            kind=SourceKind.RECORDING,
            url="https://live.sjtu.edu.cn/x/rec.mp4",
            title="t",
            download=True,
            realtime=False,
        )
        src = FFmpegSource(
            spec,
            hub=hub,
            emit=lambda event_type, **kw: events.append({"type": event_type, **kw}),
            downloader=BrokenDownloader(),
            media_dir=tmp_path,
        )
        with mock.patch("plugins.input.sources.subprocess.Popen") as popen:
            popen.return_value.stdout = None
            popen.return_value.poll.return_value = 1
            src.start()
        popen.assert_not_called()
        assert not src.running
        assert any(
            e["type"] == "input.error" and "nope" in e["message"] for e in events
        )


class TestMediaPrune:
    def test_keeps_recent_media_folders(self, tmp_path) -> None:
        from plugins.input.plugin import InputPlugin

        ctx = _make_context(tmp_path)
        plugin = InputPlugin()
        plugin.setup(ctx)
        sessions = tmp_path / "sessions"
        for index in range(6):
            folder = sessions / f"s{index}" / "media"
            folder.mkdir(parents=True)
            (folder / "a.mp4").write_bytes(b"x")
            os.utime(folder, (index + 1, index + 1))
        ctx.require("settings").set("input.keep_recordings", 2)

        plugin._prune_media()

        left = sorted(path.parent.name for path in sessions.glob("*/media"))
        assert left == ["s4", "s5"]
        plugin.teardown()


class TestSJTUCatalogUnit:
    def test_no_token_raises_input_error(self):
        from integrations.sjtu_canvas import InputError, SJTUCatalog

        cat = SJTUCatalog(token="", base_url="http://localhost")
        with pytest.raises(InputError, match="token"):
            cat.list_courses()

    def test_context_manager(self):
        from integrations.sjtu_canvas import SJTUCatalog

        with SJTUCatalog(token="test", base_url="http://localhost") as cat:
            pass
        with pytest.raises(Exception):
            cat.list_courses()

    def test_list_lives_needs_token(self):
        from integrations.sjtu_canvas import InputError, SJTUCatalog

        cat = SJTUCatalog(token="", base_url="http://localhost")
        with pytest.raises(InputError):
            cat.list_lives("2024-01-01 00:00:00", "2024-01-01 23:59:59")

    def test_get_live_devices_no_token(self):
        from integrations.sjtu_canvas import InputError, SJTUCatalog

        cat = SJTUCatalog(token="test", base_url="http://localhost")
        # No real Canvas to resolve auth → will fail trying to connect
        with pytest.raises(InputError):
            cat.get_live_devices("live123")

    def test_list_lives_name_fallback_to_subj_name(self):
        """name field maps subjName > courName > name (real API uses subjName)."""
        from integrations.sjtu_canvas import SJTUCatalog

        fake_records = [
            {
                "id": 200951,
                "subjName": "初等数论",
                "courName": None,
                "clroName": "东下院115",
                "courBeginTime": "2024-12-15 08:00:00",
                "courEndTime": "2024-12-15 09:40:00",
                "status": "0",
            },
            {
                "id": 200952,
                "subjName": None,
                "courName": "Fallback Name",
                "clroName": "东上院201",
                "courBeginTime": "2024-12-15 10:00:00",
                "courEndTime": "2024-12-15 11:40:00",
            },
        ]

        with mock.patch.object(
            SJTUCatalog, "_ensure_live_token", return_value=("fake-jwt", {})
        ):
            with mock.patch("httpx2.Client.get") as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = {
                    "code": 0,
                    "data": {"records": fake_records},
                }
                mock_get.return_value.raise_for_status = lambda: None

                cat = SJTUCatalog(token="test", base_url="http://localhost")
                lives = cat.list_lives("2024-12-15 00:00:00", "2024-12-15 23:59:59")
                assert len(lives) == 2
                assert lives[0]["name"] == "初等数论"
                assert lives[1]["name"] == "Fallback Name"
                cat.close()


class TestLocalTime:
    def test_local_time(self):
        from plugins.input.plugin import _local_time

        result = _local_time("2024-12-15 08:30:00")
        assert result
        assert "12-15" in result or "12-15" in result

    def test_local_time_empty(self):
        from plugins.input.plugin import _local_time

        assert _local_time("") == ""
        assert _local_time("not a date") == "not a date"


class TestClassPlayer:
    def _open_class(
        self, ctx, plugin, *, videos=("v1", "v2"), kind=SourceKind.RECORDING
    ):
        payload = {
            "id": "class:1:" + kind,
            "kind": kind,
            "course": "1",
            "name": "高等数学",
            "label": "11-05 08:00-09:35",
            "videos": list(videos),
        }
        if kind == SourceKind.LIVE:
            payload["live_id"] = "L1"
        return ctx.ui.dispatch_action("input.open_class", payload)

    def test_open_class_does_not_start_playback(self, monkeypatch):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        monkeypatch.setattr(plugin, "_resolve_class_streams", lambda: None)

        result = self._open_class(ctx, plugin)
        assert result.ok
        assert result.open_page == "input:session"
        assert plugin._processing is False
        assert plugin._sources.state == SourceState.IDLE, (
            "selecting a class must not start ffmpeg"
        )
        assert ctx.streams.current.session_id == "boot", (
            "no session until processing starts"
        )
        assert plugin._current_class["videos"] == ["v1", "v2"]

        plugin.teardown()

    def test_reprocess_restarts_from_the_first_part(self, monkeypatch):
        """「重新处理」must drop the streams of the part previously loaded."""

        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        monkeypatch.setattr(plugin, "_resolve_class_streams", lambda: None)

        self._open_class(ctx, plugin, videos=("v1", "v2"))
        monkeypatch.setattr(plugin, "_resolve_job_streams", lambda job: None)
        plugin._class_index = 1
        plugin._class_streams = {StreamView.MAIN: "https://example.com/v2.mp4"}
        plugin._streams_loading = False
        plugin._class_finished = True

        result = ctx.ui.dispatch_action("input.reprocess", {})
        assert result.ok
        assert plugin._class_index == 0
        assert plugin._class_streams == {}, "stale streams would re-download part 2"
        assert plugin._streams_loading is True, "part 1 streams must be re-resolved"
        assert plugin._class_finished is False

        plugin.teardown()

    def test_player_page_shows_both_video_streams(self, monkeypatch):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        monkeypatch.setattr(plugin, "_resolve_class_streams", lambda: None)
        self._open_class(ctx, plugin, kind=SourceKind.LIVE)
        plugin._class_streams = {
            StreamView.MAIN: "https://example.com/main.m3u8",
            StreamView.SCREEN: "https://example.com/screen.m3u8",
        }
        plugin._streams_loading = False

        ctx.ui.open_page("input:session")
        blocks = ctx.ui.state()["page_content"]["input:session"]
        player = next(block for block in blocks if block["type"] == BlockType.PLAYER)
        assert [src["label"] for src in player["sources"]] == ["课堂", "屏幕"]
        assert player["processing"] is False

        plugin.teardown()

    def test_recording_page_has_no_player_but_shows_progress(self, monkeypatch):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        monkeypatch.setattr(plugin, "_resolve_class_streams", lambda: None)
        self._open_class(ctx, plugin, kind=SourceKind.RECORDING)
        plugin._class_streams = {StreamView.MAIN: "https://example.com/main.mp4"}
        plugin._streams_loading = False
        plugin._processing = True
        plugin._total_seconds = 3600.0

        ctx.ui.open_page("input:session")
        blocks = ctx.ui.state()["page_content"]["input:session"]
        assert not [b for b in blocks if b["type"] == BlockType.PLAYER], (
            "recordings are processed at full speed: no in-page player"
        )
        progress = [b for b in blocks if b["type"] == BlockType.PROGRESS]
        assert progress, "recording page must show processing progress"

        ctx.streams.current.meta["media_name"] = "clip.mp4"
        ctx.ui.open_page("input:session")
        blocks = ctx.ui.state()["page_content"]["input:session"]
        links = [
            b
            for b in blocks
            if b["type"] == BlockType.MARKDOWN
            and "/api/artifacts/" in b.get("text", "")
        ]
        assert links, "a local media link must be offered"

        plugin.teardown()

    @pytest.mark.parametrize("kind", [SourceKind.RECORDING, SourceKind.LIVE])
    def test_start_processing_starts_ingest(self, monkeypatch, kind):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        monkeypatch.setattr(plugin, "_resolve_class_streams", lambda: None)
        self._open_class(ctx, plugin, kind=kind)
        ctx.streams.register_consumer(
            "vlm", MediaKind.VIDEO, gate_limit_fn=lambda: 75.0
        )
        plugin._class_streams = {StreamView.MAIN: "https://example.com/main.m3u8"}
        plugin._streams_loading = False
        started: list[SourceSpec] = []
        monkeypatch.setattr(plugin, "_start_source", lambda spec: started.append(spec))

        result = ctx.ui.dispatch_action("input.start_processing", {})
        assert result.ok
        assert plugin._processing is True
        deadline = time.monotonic() + 5
        while not started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert started and started[0].url == "https://example.com/main.m3u8"
        assert started[0].kind is kind
        assert started[0].continue_session is False
        assert started[0].realtime is (kind == SourceKind.LIVE)
        assert started[0].download is (kind == SourceKind.RECORDING)

        plugin.teardown()

    def test_pause_processing_pauses_recording(self, monkeypatch):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        monkeypatch.setattr(plugin, "_resolve_class_streams", lambda: None)
        self._open_class(ctx, plugin)
        plugin._class_streams = {StreamView.MAIN: "u"}
        plugin._processing = True
        calls: list[str] = []
        monkeypatch.setattr(plugin._sources, "pause", lambda: calls.append("pause"))
        monkeypatch.setattr(plugin._sources, "stop", lambda: calls.append("stop"))

        result = ctx.ui.dispatch_action("input.pause_processing", {})
        assert result.ok
        assert plugin._processing is False
        assert calls == ["pause"]

        plugin.teardown()

    def test_pause_processing_stops_live_ingest(self, monkeypatch):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        monkeypatch.setattr(plugin, "_resolve_class_streams", lambda: None)
        self._open_class(ctx, plugin, kind=SourceKind.LIVE)
        plugin._class_streams = {StreamView.MAIN: "u"}
        plugin._processing = True
        calls: list[str] = []
        monkeypatch.setattr(plugin._sources, "pause", lambda: calls.append("pause"))
        monkeypatch.setattr(plugin._sources, "stop", lambda: calls.append("stop"))

        ctx.ui.dispatch_action("input.pause_processing", {})
        assert calls == ["stop"], "live must not pause, it stops ingestion"

        plugin.teardown()

    def test_class_ended_advances_to_next_period(self, monkeypatch):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        try:
            job, session, first, started = _start_recording_job(
                ctx, plugin, monkeypatch
            )
            assert first.course["video_id"] == "v1"
            assert first.final_part is False
            assert first.next_source.url == "https://example.com/v2.mp4"

            plugin._on_class_ended()
            second = started.get(timeout=5.0)
            assert plugin._class_index == 1
            assert plugin._want_processing is True
            assert job.input_part_index == 1
            assert plugin._active_job is job
            assert second.course["video_id"] == "v2"
            assert second.continue_session is True
            assert second.final_part is True
            assert ctx.streams.current is session
            assert session.ended_at is None
        finally:
            plugin.teardown()

    def test_last_period_ends_the_session(self, monkeypatch):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        try:
            job, session, first, _started = _start_recording_job(
                ctx,
                plugin,
                monkeypatch,
                videos=("v1",),
            )
            assert first.final_part is True
            assert first.next_source is None
            wait = mock.Mock(wraps=plugin._wait_for_consumers)
            monkeypatch.setattr(plugin, "_wait_for_consumers", wait)

            plugin._on_class_ended()
            assert plugin._class_finished is True
            assert plugin._processing is False
            assert session.ended_at is not None
            assert session.end_reason == "playback finished"
            assert job.state == JobState.DONE
            assert job.session_id == session.session_id
            assert plugin._active_job is None
            wait.assert_called_once_with()
        finally:
            plugin.teardown()

    def test_input_ended_event_advances(self, monkeypatch):
        from types import SimpleNamespace
        import time

        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        self._open_class(ctx, plugin)
        plugin._processing = True
        advanced: list[bool] = []
        monkeypatch.setattr(plugin, "_on_class_ended", lambda: advanced.append(True))

        event = SimpleNamespace(
            type="input.ended", get=lambda key, default=None: default
        )
        plugin._on_input_event(event)

        deadline = time.monotonic() + 2.0
        while not advanced and time.monotonic() < deadline:
            time.sleep(0.01)
        assert advanced == [True]

        plugin.teardown()

    def test_stop_processing_state(self, monkeypatch):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        self._open_class(ctx, plugin)
        plugin._processing = True
        monkeypatch.setattr(plugin._sources, "stop", lambda: None)

        result = ctx.ui.dispatch_action("input.stop", {})
        assert result.ok
        assert plugin._processing is False
        assert plugin._class_finished is True

        plugin.teardown()

    def test_reselecting_finished_class_allows_replay(self, monkeypatch):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        monkeypatch.setattr(plugin, "_resolve_class_streams", lambda: None)
        self._open_class(ctx, plugin)
        plugin._class_finished = True

        result = self._open_class(ctx, plugin)
        assert result.ok
        assert plugin._class_finished is False
        assert plugin._class_index == 0

        plugin.teardown()

    def test_input_ended_without_processing_is_ignored(self, monkeypatch):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        self._open_class(ctx, plugin)
        advanced: list[bool] = []
        monkeypatch.setattr(
            plugin, "_end_fill_or_playback", lambda: advanced.append(True)
        )

        plugin._on_class_ended()
        assert plugin._class_index == 0
        assert advanced == []

        plugin.teardown()


class TestStreamMapping:
    def test_view_streams_use_view_numbers(self):
        streams = _map_view_streams({"1": "main-url", "2": "screen-url"})
        assert streams == {StreamView.MAIN: "main-url", StreamView.SCREEN: "screen-url"}

    def test_view_streams_fallback_to_first(self):
        assert _map_view_streams({"0": "only-url"}) == {StreamView.MAIN: "only-url"}

    def test_live_streams_use_devices(self):
        streams = _map_live_streams(
            [
                {"name": "camera", "url": "cam", "view_num": 1},
                {"name": "screen", "url": "scr", "view_num": 2},
            ]
        )
        assert streams == {StreamView.MAIN: "cam", StreamView.SCREEN: "scr"}

    def test_live_streams_without_view_numbers(self):
        streams = _map_live_streams([{"url": "a"}, {"url": "b"}])
        assert streams == {StreamView.MAIN: "a", StreamView.SCREEN: "b"}


def test_next_boundary_is_the_nearest_future_moment() -> None:
    from datetime import datetime

    now = datetime(2026, 11, 12, 10, 0, 0)
    lives = [
        {"begin": "2026-11-12 09:00:00", "end": "2026-11-12 10:20:00"},
        {"begin": "2026-11-12 10:30:00", "end": "2026-11-12 12:00:00"},
    ]
    assert next_boundary(lives, now=now) == datetime(2026, 11, 12, 10, 20).timestamp()


def test_next_boundary_ignores_past_windows() -> None:
    from datetime import datetime

    now = datetime(2026, 11, 12, 13, 0, 0)
    lives = [{"begin": "2026-11-12 08:00:00", "end": "2026-11-12 09:40:00"}]
    assert next_boundary(lives, now=now) == 0.0


def test_clock_tick_refreshes_once_at_the_boundary(monkeypatch) -> None:
    plugin = InputPlugin()
    plugin._tree_lock = threading.RLock()
    plugin._tree = {
        "nodes": [],
        "error": "",
        "at": 0.0,
        "started": 0.0,
        "loading": False,
        "next_boundary": 1000.0,
    }
    reloads: list[bool] = []
    refreshes: list[str] = []
    monkeypatch.setattr(
        plugin, "_ensure_tree", lambda *, force=False: reloads.append(force)
    )
    plugin._ctx = SimpleNamespace(
        ui=SimpleNamespace(refresh=lambda key: refreshes.append(key))
    )

    plugin._on_clock_tick({"ts": 999.0})
    assert reloads == []

    plugin._on_clock_tick({"ts": 1000.5})
    assert reloads == [True]
    assert refreshes == ["input.tree"]
    assert plugin._tree["next_boundary"] == 0.0

    plugin._on_clock_tick({"ts": 1001.0})
    assert reloads == [True]


def test_pick_recordings_keeps_every_overlapping_part() -> None:

    from datetime import datetime

    recordings = [
        {"video_id": "c", "begin": "2026-11-12 13:55:00", "end": "2026-11-12 14:50:00"},
        {"video_id": "a", "begin": "2026-11-12 15:55:00", "end": "2026-11-12 16:50:00"},
        {"video_id": "b", "begin": "2026-11-12 16:50:20", "end": "2026-11-12 17:45:00"},
    ]
    started = datetime(2026, 11, 12, 16, 0).timestamp()
    ended = datetime(2026, 11, 12, 18, 0).timestamp()

    picked = InputPlugin._pick_recordings(recordings, started, ended)

    assert [item["video_id"] for item in picked] == ["a", "b"]


def test_pick_recordings_handles_three_parts() -> None:
    from datetime import datetime

    recordings = [
        {
            "video_id": f"v{i}",
            "begin": f"2026-11-12 {begin}",
            "end": f"2026-11-12 {end}",
        }
        for i, (begin, end) in enumerate(
            [
                ("08:00:00", "08:50:00"),
                ("08:50:20", "09:40:00"),
                ("09:50:00", "10:40:00"),
            ]
        )
    ]
    started = datetime(2026, 11, 12, 8, 5).timestamp()
    ended = datetime(2026, 11, 12, 10, 40).timestamp()

    picked = InputPlugin._pick_recordings(recordings, started, ended)

    assert [item["video_id"] for item in picked] == ["v0", "v1", "v2"]


class TestProcessingQueue:
    def _setup(self, monkeypatch):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        monkeypatch.setattr(plugin, "_resolve_class_streams", lambda: None)
        started: list[SourceSpec] = []
        monkeypatch.setattr(plugin, "_start_source", lambda spec: started.append(spec))
        return ctx, plugin, started

    @staticmethod
    def _open(ctx, plugin, class_id, name):
        return ctx.ui.dispatch_action(
            "input.open_class",
            {
                "id": class_id,
                "kind": SourceKind.RECORDING,
                "course": "1",
                "name": name,
                "label": "11-05 08:00-09:35",
                "videos": [f"{class_id}-v1"],
            },
        )

    def test_switching_class_keeps_the_running_job(self, monkeypatch):
        ctx, plugin, started = self._setup(monkeypatch)
        self._open(ctx, plugin, "class:a", "高等数学")
        plugin._class_streams = {StreamView.MAIN: "https://example.com/a.mp4"}
        plugin._streams_loading = False
        ctx.ui.dispatch_action("input.start_processing", {})
        assert plugin._active_job is not None
        assert plugin._active_job.id == "class:a"

        self._open(ctx, plugin, "class:b", "线性代数")
        assert plugin._processing is True
        assert plugin._active_job is not None
        assert plugin._active_job.id == "class:a"
        assert [job.url for job in started] == ["https://example.com/a.mp4"]

        plugin.teardown()

    def test_second_class_waits_in_queue_then_starts(self, monkeypatch):
        ctx, plugin, started = self._setup(monkeypatch)
        self._open(ctx, plugin, "class:a", "高等数学")
        plugin._class_streams = {StreamView.MAIN: "https://example.com/a.mp4"}
        plugin._streams_loading = False
        ctx.ui.dispatch_action("input.start_processing", {})

        self._open(ctx, plugin, "class:b", "线性代数")
        plugin._class_streams = {StreamView.MAIN: "https://example.com/b.mp4"}
        plugin._streams_loading = False
        result = ctx.ui.dispatch_action("input.start_processing", {})
        assert result.ok
        assert plugin._queue == ["class:b"]
        assert plugin._active_job.id == "class:a"

        first = plugin._active_job
        plugin._finish_job(first, "test finished")
        deadline = time.monotonic() + 5
        while len(started) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert [job.url for job in started] == [
            "https://example.com/a.mp4",
            "https://example.com/b.mp4",
        ]
        assert plugin._active_job is not None
        assert plugin._active_job.id == "class:b"

        plugin.teardown()

    def test_context_names_the_selected_class(self, monkeypatch):
        ctx, plugin, _started = self._setup(monkeypatch)
        assert ctx.ui.state()["context"] is None
        self._open(ctx, plugin, "class:a", "高等数学")
        context = ctx.ui.state()["context"]
        assert context["title"] == "高等数学"
        assert "11-05" in context["subtitle"]
        assert context["selected"] is True
        plugin.teardown()

    def test_course_tree_badges_the_background_job(self, monkeypatch):
        ctx, plugin, _started = self._setup(monkeypatch)
        plugin._tree["nodes"] = [
            {
                "id": "course:1",
                "children": [
                    {
                        "id": "class:a",
                        "action": {
                            "action": "input.open_class",
                            "value": {"id": "class:a"},
                        },
                    }
                ],
            }
        ]
        self._open(ctx, plugin, "class:a", "高等数学")
        plugin._class_streams = {StreamView.MAIN: "https://example.com/a.mp4"}
        plugin._streams_loading = False
        ctx.ui.dispatch_action("input.start_processing", {})
        plugin._apply_tree_state()
        child = plugin._tree["nodes"][0]["children"][0]
        assert child["badge"] == "处理中"
        assert child["badge_class"] == TreeBadge.LIVE

        self._open(ctx, plugin, "class:b", "线性代数")
        plugin._apply_tree_state()
        child = plugin._tree["nodes"][0]["children"][0]
        assert child["badge"] == "处理中"
        plugin.teardown()

    def test_background_multipart_keeps_duration_and_selected_context(
        self, monkeypatch
    ):
        ctx, plugin, _unused = self._setup(monkeypatch)
        try:
            job, session, first, started = _start_recording_job(
                ctx, plugin, monkeypatch
            )
            assert first.next_source.url == "https://example.com/v2.mp4"
            self._open(ctx, plugin, "class:b", "线性代数")
            plugin._class_streams = {StreamView.MAIN: "https://example.com/b.mp4"}
            plugin._streams_loading = False
            assert ctx.ui.dispatch_action("input.start_processing", {}).ok
            selected = plugin._jobs["class:b"]
            assert selected.state is JobState.QUEUED

            for video_id, duration in (("v1", 3039.0), ("v2", 3287.0)):
                ctx.events.emit(
                    "input.prepared",
                    video_id=video_id,
                    duration=duration,
                    session_id=session.session_id,
                )
            assert ctx.events.flush()
            assert job.part_durations == {"v1": 3039.0, "v2": 3287.0}
            assert job.total_seconds == 6326.0
            assert selected.part_durations == {}
            assert selected.total_seconds == plugin._total_seconds == 0.0
            assert plugin._part_durations == {}

            wait = mock.Mock(side_effect=AssertionError("drained between parts"))
            monkeypatch.setattr(plugin, "_wait_for_consumers", wait)
            plugin._on_class_ended()
            second = started.get(timeout=5.0)
            assert second.url == first.next_source.url
            assert second.continue_session is True
            assert job.input_part_index == 1
            assert job.total_seconds == 6326.0
            assert plugin._active_job is job
            assert plugin._queue == [selected.id]
            assert plugin._current_class["id"] == selected.id
            assert plugin._class_index == selected.input_part_index == 0
            assert (
                plugin._class_streams
                == selected.streams
                == {
                    StreamView.MAIN: "https://example.com/b.mp4",
                }
            )
            assert ctx.streams.current is session
            assert session.ended_at is None
            assert not session.input_done
            wait.assert_not_called()
            context = ctx.ui.state()["context"]
            assert context["title"] == "线性代数"
            assert "高等数学" in context["subtitle"]
            assert context["selected"] is True
            assert not [
                b for b in plugin._render_session() if b["type"] == BlockType.PROGRESS
            ]
        finally:
            plugin.teardown()


@pytest.mark.parametrize("kind", [SourceKind.RECORDING, SourceKind.LIVE])
def test_manual_source_combines_audio_transport_and_video_consumer_gates(
    monkeypatch, kind
) -> None:
    ctx = _make_context()
    plugin = InputPlugin()
    plugin.setup(ctx)
    ctx.streams.register_consumer("stt", MediaKind.AUDIO, gate_limit_fn=lambda: 37.0)
    ctx.streams.register_consumer("vlm", MediaKind.VIDEO, gate_limit_fn=lambda: 75.0)
    started = []
    ready = threading.Event()

    def start(spec):
        started.append(spec)
        ready.set()

    monkeypatch.setattr(plugin, "_start_source", start)
    try:
        result = ctx.ui.dispatch_action(
            "input.play_url",
            {"url": "https://example.com/media", "kind": kind},
        )
        assert result.ok
        assert ready.wait(5.0)
        assert started[0].audio_gate_limit == 37.0
        assert started[0].video_gate_limit == (0.0 if kind == SourceKind.LIVE else 75.0)
    finally:
        plugin.teardown()


def test_gap_fill_combines_audio_transport_and_video_consumer_gates(
    monkeypatch, tmp_path
) -> None:
    ctx = _make_context(tmp_path)
    plugin = InputPlugin()
    plugin.setup(ctx)
    ctx.streams.register_consumer("stt", MediaKind.AUDIO, gate_limit_fn=lambda: 43.0)
    ctx.streams.register_consumer("vlm", MediaKind.VIDEO, gate_limit_fn=lambda: 75.0)
    plugin._fill_course = {"course_id": "1"}
    plugin._fill_target = "live-session"
    monkeypatch.setattr(
        plugin._catalog,
        "get_recording_streams",
        lambda course_id, video_id: {"1": "https://example.com/recording.mp4"},
    )
    started = []
    monkeypatch.setattr(plugin, "_start_source", lambda spec: started.append(spec))
    try:
        assert plugin._start_fill_part({"video_id": "v1"}, continue_session=False)
        assert started[0].audio_gate_limit == 43.0
        assert started[0].video_gate_limit == 75.0
        assert started[0].meta["fill_for"] == "live-session"
    finally:
        plugin.teardown()


def test_audio_enabled_uses_public_service_only() -> None:
    plugin = InputPlugin()
    plugin._ctx = SimpleNamespace(service=lambda name: None)
    assert not plugin._audio_enabled()
    service = SimpleNamespace(enabled=lambda: True)
    plugin._ctx = SimpleNamespace(service=lambda name: service)
    assert plugin._audio_enabled()


def _run_audio_reader(hub, chunks, *, enabled=None, emit=None):
    from plugins.input.sources import FFmpegSource

    source = FFmpegSource(
        SourceSpec(kind=SourceKind.RECORDING, url="test"),
        hub=hub,
        emit=emit or (lambda *a, **kw: None),
        reconnect=False,
        audio_enabled=enabled,
    )
    source._session_id = hub.current.session_id
    stdout = mock.Mock()
    stdout.read1.side_effect = [*chunks, b""]
    source._audio_proc = SimpleNamespace(stdout=stdout, poll=lambda: 0)
    source._read_audio()
    return source


def test_audio_reader_gates_each_small_chunk_before_append() -> None:
    hub = StreamHub()
    session = hub.current
    endpoints = []

    def allow(end):
        assert session.audio.duration < end
        assert end - session.audio.duration <= 0.2 + 1e-9
        endpoints.append(end)
        return True

    session.audio_gate.wait = allow
    _run_audio_reader(hub, [b"\x01\x02" * 8000])
    assert endpoints == pytest.approx([0.2, 0.4, 0.5])
    assert session.audio.duration == pytest.approx(0.5)
    assert session.audio.chunk_count == 3
    assert session.audio.read(0, 0.5) == b"\x01\x02" * 8000


def test_disabled_audio_discards_payload_and_resumes_real_pcm() -> None:
    hub = StreamHub()
    session = hub.current
    enabled = iter([False, False, True])
    _run_audio_reader(
        hub,
        [b"\x01\x02" * 9600],
        enabled=lambda: next(enabled),
    )
    assert session.audio.duration == pytest.approx(0.6)
    assert session.audio.chunk_count == 1
    assert session.audio.earliest_time == pytest.approx(0.4)
    assert session.audio.read(0, 0.4) == b""
    assert session.audio.read(0.4, 0.6) == b"\x01\x02" * 3200
    assert session.audio_gate.released == pytest.approx(0.4)


def test_audio_gate_failure_reports_error_without_appending() -> None:
    hub = StreamHub()
    hub.current.audio_gate.wait = lambda end: False
    events = []
    source = _run_audio_reader(
        hub,
        [b"\x01\x02" * 3200],
        emit=lambda name, **payload: events.append((name, payload)),
    )
    assert hub.current.audio.duration == 0.0
    assert not hub.current.input_done
    assert any(name == "input.error" for name, payload in events)
    assert not source.running


def test_audio_reader_stays_bound_to_original_session() -> None:
    hub = StreamHub()
    original = hub.current

    def enabled():
        hub.start_session()
        return True

    _run_audio_reader(hub, [b"\x01\x02" * 3200], enabled=enabled)
    assert original.audio.duration == pytest.approx(0.2)
    assert hub.current.audio.duration == 0.0


def test_file_eof_completes_both_readers_without_reconnect() -> None:
    from plugins.input.sources import FFmpegSource

    hub = StreamHub()
    session = hub.start_session()
    events = []
    source = FFmpegSource(
        SourceSpec(kind=SourceKind.FILE, url="/tmp/input.wav", realtime=False),
        hub=hub,
        emit=lambda name, **payload: events.append(name),
        reconnect=True,
    )
    source._session_id = session.session_id
    audio_stdout = mock.Mock()
    audio_stdout.read1.side_effect = [b"\x01\x02" * 3200, b""]
    video_stdout = mock.Mock()
    video_stdout.read1.return_value = b""
    source._audio_proc = SimpleNamespace(stdout=audio_stdout, poll=lambda: 0)
    source._video_proc = SimpleNamespace(stdout=video_stdout, poll=lambda: 0)
    source._restart_audio_process = mock.Mock(
        side_effect=AssertionError("Audio restarted after EOF")
    )
    source._restart_video_process = mock.Mock(
        side_effect=AssertionError("Video restarted after EOF")
    )

    source._read_audio()
    assert session.audio.duration == pytest.approx(0.2)
    assert not session.input_done
    source._read_video()
    assert session.input_done
    assert source.finished
    assert events.count("input.ended") == 1
    source._restart_audio_process.assert_not_called()
    source._restart_video_process.assert_not_called()


def test_multipart_progress_uses_full_duration(tmp_path, monkeypatch) -> None:
    ctx = _make_context(tmp_path)
    plugin = InputPlugin()
    plugin.setup(ctx)
    try:
        job, session, _first, _started = _start_recording_job(ctx, plugin, monkeypatch)
        cursors = {"audio": 6326.0, "video": 6000.0}
        for name, kind in (("stt", MediaKind.AUDIO), ("vlm", MediaKind.VIDEO)):
            ctx.services.register(name, SimpleNamespace(enabled=lambda: True))
            ctx.streams.register_consumer(
                name,
                kind,
                cursor_fn=lambda kind=kind: cursors[kind],
            )
        ctx.events.emit(
            "input.prepared",
            video_id="v1",
            duration=3039.0,
            session_id=session.session_id,
        )
        assert ctx.events.flush()
        assert job.part_durations == {"v1": 3039.0}
        assert plugin._total_seconds == 0.0
        assert job.total_seconds == 0.0
        blocks = plugin._render_session()
        progress = next(b for b in blocks if b["type"] == BlockType.PROGRESS)
        assert "1:40:00" in progress["label"]
        assert progress["value"] == 0.0

        ctx.events.emit(
            "input.prepared",
            video_id="v2",
            duration=3287.0,
            session_id=session.session_id,
        )
        assert ctx.events.flush()
        assert plugin._total_seconds == 6326.0
        assert job.total_seconds == 6326.0
        ctx.events.emit(
            "input.download",
            video_id="v1",
            status=DownloadState.PROGRESS,
            total=10,
            received=5,
            session_id=session.session_id,
        )
        assert ctx.events.flush()
        blocks = plugin._render_session()
        progress = [b for b in blocks if b["type"] == BlockType.PROGRESS]
        assert len(progress) == 4
        assert "1:40:00" in progress[0]["label"]
        assert "1:45:26" in progress[0]["label"]
        assert progress[0]["value"] == pytest.approx(6000 / 6326)
    finally:
        plugin.teardown()


def test_early_duration_and_completed_progress(tmp_path, monkeypatch):
    from plugins.input.plugin import _Job

    ctx = _make_context(tmp_path)
    plugin = InputPlugin()
    plugin.setup(ctx)
    session = ctx.streams.start_session()
    cls = {
        "id": "course",
        "videos": ["v1", "v2"],
        "recording_parts": [
            {
                "video_id": "v1",
                "begin": "2026-11-16 08:00:00",
                "end": "2026-11-16 08:10:00",
            },
            {
                "video_id": "v2",
                "begin": "2026-11-16 08:15:00",
                "end": "2026-11-16 08:25:00",
            },
        ],
    }
    job = _Job(
        id="course",
        cls=cls,
        session_id=session.session_id,
        state=JobState.RUNNING,
        recording_streams={
            "v1": {StreamView.MAIN: "first"},
            "v2": {StreamView.MAIN: "second"},
        },
    )
    plugin._active_job = job
    plugin._jobs[job.id] = job
    plugin._current_class = cls
    for name in ("stt", "vlm"):
        ctx.services.register(name, SimpleNamespace(enabled=lambda: True))
    try:
        ctx.streams.seal_text(session.session_id, StreamKind.TRANSCRIPT, 300)
        ctx.streams.seal_text(session.session_id, StreamKind.VISUAL, 300)
        progress = [
            b for b in plugin._render_session() if b["type"] == BlockType.PROGRESS
        ]
        assert progress[0]["value"] == 0.25
        probed = []

        def duration(url):
            probed.append(url)
            return 600.0

        monkeypatch.setattr("plugins.input.plugin.probe_duration", duration)
        plugin._probe_job_durations(job)
        assert set(probed) == {"first", "second"}
        assert session.meta["total_media_seconds"] == 1200
        ctx.streams.seal_text(
            session.session_id, StreamKind.TRANSCRIPT, 1200, final=True
        )
        ctx.streams.seal_text(session.session_id, StreamKind.VISUAL, 1200, final=True)
        job.state = JobState.DONE
        plugin._active_job = None
        progress = [
            b for b in plugin._render_session() if b["type"] == BlockType.PROGRESS
        ]
        assert [b["value"] for b in progress] == [1.0, 1.0, 1.0, 1.0]
    finally:
        plugin.teardown()


def test_part_transition_does_not_drain_consumers(tmp_path, monkeypatch) -> None:
    ctx = _make_context(tmp_path)
    plugin = InputPlugin()
    plugin.setup(ctx)
    try:
        job, session, first, started = _start_recording_job(ctx, plugin, monkeypatch)
        assert first.next_source.url == "https://example.com/v2.mp4"
        wait = mock.Mock(side_effect=AssertionError("drained between parts"))
        monkeypatch.setattr(plugin, "_wait_for_consumers", wait)

        plugin._on_class_ended()
        second = started.get(timeout=5.0)
        assert plugin._class_index == 1
        assert plugin._class_streams == {StreamView.MAIN: "https://example.com/v2.mp4"}
        assert job.input_part_index == 1
        assert plugin._active_job is job
        assert second.url == first.next_source.url
        assert second.continue_session is True
        assert ctx.streams.current is session
        assert session.ended_at is None
        assert not session.input_done
        wait.assert_not_called()
    finally:
        plugin.teardown()


def test_prefetch_and_continuous_audio(tmp_path, monkeypatch) -> None:
    import plugins.input.sources as sources
    from plugins.input.download import DownloadResult
    from plugins.stt.plugin import STTPlugin

    download_started = threading.Event()
    allow_download = threading.Event()
    downloaded = []

    class Downloader:
        def start(self, url, dest, *, stop_event=None, on_progress=None):
            downloaded.append(url)
            if "part2" in url:
                download_started.set()
                assert allow_download.wait(2.0)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"media")
            return mock.Mock(
                input_url=str(dest),
                wait=lambda: DownloadResult(path=dest, total=5, ok=True),
            )

    monkeypatch.setattr(sources, "probe_duration", lambda path: 0.2)
    monkeypatch.setattr(sources.subprocess, "Popen", lambda *a, **kw: mock.Mock())
    monkeypatch.setattr(sources.FFmpegSource, "_read_audio", lambda self: None)
    monkeypatch.setattr(sources.FFmpegSource, "_read_video", lambda self: None)
    events = []
    hub = StreamHub(emit=lambda name, **data: events.append(name))
    session_start_count = events.count("session.started")
    mgr = SourceManager(
        hub=hub,
        emit=lambda *a, **kw: None,
        media_root=tmp_path,
        downloader=Downloader(),
    )
    second = SourceSpec(
        kind=SourceKind.RECORDING,
        url="https://example.com/part2/record.mp4",
        download=True,
        realtime=False,
        continue_session=True,
        course={"video_id": "v2"},
    )
    first = SourceSpec(
        kind=SourceKind.RECORDING,
        url="https://example.com/part1/record.mp4",
        download=True,
        realtime=False,
        final_part=False,
        next_source=second,
        course={"video_id": "v1"},
    )
    try:
        mgr.start(first)
        session = hub.current
        source1 = mgr._current
        assert download_started.wait(2.0)
        assert source1.running and not source1.finished
        session.audio.append(b"\x01\x02" * 3200)
        session.audio_gate.release(0.1)
        source1._mark_reader_done(MediaKind.AUDIO)
        source1._mark_reader_done(MediaKind.VIDEO)
        assert not session.input_done
        assert not STTPlugin._finished(session)
        assert session.audio_gate.wait(0.2)

        allow_download.set()
        mgr.start(second)
        assert hub.current is session
        assert session.audio_gate.released == 0.1
        source2 = mgr._current
        assert source2._audio_offset == pytest.approx(0.2)
        session.audio.append(b"\x03\x04" * 3200)
        source2._mark_reader_done(MediaKind.AUDIO)
        source2._mark_reader_done(MediaKind.VIDEO)
        assert session.input_done and STTPlugin._finished(session)
        assert session.audio.read(0, 0.4) == b"\x01\x02" * 3200 + b"\x03\x04" * 3200
        assert session.audio.buffered_ranges() == [(0.0, 0.4)]
        assert [
            part["offset"] for part in session.meta["media_parts"]
        ] == pytest.approx([0.0, 0.2])
        assert len(set(part["path"] for part in session.meta["media_parts"])) == 2
        assert downloaded.count(first.url) == downloaded.count(second.url) == 1
        assert events.count("session.started") == session_start_count + 1
        assert "session.stopped" not in events
    finally:
        allow_download.set()
        mgr.stop()


def test_canceled_prefetch_does_not_restart_download(monkeypatch) -> None:
    from plugins.input.sources import FFmpegSource

    source = FFmpegSource(
        SourceSpec(kind=SourceKind.RECORDING, url="test"),
        hub=StreamHub(),
        emit=lambda *a, **kw: None,
    )
    source._prepare_done = threading.Event()
    source._prepare_done.set()
    source.prepare = mock.Mock(
        side_effect=AssertionError("download restarted after cancel")
    )
    source.stop()
    source.start()
    assert not source.running
    source.prepare.assert_not_called()
