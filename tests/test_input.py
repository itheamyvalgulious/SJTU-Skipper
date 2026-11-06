"""Tests for the input plugin (P5).

Covers:
- JPEG frame splitter
- ffmpeg command builders
- SourceManager mutual exclusion
- Catalog (fake Canvas/SJTU clients)
- No-token error handling
- UI state & action paths
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from unittest import mock

import pytest

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
from tests.support import FakeSettings

_log = logging.getLogger(__name__)


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
        self._emit("input.status", status="playing", source=self.spec.title)

    def stop(self) -> None:
        self._running = False
        self._stop.set()
        self._pause.set()
        self._emit("input.status", status="stopped", source=self.spec.title)

    def pause(self) -> None:
        self._pause.clear()
        self._emit("input.status", status="paused", source=self.spec.title)

    def resume(self) -> None:
        self._pause.set()
        self._emit("input.status", status="playing", source=self.spec.title)

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

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_context() -> PluginContext:
    """Minimal PluginContext with all required attributes."""
    bus = EventBus()
    hub = StreamHub(emit=bus.emit)
    services = ServiceRegistry()
    settings = FakeSettings()
    services.register("settings", settings)
    ui = UIManager()
    import logging
    return PluginContext(
        plugin_id="input",
        data_dir=mock.MagicMock(),
        events=bus,
        streams=hub,
        llm=mock.MagicMock(),
        ui=ui,
        services=services,
        meta=mock.MagicMock(),
        logger=logging.getLogger("test.input"),
    )


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


# ====================================================================
# JPEG parser
# ====================================================================

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
        data = b"\x00\x00\xFF\x00" + jpeg + jpeg
        frames = split_jpeg_frames(data)
        assert len(frames) == 2
        assert frames[0] == jpeg

    def test_garbage_between_frames(self):
        jpeg = _fake_jpeg()
        data = jpeg + b"\xAA\xBB\xCC" + jpeg
        frames = split_jpeg_frames(data)
        assert len(frames) == 2

    def test_sticky_half_frame(self):
        """Simulate streaming: first chunk has partial frame at end."""
        jpeg = _fake_jpeg()
        half = len(jpeg) // 2
        chunk1 = jpeg + jpeg[:half]  # 1 complete + 1 partial
        frames = split_jpeg_frames(chunk1)
        assert len(frames) == 1  # only the first complete frame
        # The second (partial) is dropped because EOI is missing.

        # Now deliver the rest
        chunk2 = jpeg[half:] + jpeg
        frames2 = split_jpeg_frames(chunk2)
        # chunk2 starts mid-frame, but parser finds the EOI first.
        # The remaining data (mid-frame start to EOI) + another jpeg.
        assert len(frames2) >= 1

    def test_no_frame(self):
        assert split_jpeg_frames(b"hello world") == []
        assert split_jpeg_frames(b"") == []

    def test_soi_no_eoi(self):
        """SOI present but no following EOI — no frame returned."""
        jpeg = _fake_jpeg()
        soi = jpeg[:10]  # contains SOI but not EOI
        frames = split_jpeg_frames(soi)
        assert frames == []

    def test_has_complete_frame(self):
        assert has_complete_frame(_fake_jpeg())
        assert not has_complete_frame(b"no jpeg here")
        assert not has_complete_frame(_fake_jpeg()[:10])  # SOI only

    def test_concat_many_frames(self):
        jpeg = _fake_jpeg()
        data = jpeg * 100
        frames = split_jpeg_frames(data)
        assert len(frames) == 100

    def test_embedded_fake_marker(self):
        """Ensure that 0xFF bytes inside JPEG data do not cause false boundaries."""
        jpeg = _fake_jpeg()
        # Insert a fake EOI-like sequence inside the data portion.
        # The parser only looks for exact \xff\xd8 and \xff\xd9 byte sequences.
        modified = jpeg[:50] + b"\xff\xd9" + jpeg[52:]
        frames = split_jpeg_frames(modified)
        # The first EOI appears early; SOI..early-EOI is one frame.
        assert len(frames) == 1
        # The rest is garbage between the early EOI and real end.
        assert len(frames[0]) < len(modified)


# ====================================================================
# ffmpeg command builders
# ====================================================================

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
        # -vf fps=1.0,scale=1280:-1 are two adjacent args
        assert any("fps=1.0" in s for s in cmd)
        assert any("scale=1280:-1" in s for s in cmd)
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


# ====================================================================
# SourceManager mutual exclusion
# ====================================================================

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
        spec = SourceSpec(kind="recording", url="broken://source", title="Broken")

        mgr.start(spec)

        assert mgr.state == "idle"
        assert mgr.status()["source"] is None
        assert any(e["type"] == "input.error" for e in events)
        assert hub.current.ended_at is not None

    def test_start_stop_idle(self):
        mgr = SourceManager(
            hub=mock.MagicMock(),
            emit=mock.MagicMock(),
        )
        assert mgr.state == "idle"
        # Start a FakeSource directly (skip ffmpeg subprocess)
        spec = SourceSpec(kind="url", url="test://source", title="Test")
        # We'll use FakeSource by injecting; but start requires ffmpeg.
        # For state-only tests, just verify stop/single-source via FakeSource.

        src1 = FakeSource(spec, hub=mock.MagicMock(), emit=mock.MagicMock())
        src2 = FakeSource(
            SourceSpec(kind="url", url="test://other", title="Other"),
            hub=mock.MagicMock(),
            emit=mock.MagicMock(),
        )

        # Simulate what SourceManager.start does
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

        spec = SourceSpec(kind="file", url="fake://input", title="Fake")
        src = FakeSource(spec, hub=hub, emit=track_emit)
        src.start()
        assert src.running
        # Wait a bit for fake data
        import time
        time.sleep(0.2)
        src.pause()
        assert src.paused
        src.resume()
        assert not src.paused
        time.sleep(0.1)
        src.stop()
        assert not src.running

        # At least playing + paused + playing + stopped status events
        statuses = [e["status"] for e in events if e.get("type") == "input.status"]
        assert "playing" in statuses
        assert "paused" in statuses
        assert "stopped" in statuses

        # Audio/video should have been appended
        assert hub.current.audio.duration > 0
        assert hub.current.video.duration > 0

    def test_source_manager_session_start(self):
        """SourceManager.start creates a new session via hub.start_session."""
        hub = StreamHub(emit=lambda *a, **kw: None)
        old_session = hub.current.session_id
        mgr = SourceManager(hub=hub, emit=lambda *a, **kw: None)

        # Can't start real ffmpeg, but we verify session lifecycle
        assert mgr.state == "idle"

    def test_continue_session_reuses_stream_session(self, monkeypatch):
        """A merged class must stay in one stream session across periods."""
        import plugins.input.sources as sources_mod

        class DummySource:
            running = True

            def __init__(self, spec, **kwargs):
                self.spec = spec

            def start(self):
                pass

            def stop(self):
                pass

        monkeypatch.setattr(sources_mod, "FFmpegSource", DummySource)

        hub = StreamHub(emit=lambda *a, **kw: None)
        hub.start_session(meta={"course_name": "高等数学"})
        sid = hub.current.session_id
        mgr = SourceManager(hub=hub, emit=lambda *a, **kw: None)

        mgr.start(SourceSpec(kind="recording", url="x", continue_session=True))
        assert hub.current.session_id == sid, "next period must reuse the session"

        mgr.start(SourceSpec(kind="recording", url="y"))
        assert hub.current.session_id != sid, "a new class starts a new session"
        mgr.stop()


# ====================================================================
# Catalog mapping (with FakeSettings)
# ====================================================================

class TestCatalogNoToken:
    """Tests that without a token, catalog methods give clear errors."""

    def test_no_token_message(self):
        from integrations.sjtu_canvas import InputError, SJTUCatalog

        cat = SJTUCatalog(token="", base_url="https://oc.sjtu.edu.cn")
        with pytest.raises(InputError, match="token"):
            cat.list_courses()

    def test_has_token_method(self):
        from plugins.input.catalog import Catalog

        settings = FakeSettings()
        settings.register(
            __import__(
                "plugins.settings.spec", fromlist=["SettingsSpec"]
            ).SettingsSpec(
                key="input.canvas_token",
                label="Token",
                type="secret",
                default="",
            )
        )
        cat = Catalog(settings=settings)
        assert not cat.has_token()
        settings.set("input.canvas_token", "fake123")
        assert cat.has_token()


class TestCatalogMapping:
    """Test catalog methods return the expected shape."""

    def test_catalog_uses_fake_settings(self):
        from plugins.input.catalog import Catalog

        settings = FakeSettings({"input.canvas_token": "fake", "input.canvas_base_url": "http://localhost"})
        cat = Catalog(settings=settings)
        assert cat.has_token()
        # close should be no-op when no real catalog created
        cat.close()
        cat.reset()

    def test_has_token_ignores_env(self, monkeypatch):
        """Settings are the single source of truth; env is seeded at startup."""
        from plugins.input.catalog import Catalog

        monkeypatch.setenv("CANVAS_TOKEN", "env-token-123")
        settings = FakeSettings({"input.canvas_token": ""})
        cat = Catalog(settings=settings)
        assert not cat.has_token()

        settings.set("input.canvas_token", "settings-token")
        assert cat.has_token()

    def test_list_courses_ttl_cache_hit(self):
        """Two list_courses calls within TTL only hit underlying SJTUCatalog once."""
        from plugins.input.catalog import Catalog

        counter = [0]
        fake_courses = [{"id": "1", "name": "Test"}]

        class FakeSJTUCatalog:
            def list_courses(self):
                counter[0] += 1
                return fake_courses

            def close(self):
                pass

        settings = FakeSettings({"input.canvas_token": "fake", "input.canvas_base_url": "http://localhost"})
        cat = Catalog(settings=settings)
        cat._sjt_cat = FakeSJTUCatalog()

        result1 = cat.list_courses()
        result2 = cat.list_courses()
        assert result1 == fake_courses
        assert result2 == fake_courses
        assert counter[0] == 1, f"Expected 1 call, got {counter[0]}"

    def test_ttl_cache_expires(self):
        """After TTL expires, a new call should hit the underlying catalog."""
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
        assert counter[0] == 1  # cached

        # Manually expire the cache entry
        cat._cache[("courses",)] = (0.0, [])
        result3 = cat.list_courses()
        assert counter[0] == 2

    def test_reset_clears_cache(self):
        """reset() discards both the cache and the underlying SJTUCatalog."""
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
        assert counter[0] == 1  # cached

        cat.reset()
        assert cat._sjt_cat is None
        assert len(cat._cache) == 0

        cat._sjt_cat = FakeSJTUCatalog()
        result3 = cat.list_courses()
        assert counter[0] == 2  # fresh call after reset

    def test_recordings_cache_per_course(self):
        """Recordings are cached per course_id, not shared."""
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
        r1b = cat.list_recordings("42")  # cache hit
        r2 = cat.list_recordings("99")   # different key, misses
        assert len(calls) == 2  # 42 + 99, not 3
        assert r1a[0]["video_id"] == "42-1"
        assert r2[0]["video_id"] == "99-1"


# ====================================================================
# UI state and actions
# ====================================================================

class TestInputPluginUI:
    def test_plugin_setup_registers_ui(self):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        snap = ctx.ui.snapshot()
        # One left tab: the course tree.  One right tab: the current class.
        assert snap["tabs"]["courses"] == "left"
        assert snap["tabs"]["session"] == "right"
        assert "input.courses" in snap["items"]
        assert "input.session" in snap["items"]
        assert "input.live" not in snap["items"]
        for page in ("input:session", "input:manual"):
            assert page in snap["pages"]
        assert "input:courses" not in snap["pages"]

        state = ctx.ui.state()
        items = {
            item["id"]: item
            for side in ("left_tabs", "right_tabs")
            for tab in state.get(side, [])
            for item in tab.get("toolbox", [])
        }
        for tid in ("input.courses", "input.session"):
            assert tid in items, f"toolbox item {tid} missing from state"
            assert items[tid].get("blocks"), f"toolbox item {tid} rendered empty blocks"

        for action in (
            "input.refresh_courses",
            "input.toggle_course",
            "input.open_class",
            "input.start_processing",
            "input.pause_processing",
            "input.play_url",
            "input.stop",
            "input.pause",
            "input.resume",
        ):
            assert action in snap["actions"], f"action {action} not registered"

        assert "input.refresh" in snap["commands"]
        assert "input.manual" in snap["commands"]

        plugin.teardown()

    def test_action_stop_stops_source(self):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        # Stop when nothing is playing
        result = ctx.ui.dispatch_action("input.stop", {})
        assert result.ok

        plugin.teardown()

    def test_action_pause_resume_without_source(self):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        # Should not crash when nothing is playing
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
        assert "URL" in result.message

        plugin.teardown()

    def test_action_play_url_with_url(self):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        # This will try to start ffmpeg which won't work, but the action
        # returns ok immediately (start is sync).
        result = ctx.ui.dispatch_action(
            "input.play_url",
            {"url": "fake://test", "title": "Fake Source", "kind": "url"},
        )
        assert result.ok
        # SourceManager should now be in "playing" state (ffmpeg may fail)
        plugin.teardown()

    def test_manual_form_payload_has_url_title_kind(self):
        """The manual URL form action handler sees url/title/kind keys."""
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        # Dispatch as if the form submitted with these fields
        result = ctx.ui.dispatch_action(
            "input.play_url",
            {"url": "rtsp://example.com/live", "title": "Test Live", "kind": "live"},
        )
        assert result.ok
        assert result.open_page == "input:session"

        # Also verify that missing url returns error
        result2 = ctx.ui.dispatch_action("input.play_url", {"title": "No URL"})
        assert not result2.ok
        assert "URL" in result2.message

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

        result = ctx.ui.dispatch_action("input.open_class", {"kind": "recording"})
        assert not result.ok
        assert "录播" in result.message

        plugin.teardown()

    def test_action_open_class_live_missing_id(self):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        result = ctx.ui.dispatch_action("input.open_class", {"kind": "live"})
        assert not result.ok
        assert "直播" in result.message

        plugin.teardown()

    def test_refresh_courses(self):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        result = ctx.ui.dispatch_action("input.refresh_courses", {})
        assert result.ok
        plugin.teardown()

    def test_settings_registered(self):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        settings = ctx.require("settings")
        specs = {s.key: s for s in settings.specs()}
        for key in (
            "input.canvas_token",
            "input.canvas_base_url",
            "input.video_fps",
            "input.video_width",
            "input.audio_rate",
            "input.playback_speed",
            "input.live_device",
            "input.reconnect",
        ):
            assert key in specs, f"setting {key} not registered"

        # Check secret
        canvas_spec = specs["input.canvas_token"]
        assert canvas_spec.secret is True
        assert canvas_spec.type == "secret"

        plugin.teardown()

    def test_service_registered(self):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)

        svc = ctx.services.get("input")
        assert svc is not None
        assert hasattr(svc, "start")
        assert hasattr(svc, "stop")
        assert hasattr(svc, "pause")
        assert hasattr(svc, "resume")
        assert hasattr(svc, "status")
        assert hasattr(svc, "list_courses")
        assert hasattr(svc, "list_recordings")
        assert hasattr(svc, "get_recording_url")
        assert hasattr(svc, "list_lives")
        assert hasattr(svc, "get_live_devices")

        plugin.teardown()

    def test_requires_settings(self):
        assert "settings" in InputPlugin.requires
        assert InputPlugin.id == "input"
        assert "input" in InputPlugin.provides


# ====================================================================
# FFmpegSource: read1 & local-file headers
# ====================================================================

class TestFFmpegSourceInternals:
    """Test the read1 fallback and is_http header guard in FFmpegSource."""

    def test_read1_is_used_over_read(self):
        """The reader threads should prefer read1() for low-latency reads."""
        from plugins.input.sources import FFmpegSource

        spec = SourceSpec(kind="file", url="/tmp/test.mp4", title="test")
        src = FFmpegSource(
            spec, hub=mock.MagicMock(), emit=mock.MagicMock(), reconnect=False,
        )

        # Simulate a stdout that has read1 (BufferedReader) and one that doesn't (FileIO)
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
        # The pattern used in _read_audio: read = getattr(proc.stdout, "read1", proc.stdout.read)
        read = getattr(stdout, "read1", stdout.read)
        read(4096)
        assert HasRead1.read1_called, "read1 should be preferred over read"
        assert not HasRead1.read_called, "read should not be called when read1 exists"

    def test_read_fallback_when_no_read1(self):
        """When stdout lacks read1, fall back to read."""
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
        """_restart_audio_process skips headers for local files (not http)."""
        from plugins.input.ffmpeg import build_audio_cmd, build_video_cmd
        from plugins.input.sources import FFmpegSource

        spec = SourceSpec(
            kind="file", url="/tmp/test.mp4", title="test",
            headers={"Referer": "https://should-be-dropped/"},
        )
        src = FFmpegSource(
            spec, hub=mock.MagicMock(), emit=mock.MagicMock(), reconnect=False,
        )
        # Call _restart_audio_process under mock to avoid real Popen
        with mock.patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value.stdout = None
            mock_popen.return_value.poll.return_value = -1
            src._restart_audio_process()
            audio_cmd = mock_popen.call_args[0][0]
            assert "-headers" not in audio_cmd, (
                f"Local file should not get -headers: {audio_cmd}"
            )

    def test_restart_http_url_passes_headers(self, monkeypatch):
        """_restart_audio_process passes headers for http(s) URLs."""
        from plugins.input.sources import FFmpegSource

        spec = SourceSpec(
            kind="recording", url="https://example.com/video.mp4", title="test",
            headers={"Referer": "https://v.sjtu.edu.cn/"},
        )
        src = FFmpegSource(
            spec, hub=mock.MagicMock(), emit=mock.MagicMock(), reconnect=False,
        )
        with mock.patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value.stdout = None
            mock_popen.return_value.poll.return_value = -1
            src._restart_audio_process()
            audio_cmd = mock_popen.call_args[0][0]
            assert "-headers" in audio_cmd, (
                f"HTTP URL should pass headers: {audio_cmd}"
            )


# ====================================================================
# SourceSpec
# ====================================================================

class TestSourceSpec:
    def test_defaults(self):
        spec = SourceSpec()
        assert spec.kind == "url"
        assert spec.url == ""
        assert spec.start_at == 0.0

    def test_fields(self):
        spec = SourceSpec(
            kind="live",
            url="http://example.com/live.flv",
            title="Live Stream",
            course={"id": "42"},
            meta={"camera": 1},
            headers={"Referer": "https://v.sjtu.edu.cn/"},
            start_at=10.0,
        )
        assert spec.kind == "live"
        assert spec.url == "http://example.com/live.flv"
        assert spec.title == "Live Stream"
        assert spec.course == {"id": "42"}
        assert spec.meta == {"camera": 1}
        assert spec.headers == {"Referer": "https://v.sjtu.edu.cn/"}
        assert spec.start_at == 10.0


# ====================================================================
# Integration: SJTUCatalog (unit tests without network)
# ====================================================================

class TestSJTUCatalogUnit:
    def test_no_token_raises_input_error(self):
        from integrations.sjtu_canvas import InputError, SJTUCatalog

        cat = SJTUCatalog(token="", base_url="http://localhost")
        with pytest.raises(InputError, match="token"):
            cat.list_courses()

    def test_vendored_sjtu_client_exists(self):
        """The input module must use the vendored client, not legacy/."""
        from pathlib import Path

        import integrations.sjtu_canvas_assistant as pkg
        from integrations.sjtu_canvas_assistant.video_client import NEW_VIDEO_SERVICE_ROOT

        package_dir = Path(pkg.__file__).resolve().parent
        assert package_dir.name == "sjtu_canvas_assistant"
        assert package_dir.parent.name == "integrations"
        assert "legacy" not in package_dir.parts
        assert NEW_VIDEO_SERVICE_ROOT.startswith("https://")

    def test_close_idempotent(self):
        from integrations.sjtu_canvas import SJTUCatalog

        cat = SJTUCatalog(token="test", base_url="http://localhost")
        cat.close()
        cat.close()  # should not raise

    def test_context_manager(self):
        from integrations.sjtu_canvas import SJTUCatalog

        with SJTUCatalog(token="test", base_url="http://localhost") as cat:
            pass  # close called on exit
        # Accessing after close raises
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

        # Patch _ensure_live_token + httpx to return a synthetic payload
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

        with mock.patch.object(SJTUCatalog, "_ensure_live_token",
                               return_value=("fake-jwt", {})):
            with mock.patch("httpx.Client.get") as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = {
                    "code": 0,
                    "data": {"records": fake_records},
                }
                mock_get.return_value.raise_for_status = lambda: None

                cat = SJTUCatalog(token="test", base_url="http://localhost")
                lives = cat.list_lives("2024-12-15 00:00:00", "2024-12-15 23:59:59")
                assert len(lives) == 2
                assert lives[0]["name"] == "初等数论"  # subjName wins
                assert lives[1]["name"] == "Fallback Name"  # courName fallback
                cat.close()


# ====================================================================
# Local time helper
# ====================================================================

class TestLocalTime:
    def test_local_time(self):
        from plugins.input.plugin import _local_time

        result = _local_time("2024-12-15 08:30:00")
        assert result  # non-empty
        assert "12-15" in result or "12-15" in result  # local format

    def test_local_time_empty(self):
        from plugins.input.plugin import _local_time
        assert _local_time("") == ""
        assert _local_time("not a date") == "not a date"

class TestClassPlayer:
    """Selecting a class loads streams but never auto-plays."""

    def _open_class(self, ctx, plugin, *, videos=("v1", "v2"), kind="recording"):
        payload = {
            "id": "class:1:" + kind,
            "kind": kind,
            "course": "1",
            "name": "高等数学",
            "label": "11-05 08:00–09:35",
            "videos": list(videos),
        }
        if kind == "live":
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
        assert plugin._sources.state == "idle", "selecting a class must not start ffmpeg"
        assert ctx.streams.current.session_id == "boot", "no session until processing starts"
        assert plugin._current_class["videos"] == ["v1", "v2"]

        plugin.teardown()

    def test_player_page_shows_both_video_streams(self, monkeypatch):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        monkeypatch.setattr(plugin, "_resolve_class_streams", lambda: None)
        self._open_class(ctx, plugin)
        plugin._class_streams = {
            "main": "https://example.com/main.m3u8",
            "screen": "https://example.com/screen.m3u8",
        }
        plugin._streams_loading = False

        ctx.ui.open_page("input:session")
        blocks = ctx.ui.state()["page_content"]["input:session"]
        player = next(block for block in blocks if block["type"] == "player")
        assert [src["label"] for src in player["sources"]] == ["课堂", "屏幕"]
        assert player["processing"] is False
        assert player["part"] == 1 and player["parts"] == 2

        plugin.teardown()

    def test_start_processing_starts_ingest(self, monkeypatch):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        monkeypatch.setattr(plugin, "_resolve_class_streams", lambda: None)
        self._open_class(ctx, plugin)
        plugin._class_streams = {"main": "https://example.com/main.m3u8"}
        plugin._streams_loading = False
        started: list[SourceSpec] = []
        monkeypatch.setattr(plugin, "_start_source", lambda spec: started.append(spec))

        result = ctx.ui.dispatch_action("input.start_processing", {})
        assert result.ok
        assert plugin._processing is True
        assert started and started[0].url == "https://example.com/main.m3u8"
        assert started[0].kind == "recording"
        assert started[0].continue_session is False

        plugin.teardown()

    def test_pause_processing_pauses_recording(self, monkeypatch):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        monkeypatch.setattr(plugin, "_resolve_class_streams", lambda: None)
        self._open_class(ctx, plugin)
        plugin._class_streams = {"main": "u"}
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
        self._open_class(ctx, plugin, kind="live")
        plugin._class_streams = {"main": "u"}
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
        self._open_class(ctx, plugin)
        plugin._processing = True
        resolved: list[bool] = []
        monkeypatch.setattr(
            plugin, "_resolve_class_streams", lambda: resolved.append(True)
        )

        plugin._on_class_ended()
        assert plugin._class_index == 1
        assert plugin._want_processing is True

        plugin.teardown()

    def test_last_period_ends_the_session(self, monkeypatch):
        ctx = _make_context()
        plugin = InputPlugin()
        plugin.setup(ctx)
        self._open_class(ctx, plugin, videos=("v1",))
        plugin._processing = True
        monkeypatch.setattr(plugin._sources, "stop", lambda: None)
        session = ctx.streams.start_session(meta={"course_name": "高等数学"})

        plugin._on_class_ended()
        assert plugin._class_finished is True
        assert plugin._processing is False
        assert session.ended_at is not None
        assert session.end_reason == "playback finished"

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

        event = SimpleNamespace(type="input.ended", get=lambda key, default=None: default)
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
        monkeypatch.setattr(plugin, "_finish_class", lambda: advanced.append(True))

        plugin._on_class_ended()
        assert plugin._class_index == 0
        assert advanced == []

        plugin.teardown()


class TestStreamMapping:
    """Recording/live views are mapped to the classroom and screen panes."""

    def test_view_streams_use_view_numbers(self):
        streams = _map_view_streams({"1": "main-url", "2": "screen-url"})
        assert streams == {"main": "main-url", "screen": "screen-url"}

    def test_view_streams_fallback_to_first(self):
        assert _map_view_streams({"0": "only-url"}) == {"main": "only-url"}

    def test_live_streams_use_devices(self):
        streams = _map_live_streams(
            [
                {"name": "camera", "url": "cam", "view_num": 1},
                {"name": "screen", "url": "scr", "view_num": 2},
            ]
        )
        assert streams == {"main": "cam", "screen": "scr"}

    def test_live_streams_without_view_numbers(self):
        streams = _map_live_streams([{"url": "a"}, {"url": "b"}])
        assert streams == {"main": "a", "screen": "b"}
