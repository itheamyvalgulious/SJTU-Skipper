from __future__ import annotations

from core.media import MediaKind, ProcessingMode
from core.streams import TextSource
from core.agent_types import FinishReason, GenerationErrorCode, IpcMethod

import base64
import json
import threading
import time
import io
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from core import (
    EventBus,
    MetaRegistry,
    PluginContext,
    ServiceRegistry,
    StreamHub,
    UIManager,
    VideoFrame,
)
from plugins.vlm.fusion import fuse_frames
from plugins.vlm.plugin import VlmPlugin, _fmt_time
from plugins.vlm.prompts import (
    BLANK_BOARD_MARKER,
    collapse_repetitions,
    is_trivial_board_text,
)
from tests.support import FakeLLM, FakeSettings


JPEG_QUALITY = 95

WIDTH = 640
HEIGHT = 360


def _synthetic_frame(lines: bool = True, black: bool = False) -> np.ndarray:
    if black:
        return np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    arr = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    if lines:
        arr[100, 50:200] = 220
        arr[200, 100:300] = 200
        for i in range(50):
            y = 150 + i
            x = 400 + i
            if y < HEIGHT and x < WIDTH:
                arr[y, x] = 255
    return arr


def _to_jpeg(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _make_video_frame(data: bytes, start: float, end: float) -> VideoFrame:
    return VideoFrame(index=0, start=start, end=end, data=data)


def _make_frames(
    n: int,
    *,
    offset_px: int = 0,
    fade: float = 0.0,
) -> list[VideoFrame]:
    frames: list[VideoFrame] = []
    for i in range(n):
        arr = _synthetic_frame(lines=True)
        if fade > 0:
            arr = (
                (arr.astype(np.float32) * (1.0 - fade * i))
                .clip(0, 255)
                .astype(np.uint8)
            )
        if offset_px > 0:
            arr = np.roll(arr, shift=offset_px * i, axis=1)
        jpeg = _to_jpeg(arr)
        frames.append(_make_video_frame(jpeg, float(i), float(i + 1)))
    return frames


class TestFusion:
    def test_single_frame_raises(self):
        arr = _synthetic_frame()
        jpeg = _to_jpeg(arr)
        frames = [_make_video_frame(jpeg, 0.0, 1.0)]
        with pytest.raises(ValueError, match="at least 3 frames"):
            fuse_frames(frames)

    def test_two_frames_raise(self):
        frames = _make_frames(2)
        with pytest.raises(ValueError, match="at least 3 frames"):
            fuse_frames(frames)

    def test_many_frames_output_non_empty(self):
        frames = _make_frames(20)
        result = fuse_frames(frames, max_width=320)
        assert isinstance(result, np.ndarray)
        assert result.size > 0
        assert result.ndim == 2
        assert result.max() > 0

    def test_empty_frames_raises(self):
        with pytest.raises(ValueError, match="at least 3 frames"):
            fuse_frames([])

    def test_crop_reduces_size(self):
        frames = _make_frames(10)
        full = fuse_frames(frames, max_width=0)
        cropped = fuse_frames(frames, crop=(0.2, 0.2, 0.8, 0.8), max_width=0)
        assert cropped.shape[0] < full.shape[0] or cropped.shape[1] < full.shape[1]

    def test_max_width_resizes(self):
        frames = _make_frames(10)
        result = fuse_frames(frames, max_width=320)
        assert result.shape[1] <= 320

    def test_max_width_zero_keeps_original(self):
        frames = _make_frames(10)
        result = fuse_frames(frames, max_width=0)
        assert result.shape[1] == WIDTH

    def test_black_frames_yield_dark_output(self):
        arr = _synthetic_frame(lines=False, black=True)
        jpeg = _to_jpeg(arr)
        frames = [_make_video_frame(jpeg, float(i), float(i + 1)) for i in range(10)]
        result = fuse_frames(frames)
        assert np.mean(result) < 30

    def test_corrupt_frame_raises(self):
        """Corrupt JPEGs must surface an error, not fall back silently."""
        good_arr = _synthetic_frame()
        good_jpeg = _to_jpeg(good_arr)
        frames = [
            _make_video_frame(good_jpeg, 0.0, 1.0),
            _make_video_frame(good_jpeg, 1.0, 2.0),
            _make_video_frame(b"not-a-jpeg", 2.0, 3.0),
        ]
        with pytest.raises(Exception):
            fuse_frames(frames)


class TestPrompts:
    def test_collapse_repeated_display_math(self):
        block = "$$\nT_m = \tilde{Q}_m^T A \tilde{Q}_m\n$$"
        text = "**02:28-02:46**\n\n" + "\n\n".join([block] * 8)
        out = collapse_repetitions(text)
        assert out.count("T_m =") == 1
        assert out.count("$$") == 2
        assert "02:28-02:46" in out

    def test_collapse_repeated_formula_group(self):
        group = "$$\nA = B\n$$\n\n$$\nB = C\n$$"
        text = "\n\n".join([group] * 5)
        out = collapse_repetitions(text)
        assert out.count("A = B") == 1
        assert out.count("B = C") == 1

    def test_collapse_keeps_non_repeated_content(self):
        text = "$$\nA = B\n$$\n\n$$\nC = D\n$$\n\n$$\nA = B\n$$"
        assert collapse_repetitions(text) == text


class TestTrivialBoardText:
    def test_blank_board_marker_is_not_trivial(self):
        # It is handled separately (== comparison); is_trivial_board_text
        # is for hallucinated placeholder-only outputs.
        assert not is_trivial_board_text(BLANK_BOARD_MARKER)

    def test_single_placeholder_is_trivial(self):
        assert is_trivial_board_text("![手绘示意图：复平面上的积分路径与区域]()")

    def test_multiple_placeholders_are_trivial(self):
        assert is_trivial_board_text("![图1]()\n![图2]()\n![区域示意图]()")

    def test_placeholder_with_surrounding_whitespace(self):
        assert is_trivial_board_text("\n  ![示意图]()\n\n")

    def test_real_text_with_placeholder_is_not_trivial(self):
        assert not is_trivial_board_text(
            "定理 1：设 $f$ 在 $D$ 上全纯。\n![区域示意图]()"
        )

    def test_latex_with_placeholder_is_not_trivial(self):
        assert not is_trivial_board_text(
            "$$\n\\int_{\\gamma} f(z) dz = 0\n$$\n\n![积分路径]()"
        )

    def test_plain_text_is_not_trivial(self):
        assert not is_trivial_board_text("设 $x \\in X$ 为拓扑空间中的点。")

    def test_empty_string_is_trivial(self):
        assert is_trivial_board_text("")
        assert is_trivial_board_text("   \n  ")


@pytest.fixture
def vlm_ctx(tmp_path: Path) -> PluginContext:
    bus = EventBus()
    bus.start()
    hub = StreamHub(
        emit=bus.emit,
        audio_emit_interval=100.0,
        video_emit_interval=100.0,
    )
    services = ServiceRegistry()
    fake_llm = FakeLLM()
    services.register("llm", fake_llm, plugin_id="<test>")
    fake_settings = FakeSettings()
    services.register("settings", fake_settings, plugin_id="<test>")
    ui = UIManager()

    ctx = PluginContext(
        plugin_id="vlm",
        data_dir=tmp_path,
        events=bus,
        streams=hub,
        llm=fake_llm,
        ui=ui,
        services=services,
        meta=MetaRegistry(),
        logger=logging.getLogger("test_vlm"),
    )
    return ctx


@pytest.fixture
def vlm_plugin(vlm_ctx: PluginContext) -> VlmPlugin:
    plugin = VlmPlugin()
    plugin.setup(vlm_ctx)
    plugin._stop_worker()
    plugin._stop_event.clear()
    yield plugin
    plugin.teardown()
    vlm_ctx.events.stop()


def test_loop_gap_continues_recording(vlm_plugin, vlm_ctx):
    from plugins.vlm.plugin import VLMRecognitionError

    settings = vlm_ctx.service("settings")
    settings.set("vlm.window_seconds", 5.0)
    settings.set("vlm.cadence_recording_seconds", 5.0)
    session = vlm_ctx.streams.current
    session.meta["mode"] = ProcessingMode.FAST
    for _ in range(10):
        session.video.append_frame(_to_jpeg(_synthetic_frame()), duration=1.0)
    session.input_done = True

    def recognize(start, end, frames):
        if start == 0:
            raise VLMRecognitionError(
                "Repeated board", code=GenerationErrorCode.THINKING_LOOP, attempts=3
            )
        return {"text": "后续板书 $x=2$"}

    vlm_plugin._run_window = recognize
    vlm_plugin._run_recording_session()
    assert [item.start for item in session.visual] == [5.0]
    assert not vlm_plugin._coverage(session).covers(MediaKind.VIDEO, 0, 5)
    assert session.visual_progress.final
    assert session.video_buffer_limiter.released == 10
    rows = [
        json.loads(line) for line in vlm_plugin.visual_path().read_text().splitlines()
    ]
    assert rows[0]["error"]["code"] == GenerationErrorCode.THINKING_LOOP
    assert "text" not in rows[0]
    assert list(vlm_plugin._history) == ["后续板书 $x=2$"]


def test_recording_eof_after_idle_finishes_tail(vlm_plugin, vlm_ctx):
    settings = vlm_ctx.service("settings")
    settings.set("vlm.window_seconds", 5.0)
    settings.set("vlm.cadence_recording_seconds", 5.0)
    session = vlm_ctx.streams.current
    session.meta["mode"] = ProcessingMode.FAST
    for _ in range(8):
        session.video.append_frame(_to_jpeg(_synthetic_frame()), duration=1.0)
    vlm_plugin._run_window = lambda start, end, frames: {"text": f"Board {start}"}
    vlm_plugin._run_recording_session()
    assert not session.visual_progress.final
    session.input_done = True
    vlm_plugin._run_recording_session()
    assert session.visual_progress.final
    assert session.visual_progress.before == 8.0
    assert [item.end for item in session.visual] == [5.0, 8.0]


@pytest.mark.parametrize(
    "code,attempts,max_attempts,recoverable",
    [
        (GenerationErrorCode.THINKING_LOOP, 3, 3, True),
        (GenerationErrorCode.THINKING_LOOP, 2, 3, False),
        (GenerationErrorCode.THINKING_LOOP, 2, 2, True),
        ("provider_error", 3, 3, False),
    ],
)
def test_structured_error_classification(
    vlm_plugin, vlm_ctx, code, attempts, max_attempts, recoverable
):
    from plugins.vlm.plugin import VLMRecognitionError

    vlm_ctx.service("settings").set("vlm.loop_max_attempts", max_attempts)

    vlm_ctx.service("llm").script = [
        {
            "content": "Failed partial board",
            "finish_reason": FinishReason.ERROR,
            "error": {
                "code": code,
                "attempts": attempts,
                "message": "Recognition failed",
            },
        }
    ]
    with pytest.raises(RuntimeError, match="Recognition failed") as caught:
        vlm_plugin._ask_image(b"jpeg", "00:00")
    assert isinstance(caught.value, VLMRecognitionError) is recoverable
    if recoverable:
        assert caught.value.code is code
        assert caught.value.attempts == attempts
    assert not vlm_plugin._history
    assert not vlm_ctx.streams.current.visual.items_since(0)


def test_token_limit_is_recoverable(vlm_plugin, vlm_ctx):
    from plugins.vlm.plugin import VLMRecognitionError

    vlm_ctx.service("llm").script = [
        {
            "content": "Truncated board",
            "finish_reason": FinishReason.LENGTH,
        }
    ]
    with pytest.raises(VLMRecognitionError, match="token limit") as caught:
        vlm_plugin._ask_image(b"jpeg", "00:00")
    assert caught.value.code is GenerationErrorCode.TOKEN_LIMIT
    assert not vlm_ctx.streams.current.visual.items_since(0)


def test_cancelled_loop_window_has_no_gap(vlm_plugin, vlm_ctx, monkeypatch):
    session = vlm_ctx.streams.current
    vlm_ctx.service("settings").set("vlm.fusion", False)

    def generate(method, **params):
        params["on_sent"]()
        vlm_plugin.cancel(session.session_id)
        return {
            "content": "Failed partial board",
            "finish_reason": FinishReason.ERROR,
            "error": {
                "code": GenerationErrorCode.THINKING_LOOP,
                "attempts": 3,
                "message": "Repeated board",
            },
        }

    monkeypatch.setattr(vlm_ctx.service("llm"), "call", generate)
    with pytest.raises(RuntimeError, match="VLM session cancelled"):
        vlm_plugin.process_window(0.0, 3.0, _make_frames(3))
    assert not vlm_plugin.visual_path().exists()
    assert not session.visual.items_since(0)
    assert not vlm_plugin._coverage(session).covers(MediaKind.VIDEO, 0.0, 3.0)
    assert not vlm_plugin._history


def test_gap_persistence_failure_propagates(vlm_plugin, vlm_ctx, monkeypatch):
    from plugins.vlm.plugin import VLMRecognitionError

    session = vlm_ctx.streams.current
    path = vlm_plugin.visual_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()

    def recognize(start, end, frames):
        raise VLMRecognitionError(
            "Repeated board", code=GenerationErrorCode.THINKING_LOOP, attempts=3
        )

    monkeypatch.setattr(vlm_plugin, "_run_window", recognize)
    with pytest.raises(IsADirectoryError):
        vlm_plugin.process_window(0.0, 3.0, _make_frames(3))
    assert not session.visual.items_since(0)
    assert session.video_buffer_limiter.released == 0.0
    assert vlm_plugin.status()["cursor"] == 0.0
    assert not vlm_plugin._coverage(session).covers(MediaKind.VIDEO, 0.0, 3.0)


def test_live_gap_continues_to_success(vlm_plugin, vlm_ctx):
    settings = vlm_ctx.service("settings")
    settings.set("vlm.window_seconds", 5.0)
    settings.set("vlm.cadence_live_seconds", 5.0)
    settings.set("vlm.fusion", False)
    session = vlm_ctx.streams.current
    session.video_buffer_limiter.configure(60.0)
    for _ in range(10):
        session.video.append_frame(_to_jpeg(_synthetic_frame()), duration=1.0)
    vlm_ctx.service("llm").script = [
        {
            "content": "Failed partial board",
            "finish_reason": FinishReason.ERROR,
            "error": {
                "code": GenerationErrorCode.THINKING_LOOP,
                "message": "Repeated board",
                "attempts": 3,
            },
        },
        {"content": "Complete next board", "finish_reason": FinishReason.STOP},
    ]
    failures = []
    segments = []
    vlm_ctx.events.on("input.error", lambda event: failures.append(dict(event.payload)))
    vlm_ctx.events.on("vlm.segment", lambda event: segments.append(dict(event.payload)))
    vlm_plugin._process_next_window()
    assert vlm_plugin.status()["cursor"] == 5.0
    assert session.video_buffer_limiter.released == 5.0
    assert not session.visual.items_since(0)
    assert not vlm_plugin._history
    assert not vlm_plugin._coverage(session).covers(MediaKind.VIDEO, 0.0, 5.0)
    vlm_plugin._process_next_window()
    session.input_done = True
    vlm_plugin._process_next_window()
    vlm_ctx.events.flush(timeout=2.0)
    assert [item.start for item in session.visual] == [5.0]
    assert session.visual_progress.final
    assert session.video_buffer_limiter.released == 10.0
    assert list(vlm_plugin._history) == ["Complete next board"]
    assert failures == []
    assert [segment["start"] for segment in segments] == [5.0]
    rows = [
        json.loads(line) for line in vlm_plugin.visual_path().read_text().splitlines()
    ]
    assert rows[0] == {
        "start": 0.0,
        "end": 5.0,
        "error": {
            "code": GenerationErrorCode.THINKING_LOOP,
            "message": "Repeated board",
            "attempts": 3,
        },
    }
    assert rows[1]["text"] == "Complete next board"


class TestVlmPlugin:
    def test_recording_session_commits_windows_in_order(
        self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext
    ):
        settings = vlm_ctx.service("settings")
        settings.set("vlm.enabled", True)
        settings.set("vlm.window_seconds", 5.0)
        settings.set("vlm.cadence_recording_seconds", 5.0)
        settings.set("vlm.fusion", False)

        session = vlm_ctx.streams.current
        session.meta["mode"] = ProcessingMode.FAST
        session.video_buffer_limiter.configure(60.0)
        for index in range(10):
            session.video.append_frame(_to_jpeg(_synthetic_frame()), duration=1.0)
        session.input_done = True

        # Bypass image fusion/LLM: return text derived from the window start.
        def run_window(start, end, frames):
            time.sleep(0.02 if start % 10 == 0 else 0.0)
            return {"image_path": "", "text": f"第{int(start)}秒板书"}

        vlm_plugin._run_window = run_window  # type: ignore[assignment]
        try:
            vlm_plugin._run_recording_session()
            items = list(session.visual)
            assert [item.start for item in items] == [0.0, 5.0]
            assert vlm_plugin.status()["cursor"] == 10.0
            assert session.video_buffer_limiter.released == 10.0
            assert vlm_plugin.busy() is False
        finally:
            vlm_plugin.teardown()

    def test_recording_uses_cadence_and_flushes_tail(
        self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext
    ):
        # Drive the pipeline synchronously; the background worker would race
        # this test's direct call and duplicate windows.
        vlm_plugin._stop_worker()
        vlm_plugin._stop_event.clear()
        settings = vlm_ctx.service("settings")
        settings.set("vlm.enabled", True)
        settings.set("vlm.window_seconds", 20.0)
        settings.set("vlm.cadence_recording_seconds", 300.0)
        settings.set("vlm.fusion", False)

        session = vlm_ctx.streams.current
        session.meta["mode"] = ProcessingMode.FAST
        jpeg = _to_jpeg(_synthetic_frame())
        for _ in range(1300):
            session.video.append_frame(jpeg, duration=1.0)
        session.input_done = True

        windows: list[tuple[float, float]] = []

        def run_window(start, end, frames):
            windows.append((start, end))
            return {"image_path": "", "text": f"第{int(start)}秒板书"}

        vlm_plugin._run_window = run_window  # type: ignore[assignment]
        try:
            vlm_plugin._run_recording_session()
        finally:
            vlm_plugin.teardown()

        assert [start for start, _end in windows] == [
            0.0,
            300.0,
            600.0,
            900.0,
            1200.0,
            1280.0,
        ]
        assert windows[-1][1] == 1300.0
        assert vlm_plugin.busy() is False

    def test_idle_recording_session_does_not_hold_the_plugin_lock(
        self, vlm_ctx: PluginContext
    ):
        """Idling on a finished recording must release the lock ``status()`` needs."""

        vlm_ctx.service("settings").set("vlm.enabled", False)
        plugin = VlmPlugin()
        plugin.setup(vlm_ctx)
        session = vlm_ctx.streams.current
        session.meta["mode"] = ProcessingMode.FAST
        session.ended_at = 1.0
        session.visual_progress.final = True
        plugin._recording_sessions.add(session.session_id)

        entered = threading.Event()
        release = threading.Event()

        def fake_wait(_timeout: float | None = None) -> bool:
            entered.set()
            release.wait(2.0)
            return False

        plugin._stop_event.wait = fake_wait  # type: ignore[method-assign]
        thread = threading.Thread(target=plugin._run_recording_session, daemon=True)
        thread.start()
        try:
            assert entered.wait(1.0), "the worker never reached the idle wait"
            acquired = plugin._lock.acquire(timeout=0.3)
            assert acquired, "the idle wait must not hold the plugin lock"
            if acquired:
                plugin._lock.release()
        finally:
            release.set()
            thread.join(2.0)
            plugin.teardown()

    def test_process_window_with_fake_llm(
        self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext
    ):
        settings = vlm_ctx.service("settings")
        settings.set("vlm.enabled", True)
        settings.set("vlm.model", "deepseek-flash")
        settings.set("vlm.fusion", False)

        fake_llm = vlm_ctx.service("llm")
        fake_llm.script = [
            {
                "content": "定理 1：设 $f$ 在 $D$ 上全纯。",
                "finish_reason": FinishReason.STOP,
                "model": "board-model",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }
        ]

        arr = _synthetic_frame()
        jpeg = _to_jpeg(arr)
        frames = [_make_video_frame(jpeg, 10.0, 11.0)]

        segments: list[dict] = []

        def on_segment(event):
            segments.append(dict(event.payload))

        vlm_ctx.events.on("vlm.segment", on_segment)

        result = vlm_plugin.process_window(10.0, 11.0, frames)
        vlm_ctx.events.flush(timeout=2.0)

        assert result.get("ok") is True
        visual = vlm_ctx.streams.current.visual
        assert len(visual) >= 1
        item = visual[-1]
        assert item.source == "vlm"
        assert "定理 1" in item.text
        assert item.start == 10.0
        assert item.end == 11.0
        assert "image_path" in item.meta
        assert item.meta["model"] == "board-model"
        assert item.meta["usage"]["total_tokens"] == 30

        log = (
            vlm_ctx.data_dir
            / "sessions"
            / vlm_ctx.streams.current.session_id
            / "visual.jsonl"
        )
        records = [json.loads(line) for line in log.read_text("utf-8").splitlines()]
        assert records[-1]["start"] == 10.0
        assert records[-1]["end"] == 11.0
        assert "定理 1" in records[-1]["text"]

        method, last_call = fake_llm.calls[-1]
        assert method == IpcMethod.GENERATE
        assert last_call["role"] == "board_recognition"
        assert "model" not in last_call
        assert "extra" not in last_call

        assert len(segments) >= 1
        seg = segments[0]
        assert seg["start"] == 10.0
        assert seg["end"] == 11.0

    def test_full_reply(self, vlm_plugin, vlm_ctx):
        from html import unescape

        text = "\n\n".join(f"公式 {i}: $x_{{{i}}} = {i}$" for i in range(50))
        vlm_plugin._commit_window(0, 18, {"text": text})
        vlm_plugin._commit_window(20, 38, {"text": text})
        assert vlm_plugin.status()["last_window"]["text"] == text
        blocks = vlm_plugin._render_board_page()
        reply = next(b for b in blocks if b["type"] == "html")
        assert text in unescape(reply["html"])
        assert len(vlm_ctx.streams.current.visual) == 1

    def test_process_window_skips_blank_board(
        self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext
    ):
        settings = vlm_ctx.service("settings")
        settings.set("vlm.fusion", False)

        fake_llm = vlm_ctx.service("llm")
        fake_llm.script = [BLANK_BOARD_MARKER]

        arr = _synthetic_frame(lines=False, black=True)
        jpeg = _to_jpeg(arr)
        frames = [_make_video_frame(jpeg, 5.0, 6.0)]

        visual_before = len(vlm_ctx.streams.current.visual)
        result = vlm_plugin.process_window(5.0, 6.0, frames)

        assert result.get("ok") is True
        assert len(vlm_ctx.streams.current.visual) == visual_before

    def test_process_window_skips_placeholder_only(
        self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext
    ):
        settings = vlm_ctx.service("settings")
        settings.set("vlm.fusion", False)

        fake_llm = vlm_ctx.service("llm")
        fake_llm.script = ["![手绘示意图：复平面上的积分路径与区域]()"]

        arr = _synthetic_frame(lines=False, black=True)
        jpeg = _to_jpeg(arr)
        frames = [_make_video_frame(jpeg, 5.0, 6.0)]

        visual_before = len(vlm_ctx.streams.current.visual)
        result = vlm_plugin.process_window(5.0, 6.0, frames)

        assert result.get("ok") is True
        assert len(vlm_ctx.streams.current.visual) == visual_before

    def test_process_window_keeps_changed_formula(self, vlm_plugin, vlm_ctx):
        vlm_ctx.service("settings").set("vlm.fusion", False)
        repeated = "$$\n\\begin{aligned}\na &= b \\\\\nc &= d\n\\end{aligned}\n$$"
        old = "$x_1 + x_2 + x_3 = y_1 + y_2$"
        changed = "$x_1 + x_2 + x_3 = y_1 - y_2$"
        fake_llm = vlm_ctx.service("llm")
        fake_llm.script = [
            f"{repeated}\n\n{old}",
            f"{repeated}\n\n{changed}",
            f"{repeated}\n\n{changed}",
        ]

        jpeg = _to_jpeg(_synthetic_frame())
        for start in (0.0, 1.0, 2.0):
            frames = [_make_video_frame(jpeg, start, start + 1)]
            vlm_plugin.process_window(start, start + 1, frames)

        visual = vlm_ctx.streams.current.visual
        assert [item.text for item in visual] == [f"{repeated}\n\n{old}", changed]

    def test_truncated_response_becomes_a_gap(self, vlm_plugin, vlm_ctx):
        settings = vlm_ctx.service("settings")
        settings.set("vlm.fusion", False)
        fake_llm = vlm_ctx.service("llm")
        fake_llm.script = [
            {"content": "Incomplete formula", "finish_reason": FinishReason.LENGTH}
        ]
        session = vlm_ctx.streams.current
        frames = _make_frames(3)
        result = vlm_plugin.process_window(0.0, 3.0, frames)
        assert result["ok"] is False
        assert result["code"] == GenerationErrorCode.TOKEN_LIMIT
        # The truncated board is not committed; the window is left uncovered
        # so a later pass may retry it.
        assert not session.visual.items_since(0)
        assert not vlm_plugin._coverage(session).covers(MediaKind.VIDEO, 0.0, 3.0)
        row = json.loads(vlm_plugin.visual_path().read_text().splitlines()[0])
        assert row["error"]["code"] == GenerationErrorCode.TOKEN_LIMIT

    @pytest.mark.parametrize("cancel_before_sent", [False, True])
    def test_cancel_during_submission(
        self, vlm_plugin, vlm_ctx, monkeypatch, cancel_before_sent
    ):
        session = vlm_ctx.streams.current
        fake_llm = vlm_ctx.service("llm")
        cancelled = []
        monkeypatch.setattr(fake_llm, "cancel", cancelled.append)

        def generate(method, **params):
            assert method == IpcMethod.GENERATE
            if cancel_before_sent:
                vlm_plugin.cancel(session.session_id)
                assert not cancelled
            params["on_sent"]()
            if not cancel_before_sent:
                vlm_plugin.cancel(session.session_id)
            assert cancelled == [params["session_id"]]
            return {"content": "Cancelled board", "finish_reason": FinishReason.STOP}

        monkeypatch.setattr(fake_llm, "call", generate)
        with pytest.raises(RuntimeError, match="VLM session cancelled"):
            vlm_plugin._ask_image(b"jpeg", "00:00")
        assert not vlm_plugin._requests
        assert not vlm_plugin._submitted_requests
        assert not session.visual.items_since(0)

    def test_recognition_does_not_receive_previous_transcription(
        self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext
    ):
        settings = vlm_ctx.service("settings")
        settings.set("vlm.fusion", False)

        fake_llm = vlm_ctx.service("llm")
        fake_llm.script = ["定理 A：$x=1$。", "新增：定理 B：$y=2$。"]

        arr = _synthetic_frame()
        jpeg = _to_jpeg(arr)
        vlm_plugin.process_window(0.0, 1.0, [_make_video_frame(jpeg, 0.0, 1.0)])
        vlm_plugin.process_window(5.0, 6.0, [_make_video_frame(jpeg, 5.0, 6.0)])

        assert len(fake_llm.calls) == 2
        instruction, input_message = fake_llm.calls[1][1]["messages"]
        assert instruction["role"] == "system"
        evidence, image = input_message["content"]
        assert input_message["role"] == "user"
        assert image["type"] == "image_url"
        assert evidence == {"type": "text", "text": "采样时间：00:05 - 00:06"}
        assert "定理 A" not in instruction["content"]

    def test_llm_error_emits_vlm_error(
        self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext
    ):
        settings = vlm_ctx.service("settings")
        settings.set("vlm.fusion", False)

        fake_llm = vlm_ctx.service("llm")

        def _fail(_kw):
            raise RuntimeError("simulated LLM failure")

        fake_llm.script = [_fail]

        errors: list[dict] = []

        def on_error(event):
            errors.append(dict(event.payload))

        vlm_ctx.events.on("vlm.error", on_error)

        arr = _synthetic_frame()
        jpeg = _to_jpeg(arr)
        frames = [_make_video_frame(jpeg, 0.0, 1.0)]
        with pytest.raises(RuntimeError, match="simulated LLM failure"):
            vlm_plugin.process_window(0.0, 1.0, frames)
        vlm_ctx.events.flush(timeout=1.0)

        assert len(errors) >= 1
        assert "simulated LLM failure" in errors[0].get("error", "")

    def test_selection_failure_has_no_session(self, vlm_plugin, vlm_ctx):
        failures = []
        vlm_ctx.events.on(
            "vlm.error", lambda event: failures.append(dict(event.payload))
        )

        def fail():
            vlm_plugin._stop_event.set()
            raise RuntimeError("selection failed")

        vlm_plugin._select_session = fail
        vlm_plugin._flush_remaining = lambda: None
        vlm_plugin._worker_loop()
        vlm_ctx.events.flush(timeout=1.0)
        assert failures[0]["session_id"] == ""
        assert not vlm_plugin._cancelled_sessions

    def test_worker_failure_releases_input(self, vlm_plugin, vlm_ctx):
        session = vlm_ctx.streams.start_session(
            "worker-failure", meta={"mode": ProcessingMode.FAST}
        )
        session.input_done = True
        session.video_buffer_limiter.configure(1.0)
        failures = []
        vlm_ctx.events.on(
            "input.error", lambda event: failures.append(dict(event.payload))
        )
        vlm_plugin._select_session = lambda: session

        def fail():
            vlm_plugin._stop_event.set()
            raise RuntimeError("model unavailable")

        vlm_plugin._run_recording_session = fail
        vlm_plugin._worker_loop()
        vlm_ctx.events.flush(timeout=1.0)
        assert not session.video_buffer_limiter.wait(10.0)
        assert not vlm_plugin.busy()
        assert failures[0]["session_id"] == session.session_id
        assert "model unavailable" in failures[0]["message"]


class TestUiState:
    def test_render_with_visual_items(
        self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext
    ):
        visual = vlm_ctx.streams.current.visual
        visual.append(
            "定理 1：测试内容。",
            start=0.0,
            end=10.0,
            source=TextSource.VLM,
            meta={"image_path": "/tmp/test.jpg", "model": "test"},
        )
        page = vlm_ctx.ui._pages.get("vlm:board")
        blocks = vlm_ctx.ui._render(page.render, "test")
        markdown_blocks = [b for b in blocks if b.get("type") == "markdown"]
        assert len(markdown_blocks) >= 1
        assert "定理 1" in markdown_blocks[0]["text"]


class TestHelpers:
    def test_fmt_time(self):
        assert _fmt_time(0) == "00:00"
        assert _fmt_time(65.5) == "01:05"
        assert _fmt_time(3661.0) == "61:01"


def test_prepare_overlaps_recognition(vlm_plugin, vlm_ctx, monkeypatch):
    from plugins.vlm.plugin import PreparedWindow

    vlm_plugin._stop_worker()
    vlm_plugin._stop_event.clear()
    settings = vlm_ctx.service("settings")
    settings.set("vlm.window_seconds", 5.0)
    settings.set("vlm.cadence_recording_seconds", 5.0)
    session = vlm_ctx.streams.current
    session.meta["mode"] = ProcessingMode.FAST
    session.video_buffer_limiter.configure(0)
    for _ in range(10):
        session.video.append_frame(_to_jpeg(_synthetic_frame()), duration=1)
    session.input_done = True
    vlm_plugin._processing_session = session
    prepared_next = threading.Event()
    recognition_started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    recognized = []

    def prepare(session, start, end, frames):
        if start == 5:
            prepared_next.set()
        return PreparedWindow(start, end, Path("board.jpg"), b"jpeg")

    def recognize(prepared):
        recognized.append(prepared.start)
        recognition_started.set()
        if prepared.start == 0:
            assert prepared_next.wait(2)
            assert release.wait(2)
        return {"text": f"Definition at {prepared.start}", "image_path": ""}

    monkeypatch.setattr(vlm_plugin, "_prepare_window", prepare)
    monkeypatch.setattr(vlm_plugin, "_recognize_window", recognize)

    def run():
        try:
            vlm_plugin._run_recording_session()
        finally:
            finished.set()

    worker = threading.Thread(target=run)
    vlm_plugin._worker = worker
    worker.start()
    try:
        assert prepared_next.wait(2)
        assert recognition_started.wait(2)
        assert recognized == [0.0]
        release.set()
        assert finished.wait(2)
        assert recognized == [0.0, 5.0]
        assert [item.text for item in session.visual] == [
            "Definition at 0.0",
            "Definition at 5.0",
        ]
    finally:
        release.set()
        worker.join(2)


def test_idle_session_does_not_block_recording():
    hub = StreamHub()
    old = hub.current
    current = hub.start_session("recording", meta={"mode": ProcessingMode.FAST})
    plugin = VlmPlugin()
    plugin._ctx = SimpleNamespace(streams=hub)
    plugin._sessions = {old.session_id: old, current.session_id: current}
    plugin._recording_cursor = {}
    plugin._last_session = None
    plugin._session_history = {}
    plugin._window_seconds = lambda: 20.0
    plugin._history_windows = lambda: 3
    current.video_input_done = True
    assert plugin._select_session() is current


def test_cancel_releases_preparation_pool(vlm_plugin, vlm_ctx):
    session = vlm_ctx.streams.start_session("preparation-cancel")
    assert vlm_ctx.events.flush(timeout=1)
    vlm_plugin._processing_session = session
    waiting = threading.Event()
    wait_for = session.video.wait_for

    def wait(end, timeout):
        waiting.set()
        return wait_for(end, timeout=timeout)

    with patch.object(session.video, "wait_for", side_effect=wait):
        prepared = vlm_plugin._prepare_pool.submit(
            vlm_plugin._prepare_next_window, session, 0, 20
        )
        assert waiting.wait(3)
        vlm_plugin.cancel(session.session_id)
        assert prepared.result(timeout=3) is None
        assert (
            vlm_plugin._prepare_pool.submit(lambda: "available").result(timeout=3)
            == "available"
        )


def test_same_id_reprocess_resets_board(vlm_plugin, vlm_ctx):
    plugin = vlm_plugin
    old = vlm_ctx.streams.start_session("same-class")
    vlm_ctx.events.flush(timeout=1)
    frames = [_make_video_frame(_to_jpeg(_synthetic_frame()), 0, 2)]
    plugin._run_window = lambda *args: {
        "text": "Complete board theorem",
        "image_path": "",
    }
    plugin.process_window(0, 2, frames)
    plugin._last_session = old.session_id
    plugin._recording_cursor[old.session_id] = 100
    plugin.cancel(old.session_id)
    for name in ("visual.jsonl", "coverage.json"):
        (vlm_ctx.data_dir / "sessions" / old.session_id / name).unlink()

    session = vlm_ctx.streams.start_session(old.session_id)
    vlm_ctx.events.flush(timeout=1)
    assert old.session_id not in plugin._cancelled_sessions
    assert old.session_id not in plugin._recording_cursor
    assert plugin.status()["cursor"] == 0
    assert plugin._coverage(session).covered_seconds(MediaKind.VIDEO) == 0
    plugin.process_window(0, 2, frames)
    assert [item.text for item in session.visual] == ["Complete board theorem"]
    rows = [json.loads(line) for line in plugin.visual_path().read_text().splitlines()]
    assert [row["text"] for row in rows] == ["Complete board theorem"]


def test_recording_skips_already_covered_windows(vlm_plugin, vlm_ctx):
    settings = vlm_ctx.service("settings")
    settings.set("vlm.enabled", True)
    settings.set("vlm.window_seconds", 5.0)
    settings.set("vlm.cadence_recording_seconds", 5.0)
    settings.set("vlm.fusion", False)

    session = vlm_ctx.streams.current
    session.meta["mode"] = ProcessingMode.FAST
    session.video_buffer_limiter.configure(60.0)
    for _ in range(10):
        session.video.append_frame(_to_jpeg(_synthetic_frame()), duration=1.0)
    session.input_done = True
    # A previous pass already recognized the first window.
    vlm_plugin._coverage(session).mark(MediaKind.VIDEO, 0.0, 5.0)

    calls: list[tuple[float, float]] = []

    def run_window(start, end, frames):
        calls.append((start, end))
        return {"image_path": "", "text": f"第{int(start)}秒板书"}

    vlm_plugin._run_window = run_window
    vlm_plugin._run_recording_session()

    assert calls == [(5.0, 10.0)]
    assert [item.start for item in session.visual] == [5.0]


def test_status_cursor_is_scoped_to_current_session(vlm_plugin, vlm_ctx):
    old = vlm_ctx.streams.current
    vlm_plugin._last_session = old.session_id
    vlm_plugin._cursor = 6593.6

    new = vlm_ctx.streams.start_session(
        "next-class", meta={"mode": ProcessingMode.FAST}
    )
    assert vlm_ctx.events.flush(timeout=1)

    # The worker has not switched yet (no window of video available), but the
    # new session's progress must not inherit the previous class's cursor.
    status = vlm_plugin.status()
    assert status["session_id"] == new.session_id
    assert status["cursor"] == 0.0

    with vlm_plugin._lock:
        vlm_plugin._last_session = new.session_id
        vlm_plugin._cursor = 30.0
    assert vlm_plugin.status()["cursor"] == 30.0


def test_same_id_fill_preserves_board(vlm_plugin, vlm_ctx):
    old = vlm_ctx.streams.start_session("fill-class")
    vlm_ctx.events.flush(timeout=1)
    vlm_plugin._commit_window(0, 2, {"text": "Valid board theorem"})
    session = vlm_ctx.streams.start_session(
        old.session_id, meta={"fill_for": old.session_id}
    )
    vlm_ctx.events.flush(timeout=1)
    assert vlm_plugin._coverage(session).covers(MediaKind.VIDEO, 0, 2)
    assert "Valid board theorem" in vlm_plugin.visual_path().read_text()


@pytest.mark.parametrize("prepare_ahead", [0, 1])
def test_sparse_recording_advances_bounded_input(vlm_plugin, vlm_ctx, prepare_ahead):
    settings = vlm_ctx.service("settings")
    settings.set("vlm.window_seconds", 20.0)
    settings.set("vlm.cadence_recording_seconds", 300.0)
    settings.set("vlm.prepare_ahead", prepare_ahead)
    session = vlm_ctx.streams.start_session(
        "sparse", meta={"mode": ProcessingMode.FAST}
    )
    assert vlm_ctx.events.flush(timeout=1)
    session.video_buffer_limiter.configure(vlm_plugin._buffer_limit())
    windows = []

    def recognize(start, end, frames):
        windows.append((start, end))
        return {"text": f"Board at {start}", "image_path": ""}

    vlm_plugin._run_window = recognize

    def produce():
        for _ in range(940):
            if not session.video_buffer_limiter.wait(session.video.duration):
                return
            session.video.append_frame(b"frame", duration=1.0)
        session.video_input_done = True

    producer = threading.Thread(target=produce)
    producer.start()
    peak_buffered = 0.0
    deadline = time.monotonic() + 3
    try:
        while not session.visual_progress.final and time.monotonic() < deadline:
            available = session.video.duration
            peak_buffered = max(peak_buffered, session.video.chunk_count)
            if vlm_plugin._select_session() is session:
                vlm_plugin._processing_session = session
                vlm_plugin._run_recording_session()
            else:
                session.video.wait_for(available + 1, timeout=0.01)
        assert session.visual_progress.final
        assert windows == [(0, 20), (300, 320), (600, 620), (900, 920), (920, 940)]
        assert session.video_buffer_limiter.limit == (1 + prepare_ahead) * 20
        assert peak_buffered <= session.video_buffer_limiter.limit + 21
    finally:
        session.video_buffer_limiter.close()
        producer.join(1)
