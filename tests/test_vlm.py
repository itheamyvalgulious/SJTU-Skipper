"""Tests for the VLM plugin: fusion, prompts, deduplication, worker, UI."""

from __future__ import annotations

import base64
import json
import threading
import time
import difflib
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
from llm.types import ImagePart, TextPart
from plugins.vlm.fusion import fuse_frames
from plugins.vlm.plugin import VlmPlugin, _fmt_time
from plugins.vlm.prompts import (
    BLANK_BOARD_MARKER,
    BOARD_RECOGNITION_PROMPT,
    NO_CHANGE_MARKER,
    build_user_message,
    collapse_repetitions,
    is_trivial_board_text,
)
from tests.support import FakeLLM, FakeSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

JPEG_QUALITY = 95

WIDTH = 640
HEIGHT = 360


def _synthetic_frame(lines: bool = True, black: bool = False) -> np.ndarray:
    """Create a synthetic grayscale frame (black background with white lines)."""
    if black:
        return np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    arr = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    if lines:
        # White horizontal lines
        arr[100, 50:200] = 220
        arr[200, 100:300] = 200
        # White diagonal
        for i in range(50):
            y = 150 + i
            x = 400 + i
            if y < HEIGHT and x < WIDTH:
                arr[y, x] = 255
    return arr


def _to_jpeg(arr: np.ndarray) -> bytes:
    """Encode a grayscale numpy array as JPEG bytes."""
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
    """Create n synthetic frames with optional shifting and fading."""
    frames: list[VideoFrame] = []
    for i in range(n):
        arr = _synthetic_frame(lines=True)
        if fade > 0:
            arr = (arr.astype(np.float32) * (1.0 - fade * i)).clip(0, 255).astype(np.uint8)
        if offset_px > 0:
            arr = np.roll(arr, shift=offset_px * i, axis=1)
        jpeg = _to_jpeg(arr)
        frames.append(_make_video_frame(jpeg, float(i), float(i + 1)))
    return frames


# ---------------------------------------------------------------------------
# Fusion tests
# ---------------------------------------------------------------------------

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
        assert result.max() > 0  # white lines should survive

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

    def test_no_align_still_works(self):
        frames = _make_frames(15)
        result = fuse_frames(frames, align=False)
        assert result.size > 0

    def test_black_frames_yield_dark_output(self):
        """All-black frames should produce a dark fused result."""
        arr = _synthetic_frame(lines=False, black=True)
        jpeg = _to_jpeg(arr)
        frames = [_make_video_frame(jpeg, float(i), float(i + 1)) for i in range(10)]
        result = fuse_frames(frames)
        assert np.mean(result) < 30  # mostly dark

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


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_prompt_contains_latex_rules(self):
        prompt = BOARD_RECOGNITION_PROMPT.replace("{timestamp}", "00:00-01:00")
        assert "$" in prompt
        assert "$$" in prompt
        assert "\\[" in prompt  # forbidden mention
        assert "\\]" in prompt

    def test_prompt_contains_diagram_placeholder(self):
        prompt = BOARD_RECOGNITION_PROMPT.replace("{timestamp}", "00:00")
        assert "![" in prompt
        assert "]()" in prompt

    def test_prompt_contains_blank_board_marker(self):
        prompt = BOARD_RECOGNITION_PROMPT.replace("{timestamp}", "00:00")
        assert BLANK_BOARD_MARKER in prompt

    def test_prompt_no_greeting(self):
        prompt = BOARD_RECOGNITION_PROMPT.replace("{timestamp}", "00:00")
        assert "寒暄" in prompt or "问候" in prompt  # instructs to avoid them


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

    def test_build_user_message_structure(self):
        content = build_user_message(timestamp="00:30", image_data_url="data:image/jpeg;base64,abc")
        assert isinstance(content, list)
        assert len(content) == 2
        assert isinstance(content[0], TextPart)
        assert isinstance(content[1], ImagePart)
        assert "00:30" in content[0].text
        assert content[1].url == "data:image/jpeg;base64,abc"

    def test_build_user_message_no_image_url(self):
        content = build_user_message(timestamp="00:00", image_data_url="")
        assert len(content) == 1
        assert isinstance(content[0], TextPart)

    def test_first_window_asks_for_full_transcription(self):
        content = build_user_message(timestamp="00:00")
        text = content[0].text
        assert "第一次识别" in text
        assert "请完整转录当前黑板" in text
        assert "【无变化】" not in text

    def test_history_is_passed_as_text_and_asks_for_delta(self):
        content = build_user_message(
            timestamp="10:00",
            history=["第一版板书 A", "第二版板书 B"],
        )
        text = content[0].text
        assert "第一版板书 A" in text
        assert "第二版板书 B" in text
        assert "新增、修改或擦除" in text
        assert NO_CHANGE_MARKER in text


class TestTrivialBoardText:
    def test_blank_board_marker_is_not_trivial(self):
        """The explicit blank marker itself should not be classified as trivial."""
        # It is handled separately (== comparison); is_trivial_board_text
        # is for hallucinated placeholder-only outputs.
        assert not is_trivial_board_text(BLANK_BOARD_MARKER)

    def test_single_placeholder_is_trivial(self):
        assert is_trivial_board_text("![手绘示意图：复平面上的积分路径与区域]()")

    def test_multiple_placeholders_are_trivial(self):
        assert is_trivial_board_text(
            "![图1]()\n![图2]()\n![区域示意图]()"
        )

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


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

class TestDedupe:
    def test_identical_text_high_ratio(self):
        text = "设 $x \\in X$ 为拓扑空间中的点。\n\n$$\n\\ker \\partial_n = \\operatorname{im} \\partial_{n+1}\n$$\n"
        ratio = difflib.SequenceMatcher(None, text, text).ratio()
        assert ratio == 1.0

    def test_different_text_low_ratio(self):
        a = "设 $x \\in X$。"
        b = "证明：对于任意 $\\epsilon > 0$，存在 $\\delta > 0$ 使得..."
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        assert ratio < 0.92

    def test_threshold_default(self):
        """Default threshold 0.92 blocks similar but skips different."""
        # Two very similar texts
        a = "定理 1.1 设 $f$ 在 $D$ 上全纯，则 $f$ 在 $D$ 上有无穷阶导数。"
        b = "定理 1.1 设 $f$ 在 $D$ 上全纯，则 $f$ 在 $D$ 上有无穷阶导数。"  # identical
        assert difflib.SequenceMatcher(None, a, b).ratio() >= 0.92

        c = "这是完全不同的内容，包含了新的定义和引理，涉及复分析中的留数定理。"
        assert difflib.SequenceMatcher(None, a, c).ratio() < 0.92


# ---------------------------------------------------------------------------
# Plugin / Worker tests
# ---------------------------------------------------------------------------

@pytest.fixture
def vlm_ctx(tmp_path: Path) -> PluginContext:
    """Build a minimal PluginContext with fake settings, LLM, and streams."""
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

    class LLMProxy:
        def __init__(self, svc):
            self._svc = svc
        def _service(self):
            return self._svc.get("llm")
        def chat(self, **kw):
            return self._service().chat(**kw)
        def stream(self, **kw):
            return self._service().stream(**kw)
        def config_for(self, profile):
            return self._service().config_for(profile)
        def info(self):
            return self._service().info()

    llm_proxy = LLMProxy(services)

    ctx = PluginContext(
        plugin_id="vlm",
        data_dir=tmp_path,
        events=bus,
        streams=hub,
        llm=llm_proxy,
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
    return plugin


class TestVlmPlugin:
    def test_plugin_id_and_requires(self):
        assert VlmPlugin.id == "vlm"
        assert "settings" in VlmPlugin.requires
        assert "llm" in VlmPlugin.requires

    def test_setup_registers_service(self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext):
        assert vlm_ctx.service("vlm") is not None

    def test_setup_creates_debug_page_only(self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext):
        snap = vlm_ctx.ui.snapshot()
        assert "classroom" not in snap["tabs"]
        assert "vlm.board" not in snap["items"]
        assert "vlm:board" in snap["pages"]
        assert "vlm.debug" in snap["commands"]

    def test_status_returns_dict(self, vlm_plugin: VlmPlugin):
        status = vlm_plugin.status()
        assert "cursor" in status
        assert "session_id" in status
        assert "worker_alive" in status

    def test_settings_specs_registered(self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext):
        settings = vlm_ctx.service("settings")
        assert settings is not None
        # vlm.enabled should be registered
        spec_keys = [s.key for s in settings.specs()]
        assert "vlm.enabled" in spec_keys
        assert "vlm.model" in spec_keys
        assert "vlm.window_seconds" in spec_keys

    def test_recording_session_commits_windows_in_order(
        self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext
    ):
        settings = vlm_ctx.service("settings")
        settings.set("vlm.enabled", True)
        settings.set("vlm.window_seconds", 5.0)
        settings.set("vlm.cadence_recording_seconds", 5.0)
        settings.set("vlm.concurrency", 3)
        settings.set("vlm.dedupe_threshold", 0.92)
        settings.set("vlm.fusion", False)

        session = vlm_ctx.streams.current
        session.meta["mode"] = "fast"
        session.video_gate.configure(60.0)
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
            assert session.video_gate.released == 10.0
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
        session.meta["mode"] = "fast"
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
        session.meta["mode"] = "fast"
        session.ended_at = 1.0
        plugin._recording_sessions.add(session.session_id)

        entered = threading.Event()
        release = threading.Event()

        def fake_wait(_timeout: float | None = None) -> bool:
            entered.set()
            release.wait(2.0)
            return False

        plugin._stop_event.wait = fake_wait  # type: ignore[method-assign]
        thread = threading.Thread(
            target=plugin._run_recording_session, daemon=True
        )
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

    def test_process_window_with_fake_llm(self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext):
        """process_window should call LLM, emit vlm.segment, and append visual."""
        # Setup settings
        settings = vlm_ctx.service("settings")
        settings.set("vlm.enabled", True)
        settings.set("vlm.model", "deepseek-flash")
        settings.set("vlm.dedupe_threshold", 0.0)  # disable dedup for test
        settings.set("vlm.fusion", False)  # skip fusion for simplicity

        # Inject scripted LLM response
        fake_llm = vlm_ctx.service("llm")
        fake_llm.script = ["定理 1：设 $f$ 在 $D$ 上全纯。"]

        # Create synthetic frame
        arr = _synthetic_frame()
        jpeg = _to_jpeg(arr)
        frames = [_make_video_frame(jpeg, 10.0, 11.0)]

        # Listen for vlm.segment
        segments: list[dict] = []
        def on_segment(event):
            segments.append(dict(event.payload))

        vlm_ctx.events.on("vlm.segment", on_segment)

        result = vlm_plugin.process_window(10.0, 11.0, frames)
        vlm_ctx.events.flush(timeout=2.0)

        assert result.get("ok") is True
        # Check a visual item was appended
        visual = vlm_ctx.streams.current.visual
        assert len(visual) >= 1
        item = visual[-1]
        assert item.source == "vlm"
        assert "定理 1" in item.text
        assert item.start == 10.0
        assert item.end == 11.0
        assert "image_path" in item.meta
        assert "model" in item.meta

        log = (
            vlm_ctx.data_dir
            / "sessions"
            / vlm_ctx.streams.current.session_id
            / "visual.jsonl"
        )
        records = [
            json.loads(line) for line in log.read_text("utf-8").splitlines()
        ]
        assert records[-1]["start"] == 10.0
        assert records[-1]["end"] == 11.0
        assert "定理 1" in records[-1]["text"]

        # Check that thinking was disabled in the LLM call
        assert len(fake_llm.calls) >= 1
        last_call = fake_llm.calls[-1]
        extra = last_call.get("extra") or {}
        assert extra.get("thinking") == {"type": "disabled"}
        assert extra.get("frequency_penalty") == 0.2
        assert extra.get("presence_penalty") == 0.2

        # Check vlm.segment event
        assert len(segments) >= 1
        seg = segments[0]
        assert seg["start"] == 10.0
        assert seg["end"] == 11.0

    def test_process_window_skips_blank_board(self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext):
        settings = vlm_ctx.service("settings")
        settings.set("vlm.dedupe_threshold", 0.0)
        settings.set("vlm.fusion", False)

        fake_llm = vlm_ctx.service("llm")
        fake_llm.script = [BLANK_BOARD_MARKER]

        arr = _synthetic_frame(lines=False, black=True)
        jpeg = _to_jpeg(arr)
        frames = [_make_video_frame(jpeg, 5.0, 6.0)]

        visual_before = len(vlm_ctx.streams.current.visual)
        result = vlm_plugin.process_window(5.0, 6.0, frames)

        assert result.get("ok") is True
        # No new visual items for blank board
        assert len(vlm_ctx.streams.current.visual) == visual_before

    def test_process_window_skips_placeholder_only(self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext):
        """Output consisting solely of ![...]() placeholders must be skipped."""
        settings = vlm_ctx.service("settings")
        settings.set("vlm.dedupe_threshold", 0.0)
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

    def test_process_window_deduplicates(self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext):
        settings = vlm_ctx.service("settings")
        settings.set("vlm.dedupe_threshold", 0.5)  # low threshold to catch
        settings.set("vlm.fusion", False)

        same_text = "定理 2.1 设 $G$ 是一个群。"
        fake_llm = vlm_ctx.service("llm")
        fake_llm.script = [same_text, same_text]

        arr = _synthetic_frame()
        jpeg = _to_jpeg(arr)

        # First window
        frames1 = [_make_video_frame(jpeg, 0.0, 1.0)]
        vlm_plugin.process_window(0.0, 1.0, frames1)

        # Second window with identical text
        frames2 = [_make_video_frame(jpeg, 1.0, 2.0)]
        result2 = vlm_plugin.process_window(1.0, 2.0, frames2)

        # Second should be skipped
        assert result2.get("skipped") is True
        # Only one visual item
        visual = vlm_ctx.streams.current.visual
        assert len(visual) == 1

    def test_process_window_skips_no_change_marker(
        self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext
    ):
        settings = vlm_ctx.service("settings")
        settings.set("vlm.dedupe_threshold", 0.0)
        settings.set("vlm.fusion", False)

        fake_llm = vlm_ctx.service("llm")
        fake_llm.script = [NO_CHANGE_MARKER]

        arr = _synthetic_frame()
        jpeg = _to_jpeg(arr)
        frames = [_make_video_frame(jpeg, 5.0, 6.0)]

        visual_before = len(vlm_ctx.streams.current.visual)
        result = vlm_plugin.process_window(5.0, 6.0, frames)

        assert result.get("skipped") is True
        assert "无变化" in result.get("note", "")
        assert len(vlm_ctx.streams.current.visual) == visual_before

    def test_second_window_sends_previous_transcription(
        self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext
    ):
        settings = vlm_ctx.service("settings")
        settings.set("vlm.dedupe_threshold", 0.0)
        settings.set("vlm.fusion", False)

        fake_llm = vlm_ctx.service("llm")
        fake_llm.script = ["定理 A：$x=1$。", "新增：定理 B：$y=2$。"]

        arr = _synthetic_frame()
        jpeg = _to_jpeg(arr)
        vlm_plugin.process_window(0.0, 1.0, [_make_video_frame(jpeg, 0.0, 1.0)])
        vlm_plugin.process_window(5.0, 6.0, [_make_video_frame(jpeg, 5.0, 6.0)])

        assert len(fake_llm.calls) == 2
        first = fake_llm.calls[0]["messages"][0]["content"][0].text
        assert "第一次识别" in first
        second = fake_llm.calls[1]["messages"][0]["content"][0].text
        assert "定理 A" in second
        assert "新增、修改或擦除" in second

    def test_llm_error_emits_vlm_error(self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext):
        settings = vlm_ctx.service("settings")
        settings.set("vlm.dedupe_threshold", 0.0)
        settings.set("vlm.fusion", False)

        # Make the FakeLLM raise
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


# ---------------------------------------------------------------------------
# UI state tests
# ---------------------------------------------------------------------------

class TestUiState:
    def test_debug_page_present(self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext):
        snap = vlm_ctx.ui.snapshot()
        assert "vlm:board" in snap["pages"]
        assert "vlm.debug" in snap["commands"]

    def test_render_returns_blocks(self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext):
        page = vlm_ctx.ui._pages.get("vlm:board")
        assert page is not None
        blocks = vlm_ctx.ui._render(page.render, "test")
        assert isinstance(blocks, list)
        # At minimum, we get a text placeholder or image
        assert len(blocks) > 0

    def test_render_with_visual_items(self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext):
        visual = vlm_ctx.streams.current.visual
        visual.append(
            "定理 1：测试内容。",
            start=0.0,
            end=10.0,
            source="vlm",
            meta={"image_path": "/tmp/test.jpg", "model": "test"},
        )
        page = vlm_ctx.ui._pages.get("vlm:board")
        blocks = vlm_ctx.ui._render(page.render, "test")
        # Should contain markdown with the theorem text
        markdown_blocks = [b for b in blocks if b.get("type") == "markdown"]
        assert len(markdown_blocks) >= 1
        assert "定理 1" in markdown_blocks[0]["text"]


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_fmt_time(self):
        assert _fmt_time(0) == "00:00"
        assert _fmt_time(65.5) == "01:05"
        assert _fmt_time(3661.0) == "61:01"
