"""Tests for the VLM plugin: fusion, prompts, deduplication, worker, UI."""

from __future__ import annotations

import base64
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

    def test_setup_creates_ui_tab_and_page(self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext):
        snap = vlm_ctx.ui.snapshot()
        assert "classroom" in snap["tabs"]
        assert "vlm.board" in snap["items"]
        assert "vlm:board" in snap["pages"]

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
    def test_ui_tab_and_page_present(self, vlm_plugin: VlmPlugin, vlm_ctx: PluginContext):
        snap = vlm_ctx.ui.snapshot()
        assert snap["tabs"]["classroom"] == "left"
        assert snap["items"]["vlm.board"] == "classroom"
        assert "vlm:board" in snap["pages"]

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
