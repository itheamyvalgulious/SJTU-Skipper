"""VLM plugin: blackboard recognition via fused frame windows.

Registered services
-------------------
``"vlm"``
    - ``status()`` – returns dict with cursor, session, error info.
    - ``process_window(start, end, frames)`` – force-process a window.
"""

from __future__ import annotations

import base64
import difflib
import logging
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

from core.events import Event
from core.plugin import Plugin, PluginContext
from core.streams import TextItem, VideoFrame
from plugins.settings.spec import SettingsSpec

from .fusion import fuse_frames
from .prompts import (
    BLANK_BOARD_MARKER,
    BOARD_RECOGNITION_PROMPT,
    build_user_message,
    collapse_repetitions,
    is_trivial_board_text,
)

logger = logging.getLogger("lecture_agent.plugin.vlm")


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class VlmPlugin(Plugin):
    id = "vlm"
    requires = ("settings", "llm")
    provides = ("vlm",)

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------
    def setup(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._cursor: float = 0.0  # seconds consumed in current session
        self._last_session: str = ""
        self._consecutive_errors = 0
        self._last_visual_text: str = ""
        self._latest_image_path: str = ""
        self._last_error: str = ""
        self._shutting_down = False

        self._register_settings()
        self._register_service()
        self._setup_ui()

        ctx.events.on("session.started", self._on_session_started, plugin_id=self.id)
        ctx.events.on("session.stopped", self._on_session_stopped, plugin_id=self.id)
        ctx.events.on("settings.changed", self._on_settings_changed, plugin_id=self.id)

        enabled = self._setting("vlm.enabled", True)
        if enabled:
            self._start_worker()

    def teardown(self) -> None:
        with self._lock:
            self._shutting_down = True
            self._stop_event.set()
            worker = self._worker
            self._worker = None
        if worker is not None and worker.is_alive():
            worker.join(timeout=3.0)

    # ----------------------------------------------------------------
    # Settings
    # ----------------------------------------------------------------
    def _register_settings(self) -> None:
        specs: list[SettingsSpec] = [
            SettingsSpec(
                key="vlm.enabled",
                label="启用板书识别",
                type="bool",
                default=True,
                group="VLM",
                description="是否启动 VLM 板书识别工作线程",
                group_note="VLM 复用「LLM」分组中的 Base URL / API Key；本组只配置板书识别模型与处理参数。",
            ),
            SettingsSpec(
                key="vlm.model",
                label="VLM 模型",
                type="string",
                default="deepseek-flash",
                group="VLM",
                description="视觉语言模型；连接与 API Key 使用「LLM」分组。",
            ),
            SettingsSpec(
                key="vlm.window_seconds",
                label="窗口时长（秒）",
                type="float",
                default=20.0,
                group="VLM",
                description="每次融合识别的视频窗口长度",
            ),
            SettingsSpec(
                key="vlm.fps",
                label="采样帧率",
                type="float",
                default=4.0,
                group="VLM",
                description="VLM 处理使用的帧率（通常与视频帧率一致）",
            ),
            SettingsSpec(
                key="vlm.max_frames",
                label="最大帧数",
                type="int",
                default=12,
                group="VLM",
                description="每窗口最多采样的帧数；0 表示不限制（全量帧很慢）",
            ),
            SettingsSpec(
                key="vlm.fusion",
                label="启用融合",
                type="bool",
                default=True,
                group="VLM",
                description="启用图片融合（含配准与去遮挡）；关闭时只取最后一帧",
            ),
            SettingsSpec(
                key="vlm.dedupe_threshold",
                label="去重阈值",
                type="float",
                default=0.92,
                group="VLM",
                description="difflib.SequenceMatcher 相似度阈值，超过则跳过",
            ),
            SettingsSpec(
                key="vlm.prompt",
                label="板书识别提示词",
                type="text",
                default=BOARD_RECOGNITION_PROMPT,
                group="VLM",
                description="完整提示词，可编辑；`{timestamp}` 会替换为时间范围，留空会报错。",
            ),
        ]
        settings = self._ctx.service("settings")
        if settings is not None:
            for spec in specs:
                settings.register(spec)

    # ----------------------------------------------------------------
    # Service
    # ----------------------------------------------------------------
    def _register_service(self) -> None:
        self._ctx.register_service("vlm", self)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "cursor": self._cursor,
                "session_id": self._last_session,
                "latest_image": self._latest_image_path,
                "consecutive_errors": self._consecutive_errors,
                "last_error": self._last_error,
                "worker_alive": self._worker is not None and self._worker.is_alive(),
            }

    def process_window(
        self,
        start: float,
        end: float,
        frames: list[VideoFrame],
    ) -> dict[str, Any]:
        """Manually process one window (used by tests and UI actions)."""
        return self._process_window(start, end, frames, force=True)

    # ----------------------------------------------------------------
    # UI
    # ----------------------------------------------------------------
    def _setup_ui(self) -> None:
        ui = self._ctx.ui
        # Debug view only: no sidebar surface, opened from the command palette.
        ui.add_page(
            "vlm:board",
            "板书识别",
            render=self._render_board_page,
            icon="🖼",
            order=21,
            plugin_id=self.id,
        )
        ui.register_command(
            "vlm.debug",
            "板书识别（调试）",
            subtitle="查看融合板书图像与识别文本",
            category="调试",
            icon="🖼",
            order=21,
            page_id="vlm:board",
            keywords="vlm board blackboard debug 板书",
            plugin_id=self.id,
        )

    def _render_board_page(self) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        image_path = ""
        with self._lock:
            image_path = self._latest_image_path

        if image_path:
            blocks.append(
                {
                    "type": "image",
                    "src": f"/api/artifacts/{self._ctx.streams.current.session_id}/vlm/{Path(image_path).name}",
                    "alt": "最新融合板书",
                }
            )

        # Collect recent visual items
        visual = self._ctx.streams.current.visual
        items = visual.items_since(max(0, len(visual) - 20))
        visual_items = [i for i in items if i.source == "vlm"]
        if visual_items:
            # One markdown block per window: KaTeX can render them
            # progressively instead of blocking on one huge document.
            for item in visual_items:
                ts = ""
                if item.start is not None and item.end is not None:
                    ts = f"**{_fmt_time(item.start)}–{_fmt_time(item.end)}**\n\n"
                blocks.append({"type": "markdown", "text": f"{ts}{item.text}"})
        elif not image_path:
            blocks.append({"type": "text", "text": "暂无板书识别结果"})

        return blocks

    # ----------------------------------------------------------------
    # Worker
    # ----------------------------------------------------------------
    def _ask_image(self, image_bytes: bytes, timestamp: str) -> str:
        """One VLM request for an encoded JPEG; returns stripped text."""

        if self._shutting_down:
            return ""
        ctx = self._ctx
        data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
        prompt_text = str(
            self._setting("vlm.prompt", BOARD_RECOGNITION_PROMPT)
        ).strip()
        if not prompt_text:
            raise ValueError("vlm.prompt 不能为空；请填写板书识别提示词")
        # Match the legacy single-frame call shape: one user message with the
        # prompt + image.  The extra system message is unnecessary and made
        # the model more likely to keep re-stating the same formulas.
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": build_user_message(
                    timestamp=timestamp,
                    image_data_url=data_url,
                    prompt_text=prompt_text,
                ),
            },
        ]
        response = ctx.llm.chat(
            profile="vlm",
            model=str(self._setting("vlm.model", "") or "") or None,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
            extra={
                "thinking": {"type": "disabled"},
                "frequency_penalty": 0.2,
                "presence_penalty": 0.2,
            },
        )
        text = getattr(response, "content", str(response)).strip()
        return collapse_repetitions(text)

    def _on_settings_changed(self, event: Event) -> None:
        if event.get("key") != "vlm.enabled":
            return
        if bool(event.get("new")):
            self._start_worker()
        else:
            self._stop_worker()

    def _stop_worker(self) -> None:
        with self._lock:
            worker = self._worker
            self._worker = None
        self._stop_event.set()
        if worker is not None and worker.is_alive():
            worker.join(timeout=3.0)

    def _start_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop_event.clear()
            self._worker = threading.Thread(
                target=self._worker_loop, name="vlm-worker", daemon=True
            )
            self._worker.start()

    def _worker_loop(self) -> None:
        logger.info("VLM worker started")
        while not self._stop_event.is_set():
            try:
                self._process_next_window()
            except Exception as exc:
                logger.exception("VLM worker iteration failed")
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._consecutive_errors += 1
                ctx = self._ctx
                if ctx is not None:
                    ctx.events.emit(
                        "vlm.error",
                        session_id=ctx.streams.current.session_id,
                        error=self._last_error,
                    )
                    ctx.ui.refresh("vlm.error")
                backoff = min(10.0, 0.5 * (2 ** min(self._consecutive_errors, 5)))
                self._stop_event.wait(backoff)
            else:
                self._consecutive_errors = 0
                self._last_error = ""

        # Flush remaining frames
        try:
            self._flush_remaining()
        except Exception:
            logger.exception("VLM worker flush failed")
        logger.info("VLM worker stopped")

    def _process_next_window(self) -> None:
        ctx = self._ctx
        hub = ctx.streams
        session = hub.current
        video = session.video

        # Detect session switch
        sid = session.session_id
        with self._lock:
            if self._last_session != sid:
                self._cursor = 0.0
                self._last_session = sid
                self._last_visual_text = ""

        cursor = self._cursor
        window = max(5.0, self._setting("vlm.window_seconds", 20.0))
        fps = self._setting("vlm.fps", 4.0)
        max_frames_val = int(self._setting("vlm.max_frames", 12))

        # How many frames to expect
        expected_frames = int(window * fps)
        total_seconds_needed = cursor + window

        # Wait for enough video to accumulate
        if video.duration + 1e-9 < total_seconds_needed:
            # Not enough data yet; check if session has ended
            if session.ended_at is not None:
                self._flush_remaining()
            else:
                self._stop_event.wait(0.5)
            return

        # Gather frames in [cursor, cursor+window)
        raw_frames = video.read_frames(cursor, total_seconds_needed)
        if not raw_frames:
            self._stop_event.wait(0.5)
            return

        # Sample frames: take at most max_frames, evenly spaced
        if max_frames_val > 0 and len(raw_frames) > max_frames_val:
            step = len(raw_frames) / max_frames_val
            indices = [int(i * step) for i in range(max_frames_val)]
            raw_frames = [raw_frames[i] for i in indices if i < len(raw_frames)]

        start_time = raw_frames[0].start
        end_time = raw_frames[-1].end
        self._process_window(start_time, end_time, raw_frames, force=False)

        # Advance cursor
        with self._lock:
            if self._last_session == session.session_id:
                self._cursor = end_time

    def _flush_remaining(self) -> None:
        ctx = self._ctx
        video = ctx.streams.current.video
        with self._lock:
            cursor = self._cursor
        dur = video.duration
        if cursor + 1e-9 >= dur:
            return
        remaining = video.read_frames(cursor, dur)
        if remaining:
            start_time = remaining[0].start
            end_time = remaining[-1].end
            self._process_window(start_time, end_time, remaining, force=True)
            with self._lock:
                self._cursor = end_time

    def _process_window(
        self,
        start: float,
        end: float,
        frames: list[VideoFrame],
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        ctx = self._ctx

        # --- Fuse ---
        # Full frame at native resolution: no board crop, no down-scale.
        do_fusion = self._setting("vlm.fusion", True)

        if do_fusion:
            fused = fuse_frames(
                frames, crop=(0.0, 0.0, 1.0, 1.0), max_width=0, align=True
            )
        else:
            fused = _decode_single(frames[-1].data)

        # --- Save image ---
        session_dir = ctx.session_dir()
        vlm_dir = session_dir / "vlm"
        vlm_dir.mkdir(parents=True, exist_ok=True)
        ts_start = int(start)
        ts_end = int(end)
        image_filename = f"{ts_start:06d}-{ts_end:06d}.jpg"
        image_path = vlm_dir / image_filename
        Image.fromarray(fused).save(str(image_path), quality=92)

        with self._lock:
            self._latest_image_path = str(image_path)

        # --- Call LLM ---
        timestamp = f"{_fmt_time(start)}–{_fmt_time(end)}"
        try:
            text = self._ask_image(image_path.read_bytes(), timestamp)
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            ctx.events.emit(
                "vlm.error",
                session_id=ctx.streams.current.session_id,
                start=start,
                end=end,
                error=self._last_error,
            )
            ctx.ui.refresh("vlm.error")
            raise

        # --- Emit segment ---
        ctx.events.emit(
            "vlm.segment",
            session_id=ctx.streams.current.session_id,
            start=start,
            end=end,
            chars=len(text),
            model=self._setting("vlm.model", "deepseek-flash"),
        )

        # --- Dedupe ---
        threshold = self._setting("vlm.dedupe_threshold", 0.92)
        with self._lock:
            last_text = self._last_visual_text
        if last_text and text:
            ratio = difflib.SequenceMatcher(None, last_text, text).ratio()
            if ratio >= threshold:
                logger.debug(
                    "dedupe: skipping window %.0f-%.0f (ratio=%.3f)",
                    start, end, ratio,
                )
                return {"ok": True, "start": start, "end": end, "skipped": True}

        # --- Append visual ---
        if text and text != BLANK_BOARD_MARKER and not is_trivial_board_text(text):
            visual = ctx.streams.current.visual
            visual.append(
                text,
                start=start,
                end=end,
                source="vlm",
                meta={
                    "image_path": str(image_path),
                    "model": self._setting("vlm.model", "deepseek-flash"),
                },
            )
            with self._lock:
                self._last_visual_text = text

        return {"ok": True, "start": start, "end": end, "chars": len(text)}

    # ----------------------------------------------------------------
    # Event handlers
    # ----------------------------------------------------------------
    def _on_session_started(self, event: Event) -> None:
        with self._lock:
            self._cursor = 0.0
            self._last_session = event.get("session_id", "")
            self._last_visual_text = ""
            self._latest_image_path = ""
            self._consecutive_errors = 0

    def _on_session_stopped(self, event: Event) -> None:
        try:
            self._flush_remaining()
        except Exception:
            logger.exception("VLM flush on session stop failed")

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _setting(self, key: str, default: Any = None) -> Any:
        settings = self._ctx.service("settings")
        if settings is None:
            return default
        return settings.get(key, default)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def _decode_single(data: bytes) -> np.ndarray:
    """Decode a single JPEG frame at native size (non-fusion mode)."""

    import io

    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img, dtype=np.uint8)


VLMPlugin = VlmPlugin