"""VLM plugin: blackboard recognition via fused frame windows.

Registered services
-------------------
``"vlm"``
    A :class:`VLMService` facade (six methods, no plugin internals):
    ``status()``, ``enabled()``, ``busy()``, ``flushing()``,
    ``process_window(start, end)`` and ``visual_path(session_id)``.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
from typing import Any
import numpy as np

from PIL import Image

from core.coverage import CoverageStore, skip_coverage
from core.events import Event
from core.agent_types import AgentRole, FinishReason, GenerationErrorCode, IpcMethod
from core.plugin import Plugin, PluginContext
from core.settings_spec import SettingType, SettingsSpec
from core.streams import StreamKind, TextSource, VideoFrame
from core.media import MediaKind, ProcessingMode
from core.ui import BlockType

from .fusion import fuse_frames
from .prompts import (
    BLANK_BOARD_MARKER,
    build_recognition_messages,
    collapse_repetitions,
    deduplicate_transcription,
    is_trivial_board_text,
)

logger = logging.getLogger("lecture_agent.plugin.vlm")

# The VLM plugin writes board items here. The notes plugin reads this log
# to reconstruct lecture material after a stop or a notes rerun.
VISUAL_LOG = "visual.jsonl"


class WindowOutcome(StrEnum):
    EMPTY_RESPONSE = "模型返回空"
    BLANK_BOARD = "未识别到板书（空白黑板或仅示意图）"
    DUPLICATE = "与近期板书重复，已跳过"
    WRITTEN = "已写入板书"


class VLMRecognitionError(RuntimeError):
    def __init__(self, message: str, *, code: GenerationErrorCode, attempts: int):
        super().__init__(message)
        self.code = code
        self.attempts = attempts


# ---------------------------------------------------------------------------
# Service facade
# ---------------------------------------------------------------------------


class VLMService:
    """Public board recognition operations."""

    __slots__ = ("_plugin",)

    def __init__(self, plugin: "VlmPlugin") -> None:
        self._plugin = plugin

    def status(self) -> dict[str, Any]:
        return self._plugin.status()

    def enabled(self) -> bool:
        return self._plugin.enabled()

    def busy(self) -> bool:
        return self._plugin.busy()

    def flushing(self) -> bool:
        """True while the stop-time tail flush is still writing its log."""

        return self._plugin.flushing()

    def cancel(self, session_id: str) -> None:
        self._plugin.cancel(session_id)

    def processing(self, session_id: str) -> bool:
        with self._plugin._lock:
            session = self._plugin._processing_session
            return session is not None and session.session_id == session_id

    def reset(self, session_id: str) -> None:
        """Clear cached recognition state after the session stops."""
        self._plugin.reset(session_id)

    def process_window(self, start: float, end: float) -> list[dict[str, Any]]:
        """Force-process one window of the current video stream."""

        result = self._plugin.process_range(start, end)
        return [result] if isinstance(result, dict) else []

    def visual_path(self, session_id: str = "") -> Path | None:
        """Path of this session's ``visual.jsonl`` (``None`` without ctx)."""

        return self._plugin.visual_path(session_id)


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreparedWindow:
    start: float
    end: float
    image_path: Path
    image_bytes: bytes


class VlmPlugin(Plugin):
    id = "vlm"
    requires = ("settings", "llm")
    provides = ("vlm",)

    def __init__(self) -> None:
        self._service: VLMService | None = None
        self._ctx: PluginContext | None = None
        self._page_added = False
        self._command_added = False
        self._consumer_registered = False
        self._cancelled_sessions: set[str] = set()

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------
    def setup(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        self._llm = ctx.require("llm")
        self._lock = threading.RLock()
        self._persist_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._cursor: float = 0.0  # seconds consumed in current session
        self._last_session: str = ""
        self._consecutive_errors = 0
        self._latest_image_path: str = ""
        self._last_error: str = ""
        self._last_window: dict[str, Any] = {}
        self._recording_sessions: set[str] = set()
        self._recording_cursor: dict[str, float] = {}
        self._recording_active = False
        # The notes plugin waits for the tail flush before reading visual.jsonl.
        self._flushing = False
        self._processing_session: Any = None
        self._coverage_cache: dict[str, CoverageStore] = {}
        self._shutting_down = False
        self._history: deque[str] = deque(maxlen=3)
        self._next_center: float = 10.0
        self._sessions: dict[str, Any] = {}
        self._cancelled_sessions: set[str] = set()
        self._submitted_requests: set[str] = set()
        self._requests: dict[str, str] = {}
        self._session_history: dict[str, deque[str]] = {}
        self._disabled_at: dict[str, float] = {}
        self._prepare_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vlm-prepare"
        )
        self._prepared_future = None
        self._prepared_key = None
        self._sessions[ctx.streams.current.session_id] = ctx.streams.current

        self._register_settings()
        self._register_service()
        self._setup_ui()
        self._register_consumer(ctx)

        ctx.events.on("session.started", self._on_session_started, plugin_id=self.id)
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
            requests = set(self._requests.values())
        for sid in requests:
            self.cancel(sid)
        if worker is not None and worker.is_alive():
            worker.join(timeout=3.0)
        self._prepare_pool.shutdown(wait=True, cancel_futures=True)
        ctx = self._ctx
        if ctx is not None:
            if self._consumer_registered:
                try:
                    ctx.streams.unregister_consumer(self.id)
                except Exception:  # noqa: BLE001 - hub may already be gone
                    pass
                self._consumer_registered = False
            if self._page_added:
                try:
                    ctx.ui.remove_page("vlm:board")
                except Exception:  # noqa: BLE001 - UI may already be gone
                    pass
                self._page_added = False
            if self._command_added:
                try:
                    ctx.ui.unregister_command("vlm.debug")
                except Exception:  # noqa: BLE001
                    pass
                self._command_added = False

    # ----------------------------------------------------------------
    # Backpressure contract (video consumer)
    # ----------------------------------------------------------------
    def _register_consumer(self, ctx: PluginContext) -> None:
        """Publish this plugin as the ``video`` consumer of the stream hub."""

        if ctx.streams is None:
            return
        ctx.streams.register_consumer(
            self.id,
            MediaKind.VIDEO,
            cursor_fn=lambda: float(self.status().get("cursor", 0.0) or 0.0),
            busy_fn=lambda: self.busy() or self.flushing(),
            gate_limit_fn=self._gate_limit,
        )
        self._consumer_registered = True

    def _gate_limit(self) -> float:
        """How many video seconds may buffer ahead of board recognition."""

        if not self.enabled():
            return 0.0
        try:
            prepare_ahead = min(1, max(0, int(self._setting("vlm.prepare_ahead", 1))))
        except (TypeError, ValueError):
            prepare_ahead = 1
        try:
            window = float(self._setting("vlm.window_seconds", 20.0))
        except (TypeError, ValueError):
            window = 20.0
        return (1 + prepare_ahead) * max(1.0, window)

    # ----------------------------------------------------------------
    # Settings
    # ----------------------------------------------------------------
    def _register_settings(self) -> None:
        specs: list[SettingsSpec] = [
            SettingsSpec(
                key="vlm.enabled",
                label="启用板书识别",
                type=SettingType.BOOL,
                default=True,
                group="VLM",
                description="是否启动 VLM 板书识别工作线程",
                group_note="识别模型在 LLM 的 roles.yml 中由 board_recognition 配置．",
            ),
            SettingsSpec(
                key="vlm.window_seconds",
                label="窗口时长（秒）",
                type=SettingType.FLOAT,
                default=20.0,
                group="VLM",
                description="每次融合识别的视频窗口长度；窗口以采样点为中心，前后各占一半",
            ),
            SettingsSpec(
                key="vlm.cadence_live_seconds",
                label="直播采样间隔（秒）",
                type=SettingType.FLOAT,
                default=180.0,
                group="VLM",
                description="直播时每隔多少秒取一个 20 秒窗口（默认 3 分钟）",
            ),
            SettingsSpec(
                key="vlm.cadence_recording_seconds",
                label="录播采样间隔（秒）",
                type=SettingType.FLOAT,
                default=300.0,
                group="VLM",
                description="录播全速处理时每隔多少秒取一个 20 秒窗口（默认 5 分钟）",
            ),
            SettingsSpec(
                key="vlm.history_windows",
                label="板书去重窗口数",
                type=SettingType.INT,
                default=3,
                group="VLM",
                description="只用最近写入的板书去重，不发送给模型．0 表示不跨窗口去重．",
            ),
            SettingsSpec(
                key="vlm.fps",
                label="采样帧率",
                type=SettingType.FLOAT,
                default=4.0,
                group="VLM",
                description="VLM 处理使用的帧率（通常与视频帧率一致）",
            ),
            SettingsSpec(
                key="vlm.max_frames",
                label="最大帧数",
                type=SettingType.INT,
                default=12,
                group="VLM",
                description="每个窗口最多采样的帧数；0 表示不限制（全量帧很慢）",
            ),
            SettingsSpec(
                key="vlm.loop_max_attempts",
                label="重复输出最大尝试次数",
                type=SettingType.INT,
                default=3,
                group="VLM",
                description="包含首次识别；检测到重复输出时重新尝试",
            ),
            SettingsSpec(
                key="vlm.fusion",
                label="启用融合",
                type=SettingType.BOOL,
                default=True,
                group="VLM",
                description="启用图片融合（含配准与去遮挡）；关闭时只取最后一帧",
            ),
            SettingsSpec(
                key="vlm.prepare_ahead",
                label="提前准备窗口数",
                type=SettingType.INT,
                default=1,
                group="VLM",
                description="0 为顺序准备，1 为识别时准备下一个窗口",
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
        service = VLMService(self)
        self._service = service
        self._ctx.register_service("vlm", service)

    def cancel(self, session_id: str) -> None:
        with self._lock:
            self._cancelled_sessions.add(session_id)
            requests = [
                key
                for key, sid in self._requests.items()
                if sid == session_id and key in self._submitted_requests
            ]
            session = self._ctx.streams.get_session(session_id)
        if session is not None:
            session.video_gate.close()
        for key in requests:
            try:
                self._llm.cancel(key)
            except Exception:
                logger.exception("Cannot cancel VLM request %s", key)

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._cancelled_sessions.add(session_id)
            session = self._ctx.streams.get_session(session_id)
            if session is not None:
                self._sessions[session_id] = session
            self._recording_sessions.discard(session_id)
            self._recording_cursor.pop(session_id, None)
            self._coverage_cache.pop(session_id, None)
            self._disabled_at.pop(session_id, None)
            self._session_history.pop(session_id, None)
            if self._prepared_key is not None and self._prepared_key[0] == session_id:
                self._prepared_future.cancel()
                self._prepared_future = None
                self._prepared_key = None
            if self._last_session == session_id:
                self._cursor = 0.0
                self._next_center = self._window_seconds() / 2
                self._history.clear()
                self._latest_image_path = ""
                self._last_window = {}
                self._last_error = ""
                self._consecutive_errors = 0

    def status(self) -> dict[str, Any]:
        with self._lock:
            session = self._ctx.streams.current if self._ctx is not None else None
            payload = {
                "cursor": self._cursor,
                "session_id": self._last_session,
                "latest_image": self._latest_image_path,
                "last_window": dict(self._last_window),
                "consecutive_errors": self._consecutive_errors,
                "last_error": self._last_error,
                "worker_alive": self._worker is not None and self._worker.is_alive(),
            }
        if session is not None and self._ctx is not None:
            try:
                covered = self._coverage(session).covered_seconds(MediaKind.VIDEO)
            except Exception:  # noqa: BLE001 - status must never fail
                covered = 0.0
            total = float(session.video.duration)
            payload["video_seconds"] = round(total, 3)
            payload["covered_seconds"] = round(covered, 3)
            payload["uncovered_seconds"] = round(max(0.0, total - covered), 3)
        return payload

    def process_window(
        self,
        start: float,
        end: float,
        frames: list[VideoFrame],
    ) -> dict[str, Any]:
        """Manually process one window with explicit frames (tests/UI)."""
        return self._process_window(start, end, frames, force=True)

    def process_range(self, start: float, end: float) -> dict[str, Any]:
        """Force-process ``[start, end)`` of the current video stream."""

        ctx = self._ctx
        if ctx is None:
            return {"ok": False, "error": "vlm 插件尚未初始化"}
        frames = ctx.streams.current.video.read_frames(float(start), float(end))
        if not frames:
            return {
                "ok": False,
                "start": float(start),
                "end": float(end),
                "error": "该区间没有视频帧",
            }
        return self._process_window(float(start), float(end), frames, force=True)

    def visual_path(self, session_id: str = "") -> Path | None:
        """``sessions/<id>/visual.jsonl`` for *session_id* (current if empty)."""

        ctx = self._ctx
        if ctx is None:
            return None
        sid = str(session_id or "")
        if not sid:
            try:
                sid = str(ctx.streams.current.session_id or "")
            except Exception:  # noqa: BLE001 - no session yet
                return None
        if not sid:
            return None
        return ctx.data_dir / "sessions" / sid / VISUAL_LOG

    def _persist_item(self, session_id: str, item: Any) -> None:
        """Append one written board item to ``sessions/<id>/visual.jsonl``."""

        ctx = self._ctx
        if ctx is None or not session_id:
            return
        path = ctx.data_dir / "sessions" / session_id / VISUAL_LOG
        record = {
            "start": None if item.start is None else round(float(item.start), 3),
            "end": None if item.end is None else round(float(item.end), 3),
            "text": item.text,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._persist_lock:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            # Losing the on-disk copy must never break board recognition.
            logger.warning(
                "failed to persist visual log for %s", session_id, exc_info=True
            )

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
            order=21,
            plugin_id=self.id,
        )
        ui.register_command(
            "vlm.debug",
            "板书识别（调试）",
            category="调试",
            order=21,
            page_id="vlm:board",
            keywords="vlm board blackboard debug 板书",
            plugin_id=self.id,
        )
        self._page_added = True
        self._command_added = True

    def _render_board_page(self) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        image_path = ""
        with self._lock:
            image_path = self._latest_image_path
            last = dict(self._last_window)

        if image_path:
            blocks.append(
                {
                    "type": BlockType.IMAGE,
                    "src": f"/api/artifacts/{self._ctx.streams.current.session_id}/vlm/{Path(image_path).name}",
                    "alt": "最新融合板书",
                }
            )
        if last:
            stamp = (
                f"{_fmt_time(last.get('start', 0))} - {_fmt_time(last.get('end', 0))}"
            )
            blocks.append(
                {
                    "type": BlockType.TEXT,
                    "text": f"最近窗口 {stamp}：{last.get('note', '')}",
                }
            )
            if last.get("text"):
                blocks.append(
                    {
                        "type": BlockType.HTML,
                        "html": '<pre class="vlm-raw-reply overflow-auto" aria-label="模型原始回复">'
                        + escape(f"模型原始回复：\n{last['text']}")
                        + "</pre>",
                    }
                )

        # Collect recent visual items
        visual = self._ctx.streams.current.visual
        items = visual.items_since(max(0, len(visual) - 20))
        visual_items = [i for i in items if i.source == TextSource.VLM]
        if visual_items:
            # One markdown block per window: KaTeX can render them
            # progressively instead of blocking on one huge document.
            for item in visual_items:
                ts = ""
                if item.start is not None and item.end is not None:
                    ts = f"**{_fmt_time(item.start)} - {_fmt_time(item.end)}**\n\n"
                blocks.append({"type": BlockType.MARKDOWN, "text": f"{ts}{item.text}"})
        elif not image_path:
            blocks.append({"type": BlockType.TEXT, "text": "暂无板书识别结果"})

        return blocks

    # ----------------------------------------------------------------
    # Worker
    # ----------------------------------------------------------------
    def _ask_image(
        self,
        image_bytes: bytes,
        timestamp: str,
    ) -> dict[str, Any]:
        """Generate one board transcription from an encoded JPEG."""

        if self._shutting_down:
            return {"content": ""}
        ctx = self._ctx
        data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
        messages = build_recognition_messages(
            timestamp=timestamp, image_data_url=data_url
        )
        session = self._processing_session or ctx.streams.current
        sid = session.session_id
        request_id = f"vlm_{sid}_{time.monotonic_ns()}"
        max_attempts = int(self._setting("vlm.loop_max_attempts", 3))
        if max_attempts < 1:
            raise ValueError("vlm.loop_max_attempts 必须至少为 1")
        with self._lock:
            if sid in self._cancelled_sessions:
                raise RuntimeError("VLM session cancelled")
            self._requests[request_id] = sid

        def sent():
            with self._lock:
                self._submitted_requests.add(request_id)
                cancelled = sid in self._cancelled_sessions
            if cancelled:
                self._llm.cancel(request_id)

        try:
            response = self._llm.call(
                IpcMethod.GENERATE,
                role=AgentRole.BOARD_RECOGNITION,
                messages=messages,
                temperature=0.1,
                loop_max_attempts=max_attempts,
                session_id=request_id,
                on_sent=sent,
            )
        finally:
            with self._lock:
                self._requests.pop(request_id, None)
                self._submitted_requests.discard(request_id)
        if sid in self._cancelled_sessions or self._stop_event.is_set():
            raise RuntimeError("VLM session cancelled")
        if error := response.get("error"):
            if (
                error.get("code") == GenerationErrorCode.THINKING_LOOP
                and error.get("attempts") == max_attempts
            ):
                raise VLMRecognitionError(
                    error["message"],
                    code=GenerationErrorCode(error["code"]),
                    attempts=error["attempts"],
                )
            raise RuntimeError(error.get("message", str(error)))
        if response.get("finish_reason") == FinishReason.LENGTH:
            raise RuntimeError(
                "VLM output reached the model token limit; transcription is incomplete"
            )
        response["content"] = collapse_repetitions(
            str(response.get("content") or "").strip()
        )
        return response

    def _on_settings_changed(self, event: Event) -> None:
        if event.get("key") != "vlm.enabled":
            return
        threading.Thread(
            target=self._apply_enabled,
            args=(bool(event.get("new")),),
            name="vlm-enable",
            daemon=True,
        ).start()

    def _apply_enabled(self, enabled: bool) -> None:
        session = self._processing_session or self._ctx.streams.current
        if enabled:
            start = self._disabled_at.pop(session.session_id, session.video.duration)
            self._ctx.streams.seal_text(
                session.session_id,
                StreamKind.VISUAL,
                session.video.duration,
                error={
                    "start": start,
                    "end": session.video.duration,
                    "error": "Source disabled",
                }
                if session.video.duration > start
                else None,
            )
            self._ctx.streams.set_text_enabled(
                session.session_id, StreamKind.VISUAL, True
            )
            self._start_worker()
        else:
            self._disabled_at[session.session_id] = self._cursor
            self._stop_worker()
            self._ctx.streams.set_text_enabled(
                session.session_id, StreamKind.VISUAL, False
            )

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
            session = None
            try:
                session = self._select_session()
                if session is None:
                    self._stop_event.wait(0.1)
                    continue
                self._processing_session = session
                if self._is_recording_session():
                    self._run_recording_session()
                else:
                    self._process_next_window()
            except Exception as exc:
                if session is not None and (
                    session.session_id in self._cancelled_sessions
                    or self._ctx.streams.get_session(session.session_id) is not session
                ):
                    continue
                logger.exception("VLM worker iteration failed")
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._consecutive_errors += 1
                ctx = self._ctx
                if ctx is not None:
                    ctx.events.emit(
                        "vlm.error",
                        session_id=session.session_id if session is not None else "",
                        error=self._last_error,
                    )
                    ctx.ui.refresh("vlm.error")
                if session is not None:
                    self.cancel(session.session_id)
                    ctx.events.emit(
                        "input.error",
                        session_id=session.session_id,
                        message=f"VLM failed: {self._last_error}",
                    )
                else:
                    self._stop_event.wait(0.1)
            else:
                self._consecutive_errors = 0
                self._last_error = ""

            finally:
                if self._processing_session is session:
                    self._processing_session = None
        # Flush remaining frames
        with self._lock:
            self._flushing = True
        try:
            self._flush_remaining()
        except Exception:
            logger.exception("VLM worker flush failed")
        finally:
            with self._lock:
                self._flushing = False
        logger.info("VLM worker stopped")

    def _process_next_window(self) -> None:
        ctx = self._ctx
        hub = ctx.streams
        session = self._processing_session or hub.current
        video = session.video

        # Detect session switch
        sid = session.session_id
        window = self._window_seconds()
        half = window / 2.0
        cadence = self._cadence(recording=False)
        with self._lock:
            if self._last_session != sid:
                self._cursor = 0.0
                self._last_session = sid
                self._next_center = half
                self._history.clear()
            center = self._next_center

        # Frames between windows are never used: drop them as they arrive so
        # a 3-minute cadence does not buffer minutes of MJPEG.
        self._trim_before(session, min(center - half, video.duration - window))

        # Wait for enough video to accumulate
        if video.duration + 1e-9 < center + half:
            # Not enough data yet; check if session has ended
            if (
                session.video_input_done
                or session.input_done
                or session.ended_at is not None
            ):
                with self._lock:
                    self._flushing = True
                try:
                    self._flush_remaining()
                finally:
                    with self._lock:
                        self._flushing = False
                hub.seal_text(
                    session.session_id, StreamKind.VISUAL, video.duration, final=True
                )
            else:
                self._stop_event.wait(0.5)
            return

        start = max(0.0, center - half)
        end = min(video.duration, center + half)
        raw_frames = video.read_frames(start, end)
        if not raw_frames:
            self._stop_event.wait(0.5)
            return

        # Sample frames: take at most max_frames, evenly spaced
        max_frames_val = int(self._setting("vlm.max_frames", 12))
        if max_frames_val > 0 and len(raw_frames) > max_frames_val:
            step = len(raw_frames) / max_frames_val
            indices = [int(i * step) for i in range(max_frames_val)]
            raw_frames = [raw_frames[i] for i in indices if i < len(raw_frames)]

        start_time = raw_frames[0].start
        end_time = raw_frames[-1].end
        self._process_window(start_time, end_time, raw_frames, force=False)

        # Advance to the next sampling centre
        with self._lock:
            if self._last_session == session.session_id:
                self._next_center = center + cadence
        hub.seal_text(
            session.session_id,
            StreamKind.VISUAL,
            max(0.0, min(center + cadence - half, video.duration - window)),
        )

    def _flush_remaining(self) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        session = self._processing_session or ctx.streams.current
        if (
            session.session_id in self._cancelled_sessions
            or ctx.streams.get_session(session.session_id) is not session
        ):
            return
        dur = session.video.duration
        window = self._window_seconds()
        with self._lock:
            center = self._next_center
            committed = self._cursor
        start = max(0.0, min(center - window / 2.0, dur - window))
        end = dur
        if end - start < 2.0 or end <= committed + 1.0:
            return
        remaining = session.video.read_frames(start, end)
        if remaining:
            self._process_window(start, end, remaining, force=True)
            with self._lock:
                self._cursor = end
                self._next_center = end + self._cadence(recording=False)

    def _sample_frames(self, frames: list[VideoFrame]) -> list[VideoFrame]:
        max_frames_val = int(self._setting("vlm.max_frames", 12))
        if max_frames_val <= 0 or len(frames) <= max_frames_val:
            return frames
        step = len(frames) / max_frames_val
        indices = [int(i * step) for i in range(max_frames_val)]
        return [frames[i] for i in indices if i < len(frames)]

    def _prepare_window(
        self, session, start: float, end: float, frames: list[VideoFrame]
    ) -> PreparedWindow:
        frames = self._sample_frames(frames)

        do_fusion = self._setting("vlm.fusion", True)
        if do_fusion:
            fused = fuse_frames(
                frames, crop=(0.0, 0.0, 1.0, 1.0), max_width=0, align=True
            )
        else:
            fused = _decode_single(frames[-1].data)

        buffer = io.BytesIO()
        Image.fromarray(fused).save(buffer, format="JPEG", quality=92)
        image_bytes = buffer.getvalue()
        vlm_dir = self._ctx.data_dir / "sessions" / session.session_id / "vlm"
        vlm_dir.mkdir(parents=True, exist_ok=True)
        image_path = vlm_dir / f"{int(start):06d}-{int(end):06d}.jpg"
        image_path.write_bytes(image_bytes)

        return PreparedWindow(start, end, image_path, image_bytes)

    def _recognize_window(self, prepared: PreparedWindow) -> dict[str, Any]:
        timestamp = f"{_fmt_time(prepared.start)} - {_fmt_time(prepared.end)}"
        response = self._ask_image(prepared.image_bytes, timestamp)
        return {
            "image_path": str(prepared.image_path),
            "text": response["content"],
            "model": response.get("model", ""),
            "usage": response.get("usage", {}),
        }

    def _run_window(
        self, start: float, end: float, frames: list[VideoFrame]
    ) -> dict[str, Any]:
        if not frames:
            return {"image_path": "", "text": "", "empty": True}
        session = self._processing_session or self._ctx.streams.current
        key = (session.session_id, start, end)
        if self._prepared_future is not None and self._prepared_key == key:
            prepared = self._prepared_future.result()
            self._prepared_future = None
            self._prepared_key = None
        else:
            prepared = self._prepare_window(session, start, end, list(frames))
        if prepared is None:
            prepared = self._prepare_window(session, start, end, list(frames))
        if (
            self._prepared_future is not None
            and self._prepared_key != key
            and self._prepared_future.done()
        ):
            self._prepared_future.result()
            self._prepared_future = None
            self._prepared_key = None
        if (
            threading.current_thread() is self._worker
            and int(self._setting("vlm.prepare_ahead", 1)) > 0
            and self._prepared_future is None
        ):
            cadence = self._cadence(
                recording=str(session.meta.get("mode", "")) == ProcessingMode.FAST
            )
            next_start, next_end = start + cadence, end + cadence
            self._prepared_key = (session.session_id, next_start, next_end)
            self._prepared_future = self._prepare_pool.submit(
                self._prepare_next_window,
                session,
                next_start,
                next_end,
            )
        return self._recognize_window(prepared)

    def _prepare_next_window(self, session, start: float, end: float):
        while not self._stop_event.is_set():
            if (
                session is not self._processing_session
                or session.session_id in self._cancelled_sessions
            ):
                return None
            self._trim_before(
                session, min(start, session.video.duration - self._window_seconds())
            )
            if session.video.duration + 1e-6 >= end:
                frames = session.video.read_frames(start, end)
                if frames:
                    return self._prepare_window(session, start, end, list(frames))
                return None
            if (
                session.video_input_done
                or session.input_done
                or session.ended_at is not None
            ):
                return None
            session.video.wait_for(end, timeout=0.1)
        return None

    def _commit_window(self, start: float, end: float, result: Any) -> dict[str, Any]:
        """Deduplicate and append one window, in delivery order."""

        ctx = self._ctx
        session = self._processing_session or ctx.streams.current
        if (
            ctx.streams.get_session(session.session_id) is not session
            or session.session_id in self._cancelled_sessions
        ):
            raise RuntimeError("VLM session cancelled or replaced")

        raw_text = str(result.get("text", "") or "")
        text = raw_text
        image_path = str(result.get("image_path", "") or "")
        if not text:
            note = WindowOutcome.EMPTY_RESPONSE
        elif text == BLANK_BOARD_MARKER or is_trivial_board_text(text):
            note = WindowOutcome.BLANK_BOARD
        else:
            text = deduplicate_transcription(text, self._snapshot_history())
            note = WindowOutcome.WRITTEN if text else WindowOutcome.DUPLICATE

        # The window was processed; record it even when the board was empty.
        self._coverage(session).mark(MediaKind.VIDEO, start, end)

        ctx.events.emit(
            "vlm.segment",
            session_id=session.session_id,
            start=start,
            end=end,
            chars=len(text),
            role=AgentRole.BOARD_RECOGNITION,
        )

        with self._lock:
            self._last_window = {
                "start": start,
                "end": end,
                "note": note,
                "chars": len(text),
                "text": raw_text,
            }
        logger.info(
            "VLM window %s-%s: %s (%d chars)",
            _fmt_time(start),
            _fmt_time(end),
            note,
            len(text),
        )

        if note == WindowOutcome.WRITTEN:
            visual = session.visual
            item = visual.append(
                text,
                start=start,
                end=end,
                source=TextSource.VLM,
                meta={
                    "image_path": image_path,
                    "model": result.get("model", ""),
                    "usage": result.get("usage", {}),
                },
            )
            if item is not None:
                self._persist_item(session.session_id, item)
            with self._lock:
                self._history.append(text)

        with self._lock:
            self._cursor = max(self._cursor, float(end))
        session.video_gate.release(end)
        # Drop processed frames: a 2h lecture is several GB of MJPEG.
        session.video.trim(max(0.0, float(end) - 30.0))
        return {
            "ok": True,
            "start": start,
            "end": end,
            "chars": len(text),
            "skipped": note != WindowOutcome.WRITTEN,
            "note": note,
        }

    def _skip_failed_window(self, session, start, end, error):
        with self._lock:
            if (
                session.session_id in self._cancelled_sessions
                or self._stop_event.is_set()
                or self._ctx.streams.get_session(session.session_id) is not session
            ):
                raise RuntimeError("VLM session cancelled")
            detail = {
                "code": error.code,
                "message": str(error),
                "attempts": error.attempts,
            }
            record = {"start": start, "end": end, "error": detail}
            path = self._ctx.data_dir / "sessions" / session.session_id / VISUAL_LOG
            with self._persist_lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                previous = (
                    [
                        json.loads(line)
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    if path.exists()
                    else []
                )
                if not any(
                    row.get("start") == start
                    and row.get("end") == end
                    and row.get("error", {}).get("code") == error.code
                    for row in previous
                ):
                    with path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            failure = {
                "start": start,
                "end": end,
                "error": str(error),
                "code": error.code,
                "attempts": error.attempts,
            }
            self._ctx.streams.seal_text(
                session.session_id, StreamKind.VISUAL, end, error=failure
            )
            self._last_error = str(error)
            self._last_window = {
                **failure,
                "chars": 0,
                "text": "",
                "note": "板书识别失败，已记录缺口并继续",
            }
            self._cursor = max(self._cursor, float(end))
            session.video_gate.release(end)
            session.video.trim(max(0.0, float(end) - 30.0))
        self._ctx.events.emit(
            "vlm.error", session_id=session.session_id, recoverable=True, **failure
        )
        self._ctx.ui.refresh("vlm.error")
        return {"ok": False, "skipped": True, **failure}

    def _process_window(
        self,
        start: float,
        end: float,
        frames: list[VideoFrame],
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        ctx = self._ctx
        session = self._processing_session or ctx.streams.current
        if ctx.streams.get_session(session.session_id) is not session:
            raise RuntimeError("VLM session replaced")
        self._remember_session(session)
        self._processing_session = session
        try:
            try:
                result = self._run_window(start, end, frames)
                if ctx.streams.get_session(session.session_id) is not session:
                    raise RuntimeError("VLM session replaced")
            except VLMRecognitionError as exc:
                return self._skip_failed_window(session, start, end, exc)
            except Exception as exc:
                if (
                    ctx.streams.get_session(session.session_id) is not session
                    or session.session_id in self._cancelled_sessions
                ):
                    raise
                self._last_error = f"{type(exc).__name__}: {exc}"
                ctx.streams.seal_text(
                    session.session_id,
                    StreamKind.VISUAL,
                    start,
                    error={"start": start, "end": end, "error": self._last_error},
                )
                ctx.events.emit(
                    "vlm.error",
                    session_id=session.session_id,
                    start=start,
                    end=end,
                    error=self._last_error,
                )
                ctx.ui.refresh("vlm.error")
                raise
            return self._commit_window(start, end, result)
        finally:
            if threading.current_thread() is not self._worker:
                self._processing_session = None

    # ------------------------------------------------------------------
    # Recording mode (full-speed ingest + concurrent windows)
    # ------------------------------------------------------------------
    def enabled(self) -> bool:
        return bool(self._setting("vlm.enabled", True))

    def _coverage(self, session: Any) -> CoverageStore:
        """Per-session coverage of processed video windows."""

        self._remember_session(session)
        sid = str(getattr(session, "session_id", "") or "")
        with self._lock:
            store = self._coverage_cache.get(sid)
            if store is None:
                store = CoverageStore(self._ctx.data_dir, sid)
                self._coverage_cache[sid] = store
            return store

    @staticmethod
    def _should_skip(session: Any, start: float, end: float) -> bool:
        return skip_coverage(getattr(session, "meta", None), MediaKind.VIDEO).covers(
            start, end
        )

    def busy(self) -> bool:
        if not self.enabled():
            return False
        with self._lock:
            if self._recording_active or self._requests:
                return True
            ctx = self._ctx
            session = ctx.streams.current if ctx is not None else None
            if session is not None and session.session_id in self._cancelled_sessions:
                return False
            if session is None or not session.input_done:
                return False
            if str(session.meta.get("mode", "")) != ProcessingMode.FAST:
                return False
            cursor = float(self._recording_cursor.get(session.session_id, 0.0))
            return cursor + 1.0 < session.video.duration

    def flushing(self) -> bool:
        """True while the stop-time tail flush is still writing the log."""

        with self._lock:
            return self._flushing

    def _is_recording_session(self) -> bool:
        ctx = self._ctx
        if ctx is None:
            return False
        return (
            str((self._processing_session or ctx.streams.current).meta.get("mode", ""))
            == ProcessingMode.FAST
        )

    def _run_recording_session(self) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        session = self._processing_session or ctx.streams.current
        sid = session.session_id
        window = self._window_seconds()
        half = window / 2.0
        cadence = self._cadence(recording=True)
        with self._lock:
            done = session.visual_progress.final
        if done:
            # Same as the STT worker: idling must not hold the lock that the
            # UI's ``status()`` render needs on every poll.
            self._stop_event.wait(0.5)
            return
        with self._lock:
            self._recording_sessions.discard(sid)
            center = float(self._recording_cursor.get(sid, 0.0)) + half
            self._recording_active = True
            self._next_center = center
        self._processing_session = session

        try:
            while not self._stop_event.is_set():
                if (
                    sid in self._cancelled_sessions
                    or ctx.streams.get_session(sid) is not session
                ):
                    return
                available = session.video.duration
                # Frames between sampling windows are never used; keep the
                # last window so the end-of-media flush still has data.
                self._trim_before(session, min(center - half, available - window))
                if available + 1e-6 >= center + half:
                    start = max(0.0, center - half)
                    end = min(available, center + half)
                    if self._should_skip(session, start, end):
                        # Already processed in an earlier pass: drop the
                        # frames instead of sending them to the model again.
                        with self._lock:
                            center += cadence
                            self._recording_cursor[sid] = center - half
                            self._next_center = center
                        session.video.trim(max(0.0, end - 30.0))
                        ctx.streams.seal_text(
                            sid,
                            StreamKind.VISUAL,
                            max(0.0, min(center - half, available - window)),
                        )
                        continue
                    frames = session.video.read_frames(start, end)
                    if frames and end - start >= 1.0:
                        self._process_window(start, end, frames)
                    with self._lock:
                        center += cadence
                        self._recording_cursor[sid] = center - half
                        self._next_center = center
                    ctx.streams.seal_text(
                        sid,
                        StreamKind.VISUAL,
                        max(0.0, min(center - half, available - window)),
                    )
                    continue
                if (
                    session.video_input_done
                    or session.input_done
                    or session.ended_at is not None
                ):
                    with self._lock:
                        committed = self._cursor
                    if available > committed + 1.0:
                        tail_start = max(0.0, min(center - half, available - window))
                        if available - tail_start >= 2.0:
                            frames = session.video.read_frames(tail_start, available)
                            if frames:
                                self._process_window(tail_start, available, frames)
                                with self._lock:
                                    self._cursor = available
                                    self._recording_cursor[sid] = available
                    ctx.streams.seal_text(sid, StreamKind.VISUAL, available, final=True)
                    break
                return
        finally:
            with self._lock:
                if ctx.streams.get_session(sid) is session:
                    self._recording_cursor[sid] = max(
                        float(self._recording_cursor.get(sid, 0.0)),
                        center - half,
                    )
                    self._recording_sessions.add(sid)
                self._recording_active = False

    # ----------------------------------------------------------------
    # Event handlers
    # ----------------------------------------------------------------
    def _remember_session(self, session) -> None:
        sid = session.session_id
        if self._sessions.get(sid) is session:
            return
        with self._lock:
            if self._sessions.get(sid) is session:
                return
            self._sessions[sid] = session
            self._cancelled_sessions.discard(sid)
            self._recording_sessions.discard(sid)
            self._recording_cursor.pop(sid, None)
            self._coverage_cache.pop(sid, None)
            self._disabled_at.pop(sid, None)
            self._session_history.pop(sid, None)
            if self._prepared_key is not None and self._prepared_key[0] == sid:
                self._prepared_future.cancel()
                self._prepared_future = None
                self._prepared_key = None
            if self._last_session in ("", sid):
                self._last_session = ""
                self._cursor = 0.0
                self._next_center = self._window_seconds() / 2
                self._history = deque(maxlen=self._history_windows())
                self._latest_image_path = ""
                self._last_window = {}
                self._last_error = ""
                self._consecutive_errors = 0

    def _on_session_started(self, event: Event) -> None:
        session = self._ctx.streams.get_session(str(event.get("session_id", "")))
        if session is not None and event.get("started_at") != session.started_at:
            return
        if session is not None:
            self._remember_session(session)
            self._ctx.streams.set_text_enabled(
                session.session_id, StreamKind.VISUAL, self.enabled()
            )

    def _select_session(self):
        current = self._ctx.streams.current
        self._remember_session(current)
        for session in list(self._sessions.values()):
            if session.session_id in self._cancelled_sessions or session.meta.get(
                "input_failed"
            ):
                continue
            if self._ctx.streams.text_progress(
                session.session_id, StreamKind.VISUAL
            ).final:
                continue
            cursor = self._recording_cursor.get(session.session_id, 0.0)
            finished = (
                session.video_input_done
                or session.input_done
                or session.ended_at is not None
            )
            if (
                not finished
                and session.video.duration < cursor + self._window_seconds()
            ):
                continue
            if self._last_session != session.session_id:
                if self._last_session:
                    self._session_history[self._last_session] = self._history
                self._history = self._session_history.setdefault(
                    session.session_id, deque(maxlen=self._history_windows())
                )
                self._cursor = self._recording_cursor.get(session.session_id, 0.0)
                self._next_center = self._cursor + self._window_seconds() / 2
                self._last_session = session.session_id
            return session
        return None

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _window_seconds(self) -> float:
        try:
            value = float(self._setting("vlm.window_seconds", 20.0))
        except (TypeError, ValueError):
            value = 20.0
        return max(5.0, value)

    def _cadence(self, *, recording: bool) -> float:
        key = (
            "vlm.cadence_recording_seconds" if recording else "vlm.cadence_live_seconds"
        )
        default = 300.0 if recording else 180.0
        try:
            value = float(self._setting(key, default))
        except (TypeError, ValueError):
            value = default
        return max(self._window_seconds(), value)

    def _history_windows(self) -> int:
        try:
            value = int(self._setting("vlm.history_windows", 3))
        except (TypeError, ValueError):
            value = 3
        return max(0, value)

    def _snapshot_history(self) -> list[str]:
        limit = self._history_windows()
        with self._lock:
            if limit <= 0:
                return []
            if self._history.maxlen != limit:
                self._history = deque(self._history, maxlen=limit)
            return list(self._history)

    def _trim_before(self, session: Any, before: float) -> None:
        """Drop frames that no future window will read.

        With a 3/5-minute cadence most frames between windows are never used;
        trimming them keeps the MJPEG buffer bounded and lets the producer
        run ahead through the gate.
        """

        before = max(0.0, float(before))
        if before <= 0.0:
            return
        session.video.trim(before)
        gate = getattr(session, "video_gate", None)
        if gate is not None:
            gate.release(before)

    def _setting(self, key: str, default: Any = None) -> Any:
        ctx = self._ctx
        settings = ctx.service("settings") if ctx is not None else None
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
