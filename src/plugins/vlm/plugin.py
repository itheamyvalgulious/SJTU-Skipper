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
import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image

from core.coverage import CoverageStore, skip_coverage
from core.events import Event
from core.pipeline import WindowError
from core.plugin import Plugin, PluginContext
from core.streams import VideoFrame
from plugins.settings.spec import SettingsSpec

from .fusion import fuse_frames
from .prompts import (
    BLANK_BOARD_MARKER,
    BOARD_RECOGNITION_PROMPT,
    NO_CHANGE_MARKER,
    build_user_message,
    collapse_repetitions,
    is_trivial_board_text,
)

logger = logging.getLogger("lecture_agent.plugin.vlm")

# The VLM plugin owns this file exclusively: one JSON object per written
# blackboard item, under ``data_dir/sessions/<session>/``.  The summarize
# plugin reads it later to rebuild the lecture material, so board text that
# was already recognised survives a crash, an early stop or a handout rerun.
VISUAL_LOG = "visual.jsonl"


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
        self._persist_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._cursor: float = 0.0  # seconds consumed in current session
        self._last_session: str = ""
        self._consecutive_errors = 0
        self._last_visual_text: str = ""
        self._latest_image_path: str = ""
        self._last_error: str = ""
        self._last_window: dict[str, Any] = {}
        self._recording_sessions: set[str] = set()
        self._recording_cursor: dict[str, float] = {}
        self._recording_active = False
        # True while the stop-time tail flush runs; the summarize plugin waits
        # for it before reading visual.jsonl.
        self._flushing = False
        self._processing_session: Any = None
        self._coverage_cache: dict[str, CoverageStore] = {}
        self._shutting_down = False
        self._history: deque[str] = deque(maxlen=3)
        self._next_center: float = 10.0

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
                description="每次融合识别的视频窗口长度；窗口以采样点为中心，前后各占一半",
            ),
            SettingsSpec(
                key="vlm.cadence_live_seconds",
                label="直播采样间隔（秒）",
                type="float",
                default=180.0,
                group="VLM",
                description="直播时每隔多少秒取一个 20 秒窗口（默认 3 分钟）",
            ),
            SettingsSpec(
                key="vlm.cadence_recording_seconds",
                label="录播采样间隔（秒）",
                type="float",
                default=300.0,
                group="VLM",
                description="录播全速处理时每隔多少秒取一个 20 秒窗口（默认 5 分钟）",
            ),
            SettingsSpec(
                key="vlm.history_windows",
                label="历史转录条数",
                type="int",
                default=3,
                group="VLM",
                description="把最近几次转录作为纯文本上下文，供模型输出增量",
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
                key="vlm.concurrency",
                label="录播并发数",
                type="int",
                default=1,
                group="VLM",
                description=(
                    "录播板书窗口按时间顺序串行处理（固定为 1），"
                    "以保证最近几次转录的顺序正确；该键保留用于兼容旧配置。"
                ),
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
                description=(
                    "完整提示词，可编辑；`{timestamp}` 替换为时间范围，"
                    "`{history_block}` / `{output_rule}` 替换为历史转录与增量要求"
                    "（缺失时自动追加）。留空会报错。"
                ),
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
                covered = self._coverage(session).covered_seconds("video")
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
        """Manually process one window (used by tests and UI actions)."""
        return self._process_window(start, end, frames, force=True)

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
            last = dict(self._last_window)

        if image_path:
            blocks.append(
                {
                    "type": "image",
                    "src": f"/api/artifacts/{self._ctx.streams.current.session_id}/vlm/{Path(image_path).name}",
                    "alt": "最新融合板书",
                }
            )
        if last:
            stamp = f"{_fmt_time(last.get('start', 0))}–{_fmt_time(last.get('end', 0))}"
            blocks.append(
                {
                    "type": "text",
                    "text": f"最近窗口 {stamp}：{last.get('note', '')}"
                    + (f"\n\n模型原始回复：\n{last['text']}" if last.get("text") else ""),
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
    def _ask_image(
        self,
        image_bytes: bytes,
        timestamp: str,
        *,
        history: list[str] | None = None,
    ) -> str:
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
        # One user message with the prompt + image.  Previous transcriptions
        # are passed as text (never as extra images) so the model can emit a
        # delta instead of re-stating the whole board.
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": build_user_message(
                    timestamp=timestamp,
                    image_data_url=data_url,
                    prompt_text=prompt_text,
                    history=history or (),
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
                if self._is_recording_session():
                    self._run_recording_session()
                else:
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
        session = hub.current
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
                self._last_visual_text = ""
                self._next_center = half
                self._history.clear()
            center = self._next_center

        # Frames between windows are never used: drop them as they arrive so
        # a 3-minute cadence does not buffer minutes of MJPEG.
        self._trim_before(
            session, min(center - half, video.duration - window)
        )

        # Wait for enough video to accumulate
        if video.duration + 1e-9 < center + half:
            # Not enough data yet; check if session has ended
            if session.ended_at is not None:
                with self._lock:
                    self._flushing = True
                try:
                    self._flush_remaining()
                finally:
                    with self._lock:
                        self._flushing = False
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

    def _flush_remaining(self) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        session = self._processing_session or ctx.streams.current
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

    def _run_window(
        self, start: float, end: float, frames: list[VideoFrame]
    ) -> dict[str, Any]:
        """Fuse + save + ask the model.  Safe to run on pool threads."""

        if not frames:
            return {"image_path": "", "text": "", "empty": True}
        frames = self._sample_frames(frames)

        do_fusion = self._setting("vlm.fusion", True)
        if do_fusion:
            fused = fuse_frames(
                frames, crop=(0.0, 0.0, 1.0, 1.0), max_width=0, align=True
            )
        else:
            fused = _decode_single(frames[-1].data)

        session = self._processing_session or self._ctx.streams.current
        vlm_dir = self._ctx.data_dir / "sessions" / session.session_id / "vlm"
        vlm_dir.mkdir(parents=True, exist_ok=True)
        image_path = vlm_dir / f"{int(start):06d}-{int(end):06d}.jpg"
        Image.fromarray(fused).save(str(image_path), quality=92)

        with self._lock:
            self._latest_image_path = str(image_path)

        timestamp = f"{_fmt_time(start)}–{_fmt_time(end)}"
        try:
            text = self._ask_image(
                image_path.read_bytes(), timestamp, history=self._snapshot_history()
            )
        except Exception as exc:  # noqa: BLE001 - reported by commit
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            raise
        return {"image_path": str(image_path), "text": text}

    def _commit_window(self, start: float, end: float, result: Any) -> dict[str, Any]:
        """Deduplicate and append one window, in delivery order."""

        ctx = self._ctx
        session = self._processing_session or ctx.streams.current
        if isinstance(result, WindowError):
            session.video_gate.release(end)
            session.video.trim(max(0.0, float(end) - 30.0))
            ctx.events.emit(
                "vlm.error",
                session_id=session.session_id,
                start=start,
                end=end,
                error=f"{type(result.error).__name__}: {result.error}",
            )
            ctx.ui.refresh("vlm.error")
            return {"ok": False, "start": start, "end": end, "error": str(result.error)}

        text = str(result.get("text", "") or "")
        image_path = str(result.get("image_path", "") or "")

        # The window was processed; record it even when the board was empty.
        self._coverage(session).mark("video", start, end)

        ctx.events.emit(
            "vlm.segment",
            session_id=session.session_id,
            start=start,
            end=end,
            chars=len(text),
            model=self._setting("vlm.model", "deepseek-flash"),
        )

        threshold = self._setting("vlm.dedupe_threshold", 0.92)
        with self._lock:
            last_text = self._last_visual_text

        if not text:
            note = "模型返回空"
        elif text.strip() == NO_CHANGE_MARKER:
            note = "与上一窗口无变化，已跳过"
        elif text == BLANK_BOARD_MARKER or is_trivial_board_text(text):
            note = "未识别到板书（空白黑板或仅示意图）"
        elif last_text and difflib.SequenceMatcher(None, last_text, text).ratio() >= threshold:
            note = "与上一窗口重复，已跳过"
        else:
            note = "已写入板书"

        with self._lock:
            self._last_window = {
                "start": start,
                "end": end,
                "note": note,
                "chars": len(text),
                "text": text[:400],
            }
        logger.info(
            "VLM window %s–%s: %s (%d chars)",
            _fmt_time(start),
            _fmt_time(end),
            note,
            len(text),
        )

        if note == "已写入板书":
            visual = session.visual
            item = visual.append(
                text,
                start=start,
                end=end,
                source="vlm",
                meta={
                    "image_path": image_path,
                    "model": self._setting("vlm.model", "deepseek-flash"),
                },
            )
            if item is not None:
                self._persist_item(session.session_id, item)
            with self._lock:
                self._last_visual_text = text
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
            "skipped": note != "已写入板书",
            "note": note,
        }

    def _process_window(
        self,
        start: float,
        end: float,
        frames: list[VideoFrame],
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        ctx = self._ctx
        self._processing_session = ctx.streams.current
        try:
            result = self._run_window(start, end, frames)
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
        return self._commit_window(start, end, result)

    # ------------------------------------------------------------------
    # Recording mode (full-speed ingest + concurrent windows)
    # ------------------------------------------------------------------
    def enabled(self) -> bool:
        return bool(self._setting("vlm.enabled", True))

    def _coverage(self, session: Any) -> CoverageStore:
        """Per-session coverage of processed video windows."""

        sid = str(getattr(session, "session_id", "") or "")
        with self._lock:
            store = self._coverage_cache.get(sid)
            if store is None:
                store = CoverageStore(self._ctx.data_dir, sid)
                self._coverage_cache[sid] = store
            return store

    @staticmethod
    def _should_skip(session: Any, start: float, end: float) -> bool:
        return skip_coverage(getattr(session, "meta", None), "video").covers(
            start, end
        )

    def busy(self) -> bool:
        if not self.enabled():
            return False
        with self._lock:
            if self._recording_active:
                return True
            ctx = self._ctx
            session = ctx.streams.current if ctx is not None else None
            if session is None or not session.input_done:
                return False
            if str(session.meta.get("mode", "")) != "fast":
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
        return str(ctx.streams.current.meta.get("mode", "")) == "fast"

    def _run_recording_session(self) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        session = ctx.streams.current
        sid = session.session_id
        window = self._window_seconds()
        half = window / 2.0
        cadence = self._cadence(recording=True)
        with self._lock:
            done = sid in self._recording_sessions and (
                session.input_done or session.ended_at is not None
            )
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
                if ctx.streams.current.session_id != sid:
                    break
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
                        continue
                    frames = session.video.read_frames(start, end)
                    if frames and end - start >= 1.0:
                        self._process_window(start, end, frames)
                    with self._lock:
                        center += cadence
                        self._recording_cursor[sid] = center - half
                        self._next_center = center
                    continue
                if session.input_done or session.ended_at is not None:
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
                    break
                self._stop_event.wait(0.2)
        finally:
            with self._lock:
                self._recording_cursor[sid] = max(
                    float(self._recording_cursor.get(sid, 0.0)),
                    center - half,
                )
                self._recording_sessions.add(sid)
                self._recording_active = False

    # ----------------------------------------------------------------
    # Event handlers
    # ----------------------------------------------------------------
    def _on_session_started(self, event: Event) -> None:
        with self._lock:
            self._cursor = 0.0
            self._last_session = event.get("session_id", "")
            self._last_visual_text = ""
            self._latest_image_path = ""
            self._last_window = {}
            self._consecutive_errors = 0
            self._next_center = self._window_seconds() / 2.0
            self._history.clear()

    def _on_session_stopped(self, event: Event) -> None:
        ctx = self._ctx
        if ctx is not None and str(ctx.streams.current.meta.get("mode", "")) == "fast":
            # The recording loop already flushed the tail window.
            return
        with self._lock:
            if self._recording_active:
                return
        try:
            self._flush_remaining()
        except Exception:
            logger.exception("VLM flush on session stop failed")

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
            "vlm.cadence_recording_seconds"
            if recording
            else "vlm.cadence_live_seconds"
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