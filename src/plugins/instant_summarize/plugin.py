"""Instant summarize plugin: incremental (real-time) structured lecture notes.

Content is merged into the notes as it arrives, whenever the configured
character / audio thresholds are reached.  The model is required to keep a
clear outline and may rewrite whole files to keep the structure coherent.
"""

from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from core.plugin import Plugin, PluginContext
from plugins.settings.spec import SettingsSpec

from .prompts import RETRY_PROMPT, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# ---------------------------------------------------------------------------
# Fixed file headers (written once when a session starts)
# ---------------------------------------------------------------------------
INSTANT_NOTE_HEADERS: dict[str, str] = {
    "instant_note_knowledge.md": (
        "# 课堂知识笔记（实时）\n\n"
        "> 本文件记录课程知识内容：定义、定理、证明、概念、公式、例题等。\n"
        "> 结构：按知识逻辑分章节组织；增量维护，必要时才重排章节；不得压缩细节。\n"
        "> 公式使用 $...$（行内）或 $$...$$（行间）。\n\n"
    ),
    "instant_note_gossip.md": (
        "# 课堂闲话笔记（实时）\n\n"
        "> 本文件记录与课程知识无关的内容：世界局势、生活趣事、对学生的期待、\n"
        "> AI 对数学的冲击、个人经历分享等。\n"
        "> 结构：按主题分章节组织；增量维护，必要时才重排章节；不得压缩细节。\n"
        "> 公式使用 $...$（行内）或 $$...$$（行间）。\n\n"
    ),
    "instant_note_notice.md": (
        "# 课堂通知笔记（实时）\n\n"
        "> 本文件记录作业、考试、考核要求、课程安排等通知。\n"
        "> 结构：按类型分章节组织；增量维护，必要时才重排章节；不得压缩细节。\n"
        "> 公式使用 $...$（行内）或 $$...$$（行间）。\n\n"
    ),
}

INSTANT_NOTE_FILES = tuple(INSTANT_NOTE_HEADERS)

_DOC_FILES = {
    "knowledge": "instant_note_knowledge.md",
    "gossip": "instant_note_gossip.md",
    "notice": "instant_note_notice.md",
}

_TOOLS = ["info", "read", "replace", "append", "write"]


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------
class InstantSummarizePlugin(Plugin):
    """Continuously merge lecture streams into three structured notes."""

    id = "instant-summarize"
    requires = ("settings", "llm-agent", "reader")

    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self._settings_service = ctx.require("settings")
        self._agent_service = ctx.require("agent")

        # --- settings -------------------------------------------------------
        for spec in _SETTING_SPECS:
            self._settings_service.register(spec)

        # --- service --------------------------------------------------------
        ctx.register_service("instant-summarize", self)

        # --- state ----------------------------------------------------------
        # Per-session watermarks: session_id -> {transcript:idx, visual:idx, audio:float}
        self._watermarks: dict[str, dict[str, int | float]] = {}
        self._retry_counts: dict[str, int] = {}

        # Worker guard: only one summary runs at a time across all sessions.
        self._worker_lock = threading.Lock()
        self._worker_busy = False
        self._stop_event = threading.Event()

        # --- events ---------------------------------------------------------
        ctx.events.define(
            "instant-summarize.triggered", "An incremental summary run started"
        )
        ctx.events.define(
            "instant-summarize.updated", "Summary completed, watermarks advanced"
        )
        ctx.events.define(
            "instant-summarize.error", "Summary failed after all retries"
        )

        ctx.events.on(
            "session.started", self._on_session_started, plugin_id=self.id
        )
        ctx.events.on(
            "session.stopped", self._on_session_stopped, plugin_id=self.id
        )
        ctx.events.on(
            "stream.appended", self._on_stream_appended, plugin_id=self.id
        )

        # --- UI -------------------------------------------------------------
        self._setup_ui()

    def teardown(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    @property
    def _trigger_chars(self) -> int:
        return int(
            self._settings_service.get("instant-summarize.trigger_chars", 600)
        )

    @property
    def _trigger_audio_seconds(self) -> int:
        return int(
            self._settings_service.get(
                "instant-summarize.trigger_audio_seconds", 120
            )
        )

    @property
    def _min_chars(self) -> int:
        return int(self._settings_service.get("instant-summarize.min_chars", 80))

    @property
    def _max_steps(self) -> int:
        return int(self._settings_service.get("instant-summarize.max_steps", 12))

    @property
    def _max_tool_output(self) -> int:
        return int(
            self._settings_service.get("instant-summarize.max_tool_output", 24000)
        )

    @property
    def _model(self) -> str:
        return str(self._settings_service.get("instant-summarize.model", "") or "")

    @property
    def _notes_dir_name(self) -> str:
        return str(
            self._settings_service.get("instant-summarize.notes_dir", "notes")
        )

    @property
    def _system_prompt(self) -> str:
        return str(
            self._settings_service.get("instant-summarize.system_prompt", SYSTEM_PROMPT)
            or SYSTEM_PROMPT
        )

    @property
    def _retry_max(self) -> int:
        return int(self._settings_service.get("instant-summarize.retry_max", 2))

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        ui = self.ctx.ui

        if not ui.has_tab("note"):
            ui.add_tab("right", "note", "笔记", icon="📘", order=20)
        if not ui.has_toolbox_item("instant-summarize.notes"):
            ui.add_toolbox_item(
                "note",
                "instant-summarize.notes",
                "实时笔记",
                description="课堂中持续合并的笔记",
                order=11,
                render=self._render_note_item,
            )

        ui.register_action(
            "instant-summarize.now", self._on_summarize_now, plugin_id=self.id
        )
        ui.register_action(
            "instant-summarize.read", self._on_read, plugin_id=self.id
        )

    # ------------------------------------------------------------------
    # Note tab
    # ------------------------------------------------------------------
    _DOC_LABELS = {
        "knowledge": "知识",
        "gossip": "生活趣事",
        "notice": "通知",
    }

    def _docs(self) -> list[dict[str, str]]:
        sid = self.ctx.streams.notes_session_id
        return [
            {
                "key": key,
                "label": label,
                "path": str(self._note_path(sid, _DOC_FILES[key])),
            }
            for key, label in self._DOC_LABELS.items()
        ]

    def _render_note_item(self) -> list[dict[str, Any]]:
        sid = self.ctx.streams.notes_session_id
        ready = sum(
            1
            for key in self._DOC_LABELS
            if _has_content(
                self._note_path(sid, _DOC_FILES[key]),
                INSTANT_NOTE_HEADERS[_DOC_FILES[key]],
            )
        )
        return [
            {"type": "text", "text": f"已记录 {ready}/{len(self._DOC_LABELS)} 篇"},
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": label,
                        "action": "instant-summarize.read",
                        "value": {"doc": key},
                        "class": "btn btn-sm",
                    }
                    for key, label in self._DOC_LABELS.items()
                ],
            },
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "立即总结",
                        "action": "instant-summarize.now",
                        "class": "btn btn-sm btn-primary",
                    }
                ],
            },
        ]

    def _on_read(self, payload: dict[str, Any]) -> dict[str, Any]:
        doc = str(payload.get("doc", "knowledge") or "knowledge")
        if doc not in _DOC_FILES:
            doc = "knowledge"
        reader = self.ctx.require("reader")
        reader.show(
            title=f"实时笔记 · {_course_name(self.ctx.streams.current)}",
            docs=self._docs(),
            active=doc,
        )
        return {"ok": True, "open_page": "reader:main"}

    def _note_path(self, session_id: str, filename: str) -> Path:
        return (
            self.ctx.data_dir
            / "sessions"
            / session_id
            / self._notes_dir_name
            / filename
        )

    def _on_summarize_now(self, _payload: dict[str, Any]) -> dict[str, Any]:
        """Action handler: force a summary for the current session."""
        sid = self.ctx.streams.notes_session_id
        streams = self.ctx.streams.current

        wm = self._ensure_watermarks(sid)
        new_chars = (
            streams.transcript.char_count_since(int(wm["transcript"]))
            + streams.visual.char_count_since(int(wm["visual"]))
        )
        if new_chars == 0 and streams.audio.duration <= float(wm["audio"]):
            return {"ok": True, "message": "暂无新内容需要总结", "refresh": False}

        triggered = self._maybe_trigger(sid, force=True)
        return {
            "ok": True,
            "message": "总结已开始" if triggered else "总结已在运行中",
            "refresh": True,
        }

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def _on_session_started(self, event: Any) -> None:
        sid = event.get("session_id", "")
        if not sid:
            return
        self._create_note_files(sid)
        # Reset watermarks for this session.
        self._watermarks[sid] = {
            "transcript": 0,
            "visual": 0,
            "audio": 0.0,
        }
        self._retry_counts.pop(sid, None)

    def _on_session_stopped(self, event: Any) -> None:
        sid = event.get("session_id", "")
        if not sid:
            return
        # Flush any remaining content.
        self._maybe_trigger(sid, force=True)

    def _create_note_files(self, session_id: str) -> None:
        notes_dir = (
            self.ctx.data_dir
            / "sessions"
            / session_id
            / self._notes_dir_name
        )
        notes_dir.mkdir(parents=True, exist_ok=True)
        for filename in INSTANT_NOTE_FILES:
            path = notes_dir / filename
            if not path.exists():
                path.write_text(INSTANT_NOTE_HEADERS[filename], "utf-8")

    # ------------------------------------------------------------------
    # Trigger logic
    # ------------------------------------------------------------------
    def _ensure_watermarks(self, session_id: str) -> dict[str, int | float]:
        if session_id not in self._watermarks:
            self._watermarks[session_id] = {
                "transcript": 0,
                "visual": 0,
                "audio": 0.0,
            }
        return self._watermarks[session_id]

    def _on_stream_appended(self, event: Any) -> None:
        sid = event.get("session_id", "")
        if not sid:
            return
        # Ensure we have watermarks for this session.
        self._ensure_watermarks(sid)
        # Check trigger (non-force: respect thresholds).
        self._maybe_trigger(sid, force=False)

    def _maybe_trigger(self, session_id: str, *, force: bool = False) -> bool:
        """Check trigger conditions and start worker if possible.

        Returns True if a summary was started (or already running).
        """
        wm = self._ensure_watermarks(session_id)
        streams = self.ctx.streams.current
        if streams.session_id != session_id:
            # Session has changed; skip (the old session is no longer current).
            return False
        if str(streams.meta.get("mode", "")) == "fast":
            # Recordings are processed at full speed and summarized once at
            # the end; incremental notes would just fire on every window.
            return False

        # Calculate deltas since last watermark.
        new_transcript_chars = streams.transcript.char_count_since(
            int(wm["transcript"])
        )
        new_visual_chars = streams.visual.char_count_since(int(wm["visual"]))
        total_new_chars = new_transcript_chars + new_visual_chars
        new_audio = streams.audio.duration - float(wm["audio"])

        # Check trigger conditions.
        should_trigger = force
        if not should_trigger:
            if total_new_chars >= self._trigger_chars:
                should_trigger = True
            elif (
                new_audio >= self._trigger_audio_seconds
                and total_new_chars >= self._min_chars
            ):
                should_trigger = True

        if not should_trigger:
            return False

        # If worker is already busy, the pending content will be picked up
        # when it finishes (watermarks haven't moved yet).
        if self._worker_busy:
            return False

        # Snapshot current positions *before* acquiring lock.
        target_wm = {
            "transcript": streams.transcript.snapshot_index(),
            "visual": streams.visual.snapshot_index(),
            "audio": streams.audio.duration,
        }
        # Don't run if nothing new (can happen with force + empty streams).
        if not force and (
            target_wm["transcript"] <= int(wm["transcript"])
            and target_wm["visual"] <= int(wm["visual"])
            and target_wm["audio"] <= float(wm["audio"])
        ):
            return False

        if not self._worker_lock.acquire(blocking=False):
            return False
        try:
            if self._worker_busy:
                return False
            self._worker_busy = True
        finally:
            self._worker_lock.release()

        # Reset retry count for fresh trigger.
        self._retry_counts[session_id] = 0

        base_wm = dict(wm)  # copy current watermarks as processing baseline

        t = threading.Thread(
            target=self._run_summary,
            args=(session_id, base_wm, target_wm, force),
            name=f"instant-summarize-{session_id[:8]}",
            daemon=True,
        )
        t.start()
        return True

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _run_summary(
        self,
        session_id: str,
        base_wm: dict[str, int | float],
        target_wm: dict[str, int | float],
        force: bool,
    ) -> None:
        self.ctx.events.emit("instant-summarize.triggered", session_id=session_id)

        max_attempts = self._retry_max + 1  # 1 initial + N retries
        success = False

        for attempt in range(max_attempts):
            try:
                self._run_summary_inner(session_id, base_wm, force)
                success = True
                break
            except Exception:
                if attempt < max_attempts - 1:
                    delay = min(2**attempt, 30)
                    self.ctx.logger.warning(
                        "instant summarize attempt %d/%d failed for session %s, "
                        "retry in %.1fs",
                        attempt + 1,
                        max_attempts,
                        session_id,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    self.ctx.logger.error(
                        "instant summarize failed after %d attempts for session %s",
                        max_attempts,
                        session_id,
                    )

        with self._worker_lock:
            self._worker_busy = False

        if success:
            self._watermarks[session_id] = target_wm
            self._retry_counts.pop(session_id, None)
            self.ctx.events.emit(
                "instant-summarize.updated", session_id=session_id
            )
            self.ctx.ui.refresh("instant-summarize.updated")
        else:
            self.ctx.events.emit("instant-summarize.error", session_id=session_id)
            # Advance watermarks to avoid infinite loops.
            self._watermarks[session_id] = target_wm
            self._retry_counts.pop(session_id, None)

        # Check if more content arrived while we were working.
        if not self._stop_event.is_set():
            self._maybe_trigger(session_id, force=False)

    def _run_summary_inner(
        self,
        session_id: str,
        base_wm: dict[str, int | float],
        force: bool,
    ) -> None:
        """Single summary attempt (may be retried on failure)."""
        streams = self.ctx.streams.current
        if streams.session_id != session_id:
            raise RuntimeError("Session changed during summary")

        # Gather new content.
        transcript_items = streams.transcript.items_since(
            int(base_wm["transcript"])
        )
        visual_items = streams.visual.items_since(int(base_wm["visual"]))
        audio_start = float(base_wm["audio"])
        audio_end = streams.audio.duration

        transcript_text = _format_stream_items(transcript_items)
        visual_text = _format_stream_items(visual_items)

        time_start = _format_seconds(audio_start)
        time_end = _format_seconds(audio_end)

        if not transcript_text.strip() and not visual_text.strip():
            # Nothing to summarize (even a manual/stop flush).  Do not ask the
            # model to write placeholder noise into the note files.
            return

        # Build user prompt.
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        course_name = _course_name(self.ctx.streams.current)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            session_id=session_id,
            date=now,
            course_name=course_name,
            time_start=time_start,
            time_end=time_end,
            transcript_text=transcript_text,
            visual_text=visual_text,
        )

        # Create notes directory (in case session started before we created files).
        notes_dir = (
            self.ctx.data_dir
            / "sessions"
            / session_id
            / self._notes_dir_name
        )
        notes_dir.mkdir(parents=True, exist_ok=True)

        # Ensure files exist.
        file_paths: dict[str, str] = {}
        for filename in INSTANT_NOTE_FILES:
            path = notes_dir / filename
            if not path.exists():
                path.write_text(INSTANT_NOTE_HEADERS[filename], "utf-8")
            file_paths[filename] = str(path)

        # Snapshot file contents before run.
        hashes_before = {
            filename: _file_md5(path) for filename, path in file_paths.items()
        }

        # Create agent session and run.
        agent = self._agent_service
        model = self._model or None
        session = agent.create_session(
            name=f"instant-summarize-{session_id[:12]}",
            system_prompt=self._system_prompt,
            files=file_paths,
            tools=list(_TOOLS),
            model=model,
            max_steps=self._max_steps,
            max_tool_output=self._max_tool_output,
        )
        agent.run(session.id, user_prompt)

        # Check if files changed.
        hashes_after = {
            filename: _file_md5(path) for filename, path in file_paths.items()
        }
        if hashes_after == hashes_before:
            # Retry with stricter prompt.
            retry_session = agent.create_session(
                name=f"instant-summarize-retry-{session_id[:12]}",
                system_prompt=self._system_prompt,
                files=file_paths,
                tools=list(_TOOLS),
                model=model,
                max_steps=self._max_steps,
                max_tool_output=self._max_tool_output,
            )
            agent.run(retry_session.id, RETRY_PROMPT)




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _course_name(streams: Any) -> str:
    meta = getattr(streams, "meta", {}) or {}
    name: Any = meta.get("course_name") or meta.get("course")
    if isinstance(name, dict):
        name = name.get("course_name") or name.get("name") or ""
    text = str(name or "").strip()
    return text or "未知课程"


def _has_content(path: Path, header: str) -> bool:
    try:
        return path.stat().st_size > len(header.encode("utf-8"))
    except OSError:
        return False


def _format_stream_items(items: list[Any]) -> str:
    """Format a list of TextItems into readable blocks, tagged by source.

    Timestamps are intentionally omitted: notes must not be a timestamped log,
    and the stream timestamps are frequently unreliable (0.0 for VLM items).
    """
    if not items:
        return ""
    blocks: list[str] = []
    for item in items:
        text = str(getattr(item, "text", "") or "").strip()
        if not text:
            continue
        source = str(getattr(item, "source", "") or "")
        if source == "vlm":
            blocks.append(f"【板书】{text}")
        elif source == "stt":
            blocks.append(f"【讲授】{text}")
        else:
            blocks.append(text)
    return "\n\n".join(blocks)


def _format_seconds(seconds: float) -> str:
    """Format float seconds as HH:MM:SS."""
    s = max(0.0, seconds)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _file_md5(path: str) -> str:
    """Return hex MD5 digest of a file's contents."""
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except FileNotFoundError:
        return ""


# ---------------------------------------------------------------------------
# Setting specs
# ---------------------------------------------------------------------------
_SETTING_SPECS: list[SettingsSpec] = [
    SettingsSpec(
        key="instant-summarize.trigger_chars",
        label="触发字符数",
        type="int",
        default=600,
        group="Instant Summarize",
        description="新增文字达到此数量时触发总结",
    ),
    SettingsSpec(
        key="instant-summarize.trigger_audio_seconds",
        label="触发音频秒数",
        type="int",
        default=120,
        group="Instant Summarize",
        description="新增音频达到此时长且文字足够时触发",
    ),
    SettingsSpec(
        key="instant-summarize.min_chars",
        label="最小字符数",
        type="int",
        default=80,
        group="Instant Summarize",
        description="音频触发时至少需要的文字量",
    ),
    SettingsSpec(
        key="instant-summarize.max_steps",
        label="最大步数",
        type="int",
        default=12,
        group="Instant Summarize",
        description="Agent 最大工具调用步数",
    ),
    SettingsSpec(
        key="instant-summarize.max_tool_output",
        label="工具输出上限",
        type="int",
        default=24000,
        group="Instant Summarize",
        description="单次 read 输出上限（字符），需要足够大才能整篇重写",
    ),
    SettingsSpec(
        key="instant-summarize.model",
        label="模型",
        type="string",
        default="",
        group="Instant Summarize",
        description="总结用模型（空=agent 默认）",
    ),
    SettingsSpec(
        key="instant-summarize.notes_dir",
        label="笔记目录",
        type="string",
        default="notes",
        group="Instant Summarize",
        description="笔记文件存放子目录名",
    ),
    SettingsSpec(
        key="instant-summarize.system_prompt",
        label="系统提示词",
        type="text",
        default=SYSTEM_PROMPT,
        group="Instant Summarize",
        description="Agent 系统提示词",
    ),
    SettingsSpec(
        key="instant-summarize.retry_max",
        label="最大重试次数",
        type="int",
        default=2,
        group="Instant Summarize",
        description="总结失败后最大重试次数",
    ),
]
