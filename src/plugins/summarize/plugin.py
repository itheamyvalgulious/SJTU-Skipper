"""P4 Summarize plugin: incremental lecture note-taking via LLM agent."""

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
NOTES_HEADERS: dict[str, str] = {
    "knowledge.md": (
        "# 课堂知识笔记\n\n"
        "> 本文件记录课程知识性内容：定理、证明、概念、公式、例题等。\n"
        "> 格式：Markdown，数学公式使用 $...$（行内）或 $$...$$（行间）。\n"
        "> 每条记录附带时间戳。\n\n"
    ),
    "gossip.md": (
        "# 课堂闲话笔记\n\n"
        "> 本文件记录与课程知识无关的内容：世界局势、生活趣事、对学生的期待、\n"
        "> AI 对数学的冲击、个人经历分享等。\n"
        "> 格式：Markdown，数学公式使用 $...$（行内）或 $$...$$（行间）。\n"
        "> 每条记录附带时间戳。\n\n"
    ),
    "notice.md": (
        "# 课堂通知笔记\n\n"
        "> 本文件记录作业、考试、考核要求、课程安排等通知。\n"
        "> 格式：Markdown，数学公式使用 $...$（行内）或 $$...$$（行间）。\n"
        "> 每条记录附带时间戳。\n\n"
    ),
}

NOTE_FILES = ("knowledge.md", "gossip.md", "notice.md")


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------
class SummarizePlugin(Plugin):
    """Incrementally summarize lecture streams into three structured notes."""

    id = "summarize"
    requires = ("settings", "llm-agent")

    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self._settings_service = ctx.require("settings")
        self._agent_service = ctx.require("agent")

        # --- settings -------------------------------------------------------
        for spec in _SETTING_SPECS:
            self._settings_service.register(spec)

        # --- service --------------------------------------------------------
        ctx.register_service("summarize", self)

        # --- state ----------------------------------------------------------
        # Per-session watermarks: session_id -> {transcript:idx, visual:idx, audio:float}
        self._watermarks: dict[str, dict[str, int | float]] = {}
        self._retry_counts: dict[str, int] = {}

        # Worker guard: only one summary runs at a time across all sessions.
        self._worker_lock = threading.Lock()
        self._worker_busy = False
        self._stop_event = threading.Event()

        # --- events ---------------------------------------------------------
        ctx.events.define("summarize.triggered", "A summary run has started")
        ctx.events.define("summarize.updated", "Summary completed, watermarks advanced")
        ctx.events.define("summarize.error", "Summary failed after all retries")

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
        return int(self._settings_service.get("summarize.trigger_chars", 600))

    @property
    def _trigger_audio_seconds(self) -> int:
        return int(self._settings_service.get("summarize.trigger_audio_seconds", 120))

    @property
    def _min_chars(self) -> int:
        return int(self._settings_service.get("summarize.min_chars", 80))

    @property
    def _max_steps(self) -> int:
        return int(self._settings_service.get("summarize.max_steps", 8))

    @property
    def _model(self) -> str:
        return str(self._settings_service.get("summarize.model", "") or "")

    @property
    def _notes_dir_name(self) -> str:
        return str(self._settings_service.get("summarize.notes_dir", "notes"))

    @property
    def _system_prompt(self) -> str:
        return str(self._settings_service.get("summarize.system_prompt", SYSTEM_PROMPT) or SYSTEM_PROMPT)

    @property
    def _retry_max(self) -> int:
        return int(self._settings_service.get("summarize.retry_max", 2))

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        ui = self.ctx.ui

        # Ensure left tab "classroom" exists.
        if not ui.has_tab("classroom"):
            ui.add_tab("left", "classroom", "课堂", icon="📖", order=10)

        # Toolbox item for notes.
        if not ui.has_toolbox_item("summarize.notes"):
            ui.add_toolbox_item(
                "classroom",
                "summarize.notes",
                "课堂笔记",
                description="查看课堂笔记",
                order=20,
                render=self._render_toolbox_notes,
            )

        # Notes page.
        ui.add_page(
            "summarize:notes",
            "课堂笔记",
            render=self._render_notes_page,
            icon="📝",
            order=30,
        )

        # Action: force-summarize now.
        ui.register_action(
            "summarize.now", self._on_summarize_now, plugin_id=self.id
        )

    def _render_toolbox_notes(self) -> list[dict[str, Any]]:
        """Toolbox preview for the notes."""
        sid = self.ctx.streams.current.session_id
        note_paths = [
            (fname, self._note_path(sid, fname))
            for fname in NOTE_FILES
        ]
        existing = [
            (fname, p)
            for fname, p in note_paths
            if p.exists() and p.stat().st_size > len(NOTES_HEADERS[fname].encode("utf-8"))
        ]
        lines: list[str] = []
        if existing:
            lines.append(f"已有 {len(existing)} 个笔记文件包含内容")
        else:
            lines.append("笔记文件尚未建立或为空（开始课堂后自动创建）")
        return [
            {"type": "text", "text": "\n".join(lines)},
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "📝 打开笔记页",
                        "action": "ui.open_page",
                        "value": {"page_id": "summarize:notes"},
                    },
                    {"label": "⚡ 立即总结", "action": "summarize.now", "style": "primary"},
                ],
            },
        ]

    def _render_notes_page(self) -> list[dict[str, Any]]:
        """Render the notes page with three tabs (knowledge / gossip / notice)."""
        tab_labels = [
            ("knowledge.md", "知识"),
            ("gossip.md", "生活趣事"),
            ("notice.md", "通知"),
        ]
        tabs: list[dict[str, Any]] = []
        for filename, title in tab_labels:
            text = self._read_note_file(filename)
            if text is None:
                blocks: list[dict[str, Any]] = [
                    {"type": "text", "text": "（会话未开始）"}
                ]
            elif text == "":
                blocks = [{"type": "text", "text": "（暂无内容）"}]
            else:
                blocks = [{"type": "markdown", "text": text}]
            tabs.append({"title": title, "blocks": blocks})

        return [
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "立即总结",
                        "action": "summarize.now",
                        "style": "primary",
                    }
                ],
            },
            {"type": "tabs", "tabs": tabs},
        ]

    def _read_note_file(self, filename: str) -> str | None:
        """Return the current session's note file content, or None."""
        sid = self.ctx.streams.current.session_id
        path = self._note_path(sid, filename)
        if path.exists():
            return path.read_text("utf-8")
        # File hasn't been created yet – return empty string so UI shows placeholder.
        notes_dir = self.ctx.data_dir / "sessions" / sid / self._notes_dir_name
        if notes_dir.exists():
            return ""
        return None

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
        sid = self.ctx.streams.current.session_id
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
        for filename in NOTE_FILES:
            path = notes_dir / filename
            if not path.exists():
                path.write_text(NOTES_HEADERS[filename], "utf-8")

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
            name=f"summarize-{session_id[:8]}",
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
        self.ctx.events.emit("summarize.triggered", session_id=session_id)

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
                        "summarize attempt %d/%d failed for session %s, "
                        "retry in %.1fs",
                        attempt + 1,
                        max_attempts,
                        session_id,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    self.ctx.logger.error(
                        "summarize failed after %d attempts for session %s",
                        max_attempts,
                        session_id,
                    )

        with self._worker_lock:
            self._worker_busy = False

        if success:
            self._watermarks[session_id] = target_wm
            self._retry_counts.pop(session_id, None)
            self.ctx.events.emit("summarize.updated", session_id=session_id)
            self.ctx.ui.refresh("summarize.updated")
        else:
            self.ctx.events.emit("summarize.error", session_id=session_id)
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
        course_name = (
            self.ctx.meta.snapshot().get("course")
            or self.ctx.streams.current.meta.get("course")
            or "未知课程"
        )
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
        for filename in NOTE_FILES:
            path = notes_dir / filename
            if not path.exists():
                path.write_text(NOTES_HEADERS[filename], "utf-8")
            file_paths[filename] = str(path)

        # Snapshot file contents before run.
        hashes_before = {
            filename: _file_md5(path) for filename, path in file_paths.items()
        }

        # Create agent session and run.
        agent = self._agent_service
        model = self._model or None
        session = agent.create_session(
            name=f"summarize-{session_id[:12]}",
            system_prompt=self._system_prompt,
            files=file_paths,
            tools=["info", "read", "replace", "append"],
            model=model,
            max_steps=self._max_steps,
        )
        agent.run(session.id, user_prompt)

        # Check if files changed.
        hashes_after = {
            filename: _file_md5(path) for filename, path in file_paths.items()
        }
        if hashes_after == hashes_before:
            # Retry with stricter prompt.
            retry_session = agent.create_session(
                name=f"summarize-retry-{session_id[:12]}",
                system_prompt=self._system_prompt,
                files=file_paths,
                tools=["info", "read", "replace", "append"],
                model=model,
                max_steps=self._max_steps,
            )
            agent.run(retry_session.id, RETRY_PROMPT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_stream_items(items: list[Any]) -> str:
    """Format a list of TextItems into a readable text block with timestamps."""
    if not items:
        return "（无内容）"
    lines: list[str] = []
    for item in items:
        ts = ""
        if item.start is not None:
            ts = _format_seconds(item.start)
            if item.end is not None:
                ts += f" - {_format_seconds(item.end)}"
        prefix = f"[{ts}] " if ts else ""
        lines.append(f"{prefix}{item.text}")
    return "\n\n".join(lines)


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
        key="summarize.trigger_chars",
        label="触发字符数",
        type="int",
        default=600,
        group="Summarize",
        description="新增文字达到此数量时触发总结",
    ),
    SettingsSpec(
        key="summarize.trigger_audio_seconds",
        label="触发音频秒数",
        type="int",
        default=120,
        group="Summarize",
        description="新增音频达到此时长且文字足够时触发",
    ),
    SettingsSpec(
        key="summarize.min_chars",
        label="最小字符数",
        type="int",
        default=80,
        group="Summarize",
        description="音频触发时至少需要的文字量",
    ),
    SettingsSpec(
        key="summarize.max_steps",
        label="最大步数",
        type="int",
        default=8,
        group="Summarize",
        description="Agent 最大工具调用步数",
    ),
    SettingsSpec(
        key="summarize.model",
        label="模型",
        type="string",
        default="",
        group="Summarize",
        description="总结用模型（空=agent 默认）",
    ),
    SettingsSpec(
        key="summarize.notes_dir",
        label="笔记目录",
        type="string",
        default="notes",
        group="Summarize",
        description="笔记文件存放子目录名",
    ),
    SettingsSpec(
        key="summarize.system_prompt",
        label="系统提示词",
        type="text",
        default=SYSTEM_PROMPT,
        group="Summarize",
        description="Agent 系统提示词",
    ),
    SettingsSpec(
        key="summarize.retry_max",
        label="最大重试次数",
        type="int",
        default=2,
        group="Summarize",
        description="总结失败后最大重试次数",
    ),
]