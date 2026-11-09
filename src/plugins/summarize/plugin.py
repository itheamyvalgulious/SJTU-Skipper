"""Summarize plugin: non-instant, whole-lecture structured notes.

Unlike the instant plugin, this one does not run during the lecture.  It runs
once over the entire session (manually via the UI action, or automatically
when the session stops).

The whole transcript + blackboard record is written to a single material file
and handed to one agent run.  The agent reads it in slices (its older tool
results are spilled to hidden files by the agent runtime) and maintains the
three note files incrementally; there is no batching and no whole-document
rewrite pass.
"""

from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path
from typing import Any

from core.plugin import Plugin, PluginContext
from plugins.settings.spec import SettingsSpec

from .prompts import RETRY_PROMPT, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# ---------------------------------------------------------------------------
# Fixed file headers (written once when a session starts)
# ---------------------------------------------------------------------------
NOTE_HEADERS: dict[str, str] = {
    "note_knowledge.md": (
        "# 课堂讲义：知识\n\n"
        "> 由整堂课的完整记录整理而成：定义、定理、证明、例题、方法。\n"
        "> 结构：按知识逻辑分章节组织；细节完整，不作压缩。\n"
        "> 公式使用 $...$（行内）或 $$...$$（行间）。\n\n"
    ),
    "note_gossip.md": (
        "# 课堂讲义：闲话与分享\n\n"
        "> 记录与课程知识无关的内容：时事、生活趣事、个人经历、对学生的期待等。\n"
        "> 结构：按主题分章节组织；细节完整，不作压缩。\n"
        "> 公式使用 $...$（行内）或 $$...$$（行间）。\n\n"
    ),
    "note_notice.md": (
        "# 课堂讲义：通知\n\n"
        "> 记录作业、考试、考核要求、课程安排等通知。\n"
        "> 结构：按类型分章节组织；细节完整，不作压缩。\n"
        "> 公式使用 $...$（行内）或 $$...$$（行间）。\n\n"
    ),
}

NOTE_FILES = tuple(NOTE_HEADERS)

_TOOLS = ["info", "read", "replace", "append"]

# The raw lecture record handed to the agent (kept next to ``notes/``).
SOURCE_FILENAME = "lecture_source.md"

_DOC_FILES = {
    "knowledge": "note_knowledge.md",
    "gossip": "note_gossip.md",
    "notice": "note_notice.md",
}



# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------
class SummarizePlugin(Plugin):
    """One-shot whole-lecture summarizer producing structured handouts."""

    id = "summarize"
    requires = ("settings", "llm-agent", "reader")

    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self._settings_service = ctx.require("settings")
        self._agent_service = ctx.require("agent")

        for spec in _SETTING_SPECS:
            self._settings_service.register(spec)

        ctx.register_service("summarize", self)

        self._job_lock = threading.Lock()
        self._job_busy = False
        self._pending_session = ""
        self._stop_event = threading.Event()
        self._status = "尚未生成讲义"

        ctx.events.define("summarize.started", "A whole-lecture summary started")
        ctx.events.define("summarize.finished", "Whole-lecture summary finished")
        ctx.events.define("summarize.error", "Whole-lecture summary failed")

        ctx.events.on(
            "session.started", self._on_session_started, plugin_id=self.id
        )
        ctx.events.on(
            "session.stopped", self._on_session_stopped, plugin_id=self.id
        )

        self._setup_ui()

    def teardown(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    @property
    def _max_tool_output(self) -> int:
        return max(10000, int(self._settings_service.get("summarize.max_tool_output", 200000)))

    @property
    def _max_history_chars(self) -> int:
        return max(0, int(self._settings_service.get("summarize.max_history_chars", 0)))

    @property
    def _max_tokens(self) -> int:
        return max(2000, int(self._settings_service.get("summarize.max_tokens", 32000)))

    @property
    def _model(self) -> str:
        return str(self._settings_service.get("summarize.model", "") or "")

    @property
    def _notes_dir_name(self) -> str:
        return str(self._settings_service.get("summarize.notes_dir", "notes"))

    @property
    def _system_prompt(self) -> str:
        return str(
            self._settings_service.get("summarize.system_prompt", SYSTEM_PROMPT)
            or SYSTEM_PROMPT
        )

    @property
    def _retry_max(self) -> int:
        return int(self._settings_service.get("summarize.retry_max", 1))

    @property
    def _auto_on_stop(self) -> bool:
        value = self._settings_service.get("summarize.auto_on_stop", True)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}

    @property
    def busy(self) -> bool:
        return self._job_busy

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        ui = self.ctx.ui

        # Right sidebar: the shared "note" area, one item per notes plugin.
        if not ui.has_tab("note"):
            ui.add_tab("right", "note", "笔记", icon="📘", order=20)
        if not ui.has_toolbox_item("summarize.notes"):
            ui.add_toolbox_item(
                "note",
                "summarize.notes",
                "课堂讲义",
                description="整堂课整理出的结构化讲义",
                order=10,
                render=self._render_note_item,
            )

        ui.register_action(
            "summarize.full", self._on_summarize_full, plugin_id=self.id
        )
        ui.register_action(
            "summarize.read", self._on_read, plugin_id=self.id
        )

    # ------------------------------------------------------------------
    # Note tab
    # ------------------------------------------------------------------
    _DOC_LABELS = {
        "knowledge": "知识",
        "gossip": "闲话与分享",
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
            if _has_content(self._note_path(sid, _DOC_FILES[key]), NOTE_HEADERS[_DOC_FILES[key]])
        )
        blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": self._status if ready == 0 else f"已生成 {ready}/{len(self._DOC_LABELS)} 篇",
            },
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": label,
                        "action": "summarize.read",
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
                        "label": "生成完整讲义",
                        "action": "summarize.full",
                        "class": "btn btn-sm btn-primary",
                    }
                ],
            },
        ]
        return blocks

    def _on_read(self, payload: dict[str, Any]) -> dict[str, Any]:
        doc = str(payload.get("doc", "knowledge") or "knowledge")
        if doc not in _DOC_FILES:
            doc = "knowledge"
        reader = self.ctx.require("reader")
        page_id = reader.show(
            title=f"课堂讲义 · {self._course_name()}",
            docs=self._docs(),
            active=doc,
            page_key=self.ctx.streams.notes_session_id,
        )
        return {"ok": True, "open_page": page_id}

    def _course_name(self) -> str:
        meta = self.ctx.streams.notes_meta
        if meta:
            return _course_name(SimpleNamespace(meta=meta))
        return _course_name(self.ctx.streams.current)

    def _note_path(self, session_id: str, filename: str) -> Path:
        return (
            self.ctx.data_dir
            / "sessions"
            / session_id
            / self._notes_dir_name
            / filename
        )

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------
    def _on_summarize_full(self, payload: dict[str, Any]) -> dict[str, Any]:
        sid = str(
            payload.get("session_id") or self.ctx.streams.notes_session_id
        )
        if self.busy:
            # Queued behind the running job; its material is already on disk.
            self.start(sid)
            return {"ok": True, "message": "讲义生成已在运行中，本次已排队", "refresh": True}
        started = self.start(sid)
        if started:
            return {"ok": True, "message": "完整讲义生成已开始", "refresh": True}
        return {"ok": True, "message": "暂无内容可整理", "refresh": False}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def _on_session_started(self, event: Any) -> None:
        sid = event.get("session_id", "")
        if not sid:
            return
        self._create_note_files(sid)
        self._status = "课堂进行中，讲义将在课后生成"

    def _on_session_stopped(self, event: Any) -> None:
        sid = event.get("session_id", "")
        if not sid:
            return
        if not self._auto_on_stop:
            self.ctx.logger.info("auto summarize disabled; skipping %s", sid)
            return
        started = self.start(sid)
        self.ctx.logger.info(
            "auto summarize for %s: %s", sid, "started" if started else "skipped"
        )

    def _create_note_files(self, session_id: str) -> None:
        notes_dir = (
            self.ctx.data_dir / "sessions" / session_id / self._notes_dir_name
        )
        notes_dir.mkdir(parents=True, exist_ok=True)
        for filename in NOTE_FILES:
            path = notes_dir / filename
            if not path.exists():
                path.write_text(NOTE_HEADERS[filename], "utf-8")

    # ------------------------------------------------------------------
    # Job control
    # ------------------------------------------------------------------
    def start(self, session_id: str) -> bool:
        """Start a whole-lecture summary job. Returns False when not started."""
        if not session_id:
            return False
        streams = self.ctx.streams.current
        material = ""
        if streams.session_id == session_id:
            material = _collect_material(streams)
        if not material.strip():
            # The app may have restarted since the lecture: the raw record is
            # still on disk next to the notes.
            material = self._load_source(session_id)
        if not material.strip():
            self.ctx.logger.info("summarize skipped: no transcript/visual content")
            return False
        if self._stop_event.is_set():
            return False

        with self._job_lock:
            if self._job_busy:
                # One handout job at a time; run this session afterwards
                # (its lecture_source.md is already on disk).
                self._pending_session = session_id
                return True
            self._job_busy = True

        # Snapshot everything the job needs now: the current stream may be
        # replaced by a new session as soon as the old one stops.
        course_name = (
            _course_name(streams) if streams.session_id == session_id else "未知课程"
        )
        duration = streams.audio.duration if streams.session_id == session_id else 0.0

        self._status = "正在生成完整讲义…"
        thread = threading.Thread(
            target=self._run_full,
            args=(session_id, material, course_name, duration),
            name=f"summarize-{session_id[:8]}",
            daemon=True,
        )
        thread.start()
        return True

    def _load_source(self, session_id: str) -> str:
        path = self.ctx.data_dir / "sessions" / session_id / SOURCE_FILENAME
        try:
            return path.read_text("utf-8")
        except OSError:
            return ""

    def _run_full(
        self,
        session_id: str,
        material: str,
        course_name: str,
        duration: float,
    ) -> None:
        self.ctx.events.emit("summarize.started", session_id=session_id)
        self.ctx.ui.refresh("summarize.started")
        success = False
        try:
            self._run_once(session_id, material, course_name, duration)
            success = True
            self._status = "讲义已生成"
        except Exception:
            self.ctx.logger.exception(
                "whole-lecture summarize failed for session %s", session_id
            )
            self._status = "讲义生成失败"
        finally:
            with self._job_lock:
                self._job_busy = False
                pending = self._pending_session
                self._pending_session = ""

        if success:
            self.ctx.events.emit("summarize.finished", session_id=session_id)
            self.ctx.ui.refresh("summarize.finished")
        else:
            self.ctx.events.emit("summarize.error", session_id=session_id)
            self.ctx.ui.refresh("summarize.error")

        if pending:
            self.ctx.logger.info("summarize queued session %s", pending)
            self.start(pending)

    def _run_once(
        self,
        session_id: str,
        material: str,
        course_name: str,
        duration: float,
    ) -> None:
        """Hand the whole record to one agent run; it reads and writes itself."""

        session_dir = self.ctx.data_dir / "sessions" / session_id
        notes_dir = session_dir / self._notes_dir_name
        notes_dir.mkdir(parents=True, exist_ok=True)

        file_paths: dict[str, str] = {}
        for filename in NOTE_FILES:
            path = notes_dir / filename
            if not path.exists():
                path.write_text(NOTE_HEADERS[filename], "utf-8")
            file_paths[filename] = str(path)

        source_path = session_dir / SOURCE_FILENAME
        source_path.write_text(material, "utf-8")
        file_paths[SOURCE_FILENAME] = str(source_path)

        prompt = USER_PROMPT_TEMPLATE.format(
            course_name=course_name,
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            session_id=session_id,
            time_start=_format_seconds(0.0),
            time_end=_format_seconds(duration),
            source=SOURCE_FILENAME,
            chars=len(material),
            lines=material.count("\n") + 1,
        )

        agent = self._agent_service
        model = self._model or None
        hashes_before = {
            filename: _file_md5(path)
            for filename, path in file_paths.items()
            if filename != SOURCE_FILENAME
        }

        session = agent.create_session(
            name=f"summarize-{session_id[:12]}",
            system_prompt=self._system_prompt,
            files=file_paths,
            tools=list(_TOOLS),
            model=model,
            max_steps=0,  # unlimited: the model runs until the handout is done
            max_tool_output=self._max_tool_output,
            max_history_chars=self._max_history_chars,
            max_tokens=self._max_tokens,
            llm_extra={"reasoning_effort": "max"},
        )
        agent.run(session.id, prompt)

        hashes_after = {
            filename: _file_md5(str(file_paths[filename]))
            for filename in hashes_before
        }
        if hashes_after == hashes_before:
            retry = agent.create_session(
                name=f"summarize-{session_id[:12]}-retry",
                system_prompt=self._system_prompt,
                files=file_paths,
                tools=list(_TOOLS),
                model=model,
                max_steps=0,  # unlimited: the model runs until the handout is done
                max_tool_output=self._max_tool_output,
                max_history_chars=self._max_history_chars,
                max_tokens=self._max_tokens,
                llm_extra={"reasoning_effort": "max"},
            )
            agent.run(retry.id, RETRY_PROMPT)

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


def _fmt_clock(seconds: float) -> str:
    total = max(0, int(seconds or 0))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _collect_material(streams: Any) -> str:
    """Render transcript + blackboard items into one chronological document."""

    entries: list[tuple[float, int, str]] = []
    for order, item in enumerate(streams.transcript.items_since(0)):
        text = str(getattr(item, "text", "") or "").strip()
        if not text:
            continue
        start = _item_start(item)
        stamp = f"[{_fmt_clock(start or 0.0)}]"
        entries.append((start or 0.0, order, f"{stamp} 讲授：{text}"))
    offset = len(entries)
    for order, item in enumerate(streams.visual.items_since(0)):
        text = str(getattr(item, "text", "") or "").strip()
        if not text:
            continue
        start = _item_start(item)
        end_value = getattr(item, "end", None)
        try:
            end = float(end_value) if end_value is not None else (start or 0.0)
        except (TypeError, ValueError):
            end = start or 0.0
        stamp = f"[{_fmt_clock(start or 0.0)}–{_fmt_clock(end)}]"
        entries.append((start or 0.0, offset + order, f"{stamp} 板书：\n{text}"))

    if not entries:
        return ""
    entries.sort(key=lambda entry: (entry[0], entry[1]))
    body = "\n\n".join(entry[2] for entry in entries)
    return f"# 课堂原始记录\n\n{body}\n"


def _item_start(item: Any) -> float | None:
    value = getattr(item, "start", None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _has_content(path: Path, header: str) -> bool:
    """True when the note file exists and holds more than its fixed header."""
    try:
        return path.stat().st_size > len(header.encode("utf-8"))
    except OSError:
        return False




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
        key="summarize.max_tool_output",
        label="工具输出上限",
        type="int",
        default=200000,
        group="Summarize",
        description="单次 read 的输出上限（字符）；足够大才能一次读整段材料",
    ),
    SettingsSpec(
        key="summarize.max_history_chars",
        label="上下文工具结果上限",
        type="int",
        default=0,
        group="Summarize",
        description="0 = 不裁剪（模型上下文足够大）；>0 时旧的 read 结果转存为隐藏文件",
    ),
    SettingsSpec(
        key="summarize.max_tokens",
        label="输出 Token 上限",
        type="int",
        default=32000,
        group="Summarize",
        description="单次调用允许的输出 token 数（含思考 token）",
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
        description="讲义生成提示词",
    ),
    SettingsSpec(
        key="summarize.retry_max",
        label="重试次数",
        type="int",
        default=1,
        group="Summarize",
        description="一次运行后三份笔记都没变化时的重试次数",
    ),
    SettingsSpec(
        key="summarize.auto_on_stop",
        label="课后自动生成",
        type="bool",
        default=True,
        group="Summarize",
        description="会话结束时自动生成完整讲义",
    ),
]

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

# ---------------------------------------------------------------------------
# Settings coercion helpers
# ---------------------------------------------------------------------------
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
