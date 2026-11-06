"""Summarize plugin: non-instant, whole-lecture structured notes.

Unlike the instant plugin, this one does not run during the lecture.  It runs
once over the entire session (manually via the UI action, or automatically
when the session stops) and rewrites the three note files into a complete,
outline-structured handout.

Long lectures are processed in sequential batches: every batch is merged into
the same outline, so the notes stay coherent from the first batch to the last.
"""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from core.plugin import Plugin, PluginContext
from plugins.settings.spec import SettingsSpec

from .prompts import (
    FINAL_INSTRUCTION,
    RETRY_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

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

_DOC_FILES = {
    "knowledge": "note_knowledge.md",
    "gossip": "note_gossip.md",
    "notice": "note_notice.md",
}

_TOOLS = ["info", "read", "replace", "append", "write"]


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
    def _max_steps(self) -> int:
        return int(self._settings_service.get("summarize.max_steps", 16))

    @property
    def _max_tool_output(self) -> int:
        return int(self._settings_service.get("summarize.max_tool_output", 24000))

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
    def _chunk_chars(self) -> int:
        return max(200, int(self._settings_service.get("summarize.chunk_chars", 6000)))

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
        sid = self.ctx.streams.current.session_id
        return [
            {
                "key": key,
                "label": label,
                "path": str(self._note_path(sid, _DOC_FILES[key])),
            }
            for key, label in self._DOC_LABELS.items()
        ]

    def _render_note_item(self) -> list[dict[str, Any]]:
        sid = self.ctx.streams.current.session_id
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
        reader.show(
            title=f"课堂讲义 · {self._course_name()}",
            docs=self._docs(),
            active=doc,
        )
        return {"ok": True, "open_page": "reader:main"}

    def _course_name(self) -> str:
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
    def _on_summarize_full(self, _payload: dict[str, Any]) -> dict[str, Any]:
        sid = self.ctx.streams.current.session_id
        started = self.start(sid)
        if started:
            return {"ok": True, "message": "完整讲义生成已开始", "refresh": True}
        if self.busy:
            return {"ok": True, "message": "讲义生成已在运行中", "refresh": False}
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
        if self._auto_on_stop:
            self.start(sid)

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
        if streams.session_id != session_id:
            return False
        blocks = _collect_blocks(streams)
        if not blocks:
            return False
        if self._stop_event.is_set():
            return False

        with self._job_lock:
            if self._job_busy:
                return False
            self._job_busy = True

        # Snapshot everything the job needs now: the current stream may be
        # replaced by a new session as soon as the old one stops.
        course_name = _course_name(streams)
        duration = streams.audio.duration

        self._status = "正在生成完整讲义…"
        thread = threading.Thread(
            target=self._run_full,
            args=(session_id, blocks, course_name, duration),
            name=f"summarize-{session_id[:8]}",
            daemon=True,
        )
        thread.start()
        return True

    def _run_full(
        self,
        session_id: str,
        blocks: list[str],
        course_name: str,
        duration: float,
    ) -> None:
        self.ctx.events.emit("summarize.started", session_id=session_id)
        self.ctx.ui.refresh("summarize.started")
        success = False
        try:
            passes = self._run_passes(session_id, blocks, course_name, duration)
            success = True
            self._status = f"讲义已生成（{passes} 批材料）"
        except Exception:
            self.ctx.logger.exception(
                "whole-lecture summarize failed for session %s", session_id
            )
            self._status = "讲义生成失败"
        finally:
            with self._job_lock:
                self._job_busy = False

        if success:
            self.ctx.events.emit("summarize.finished", session_id=session_id)
            self.ctx.ui.refresh("summarize.finished")
        else:
            self.ctx.events.emit("summarize.error", session_id=session_id)
            self.ctx.ui.refresh("summarize.error")

    def _run_passes(
        self,
        session_id: str,
        blocks: list[str],
        course_name: str,
        duration: float,
    ) -> int:
        notes_dir = (
            self.ctx.data_dir / "sessions" / session_id / self._notes_dir_name
        )
        notes_dir.mkdir(parents=True, exist_ok=True)

        file_paths: dict[str, str] = {}
        for filename in NOTE_FILES:
            path = notes_dir / filename
            if not path.exists():
                path.write_text(NOTE_HEADERS[filename], "utf-8")
            file_paths[filename] = str(path)

        chunks = _chunk_blocks(blocks, self._chunk_chars)
        total = len(chunks)

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        time_start = _format_seconds(0.0)
        time_end = _format_seconds(duration)

        for index, (chunk, chunk_start, chunk_end) in enumerate(chunks, start=1):
            prompt = USER_PROMPT_TEMPLATE.format(
                course_name=course_name,
                date=now,
                session_id=session_id,
                time_start=time_start,
                time_end=time_end,
                part=index,
                parts=total,
                part_range=f"第 {chunk_start + 1}–{chunk_end} 字",
                content="\n\n".join(chunk),
                final_instruction=FINAL_INSTRUCTION if index == total else "",
            )
            self._run_pass(session_id, index, file_paths, prompt)

        return total

    def _run_pass(
        self,
        session_id: str,
        index: int,
        file_paths: dict[str, str],
        prompt: str,
    ) -> None:
        """Run one batch (with one retry when nothing was written)."""
        agent = self._agent_service
        model = self._model or None
        hashes_before = {
            filename: _file_md5(path) for filename, path in file_paths.items()
        }

        session = agent.create_session(
            name=f"summarize-{session_id[:12]}-{index}",
            system_prompt=self._system_prompt,
            files=file_paths,
            tools=list(_TOOLS),
            model=model,
            max_steps=self._max_steps,
            max_tool_output=self._max_tool_output,
        )
        agent.run(session.id, prompt)

        hashes_after = {
            filename: _file_md5(path) for filename, path in file_paths.items()
        }
        if hashes_after == hashes_before:
            retry_session = agent.create_session(
                name=f"summarize-retry-{session_id[:12]}-{index}",
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


def _item_start(item: Any) -> float | None:
    value = getattr(item, "start", None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _collect_blocks(streams: Any) -> list[str]:
    """Flatten transcript + visual streams into tagged text blocks in order.

    Both streams are merged chronologically only when every item carries a
    usable start time; otherwise the transcript is kept first and the visual
    stream follows, each in arrival order.
    """
    transcript = [
        item
        for item in streams.transcript.items_since(0)
        if str(getattr(item, "text", "") or "").strip()
    ]
    visual = [
        item
        for item in streams.visual.items_since(0)
        if str(getattr(item, "text", "") or "").strip()
    ]

    tagged: list[tuple[float | None, int, str]] = []
    for order, item in enumerate(transcript):
        tagged.append(
            (_item_start(item), order, f"【讲授】{str(item.text).strip()}")
        )
    offset = len(transcript)
    for order, item in enumerate(visual):
        tagged.append(
            (_item_start(item), offset + order, f"【板书】{str(item.text).strip()}")
        )

    if tagged and all(entry[0] is not None for entry in tagged):
        tagged.sort(key=lambda entry: (entry[0], entry[1]))
    return [entry[2] for entry in tagged]


def _chunk_blocks(blocks: list[str], limit: int) -> list[tuple[list[str], int, int]]:
    """Pack blocks into batches of at most *limit* characters.

    Returns ``(blocks, start_char, end_char)`` triples, where the offsets are
    cumulative character positions inside the full lecture record.
    """
    chunks: list[tuple[list[str], int, int]] = []
    current: list[str] = []
    size = 0
    cursor = 0
    chunk_start = 0
    for block in blocks:
        if current and size + len(block) > limit:
            chunks.append((current, chunk_start, cursor))
            current = []
            size = 0
            chunk_start = cursor
        current.append(block)
        size += len(block) + 2
        cursor += len(block) + 2
    if current:
        chunks.append((current, chunk_start, cursor))
    return chunks or [([], 0, 0)]


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
        key="summarize.max_steps",
        label="最大步数",
        type="int",
        default=16,
        group="Summarize",
        description="每批材料的 Agent 最大工具调用步数",
    ),
    SettingsSpec(
        key="summarize.max_tool_output",
        label="工具输出上限",
        type="int",
        default=24000,
        group="Summarize",
        description="单次 read 输出上限（字符），需要足够大才能整篇重写",
    ),
    SettingsSpec(
        key="summarize.chunk_chars",
        label="分批字符数",
        type="int",
        default=6000,
        group="Summarize",
        description="每批材料的目标字符数，过大可能超出上下文",
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
        default=1,
        group="Summarize",
        description="某批材料未写入任何内容时的重试次数",
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
