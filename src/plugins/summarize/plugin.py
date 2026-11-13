"""Summarize plugin: non-instant, whole-lecture structured notes.

Unlike the instant plugin, this one does not run during the lecture.  It runs
once over the entire session (manually via the UI action, or automatically
when the session stops).

The whole transcript + blackboard record is written to a single material file
and handed to a **writer** agent that maintains the three note files in place.
A second, independent **reviewer** agent (read-only) checks the notes against
the material and the mathematics; its findings are sent back to the writer,
which revises the existing document, and the reviewer re-checks.  Both agent
contexts persist for the whole job; the loop ends only when the reviewer says
``REVIEW: PASS`` (or one of the no-progress guards fires).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path
from typing import Any

from core.plugin import Plugin, PluginContext
from plugins.settings.spec import SettingsSpec

from .prompts import (
    DRAFT_SECTION_TEMPLATE,
    MERGE_PROMPT,
    NO_CHANGE_PROMPT,
    RETRY_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    REVIEW_FORMAT_NUDGE,
    REVIEW_PROMPT,
    REVISION_PROMPT_TEMPLATE,
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

# Agent tool declarations for the whole-lecture jobs (tool names + params).
WRITER_TOOLS = [("info", {}), ("read", {}), ("replace", {}), ("append", {})]
REVIEWER_TOOLS = [("info", {}), ("read", {})]

# The raw lecture record handed to the agent (kept next to ``notes/``).
SOURCE_FILENAME = "lecture_source.md"

# Draft assembled from the live "instant" notes, handed to the writer as a
# starting point.  It is explicitly *not* a source of truth: the prompt tells
# the writer to drop anything the lecture record cannot confirm.
INSTANT_DRAFT_FILENAME = "instant_draft.md"

_DOC_FILES = {
    "knowledge": "note_knowledge.md",
    "gossip": "note_gossip.md",
    "notice": "note_notice.md",
}

# Written by the stt / vlm plugins, one JSON object per line.  Each plugin owns
# its own file; this plugin only reads and merges them at handout time.
TRANSCRIPT_LOG = "transcript.jsonl"
VISUAL_LOG = "visual.jsonl"



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
        self._running_session = ""
        self._running_label = ""
        self._running_started = 0.0
        self._agent_session_id = ""
        self._running_phase = ""
        self._stop_event = threading.Event()
        self._status = "尚未生成讲义"
        self._notice = ""

        ctx.events.define("summarize.started", "A whole-lecture summary started")
        ctx.events.define("summarize.finished", "Whole-lecture summary finished")
        ctx.events.define("summarize.error", "Whole-lecture summary failed")

        ctx.ui.register_task_provider(self.running_tasks)
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
    def _writer_prompt(self) -> str:
        # ``summarize.system_prompt`` stays the editable writer prompt so
        # existing settings keep working.
        return self._system_prompt

    @property
    def _reviewer_prompt(self) -> str:
        return str(
            self._settings_service.get(
                "summarize.reviewer_prompt", REVIEWER_SYSTEM_PROMPT
            )
            or REVIEWER_SYSTEM_PROMPT
        )

    @property
    def _review_enabled(self) -> bool:
        value = self._settings_service.get("summarize.review_enabled", True)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}

    @property
    def _review_model(self) -> str:
        return str(self._settings_service.get("summarize.review_model", "") or "")

    @property
    def _review_max_steps(self) -> int:
        # 0 = unlimited: the reviewer reads until it has a verdict.
        value = int(self._settings_service.get("summarize.review_max_steps", 0))
        return max(0, value)

    @property
    def _review_max_tokens(self) -> int:
        # Reasoning tokens count against this cap; 8k was too small and made
        # the reviewer return empty content.  Match the writer.
        value = int(
            self._settings_service.get("summarize.review_max_tokens", 32000)
        )
        return max(2000, value)

    @property
    def _review_max_rounds(self) -> int:
        # 0 = keep revising until the reviewer passes.
        return max(0, int(self._settings_service.get("summarize.review_max_rounds", 0)))

    @property
    def _retry_max(self) -> int:
        return int(self._settings_service.get("summarize.retry_max", 1))

    @property
    def _auto_on_stop(self) -> bool:
        """Legacy hard switch; the dropdown below is the normal control."""

        value = self._settings_service.get("summarize.auto_on_stop", True)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}

    ON_STOP_MODES = ("direct", "fill", "skip")
    ON_STOP_LABELS = (
        ("direct", "直接总结（只用手上已有的内容）"),
        ("fill", "先补全缺课部分，再总结"),
        ("skip", "不生成讲义"),
    )

    @property
    def _on_stop_mode(self) -> str:
        """What to do when a class ends: direct / fill / skip."""

        value = str(self._settings_service.get("summarize.on_stop", "direct") or "")
        value = value.strip().lower()
        return value if value in self.ON_STOP_MODES else "direct"

    @property
    def busy(self) -> bool:
        return self._job_busy

    @property
    def _fill_wait_minutes(self) -> int:
        try:
            return max(1, int(self._settings_service.get("summarize.fill_wait_minutes", 20)))
        except (TypeError, ValueError):
            return 20

    @property
    def _fill_max_attempts(self) -> int:
        try:
            return max(1, int(self._settings_service.get("summarize.fill_max_attempts", 6)))
        except (TypeError, ValueError):
            return 6

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
        ui.register_action(
            "summarize.set_mode", self._on_set_mode, plugin_id=self.id
        )
        ui.register_action(
            "summarize.fill_again", self._on_fill_again, plugin_id=self.id
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

    def running_tasks(self) -> list[dict[str, Any]]:
        if not self._job_busy:
            return []
        step = 0
        getter = getattr(self._agent_service, "get_session", None)
        agent_session = (
            getter(self._agent_session_id)
            if (getter is not None and self._agent_session_id)
            else None
        )
        if agent_session is not None:
            try:
                step = int(agent_session.meta.get("step", 0) or 0)
            except (TypeError, ValueError):
                step = 0
        elapsed = max(0, int(time.time() - self._running_started))
        phase = {
            "writing": "撰写",
            "reviewing": "审阅",
            "revising": "修改",
        }.get(self._running_phase, "")
        detail = f"第 {step} 步 · {elapsed // 60}分{elapsed % 60}秒" if step else f"{elapsed // 60}分{elapsed % 60}秒"
        if phase:
            detail = f"{phase} · {detail}"
        return [
            {
                "id": "summarize",
                "label": f"生成讲义 · {self._running_label or '课程'}",
                "detail": detail,
                "started": self._running_started,
            }
        ]

    def _render_note_item(self) -> list[dict[str, Any]]:
        sid = self.ctx.streams.notes_session_id
        ready = sum(
            1
            for key in self._DOC_LABELS
            if _has_content(self._note_path(sid, _DOC_FILES[key]), NOTE_HEADERS[_DOC_FILES[key]])
        )
        running = (
            self.running_tasks()
            if self._job_busy and self._running_session == sid
            else []
        )
        status_text = self._status if ready == 0 else f"已生成 {ready}/{len(self._DOC_LABELS)} 篇"
        if running:
            status_text = f"正在生成讲义…（{running[0]['detail']}）"
        blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": status_text,
            },
        ]
        if self._notice:
            # Persistent: a toast would vanish before the user looks.
            blocks.append({"type": "markdown", "text": f"**⚠️ {self._notice}**"})
            if not running:
                blocks.append(
                    {
                        "type": "buttons",
                        "buttons": [
                            {
                                "label": "补全并重新生成",
                                "action": "summarize.fill_again",
                                "class": "btn btn-sm btn-primary",
                            }
                        ],
                    }
                )
        blocks.append(
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
            }
        )
        # The first handout is generated automatically when the session ends;
        # there is deliberately no "generate now" button.  A manual button only
        # appears once a handout exists (to rebuild it), and never while a job
        # is running — clicking it after the video download but before the
        # transcription finishes used to freeze the material at a partial
        # snapshot.
        if ready and not running:
            blocks.append(
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "重新生成讲义",
                            "action": "summarize.full",
                            "class": "btn btn-sm",
                        }
                    ],
                }
            )
        blocks.append(
            {
                "type": "form",
                "fields": [
                    {
                        "name": "mode",
                        "label": "课后处理",
                        "type": "select",
                        "options": [
                            {"value": value, "label": label}
                            for value, label in self.ON_STOP_LABELS
                        ],
                        "value": self._on_stop_mode,
                    }
                ],
                "submit": {"label": "保存", "action": "summarize.set_mode"},
            }
        )
        return blocks

    def _on_set_mode(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload.get("mode") or "").strip().lower()
        labels = dict(self.ON_STOP_LABELS)
        if mode not in labels:
            return {"ok": False, "message": "未知的课后处理方式", "refresh": True}
        self._settings_service.set("summarize.on_stop", mode)
        return {
            "ok": True,
            "message": f"课后处理方式：{labels[mode]}",
            "refresh": True,
        }

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
            owner=self.id,
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
        self._notice = ""
        self._create_note_files(sid)
        self._status = "课堂进行中，讲义将在课后生成"

    def _on_session_stopped(self, event: Any) -> None:
        sid = event.get("session_id", "")
        if not sid:
            return
        meta = self._session_meta(sid)
        fill_for = str(meta.get("fill_for") or "")
        if fill_for:
            # A gap-filling pass over the class recording just finished: build
            # the handout for the original lecture from both sets of logs.
            self.ctx.logger.info("gap fill for %s finished; summarizing", fill_for)
            self._spawn_worker(self._auto_start_when_ready, fill_for)
            return
        if not self._auto_on_stop:
            self.ctx.logger.info("auto summarize disabled; skipping %s", sid)
            return
        mode = self._on_stop_mode
        if mode == "skip":
            self.ctx.logger.info("summarize.on_stop=skip; not summarizing %s", sid)
            return
        if mode == "fill":
            self._spawn_worker(self._auto_fill_then_summarize, sid)
            return
        # The stop event races the STT/VLM tail flush: both workers append
        # their last lines asynchronously, so reading the logs immediately
        # silently truncates the lecture.  Prepare in a background thread
        # instead of blocking the event bus.
        self._spawn_worker(self._auto_start_when_ready, sid)

    def _spawn_worker(self, target: Any, session_id: str) -> None:
        threading.Thread(
            target=target,
            args=(session_id,),
            name=f"summarize-{session_id[:8]}",
            daemon=True,
        ).start()

    def _session_meta(self, session_id: str) -> dict[str, Any]:
        """Meta of a session that may already have ended (best effort)."""

        try:
            for info in self.ctx.streams.list_sessions():
                if info.session_id == session_id:
                    return dict(info.meta)
        except Exception:  # noqa: BLE001 - meta lookup is best effort
            pass
        return {}

    def _auto_fill_then_summarize(self, session_id: str) -> None:
        """Wait for the class recording, process its gaps, then summarize.

        The recording is usually published within minutes, so the first lookup
        is immediate; later attempts repeat every ``summarize.fill_wait_minutes``
        (20 by default).  When every attempt fails the plugin says so in the
        notes panel and falls back to a direct handout — the warning stays
        visible for the rest of the class, not just as a toast.
        """

        service = self.ctx.service("input")
        fill = getattr(service, "fill_gaps", None) if service is not None else None
        if not callable(fill):
            self._note_fallback(session_id, "输入服务不可用")
            self._auto_start_when_ready(session_id)
            return

        wait_seconds = max(1, self._fill_wait_minutes) * 60
        attempts = max(1, self._fill_max_attempts)
        for attempt in range(1, attempts + 1):
            self._wait_for_processors()
            self._status = f"正在查找本节课录播（第 {attempt}/{attempts} 次）…"
            self.ctx.ui.refresh("summarize.fill.lookup")
            started = False
            try:
                started = bool(fill(session_id))
            except Exception:  # noqa: BLE001 - fall back to a direct handout
                self.ctx.logger.exception("gap fill failed to start")
            if started:
                self._status = "正在补全课堂缺口…"
                self.ctx.ui.refresh("summarize.fill.started")
                self.ctx.logger.info("gap fill started for %s", session_id)
                return
            if attempt >= attempts:
                break
            self._status = (
                f"录播尚未发布，{wait_seconds // 60} 分钟后重试"
                f"（第 {attempt}/{attempts} 次）"
            )
            self.ctx.ui.refresh("summarize.fill.waiting")
            if self._stop_event.wait(wait_seconds):
                return

        self._note_fallback(session_id, f"已重试 {attempts} 次仍未取得录播")
        self._auto_start_when_ready(session_id)

    def _note_fallback(self, session_id: str, reason: str) -> None:
        """Persistent, visible warning that the handout skipped the recording."""

        self._notice = (
            f"未获取到本节课录播（{reason}），已直接使用课堂上已录制的内容生成讲义。"
            "稍后录播发布时，可点下面的「补全并重新生成」再补一次。"
        )
        self.ctx.logger.warning("summarize fallback for %s: %s", session_id, reason)
        self.ctx.ui.refresh("summarize.fallback")

    def _on_fill_again(self, _payload: dict[str, Any]) -> dict[str, Any]:
        """Manual retry after the automatic waits were exhausted."""

        sid = self.ctx.streams.notes_session_id
        if not sid:
            return {"ok": False, "message": "没有可补全的课堂", "refresh": False}
        self._notice = ""
        self._status = "正在查找本节课录播…"
        self._spawn_worker(self._auto_fill_then_summarize, sid)
        return {"ok": True, "message": "正在查找录播…", "refresh": True}

    def _auto_start_when_ready(self, session_id: str) -> None:
        """Wait for the transcribers to go idle, then start the handout."""

        self._status = "正在等待语音/板书处理完成…"
        self._wait_for_processors()
        started = self.start(session_id)
        self.ctx.logger.info(
            "auto summarize for %s: %s", session_id, "started" if started else "skipped"
        )

    def _wait_for_processors(self, *, timeout: float = 120.0) -> None:
        """Let STT/VLM flush their buffers before their logs are read."""

        probes = []
        for name in ("stt", "vlm"):
            service = self.ctx.service(name)
            for attr in ("busy", "flushing"):
                probe = getattr(service, attr, None)
                if callable(probe):
                    probes.append(probe)
        if not probes:
            return
        # Live VLM windows are flushed after the stop event and do not show up
        # as ``busy`` in live mode, so always allow a short grace period.
        if self._stop_event.wait(15.0):
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            busy = False
            for probe in probes:
                try:
                    if probe():
                        busy = True
                        break
                except Exception:  # noqa: BLE001 - a broken probe is not fatal
                    continue
            if not busy:
                return
            if self._stop_event.wait(0.5):
                return

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
        material = self._material_from_logs(session_id)
        if not material.strip() and streams.session_id == session_id:
            # Nothing on disk yet (manual streams / tests): use the live one.
            material = _collect_material(streams)
        if not material.strip():
            # Sessions that predate the per-plugin logs only have the snapshot.
            material = self._load_source(session_id)
        if not material.strip():
            self.ctx.logger.info("summarize skipped: no transcript/visual content")
            return False
        if self._stop_event.is_set():
            return False

        with self._job_lock:
            if self._job_busy:
                if session_id == self._running_session:
                    return True
                # One handout job at a time; run this session afterwards
                # (its lecture_source.md is already on disk).
                self._pending_session = session_id
                return True
            self._job_busy = True
            self._running_session = session_id
            self._running_started = time.time()

        # Snapshot everything the job needs now: the current stream may be
        # replaced by a new session as soon as the old one stops.
        course_name = (
            _course_name(streams) if streams.session_id == session_id else "未知课程"
        )
        duration = streams.audio.duration if streams.session_id == session_id else 0.0

        self._running_label = course_name
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

    def _material_from_logs(self, session_id: str) -> str:
        """Merge this session's logs with any gap-filling passes.

        A gap-fill runs as its own session (the recording is a separate media
        timeline); its records are shifted into the lecture session's timeline
        by ``media_begin - started_at`` so the merged material stays in order.
        """

        session_dir = self.ctx.data_dir / "sessions" / session_id
        transcript = _read_log(session_dir / TRANSCRIPT_LOG)
        visual = _read_log(session_dir / VISUAL_LOG)

        primary_started = _session_started_at(self.ctx.data_dir, session_id)
        for fill_id, media_begin in _fill_sessions(self.ctx.data_dir, session_id):
            shift = (
                float(media_begin) - float(primary_started)
                if media_begin and primary_started
                else 0.0
            )
            fill_dir = self.ctx.data_dir / "sessions" / fill_id
            transcript.extend(
                _shift_records(_read_log(fill_dir / TRANSCRIPT_LOG), shift)
            )
            visual.extend(_shift_records(_read_log(fill_dir / VISUAL_LOG), shift))

        return _merge_logs(transcript, visual)

    def _instant_draft_text(self, session_id: str) -> str:
        """Merge the live instant notes into one draft document.

        The instant plugin writes ``instant_note_*.md`` next to the handout
        files while the lecture is running.  Handing them to the writer as a
        draft avoids starting from a blank page; the prompt makes it verify
        every claim against the lecture record and delete the rest.
        """

        notes_dir = (
            self.ctx.data_dir / "sessions" / session_id / self._notes_dir_name
        )
        parts: list[str] = []
        for key, filename in _DOC_FILES.items():
            path = notes_dir / f"instant_{filename}"
            try:
                text = path.read_text("utf-8")
            except OSError:
                continue
            if text.strip():
                label = self._DOC_LABELS.get(key, key)
                parts.append(f"## {label}\n\n{text.strip()}\n")
        if not parts:
            return ""
        return (
            "# 即时笔记底稿\n\n"
            "> 课堂上实时生成，含识别错误与未经核实的推断。\n"
            "> 只作为组织线索；事实一律以 lecture_source.md 为准。\n\n"
            + "\n".join(parts)
        )

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
                self._running_session = ""
                self._running_label = ""
                self._running_started = 0.0
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
        """Write the handout, then let a reviewer drive revisions."""

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

        draft_text = self._instant_draft_text(session_id)
        draft_section = ""
        if draft_text:
            draft_path = notes_dir / INSTANT_DRAFT_FILENAME
            draft_path.write_text(draft_text, "utf-8")
            file_paths[INSTANT_DRAFT_FILENAME] = str(draft_path)
            draft_section = DRAFT_SECTION_TEMPLATE.format(
                draft=INSTANT_DRAFT_FILENAME,
                draft_chars=len(draft_text),
                source=SOURCE_FILENAME,
            )

        prompt = USER_PROMPT_TEMPLATE.format(
            course_name=course_name,
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            session_id=session_id,
            time_start=_format_seconds(0.0),
            time_end=_format_seconds(duration),
            source=SOURCE_FILENAME,
            chars=len(material),
            lines=material.count("\n") + 1,
            draft_section=draft_section,
        )
        existing = any(
            _has_content(Path(path), NOTE_HEADERS[filename])
            for filename, path in file_paths.items()
            if filename in NOTE_HEADERS
        )
        if existing:
            prompt += MERGE_PROMPT

        agent = self._agent_service
        model = self._model or None
        writer = agent.create_session(
            name=f"summarize-{session_id[:12]}",
            system_prompt=self._writer_prompt,
            files=file_paths,
            tools=list(WRITER_TOOLS),
            model=model,
            max_steps=0,  # unlimited: the model runs until the handout is done
            max_tool_output=self._max_tool_output,
            max_history_chars=self._max_history_chars,
            max_tokens=self._max_tokens,
            llm_extra={"reasoning_effort": "max"},
            meta={"lecture_session": session_id, "role": "writer"},
        )

        reviewer = None
        if self._review_enabled:
            reviewer = agent.create_session(
                name=f"review-{session_id[:12]}",
                system_prompt=self._reviewer_prompt,
                files=file_paths,
                tools=list(REVIEWER_TOOLS),
                model=self._review_model or model,
                max_steps=self._review_max_steps,
                max_tool_output=self._max_tool_output,
                max_history_chars=self._max_history_chars,
                max_tokens=self._review_max_tokens,
                llm_extra={"reasoning_effort": "max"},
                meta={"lecture_session": session_id, "role": "reviewer"},
            )

        self._running_phase = "writing"
        self._agent_session_id = writer.id
        hashes_before = {
            filename: _file_md5(path)
            for filename, path in file_paths.items()
            if filename not in (SOURCE_FILENAME, INSTANT_DRAFT_FILENAME)
        }
        try:
            agent.run(writer.id, prompt)
            if not _notes_changed(file_paths, hashes_before):
                self.ctx.logger.warning(
                    "writer made no changes for %s; nudging it to merge the material",
                    session_id,
                )
                agent.run(writer.id, NO_CHANGE_PROMPT)
            if reviewer is not None:
                try:
                    self._review_loop(
                        agent, writer.id, reviewer.id, file_paths, session_id
                    )
                except Exception:
                    # The handout exists; a broken reviewer must not turn the
                    # whole job into a failure.
                    self.ctx.logger.exception(
                        "review loop failed for %s; keeping the writer's notes",
                        session_id,
                    )
            else:
                self._retry_if_unchanged(
                    agent, file_paths, session_id, hashes_before
                )
        finally:
            self._agent_session_id = ""
            self._running_phase = ""

    def _review_loop(
        self,
        agent: Any,
        writer_id: str,
        reviewer_id: str,
        file_paths: dict[str, str],
        session_id: str,
    ) -> None:
        """Send the reviewer's findings back to the writer until it passes.

        Both contexts are persistent: the reviewer remembers its previous
        findings (so it can check whether they were fixed), and the writer
        remembers every revision it already made.  Nothing is reset.
        """

        previous_feedback = ""
        round_no = 0
        format_nudges = 0
        max_rounds = self._review_max_rounds  # 0 = no fixed limit
        next_prompt = REVIEW_PROMPT
        while True:
            round_no += 1
            if max_rounds and round_no > max_rounds:
                self.ctx.logger.warning(
                    "review for %s stopped at the configured limit (%d rounds)",
                    session_id,
                    max_rounds,
                )
                return

            self._running_phase = "reviewing"
            self._agent_session_id = reviewer_id
            result = agent.run(reviewer_id, next_prompt)
            if getattr(result, "error", None):
                self.ctx.logger.warning(
                    "reviewer failed for %s: %s", session_id, result.error
                )
                return
            if (
                self._review_max_steps
                and getattr(result, "steps", 0) >= self._review_max_steps
            ):
                self.ctx.logger.warning(
                    "reviewer hit its step limit (%d) for %s",
                    self._review_max_steps,
                    session_id,
                )
            message = _result_text(result)
            verdict, body = _parse_review(message)
            if verdict == "PASS":
                self.ctx.logger.info(
                    "review passed for %s after %d round(s)", session_id, round_no
                )
                return
            feedback = body.strip()
            if not feedback:
                if format_nudges < 2:
                    format_nudges += 1
                    self.ctx.logger.warning(
                        "reviewer returned no usable findings for %s; asking again",
                        session_id,
                    )
                    next_prompt = REVIEW_FORMAT_NUDGE
                    continue
                self.ctx.logger.warning(
                    "reviewer returned FAIL without findings for %s", session_id
                )
                return
            next_prompt = REVIEW_PROMPT
            if feedback == previous_feedback:
                self.ctx.logger.warning(
                    "reviewer repeated the same findings for %s; stopping", session_id
                )
                return
            previous_feedback = feedback

            before = {
                filename: _file_md5(path)
                for filename, path in file_paths.items()
                if filename not in (SOURCE_FILENAME, INSTANT_DRAFT_FILENAME)
            }
            self._running_phase = "revising"
            self._agent_session_id = writer_id
            agent.run(
                writer_id,
                REVISION_PROMPT_TEMPLATE.format(review=feedback),
            )
            after = {
                filename: _file_md5(str(file_paths[filename]))
                for filename in before
            }
            if after == before:
                self.ctx.logger.warning(
                    "writer produced no change after review for %s; stopping",
                    session_id,
                )
                return

    def _retry_if_unchanged(
        self,
        agent: Any,
        file_paths: dict[str, str],
        session_id: str,
        hashes_before: dict[str, str],
    ) -> None:
        """Legacy fallback when review is disabled."""

        hashes_after = {
            filename: _file_md5(str(file_paths[filename]))
            for filename in hashes_before
        }
        if hashes_after != hashes_before:
            return
        retry = agent.create_session(
            name=f"summarize-{session_id[:12]}-retry",
            system_prompt=self._writer_prompt,
            files=file_paths,
            tools=list(WRITER_TOOLS),
            model=self._model or None,
            max_steps=0,
            max_tool_output=self._max_tool_output,
            max_history_chars=self._max_history_chars,
            max_tokens=self._max_tokens,
            llm_extra={"reasoning_effort": "max"},
        )
        self._agent_session_id = retry.id
        agent.run(retry.id, RETRY_PROMPT)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_REVIEW_MARKER_RE = re.compile(
    r"(?im)^[ \t>*`]*(?:REVIEW|审阅结论)[ \t]*[:：][ \t*`]*"
    r"(PASS|FAIL|通过|不通过)[ \t*`]*[.。]?[ \t]*$"
)


def _result_text(result: Any) -> str:
    """Return the final message of an ``AgentRunResult`` (or a test double)."""

    for attr in ("final_message", "content"):
        value = getattr(result, attr, None)
        if value:
            return str(value)
    return ""


def _notes_changed(
    file_paths: dict[str, str], hashes_before: dict[str, str]
) -> bool:
    """True when at least one note file changed since *hashes_before*."""

    return any(
        _file_md5(str(file_paths[name])) != digest
        for name, digest in hashes_before.items()
    )


def _parse_review(text: str) -> tuple[str, str]:
    """Split a reviewer reply into ``("PASS"|"FAIL", findings)``.

    A missing marker is treated as FAIL so the writer still receives the
    text as feedback; the loop's repetition guard stops a model that never
    emits the marker.
    """

    text = (text or "").strip()
    matches = _REVIEW_MARKER_RE.findall(text)
    if not matches:
        return "FAIL", text
    verdict = matches[-1].strip()
    passed = verdict.upper() == "PASS" or verdict == "通过"
    body = _REVIEW_MARKER_RE.sub("", text).strip()
    return ("PASS" if passed else "FAIL"), body


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


def _read_log(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL log written by the stt / vlm plugin."""

    try:
        raw = path.read_text("utf-8")
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and str(record.get("text") or "").strip():
            records.append(record)
    return records


def _log_start(record: dict[str, Any]) -> float:
    try:
        value = record.get("start")
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _merge_logs(
    transcript: list[dict[str, Any]], visual: list[dict[str, Any]]
) -> str:
    """Merge the two per-plugin logs exactly like ``_collect_material`` does."""

    entries: list[tuple[float, int, int, str]] = []
    for index, record in enumerate(transcript):
        start = _log_start(record)
        text = str(record.get("text") or "").strip()
        entries.append((start, 0, index, f"[{_fmt_clock(start)}] 讲授：{text}"))
    for index, record in enumerate(visual):
        start = _log_start(record)
        try:
            end_value = record.get("end")
            end = float(end_value) if end_value is not None else start
        except (TypeError, ValueError):
            end = start
        text = str(record.get("text") or "").strip()
        entries.append(
            (
                start,
                1,
                index,
                f"[{_fmt_clock(start)}–{_fmt_clock(end)}] 板书：\n{text}",
            )
        )
    if not entries:
        return ""
    entries.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    body = "\n\n".join(entry[3] for entry in entries)
    return f"# 课堂原始记录\n\n{body}\n"


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
        label="写作提示词",
        type="text",
        default=SYSTEM_PROMPT,
        group="Summarize",
        description="讲义写作 agent 的系统提示词",
    ),
    SettingsSpec(
        key="summarize.review_enabled",
        label="启用审阅",
        type="bool",
        default=True,
        group="Summarize",
        description="写完后由独立审阅 agent 检查，并把意见发回写作 agent 修改",
    ),
    SettingsSpec(
        key="summarize.reviewer_prompt",
        label="审阅提示词",
        type="text",
        default=REVIEWER_SYSTEM_PROMPT,
        group="Summarize",
        description="审阅 agent 的系统提示词（只读，不写文件）",
    ),
    SettingsSpec(
        key="summarize.review_model",
        label="审阅模型",
        type="string",
        default="",
        group="Summarize",
        description="审阅 agent 使用的模型（空=与写作相同）",
    ),
    SettingsSpec(
        key="summarize.review_max_steps",
        label="审阅步数上限",
        type="int",
        default=0,
        group="Summarize",
        description="0 = 不设上限（审阅 agent 自己读到自己满意）；>0 时达到步数后停止",
    ),
    SettingsSpec(
        key="summarize.review_max_tokens",
        label="审阅输出上限",
        type="int",
        default=32000,
        group="Summarize",
        description="审阅 agent 单次调用的输出 token 上限（含思考 token）",
    ),
    SettingsSpec(
        key="summarize.review_max_rounds",
        label="审阅轮数上限",
        type="int",
        default=0,
        group="Summarize",
        description="0 = 不设上限（有问题就一直改）；>0 时达到轮数后停止",
    ),
    SettingsSpec(
        key="summarize.retry_max",
        label="重试次数",
        type="int",
        default=1,
        group="Summarize",
        description="关闭审阅时，一次运行后三份笔记都没变化的重试次数",
    ),
    SettingsSpec(
        key="summarize.on_stop",
        label="课后处理方式",
        type="select",
        default="direct",
        options=[
            ("direct", "直接总结（只用手上已有的内容）"),
            ("fill", "先补全缺课部分，再总结"),
            ("skip", "不生成讲义"),
        ],
        group="Summarize",
        description=(
            "课堂直播结束后的动作。选「先补全」时，会查找这门课的录播，"
            "只处理直播期间没有覆盖到的时间段，再生成讲义。"
        ),
    ),
    SettingsSpec(
        key="summarize.fill_wait_minutes",
        label="等待录播（分钟）",
        type="int",
        default=20,
        group="Summarize",
        description=(
            "选「先补全」时，第一次查找失败后每隔多少分钟重试一次。"
            "录播通常在几分钟内发布，默认 20 分钟足够。"
        ),
    ),
    SettingsSpec(
        key="summarize.fill_max_attempts",
        label="录播最多重试次数",
        type="int",
        default=6,
        group="Summarize",
        description=(
            "重试多少次仍拿不到录播就回退为「直接总结」；回退会在讲义面板里常驻提示，"
            "之后也可以点「补全并重新生成」手动再试。"
        ),
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
# Session meta / gap-fill helpers
# ---------------------------------------------------------------------------


def _read_session_meta(data_dir: Path, session_id: str) -> dict[str, Any]:
    try:
        raw = json.loads(
            (Path(data_dir) / "sessions" / session_id / "meta.json").read_text("utf-8")
        )
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _session_started_at(data_dir: Path, session_id: str) -> float:
    raw = _read_session_meta(data_dir, session_id)
    try:
        return float(raw.get("started_at") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _fill_sessions(data_dir: Path, target_id: str) -> list[tuple[str, float]]:
    """Sessions whose ``fill_for`` points at *target_id*."""

    out: list[tuple[str, float]] = []
    root = Path(data_dir) / "sessions"
    try:
        dirs = list(root.iterdir())
    except OSError:
        return out
    for path in dirs:
        if not path.is_dir():
            continue
        raw = _read_session_meta(data_dir, path.name)
        meta = raw.get("meta")
        if not isinstance(meta, dict):
            continue
        if str(meta.get("fill_for") or "") != target_id:
            continue
        try:
            begin = float(meta.get("media_begin") or 0.0)
        except (TypeError, ValueError):
            begin = 0.0
        out.append((path.name, begin))
    return out


def _shift_records(
    records: list[dict[str, Any]], shift: float
) -> list[dict[str, Any]]:
    """Move records onto the lecture timeline (no-op when *shift* is 0)."""

    if not shift:
        return records
    shifted: list[dict[str, Any]] = []
    for record in records:
        moved = dict(record)
        for key in ("start", "end"):
            if moved.get(key) is not None:
                try:
                    moved[key] = float(moved[key]) + shift
                except (TypeError, ValueError):
                    pass
        shifted.append(moved)
    return shifted
