"""Ask plugin: in-note question answering with streaming and two intensities.

The plugin owns *questions*, not the agent loop: the ``agent`` plugin runs
the model, builds the tools from the declarations below and streams deltas
back.  Answers live in the note as ``qa`` markdown blocks (owned by ``reader``).
"""

from __future__ import annotations

import json
import threading
import time
import tempfile
from pathlib import Path
from typing import Any, Mapping

from core.agent_types import (
    AgentEventType,
    AgentProfile,
    AgentRole,
    AgentTool,
    AgentToolName,
    ExecutionRole,
    ExecutionBlockKind,
    ExecutionBlockStatus,
    ExecutionStatus,
)
from core.note_document import AskIntensity, QaStatus
from core.navigation import current_navigation
from core.plugin import Plugin, PluginContext
from core.settings_spec import SettingsSpec, SettingType
from core.ui import render_markdown
from core.file_access import publication_lock
from core.workspace import FileWorkspace
from plugins.notes.types import NotePublication
from plugins.reader.types import SelectionKind

from .prompts import ASK_PROMPT_TEMPLATE, INSTANT_SYSTEM_PROMPT, RESEARCH_SYSTEM_PROMPT

ACTION = "ask.ask"
SELECTION_ID = "ask.ask"
# The agent plugin owns this channel. Show it only when the action exists.
MESSAGE_ACTION = "agent.message"


class AskService:
    __slots__ = ("_plugin",)

    def __init__(self, plugin: "AskPlugin") -> None:
        self._plugin = plugin

    def ask(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ask one question (same payload as the ``ask.ask`` UI action)."""

        return self._plugin._on_ask(payload)  # noqa: SLF001 - the plugin owns it

    def activity(self, path: str, qa_id: str) -> dict[str, Any] | None:
        return self._plugin._activity(path, qa_id)


class AskPlugin(Plugin):
    id = "ask"
    requires = ("settings", "agent", "reader")
    optional_requires = ("notes",)
    provides = ("ask",)
    uses_actions = (MESSAGE_ACTION,)

    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self._settings = ctx.require("settings")
        self._agent = ctx.require("agent")
        self._reader = ctx.require("reader")
        self._active: set[str] = set()
        self._live: dict[tuple[str, str], _Live] = {}
        self._active_lock = threading.Lock()
        self._closed = False

        for spec in _SETTING_SPECS:
            self._settings.register(spec)

        ctx.register_service("ask", AskService(self))
        ctx.ui.register_action(ACTION, self._on_ask, plugin_id=self.id)
        self._reader.register_activity_provider(self.id, self._activity)

        self._reader.register_selection_action(
            "",
            {
                "id": SELECTION_ID,
                "label": "提问",
                "kind": SelectionKind.PROMPT,
                "action": ACTION,
                "placeholder": "针对选中的内容提问，Enter 发送",
                "intensities": [
                    {"value": AskIntensity.INSTANT, "label": "快速"},
                    {"value": AskIntensity.RESEARCH, "label": "研究"},
                ],
                "order": 10,
            },
            owner=self.id,
        )

    def teardown(self) -> None:

        with self._active_lock:
            self._closed = True
            for session_id in tuple(self._active):
                self._agent.cancel(session_id)
        ctx = getattr(self, "ctx", None)
        if ctx is not None:
            try:
                ctx.ui.unregister_action(ACTION)
            except Exception:  # noqa: BLE001 - shutdown is best effort
                pass
        reader = getattr(self, "_reader", None)
        if reader is not None:
            try:
                reader.unregister_selection_action("", SELECTION_ID, owner=self.id)
                reader.unregister_activity_provider(self.id)
            except Exception:  # noqa: BLE001 - shutdown is best effort
                pass

    def _activity(self, path, qa_id):
        with self._active_lock:
            live = self._live.get((str(Path(path).resolve()), qa_id))
        return live.activity() if live is not None else None

    def _setting(self, key: str, default: Any) -> Any:
        return self._settings.get(f"ask.{key}", default)

    @property
    def _before_chars(self) -> int:
        return max(0, int(self._setting("before_chars", 2000)))

    @property
    def _after_chars(self) -> int:
        return max(0, int(self._setting("after_chars", 2000)))

    @property
    def _history_chars(self) -> int:
        return max(0, int(self._setting("history_chars", 1000)))

    @property
    def _selection_chars(self) -> int:
        return max(0, int(self._setting("selection_chars", 800)))

    @property
    def _instant_prompt(self) -> str:
        return str(
            self._setting("instant_prompt", INSTANT_SYSTEM_PROMPT)
            or INSTANT_SYSTEM_PROMPT
        )

    @property
    def _research_prompt(self) -> str:
        return str(
            self._setting("research_prompt", RESEARCH_SYSTEM_PROMPT)
            or RESEARCH_SYSTEM_PROMPT
        )

    def _on_ask(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question", "") or "").strip()
        selection = str(payload.get("selection", "") or "").strip()
        try:
            intensity = AskIntensity(payload.get("intensity") or AskIntensity.INSTANT)
        except ValueError:
            intensity = AskIntensity.INSTANT
        path = str(payload.get("path", "") or "")
        if not question:
            return {"ok": False, "message": "问题不能为空"}

        created = self._reader.qa_begin(
            path,
            question=question,
            selection=selection,
            intensity=intensity,
        )
        qa_id = str(created.get("id", "") or "")
        course_name = self._course_name(path)
        threading.Thread(
            target=self._run,
            args=(path, qa_id, selection, intensity, course_name),
            name=f"ask-{qa_id}",
            daemon=True,
        ).start()

        message = "已提交，Agent 正在回答…"
        if not created.get("anchored", True):
            message += "（未能定位选中片段，已附在笔记末尾）"
        return {"ok": True, "message": message, "refresh": True}

    def _run(
        self,
        path: str,
        qa_id: str,
        selection: str,
        intensity: AskIntensity,
        course_name: str,
    ) -> None:
        live = _Live(self, path, qa_id)
        key = (str(Path(path).resolve()), qa_id)
        with self._active_lock:
            self._live[key] = live
        status = QaStatus.DONE
        answer = ""
        started = time.time()
        try:
            answer = self._answer(path, qa_id, selection, intensity, live, course_name)
        except Exception as exc:  # noqa: BLE001 - the failure belongs in the note
            self.ctx.logger.exception("ask %s failed", qa_id)
            status = QaStatus.ERROR
            answer = f"（回答失败：{exc}）"

        try:
            self._reader.qa_finish(
                path,
                qa_id,
                answer=answer,
                status=status,
                collapsed=(status == QaStatus.DONE),
            )
            live.finish(status=status, answer=answer, elapsed=time.time() - started)
        finally:
            with self._active_lock:
                self._live.pop(key, None)

    def _answer(
        self,
        path: str,
        qa_id: str,
        selection: str,
        intensity: AskIntensity,
        live: "_Live",
        course_name: str,
    ) -> str:
        context = self._reader.qa_context(
            path,
            qa_id,
            before_chars=self._before_chars,
            after_chars=self._after_chars,
            history_chars=self._history_chars,
        )
        prompt = self._build_prompt(path, context, selection, intensity, course_name)
        files = (
            self._workspace_files(path) if intensity == AskIntensity.RESEARCH else {}
        )
        started = time.time()
        with tempfile.TemporaryDirectory(prefix="notes-ask-") as snapshot:
            frozen = {}
            with publication_lock(Path(path).parent):
                for name, original in files.items():
                    destination = Path(snapshot) / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(Path(original).read_bytes())
                    frozen[name] = destination
            profile = self._profile(intensity)
            with self._active_lock:
                if self._closed:
                    raise RuntimeError("Ask service is closed")
                session = self._agent.create_session(
                    name=f"ask-{qa_id}",
                    agent=profile,
                    workspace=FileWorkspace(frozen, readonly=frozen),
                    meta={
                        "role": ExecutionRole.QUESTION_ANSWER,
                        "qa_id": qa_id,
                        "intensity": intensity,
                    },
                )
                self._active.add(session.id)
            try:
                live.bind_session(session.id)
                result = self._agent.run(session.id, prompt, on_event=live.on_event)
            finally:
                try:
                    self._agent.close(session.id)
                finally:
                    with self._active_lock:
                        self._active.discard(session.id)

        self.ctx.logger.info(
            "ask %s finished in %.1fs: intensity=%s steps=%s tokens=%s error=%s",
            qa_id,
            time.time() - started,
            intensity,
            result.steps,
            result.total_tokens,
            result.error,
        )
        if result.error:
            raise RuntimeError(result.error)
        return result.final_message.strip() or "（模型没有返回内容）"

    def _profile(self, intensity: AskIntensity) -> AgentProfile:
        return AgentProfile(
            name=f"ask.{intensity}",
            role=AgentRole.ASK_INSTANT
            if intensity == AskIntensity.INSTANT
            else AgentRole.ASK_RESEARCH,
            system_prompt=self._instant_prompt
            if intensity == AskIntensity.INSTANT
            else self._research_prompt,
            tools=[]
            if intensity == AskIntensity.INSTANT
            else [
                AgentTool(name)
                for name in (AgentToolName.READ, AgentToolName.GREP, AgentToolName.GLOB)
            ],
        )

    def _build_prompt(
        self,
        path: str,
        context: Mapping[str, Any],
        selection: str,
        intensity: AskIntensity,
        course_name: str,
    ) -> str:
        name = Path(path).name
        if intensity == AskIntensity.RESEARCH:
            readable = "、".join(f"`{item}`" for item in self._workspace_files(path))
            instructions = (
                f"可用 read、grep、glob 核对完整授权材料．可读文件：{readable}．"
            )
        else:
            instructions = "材料已经足够，直接回答；本题没有工具可用。"
        return ASK_PROMPT_TEMPLATE.format(
            course_name=course_name,
            file_label=self._doc_label(name),
            file_name=name,
            selection=_truncate_chars(selection, self._selection_chars)
            or "（未提供选中内容）",
            before_chars=self._before_chars,
            before=context.get("before") or "（无）",
            after_chars=self._after_chars,
            after=context.get("after") or "（无）",
            history_chars=self._history_chars,
            history=_format_history(context.get("history") or []),
            question=context.get("question") or "",
            instructions=instructions,
        )

    def _workspace_files(self, path: str) -> dict[str, str]:
        note = Path(path)
        names = set()
        names.add(note.name)
        pipeline = self.ctx.service("notes")
        if pipeline is not None:
            for kind in NotePublication:
                names.update(name for name, _ in pipeline.note_files(kind).values())
        files = {
            name: str(note.parent / name)
            for name in sorted(names)
            if (note.parent / name).is_file()
        }
        source = self._source_file(note)
        if source is not None:
            files[source.name] = str(source)
        return files

    def _source_file(self, note: Path) -> Path | None:
        """Locate the complete source through its owning pipeline."""

        pipeline = self.ctx.service("notes")
        if pipeline is not None:
            found = pipeline.source_path(note.parent.parent.name)
            if found is not None and Path(found).exists():
                return Path(found)
        fallback = note.parent.parent / "lecture_source.md"
        return fallback if fallback.exists() else None

    def _doc_label(self, name: str) -> str:
        """Human title of a note file, asked from the plugin that writes it."""

        pipeline = self.ctx.service("notes")
        if pipeline is not None:
            for kind in NotePublication:
                for filename, title in pipeline.note_files(kind).values():
                    if filename == name:
                        return title
        return _DEFAULT_DOC_LABELS.get(name, Path(name).stem)

    def _course_name(self, path: str) -> str:
        navigation = current_navigation()
        meta = navigation.class_meta
        try:
            relative = (
                Path(path)
                .resolve()
                .relative_to((self.ctx.data_dir / "sessions").resolve())
            )
        except ValueError:
            pass
        else:
            sid = relative.parts[0]
            session = self.ctx.streams.get_session(sid)
            if session is not None:
                meta = session.meta
            else:
                metadata_path = self.ctx.data_dir / "sessions" / sid / "meta.json"
                if metadata_path.exists():
                    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
                    meta = raw.get("meta", raw)
                elif navigation.class_id != sid:
                    meta = {}
        name: Any = meta.get("course_name") or meta.get("course")
        if isinstance(name, dict):
            name = name.get("course_name") or name.get("name") or ""
        return str(name or "").strip() or "未知课程"


class _Live:
    """Accumulate native execution events and throttle full UI snapshots."""

    _INTERVAL = 0.1

    def __init__(self, plugin: AskPlugin, path: str, qa_id: str) -> None:
        self._plugin = plugin
        self._path = path
        self._qa_id = qa_id
        self._agent: dict[str, Any] = {
            "id": "main",
            "session_id": "",
            "role": ExecutionRole.QUESTION_ANSWER,
            "label": "Agent 回答",
            "status": ExecutionStatus.RUNNING,
            "model": "",
            "thinking": "",
            "context_tokens": 0,
            "context_window": 0,
            "llm_seconds": 0,
            "tool_seconds": 0,
            "output_tokens": 0,
            "tokens_per_second": 0,
            "blocks": [],
        }
        self._started = time.monotonic()
        self._last = self._started
        self._lock = threading.Lock()
        self.publish(force=True)

    def bind_session(self, session_id: str) -> None:
        with self._lock:
            self._agent["session_id"] = session_id
        self.publish(force=True)

    def on_event(self, event: Any) -> None:
        if event.type != AgentEventType.EXECUTION:
            return
        data = event.data or {}
        with self._lock:
            self._agent.update(data.get("metrics") or {})
            block = data.get("block")
            if block is not None:
                blocks = self._agent["blocks"]
                for index, existing in enumerate(blocks):
                    if existing["id"] == block["id"]:
                        blocks[index] = self._execution_block(block)
                        break
                else:
                    blocks.append(self._execution_block(block))
        self.publish()

    @staticmethod
    def _execution_block(block: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **block,
            "kind": ExecutionBlockKind(block["kind"]),
            "status": ExecutionBlockStatus(block["status"]),
        }

    def activity(self) -> dict[str, Any]:
        with self._lock:
            return self._activity()

    def _activity(self) -> dict[str, Any]:
        owner = self._plugin.ctx.ui.has_action(MESSAGE_ACTION)
        return {
            "id": f"qa-{self._qa_id}",
            "send_action": MESSAGE_ACTION if owner else "",
            "send_values": {},
            "agents": [
                {
                    **self._agent,
                    "blocks": [dict(block) for block in self._agent["blocks"]],
                }
            ],
        }

    def publish(self, *, force: bool = False) -> None:
        with self._lock:
            now = time.monotonic()
            if not force and now - self._last < self._INTERVAL:
                return
            self._last = now
            payload = {
                "path": self._path,
                "id": self._qa_id,
                "status": QaStatus.PENDING,
                "activity": self._activity(),
                "elapsed": round(now - self._started, 1),
            }
        self._plugin.ctx.ui.notify("qa.live", payload)

    def finish(self, *, status: QaStatus, answer: str, elapsed: float) -> None:
        with self._lock:
            execution_status = (
                ExecutionStatus.COMPLETED
                if status == QaStatus.DONE
                else ExecutionStatus.FAILED
            )
            self._agent["status"] = execution_status
            for block in self._agent["blocks"]:
                if block["status"] in (
                    ExecutionBlockStatus.STREAMING,
                    ExecutionBlockStatus.RUNNING,
                ):
                    block["status"] = ExecutionBlockStatus(execution_status)
            payload = {
                "path": self._path,
                "id": self._qa_id,
                "status": status,
                "activity": self._activity(),
                "answer": answer,
                "answer_html": render_markdown(answer),
                "collapsed": status == QaStatus.DONE,
                "elapsed": round(elapsed, 1),
            }
        self._plugin.ctx.logger.info(
            "ask %s answer (%s, %.1fs):\n%s",
            self._qa_id,
            status,
            elapsed,
            answer,
        )
        self._plugin.ctx.ui.notify("qa.live", payload)


# Fallback titles when the notes plugins are not loaded (service API first).
_DEFAULT_DOC_LABELS = {
    "note_knowledge.md": "知识",
    "note_gossip.md": "闲话与分享",
    "note_notice.md": "通知",
    "chunk_note_knowledge.md": "分段知识",
    "chunk_note_gossip.md": "分段闲话",
    "chunk_note_notice.md": "分段通知",
}


def _truncate_chars(text: str, limit: int) -> str:
    text = (text or "").strip()
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n……（选中内容过长，已截断）"


def _format_history(entries: list[dict[str, str]]) -> str:
    if not entries:
        return "（无）"
    parts: list[str] = []
    for index, entry in enumerate(entries, 1):
        parts.append(
            f"### 问答 {index}\n\n问：{entry.get('question', '')}\n\n"
            f"答：{entry.get('answer', '')}"
        )
    return "\n\n".join(parts)


_SETTING_SPECS: list[SettingsSpec] = [
    SettingsSpec(
        key="ask.before_chars",
        label="选中前文字符数",
        type=SettingType.INT,
        default=2000,
        group="提问",
        description="选中段落之前的笔记内容上限，包含标点、空格和换行",
    ),
    SettingsSpec(
        key="ask.after_chars",
        label="选中后文字符数",
        type=SettingType.INT,
        default=2000,
        group="提问",
        description="上下文里选中段落之后的笔记内容",
    ),
    SettingsSpec(
        key="ask.history_chars",
        label="历史问答字符数",
        type=SettingType.INT,
        default=1000,
        group="提问",
        description="同一条笔记中此前问答记录的独立限额",
    ),
    SettingsSpec(
        key="ask.selection_chars",
        label="选中内容字符数",
        type=SettingType.INT,
        default=800,
        group="提问",
        description="选中片段送入模型的上限，超出部分截断",
    ),
    SettingsSpec(
        key="ask.instant_prompt",
        label="快速模式提示词",
        type=SettingType.TEXT,
        default=INSTANT_SYSTEM_PROMPT,
        group="提问",
        description="快速模式（无工具、一步回答）的系统提示词",
    ),
    SettingsSpec(
        key="ask.research_prompt",
        label="研究模式提示词",
        type=SettingType.TEXT,
        default=RESEARCH_SYSTEM_PROMPT,
        group="提问",
        description="研究模式只读答疑的系统提示词",
    ),
]


__all__ = ["AskPlugin", "ACTION"]
