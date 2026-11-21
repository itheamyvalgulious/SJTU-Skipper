"""Ask plugin: in-note question answering with streaming and two intensities.

The plugin owns *questions*, not the agent loop: the ``agent`` plugin runs
the model, builds the tools from the declarations below and streams deltas
back.  Answers live in the note as ``qa`` markdown blocks (owned by ``reader``).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from core.agent_types import (
    AgentEventType,
    AgentProfile,
    AgentTool,
    AgentToolName,
    ExecutionRole,
    ExecutionBlockKind,
    ExecutionBlockStatus,
    ExecutionStatus,
)
from core.note_document import AskIntensity, QaStatus, strip_blocks
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

# Fallback turn prompt when the tutor plugin (or its editable setting) is absent.
_DEFAULT_TURN_PROMPT = r"""现在学生要针对上面的笔记向你提问。

- 进入提问模式：只回答问题，不要继续写笔记，不要修改任何文件；回答由程序插入笔记。
- 依据上面的课堂材料和笔记回答；笔记/材料里没有的，明确说「笔记里没有」。
- 回答直接、面向本科生；学生追问时再展开，不要一次给太多。
- 笔记里已有定义/记号时优先沿用；补充课外内容时标「补充（课外）」。
- 不要写「根据笔记/根据材料」这类过程性文字。
- 本轮附带的「选中片段 + 附近文字」只用于定位，不要把它当成完整上下文重复复述。
- 工具列表中的文件工具不可用；文件工具之外的其它工具仍可正常使用。"""


class AskService:
    __slots__ = ("_plugin",)

    def __init__(self, plugin: "AskPlugin") -> None:
        self._plugin = plugin

    def ask(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ask one question (same payload as the ``ask.ask`` UI action)."""

        return self._plugin._on_ask(payload)  # noqa: SLF001 - the plugin owns it

    def activity(self, path: str, qa_id: str) -> dict[str, Any] | None:
        return self._plugin._activity(path, qa_id)

    def invalidate_conversations(self) -> None:
        """Drop cached ask conversations (called after prompt/preference edits)."""

        self._plugin.invalidate_conversations()


class AskPlugin(Plugin):
    id = "ask"
    requires = ("settings", "agent", "reader")
    optional_requires = ("notes",)
    provides = ("ask",)
    uses_actions = (MESSAGE_ACTION,)

    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        ctx.llm.register_role(self.id, "ask_instant", "快速问答")
        ctx.llm.register_role(self.id, "ask_research", "研究问答")
        self._settings = ctx.require("settings")
        self._agent = ctx.require("agent")
        self._reader = ctx.require("reader")
        self._active: set[str] = set()
        self._live: dict[tuple[str, str], _Live] = {}
        self._conversations: dict[tuple[str, str], dict[str, Any]] = {}
        self._conv_lock = threading.Lock()
        self._usages: dict[str, dict[str, int]] = {}
        self._active_lock = threading.Lock()
        self._closed = False

        for spec in _SETTING_SPECS:
            self._settings.register(spec)

        ctx.register_service("ask", AskService(self))
        ctx.ui.register_action(ACTION, self._on_ask, plugin_id=self.id)
        ctx.ui.register_action(
            "ask.restore_prompts", self._restore_prompts, plugin_id=self.id
        )
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
        self.invalidate_conversations()
        ctx = getattr(self, "ctx", None)
        if ctx is not None:
            for action in (ACTION, "ask.restore_prompts"):
                try:
                    ctx.ui.unregister_action(action)
                except Exception:  # noqa: BLE001 - shutdown is best effort
                    pass
        reader = getattr(self, "_reader", None)
        if reader is not None:
            try:
                reader.unregister_selection_action("", SELECTION_ID, owner=self.id)
                reader.unregister_activity_provider(self.id)
            except Exception:  # noqa: BLE001 - shutdown is best effort
                pass

    def _restore_prompts(self, payload):
        defaults = {
            "ask.instant_prompt": INSTANT_SYSTEM_PROMPT,
            "ask.research_prompt": RESEARCH_SYSTEM_PROMPT,
        }
        for key, value in defaults.items():
            self._settings.set(key, value)
        return {"ok": True, "message": "已恢复 ask 默认提示词", "refresh": True}

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
            args=(path, qa_id, selection, intensity, course_name, question),
            name=f"ask-{qa_id}",
            daemon=True,
        ).start()

        message = "已提交，Agent 正在回答…"
        if not created.get("anchored", True):
            message += "（未能定位选中片段，已附在笔记末尾）"
        return {
            "ok": True,
            "message": message,
            "refresh": True,
            "data": {"qa_id": qa_id},
        }

    def invalidate_conversations(self) -> None:
        with self._conv_lock:
            conversations = list(self._conversations.values())
            self._conversations.clear()
        for conversation in conversations:
            try:
                self._agent.close(conversation["session_id"])
            except Exception:  # noqa: BLE001 - best effort
                pass
            shutil.rmtree(conversation.get("snapshot", ""), ignore_errors=True)

    def _run(
        self,
        path: str,
        qa_id: str,
        selection: str,
        intensity: AskIntensity,
        course_name: str,
        question: str,
    ) -> None:
        live = _Live(self, path, qa_id)
        key = (str(Path(path).resolve()), qa_id)
        with self._active_lock:
            self._live[key] = live
        status = QaStatus.DONE
        answer = ""
        usage: dict[str, int] = {}
        started = time.time()
        try:
            answer = self._answer(path, qa_id, selection, intensity, live, course_name, question)
            usage = self._usages.pop(qa_id, {})
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
            self._log_turn(path, qa_id, question, answer, intensity, status, usage)
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
        question: str,
    ) -> str:
        context = self._reader.qa_context(
            path,
            qa_id,
            before_chars=min(400, self._before_chars),
            after_chars=min(400, self._after_chars),
            history_chars=self._history_chars,
        )
        conversation, first = self._conversation(path, intensity)
        prompt = self._prompt_for(
            path, question, selection, context, intensity, first, conversation
        )

        started = time.time()
        try:
            result = self._run_turn(conversation, prompt, live)
        except Exception:
            # A stale conversation (e.g. runtime restarted) is rebuilt once.
            self._drop_conversation(path, intensity)
            conversation, first = self._conversation(path, intensity)
            prompt = self._prompt_for(
                path, question, selection, context, intensity, first, conversation
            )
            result = self._run_turn(conversation, prompt, live)

        self.ctx.logger.info(
            "ask %s finished in %.1fs: intensity=%s steps=%s tokens=%s error=%s",
            qa_id,
            time.time() - started,
            intensity,
            result.steps,
            result.total_tokens,
            result.error,
        )
        try:
            conversation["chars"] = int(conversation.get("chars", 0)) + len(
                prompt
            ) + len(result.final_message or "")
            tokens = int(result.input_tokens or 0) or int(conversation["chars"] * 0.56)
            if self.ctx is not None:
                self.ctx.events.emit(
                    "ask.context_usage",
                    session_id=path,
                    chars=int(conversation["chars"]),
                    tokens=tokens,
                )
        except Exception:  # noqa: BLE001 - usage reporting is best effort
            pass
        self._usages[qa_id] = {
            "input_tokens": int(result.input_tokens),
            "output_tokens": int(result.output_tokens),
            "cache_read_tokens": int(result.cache_read_tokens),
            "cache_miss_tokens": int(result.cache_miss_tokens),
            "total_tokens": int(result.total_tokens),
        }
        if result.error:
            raise RuntimeError(result.error)
        return result.final_message.strip() or "（模型没有返回内容）"

    def _prompt_for(
        self,
        path: str,
        question: str,
        selection: str,
        context: Mapping[str, Any],
        intensity: AskIntensity,
        first: bool,
        conversation: dict[str, Any] | None = None,
    ) -> str:
        parts = [self._turn_prompt()]
        preferences = self._preferences_text()
        prefs_hash = _text_hash(preferences)
        writer = bool((conversation or {}).get("writer"))
        if first and not writer:
            snapshot = self._notes_snapshot(path)
            if preferences:
                parts.append("## 学生长期偏好（已确认）\n\n" + preferences)
            parts.append("## 笔记\n\n" + (snapshot or "（暂无笔记正文）"))
            history = _format_history(context.get("history") or [])
            if history and history != "（无）":
                parts.append("## 此前问答记录\n\n" + history)
        elif conversation is not None and preferences:
            if conversation.get("prefs_hash", "") != prefs_hash:
                parts.append(
                    "## 偏好更新（以本节为准，之前提到的偏好不再生效）\n\n"
                    + preferences
                )
        parts.append("## 本轮问题位置\n\n" + self._locator(selection, context))
        parts.append("## 学生的问题\n\n" + question)
        if conversation is not None:
            conversation["prefs_hash"] = prefs_hash
        if writer:
            if intensity == AskIntensity.INSTANT:
                parts.append(
                    "快速模式禁止调用任何文件工具（read、grep、glob、edit、write）；"
                    "文件工具之外的其它工具仍可正常使用，优先直接回答。"
                )
            else:
                parts.append("可用 read、grep、glob 只读核对材料；不得修改任何文件。")
        elif intensity == AskIntensity.RESEARCH:
            readable = "、".join(f"`{item}`" for item in self._workspace_files(path))
            parts.append(f"需要核对材料时可用 read、grep、glob 读取：{readable}。")
        else:
            parts.append("本题没有文件工具，材料已经足够时直接回答。")
        return "\n\n".join(parts)

    def _run_turn(self, conversation: dict[str, Any], prompt: str, live: "_Live"):
        session_id = conversation["session_id"]
        live.bind_session(session_id)
        result = self._agent.run(session_id, prompt, on_event=live.on_event)
        conversation["turns"] = int(conversation.get("turns", 0)) + 1
        return result

    def _drop_conversation(self, path: str, intensity: AskIntensity) -> None:
        sid = Path(path).parent.parent.name
        with self._conv_lock:
            conversation = None
            for key in list(self._conversations):
                if key[0] == sid and (
                    str(key[1]).startswith("writer:")
                    or key == self._conversation_key(path, intensity)
                ):
                    conversation = conversation or self._conversations.pop(key, None)
            conversation = conversation or self._conversations.pop(
                self._conversation_key(path, intensity), None
            )
        if conversation is None:
            return
        try:
            self._agent.close(conversation["session_id"])
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(conversation.get("snapshot", ""), ignore_errors=True)

    def _conversation_key(self, path: str, intensity: AskIntensity):
        sid = Path(path).parent.parent.name
        return (sid, str(intensity))

    def _conversation(self, path: str, intensity: AskIntensity):
        notes = self.ctx.service("notes") if self.ctx is not None else None
        resume = getattr(notes, "resume_writer_for_ask", None)
        writer_id = getattr(notes, "writer_session_id", None)
        sid = Path(path).parent.parent.name
        has_writer = bool(callable(writer_id) and writer_id(sid))
        if callable(resume) and has_writer:
            return self._writer_conversation(path, resume, intensity)
        return self._fallback_conversation(path, intensity)

    def _writer_conversation(self, path: str, resume, intensity: AskIntensity):
        sid = Path(path).parent.parent.name
        key = (sid, f"writer:{intensity}")
        signature = self._note_signature(path)
        with self._conv_lock:
            conversation = self._conversations.get(key)
            if conversation is not None and conversation.get("signature") == signature:
                return conversation, int(conversation.get("turns", 0)) == 0
            old = self._conversations.pop(key, None)
            # Only one resumed writer session may be live at a time.
            others = [
                (other_key, other)
                for other_key, other in self._conversations.items()
                if other_key[0] == sid
                and other_key[1] != key[1]
                and str(other_key[1]).startswith("writer:")
            ]
            for other_key, _other in others:
                self._conversations.pop(other_key, None)
        if old is not None:
            try:
                self._agent.close(old["session_id"])
            except Exception:  # noqa: BLE001
                pass
        for _other_key, other in others:
            try:
                self._agent.close(other["session_id"])
            except Exception:  # noqa: BLE001
                pass
        tools: list[str] = []
        if intensity == AskIntensity.RESEARCH:
            tools = [
                AgentToolName.READ,
                AgentToolName.GREP,
                AgentToolName.GLOB,
            ]
        try:
            info = resume(sid, tools)
        except Exception as exc:  # noqa: BLE001 - surfaced to the student
            raise RuntimeError(str(exc)) from exc
        conversation = {
            "session_id": info.id,
            "signature": signature,
            "turns": 0,
            "snapshot": "",
            "prefs_hash": "",
            "writer": True,
        }
        with self._conv_lock:
            self._conversations[key] = conversation
        return conversation, True

    def _fallback_conversation(self, path: str, intensity: AskIntensity):
        key = self._conversation_key(path, intensity)
        signature = self._note_signature(path)
        with self._conv_lock:
            conversation = self._conversations.get(key)
            if conversation is not None and conversation.get("signature") == signature:
                return conversation, int(conversation.get("turns", 0)) == 0
            old = self._conversations.pop(key, None)
        if old is not None:
            try:
                self._agent.close(old["session_id"])
            except Exception:  # noqa: BLE001
                pass
            shutil.rmtree(old.get("snapshot", ""), ignore_errors=True)

        snapshot = ""
        files: dict[str, str] = {}
        if intensity == AskIntensity.RESEARCH:
            snapshot = tempfile.mkdtemp(prefix="notes-ask-conv-")
            for name, original in self._workspace_files(path).items():
                destination = Path(snapshot) / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(Path(original).read_bytes())
                files[name] = str(destination)
        profile = self._profile(intensity)
        session = self._agent.create_session(
            name=f"ask-conv-{key[0][:12]}-{key[1]}",
            agent=profile,
            workspace=FileWorkspace(files, readonly=files),
            meta={
                "role": ExecutionRole.QUESTION_ANSWER,
                "conversation": True,
                "notes_session": key[0],
            },
        )
        conversation = {
            "session_id": session.id,
            "signature": signature,
            "turns": 0,
            "snapshot": snapshot,
            "prefs_hash": "",
        }
        with self._conv_lock:
            self._conversations[key] = conversation
        return conversation, True

    def _note_signature(self, path: str) -> str:
        digest = hashlib.sha256()
        try:
            sid = Path(path).parent.parent.name
            notes = self.ctx.data_dir / "sessions" / sid / "notes"
            pipeline = self.ctx.service("notes")
            names = [Path(path).name]
            if pipeline is not None:
                for kind in NotePublication:
                    names.extend(
                        name for name, _label in pipeline.note_files(kind).values()
                    )
            for name in sorted(set(names)):
                candidate = notes / name
                if candidate.is_file():
                    digest.update(name.encode())
                    digest.update(
                        strip_blocks(
                            candidate.read_text("utf-8", errors="replace")
                        ).encode("utf-8")
                    )
        except OSError:
            return ""
        return digest.hexdigest()

    def _notes_snapshot(self, path: str) -> str:
        notes = Path(path).parent
        pipeline = self.ctx.service("notes")
        names: list[tuple[str, str]] = []
        if pipeline is not None:
            for _key, (name, label) in pipeline.note_files(NotePublication.FINAL).items():
                names.append((name, label))
        else:
            names = [(Path(path).name, "笔记")]
        parts: list[str] = []
        for name, label in names:
            candidate = notes / name
            if not candidate.is_file():
                continue
            text = strip_blocks(
                candidate.read_text("utf-8", errors="replace")
            ).strip()
            if not text:
                continue
            parts.append(f"### {label}（{name}）\n\n{text}")
        return "\n\n".join(parts)

    def _locator(self, selection: str, context: Mapping[str, Any]) -> str:
        selected = _truncate_chars(selection, self._selection_chars) or "（未选中，问题针对整篇笔记）"
        before = _truncate_chars(str(context.get("before") or ""), 400) or "（无）"
        after = _truncate_chars(str(context.get("after") or ""), 400) or "（无）"
        return f"选中片段：\n{selected}\n\n前文：\n{before}\n\n后文：\n{after}"

    def _preferences_text(self) -> str:
        try:
            return str(self._settings.get("tutor.preferences_text", "") or "").strip()
        except Exception:  # noqa: BLE001
            return ""

    def _turn_prompt(self) -> str:
        return str(
            self._settings.get("tutor.ask_prompt", _DEFAULT_TURN_PROMPT)
            or _DEFAULT_TURN_PROMPT
        )

    def _log_turn(
        self,
        path: str,
        qa_id: str,
        question: str,
        answer: str,
        intensity: AskIntensity,
        status: QaStatus,
        usage: dict[str, int] | None = None,
    ) -> None:
        try:
            sid = Path(path).parent.parent.name
            directory = self.ctx.data_dir / "sessions" / sid
            directory.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "qa_id": qa_id,
                "intensity": str(intensity),
                "status": str(status),
                "question": question,
                "answer": answer,
                **{
                    key: int(value)
                    for key, value in (usage or {}).items()
                },
            }
            with (directory / "ask_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            self.ctx.logger.debug("ask log write failed", exc_info=True)

    def _profile(self, intensity: AskIntensity) -> AgentProfile:
        return AgentProfile(
            name=f"ask.{intensity}",
            role="ask_instant" if intensity == AskIntensity.INSTANT else "ask_research",
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


def _text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


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
        label="前文上限（字符）",
        type=SettingType.INT,
        default=2000,
        group="提问",
        section="提问上下文",
        order=10,
        description="选中段落之前的笔记内容．",
    ),
    SettingsSpec(
        key="ask.after_chars",
        label="后文上限（字符）",
        type=SettingType.INT,
        default=2000,
        group="提问",
        section="提问上下文",
        order=20,
        description="选中段落之后的笔记内容．",
    ),
    SettingsSpec(
        key="ask.history_chars",
        label="历史问答上限（字符）",
        type=SettingType.INT,
        default=1000,
        group="提问",
        section="提问上下文",
        order=30,
        description="同一条笔记中的此前问答记录．",
    ),
    SettingsSpec(
        key="ask.selection_chars",
        label="选中内容上限（字符）",
        type=SettingType.INT,
        default=800,
        group="提问",
        section="提问上下文",
        order=0,
        description="超出上限的选中内容不会发送给模型．",
    ),
    SettingsSpec(
        key="ask.instant_prompt",
        label="快速模式提示词",
        type=SettingType.TEXT,
        default=INSTANT_SYSTEM_PROMPT,
        group="提问",
        section="回答提示词",
        order=100,
        description="快速模式不使用工具，一次生成回答．留空时使用默认提示词．",
    ),
    SettingsSpec(
        key="ask.research_prompt",
        label="研究模式提示词",
        type=SettingType.TEXT,
        default=RESEARCH_SYSTEM_PROMPT,
        group="提问",
        section="回答提示词",
        order=110,
        description="研究模式使用只读工具查阅资料．留空时使用默认提示词．",
    ),
]


__all__ = ["AskPlugin", "ACTION"]
