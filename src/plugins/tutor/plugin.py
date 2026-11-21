"""Tutor plugin: user-editable note modes, shared preferences, prompt studio.

This plugin is the product layer on top of ``notes`` and ``ask``:

* it owns the editable *prompt studio* (common prompt, four note modes, ask
  turn prompt) and writes the composed notes prompt into ``notes.mode_prompt``
  before starting a generation;
* it owns the shared preference profile and a real agent tool
  (``propose_preference``) that can only *propose* changes — the user accepts
  them on the settings page;
* it injects preferences into the ask roles through ``AgentExtension`` and
  composes them into the notes writer prompt;
* it adds a "生成与迭代" toolbox panel to the existing 笔记 tab with mode
  selection, note generation, ask-driven note iteration and version history;
* it keeps note versions under ``sessions/<sid>/notes_history/<ts>/``.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from core.agent_extensions import AgentExtension, AgentExtensionTool
from core.file_access import publication_lock
from core.note_document import parse_blocks, strip_blocks
from core.plugin import Plugin, PluginContext
from core.settings_spec import SettingsSpec, SettingType
from core.ui import BlockType, FormInputType

from .modes import MAX_PROMPT, ModeStore
from .prompts import (
    ASK_DEFAULT,
    COMMON_PROMPT,
    MODES,
    PROPOSAL_TOOL_PROMPT,
    WRITER_CONTRACT,
)
from .store import (
    CATEGORIES,
    CATEGORY_LABELS,
    MAX_TEXT,
    SCOPES,
    SCOPE_LABELS,
    PreferenceStore,
)

LOG = logging.getLogger("lecture_agent.plugin.tutor")

SECTION_ID = "tutor"
PAGE_HISTORY = "tutor:history"
ASK_ROLES = ("ask_instant", "ask_research")


def _values(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("values")
    return nested if isinstance(nested, Mapping) else payload


def _text(payload: Mapping[str, Any], *names: str) -> str:
    values = _values(payload)
    for name in names:
        value = values.get(name)
        if value is None:
            value = payload.get(name)
        if value is not None:
            return str(value).strip()
    return ""


def _field(
    name: str,
    label: str,
    value: Any = "",
    *,
    type: FormInputType = FormInputType.STRING,
    input: FormInputType = FormInputType.TEXT,
    **options: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "key": name,
        "label": label,
        "type": type,
        "input": input,
        "value": value,
        **options,
    }


class TutorService:
    """Narrow facade exposed to other plugins and read lazily by the UI."""

    __slots__ = ("_plugin",)

    def __init__(self, plugin: "TutorPlugin") -> None:
        self._plugin = plugin

    def preferences(self) -> str:
        return self._plugin.preference_text(("both", "ask"))

    def note_modes(self) -> list[tuple[str, str]]:
        return self._plugin.modes.options()

    def mode_prompt(self, mode_id: str) -> str:
        return self._plugin.compose_notes_prompt(mode_id)

    def ask_prompt(self) -> str:
        return self._plugin.ask_prompt()


class TutorPlugin(Plugin):
    id = "tutor"
    requires = ("settings", "agent")
    optional_requires = ("ask", "notes")
    provides = ("tutor",)
    action_namespace = "tutor"

    def __init__(self) -> None:
        self.ctx: PluginContext | None = None
        self.settings: Any = None
        self.agent: Any = None
        self.prefs: PreferenceStore | None = None
        self.modes: ModeStore | None = None
        self._lock = threading.RLock()
        self._history_pages: set[str] = set()
        self._compacting: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self.settings = ctx.require("settings")
        self.agent = ctx.require("agent")
        self.prefs = PreferenceStore(
            ctx.data_dir / "tutor" / "preferences.json",
            defaults_path=Path(__file__).with_name("defaults.json"),
        )
        self.prefs.load()
        self.modes = ModeStore(ctx.data_dir / "tutor" / "modes.json", MODES)
        self.modes.load()

        for spec in (
            SettingsSpec(
                key="tutor.common_prompt",
                label="通用提示词",
                type=SettingType.TEXT,
                default=COMMON_PROMPT,
                group="Tutor",
                hidden=True,
            ),
            SettingsSpec(
                key="tutor.ask_prompt",
                label="提问提示词",
                type=SettingType.TEXT,
                default=ASK_DEFAULT,
                group="Tutor",
                hidden=True,
            ),
            SettingsSpec(
                key="tutor.writer_contract",
                label="笔记输出约定",
                type=SettingType.TEXT,
                default=WRITER_CONTRACT,
                group="Tutor",
                hidden=True,
            ),
            SettingsSpec(
                key="tutor.proposal_prompt",
                label="偏好工具说明",
                type=SettingType.TEXT,
                default=PROPOSAL_TOOL_PROMPT,
                group="Tutor",
                hidden=True,
            ),
            SettingsSpec(
                key="tutor.context_budget",
                label="问答上下文预算（token）",
                type=SettingType.INT,
                default=120000,
                group="Tutor",
                hidden=True,
            ),
            SettingsSpec(
                key="tutor.compact_warn_ratio",
                label="建议更新笔记的比例",
                type=SettingType.FLOAT,
                default=0.6,
                group="Tutor",
                hidden=True,
            ),
            SettingsSpec(
                key="tutor.compact_auto_ratio",
                label="自动更新笔记的比例",
                type=SettingType.FLOAT,
                default=0.85,
                group="Tutor",
                hidden=True,
            ),
            SettingsSpec(
                key="tutor.propose_enabled",
                label="允许助教提议偏好",
                type=SettingType.BOOL,
                default=True,
                group="Tutor",
                hidden=True,
            ),
        ):
            self.settings.register(spec, plugin_id=self.id)

        ctx.register_service("tutor", TutorService(self))
        for action, handler in (
            ("tutor.notes.generate", self._generate_notes),
            ("tutor.notes.iterate", self._iterate_notes),
            ("tutor.notes.history", self._open_history),
            ("tutor.notes.ask_log", self._open_ask_log),
            ("tutor.mode.add", self._mode_add),
            ("tutor.mode.save", self._mode_save),
            ("tutor.mode.delete", self._mode_delete),
            ("tutor.mode.default", self._mode_default),
            ("tutor.prompt.save", self._prompt_save),
            ("tutor.prompt.restore", self._prompt_restore),
            ("tutor.pref.add", self._pref_add),
            ("tutor.pref.remove", self._pref_remove),
            ("tutor.pref.accept", self._pref_accept),
            ("tutor.pref.reject", self._pref_reject),
            ("tutor.pref.export", self._pref_export),
            ("tutor.pref.reset", self._pref_reset),
            ("tutor.pref.clear", self._pref_clear),
        ):
            ctx.ui.register_action(action, handler, plugin_id=self.id)
        self.settings.register_section(SECTION_ID, "教学与偏好", self._render)

        ctx.events.on("ask.context_usage", self._on_context_usage, plugin_id=self.id)
        self._sync_mode_prompt()
        self._sync_preferences_text()
        if ctx.service("notes") is not None:
            ctx.ui.add_toolbox_item(
                "note",
                "tutor.notes",
                "",
                description="选择模式生成笔记、根据问答迭代笔记",
                order=12,
                render=self._render_notes_panel,
                plugin_id=self.id,
            )
        self._register_extensions()

    def teardown(self) -> None:
        ctx = self.ctx
        if ctx is None:
            return
        try:
            self.agent.unregister_extensions(self.id)
        except Exception:  # noqa: BLE001
            LOG.debug("unregister extensions failed", exc_info=True)
        for action in (
            "tutor.notes.generate",
            "tutor.notes.iterate",
            "tutor.notes.history",
            "tutor.notes.ask_log",
            "tutor.mode.add",
            "tutor.mode.save",
            "tutor.mode.delete",
            "tutor.mode.default",
            "tutor.prompt.save",
            "tutor.prompt.restore",
            "tutor.pref.add",
            "tutor.pref.remove",
            "tutor.pref.accept",
            "tutor.pref.reject",
            "tutor.pref.export",
            "tutor.pref.reset",
            "tutor.pref.clear",
        ):
            try:
                ctx.ui.unregister_action(action)
            except Exception:  # noqa: BLE001
                pass
        try:
            ctx.ui.remove_toolbox_item("tutor.notes")
        except Exception:  # noqa: BLE001
            pass
        for page in list(self._history_pages):
            try:
                ctx.ui.remove_page(page)
            except Exception:  # noqa: BLE001
                pass
        try:
            self.settings.unregister_section(SECTION_ID)
        except Exception:  # noqa: BLE001
            pass
        self.ctx = None

    # ------------------------------------------------------------------
    # Prompt composition
    # ------------------------------------------------------------------
    def _current_sid(self, payload: Mapping[str, Any] | None = None) -> str:
        sid = str((payload or {}).get("session_id") or "").strip()
        if sid and sid != "boot":
            return sid
        try:
            from core.navigation import current_navigation

            sid = str(current_navigation().class_id or "")
        except Exception:  # noqa: BLE001
            sid = ""
        return "" if sid == "boot" else sid

    def _default_mode_id(self) -> str:
        modes = self.modes.items() if self.modes is not None else []
        if not modes:
            return ""
        configured = str(self.settings.get("tutor.default_mode", "") or "")
        if configured and any(item["id"] == configured for item in modes):
            return configured
        for item in modes:
            if item["id"] == "detailed":
                return configured or "detailed"
        return modes[0]["id"]

    def _course_name(self, sid: str) -> str:
        ctx = self.ctx
        if ctx is None:
            return "本课程"
        session = ctx.streams.get_session(sid) if sid else None
        meta = getattr(session, "meta", None) or {}
        path = ctx.data_dir / "sessions" / sid / "meta.json"
        if not meta and sid and path.exists():
            try:
                raw = json.loads(path.read_text("utf-8"))
                meta = raw.get("meta") or raw
            except (OSError, json.JSONDecodeError):
                meta = {}
        return str(meta.get("course_name") or meta.get("title") or "本课程")

    def common_prompt(self) -> str:
        return str(self.settings.get("tutor.common_prompt", COMMON_PROMPT) or COMMON_PROMPT)

    def ask_prompt(self) -> str:
        return str(self.settings.get("tutor.ask_prompt", ASK_DEFAULT) or ASK_DEFAULT)

    def writer_contract(self) -> str:
        return str(
            self.settings.get("tutor.writer_contract", WRITER_CONTRACT)
            or WRITER_CONTRACT
        )

    def proposal_prompt(self) -> str:
        return str(
            self.settings.get("tutor.proposal_prompt", PROPOSAL_TOOL_PROMPT)
            or PROPOSAL_TOOL_PROMPT
        )

    def preference_text(self, scopes: tuple[str, ...] = ("both", "ask", "notes")) -> str:
        store = self.prefs
        return store.block(scopes) if store is not None else ""

    def _prefs_enabled(self) -> bool:
        return bool(self.settings.get("tutor.propose_enabled", True))

    def compose_notes_prompt(self, mode_id: str, sid: str = "") -> str:
        modes = self.modes
        if modes is None:
            return ""
        mode = modes.get(mode_id) or modes.get(self._default_mode_id()) or modes.items()[0]
        parts = [
            self.common_prompt().replace("{{course}}", self._course_name(sid)),
            mode["prompt"],
            self.writer_contract(),
            self.preference_text(("both", "notes")),
        ]
        return "\n\n".join(part for part in parts if part.strip())

    def _apply_mode(self, mode_id: str = "") -> None:
        """Publish the writer system prompt / instructions / reviewer mode text.

        * writer system prompt = 公共提示词 (task agnostic, so the same
          conversation can be continued by ask);
        * writer instruction   = mode prompt + output contract + preferences;
        * notes.mode_prompt    = mode prompt only, used by the reviewer.
        """

        if self.modes is None:
            return
        mode = self.modes.get(mode_id) or self.modes.get(self._default_mode_id())
        if mode is None:
            return
        self.settings.set("notes.writer_system_prompt", self.common_prompt())
        writer_extra = "\n\n".join(
            part
            for part in (
                mode["prompt"],
                self.writer_contract(),
                self.preference_text(("both", "notes")),
            )
            if part.strip()
        )
        self.settings.set("notes.writer_instruction", writer_extra)
        self.settings.set("notes.mode_prompt", mode["prompt"])

    def _sync_mode_prompt(self) -> None:
        self._apply_mode("")

    def _open_ask_log(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        reader = self.ctx.service("reader") if self.ctx is not None else None
        if reader is None:
            return {"ok": False, "message": "阅读器未启用"}
        sid = self._current_sid(payload)
        if not sid or sid == "boot":
            return {"ok": False, "message": "请先选择课次"}
        path = self.ctx.data_dir / "sessions" / sid / "ask_log.jsonl"
        preview = self.ctx.data_dir / "sessions" / sid / "ask_log.md"
        lines = []
        if path.exists():
            for line in path.read_text("utf-8", errors="replace").splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = entry.get("type")
                if kind == "note_updated":
                    lines.append(
                        "> 笔记已根据问答更新"
                        + (f"（旧版归档：{entry.get('archived')}）" if entry.get("archived") else "")
                    )
                elif kind == "note_update_failed":
                    lines.append("> 笔记更新失败，问答记录保留")
                elif kind == "context_warning":
                    lines.append("> 上下文接近上限，建议手动更新笔记")
                else:
                    lines.append(
                        f"## 问：{entry.get('question', '')}\n\n{entry.get('answer', '')}"
                    )
        preview.write_text("# 问答记录\n\n" + "\n\n".join(lines) if lines else "# 问答记录\n\n（暂无）", "utf-8")
        page = reader.show(
            title="问答记录",
            docs=[{"key": "ask_log", "label": "问答记录", "path": str(preview)}],
            page_key=f"{sid}:ask-log",
            owner=self.id,
        )
        return {"ok": True, "open_page": page}

    # ------------------------------------------------------------------
    # Agent extension: ask prompt layer + propose_preference tool
    # ------------------------------------------------------------------
    def _tool_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["add", "update", "remove"]},
                            "text": {"type": "string"},
                            "category": {
                                "type": "string",
                                "enum": list(CATEGORIES),
                            },
                            "scope": {"type": "string", "enum": list(SCOPES)},
                            "id": {"type": "string"},
                        },
                        "required": ["op"],
                        "additionalProperties": False,
                    },
                },
                "reason": {"type": "string"},
            },
            "required": ["operations"],
            "additionalProperties": False,
        }

    def _propose_tool(self, context: Mapping[str, Any]) -> dict[str, Any]:
        store = self.prefs
        if store is None or not self._prefs_enabled():
            return {
                "content": [{"type": "text", "text": "偏好提议当前不可用。"}],
                "isError": True,
                "details": {},
            }
        params = context.get("parameters") or {}
        operations = params.get("operations") if isinstance(params, Mapping) else None
        reason = str(params.get("reason") or "") if isinstance(params, Mapping) else ""
        proposal = store.propose(operations or [], source="ask", reason=reason)
        if proposal is None:
            return {
                "content": [
                    {"type": "text", "text": "没有可记录的偏好（提议为空、无效或已达上限），本轮不要声称记住了任何内容。"}
                ],
                "isError": True,
                "details": {},
            }
        self.refresh()
        count = len(proposal.operations)
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"已提交 {count} 条偏好建议（#{proposal.id}），等待学生在设置页确认后才会生效。",
                }
            ],
            "details": {"proposal_id": proposal.id, "count": count},
        }

    def _ask_extension(self) -> AgentExtension:
        prompt = self.proposal_prompt() if self._prefs_enabled() else ""
        tools: tuple[AgentExtensionTool, ...] = ()
        if self._prefs_enabled():
            tools = (
                AgentExtensionTool(
                    name="propose_preference",
                    label="提议修改偏好",
                    description=(
                        "向学生提议记录或修改跨对话的长期偏好。只提交建议，"
                        "学生确认前不会生效；返回建议编号。"
                    ),
                    parameters=self._tool_parameters(),
                    handler=self._propose_tool,
                ),
            )
        return AgentExtension(
            id="preferences",
            roles=ASK_ROLES,
            prompt=prompt,
            tools=tools,
        )

    def _register_extensions(self) -> None:
        ctx = self.ctx
        if ctx is None:
            return
        ask_extension = self._ask_extension()
        if ctx.service("ask") is not None:
            try:
                self.agent.register_extension(self.id, ask_extension)
            except Exception:  # noqa: BLE001 - roles may be missing while loading
                LOG.warning("tutor ask extension not registered", exc_info=True)
        if ctx.service("notes") is not None:
            try:
                # ask continues the writer conversation, whose role is
                # note_writer; it needs the same proposal tool.
                self.agent.register_extension(
                    self.id,
                    AgentExtension(
                        id="preferences-writer",
                        roles=("note_writer",),
                        prompt=ask_extension.prompt,
                        tools=ask_extension.tools,
                    ),
                )
            except Exception:  # noqa: BLE001
                LOG.warning("tutor writer extension not registered", exc_info=True)

    def _refresh_extensions(self) -> None:
        with self._lock:
            try:
                self.agent.unregister_extensions(self.id)
            except Exception:  # noqa: BLE001
                LOG.debug("unregister failed", exc_info=True)
            self._register_extensions()

    def _sync_preferences_text(self) -> None:
        """Publish preferences as a setting so ask turns can inject them lazily.

        Preferences are NOT part of the ask system prompt any more: changing
        them must not invalidate an ongoing conversation (that would throw away
        the whole cached prefix).  The ask plugin appends the current text as a
        short turn-tail block, so only the tail changes when preferences change.
        """

        self.settings.set("tutor.preferences_text", self.preference_text(("both", "ask")))

    def _prefs_changed(self) -> None:
        self._sync_mode_prompt()
        self._sync_preferences_text()
        self.refresh()

    def refresh(self) -> None:
        ctx = self.ctx
        if ctx is None:
            return
        try:
            ctx.ui.refresh("tutor")
        except Exception:  # noqa: BLE001
            LOG.debug("tutor refresh failed", exc_info=True)

    # ------------------------------------------------------------------
    # Notes actions (mode generation / ask-driven iteration / history)
    # ------------------------------------------------------------------
    def _notes(self):
        return self.ctx.service("notes") if self.ctx is not None else None

    def _generate_notes(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        notes = self._notes()
        if notes is None:
            return {"ok": False, "message": "笔记插件未启用"}
        sid = self._current_sid(payload)
        if not sid or sid == "boot":
            return {"ok": False, "message": "请先选择课次"}
        modes = self.modes
        if modes is None:
            return {"ok": False, "message": "模式未就绪"}
        mode_id = str(payload.get("mode") or "").strip()
        mode = modes.get(mode_id) or modes.get(self._default_mode_id()) or modes.items()[0]
        self._apply_mode(mode["id"])
        started = notes.start(sid)
        if not started:
            return {"ok": False, "refresh": True, "message": "笔记未开始生成，请检查状态和来源"}
        return {
            "ok": True,
            "refresh": True,
            "message": f"正在按「{mode['name']}」模式生成笔记",
        }

    def _notes_dir(self, sid: str) -> Path:
        return self.ctx.data_dir / "sessions" / sid / "notes"  # type: ignore[union-attr]

    def _archive_notes(self, sid: str) -> Path | None:
        notes = self._notes_dir(sid)
        if not notes.is_dir():
            return None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = self.ctx.data_dir / "sessions" / sid / "notes_history" / stamp  # type: ignore[union-attr]
        target.mkdir(parents=True, exist_ok=True)
        copied = 0
        for path in sorted(notes.glob("note_*.md")):
            shutil.copy2(path, target / path.name)
            copied += 1
        return target if copied else None

    def _qa_digest(self, sid: str) -> str:
        notes = self._notes_dir(sid)
        if not notes.is_dir():
            return ""
        lines: list[str] = []
        for path in sorted(notes.glob("note_*.md")):
            try:
                text = path.read_text("utf-8")
            except OSError:
                continue
            for block in parse_blocks(text):
                question = (block.question or "").strip()
                answer = (block.answer or "").strip()
                if question and answer and block.status.value == "done":
                    lines.append(f"问：{question}\n答：{answer}")
        return "\n\n".join(lines)

    def _iterate_notes(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        notes = self._notes()
        if notes is None:
            return {"ok": False, "message": "笔记插件未启用"}
        sid = self._current_sid(payload)
        if not sid or sid == "boot":
            return {"ok": False, "message": "请先选择课次"}
        digest = self._qa_digest(sid)
        if not digest:
            return {"ok": False, "message": "还没有可用的问答记录"}
        archived = self._archive_notes(sid)
        # The note update will invalidate the whole ask cache: close the
        # resumed writer/ask conversation first, then let notes reopen it with
        # write tools.  Records are marked only after publication succeeds.
        self._release_ask(sid)
        message = (
            "根据下面学生与助教的问答，更新最终笔记（保持当前模式的结构与风格）。\n"
            "把问答中有价值的新解释、例子、纠正并入正文对应位置，不要保留问答格式；"
            "没有新增价值的问答忽略。只修改 note_knowledge.md 等笔记文稿。\n\n"
            + digest
        )
        result = notes.message({"session_id": sid, "message": message})
        ok = bool(result.get("ok"))
        reply = result.get("message") or ("已提交笔记更新，完成后会发布新版" if ok else "笔记更新未提交")
        if ok:
            threading.Thread(
                target=self._watch_iteration,
                args=(sid, archived.name if archived else ""),
                name=f"tutor-iterate-{sid}",
                daemon=True,
            ).start()
        return {"ok": ok, "refresh": True, "message": reply}

    def _release_ask(self, sid: str) -> None:
        ask = self.ctx.service("ask") if self.ctx is not None else None
        close = getattr(ask, "invalidate_conversations", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                LOG.debug("ask conversation release failed", exc_info=True)

    def _watch_iteration(self, sid: str, archived: str) -> None:
        notes = self._notes()
        if notes is None:
            return
        deadline = time.time() + 1800
        while time.time() < deadline:
            status = notes.status(sid)
            phase = str(status.get("phase") or "")
            if phase == "published":
                self._clear_qa_blocks(sid)
                self._mark_ask_log(sid, archived)
                self.refresh()
                return
            if phase in ("failed", "cancelled"):
                self._mark_ask_log(sid, archived, failed=True)
                self.refresh()
                return
            time.sleep(2)

    def _clear_qa_blocks(self, sid: str) -> int:
        notes = self._notes_dir(sid)
        if not notes.is_dir():
            return 0
        removed = 0
        with publication_lock(notes):
            for path in sorted(notes.glob("note_*.md")):
                try:
                    text = path.read_text("utf-8")
                except OSError:
                    continue
                cleaned = strip_blocks(text).rstrip() + "\n"
                if cleaned != text:
                    removed += 1
                    path.write_text(cleaned, "utf-8")
        return removed

    def _mark_ask_log(
        self, sid: str, archived: str, failed: bool = False, warning: bool = False
    ) -> None:
        try:
            directory = self.ctx.data_dir / "sessions" / sid
            directory.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": (
                    "note_update_failed"
                    if failed
                    else "context_warning"
                    if warning
                    else "note_updated"
                ),
                "archived": archived,
            }
            with (directory / "ask_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            LOG.debug("ask log marker failed", exc_info=True)

    def _on_context_usage(self, event: Any) -> None:
        path = str(event.get("session_id") or "")
        sid = Path(path).parent.parent.name if path else ""
        if not sid or sid == "boot":
            return
        tokens = int(event.get("tokens") or 0)
        budget = max(1, int(self.settings.get("tutor.context_budget", 120000) or 120000))
        ratio = tokens / budget
        warn = float(self.settings.get("tutor.compact_warn_ratio", 0.6) or 0.6)
        auto = float(self.settings.get("tutor.compact_auto_ratio", 0.85) or 0.85)
        if ratio < warn:
            return
        if ratio < auto:
            self._mark_ask_log(sid, "", warning=True)
            self._notify(
                f"问答上下文已用到约 {ratio:.0%}，建议手动点「根据问答更新笔记」。"
            )
            return
        if sid in self._compacting:
            return
        self._compacting.add(sid)
        self._notify(f"问答上下文已到约 {ratio:.0%}，正在自动更新笔记…")
        threading.Thread(
            target=self._auto_compact, args=(sid,), name=f"tutor-auto-compact-{sid}", daemon=True
        ).start()

    def _auto_compact(self, sid: str) -> None:
        try:
            self._iterate_notes({"session_id": sid})
        except Exception:  # noqa: BLE001
            LOG.exception("auto compact failed")
        finally:
            self._compacting.discard(sid)

    def _notify(self, message: str) -> None:
        ctx = self.ctx
        if ctx is None:
            return
        try:
            ctx.ui.notify("ui.toast", {"message": message})
        except Exception:  # noqa: BLE001
            LOG.debug("toast failed", exc_info=True)

    def _mark_ask_log_warning(self, sid: str) -> None:
        self._mark_ask_log(sid, "", warning=True)

    def _open_history(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        reader = self.ctx.service("reader") if self.ctx is not None else None
        if reader is None:
            return {"ok": False, "message": "阅读器未启用"}
        sid = self._current_sid(payload)
        root = self.ctx.data_dir / "sessions" / sid / "notes_history"  # type: ignore[union-attr]
        docs: list[dict[str, str]] = []
        if root.is_dir():
            for version in sorted(root.iterdir(), reverse=True):
                if not version.is_dir():
                    continue
                for path in sorted(version.glob("note_*.md")):
                    docs.append(
                        {
                            "key": f"{version.name}:{path.stem}",
                            "label": f"{version.name} / {path.stem}",
                            "path": str(path),
                        }
                    )
        if not docs:
            return {"ok": False, "message": "还没有历史版本"}
        page = reader.show(
            title="笔记历史",
            docs=docs,
            page_key=f"{sid}:notes-history",
            owner=self.id,
        )
        return {"ok": True, "open_page": page}

    def _history_versions(self, sid: str) -> list[str]:
        root = self.ctx.data_dir / "sessions" / sid / "notes_history"  # type: ignore[union-attr]
        if not root.is_dir():
            return []
        return [path.name for path in sorted(root.iterdir(), reverse=True) if path.is_dir()]

    # ------------------------------------------------------------------
    # Mode / prompt actions
    # ------------------------------------------------------------------
    def _mode_add(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.modes is None:
            return {"ok": False, "message": "模式未就绪"}
        ok, message = self.modes.add(
            _text(payload, "name"),
            _text(payload, "prompt"),
            _text(payload, "description"),
        )
        self.refresh()
        return {"ok": ok, "message": message, "refresh": True}

    def _mode_save(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.modes is None:
            return {"ok": False, "message": "模式未就绪"}
        mode_id = _text(payload, "mode_id", "id")
        ok, message = self.modes.update(
            mode_id,
            _text(payload, "name"),
            _text(payload, "prompt"),
            _text(payload, "description"),
        )
        self._sync_mode_prompt()
        self.refresh()
        return {"ok": ok, "message": message, "refresh": True}

    def _mode_delete(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.modes is None:
            return {"ok": False, "message": "模式未就绪"}
        ok = self.modes.delete(_text(payload, "mode_id", "id"))
        self._sync_mode_prompt()
        self.refresh()
        return {
            "ok": ok,
            "message": "已删除模式" if ok else "至少保留一个模式",
            "refresh": True,
        }

    def _mode_default(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        mode_id = _text(payload, "mode")
        if self.modes is None or self.modes.get(mode_id) is None:
            return {"ok": False, "message": "没有找到该模式"}
        self.settings.set("tutor.default_mode", mode_id)
        self.refresh()
        return {"ok": True, "message": "已设为默认模式", "refresh": True}

    def _prompt_save(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        common = _text(payload, "common_prompt")
        ask = _text(payload, "ask_prompt")
        mode = _text(payload, "mode")
        contract = _text(payload, "writer_contract")
        proposal = _text(payload, "proposal_prompt")
        if contract:
            self.settings.set("tutor.writer_contract", contract)
        if proposal:
            self.settings.set("tutor.proposal_prompt", proposal)
        if common:
            self.settings.set("tutor.common_prompt", common)
        if ask:
            self.settings.set("tutor.ask_prompt", ask)
        if mode and self.modes is not None and self.modes.get(mode) is not None:
            self.settings.set("tutor.default_mode", mode)
        self._sync_mode_prompt()
        self._sync_preferences_text()
        self._refresh_extensions()
        self.refresh()
        return {"ok": True, "message": "已保存提示词", "refresh": True}

    def _prompt_restore(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        scope = _text(payload, "scope") or "all"
        if scope in ("common", "all"):
            self.settings.set("tutor.common_prompt", COMMON_PROMPT)
            self.settings.set("tutor.writer_contract", WRITER_CONTRACT)
            self.settings.set("tutor.proposal_prompt", PROPOSAL_TOOL_PROMPT)
        if scope in ("ask", "all"):
            self.settings.set("tutor.ask_prompt", ASK_DEFAULT)
        if scope in ("modes", "all") and self.modes is not None:
            self.modes.reset_to_defaults()
        self._sync_mode_prompt()
        self._sync_preferences_text()
        self._refresh_extensions()
        self.refresh()
        return {"ok": True, "message": "已恢复默认值", "refresh": True}

    # ------------------------------------------------------------------
    # Preference actions
    # ------------------------------------------------------------------
    def _pref_add(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        store = self.prefs
        if store is None:
            return {"ok": False, "message": "偏好服务未就绪"}
        ok, message = store.add(
            _text(payload, "text", "pref"),
            category=_text(payload, "category") or "style",
            scope=_text(payload, "scope") or "both",
        )
        self._prefs_changed()
        return {"ok": ok, "message": message, "refresh": True}

    def _pref_remove(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        store = self.prefs
        if store is None:
            return {"ok": False, "message": "偏好服务未就绪"}
        ok = store.remove(_text(payload, "id"))
        self._prefs_changed()
        return {
            "ok": ok,
            "message": "已删除偏好" if ok else "没有找到该偏好",
            "refresh": True,
        }

    def _pref_accept(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        store = self.prefs
        if store is None:
            return {"ok": False, "message": "偏好服务未就绪"}
        ok, message = store.accept(_text(payload, "id"))
        self._prefs_changed()
        return {"ok": ok, "message": message, "refresh": True}

    def _pref_reject(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        store = self.prefs
        if store is None:
            return {"ok": False, "message": "偏好服务未就绪"}
        ok = store.reject(_text(payload, "id"))
        self.refresh()
        return {
            "ok": ok,
            "message": "已忽略该建议" if ok else "没有找到该建议",
            "refresh": True,
        }

    def _pref_export(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        store = self.prefs
        if store is None:
            return {"ok": False, "message": "偏好服务未就绪"}
        try:
            path = store.export_defaults()
        except (OSError, ValueError) as exc:
            return {"ok": False, "message": f"导出失败：{exc}"}
        return {
            "ok": True,
            "message": f"已导出为默认偏好：{path.name}",
            "refresh": True,
        }

    def _pref_reset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        store = self.prefs
        if store is None:
            return {"ok": False, "message": "偏好服务未就绪"}
        count = store.reset_to_defaults()
        self._prefs_changed()
        return {"ok": True, "message": f"已恢复默认偏好（{count} 条）", "refresh": True}

    def _pref_clear(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        store = self.prefs
        if store is None:
            return {"ok": False, "message": "偏好服务未就绪"}
        count = store.clear()
        self._prefs_changed()
        return {"ok": True, "message": f"已清空 {count} 条偏好", "refresh": True}

    # ------------------------------------------------------------------
    # Notes tab toolbox panel
    # ------------------------------------------------------------------
    def _render_notes_panel(self) -> list[dict[str, Any]]:
        sid = ""
        try:
            from core.navigation import current_navigation

            sid = str(current_navigation().class_id or "")
        except Exception:  # noqa: BLE001
            sid = ""
        modes = self.modes.items() if self.modes is not None else []
        current = self._default_mode_id()
        versions = self._history_versions(sid) if sid and sid != "boot" else []
        blocks: list[dict[str, Any]] = [
            {
                "type": BlockType.MARKDOWN,
                "text": (
                    "### 生成与迭代\n\n"
                    f"共 {len(modes)} 个笔记模式；历史版本 {len(versions)} 个。"
                    "生成会使用设置页中当前所选模式的提示词；更新会先归档旧版。"
                ),
            },
            {
                "type": BlockType.FORM,
                "fields": [
                    _field(
                        "mode",
                        "笔记模式",
                        current,
                        type=FormInputType.SELECT,
                        input=FormInputType.SELECT,
                        options=[
                            {"value": item["id"], "label": item["name"]} for item in modes
                        ],
                    ),
                ],
                "submit": {"label": "按所选模式生成笔记", "action": "tutor.notes.generate"},
            },
            {
                "type": BlockType.BUTTONS,
                "buttons": [
                    {
                        "label": "根据问答更新笔记",
                        "action": "tutor.notes.iterate",
                        "class": "btn btn-sm btn-primary",
                    },
                    {
                        "label": "查看历史版本",
                        "action": "tutor.notes.history",
                        "class": "btn btn-sm",
                    },
                    {
                        "label": "查看问答记录",
                        "action": "tutor.notes.ask_log",
                        "class": "btn btn-sm",
                    },
                ],
            },
        ]
        return blocks

    # ------------------------------------------------------------------
    # Settings section
    # ------------------------------------------------------------------
    def _render(self) -> list[dict[str, Any]]:
        ctx = self.ctx
        if ctx is None or self.modes is None or self.prefs is None:
            return [{"type": BlockType.TEXT, "text": "偏好服务未就绪"}]
        modes = self.modes.items()
        default_mode = self._default_mode_id()
        prefs = self.prefs.items()
        pending = self.prefs.pending()
        sid = ""
        try:
            from core.navigation import current_navigation

            sid = str(current_navigation().class_id or "")
        except Exception:  # noqa: BLE001
            pass
        blocks: list[dict[str, Any]] = [
            {
                "type": BlockType.MARKDOWN,
                "text": (
                    "### 教学与偏好\n\n"
                    f"已记住 **{len(prefs)}** 条偏好，待确认建议 **{len(pending)}** 条；"
                    f"已启用 **{len(modes)}** 个笔记模式。\n\n"
                    "下面所有提示词都可以直接改写；改坏了点「恢复默认值」。"
                ),
            },
            {
                "type": BlockType.FORM,
                "fields": [
                    _field(
                        "mode",
                        "默认笔记模式",
                        default_mode,
                        type=FormInputType.SELECT,
                        input=FormInputType.SELECT,
                        options=[
                            {"value": item["id"], "label": item["name"]} for item in modes
                        ],
                    ),
                    _field(
                        "common_prompt",
                        "通用提示词（所有模式共用）",
                        self.common_prompt(),
                        type=FormInputType.TEXT,
                        input=FormInputType.TEXTAREA,
                    ),
                    _field(
                        "ask_prompt",
                        "提问提示词（每轮注入）",
                        self.ask_prompt(),
                        type=FormInputType.TEXT,
                        input=FormInputType.TEXTAREA,
                    ),
                    _field(
                        "writer_contract",
                        "笔记输出约定（三份文稿规则）",
                        self.writer_contract(),
                        type=FormInputType.TEXT,
                        input=FormInputType.TEXTAREA,
                    ),
                    _field(
                        "proposal_prompt",
                        "偏好工具说明（答题时注入）",
                        self.proposal_prompt(),
                        type=FormInputType.TEXT,
                        input=FormInputType.TEXTAREA,
                    ),
                ],
                "submit": {"label": "保存提示词", "action": "tutor.prompt.save"},
            },
            {
                "type": BlockType.BUTTONS,
                "buttons": [
                    {
                        "label": "通用提示词恢复默认",
                        "action": "tutor.prompt.restore",
                        "value": {"scope": "common"},
                        "class": "btn btn-sm",
                    },
                    {
                        "label": "提问提示词恢复默认",
                        "action": "tutor.prompt.restore",
                        "value": {"scope": "ask"},
                        "class": "btn btn-sm",
                    },
                    *(
                        [
                            {
                                "label": "ask 基础提示词恢复默认",
                                "action": "ask.restore_prompts",
                                "class": "btn btn-sm",
                            }
                        ]
                        if ctx.ui.has_action("ask.restore_prompts")
                        else []
                    ),
                    {
                        "label": "全部恢复默认",
                        "action": "tutor.prompt.restore",
                        "value": {"scope": "all"},
                        "class": "btn btn-sm btn-danger",
                    },
                ],
            },
        ]
        # --- note modes -------------------------------------------------
        blocks.append({"type": BlockType.MARKDOWN, "text": "#### 笔记模式"})
        for mode in modes:
            blocks.append(
                {
                    "type": BlockType.FORM,
                    "fields": [
                        _field("mode_id", "模式 ID", mode["id"], hidden=True),
                        _field("name", "名称", mode["name"]),
                        _field("description", "说明", mode["description"]),
                        _field(
                            "prompt",
                            "提示词",
                            mode["prompt"],
                            type=FormInputType.TEXT,
                            input=FormInputType.TEXTAREA,
                        ),
                    ],
                    "submit": {"label": "保存此模式", "action": "tutor.mode.save"},
                }
            )
            blocks.append(
                {
                    "type": BlockType.BUTTONS,
                    "buttons": [
                        {
                            "label": "删除此模式",
                            "action": "tutor.mode.delete",
                            "value": {"mode_id": mode["id"]},
                            "class": "btn btn-sm btn-danger",
                        }
                    ],
                }
            )
        blocks.append(
            {
                "type": BlockType.FORM,
                "fields": [
                    _field("name", "新模式名称", ""),
                    _field("description", "说明（可选）", ""),
                    _field(
                        "prompt",
                        "新模式提示词",
                        "",
                        type=FormInputType.TEXT,
                        input=FormInputType.TEXTAREA,
                    ),
                ],
                "submit": {"label": "添加模式", "action": "tutor.mode.add"},
            }
        )
        # --- preferences ------------------------------------------------
        blocks.append({"type": BlockType.MARKDOWN, "text": "#### 学生偏好"})
        if prefs:
            rows = [
                {
                    "id": item.id,
                    "text": item.text,
                    "category": item.label(),
                    "scope": item.scope_label(),
                    "source": {
                        "manual": "手动",
                        "ask": "助教建议",
                        "default": "默认",
                    }.get(item.source, item.source),
                }
                for item in prefs
            ]
            blocks.append(
                {
                    "type": BlockType.TABLE,
                    "columns": [
                        {"key": "text", "label": "偏好"},
                        {"key": "category", "label": "分类"},
                        {"key": "scope", "label": "范围"},
                        {"key": "source", "label": "来源"},
                        {"key": "id", "label": "编号"},
                    ],
                    "rows": rows,
                    "row_actions": [
                        {
                            "label": "删除",
                            "action": "tutor.pref.remove",
                            "keys": ["id"],
                            "class": "btn btn-sm",
                        }
                    ],
                }
            )
        else:
            blocks.append({"type": BlockType.MARKDOWN, "text": "（暂无偏好）"})
        if pending:
            blocks.append({"type": BlockType.MARKDOWN, "text": "#### 待确认的偏好建议"})
            for proposal in pending:
                lines = [
                    f"- {op.get('op')}: {op.get('text') or op.get('id')}"
                    for op in proposal.operations
                ]
                reason = f"\n\n> 理由：{proposal.reason}" if proposal.reason else ""
                blocks.append(
                    {
                        "type": BlockType.MARKDOWN,
                        "text": f"**建议 #{proposal.id}**\n" + "\n".join(lines) + reason,
                    }
                )
                blocks.append(
                    {
                        "type": BlockType.BUTTONS,
                        "buttons": [
                            {
                                "label": "接受",
                                "action": "tutor.pref.accept",
                                "value": {"id": proposal.id},
                                "class": "btn btn-sm btn-primary",
                            },
                            {
                                "label": "拒绝",
                                "action": "tutor.pref.reject",
                                "value": {"id": proposal.id},
                                "class": "btn btn-sm",
                            },
                        ],
                    }
                )
        blocks.append(
            {
                "type": BlockType.FORM,
                "fields": [
                    _field(
                        "text",
                        "新偏好",
                        "",
                        placeholder=f"例如：先给结论再展开（≤{MAX_TEXT} 字）",
                    ),
                    _field(
                        "category",
                        "分类",
                        "style",
                        type=FormInputType.SELECT,
                        input=FormInputType.SELECT,
                        options=[
                            {"value": key, "label": CATEGORY_LABELS[key]}
                            for key in CATEGORIES
                        ],
                    ),
                    _field(
                        "scope",
                        "作用范围",
                        "both",
                        type=FormInputType.SELECT,
                        input=FormInputType.SELECT,
                        options=[
                            {"value": key, "label": SCOPE_LABELS[key]} for key in SCOPES
                        ],
                    ),
                ],
                "submit": {"label": "添加偏好", "action": "tutor.pref.add"},
            }
        )
        blocks.append(
            {
                "type": BlockType.BUTTONS,
                "buttons": [
                    {"label": "导出为默认", "action": "tutor.pref.export", "class": "btn btn-sm"},
                    {"label": "恢复默认偏好", "action": "tutor.pref.reset", "class": "btn btn-sm"},
                    {"label": "清空偏好", "action": "tutor.pref.clear", "class": "btn btn-sm btn-danger"},
                ],
            }
        )
        return blocks


__all__ = ["TutorPlugin", "TutorService"]
