"""Interactive teaching plugin: plan file + guided conversation + review."""

from __future__ import annotations

import difflib
import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from core.agent_extensions import AgentExtension, AgentExtensionTool
from core.agent_types import AgentEventType, AgentToolName
from core.navigation import current_navigation
from core.plugin import Plugin, PluginContext
from core.settings_spec import SettingsSpec, SettingType
from core.ui import BlockType, FormInputType, Side, render_markdown

from .prompts import ASK_INSTRUCTION, PLAN_INSTRUCTION, REVIEW_PROMPT, STYLE_RULES, SYSTEM_PROMPT, TEACH_INSTRUCTION

LOG = logging.getLogger("lecture_agent.plugin.teaching")
PAGE_ID = "teaching:main"
TAB_ID = "teaching-page"
SETTINGS_ITEM = "teaching.settings"
PAGE_ITEM = "teaching.page"
TOOLS = [AgentToolName.READ, AgentToolName.GREP, AgentToolName.GLOB, AgentToolName.EDIT, AgentToolName.WRITE]


def _values(payload):
    nested = payload.get("values")
    return nested if isinstance(nested, Mapping) else payload


def _text(payload, *names):
    values = _values(payload)
    for name in names:
        if values.get(name) is not None:
            return str(values.get(name)).strip()
    return ""


class TeachingPlugin(Plugin):
    id = "teaching"
    requires = ("settings", "agent", "reader")
    optional_requires = ("notes", "conversation", "preference")
    provides = ("teaching",)

    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self.settings = ctx.require("settings")
        self.agent = ctx.require("agent")
        self.reader = ctx.require("reader")
        self._delete_pending: set[str] = set()
        ctx.llm.register_role(self.id, "teaching", "交互式教学")
        ctx.llm.register_role(self.id, "teaching_review", "教案审阅")
        self.settings.register(
            SettingsSpec(
                key="teaching.materials", label="教学材料", type=SettingType.SELECT, default="both",
                options=(("notes", "笔记"), ("transcript", "原始转写"), ("both", "都读入")),
                group="交互式教学", description="读入笔记、课程原始转写，或都读入。",
            ),
            plugin_id=self.id,
        )
        self.settings.register(
            SettingsSpec(key="teaching.ask_learning_state", label="允许教案 agent 主动问学情", type=SettingType.BOOL, default=True, group="交互式教学", hidden=True),
            plugin_id=self.id,
        )
        self.settings.register(
            SettingsSpec(key="teaching.include_previous_notes", label="读取这门课之前的笔记", type=SettingType.BOOL, default=False, group="交互式教学", hidden=True),
            plugin_id=self.id,
        )
        for key, label, default in (
            ("teaching.system_prompt", "教学系统提示词", SYSTEM_PROMPT),
            ("teaching.plan_prompt", "教案生成提示词", PLAN_INSTRUCTION),
            ("teaching.teach_prompt", "教学对话提示词", TEACH_INSTRUCTION),
            ("teaching.review_prompt", "课代表审阅提示词", REVIEW_PROMPT),
        ):
            self.settings.register(
                SettingsSpec(key=key, label=label, type=SettingType.TEXT, default=default, group="交互式教学", hidden=True),
                plugin_id=self.id,
            )
        self.settings.register(
            SettingsSpec(
                key="teaching.review_rounds", label="教案审阅轮数", type=SettingType.INT, default=1,
                group="交互式教学", description="(审阅-重写) 轮数；0 表示不审阅。",
            ),
            plugin_id=self.id,
        )
        agent_service = ctx.service("agent")
        if (
            agent_service is not None
            and getattr(agent_service, "supports_callbacks", lambda: False)()
        ):
            try:
                agent_service.register_extension(
                    self.id,
                    AgentExtension(
                        id="teaching-tools",
                        roles=("teaching",),
                        tools=(
                            AgentExtensionTool(
                                name="plan_message_reviewer",
                                label="给课代表留言",
                                description=(
                                    "给教案审阅的课代表留一条说明（例如你为什么不改某处）。"
                                    "留言会在下一次审阅时提供给它。"
                                ),
                                parameters={
                                    "type": "object",
                                    "properties": {"message": {"type": "string", "minLength": 1}},
                                    "required": ["message"],
                                    "additionalProperties": False,
                                },
                                handler=self._reviewer_message,
                            ),
                        ),
                    ),
                )
            except Exception:  # noqa: BLE001
                LOG.warning("teaching tools registration failed", exc_info=True)
        for action, handler in (
            ("teaching.open", self._open),
            ("teaching.start", self._start),
            ("teaching.send", self._send),
            ("teaching.open_plan", self._open_plan),
            ("teaching.delete_plan", self._delete_plan),
            ("teaching.save_settings", self._save_settings),
        ):
            ctx.ui.register_action(action, handler, plugin_id=self.id)
        conversation_service = ctx.service("conversation")
        if conversation_service is not None:
            conversation_service.register_provider("teaching", self._conversation_send, None, self._conversation_composer)
        ctx.ui.add_page(PAGE_ID, "交互式教学", render=self._render_page, order=8, plugin_id=self.id)
        ctx.ui.add_tab(Side.RIGHT, TAB_ID, "教学", order=21, plugin_id=self.id)
        ctx.ui.add_toolbox_item(
            TAB_ID, PAGE_ITEM, "", render=self._render_entry, order=0, plugin_id=self.id
        )
        ctx.ui.add_toolbox_item(
            TAB_ID, SETTINGS_ITEM, "", render=self._render_settings, order=1, plugin_id=self.id
        )

    def teardown(self) -> None:
        ctx = getattr(self, "ctx", None)
        if ctx is None:
            return
        self._delete_pending.clear()
        try:
            ctx.service("agent").unregister_extensions(self.id)
        except Exception:  # noqa: BLE001
            pass
        try:
            service = ctx.service("conversation")
            if service is not None:
                service.unregister_provider("teaching")
        except Exception:  # noqa: BLE001
            pass
        for action in ("open", "start", "send", "open_plan", "delete_plan", "save_settings"):
            try:
                ctx.ui.unregister_action(f"teaching.{action}")
            except Exception:  # noqa: BLE001
                pass
        for tab in (TAB_ID, "teaching"):
            try:
                ctx.ui.remove_tab(tab)
            except Exception:  # noqa: BLE001
                pass
        try:
            ctx.ui.remove_page(PAGE_ID)
        except Exception:  # noqa: BLE001
            pass

    # -- paths/state ---------------------------------------------------
    def _sid(self) -> str:
        try:
            sid = str(current_navigation().class_id or "")
        except Exception:  # noqa: BLE001
            sid = ""
        return "" if sid == "boot" else sid

    def _dir(self, sid: str) -> Path:
        return self.ctx.data_dir / "sessions" / sid / "teaching"

    def _state(self, sid: str) -> dict[str, Any]:
        path = self._dir(sid) / "state.json"
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_state(self, sid: str, state: dict[str, Any]) -> None:
        directory = self._dir(sid)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", "utf-8")

    def _append(self, sid: str, role: str, text: str) -> None:
        directory = self._dir(sid)
        directory.mkdir(parents=True, exist_ok=True)
        entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "role": role, "text": text}
        with (directory / "conversation.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _conversation(self, sid: str) -> list[dict[str, str]]:
        path = self._dir(sid) / "conversation.jsonl"
        rows = []
        if path.exists():
            for line in path.read_text("utf-8", errors="replace").splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def _prompt(self, key: str, default: str) -> str:
        return str(self.settings.get(key, default) or default)

    def _course_name(self, sid: str) -> str:
        try:
            raw = (self.ctx.data_dir / "sessions" / sid / "meta.json").read_text("utf-8")
            meta = json.loads(raw).get("meta") or {}
        except (OSError, json.JSONDecodeError):
            return ""
        return str(meta.get("course_name") or meta.get("course") or "")

    def _shared_system_prompt(self, sid: str, base: str) -> str:
        return base + "\n\n" + STYLE_RULES

    def _ensure_session(self, sid: str) -> str:
        """Return a live teacher session id, reopening the saved conversation."""

        state = self._state(sid)
        session_id = str(state.get("session_id") or "")
        if session_id and self.agent.get_session(session_id) is not None:
            return session_id
        spec = state.get("spec") or {}
        if not session_id or not spec:
            return ""
        workspace = __import__("core.workspace", fromlist=["FileWorkspace"]).FileWorkspace(
            {name: Path(path) for name, path in (spec.get("files") or {}).items()},
            readonly=spec.get("readonly") or [],
        )
        try:
            self.agent.reopen(
                session_id,
                name=str(spec.get("name") or "交互式教学"),
                workspace=workspace,
                system_prompt=str(spec.get("system_prompt") or self._prompt("teaching.system_prompt", SYSTEM_PROMPT)),
                role="teaching",
                tools=list(spec.get("tools") or [str(tool) for tool in TOOLS]),
                meta={},
            )
            return session_id
        except Exception as exc:  # noqa: BLE001
            # Runtime transcripts are not guaranteed to survive a restart.
            # Keep teaching usable by starting a fresh session with a compact
            # replay of the recent conversation instead of dead-ending.
            try:
                history = self._conversation(sid)[-12:]
                replay = "\n".join(
                    f"{'学生' if item.get('role') == 'user' else '助教'}: {item.get('text', '')}"
                    for item in history
                )
                fresh = self.agent.create_session(
                    name=str(spec.get("name") or "交互式教学"),
                    role="teaching",
                    system_prompt=(
                        str(spec.get("system_prompt") or self._prompt("teaching.system_prompt", SYSTEM_PROMPT))
                        + ("\n\n## 重启前的对话（用于衔接）\n" + replay if replay else "")
                    ),
                    tools=list(spec.get("tools") or [str(tool) for tool in TOOLS]),
                    workspace=workspace,
                )
                state = self._state(sid)
                state["session_id"] = fresh.id
                self._save_state(sid, state)
                self._append(sid, "system", f"原会话无法恢复，已新建教学会话继续（原因：{exc}）")
                return fresh.id
            except Exception as second:  # noqa: BLE001
                self._append(sid, "system", f"教学会话恢复失败：{second}")
                return ""

    def _materials(self, sid: str) -> dict[str, Path]:
        source = self.ctx.data_dir / "sessions" / sid
        mode = str(self.settings.get("teaching.materials", "both") or "both")
        files: dict[str, Path] = {}
        if mode in ("notes", "both"):
            notes = source / "notes"
            if notes.is_dir():
                for path in sorted(notes.glob("note_*.md")):
                    files[f"notes_{path.name}"] = path
        if mode in ("transcript", "both"):
            transcript = source / "lecture_source.md"
            if transcript.exists():
                files["lecture_source.md"] = transcript
        if bool(self.settings.get("teaching.include_previous_notes", False)):
            notes_service = self.ctx.service("notes")
            previous = getattr(notes_service, "previous_notes", None)
            if callable(previous):
                for name, path in previous(sid).items():
                    files[name] = path
        return files

    # -- actions -------------------------------------------------------
    def _conversation_service(self):
        return self.ctx.service("conversation") if self.ctx is not None else None

    def _open(self, payload):
        sid = self._sid()
        service = self._conversation_service()
        conversation_id = str(self._state(sid).get("conversation_id") or "") if sid else ""
        if service is not None and conversation_id and service.get(conversation_id) is not None:
            service.open(conversation_id)
            return {"ok": True, "open_page": f"conversation:{conversation_id}"}
        self.ctx.ui.open_page(PAGE_ID)
        return {"ok": True, "open_page": PAGE_ID}

    def _save_settings(self, payload):
        values = _values(payload)
        if values.get("materials"):
            self.settings.set("teaching.materials", str(values.get("materials")))
        if "ask_learning_state" in values:
            self.settings.set("teaching.ask_learning_state", bool(values.get("ask_learning_state")))
        if "include_previous_notes" in values:
            self.settings.set("teaching.include_previous_notes", bool(values.get("include_previous_notes")))
        if values.get("review_rounds") is not None:
            try:
                self.settings.set("teaching.review_rounds", int(values.get("review_rounds")))
            except (TypeError, ValueError):
                return {"ok": False, "message": "审阅轮数需要整数"}
        for key in ("system_prompt", "plan_prompt", "teach_prompt", "review_prompt"):
            text = str(values.get(key) or "").strip()
            if text:
                self.settings.set(f"teaching.{key}", text)
        return {"ok": True, "message": "已保存教学设置", "refresh": True}

    def _start(self, payload):
        sid = self._sid()
        if not sid:
            return {"ok": False, "message": "请先选择课次"}
        state = self._state(sid)
        if state.get("session_id") and self.agent.get_session(state["session_id"]) is not None:
            return {"ok": False, "message": "这节课已经在教学了"}
        directory = self._dir(sid)
        directory.mkdir(parents=True, exist_ok=True)
        plan_path = directory / "plan.md"
        if not plan_path.exists():
            plan_path.write_text("# 教案\n\n", "utf-8")
        materials = self._materials(sid)
        if not materials:
            return {"ok": False, "message": "没有可用的教学材料；请先生成笔记或检查转写"}
        system = self._prompt("teaching.system_prompt", SYSTEM_PROMPT)
        system = self._shared_system_prompt(sid, system)
        if not bool(self.settings.get("teaching.ask_learning_state", True)):
            system += "\n\n不要主动向学生询问学情；无法确定的内容按默认方式处理，不要因此打断教学。"
        files_spec = {name: str(path) for name, path in materials.items()}
        files_spec["plan.md"] = str(plan_path)
        spec = {
            "name": f"{sid} 交互式教学",
            "system_prompt": system,
            "tools": [str(tool) for tool in TOOLS],
            "files": files_spec,
            "readonly": [name for name in materials],
            }
        state = {"phase": "preparing", "review_round": 0, "spec": spec}
        service = self._conversation_service()
        if service is not None:
            meta = {}
            try:
                raw = (self.ctx.data_dir / "sessions" / sid / "meta.json").read_text("utf-8")
                meta = (json.loads(raw).get("meta") or {})
            except (OSError, json.JSONDecodeError):
                meta = {}
            course = str(meta.get("course_name") or meta.get("course_id") or "未归属课程")
            session_title = str(meta.get("class_label") or meta.get("title") or sid)
            conversation_id = service.create(
                "teaching",
                f"{session_title} 交互式教学",
                {"session_id": sid, "session_title": session_title, "course": course, "path": str(plan_path)},
            )
            state["conversation_id"] = conversation_id
        self._save_state(sid, state)
        threading.Thread(target=self._prepare, args=(sid, materials, plan_path, system), name=f"teaching-{sid}", daemon=True).start()
        return {"ok": True, "refresh": True, "message": "正在准备教案…"}

    def _conversation_send(self, conversation_id: str, text: str, context: Mapping[str, Any]) -> None:
        sid = str(context.get("session_id") or self._sid())
        self._append(sid, "user", text)
        session_id = self._ensure_session(sid)
        if not session_id:
            service = self._conversation_service()
            if service is not None:
                service.append(conversation_id, "system", "教学会话已过期，请重新开始")
            return
        threading.Thread(target=self._conversation_turn, args=(sid, conversation_id, session_id, text), daemon=True).start()

    def _run_agent(self, sid: str, session_id: str, prompt: str):
        conversation_id = str(self._state(sid).get("conversation_id") or "")
        service = self._conversation_service()
        seen: set[str] = set()

        def on_event(event):
            if getattr(event, "type", None) != AgentEventType.EXECUTION:
                return
            block = (getattr(event, "data", None) or {}).get("block") or {}
            if str(block.get("kind") or "") != "tool":
                return
            if "completed" not in str(block.get("status") or ""):
                return
            block_id = str(block.get("id") or "")
            if not block_id or block_id in seen:
                return
            seen.add(block_id)
            name = str(block.get("name") or block.get("title") or "tool")
            line = f"调用工具 `{name}`"
            self._append(sid, "system", line)
            if service is not None and conversation_id:
                service.append(conversation_id, "system", line)

        return self.agent.run(session_id, prompt, on_event=on_event)

    def _conversation_turn(self, sid: str, conversation_id: str, session_id: str, text: str):
        try:
            result = self._run_agent(sid, session_id, text)
            answer = result.final_message or "（没有回答）"
        except Exception as exc:  # noqa: BLE001
            answer = f"（回答失败：{exc}）"
        state = self._state(sid)
        plan_path = self._dir(sid) / "plan.md"
        if state.get("phase") == "asking" and self._plan_ready(plan_path):
            spec = state.get("spec") or {}
            files = spec.get("files") or {}
            materials = {
                name: Path(path) for name, path in files.items() if name != "plan.md"
            }
            system = str(spec.get("system_prompt") or self._prompt("teaching.system_prompt", SYSTEM_PROMPT))
            self._finish_after_plan(sid, session_id, materials, plan_path, system)
            return
        self._append(sid, "assistant", answer)
        service = self._conversation_service()
        if service is not None:
            service.append(conversation_id, "assistant", answer)

    def _conversation_composer(self, conversation_id: str, context: Mapping[str, Any]):
        hints = ["给出第一步", "给出大体思路", "给出第一个关键观察", "指出需要观察什么对象", "联想相似知识点", "给出关键结论", "给一个代表性特殊情况", "选择最适合的提示", "直接给解答"]
        return [{"label": hint, "text": hint} for hint in hints]

    def _prepare(self, sid: str, materials: dict[str, Path], plan_path: Path, system: str):
        workspace_files = dict(materials)
        workspace_files["plan.md"] = plan_path
        readonly = [name for name in workspace_files if name != "plan.md"]
        try:
            session = self.agent.create_session(
                name=f"{sid} 交互式教学",
                role="teaching",
                system_prompt=system,
                tools=TOOLS,
                workspace=__import__("core.workspace", fromlist=["FileWorkspace"]).FileWorkspace(workspace_files, readonly=readonly),
            )
            self._save_state(sid, {**self._state(sid), "session_id": session.id, "phase": "preparing"})
            if bool(self.settings.get("teaching.ask_learning_state", True)):
                result = self._run_agent(sid, session.id, ASK_INSTRUCTION)
                if self._plan_ready(plan_path):
                    # The plan was produced in this turn; do not append a second
                    # "first teaching message".
                    self._finish_after_plan(sid, session.id, materials, plan_path, system)
                else:
                    answer = result.final_message or "在开始前，我想先确认一下你的学情。"
                    self._append(sid, "assistant", answer)
                    service = self._conversation_service()
                    conversation_id = str(self._state(sid).get("conversation_id") or "")
                    if service is not None and conversation_id:
                        service.append(conversation_id, "assistant", answer)
                    self._save_state(sid, {**self._state(sid), "phase": "asking"})
            else:
                self._run_agent(sid, session.id, self._prompt("teaching.plan_prompt", PLAN_INSTRUCTION))
                self._append(sid, "system", "教案已生成（默认不展示，可点“查看教案”）。")
                self._finish_after_plan(sid, session.id, materials, plan_path, system)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("teaching prepare failed")
            self._append(sid, "system", f"准备失败：{exc}")
            self._save_state(sid, {**self._state(sid), "phase": "failed"})
            service = self._conversation_service()
            conversation_id = str(self._state(sid).get("conversation_id") or "")
            conversation = service.get(conversation_id) if service is not None else None
            if conversation is not None and not conversation.messages():
                service.delete(conversation_id)
        self.ctx.ui.refresh("teaching")

    def _sid_for_teaching_session(self, session_id: str) -> str:
        root = self.ctx.data_dir / "sessions"
        if not root.is_dir():
            return ""
        for path in root.glob("*/teaching/state.json"):
            try:
                state = json.loads(path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(state.get("session_id") or "") == session_id:
                return path.parent.parent.name
        return ""

    def _reviewer_message(self, context):
        sid = self._sid_for_teaching_session(str(context.get("session_id") or ""))
        message = str((context.get("parameters") or {}).get("message") or "").strip()
        if not sid or not message:
            return {"content": [{"type": "text", "text": "没有可用的审阅会话"}], "isError": True, "details": {}}
        path = self._dir(sid) / "reviewer_messages.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "message": message}, ensure_ascii=False) + "\n")
        return {"content": [{"type": "text", "text": "已给课代表留言。"}], "details": {}}

    def _archive_plan(self, sid: str, plan_path: Path) -> None:
        if not plan_path.exists():
            return
        target = self._dir(sid) / "plan_history"
        target.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        try:
            (target / f"{stamp}.md").write_text(plan_path.read_text("utf-8"), "utf-8")
        except OSError:
            pass

    @staticmethod
    def _plan_ready(plan_path: Path) -> bool:
        try:
            text = plan_path.read_text("utf-8")
        except OSError:
            return False
        body = text.replace("# 教案", "", 1).strip()
        return bool(body)

    def _finish_after_plan(self, sid: str, session_id: str, materials: dict[str, Path], plan_path: Path, system: str):
        service = self._conversation_service()
        self._review(sid, materials, plan_path, system, session_id)
        result = self._run_agent(sid, session_id, self._prompt("teaching.teach_prompt", TEACH_INSTRUCTION))
        first = result.final_message or "我们开始吧。"
        self._append(sid, "assistant", first)
        conversation_id = str(self._state(sid).get("conversation_id") or "")
        if service is not None and conversation_id:
            service.append(conversation_id, "assistant", first)
        self._save_state(sid, {**self._state(sid), "phase": "teaching"})

    def _review(self, sid: str, materials: dict[str, Path], plan_path: Path, system: str, teacher_id: str):
        rounds = max(0, int(self.settings.get("teaching.review_rounds", 1) or 0))
        if rounds == 0:
            return
        workspace_files = dict(materials)
        workspace_files["plan.md"] = plan_path
        readonly = list(workspace_files)
        reviewer = self.agent.create_session(
            name=f"{sid} 教案审阅",
            role="teaching_review",
            system_prompt=self._shared_system_prompt(sid, self._prompt("teaching.review_prompt", REVIEW_PROMPT)),
            tools=[AgentToolName.READ, AgentToolName.GREP, AgentToolName.GLOB],
            workspace=__import__("core.workspace", fromlist=["FileWorkspace"]).FileWorkspace(workspace_files, readonly=readonly),
        )
        try:
            messages_path = self._dir(sid) / "reviewer_messages.jsonl"
            notes = ""
            if messages_path.exists():
                notes = "\n\n教学 agent 给你的留言：\n" + messages_path.read_text("utf-8", errors="replace")
            try:
                last_plan = plan_path.read_text("utf-8")
            except OSError:
                last_plan = ""
            for index in range(rounds):
                diff = ""
                try:
                    current_plan = plan_path.read_text("utf-8")
                except OSError:
                    current_plan = ""
                if last_plan and current_plan and current_plan != last_plan:
                    diff = "\n\n教案相对上次审阅的 diff：\n```diff\n" + "".join(
                        difflib.unified_diff(
                            last_plan.splitlines(keepends=True),
                            current_plan.splitlines(keepends=True),
                            fromfile="上次",
                            tofile="本次",
                            n=2,
                        )
                    ) + "```"
                    last_plan = current_plan
                result = self.agent.run(
                    reviewer.id,
                    "请读取 plan.md，按课代表视角给出建议；没有建议就输出“审阅通过”。" + notes + diff,
                )
                text = (result.final_message or "").strip()
                self._append(sid, "system", f"课代表审阅第 {index + 1} 轮：{text[:500]}")
                if not text or "审阅通过" in text:
                    break
                self.agent.run(
                    teacher_id,
                    "课代表对教案提出了以下建议，请用 edit 只修改有问题的地方（不要整份重写），然后停止：\n\n" + text,
                )
                self._archive_plan(sid, plan_path)
                self._save_state(sid, {**self._state(sid), "review_round": index + 1})
        finally:
            self.agent.close(reviewer.id)

    def _send(self, payload):
        sid = self._sid()
        message = _text(payload, "message", "text")
        if not sid or not message:
            return {"ok": False, "message": "请输入内容"}
        session_id = self._ensure_session(sid)
        if not session_id:
            return {"ok": False, "message": "教学会话无法恢复，请重新开始"}
        self._append(sid, "user", message)
        threading.Thread(target=self._run_turn, args=(sid, session_id, message), name=f"teaching-send-{sid}", daemon=True).start()
        return {"ok": True, "refresh": True, "message": "已发送"}

    def _run_turn(self, sid: str, session_id: str, message: str):
        try:
            result = self._run_agent(sid, session_id, message)
            self._append(sid, "assistant", result.final_message or "（没有回答）")
        except Exception as exc:  # noqa: BLE001
            self._append(sid, "system", f"回答失败：{exc}")
        self.ctx.ui.refresh("teaching")

    def _delete_plan(self, payload):
        sid = self._sid()
        if not sid:
            return {"ok": False, "message": "请先选择课次"}
        if sid not in self._delete_pending:
            self._delete_pending.add(sid)
            self.ctx.ui.refresh("teaching.delete_plan")
            return {"ok": True, "refresh": True, "message": "请再次点击「确认删除教案」完成删除"}
        self._delete_pending.discard(sid)
        state = self._state(sid)
        session_id = str(state.get("session_id") or "")
        if session_id:
            try:
                self.agent.close(session_id)
            except Exception:  # noqa: BLE001
                pass
        service = self._conversation_service()
        conversation_id = str(state.get("conversation_id") or "")
        if service is not None and conversation_id:
            try:
                service.delete(conversation_id)
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(self._dir(sid), ignore_errors=True)
        self.ctx.ui.refresh("teaching.delete_plan")
        return {"ok": True, "refresh": True, "message": "已删除教案、历史与相关对话"}

    def _open_plan(self, payload):
        sid = self._sid()
        if not sid:
            return {"ok": False, "message": "请先选择课次"}
        plan = self._dir(sid) / "plan.md"
        if not plan.exists():
            return {"ok": False, "message": "还没有教案"}
        page = self.reader.show(
            title="教案",
            docs=[{"key": "plan", "label": "教案", "path": str(plan)}],
            page_key=f"{sid}:teaching-plan",
            owner=self.id,
        )
        return {"ok": True, "open_page": page}

    # -- UI ------------------------------------------------------------
    def _render_page(self):
        sid = self._sid()
        if not sid:
            return [{"type": BlockType.TEXT, "text": "请先选择课次。"}]
        state = self._state(sid)
        blocks: list[dict[str, Any]] = [
            {"type": BlockType.MARKDOWN, "text": f"### 交互式教学\n\n状态：{state.get('phase', '未开始')}"}
        ]
        service = self._conversation_service()
        conversation_id = str(state.get("conversation_id") or "")
        if service is not None and conversation_id and service.get(conversation_id) is not None:
            return service.render(conversation_id)
        if not state.get("session_id"):
            blocks.append({"type": BlockType.BUTTONS, "buttons": [
                {"label": "开始教学", "action": "teaching.start", "class": "btn-solid theme-primary"}
            ]})
            return blocks
        blocks.append({"type": BlockType.BUTTONS, "buttons": [
            {"label": "查看教案", "action": "teaching.open_plan", "class": "btn btn-sm"}
        ]})
        bubbles = []
        for entry in self._conversation(sid):
            role = {"user": "你", "assistant": "助教", "system": "系统"}.get(entry.get("role"), entry.get("role"))
            bubbles.append(
                f'<div class="conv-row conv-{entry.get("role")}"><div class="conv-role">{role}</div>'
                f'<div class="conv-bubble">{render_markdown(str(entry.get("text", "")))}</div></div>'
            )
        if bubbles:
            blocks.append({"type": BlockType.HTML, "html": '<div class="markdown prose conv-messages">' + "".join(bubbles) + "</div>"})
        blocks.append({
            "type": BlockType.FORM,
            "fields": [{"name": "message", "key": "message", "label": "你的回答/问题", "type": FormInputType.TEXT, "input": FormInputType.TEXTAREA, "value": ""}],
            "submit": {"label": "发送", "action": "teaching.send"},
        })
        hints = ["给出第一步", "给出大体思路", "给出第一个关键观察", "指出需要观察什么对象", "联想相似知识点", "给出关键结论", "给一个代表性特殊情况", "选择最适合的提示", "直接给解答"]
        blocks.append({"type": BlockType.HTML, "html": "".join(
            f'<button type="button" class="btn btn-sm" data-fill-text="{hint}">{i+1}. {hint}</button> '
            for i, hint in enumerate(hints)
        )})
        return blocks

    def _render_entry(self):
        return [
            {
                "type": BlockType.BUTTONS,
                "buttons": [
                    {"label": "生成教案", "action": "teaching.start", "class": "btn-solid theme-primary"},
                    {"label": "查看教案", "action": "teaching.open_plan", "class": "btn btn-sm"},
                    {"label": "确认删除教案" if self._sid() in self._delete_pending else "删除教案", "action": "teaching.delete_plan", "class": "btn btn-sm theme-danger"},
                    {"label": "进入对话", "action": "teaching.open", "class": "btn btn-sm"},
                ],
            }
        ]

    def _render_settings(self):
        materials = str(self.settings.get("teaching.materials", "both") or "both")
        rounds = int(self.settings.get("teaching.review_rounds", 1) or 0)
        return [{
            "type": BlockType.FORM,
            "fields": [
                {"name": "materials", "key": "materials", "label": "教学材料", "type": FormInputType.SELECT, "input": FormInputType.SELECT, "value": materials,
                 "options": [{"value": "notes", "label": "笔记"}, {"value": "transcript", "label": "原始转写"}, {"value": "both", "label": "都读入"}]},
                {"name": "review_rounds", "key": "review_rounds", "label": "教案审阅轮数", "type": FormInputType.NUMBER, "input": FormInputType.NUMBER, "value": rounds},
                {"name": "ask_learning_state", "key": "ask_learning_state", "label": "允许教案 agent 主动问学情", "type": FormInputType.CHECKBOX, "input": FormInputType.CHECKBOX, "value": bool(self.settings.get("teaching.ask_learning_state", True))},
                {"name": "include_previous_notes", "key": "include_previous_notes", "label": "读取这门课之前的笔记", "type": FormInputType.CHECKBOX, "input": FormInputType.CHECKBOX, "value": bool(self.settings.get("teaching.include_previous_notes", False))},
                {"name": "system_prompt", "key": "system_prompt", "label": "教学系统提示词", "type": FormInputType.TEXT, "input": FormInputType.TEXTAREA, "value": self._prompt("teaching.system_prompt", SYSTEM_PROMPT)},
                {"name": "plan_prompt", "key": "plan_prompt", "label": "教案生成提示词", "type": FormInputType.TEXT, "input": FormInputType.TEXTAREA, "value": self._prompt("teaching.plan_prompt", PLAN_INSTRUCTION)},
                {"name": "teach_prompt", "key": "teach_prompt", "label": "教学对话提示词", "type": FormInputType.TEXT, "input": FormInputType.TEXTAREA, "value": self._prompt("teaching.teach_prompt", TEACH_INSTRUCTION)},
                {"name": "review_prompt", "key": "review_prompt", "label": "课代表审阅提示词", "type": FormInputType.TEXT, "input": FormInputType.TEXTAREA, "value": self._prompt("teaching.review_prompt", REVIEW_PROMPT)},
            ],
            "submit": {"label": "保存教学设置", "action": "teaching.save_settings"},
        }]


__all__ = ["TeachingPlugin"]
