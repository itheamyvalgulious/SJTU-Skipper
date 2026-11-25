"""Learning-state memory: knowledge aliases + a five-level mastery value."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from core.agent_extensions import AgentExtension, AgentExtensionTool
from core.plugin import Plugin, PluginContext
from core.ui import BlockType, FormInputType

from .store import MAX_KEYWORDS, VALUES, VALUE_LABELS, LearningStore, value_label

LOG = logging.getLogger("lecture_agent.plugin.learning")
READ_ONLY_ROLES = {"note_chunk", "note_writer", "note_review", "teaching_review"}


def _values(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("values")
    return nested if isinstance(nested, Mapping) else payload


def _text(payload: Mapping[str, Any], *names: str) -> str:
    values = _values(payload)
    for name in names:
        value = values.get(name)
        if value is not None:
            return str(value).strip()
    return ""


class LearningService:
    __slots__ = ("_plugin",)

    def __init__(self, plugin: "LearningPlugin") -> None:
        self._plugin = plugin

    def search(self, keywords, *, limit=20):
        return self._plugin.store.search(keywords, limit=limit)

    def upsert(self, keywords, value, *, mode="insert", entry_id=None):
        return self._plugin.store.upsert(keywords, value, mode=mode, entry_id=entry_id)


class LearningPlugin(Plugin):
    id = "learning"
    requires = ("settings", "agent")
    provides = ("learning",)
    action_namespace = "learning"
    optional_requires = ("conversation", "teaching")

    def __init__(self) -> None:
        self.ctx: PluginContext | None = None
        self.agent: Any = None
        self.store: LearningStore | None = None
        self._query = ""
        self._admin_sessions: dict[str, str] = {}
        self._pending: dict[str, dict[str, Any]] = {}
        self._pending_lock = threading.RLock()

    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self.agent = ctx.require("agent")
        self.settings = ctx.require("settings")
        self.store = LearningStore(ctx.data_dir / "learning" / "learning.db")
        self._load_pending()
        ctx.register_service("learning", LearningService(self))
        for action, handler in (
            ("learning.remove", self._remove),
            ("learning.update_value", self._update_value),
            ("learning.search", self._search_action),
            ("learning.admin_open", self._admin_open),
            ("learning.close_admin", self._close_admin),
            ("learning.answer_choice", self._answer_choice),
        ):
            ctx.ui.register_action(action, handler, plugin_id=self.id)
        conversation_service = ctx.service("conversation")
        if conversation_service is not None:
            conversation_service.register_provider("learning-admin", self._admin_send, self._admin_extra)
        self.settings.register_section("learning", "学情记忆", self._render_settings)
        self._register()

    def teardown(self) -> None:
        ctx = self.ctx
        if ctx is None:
            return
        try:
            self.agent.unregister_extensions(self.id)
        except Exception:  # noqa: BLE001
            pass
        for action in ("remove", "update_value", "search", "admin_open", "close_admin", "answer_choice"):
            try:
                ctx.ui.unregister_action(f"learning.{action}")
            except Exception:  # noqa: BLE001
                pass
        try:
            self.settings.unregister_section("learning")
        except Exception:  # noqa: BLE001
            pass
        try:
            service = ctx.service("conversation")
            if service is not None:
                service.unregister_provider("learning-admin")
        except Exception:  # noqa: BLE001
            pass
        for session_id in list(self._admin_sessions.values()):
            try:
                self.agent.close(session_id)
            except Exception:  # noqa: BLE001
                pass
        self._admin_sessions.clear()
        self.ctx = None

    # -- tools ---------------------------------------------------------
    def _register(self) -> None:
        tools = (
            AgentExtensionTool(
                name="learning_search",
                label="查询学情记忆",
                description=(
                    "查询方法学情。学情只记录“方法/技能”，不记录纯概念。\n"
                    "keywords：传 1–5 个部分关键词，不要传完整长句。例如查仿射簇相关方法时只传"
                    "“坐标环”；搜索会做子串匹配，返回所有名称里含“坐标环”的方法；"
                    "可以一次传多个相关词（OR），避免漏掉相关方法。\n"
                    "返回 entries：[{id, keywords, value, label, updated_at}]。value 五级："
                    "unknown=不了解；impression=有大概印象；understand=肯定能看懂；"
                    "proficient=比较熟练；preferred=喜欢用这个。\n"
                    "entries 为空表示“不知道学生对这个方法会不会”，不等于“学生不会”；"
                    "不要因此跳过内容。\n"
                    "如果 search 结果存在就相信；只有出现“结果说没学过，但学生说自己会”"
                    "这类明显过时的迹象时，才重新问学生。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "keywords": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": ["keywords"],
                    "additionalProperties": False,
                },
                handler=self._search_tool,
            ),
            AgentExtensionTool(
                name="learning_upsert",
                label="写入学情记忆",
                description=(
                    "记录/更新学生对某个“方法”的掌握程度。学情只记录方法，不记录纯概念。\n"
                    "insert（mode=insert）：新建一条学情。keywords 第一个放完整方法名，用动词短语，"
                    "例如“计算仿射簇的坐标环”“判断多项式映射是否是簇同构”；不要写“仿射簇”“域论”"
                    "这类纯概念。搜索时用短关键词即可子串命中完整方法名，所以不要把方法名里的每个字"
                    "都拆成额外关键词。\n"
                    "replace（mode=replace）：只更新已有条目的 value，必须提供 id——"
                    "id 来自 learning_search 返回结果。replace 不允许用 keywords 匹配，也不能改方法名；"
                    "没有 id 时不要 replace，先 search。\n"
                    "value 五级：\n"
                    "- unknown：没学过/完全不了解；\n"
                    "- impression：大概知道思路、流程或用途，但细节不清楚；\n"
                    "- understand：看到别人使用它的时候能够完全看懂，但自己可能无法主动想到要用；\n"
                    "- proficient：能够熟练使用；\n"
                    "- preferred：能够熟练使用，并且相比其它方法，更倾向于或习惯使用这个方法。\n"
                    "只在学生明确表达“没学过/有印象/能看懂/熟练/喜欢用”时调用；"
                    "不要从答错、要求举例、听不懂某一步推断。\n"
                    "给用户讲解之后，如果用户表示“懂了”，最多记为 understand（能理解），"
                    "不能记 proficient 或 preferred；学生说“先跳过/以后再说”时不要更新。\n"
                    "不要写“上节课没听”这类课堂事件或时效性强的内容。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "keywords": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "value": {"type": "string", "enum": list(VALUES)},
                        "mode": {"type": "string", "enum": ["insert", "replace"]},
                        "id": {"type": "string", "description": "replace 时必填：learning_search 返回的条目 id"},
                    },
                    "required": ["value"],
                    "additionalProperties": False,
                },
                handler=self._upsert_tool,
            ),
        )
        try:
            self.agent.register_extension(
                self.id,
                AgentExtension(id="learning-memory", roles=("*",), tools=tools),
            )
        except Exception:  # noqa: BLE001
            LOG.warning("learning extension registration failed", exc_info=True)
        # The choice-card tool only makes sense for the teaching conversation.
        try:
            roles = {str(item.get("id")) for item in self.agent.list_roles()}
            if "teaching" in roles and self.ctx is not None and self.ctx.service("conversation") is not None:
                self.agent.register_extension(
                    self.id,
                    AgentExtension(
                        id="learning-ask",
                        roles=("teaching",),
                        tools=(self._ask_tool(),),
                    ),
                )
        except Exception:  # noqa: BLE001
            LOG.warning("learning ask tool registration failed", exc_info=True)

    def _ask_tool(self) -> AgentExtensionTool:
        return AgentExtensionTool(
            name="learning_ask",
            label="向学生发学情选择题",
            description=(
                "当你不确定学生对某个“方法”的掌握程度、且它会影响接下来怎么讲时，"
                "向学生发送选择题。一次只问一个方法；学生点击选项后会直接写入学情记忆，"
                "不需要你再调用 learning_upsert。\n"
                "keywords 的第一个必须是完整方法名，不能是“仿射簇”“域论”这类纯概念。"
                "原因：概念没有可执行的掌握程度，只有“会不会做某个方法”才有教学意义。"
                "如果 learning_search 已有该方法，把它的 id 一并传入，选择题答案会更新原条目；"
                "没有 id 就按新方法 insert。display_name 是给学生看的简短方法名。\n"
                "学生可选：A. 不了解；B. 有大概印象；C. 肯定能看懂；D. 比较熟练；E. 喜欢用这个。\n"
                "调用后等待学生选择，不要猜测答案，也不要继续追问同一方法。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": MAX_KEYWORDS},
                    "display_name": {"type": "string"},
                    "id": {"type": "string", "description": "已有方法的 learning_search id；新方法可省略"},
                },
                "required": ["keywords"],
                "additionalProperties": False,
            },
            handler=self._ask,
        )

    def _search_tool(self, context: Mapping[str, Any]):
        params = context.get("parameters") or {}
        rows = self.store.search(params.get("keywords") or [], limit=int(params.get("limit") or 20))
        return {
            "content": [{"type": "text", "text": json.dumps(rows, ensure_ascii=False)}],
            "details": {"rows": rows},
        }

    def _upsert_tool(self, context: Mapping[str, Any]):
        role = str(context.get("role") or "")
        if role in READ_ONLY_ROLES:
            return {
                "content": [{"type": "text", "text": "这个角色只能查询学情，不能写入。"}],
                "isError": True,
                "details": {},
            }
        params = context.get("parameters") or {}
        result = self.store.upsert(
            params.get("keywords") or [],
            str(params.get("value") or ""),
            mode=str(params.get("mode") or "insert"),
            entry_id=params.get("id"),
        )
        self.refresh()
        return {
            "content": [{"type": "text", "text": str(result.get("message") or "已更新")}],
            "isError": not bool(result.get("ok")),
            "details": result,
        }

    def _ask(self, context: Mapping[str, Any]):
        params = context.get("parameters") or {}
        session_id = str(context.get("session_id") or "")
        sid, conversation_id = self._teaching_context(session_id)
        if not sid or not conversation_id:
            return {
                "content": [{"type": "text", "text": "当前没有可用的教学对话，无法发送选择题。"}],
                "isError": True,
                "details": {},
            }
        keywords = [str(item).strip() for item in (params.get("keywords") or []) if str(item).strip()][:MAX_KEYWORDS]
        if not keywords:
            return {"content": [{"type": "text", "text": "需要至少一个知识点关键词。"}], "isError": True, "details": {}}
        display_name = str(params.get("display_name") or keywords[0]).strip()
        entry_id = str(params.get("id") or "").strip()
        question_id = uuid.uuid4().hex[:10]
        pending = {
            "question_id": question_id,
            "session_id": session_id,
            "teaching_session": sid,
            "conversation_id": conversation_id,
            "keywords": keywords,
            "display_name": display_name,
            "entry_id": entry_id,
            "created_at": time.time(),
        }
        with self._pending_lock:
            self._pending[question_id] = pending
            self._save_pending()
        service = self.ctx.service("conversation") if self.ctx is not None else None
        if service is None:
            return {"content": [{"type": "text", "text": "conversation 插件未启用。"}], "isError": True, "details": {}}
        service.append_choice(
            conversation_id,
            question_id,
            f"关于「{display_name}」，你目前属于哪一种？",
            [
                {"value": value, "label": f"{letter}. {VALUE_LABELS[value]}"}
                for letter, value in zip("ABCDE", VALUES)
            ],
        )
        self.refresh()
        return {
            "content": [{"type": "text", "text": "已向学生发送选择题。等待学生选择，不要继续猜测或重复提问。"}],
            "details": {"question_id": question_id},
        }

    # -- choice answers -------------------------------------------------
    def _answer_choice(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        values = _values(payload)
        question_id = str(values.get("question_id") or "")
        value = str(values.get("value") or "")
        with self._pending_lock:
            pending = self._pending.pop(question_id, None)
            if pending is not None:
                self._save_pending()
        if pending is None:
            return {"ok": False, "message": "这道选择题已失效或已回答"}
        if value not in VALUES:
            return {"ok": False, "message": "未知的掌握程度"}
        entry_id = str(pending.get("entry_id") or "")
        result = self.store.upsert(
            pending["keywords"],
            value,
            mode="replace" if entry_id else "insert",
            entry_id=entry_id or None,
        )
        if not result.get("ok"):
            return {"ok": False, "message": str(result.get("message") or "写入失败")}
        service = self.ctx.service("conversation") if self.ctx is not None else None
        label = f"{VALUE_LABELS[value]}"
        answer = f"[选择题] {pending['display_name']}：{label}"
        if service is not None:
            service.append(pending["conversation_id"], "user", answer)
        threading.Thread(
            target=self._resume_after_choice,
            args=(pending, value),
            name=f"learning-choice-{question_id}",
            daemon=True,
        ).start()
        return {"ok": True, "refresh": True, "message": f"已记录为「{label}」"}

    def _resume_after_choice(self, pending: Mapping[str, Any], value: str) -> None:
        session_id = str(pending.get("session_id") or "")
        conversation_id = str(pending.get("conversation_id") or "")
        service = self.ctx.service("conversation") if self.ctx is not None else None
        if not session_id or self.agent.get_session(session_id) is None:
            if service is not None:
                service.append(conversation_id, "system", "该学情已记录；当前教学会话已结束。")
            return
        label = VALUE_LABELS.get(value, value)
        prompt = (
            f"学生对「{pending.get('display_name')}」的选择是：{label}（{value}）。"
            "该学情已直接写入学情记忆，不要重复调用 learning_upsert。"
            "请根据这个回答继续教学。"
        )
        try:
            result = self.agent.run(session_id, prompt)
            if service is not None:
                service.append(conversation_id, "assistant", result.final_message or "（没有回答）")
        except Exception as exc:  # noqa: BLE001
            if service is not None:
                service.append(conversation_id, "system", f"选择题已记录，但继续教学失败：{exc}")

    # -- pending persistence -------------------------------------------
    def _pending_path(self) -> Path:
        return self.ctx.data_dir / "learning" / "pending.json"

    def _load_pending(self) -> None:
        try:
            raw = json.loads(self._pending_path().read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(raw, dict):
            self._pending = {str(key): value for key, value in raw.items() if isinstance(value, dict)}

    def _save_pending(self) -> None:
        if self.ctx is None:
            return
        path = self._pending_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(self._pending, ensure_ascii=False, indent=2) + "\n", "utf-8")
        temp.replace(path)

    def _teaching_context(self, session_id: str) -> tuple[str, str]:
        root = self.ctx.data_dir / "sessions" if self.ctx is not None else None
        if root is None or not root.is_dir():
            return "", ""
        for path in root.glob("*/teaching/state.json"):
            try:
                state = json.loads(path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(state.get("session_id") or "") == session_id:
                return path.parent.parent.name, str(state.get("conversation_id") or "")
        return "", ""

    # -- UI actions -----------------------------------------------------
    def _remove(self, payload):
        ok = self.store.remove(int(_text(payload, "id") or 0))
        self.refresh()
        return {"ok": ok, "message": "已删除" if ok else "没有找到条目", "refresh": True}

    def _update_value(self, payload):
        values = _values(payload)
        entry_id = int(values.get("id") or 0)
        value = str(values.get("value") or "")
        if not self.store.update_value(entry_id, value):
            return {"ok": False, "message": "没有找到条目或状态无效"}
        self.refresh()
        return {"ok": True, "message": "状态已更新", "refresh": True}

    def _search_action(self, payload):
        self._query = _text(payload, "query")
        self.refresh()
        return {"ok": True, "refresh": True}

    def refresh(self) -> None:
        if self.ctx is not None:
            self.ctx.ui.refresh("learning")

    # -- admin conversation -------------------------------------------
    def _admin_open(self, payload):
        service = self.ctx.service("conversation") if self.ctx is not None else None
        if service is None:
            return {"ok": False, "message": "conversation 插件未启用"}
        conversation_id = service.create(
            "learning-admin",
            "修改学情记忆",
            {"session_id": "", "session_title": "学情记忆"},
            transient=True,
        )
        return {"ok": True, "open_page": f"conversation:{conversation_id}", "refresh": True}

    def _admin_extra(self, conversation_id: str, context: Mapping[str, Any]):
        return [{
            "type": BlockType.BUTTONS,
            "buttons": [{
                "label": "结束修改对话",
                "action": "learning.close_admin",
                "value": {"conversation_id": conversation_id},
                "class": "btn btn-sm",
            }],
        }]

    def _close_admin(self, payload):
        conversation_id = str(_values(payload).get("conversation_id") or "")
        session_id = self._admin_sessions.pop(conversation_id, "")
        if session_id:
            try:
                self.agent.close(session_id)
            except Exception:  # noqa: BLE001
                pass
        service = self.ctx.service("conversation") if self.ctx is not None else None
        if service is not None:
            service.delete(conversation_id)
        return {"ok": True, "refresh": True, "message": "已结束修改对话"}

    def _admin_send(self, conversation_id: str, text: str, context: Mapping[str, Any]) -> None:
        service = self.ctx.service("conversation")
        if service is None:
            return
        session_id = self._admin_sessions.get(conversation_id, "")
        if not session_id:
            try:
                session = self.agent.create_session(
                    name="学情修改对话",
                    role="default",
                    system_prompt=(
                        "你是学情记忆管理员。用户会描述要修改的学情；先调用 learning_search "
                        "找到条目，再用 learning_upsert 修改 value（keywords 由你按 tool "
                        "描述维护，不要凭空改无关条目）。value 只能是 unknown/impression/"
                        "understand/proficient/preferred。每次只做用户要求的修改，完成后简短说明。"
                    ),
                    tools=[],
                )
                session_id = session.id
                self._admin_sessions[conversation_id] = session_id
            except Exception as exc:  # noqa: BLE001
                service.append(conversation_id, "system", f"无法创建对话：{exc}")
                return
        try:
            result = self.agent.run(session_id, text)
            service.append(conversation_id, "assistant", result.final_message or "（没有回答）")
        except Exception as exc:  # noqa: BLE001
            service.append(conversation_id, "system", f"修改失败：{exc}")
        self.refresh()

    # -- settings page --------------------------------------------------
    def _render_settings(self):
        store = self.store
        if store is None:
            return [{"type": BlockType.TEXT, "text": "学情服务未就绪"}]
        entries = [
            entry
            for entry in store.items()
            if not self._query
            or self._query.lower() in (entry["label"] + " " + " ".join(entry["keywords"])).lower()
        ]
        blocks: list[dict[str, Any]] = [
            {
                "type": BlockType.MARKDOWN,
                "text": (
                    f"### 学情记忆\n\n共 **{len(entries)}** 条。agent 自动写入；"
                    "用户只能搜索/删除/改掌握程度，keywords 由 agent 维护。"
                ),
            },
            {
                "type": BlockType.FORM,
                "fields": [{"name": "query", "key": "query", "label": "搜索（状态或关键词）", "type": FormInputType.STRING, "input": FormInputType.TEXT, "value": self._query}],
                "submit": {"label": "搜索", "action": "learning.search"},
            },
        ]
        options = [{"value": value, "label": VALUE_LABELS[value]} for value in VALUES]
        for entry in entries:
            blocks.append({"type": BlockType.MARKDOWN, "text": f"**#{entry['id']}** 关键词：{'、'.join(entry['keywords'])}"})
            blocks.append({
                "type": BlockType.FORM,
                "fields": [
                    {"name": "id", "key": "id", "label": "", "type": FormInputType.STRING, "input": FormInputType.TEXT, "value": entry["id"], "hidden": True},
                    {"name": "value", "key": "value", "label": "掌握程度", "type": FormInputType.SELECT, "input": FormInputType.SELECT, "value": entry["value"], "options": options},
                ],
                "submit": {"label": "保存状态", "action": "learning.update_value"},
            })
            blocks.append({"type": BlockType.BUTTONS, "buttons": [{"label": "删除", "action": "learning.remove", "value": {"id": entry["id"]}, "class": "btn btn-sm btn-danger"}]})
        if self.ctx.service("conversation") is not None:
            blocks.append({"type": BlockType.BUTTONS, "buttons": [{"label": "用对话修改学情", "action": "learning.admin_open", "class": "btn-solid theme-primary"}]})
        return blocks


__all__ = ["LearningPlugin", "LearningService"]
