"""Shared preference memory: manual editing + propose-only agent tool."""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Any, Mapping

from core.agent_extensions import AgentExtension, AgentExtensionTool
from core.plugin import Plugin, PluginContext
from core.settings_spec import SettingsSpec, SettingType
from core.ui import BlockType, FormInputType, Side

from .store import CATEGORIES, CATEGORY_LABELS, MAX_TEXT, SCOPES, SCOPE_LABELS, PreferenceStore

LOG = logging.getLogger("lecture_agent.plugin.preference")
TAB_ID = "preference"
ITEM_ID = "preference.panel"


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


def _field(name, label, value="", *, type=FormInputType.STRING, input=FormInputType.TEXT, **options):
    return {"name": name, "key": name, "label": label, "type": type, "input": input, "value": value, **options}


class PreferenceService:
    __slots__ = ("_plugin",)

    def __init__(self, plugin: "PreferencePlugin") -> None:
        self._plugin = plugin

    def block(self, scopes=("both", "ask", "notes")) -> str:
        return self._plugin.store.block(scopes)

    def snapshot(self):
        return self._plugin.store.snapshot()


class PreferencePlugin(Plugin):
    id = "preference"
    requires = ("settings", "agent")
    provides = ("preference",)
    action_namespace = "preference"

    def __init__(self) -> None:
        self.ctx: PluginContext | None = None
        self.agent: Any = None
        self.store: PreferenceStore | None = None

    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self.agent = ctx.require("agent")
        self.settings = ctx.require("settings")
        self.store = PreferenceStore(
            ctx.data_dir / "preference" / "preferences.json",
            defaults_path=Path(__file__).with_name("defaults.json"),
        )
        self.store.load()
        for key, label in (
            ("preference.inject_note_ask", "向 note / ask 注入偏好"),
            ("preference.tool_note_ask", "允许 note / ask 调用偏好工具"),
        ):
            self.settings.register(
                SettingsSpec(
                    key=key,
                    label=label,
                    type=SettingType.BOOL,
                    default=False,
                    group="偏好",
                    hidden=True,
                ),
                plugin_id=self.id,
            )
        ctx.register_service("preference", PreferenceService(self))
        self.agent.register_prompt_provider(self.id, self._prompt_context)
        for action, handler in (
            ("preference.add", self._add),
            ("preference.remove", self._remove),
            ("preference.accept", self._accept),
            ("preference.reject", self._reject),
            ("preference.export", self._export),
            ("preference.reset", self._reset),
            ("preference.clear", self._clear),
            ("preference.save_flags", self._save_flags),
        ):
            ctx.ui.register_action(action, handler, plugin_id=self.id)
        ctx.ui.add_tab(Side.RIGHT, TAB_ID, "偏好", order=16, plugin_id=self.id)
        ctx.ui.add_toolbox_item(
            TAB_ID, ITEM_ID, "", render=self._render, order=0, plugin_id=self.id
        )
        self._refresh_extension()

    def teardown(self) -> None:
        ctx = self.ctx
        if ctx is None:
            return
        try:
            self.agent.unregister_extensions(self.id)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.agent.unregister_prompt_provider(self.id)
        except Exception:  # noqa: BLE001
            pass
        for action in ("add", "remove", "accept", "reject", "export", "reset", "clear", "save_flags"):
            try:
                ctx.ui.unregister_action(f"preference.{action}")
            except Exception:  # noqa: BLE001
                pass
        try:
            ctx.ui.remove_tab(TAB_ID)
        except Exception:  # noqa: BLE001
            pass
        self.ctx = None

    # -- tool ---------------------------------------------------------
    def _tool(self) -> AgentExtensionTool:
        return AgentExtensionTool(
            name="propose_preference",
            label="提议修改学生偏好",
            description=(
                "向学生提议新增/修改/删除长期偏好。只能提议，不能直接修改；"
                "学生需要在右侧偏好页确认后才生效。\n\n"
                "什么时候调用：\n"
                "- 学生明确表达跨对话稳定的偏好、习惯、长期目标，例如「以后先给结论」"
                "「我不喜欢太多计算」「我习惯用 ε-δ 语言」；\n"
                "- 学生指出已有偏好不适用，需要 update/remove；\n"
                "- 学生对你的讲解表达了明确不满。\n"
                "不要用于一次性要求、临时指令、敏感信息；不确定是否长期时先不提议。\n\n"
                "operations 规则：\n"
                "- add：必须提供 text；category 默认 style，scope 默认 both；\n"
                "- update：必须提供 id；text/category/scope 至少改一个；\n"
                "- remove：必须提供 id。\n"
                "id 必须来自上下文里「学生长期偏好」列表的 #id；不要编造 id。\n"
                "一次最多 3 条；不要重复添加已有偏好。\n\n"
                "category 取值：style 表达风格、depth 讲解深度、content 内容偏好、"
                "format 格式偏好、dislike 不喜欢、goal 学习目标。\n"
                "scope 取值：both 问答+笔记、ask 仅问答、notes 仅笔记。\n\n"
                "调用后只告诉学生「已提交，等待确认」，不要声称已经记住或已经生效，"
                "因为学生可能会拒绝添加此条偏好。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "op": {"type": "string", "enum": ["add", "update", "remove"]},
                                "text": {"type": "string"},
                                "category": {"type": "string", "enum": list(CATEGORIES)},
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
            },
            handler=self._propose,
        )

    def _propose(self, context: Mapping[str, Any]) -> dict[str, Any]:
        store = self.store
        params = context.get("parameters") or {}
        operations = params.get("operations") if isinstance(params, Mapping) else None
        proposal = store.propose(operations or [], reason=str(params.get("reason") or "")) if store else None
        if proposal is None:
            return {
                "content": [{"type": "text", "text": "没有可记录的偏好（提议无效或已达上限）。"}],
                "isError": True,
                "details": {},
            }
        self.refresh()
        return {
            "content": [{"type": "text", "text": f"已提交 {len(proposal.operations)} 条偏好建议，等待学生确认。"}],
            "details": {"proposal_id": proposal.id},
        }

    @staticmethod
    def _note_or_ask(role: str) -> bool:
        return role in {"ask", "note"} or role.startswith("ask_") or role.startswith("note_")

    def _prompt_context(self, role: str, meta: Mapping[str, Any]) -> str:
        if self.store is None:
            return ""
        if self._note_or_ask(role) and not bool(
            self.settings.get("preference.inject_note_ask", False)
        ):
            return ""
        return self.store.block(("both", "ask", "notes", "teaching"))

    def _tool_roles(self) -> tuple[str, ...]:
        allow = bool(self.settings.get("preference.tool_note_ask", False))
        roles = [
            str(item.get("id"))
            for item in self.agent.list_roles()
            if allow or not self._note_or_ask(str(item.get("id") or ""))
        ]
        return tuple(roles) or ("default",)

    def _refresh_extension(self) -> None:
        try:
            self.agent.unregister_extensions(self.id)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.agent.register_extension(
                self.id,
                AgentExtension(
                    id="preferences",
                    roles=self._tool_roles(),
                    tools=(self._tool(),),
                ),
            )
        except Exception:  # noqa: BLE001
            LOG.warning("preference extension registration failed", exc_info=True)

    def refresh(self) -> None:
        if self.ctx is not None:
            self._refresh_extension()
            try:
                self.ctx.ui.refresh("preference")
            except Exception:  # noqa: BLE001
                pass

    # -- actions ------------------------------------------------------
    def _add(self, payload):
        ok, message = self.store.add(
            _text(payload, "text"), category=_text(payload, "category") or "style", scope=_text(payload, "scope") or "both"
        )
        self.refresh()
        return {"ok": ok, "message": message, "refresh": True}

    def _remove(self, payload):
        ok = self.store.remove(_text(payload, "id"))
        self.refresh()
        return {"ok": ok, "message": "已删除偏好" if ok else "没有找到该偏好", "refresh": True}

    def _accept(self, payload):
        ok, message = self.store.accept(_text(payload, "id"))
        self._refresh_extension()
        self.refresh()
        return {"ok": ok, "message": message, "refresh": True}

    def _reject(self, payload):
        ok = self.store.reject(_text(payload, "id"))
        self.refresh()
        return {"ok": ok, "message": "已忽略建议" if ok else "没有找到建议", "refresh": True}

    def _export(self, payload):
        try:
            path = self.store.export_defaults()
        except (OSError, ValueError) as exc:
            return {"ok": False, "message": f"导出失败：{exc}"}
        return {"ok": True, "message": f"已导出：{path.name}", "refresh": True}

    def _reset(self, payload):
        count = self.store.reset_to_defaults()
        self._refresh_extension()
        self.refresh()
        return {"ok": True, "message": f"已恢复默认（{count} 条）", "refresh": True}

    def _save_flags(self, payload):
        values = _values(payload)
        self.settings.set("preference.inject_note_ask", bool(values.get("inject_note_ask")))
        self.settings.set("preference.tool_note_ask", bool(values.get("tool_note_ask")))
        self._refresh_extension()
        self.refresh()
        return {"ok": True, "refresh": True, "message": "已保存偏好开关"}

    def _clear(self, payload):
        count = self.store.clear()
        self._refresh_extension()
        self.refresh()
        return {"ok": True, "message": f"已清空 {count} 条偏好", "refresh": True}

    # -- UI -----------------------------------------------------------
    def _render(self) -> list[dict[str, Any]]:
        store = self.store
        if store is None:
            return [{"type": BlockType.TEXT, "text": "偏好服务未就绪"}]
        prefs = store.items()
        pending = store.pending()
        blocks: list[dict[str, Any]] = [
            {
                "type": BlockType.MARKDOWN,
                "text": f"### 学生偏好\n\n已确认 **{len(prefs)}** 条，待确认建议 **{len(pending)}** 条。",
            }
        ]
        if pending:
            blocks.append({"type": BlockType.MARKDOWN, "text": "#### 待处理请求"})
            for proposal in pending:
                lines = [f"- {op.get('op')}: {op.get('text') or op.get('id')}" for op in proposal.operations]
                blocks.append({"type": BlockType.MARKDOWN, "text": f"**建议 #{proposal.id}**\n" + "\n".join(lines)})
                blocks.append(
                    {
                        "type": BlockType.BUTTONS,
                        "buttons": [
                            {"label": "接受", "action": "preference.accept", "value": {"id": proposal.id}, "class": "btn btn-sm btn-primary"},
                            {"label": "拒绝", "action": "preference.reject", "value": {"id": proposal.id}, "class": "btn btn-sm"},
                        ],
                    }
                )
        if prefs:
            rows = []
            for item in prefs:
                payload = html.escape(json.dumps({"id": item.id}), quote=True)
                rows.append(
                    '<div class="pref-item">'
                    f'<div class="pref-text">{html.escape(item.text)}</div>'
                    f'<div class="pref-meta">{html.escape(CATEGORY_LABELS.get(item.category, item.category))}'
                    f' · {html.escape(SCOPE_LABELS.get(item.scope, item.scope))}'
                    f' · {html.escape({"manual": "手动", "ask": "助教建议", "default": "默认"}.get(item.source, item.source))}'
                    f' · #{item.id}</div>'
                    f'<button type="button" class="pref-remove" data-action="preference.remove" data-value="{payload}" title="删除">删除</button>'
                    "</div>"
                )
            blocks.append({"type": BlockType.HTML, "html": '<div class="pref-list">' + "".join(rows) + "</div>"})
        blocks.append(
            {
                "type": BlockType.FORM,
                "fields": [
                    _field("text", "新偏好", "", placeholder=f"例如：先给结论再展开（≤{MAX_TEXT} 字）"),
                    _field("category", "分类", "style", type=FormInputType.SELECT, input=FormInputType.SELECT, options=[{"value": k, "label": CATEGORY_LABELS[k]} for k in CATEGORIES]),
                    _field("scope", "范围", "both", type=FormInputType.SELECT, input=FormInputType.SELECT, options=[{"value": k, "label": SCOPE_LABELS[k]} for k in SCOPES]),
                ],
                "submit": {"label": "添加偏好", "action": "preference.add"},
            }
        )
        blocks.append(
            {
                "type": BlockType.FORM,
                "fields": [
                    {"name": "inject_note_ask", "key": "inject_note_ask", "label": "向 note / ask 注入偏好", "type": FormInputType.CHECKBOX, "input": FormInputType.CHECKBOX, "value": bool(self.settings.get("preference.inject_note_ask", False))},
                    {"name": "tool_note_ask", "key": "tool_note_ask", "label": "允许 note / ask 调用偏好工具", "type": FormInputType.CHECKBOX, "input": FormInputType.CHECKBOX, "value": bool(self.settings.get("preference.tool_note_ask", False))},
                ],
                "submit": {"label": "保存开关", "action": "preference.save_flags"},
            }
        )
        blocks.append(
            {
                "type": BlockType.BUTTONS,
                "buttons": [
                    {"label": "导出为默认", "action": "preference.export", "class": "btn btn-sm"},
                    {"label": "恢复默认", "action": "preference.reset", "class": "btn btn-sm"},
                    {"label": "清空", "action": "preference.clear", "class": "btn btn-sm btn-danger"},
                ],
            }
        )
        return blocks


__all__ = ["PreferencePlugin", "PreferenceService"]
