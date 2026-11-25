"""Generic chat UI: one page per conversation; providers own behavior.

Conversations are persisted under ``data/agent/conversations/<id>/`` and
restored when the plugin loads.  A provider is registered by kind (ask,
teaching, ...) so restored conversations can be sent to after a restart.
"""

from __future__ import annotations

import html
import json
import logging
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from core.plugin import Plugin, PluginContext
from core.ui import BlockType, FormInputType, Side, render_markdown

LOG = logging.getLogger("lecture_agent.plugin.conversation")
LIST_TAB = "conversations"
PAGE_PREFIX = "conversation:"

SendHandler = Callable[[str, str, Mapping[str, Any]], None]
ExtraHandler = Callable[[str, Mapping[str, Any]], list[dict[str, Any]]]


def _values(payload):
    nested = payload.get("values")
    return nested if isinstance(nested, Mapping) else payload


class Conversation:
    def __init__(self, conversation_id, kind, title, context, path, *, transient=False):
        self.id = conversation_id
        self.kind = kind
        self.title = title
        self.context = dict(context)
        self.path = Path(path)
        self.transient = bool(transient)

    @property
    def directory(self) -> Path:
        return self.path.parent

    def messages(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self.path.exists():
            return rows
        try:
            lines = self.path.read_text("utf-8", errors="replace").splitlines()
        except OSError:
            return rows
        for line in lines:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows


class ConversationService:
    __slots__ = ("_plugin",)

    def __init__(self, plugin: "ConversationPlugin") -> None:
        self._plugin = plugin

    def register_provider(self, kind: str, send: SendHandler, extra: ExtraHandler | None = None, composer=None) -> None:
        self._plugin.register_provider(kind, send, extra, composer)

    def unregister_provider(self, kind: str, send: SendHandler | None = None) -> None:
        self._plugin.unregister_provider(kind)

    def create(self, kind, title, context, *, transient=False) -> str:
        return self._plugin.create(kind, title, context, transient=transient)

    def delete(self, conversation_id) -> bool:
        return self._plugin.delete(conversation_id)

    def append(self, conversation_id, role, text) -> None:
        self._plugin.append(conversation_id, role, text)

    def append_choice(self, conversation_id, question_id, prompt, choices) -> None:
        self._plugin.append_choice(conversation_id, question_id, prompt, choices)

    def get(self, conversation_id):
        return self._plugin.get(conversation_id)

    def for_class(self, class_id):
        return self._plugin.for_class(class_id)

    def open(self, conversation_id) -> bool:
        return self._plugin.open(conversation_id)

    def preset_html(self, presets) -> str:
        return self._plugin.preset_html(presets)

    def update_context(self, conversation_id, updates) -> None:
        self._plugin.update_context(conversation_id, updates)

    def render(self, conversation_id):
        return self._plugin._render(str(conversation_id))


class ConversationPlugin(Plugin):
    id = "conversation"
    requires = ("settings", "reader")
    optional_requires = ("input",)
    provides = ("conversation",)
    action_namespace = "conversation"

    def __init__(self):
        self.ctx: PluginContext | None = None
        self._items: dict[str, Conversation] = {}
        self._providers: dict[str, tuple[SendHandler, ExtraHandler | None, Any]] = {}
        self._pages: set[str] = set()
        self._lock = threading.RLock()

    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        ctx.register_service("conversation", ConversationService(self))
        for action, handler in (
            ("conversation.send", self._send),
            ("conversation.open", self._open_action),
            ("conversation.delete", self._delete_action),
        ):
            ctx.ui.register_action(action, handler, plugin_id=self.id)
        ctx.ui.add_tab(Side.LEFT, LIST_TAB, "对话", order=10, plugin_id=self.id)
        ctx.ui.add_toolbox_item(
            LIST_TAB, "conversation.list", "", render=self._render_list, order=0, plugin_id=self.id
        )
        input_service = ctx.service("input")
        if input_service is not None:
            input_service.register_class_children(self.id, self._tree_children)
        self._load_existing()

    def teardown(self) -> None:
        ctx = self.ctx
        if ctx is None:
            return
        for action in ("send", "open", "delete"):
            try:
                ctx.ui.unregister_action(f"conversation.{action}")
            except Exception:  # noqa: BLE001
                pass
        try:
            ctx.ui.remove_tab(LIST_TAB)
        except Exception:  # noqa: BLE001
            pass
        input_service = ctx.service("input")
        if input_service is not None:
            input_service.unregister_class_children(self.id)
        with self._lock:
            pages = list(self._pages)
            self._pages.clear()
            self._items.clear()
            self._providers.clear()
        for page in pages:
            try:
                ctx.ui.remove_page(page)
            except Exception:  # noqa: BLE001
                pass
        self.ctx = None

    # -- providers -----------------------------------------------------
    def register_provider(self, kind: str, send: SendHandler, extra: ExtraHandler | None = None, composer=None) -> None:
        with self._lock:
            if kind in self._providers:
                raise ValueError(f"Conversation provider is already registered: {kind}")
            self._providers[str(kind)] = (send, extra, composer)

    def unregister_provider(self, kind: str) -> None:
        with self._lock:
            self._providers.pop(str(kind), None)

    def _provider(self, kind: str):
        with self._lock:
            return self._providers.get(str(kind))

    # -- persistence ---------------------------------------------------
    def _root(self) -> Path:
        return self.ctx.data_dir / "conversations"

    def _load_existing(self) -> None:
        root = self._root()
        if not root.is_dir():
            return
        for directory in sorted(root.iterdir()):
            if not directory.is_dir():
                continue
            meta_path = directory / "meta.json"
            try:
                meta = json.loads(meta_path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                LOG.warning("ignoring unreadable conversation metadata: %s", meta_path)
                continue
            conversation_id = str(meta.get("id") or directory.name)
            if not conversation_id or conversation_id != directory.name:
                LOG.warning("ignoring conversation with mismatched id: %s", directory)
                continue
            kind = str(meta.get("kind") or "")
            if not kind:
                continue
            conversation = Conversation(
                conversation_id,
                kind,
                str(meta.get("title") or conversation_id),
                meta.get("context") or {},
                directory / "messages.jsonl",
                transient=bool(meta.get("transient")),
            )
            with self._lock:
                self._items[conversation_id] = conversation
            self._add_page(conversation)

    def _add_page(self, conversation: Conversation) -> None:
        page = PAGE_PREFIX + conversation.id
        try:
            if self.ctx.ui.has_page(page):
                return
            self.ctx.ui.add_page(
                page,
                conversation.title,
                render=lambda cid=conversation.id: self._render(cid),
                order=9,
                plugin_id=self.id,
            )
            with self._lock:
                self._pages.add(page)
        except Exception:  # noqa: BLE001
            LOG.debug("cannot register conversation page %s", page, exc_info=True)

    def create(self, kind, title, context, *, transient=False) -> str:
        conversation_id = uuid.uuid4().hex[:12]
        directory = self._root() / conversation_id
        directory.mkdir(parents=True, exist_ok=True)
        conversation = Conversation(
            conversation_id, str(kind), str(title), context, directory / "messages.jsonl",
            transient=transient,
        )
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "id": conversation_id,
                    "kind": conversation.kind,
                    "title": conversation.title,
                    "context": conversation.context,
                    "transient": conversation.transient,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            "utf-8",
        )
        with self._lock:
            self._items[conversation_id] = conversation
        self._add_page(conversation)
        self.ctx.ui.open_page(PAGE_PREFIX + conversation_id)
        self.ctx.ui.refresh("conversation.create")
        return conversation_id

    def delete(self, conversation_id) -> bool:
        conversation = self.get(conversation_id)
        if conversation is None:
            return False
        page = PAGE_PREFIX + conversation.id
        with self._lock:
            self._items.pop(conversation.id, None)
            self._pages.discard(page)
        try:
            shutil.rmtree(conversation.directory, ignore_errors=True)
        except OSError:
            pass
        try:
            self.ctx.ui.remove_page(page)
        except Exception:  # noqa: BLE001
            pass
        self.ctx.ui.refresh("conversation.delete")
        return True

    def get(self, conversation_id):
        with self._lock:
            return self._items.get(str(conversation_id))

    def append(self, conversation_id, role, text) -> None:
        conversation = self.get(conversation_id)
        if conversation is None:
            return
        entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "role": str(role), "text": str(text)}
        with self._lock:
            conversation.directory.mkdir(parents=True, exist_ok=True)
            with conversation.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.ctx.ui.refresh("conversation")

    def append_choice(self, conversation_id, question_id, prompt, choices) -> None:
        conversation = self.get(conversation_id)
        if conversation is None:
            return
        entry = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "role": "assistant",
            "kind": "choice",
            "question_id": str(question_id),
            "prompt": str(prompt),
            "choices": [
                {"value": str(item.get("value")), "label": str(item.get("label"))}
                for item in choices
            ],
        }
        with self._lock:
            conversation.directory.mkdir(parents=True, exist_ok=True)
            with conversation.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.ctx.ui.refresh("conversation.choice")

    def open(self, conversation_id) -> bool:
        conversation = self.get(conversation_id)
        if conversation is None:
            return False
        class_id = str(conversation.context.get("session_id") or "")
        if class_id:
            try:
                input_service = self.ctx.service("input") if self.ctx is not None else None
                if input_service is not None:
                    input_service.open_class(class_id)
            except Exception:  # noqa: BLE001
                LOG.debug("cannot select class for conversation", exc_info=True)
        self.ctx.ui.open_page(PAGE_PREFIX + str(conversation_id))
        return True

    def update_context(self, conversation_id, updates) -> None:
        conversation = self.get(conversation_id)
        if conversation is None:
            return
        with self._lock:
            conversation.context.update({str(key): value for key, value in dict(updates).items()})
            meta_path = conversation.directory / "meta.json"
            meta = {
                "id": conversation.id,
                "kind": conversation.kind,
                "title": conversation.title,
                "context": conversation.context,
                "transient": conversation.transient,
            }
            try:
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", "utf-8")
            except OSError:
                LOG.debug("cannot persist conversation context", exc_info=True)
        self.ctx.ui.refresh("conversation.context")

    def for_class(self, class_id):
        with self._lock:
            items = [item for item in self._items.values() if not item.transient]
        rows = []
        for item in items:
            if str(item.context.get("session_id") or "") != str(class_id):
                continue
            kind = "教学" if item.kind == "teaching" else "提问"
            rows.append({"id": item.id, "label": kind, "title": item.title, "kind": item.kind})
        rows.sort(key=lambda row: row["title"])
        return rows

    def _tree_children(self, class_id: str):
        return [
            {
                "id": f"conversation:{row['id']}",
                "label": row["label"],
                "subtitle": row["title"],
                "action": {"action": "conversation.open", "value": {"id": row["id"]}},
            }
            for row in self.for_class(class_id)
        ]

    # -- actions -------------------------------------------------------
    def _send(self, payload):
        values = _values(payload)
        conversation_id = str(values.get("conversation_id") or "")
        text = str(values.get("message") or "").strip()
        conversation = self.get(conversation_id)
        if conversation is None or not text:
            return {"ok": False, "message": "会话或消息不可用"}
        provider = self._provider(conversation.kind)
        if provider is None:
            return {"ok": False, "message": f"没有为 {conversation.kind} 注册会话 provider"}
        self.append(conversation_id, "user", text)
        try:
            provider[0](conversation_id, text, conversation.context)
        except Exception as exc:  # noqa: BLE001
            self.append(conversation_id, "system", f"发送失败：{exc}")
        self.ctx.ui.refresh("conversation.send")
        return {"ok": True, "refresh": True}

    def _delete_action(self, payload):
        conversation_id = str(_values(payload).get("id") or "")
        if not conversation_id:
            return {"ok": False, "message": "没有指定对话"}
        return {"ok": self.delete(conversation_id), "refresh": True, "message": "已删除对话"}

    def _open_action(self, payload):
        conversation_id = str(_values(payload).get("id") or "")
        return {"ok": self.open(conversation_id), "open_page": PAGE_PREFIX + conversation_id}

    # -- UI ------------------------------------------------------------
    @staticmethod
    def preset_html(presets):
        buttons = "".join(
            f'<button type="button" class="btn btn-sm conv-preset" data-fill-text="{html.escape(item, quote=True)}">{html.escape(item)}</button>'
            for item in presets
        )
        return (
            '<details class="conv-presets">'
            '<summary class="btn btn-sm">提示 ▾</summary>'
            f'<div class="conv-preset-menu">{buttons}</div>'
            "</details>"
        )

    def _render_list(self):
        with self._lock:
            items = [item for item in self._items.values() if not item.transient]
        groups: dict[str, dict[str, list[Conversation]]] = {}
        for item in items:
            course = str(item.context.get("course") or "未归属课程")
            session = str(item.context.get("session_title") or item.context.get("session_id") or "未分组")
            groups.setdefault(course, {}).setdefault(session, []).append(item)
        blocks = [{"type": BlockType.HTML, "html": '<div class="conv-side-title">对话</div>'}]
        if not groups:
            blocks.append({"type": BlockType.TEXT, "text": "还没有对话。"})
            return blocks
        for course, sessions in groups.items():
            blocks.append({"type": BlockType.HTML, "html": f'<div class="conv-side-course">{html.escape(course)}</div>'})
            for session, conversations in sessions.items():
                blocks.append({"type": BlockType.HTML, "html": f'<div class="conv-side-session">{html.escape(session)}</div>'})
                rows = []
                for item in conversations:
                    payload = html.escape(json.dumps({"id": item.id}, ensure_ascii=False), quote=True)
                    rows.append(
                        '<div class="conv-list-row">'
                        f'<button type="button" class="conv-list-item" data-action="conversation.open" data-value="{payload}">'
                        f'<span class="conv-list-title">{html.escape(item.title)}</span>'
                        f'<span class="conv-list-kind">{html.escape("教学" if item.kind == "teaching" else "提问")}</span>'
                        "</button>"
                        f'<button type="button" class="conv-list-delete" data-action="conversation.delete" data-value="{payload}" title="删除对话">×</button>'
                        "</div>"
                    )
                blocks.append({"type": BlockType.HTML, "html": '<div class="conv-list">' + "".join(rows) + "</div>"})
        return blocks

    def _render(self, conversation_id: str):
        conversation = self.get(conversation_id)
        if conversation is None:
            return [{"type": BlockType.TEXT, "text": "对话已不存在。"}]
        blocks = [{"type": BlockType.MARKDOWN, "text": f"### {conversation.title}"}]
        bubbles = []
        with self._lock:
            entries = conversation.messages()
        for entry in entries:
            role = str(entry.get("role") or "")
            label = {"user": "你", "assistant": "助教", "system": "系统"}.get(role, role)
            if entry.get("kind") == "choice":
                buttons = "".join(
                    f'<button type="button" class="conv-choice-option" '
                    f'data-action="learning.answer_choice" '
                    f'data-value="{html.escape(json.dumps({"question_id": entry.get("question_id"), "value": choice.get("value")}, ensure_ascii=False), quote=True)}">'
                    f'{html.escape(str(choice.get("label") or ""))}</button>'
                    for choice in (entry.get("choices") or [])
                )
                bubbles.append(
                    f'<div class="conv-row conv-{html.escape(role)}">'
                    f'<div class="conv-role">{html.escape(label)}</div>'
                    f'<div class="conv-bubble conv-choice">{render_markdown(str(entry.get("prompt") or ""))}'
                    f'<div class="conv-choice-options">{buttons}</div></div>'
                    f'</div>'
                )
                continue
            bubbles.append(
                f'<div class="conv-row conv-{html.escape(role)}">'
                f'<div class="conv-role">{html.escape(label)}</div>'
                f'<div class="conv-bubble">{render_markdown(str(entry.get("text", "")))}</div>'
                f'</div>'
            )
        if bubbles:
            blocks.append({
                "type": BlockType.HTML,
                "html": '<div class="markdown prose conv-messages">' + "".join(bubbles) + "</div>",
            })
        provider = self._provider(conversation.kind)
        commands = []
        if provider is not None and provider[2] is not None:
            try:
                commands = list(provider[2](conversation.id, conversation.context) or [])
            except Exception:  # noqa: BLE001
                LOG.debug("conversation composer failed", exc_info=True)
        menu = []
        for command in commands:
            label = html.escape(str(command.get("label") or ""), quote=True)
            if command.get("action"):
                value = html.escape(json.dumps(command.get("value") or {}, ensure_ascii=False), quote=True)
                menu.append(
                    f'<button type="button" class="conv-command" data-action="{html.escape(str(command["action"]), quote=True)}" data-value="{value}">{label}</button>'
                )
            else:
                menu.append(
                    f'<button type="button" class="conv-command" data-fill-text="{html.escape(str(command.get("text") or command.get("label") or ""), quote=True)}">{label}</button>'
                )
        plus = (
            '<div class="conv-plus-wrap">'
            '<button type="button" class="conv-plus" aria-label="更多">+</button>'
            '<div class="conv-plus-menu">' + "".join(menu) + "</div>"
            "</div>"
        ) if menu else ""
        blocks.append({
            "type": BlockType.HTML,
            "html": (
                '<form class="conv-composer" data-action="conversation.send">'
                f'<input type="hidden" name="conversation_id" value="{html.escape(conversation_id, quote=True)}">'
                '<div class="conv-composer-row">'
                + plus
                + '<textarea name="message" rows="1" placeholder="输入你的问题…"></textarea>'
                '<button type="submit" class="conv-send" aria-label="发送">发送</button>'
                "</div></form>"
            ),
        })
        return blocks


__all__ = ["ConversationPlugin", "ConversationService"]
