"""Declarative UI model: tabs, toolboxes, pages, blocks, actions and commands."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Any, Callable, Iterable, Mapping

from html import escape as html_escape

from markdown_it import MarkdownIt
from mdit_py_plugins.footnote import footnote_plugin

logger = logging.getLogger("lecture_agent.ui")

# Menu bar category order (unknown categories fall between 视图 and 调试).
_CATEGORY_ORDER = {"课程": 0, "视图": 10, "调试": 90}
_DEFAULT_CATEGORY_ORDER = 50


class Side(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class BlockType(StrEnum):
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    TABS = "tabs"
    FORM = "form"
    TREE = "tree"
    BUTTONS = "buttons"
    KEYVALUE = "keyvalue"
    ROWS = "rows"
    PLAYER = "player"
    PROGRESS = "progress"
    SWITCH = "switch"
    OAUTH = "oauth"
    QA = "qa"
    AGENT_ACTIVITY = "agent_activity"
    IMAGE = "image"
    TASK_QUEUE = "task_queue"
    TABLE = "table"
    CARDS = "cards"
    LIST = "list"
    LOG = "log"
    VIDEO = "video"
    BADGE = "badge"


class FormInputType(StrEnum):
    STRING = "string"
    TEXT = "text"
    TEXTAREA = "textarea"
    NUMBER = "number"
    CHECKBOX = "checkbox"
    SELECT = "select"
    PASSWORD = "password"
    BUTTON = "button"
    SUBMIT = "submit"


class TaskState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"


Renderer = Callable[[], Any]
ActionHandler = Callable[[dict[str, Any]], Any]


def _markdown_renderer() -> MarkdownIt:
    # Raw HTML is allowed because plugins use small inline buttons in notes /
    # tables.  LLM-generated markdown is passed through ``_sanitize_html``.
    md = MarkdownIt(
        "commonmark", {"html": True, "linkify": False, "typographer": False}
    )
    for rule in ("table", "strikethrough"):
        try:
            md.enable(rule)
        except Exception:  # pragma: no cover - rule availability differs
            pass
    md.use(footnote_plugin)
    return md


_MD = _markdown_renderer()
_SCRIPT_RE = re.compile(
    r"<\s*(script|iframe|object|embed|link|meta)\b[^>]*>.*?<\s*/\s*\1\s*>", re.I | re.S
)
_SELF_CLOSING_RE = re.compile(
    r"<\s*(script|iframe|object|embed|link|meta)\b[^>]*/?>", re.I
)
_EVENT_ATTR_RE = re.compile(r"\son[a-z]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)", re.I)
_JS_URL_RE = re.compile(r"\b(href|src)\s*=\s*(['\"])\s*javascript:[^'\"]*\2", re.I)


_FENCE_RE = re.compile(
    r"(^```.*?^```\s*$|^~~~.*?^~~~\s*$|```.+?```|~~~.+?~~~)",
    re.S | re.M,
)
_BLOCK_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)")


def _normalise_math(text: str) -> tuple[str, dict[str, str]]:
    """Pull math out of Markdown before it is parsed.

    Markdown treats ``_``/``*`` inside formulas as emphasis and splits the
    surrounding ``$$`` block across nodes, which defeats KaTeX auto-render.
    Formulas are replaced by inert tokens here and restored after rendering.

    Returns ``(markdown_with_tokens, {token: formula})``.
    """

    blocked: dict[str, str] = {}
    counter = 0

    def token(formula: str) -> str:
        nonlocal counter
        name = f"MATHTKN{counter}ENDMATH"
        counter += 1
        blocked[name] = formula
        return name

    def block(match: re.Match[str]) -> str:
        body = re.sub(r"\n\s*\n", "\n", match.group(1)).strip()
        return "\n\n" + token("$$\n" + body + "\n$$") + "\n\n"

    def inline(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        return token("$" + body + "$")

    parts = _FENCE_RE.split(text)
    for index in range(0, len(parts), 2):
        segment = _BLOCK_MATH_RE.sub(block, parts[index])
        segment = _INLINE_MATH_RE.sub(inline, segment)
        parts[index] = segment
    return "".join(parts), blocked


def _restore_math(html: str, blocked: dict[str, str]) -> str:
    for name, formula in blocked.items():
        html = html.replace(name, html_escape(formula))
    return html


def _sanitize_html(html: str) -> str:
    """Small, dependency-free guard for LLM-authored markdown."""

    html = _SCRIPT_RE.sub("", html)
    html = _SELF_CLOSING_RE.sub("", html)
    html = _EVENT_ATTR_RE.sub("", html)
    html = _JS_URL_RE.sub(r"\1=\2#\2", html)
    return html


def render_markdown(text: str, *, document_id: str = "") -> str:
    prepared, blocked = _normalise_math(text)
    env = {"docId": sha256(document_id.encode()).hexdigest()} if document_id else {}
    html = _MD.render(prepared, env)
    html = _restore_math(html, blocked)
    return _sanitize_html(html)


@dataclass(slots=True)
class TabHandle:
    ui: "UIManager"
    id: str

    def remove(self) -> None:
        self.ui.remove_tab(self.id)

    def add_item(self, id: str, title: str, **kwargs: Any) -> "ItemHandle":
        return self.ui.add_toolbox_item(self.id, id, title, **kwargs)


@dataclass(slots=True)
class ItemHandle:
    ui: "UIManager"
    id: str

    def remove(self) -> None:
        self.ui.remove_toolbox_item(self.id)


@dataclass(slots=True)
class PageHandle:
    ui: "UIManager"
    id: str

    def open(self) -> None:
        self.ui.open_page(self.id)

    def remove(self) -> None:
        self.ui.remove_page(self.id)

    def update(self, render: Renderer | None = None) -> None:
        if render is not None:
            self.ui.set_page_render(self.id, render)
        self.ui.refresh(f"page:{self.id}")


@dataclass(slots=True)
class ActionResult:
    ok: bool = True
    message: str = ""
    refresh: bool = True
    open_page: str | None = None
    # External page the browser should open (device authorizations and the like).
    open_url: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "ActionResult":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(
                ok=bool(value.get("ok", True)),
                message=str(value.get("message", "")),
                refresh=bool(value.get("refresh", True)),
                open_page=value.get("open_page"),
                open_url=value.get("open_url"),
                data=dict(value.get("data", {}) or {}),
            )
        return cls(message=str(value))


@dataclass(slots=True)
class _Tab:
    id: str
    side: Side
    title: str
    icon: str | None
    order: float
    plugin_id: str
    # Plugins that share this tab (a tab may be contributed by several
    # plugins, e.g. "note" is used by both summarizers).
    owners: set[str] = field(default_factory=set)


@dataclass(slots=True)
class _Item:
    id: str
    tab_id: str
    title: str
    description: str
    order: float
    render: Renderer | None
    plugin_id: str
    page_id: str | None = None


@dataclass(slots=True)
class _Page:
    id: str
    title: str
    icon: str | None
    order: float
    closable: bool
    render: Renderer | None
    plugin_id: str


@dataclass(slots=True)
class _Action:
    id: str
    handler: ActionHandler
    plugin_id: str


@dataclass(slots=True)
class _Command:
    id: str
    title: str
    category: str
    order: float
    page_id: str | None
    action: str | None
    keywords: str
    plugin_id: str


class UIManager:
    """Thread-safe declarative UI registry.

    Plugins contribute tabs, toolbox items and pages.  A page renderer returns
    a block list (or a string/Markdown); the web layer serialises ``state()``.
    """

    def __init__(
        self,
        *,
        on_change: Callable[[str], None] | None = None,
        on_notify: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._on_change = on_change
        self._on_notify = on_notify
        self._status_provider: Callable[[], dict[str, Any]] | None = None
        self._status_provider_owner: str = ""
        self._context_provider: Callable[[], dict[str, Any] | None] | None = None
        self._context_provider_owner: str = ""
        self._task_providers: list[tuple[str, Callable[[], list[dict[str, Any]]]]] = []
        self._lock = threading.RLock()
        self._tabs: dict[str, _Tab] = {}
        self._items: dict[str, _Item] = {}
        self._pages: dict[str, _Page] = {}
        self._actions: dict[str, _Action] = {}
        self._commands: dict[str, _Command] = {}
        self._open_pages: list[str] = []
        self._active_page: str | None = None
        self._version = 0
        self._last_refresh = 0.0
        self.register_action(
            "ui.open_page",
            lambda payload: {
                "ok": True,
                "open_page": payload.get("page_id") or payload.get("value"),
            },
            plugin_id="core",
        )
        self.add_page(
            "ui:tasks",
            "任务队列",
            render=lambda: [{"type": BlockType.TASK_QUEUE}],
            plugin_id="core",
        )
        self.register_command(
            "ui.tasks",
            "任务队列",
            category="视图",
            page_id="ui:tasks",
            keywords="tasks queue 任务 队列",
            plugin_id="core",
        )

    def add_tab(
        self,
        side: Side,
        id: str,
        title: str,
        *,
        icon: str | None = None,
        order: float = 0,
        plugin_id: str = "",
    ) -> TabHandle:
        if not isinstance(side, Side):
            raise TypeError("side must be a Side member")
        with self._lock:
            existing = self._tabs.get(id)
            if existing is not None:
                # A plugin may *share* an existing tab (same side + title), but
                # an anonymous duplicate is still a bug in the caller.
                if not plugin_id:
                    raise ValueError(f"UI tab {id!r} already exists")
                if existing.side != side or existing.title != title:
                    raise ValueError(
                        f"UI tab {id!r} already exists with different side/title"
                    )
                existing.owners.add(plugin_id)
                return TabHandle(self, id)
            tab = _Tab(
                id=id,
                side=side,
                title=title,
                icon=icon,
                order=float(order),
                plugin_id=plugin_id,
            )
            if plugin_id:
                tab.owners.add(plugin_id)
            self._tabs[id] = tab
        return TabHandle(self, id)

    def remove_tab(self, id: str) -> None:
        with self._lock:
            self._tabs.pop(id, None)
            item_ids = [item.id for item in self._items.values() if item.tab_id == id]
            for item_id in item_ids:
                self._items.pop(item_id, None)
        self.refresh(f"tab-removed:{id}")

    def add_toolbox_item(
        self,
        tab_id: str,
        id: str,
        title: str,
        *,
        description: str = "",
        order: float = 0,
        render: Renderer | None = None,
        plugin_id: str = "",
        page_id: str | None = None,
    ) -> ItemHandle:
        with self._lock:
            if tab_id not in self._tabs:
                raise ValueError(f"unknown UI tab {tab_id!r}")
            if id in self._items:
                raise ValueError(f"UI toolbox item {id!r} already exists")
            self._items[id] = _Item(
                id=id,
                tab_id=tab_id,
                title=title,
                description=description,
                order=float(order),
                render=render,
                plugin_id=plugin_id,
                page_id=page_id,
            )
        return ItemHandle(self, id)

    def remove_toolbox_item(self, id: str) -> None:
        with self._lock:
            self._items.pop(id, None)
        self.refresh(f"item-removed:{id}")

    def add_page(
        self,
        id: str,
        title: str,
        *,
        render: Renderer | None = None,
        icon: str | None = None,
        order: float = 0,
        closable: bool = True,
        plugin_id: str = "",
    ) -> PageHandle:
        with self._lock:
            if id in self._pages:
                raise ValueError(f"UI page {id!r} already exists")
            self._pages[id] = _Page(
                id=id,
                title=title,
                icon=icon,
                order=float(order),
                closable=closable,
                render=render,
                plugin_id=plugin_id,
            )
        return PageHandle(self, id)

    def remove_page(self, id: str) -> None:
        with self._lock:
            self._pages.pop(id, None)
            if id in self._open_pages:
                self._open_pages.remove(id)
            if self._active_page == id:
                self._active_page = self._open_pages[-1] if self._open_pages else None
        self.refresh(f"page-removed:{id}")

    def has_tab(self, id: str) -> bool:
        with self._lock:
            return id in self._tabs

    def has_toolbox_item(self, id: str) -> bool:
        with self._lock:
            return id in self._items

    def has_page(self, id: str) -> bool:
        with self._lock:
            return id in self._pages

    def has_action(self, id: str) -> bool:
        with self._lock:
            return id in self._actions

    def set_page_title(self, id: str, title: str) -> None:
        with self._lock:
            page = self._pages.get(id)
            if page is None:
                raise KeyError(f"unknown UI page {id!r}")
            page.title = str(title)

    def set_page_render(self, id: str, render: Renderer) -> None:
        with self._lock:
            page = self._pages.get(id)
            if page is None:
                raise KeyError(f"unknown UI page {id!r}")
            page.render = render

    def open_page(self, id: str) -> None:
        with self._lock:
            if id not in self._pages:
                raise KeyError(f"unknown UI page {id!r}")
            if id not in self._open_pages:
                self._open_pages.append(id)
            self._active_page = id
        self.refresh(f"page-opened:{id}")

    def show_home(self) -> None:
        with self._lock:
            self._active_page = None
        self.refresh("page-home")

    def close_page(self, id: str) -> None:
        with self._lock:
            if id in self._open_pages:
                self._open_pages.remove(id)
            if self._active_page == id:
                self._active_page = self._open_pages[-1] if self._open_pages else None
        self.refresh(f"page-closed:{id}")

    def register_action(
        self, action_id: str, handler: ActionHandler, *, plugin_id: str = ""
    ) -> None:
        if not callable(handler):
            raise TypeError("action handler must be callable")
        with self._lock:
            if action_id in self._actions:
                raise ValueError(f"UI action {action_id!r} already registered")
            self._actions[action_id] = _Action(action_id, handler, plugin_id)

    def action_owner(self, action_id: str) -> str | None:
        """Return the plugin that registered *action_id*, or ``None``."""

        with self._lock:
            action = self._actions.get(action_id)
        return None if action is None else action.plugin_id or None

    def unregister_action(
        self, action_id: str, *, plugin_id: str | None = None
    ) -> None:
        """Drop an action; ``plugin_id`` restricts it to its owner."""

        with self._lock:
            action = self._actions.get(action_id)
            if action is None:
                return
            if plugin_id is not None and action.plugin_id != plugin_id:
                raise ValueError(
                    f"UI action {action_id!r} belongs to {action.plugin_id!r}, "
                    f"not {plugin_id!r}"
                )
            self._actions.pop(action_id, None)

    def register_command(
        self,
        command_id: str,
        title: str,
        *,
        category: str = "",
        order: float = 0,
        page_id: str | None = None,
        action: str | None = None,
        keywords: str = "",
        plugin_id: str = "",
    ) -> None:
        """Register a palette/menu entry that opens a page or runs an action."""

        if not page_id and not action:
            raise ValueError("command needs a page_id or an action")
        with self._lock:
            self._commands[command_id] = _Command(
                id=command_id,
                title=title,
                category=category,
                order=float(order),
                page_id=page_id,
                action=action,
                keywords=keywords,
                plugin_id=plugin_id,
            )

    def unregister_command(self, command_id: str) -> None:
        with self._lock:
            self._commands.pop(command_id, None)

    def remove_plugin_commands(self, plugin_id: str) -> None:
        with self._lock:
            for command_id, command in list(self._commands.items()):
                if command.plugin_id == plugin_id:
                    self._commands.pop(command_id, None)

    def list_commands(self) -> list[dict[str, Any]]:
        with self._lock:
            commands = sorted(
                self._commands.values(),
                key=lambda c: (
                    _CATEGORY_ORDER.get(c.category, _DEFAULT_CATEGORY_ORDER),
                    c.order,
                    c.id,
                ),
            )
        return [
            {
                "id": c.id,
                "title": c.title,
                "category": c.category,
                "page": c.page_id,
                "action": c.action,
                "keywords": c.keywords,
            }
            for c in commands
        ]

    def dispatch_action(
        self,
        action_id: str,
        payload: dict[str, Any] | None = None,
        *,
        page_id: str | None = None,
        item_id: str | None = None,
    ) -> ActionResult:
        merged = dict(payload or {})
        merged.setdefault("page_id", page_id)
        merged.setdefault("item_id", item_id)
        with self._lock:
            action = self._actions.get(action_id)
        if action is None:
            return ActionResult(ok=False, message=f"未知操作：{action_id}")
        try:
            result = ActionResult.from_value(action.handler(merged))
        except Exception as exc:  # noqa: BLE001 - actions are plugin code
            logger.exception("UI action failed: %s", action_id)
            return ActionResult(ok=False, message=f"操作失败：{exc}", refresh=True)
        if result.open_page:
            try:
                self.open_page(result.open_page)
            except KeyError:
                pass
        if result.refresh:
            self.refresh(f"action:{action_id}")
        return result

    def refresh(self, reason: str = "") -> None:
        with self._lock:
            self._version += 1
            self._last_refresh = time.time()
        if self._on_change is not None:
            try:
                self._on_change(reason)
            except Exception:  # pragma: no cover - notification is best effort
                logger.exception("UI change listener failed")

    def notify(self, topic: str, payload: Mapping[str, Any] | None = None) -> None:
        """Push a transient message to connected clients.

        Unlike :meth:`refresh` this does not invalidate the state: the browser
        patches the named target in place, so streaming output never causes a
        full-page re-render.
        """

        if self._on_notify is None:
            return
        self._on_notify(str(topic), dict(payload or {}))

    def set_status_provider(
        self,
        provider: Callable[[], dict[str, Any]] | None,
        *,
        plugin_id: str = "",
    ) -> None:
        with self._lock:
            self._status_provider = provider
            self._status_provider_owner = plugin_id if provider is not None else ""

    def set_context_provider(
        self,
        provider: Callable[[], dict[str, Any] | None] | None,
        *,
        plugin_id: str = "",
    ) -> None:
        """Register the "what is this panel about" descriptor.

        The web layer renders it at the top of the right sidebar (and hides the
        panel when it is empty), so a sidebar never silently refers to an
        unknown course.
        """

        with self._lock:
            self._context_provider = provider
            self._context_provider_owner = plugin_id if provider is not None else ""

    def register_task_provider(
        self,
        provider: Callable[[], list[dict[str, Any]]],
        *,
        plugin_id: str = "",
    ) -> None:
        """Register a provider of pending and running in-memory tasks.

        Tasks contain id, label, detail, state, and optional session_id.
        An optional cancel descriptor contains an action and its payload.
        Providers without a state describe running tasks.
        """

        with self._lock:
            if any(existing is provider for _owner, existing in self._task_providers):
                return
            self._task_providers.append((plugin_id, provider))

    def get_tasks(self) -> list[dict[str, Any]]:
        """Collect active tasks without retaining task history."""

        with self._lock:
            providers = list(self._task_providers)
        tasks: list[dict[str, Any]] = []
        for _owner, provider in providers:
            try:
                tasks.extend(
                    task
                    for task in (provider() or [])
                    if task.get("state", TaskState.RUNNING)
                    in {TaskState.PENDING, TaskState.RUNNING}
                )
            except Exception:  # noqa: BLE001 - isolate task providers
                logger.exception("UI task provider failed")
        return tasks

    def remove_plugin(self, plugin_id: str) -> None:
        """Drop every contribution owned by *plugin_id* (plugin teardown).

        Shared tabs stay alive as long as another owner remains; a tab whose
        declaring plugin leaves and that has no owners and no items left is
        removed.
        """

        if not plugin_id:
            return
        with self._lock:
            for page_id in [
                page.id for page in self._pages.values() if page.plugin_id == plugin_id
            ]:
                self._pages.pop(page_id, None)
                if page_id in self._open_pages:
                    self._open_pages.remove(page_id)
                if self._active_page == page_id:
                    self._active_page = (
                        self._open_pages[-1] if self._open_pages else None
                    )
            for item_id in [
                item.id for item in self._items.values() if item.plugin_id == plugin_id
            ]:
                self._items.pop(item_id, None)
            for action_id in [
                action.id
                for action in self._actions.values()
                if action.plugin_id == plugin_id
            ]:
                self._actions.pop(action_id, None)
            for command_id in [
                command.id
                for command in self._commands.values()
                if command.plugin_id == plugin_id
            ]:
                self._commands.pop(command_id, None)
            self._task_providers = [
                (owner, provider)
                for owner, provider in self._task_providers
                if owner != plugin_id
            ]
            for tab_id in list(self._tabs):
                tab = self._tabs[tab_id]
                tab.owners.discard(plugin_id)
                if tab.plugin_id == plugin_id:
                    if tab.owners:
                        # Ownership passes to a remaining sharer so a later
                        # teardown of *that* plugin still cleans the tab up.
                        tab.plugin_id = sorted(tab.owners)[0]
                    else:
                        tab.plugin_id = ""
                if tab.plugin_id:
                    continue
                if any(item.tab_id == tab_id for item in self._items.values()):
                    continue
                self._tabs.pop(tab_id, None)
            if self._status_provider_owner == plugin_id:
                self._status_provider = None
                self._status_provider_owner = ""
            if self._context_provider_owner == plugin_id:
                self._context_provider = None
                self._context_provider_owner = ""
        self.refresh(f"plugin-removed:{plugin_id}")

    def _render(self, renderer: Renderer | None, label: str) -> list[dict[str, Any]]:
        if renderer is None:
            return []
        try:
            return self._normalize(renderer())
        except Exception as exc:  # noqa: BLE001 - a broken page must not kill UI
            logger.exception("UI render failed: %s", label)
            return [
                {
                    "type": BlockType.HTML,
                    "html": f'<div class="alert theme-danger" role="alert">渲染失败：{_escape(str(exc))}</div>',
                }
            ]

    def _normalize(self, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, str):
            return [
                {
                    "type": BlockType.MARKDOWN,
                    "text": value,
                    "html": render_markdown(value),
                }
            ]
        if isinstance(value, dict):
            if "blocks" in value and "type" not in value:
                return self._normalize(value["blocks"])
            return self._normalize_block(value)
        if isinstance(value, Iterable):
            blocks: list[dict[str, Any]] = []
            for item in value:
                blocks.extend(self._normalize(item))
            return blocks
        return [{"type": BlockType.TEXT, "text": str(value)}]

    def _normalize_block(self, block: dict[str, Any]) -> list[dict[str, Any]]:
        block = dict(block)
        btype: BlockType | None = block.get("type")
        if btype == BlockType.MARKDOWN:
            text = str(block.get("text", ""))
            block.setdefault("html", render_markdown(text))
        elif btype == BlockType.TABS:
            tabs = []
            for tab in block.get("tabs", []):
                tab = dict(tab)
                tab["blocks"] = self._normalize(tab.get("blocks", []))
                tabs.append(tab)
            block["tabs"] = tabs
        elif btype == BlockType.FORM:
            block.setdefault("fields", [])
        elif btype == BlockType.TREE:
            block["nodes"] = _normalize_tree_nodes(block.get("nodes", []))
        return [block]

    def state(self) -> dict[str, Any]:
        with self._lock:
            tabs = sorted(self._tabs.values(), key=lambda t: (t.order, t.id))
            items = sorted(self._items.values(), key=lambda i: (i.order, i.id))
            pages = sorted(self._pages.values(), key=lambda p: (p.order, p.id))
            open_pages = list(self._open_pages)
            active = self._active_page
            version = self._version

        def tab_payload(side: Side) -> list[dict[str, Any]]:
            out = []
            for tab in tabs:
                if tab.side != side:
                    continue
                tab_items = []
                for item in items:
                    if item.tab_id != tab.id:
                        continue
                    tab_items.append(
                        {
                            "id": item.id,
                            "tab_id": item.tab_id,
                            "title": item.title,
                            "description": item.description,
                            "order": item.order,
                            "page": item.page_id,
                            "blocks": self._render(item.render, f"item:{item.id}"),
                        }
                    )
                out.append(
                    {
                        "id": tab.id,
                        "side": tab.side,
                        "title": tab.title,
                        "icon": tab.icon,
                        "order": tab.order,
                        "toolbox": tab_items,
                    }
                )
            return out

        page_content: dict[str, list[dict[str, Any]]] = {}
        # Avoid rendering inactive documents on every state poll.
        for page in pages:
            if page.id == active:
                page_content[page.id] = self._render(page.render, f"page:{page.id}")

        status: dict[str, Any] = {}
        if self._status_provider is not None:
            try:
                status = dict(self._status_provider())
            except Exception as exc:  # noqa: BLE001
                status = {"error": str(exc)}
        tasks = self.get_tasks()

        context: dict[str, Any] | None = None
        if self._context_provider is not None:
            try:
                value = self._context_provider()
                context = dict(value) if isinstance(value, dict) and value else None
            except Exception:  # noqa: BLE001 - context is decoration
                logger.exception("UI context provider failed")
                context = None

        return {
            "version": version,
            "left_tabs": tab_payload(Side.LEFT),
            "right_tabs": tab_payload(Side.RIGHT),
            "pages": [
                {
                    "id": p.id,
                    "title": p.title,
                    "icon": p.icon,
                    "order": p.order,
                    "closable": p.closable,
                }
                for p in pages
            ],
            "open_pages": open_pages,
            "active_page": active,
            "page_content": page_content,
            "tasks": tasks,
            "context": context,
            "commands": self.list_commands(),
            "status": status,
        }

    def snapshot(self) -> dict[str, Any]:
        """Small introspection payload for tests/diagnostics."""

        with self._lock:
            return {
                "tabs": {t.id: t.side for t in self._tabs.values()},
                "items": {i.id: i.tab_id for i in self._items.values()},
                "pages": {p.id: p.title for p in self._pages.values()},
                "actions": sorted(self._actions),
                "commands": sorted(self._commands),
                "open_pages": list(self._open_pages),
                "active_page": self._active_page,
            }


def _normalize_tree_nodes(nodes: Iterable[Any], depth: int = 0) -> list[dict[str, Any]]:
    """Normalise tree nodes recursively (``nodes`` → ``children``)."""

    out: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        item = dict(node)
        item["depth"] = depth
        item.setdefault("label", str(item.get("title", "") or ""))
        item["children"] = _normalize_tree_nodes(
            item.get("children", []) or [], depth + 1
        )
        out.append(item)
    return out


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
