"""Declarative UI model: tabs, toolboxes, pages, blocks, actions and commands."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal

from html import escape as html_escape

from markdown_it import MarkdownIt

logger = logging.getLogger("lecture_agent.ui")

# Menu bar category order (unknown categories fall between 视图 and 调试).
_CATEGORY_ORDER = {"课程": 0, "视图": 10, "调试": 90}
_DEFAULT_CATEGORY_ORDER = 50

Side = Literal["left", "right"]
Renderer = Callable[[], Any]
ActionHandler = Callable[[dict[str, Any]], Any]


def _markdown_renderer() -> MarkdownIt:
    # Raw HTML is allowed because plugins use small inline buttons in notes /
    # tables.  LLM-generated markdown is passed through ``_sanitize_html``.
    md = MarkdownIt("commonmark", {"html": True, "linkify": False, "typographer": False})
    for rule in ("table", "strikethrough"):
        try:
            md.enable(rule)
        except Exception:  # pragma: no cover - rule availability differs
            pass
    return md


_MD = _markdown_renderer()
_SCRIPT_RE = re.compile(r"<\s*(script|iframe|object|embed|link|meta)\b[^>]*>.*?<\s*/\s*\1\s*>", re.I | re.S)
_SELF_CLOSING_RE = re.compile(r"<\s*(script|iframe|object|embed|link|meta)\b[^>]*/?>", re.I)
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


def render_markdown(text: str) -> str:
    prepared, blocked = _normalise_math(text)
    html = _MD.render(prepared)
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
    subtitle: str
    category: str
    icon: str | None
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

    def __init__(self, *, on_change: Callable[[str], None] | None = None) -> None:
        self._on_change = on_change
        self._status_provider: Callable[[], dict[str, Any]] | None = None
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

    # ------------------------------------------------------------------
    # Contributions
    # ------------------------------------------------------------------
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
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        with self._lock:
            if id in self._tabs:
                raise ValueError(f"UI tab {id!r} already exists")
            self._tabs[id] = _Tab(
                id=id,
                side=side,
                title=title,
                icon=icon,
                order=float(order),
                plugin_id=plugin_id,
            )
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

    def close_page(self, id: str) -> None:
        with self._lock:
            if id in self._open_pages:
                self._open_pages.remove(id)
            if self._active_page == id:
                self._active_page = self._open_pages[-1] if self._open_pages else None
        self.refresh(f"page-closed:{id}")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def register_action(
        self, action_id: str, handler: ActionHandler, *, plugin_id: str = ""
    ) -> None:
        if not callable(handler):
            raise TypeError("action handler must be callable")
        with self._lock:
            if action_id in self._actions:
                raise ValueError(f"UI action {action_id!r} already registered")
            self._actions[action_id] = _Action(action_id, handler, plugin_id)

    def unregister_action(self, action_id: str) -> None:
        with self._lock:
            self._actions.pop(action_id, None)

    # ------------------------------------------------------------------
    # Commands (command palette / menu bar)
    # ------------------------------------------------------------------
    def register_command(
        self,
        command_id: str,
        title: str,
        *,
        subtitle: str = "",
        category: str = "",
        icon: str | None = None,
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
                subtitle=subtitle,
                category=category,
                icon=icon,
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
                "subtitle": c.subtitle,
                "category": c.category,
                "icon": c.icon,
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

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def refresh(self, reason: str = "") -> None:
        with self._lock:
            self._version += 1
            self._last_refresh = time.time()
        if self._on_change is not None:
            try:
                self._on_change(reason)
            except Exception:  # pragma: no cover - notification is best effort
                logger.exception("UI change listener failed")

    def set_status_provider(self, provider: Callable[[], dict[str, Any]] | None) -> None:
        self._status_provider = provider

    def _render(self, renderer: Renderer | None, label: str) -> list[dict[str, Any]]:
        if renderer is None:
            return []
        try:
            return self._normalize(renderer())
        except Exception as exc:  # noqa: BLE001 - a broken page must not kill UI
            logger.exception("UI render failed: %s", label)
            return [
                {
                    "type": "html",
                    "html": f'<div class="ui-error">渲染失败：{_escape(str(exc))}</div>',
                }
            ]

    def _normalize(self, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, str):
            return [{"type": "markdown", "text": value, "html": render_markdown(value)}]
        if isinstance(value, dict):
            if "blocks" in value and "type" not in value:
                return self._normalize(value["blocks"])
            return self._normalize_block(value)
        if isinstance(value, Iterable):
            blocks: list[dict[str, Any]] = []
            for item in value:
                blocks.extend(self._normalize(item))
            return blocks
        return [{"type": "text", "text": str(value)}]

    def _normalize_block(self, block: dict[str, Any]) -> list[dict[str, Any]]:
        block = dict(block)
        btype = block.get("type")
        if btype == "markdown":
            text = str(block.get("text", ""))
            block.setdefault("html", render_markdown(text))
        elif btype == "tabs":
            tabs = []
            for tab in block.get("tabs", []):
                tab = dict(tab)
                tab["blocks"] = self._normalize(tab.get("blocks", []))
                tabs.append(tab)
            block["tabs"] = tabs
        elif btype == "form":
            block.setdefault("fields", [])
        elif btype == "tree":
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
        if active is None and open_pages:
            active = open_pages[-1]

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
        # Only render the visible page.  Rendering every open page on each
        # state poll made large board/notes pages expensive server-side and
        # shipped hundreds of KB the browser did not need.
        for page in pages:
            if page.id == active:
                page_content[page.id] = self._render(page.render, f"page:{page.id}")

        status: dict[str, Any] = {}
        if self._status_provider is not None:
            try:
                status = dict(self._status_provider())
            except Exception as exc:  # noqa: BLE001
                status = {"error": str(exc)}

        return {
            "version": version,
            "left_tabs": tab_payload("left"),
            "right_tabs": tab_payload("right"),
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
