"""Reader plugin: display markdown documents (lecture notes) in reader pages.

The notes plugin owns its files and gives the reader a list of
``{key, label, path}`` documents. Each lecture has its own reader pages.
The reader owns the selection menu. Plugins register actions for selected text.

Q&A pairs written by such a menu item live *inside* the note file as
``qa`` blocks (see :mod:`core.note_document`); the reader renders them with its
own collapsible component and offers one-click deletion.
"""

from __future__ import annotations

import threading
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from core.plugin import Plugin, PluginContext
from core.file_access import path_lock, publication_lock
from contextlib import ExitStack
from core.navigation import current_navigation
from core.ui import BlockType, render_markdown

from core import note_document as qa
from .types import SelectionKind

PAGE_ID = "reader:main"
PAGE_PREFIX = "reader:notes:"
OPEN_ACTION = "reader.open"
DELETE_QA_ACTION = "reader.qa_delete"
COLLAPSE_QA_ACTION = "reader.qa_collapse"


class ReaderService:
    """Public reader operations for documents and request navigation.

    Notes plugins and the Q&A plugin talk to the reader only through this
    object, never through the plugin instance.
    """

    __slots__ = ("_plugin",)

    def __init__(self, plugin: "ReaderPlugin") -> None:
        self._plugin = plugin

    def show(
        self,
        *,
        title: str = "笔记",
        docs: list[dict[str, Any]] | None = None,
        active: str = "",
        page_key: str = "",
        open_page: bool = True,
        owner: str = "",
    ) -> str:
        return self._plugin.show(
            title=title,
            docs=docs,
            active=active,
            page_key=page_key,
            open_page=open_page,
            owner=owner,
        )

    def register_activity_provider(self, owner, provider):
        with self._plugin._lock:
            self._plugin._activity_providers[owner] = provider

    def unregister_activity_provider(self, owner):
        with self._plugin._lock:
            self._plugin._activity_providers.pop(owner, None)

    def register_selection_action(
        self, page: str, item: dict[str, Any], *, owner: str = ""
    ) -> dict[str, Any]:
        """Add one selection-menu item; returns the normalized item."""

        return self._plugin.register_selection_action(page, item, owner=owner)

    def unregister_selection_action(
        self, page: str = "", item_id: str = "", *, owner: str = ""
    ) -> None:
        """Remove one item, or every item registered by *owner* (item_id "")."""

        self._plugin.unregister_selection_action(page, item_id, owner=owner)

    def selection_actions(self, page: str, owner: str = "") -> list[dict[str, Any]]:
        return self._plugin.selection_actions(page, owner)

    def qa_begin(
        self,
        path: str | Path,
        *,
        question: str,
        selection: str = "",
        intensity: qa.AskIntensity = qa.AskIntensity.INSTANT,
    ) -> dict[str, Any]:
        return self._plugin.qa_begin(
            path,
            question=question,
            selection=selection,
            intensity=intensity,
        )

    def qa_finish(
        self,
        path: str | Path,
        qa_id: str,
        *,
        answer: str,
        status: qa.QaStatus = qa.QaStatus.DONE,
        collapsed: bool = True,
    ) -> dict[str, Any] | None:
        return self._plugin.qa_finish(
            path,
            qa_id,
            answer=answer,
            status=status,
            collapsed=collapsed,
        )

    def qa_set_collapsed(
        self, path: str | Path, qa_id: str, collapsed: bool
    ) -> dict[str, Any] | None:
        return self._plugin.qa_set_collapsed(path, qa_id, collapsed)

    def qa_context(
        self,
        path: str | Path,
        qa_id: str,
        *,
        before_chars: int = 2000,
        after_chars: int = 2000,
        history_chars: int = 1000,
    ) -> dict[str, Any]:
        return self._plugin.qa_context(
            path,
            qa_id,
            before_chars=before_chars,
            after_chars=after_chars,
            history_chars=history_chars,
        )

    def navigation(self, page_id: str) -> str:
        navigation = current_navigation()
        with self._plugin._lock:
            docs = self._plugin._states.get(page_id, {}).get("docs", [])
            keys = [doc["key"] for doc in docs]
        requested = navigation.doc if navigation.page == page_id else ""
        active = requested if requested in keys else keys[0] if keys else ""
        if navigation.page == page_id:
            navigation.doc = active
        return active


class ReaderPlugin(Plugin):
    id = "reader"
    requires: tuple[str, ...] = ()
    provides = ("reader",)

    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self._lock = threading.RLock()
        self._states: dict[str, dict[str, Any]] = {}
        self._activity_providers = {}
        # page id → owner → item id → normalized menu item
        self._selection: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        # item id → normalized menu item, offered on every reader page
        self._global_selection: dict[str, dict[str, Any]] = {}

        ctx.register_service("reader", ReaderService(self))

        ctx.ui.add_page(
            PAGE_ID,
            "阅读",
            render=lambda: self._render_page(PAGE_ID),
            order=5,
            plugin_id=self.id,
        )
        ctx.ui.register_action(OPEN_ACTION, self._act_open, plugin_id=self.id)
        ctx.ui.register_action(DELETE_QA_ACTION, self._act_delete_qa, plugin_id=self.id)
        ctx.ui.register_action(
            COLLAPSE_QA_ACTION, self._act_collapse_qa, plugin_id=self.id
        )

    def teardown(self) -> None:

        ctx = getattr(self, "ctx", None)
        if ctx is None:
            return
        for action_id in (OPEN_ACTION, DELETE_QA_ACTION, COLLAPSE_QA_ACTION):
            try:
                ctx.ui.unregister_action(action_id)
            except Exception:  # noqa: BLE001 - shutdown is best effort
                pass
        with self._lock:
            page_ids = list(self._states)
            self._states.clear()
            self._selection.clear()
            self._global_selection.clear()
        for page_id in [PAGE_ID, *page_ids]:
            try:
                ctx.ui.remove_page(page_id)
            except Exception:  # noqa: BLE001 - shutdown is best effort
                pass

    def show(
        self,
        *,
        title: str = "笔记",
        docs: list[dict[str, Any]] | None = None,
        active: str = "",
        page_key: str = "",
        open_page: bool = True,
        owner: str = "",
    ) -> str:
        """Display *docs* in the page for *page_key*; returns the page id.

        *owner* tags every document with the plugin that supplied it, so the
        selection menu only exposes the items registered by that plugin.
        """

        page_id = self._page_id(page_key)
        cleaned: list[dict[str, Any]] = []
        for doc in docs or []:
            path = str(doc.get("path", "") or "")
            key = str(doc.get("key", "") or Path(path).stem)
            label = str(doc.get("label", "") or key)
            if not path:
                continue
            cleaned.append(
                {
                    "key": key,
                    "label": label,
                    "path": path,
                    "owner": str(doc.get("owner", "") or owner),
                }
            )

        with self._lock:
            state = self._states.setdefault(page_id, {})
            state["title"] = str(title or "")
            state["docs"] = cleaned
            state["owner"] = str(owner or "")

        if not self.ctx.ui.has_page(page_id):
            self.ctx.ui.add_page(
                page_id,
                str(title or "阅读"),
                render=lambda pid=page_id: self._render_page(pid),
                order=5,
                plugin_id=self.id,
            )
        else:
            self.ctx.ui.set_page_title(page_id, str(title or "阅读"))

        self.ctx.ui.refresh("reader.show")
        if open_page:
            navigation = current_navigation()
            keys = {doc["key"] for doc in cleaned}
            requested = str(active or "")
            if not requested and navigation.page == page_id:
                requested = navigation.doc
            navigation.doc = (
                requested if requested in keys else cleaned[0]["key"] if cleaned else ""
            )
            self.ctx.ui.open_page(page_id)
        return page_id

    def clear(self, page_key: str = "") -> None:
        page_id = self._page_id(page_key)
        with self._lock:
            self._states.pop(page_id, None)
            self._selection.pop(page_id, None)

    @staticmethod
    def page_id_for(page_key: str) -> str:
        return f"{PAGE_PREFIX}{page_key}" if page_key else PAGE_ID

    def _page_id(self, page_key: str) -> str:
        return self.page_id_for(str(page_key or ""))

    def register_selection_action(
        self,
        page: str,
        item: Mapping[str, Any],
        *,
        owner: str = "",
    ) -> dict[str, Any]:
        """Add one selection-menu item to a reader page (or to every page).

        *page* is a page id (``reader:...``), a page key, or empty / ``"*"``
        for an item offered on every reader page.  Per-page items are gated by
        *owner* (they only appear on documents that owner supplied); global
        items are not gated and must validate the document themselves.

        Returns the normalized item.
        """

        item_id = str(item.get("id", "") or "").strip()
        action = str(item.get("action", "") or "").strip()
        if not item_id or not action:
            raise ValueError("selection item needs an id and an action")
        spec = {
            "id": item_id,
            "label": str(item.get("label", "") or item_id),
            "icon": str(item.get("icon", "") or ""),
            "action": action,
            "kind": SelectionKind(item.get("kind") or SelectionKind.ACTION),
            "placeholder": str(item.get("placeholder", "") or ""),
            "order": float(item.get("order", 0) or 0),
            "value": dict(item.get("value", {}) or {}),
            "intensities": list(item.get("intensities", []) or []),
            "owner": str(owner or item.get("owner", "") or ""),
        }
        text = str(page or "")
        with self._lock:
            if text in ("", "*"):
                self._global_selection[item_id] = spec
            else:
                page_id = self._as_page_id(text)
                per_owner = self._selection.setdefault(page_id, {})
                per_owner.setdefault(spec["owner"], {})[item_id] = spec
        self.ctx.ui.refresh(f"reader.selection:{text or '*'}")
        return dict(spec)

    def unregister_selection_action(
        self, page: str = "", item_id: str = "", *, owner: str = ""
    ) -> None:
        """Remove one menu item or every item registered by *owner*.

        With an empty *item_id* and a non-empty *owner* this purges that
        plugin's items from every page and from the global bucket, which is
        what plugin teardown needs.
        """

        text = str(page or "")
        if item_id:
            with self._lock:
                if text in ("", "*"):
                    self._global_selection.pop(item_id, None)
                    return
                page_id = self._as_page_id(text)
                per_owner = self._selection.get(page_id) or {}
                bucket = per_owner.get(str(owner or ""))
                if bucket:
                    bucket.pop(item_id, None)
            return
        if not owner:
            return
        with self._lock:
            for per_owner in self._selection.values():
                per_owner.pop(owner, None)
            for key, spec in list(self._global_selection.items()):
                if spec.get("owner") == owner:
                    self._global_selection.pop(key, None)
        self.ctx.ui.refresh(f"reader.selection:purge:{owner}")

    def selection_actions(self, page: str, owner: str = "") -> list[dict[str, Any]]:
        """Return the menu items offered on *page* for a document of *owner*."""

        page_id = self._as_page_id(page)
        with self._lock:
            per_owner = self._selection.get(page_id) or {}
            bucket = dict(per_owner.get(str(owner or "")) or {})
            merged = {**self._global_selection, **bucket}
        items = sorted(merged.values(), key=lambda spec: (spec["order"], spec["id"]))
        return [dict(spec) for spec in items]

    def _as_page_id(self, page: str) -> str:
        text = str(page or "")
        if not text:
            return PAGE_ID
        if text.startswith("reader:"):
            return text
        return self._page_id(text)

    # Q&A blocks: the reader owns the on-disk format, plugins only ask
    def qa_begin(
        self,
        path: str | Path,
        *,
        question: str,
        selection: str = "",
        intensity: qa.AskIntensity = qa.AskIntensity.INSTANT,
    ) -> dict[str, Any]:
        """Insert a *pending* Q&A block below the selected paragraph.

        Returns ``{"id", "ts", "anchored"}``.  Raises :class:`ValueError` when
        the path is not a document this reader is displaying.
        """

        return self._qa_write(
            path,
            lambda text: self._qa_insert(text, question, selection, intensity),
        )

    def _qa_insert(
        self, text: str, question: str, selection: str, intensity: qa.AskIntensity
    ) -> tuple[str, dict[str, Any]]:
        qa_id = qa.new_id()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        block = qa.build_block(
            qa_id,
            ts,
            status=qa.QaStatus.PENDING,
            question=question,
            answer="",
            intensity=intensity,
        )
        updated, anchored = qa.insert_block(text, block, selection)
        return updated, {"id": qa_id, "ts": ts, "anchored": anchored}

    def qa_finish(
        self,
        path: str | Path,
        qa_id: str,
        *,
        answer: str,
        status: qa.QaStatus = qa.QaStatus.DONE,
        collapsed: bool = True,
    ) -> dict[str, Any] | None:

        def mutate(text: str) -> tuple[str, dict[str, Any]]:
            block = None
            for existing in qa.parse_blocks(text):
                if existing.id == qa_id:
                    block = existing
                    break
            if block is None:
                raise ValueError("这条问答已不存在")
            replacement = qa.build_block(
                qa_id,
                block.ts,
                status=status,
                question=block.question,
                answer=answer,
                intensity=block.intensity,
                collapsed=collapsed,
            )
            updated = qa.replace_block(text, qa_id, replacement)
            if updated is None:
                raise ValueError("这条问答已不存在")
            return updated, {}

        return self._qa_write(path, mutate)

    def qa_set_collapsed(
        self, path: str | Path, qa_id: str, collapsed: bool
    ) -> dict[str, Any] | None:

        def mutate(text: str) -> tuple[str, dict[str, Any]]:
            for existing in qa.parse_blocks(text):
                if existing.id != qa_id:
                    continue
                replacement = qa.build_block(
                    qa_id,
                    existing.ts,
                    status=existing.status,
                    question=existing.question,
                    answer=existing.answer,
                    intensity=existing.intensity,
                    collapsed=bool(collapsed),
                )
                updated = qa.replace_block(text, qa_id, replacement)
                if updated is None:
                    raise ValueError("这条问答已不存在")
                return updated, {}
            raise ValueError("这条问答已不存在")

        return self._qa_write(path, mutate)

    def qa_context(
        self,
        path: str | Path,
        qa_id: str,
        *,
        before_chars: int = 500,
        after_chars: int = 500,
        history_chars: int = 1000,
    ) -> dict[str, Any]:

        resolved = self._known_path(path)
        if resolved is None:
            raise ValueError("未知的文稿路径")
        with publication_lock(resolved.parent), path_lock(resolved):
            text = resolved.read_text("utf-8")
        blocks = qa.parse_blocks(text)
        current = next((b for b in blocks if b.id == qa_id), None)
        if current is None:
            raise ValueError("未找到这条问答")

        before_text = qa.strip_blocks(text[: current.start])
        before = before_text[-before_chars:] if before_chars > 0 else ""
        after = qa.strip_blocks(text[current.end :])[: max(0, after_chars)]

        history: list[dict[str, str]] = []
        remaining = max(0, history_chars)
        for block in reversed(blocks):
            if remaining == 0:
                break
            if block.id == qa_id or block.start >= current.start:
                continue
            question = block.question[:remaining]
            remaining -= len(question)
            answer = block.answer[:remaining]
            remaining -= len(answer)
            history.append({"question": question, "answer": answer})
        history.reverse()

        return {
            "id": current.id,
            "ts": current.ts,
            "intensity": current.intensity,
            "question": current.question,
            "answer": current.answer,
            "before": before,
            "after": after,
            "history": history,
            "file_name": resolved.name,
        }

    def _qa_write(self, path: str | Path, mutate: Any) -> dict[str, Any]:
        resolved = self._known_path(path)
        if resolved is None:
            raise ValueError("未知的文稿路径")
        with publication_lock(resolved.parent), path_lock(resolved):
            text = resolved.read_text("utf-8")
            updated, info = mutate(text)
            resolved.write_text(updated, "utf-8")
        return info

    def _act_open(self, payload: dict[str, Any]) -> dict[str, Any]:
        docs = payload.get("docs")
        if not isinstance(docs, list) or not docs:
            return {"ok": False, "message": "没有可显示的文稿"}
        page_key = str(payload.get("page_key", "") or payload.get("session_id", ""))
        page_id = self.show(
            title=str(payload.get("title", "") or ""),
            docs=[dict(doc) for doc in docs if isinstance(doc, dict)],
            active=str(payload.get("active", "") or ""),
            page_key=page_key,
            owner=str(payload.get("owner", "") or ""),
        )
        return {"ok": True, "open_page": page_id}

    def _act_delete_qa(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = str(payload.get("path", "") or "")
        qa_id = str(payload.get("id", "") or "")
        if not path or not qa_id:
            return {"ok": False, "message": "缺少要删除的问答"}
        resolved = self._known_path(path)
        if resolved is None:
            return {"ok": False, "message": "未知的文稿路径"}
        try:
            with publication_lock(resolved.parent), path_lock(resolved):
                text = resolved.read_text("utf-8")
                updated = qa.remove_block(text, qa_id)
                if updated is None:
                    return {"ok": False, "message": "未找到这条问答"}
                resolved.write_text(updated, "utf-8")
        except OSError as exc:
            return {"ok": False, "message": f"写入失败：{exc}"}
        return {"ok": True, "message": "已删除这条问答"}

    def _act_collapse_qa(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = str(payload.get("path", "") or "")
        qa_id = str(payload.get("id", "") or "")
        collapsed = bool(payload.get("collapsed"))
        if not path or not qa_id:
            return {"ok": False, "message": "缺少要折叠的问答"}
        self.qa_set_collapsed(path, qa_id, collapsed)
        return {"ok": True}

    def _known_path(self, path: str | Path) -> Path | None:
        """Only documents the reader is actually displaying may be edited."""

        try:
            target = Path(path).resolve()
        except OSError:
            return None
        with self._lock:
            states = list(self._states.values())
        for state in states:
            for doc in state.get("docs") or []:
                try:
                    if Path(doc["path"]).resolve() == target:
                        return target
                except OSError:
                    continue
        return None

    def _render_page(self, page_id: str) -> list[dict[str, Any]]:
        with self._lock:
            state = dict(self._states.get(page_id) or {})
            docs = [dict(doc) for doc in state.get("docs") or []]
        if not docs:
            return [{"type": BlockType.TEXT, "text": "没有可显示的文稿"}]

        navigation = current_navigation()
        active = navigation.doc if navigation.page == page_id else ""
        if active not in {doc["key"] for doc in docs}:
            active = docs[0]["key"]

        tabs: list[dict[str, Any]] = []
        with ExitStack() as stack:
            for directory in sorted(
                {Path(doc["path"]).resolve().parent for doc in docs}
            ):
                stack.enter_context(publication_lock(directory))
            texts = {doc["path"]: _read_text(doc["path"]) for doc in docs}
        for doc in docs:
            text = texts[doc["path"]]
            owner = str(doc.get("owner", "") or state.get("owner") or "")
            blocks = _document_blocks(text, doc["path"])
            with self._lock:
                providers = tuple(self._activity_providers.values())
            for block in blocks:
                if (
                    block.get("type") == BlockType.QA
                    and block.get("status") == qa.QaStatus.PENDING
                ):
                    for provider in providers:
                        activity = provider(doc["path"], block["id"])
                        if activity is not None:
                            block["activity"] = activity
                            break
            tabs.append(
                {
                    "title": doc["label"],
                    "active": doc["key"] == active,
                    "meta": {
                        "doc_key": doc["key"],
                        "path": doc["path"],
                        "owner": owner,
                        "page_id": page_id,
                        "selection_actions": self.selection_actions(page_id, owner),
                    },
                    "blocks": blocks,
                }
            )

        return [
            {
                "type": BlockType.TABS,
                "id": page_id,
                "tabs": tabs,
            },
        ]


def _document_blocks(text: str, path: str) -> list[dict[str, Any]]:
    """Parse the whole note before separating its Q&A components."""

    if not text.strip():
        return [{"type": BlockType.TEXT, "text": "（暂无内容）"}]

    segments = qa.segments(text)
    prefix = f"<!--reader-qa-{sha256(text.encode()).hexdigest()}-"
    source = []
    for index, (kind, payload) in enumerate(segments):
        source.append(
            str(payload) if kind == BlockType.MARKDOWN else f"{prefix}{index}-->"
        )
    # Footnote definitions must remain visible to references across Q&A boundaries.
    remaining_html = render_markdown("\n\n".join(source), document_id=path)
    blocks: list[dict[str, Any]] = []
    markdown_text = ""
    for index, (kind, payload) in enumerate(segments):
        if kind == BlockType.MARKDOWN:
            markdown_text += str(payload)
            continue
        html, _, remaining_html = remaining_html.partition(f"{prefix}{index}-->")
        if html.strip():
            blocks.append(
                {"type": BlockType.MARKDOWN, "text": markdown_text, "html": html}
            )
        markdown_text = ""
        block = payload
        assert isinstance(block, qa.QABlock)
        blocks.append(
            {
                "type": BlockType.QA,
                "id": block.id,
                "path": path,
                "ts": block.ts,
                "status": block.status,
                "intensity": block.intensity,
                "collapsed": block.collapsed,
                "question": block.question,
                "answer": block.answer,
                "answer_html": (
                    render_markdown(
                        block.answer, document_id=f"{path}:{block.id}:answer"
                    )
                    if block.answer
                    else ""
                ),
                "delete_action": DELETE_QA_ACTION,
                "collapse_action": COLLAPSE_QA_ACTION,
            }
        )
    if remaining_html.strip():
        blocks.append(
            {"type": BlockType.MARKDOWN, "text": markdown_text, "html": remaining_html}
        )
    return blocks or [{"type": BlockType.TEXT, "text": "（暂无内容）"}]


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text("utf-8")
    except OSError:
        return ""
