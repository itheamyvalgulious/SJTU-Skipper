"""Reader plugin: display markdown documents (lecture notes) in reader pages.

Notes plugins (summarize / instant-summarize) own their files; they hand the
reader a list of ``{key, label, path}`` documents.  Every notes session gets
its **own page** (``reader:notes:<session_id>``), so several courses' handouts
can stay open at the same time instead of sharing one page.

The reader also owns the **selection menu**: when the user selects a passage
the browser shows a small floating menu whose items are contributed by the
plugin that created the page (``register_selection_action``).  Registration is
per page, not global: ``summarize`` registers "提问" on the pages it opens, so
the same page re-used by ``instant-summarize`` shows no summarize menu item.

Q&A pairs written by such a menu item live *inside* the note file as
``qa`` blocks (see :mod:`plugins.reader.qa`); the reader renders them with its
own collapsible component and offers one-click deletion.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from core.plugin import Plugin, PluginContext
from core.ui import render_markdown

from . import qa

PAGE_ID = "reader:main"
PAGE_PREFIX = "reader:notes:"
OPEN_ACTION = "reader.open"
DELETE_QA_ACTION = "reader.qa_delete"
COLLAPSE_QA_ACTION = "reader.qa_collapse"

_DEFAULT_SELECTION_KIND = "action"


class ReaderPlugin(Plugin):
    """One reader page per notes session (plus a fallback ``reader:main``)."""

    id = "reader"

    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self._lock = threading.RLock()
        self._states: dict[str, dict[str, Any]] = {}
        # page id → owner → item id → normalized menu item
        self._selection: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        # item id → normalized menu item, offered on every reader page
        self._global_selection: dict[str, dict[str, Any]] = {}

        ctx.register_service("reader", self)

        ctx.ui.add_page(
            PAGE_ID,
            "阅读",
            render=lambda: self._render_page(PAGE_ID),
            icon="📖",
            order=5,
            plugin_id=self.id,
        )
        ctx.ui.register_action(OPEN_ACTION, self._act_open, plugin_id=self.id)
        ctx.ui.register_action(
            DELETE_QA_ACTION, self._act_delete_qa, plugin_id=self.id
        )
        ctx.ui.register_action(
            COLLAPSE_QA_ACTION, self._act_collapse_qa, plugin_id=self.id
        )

    def teardown(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Service API
    # ------------------------------------------------------------------
    def show(
        self,
        *,
        title: str,
        docs: list[dict[str, str]],
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
        cleaned: list[dict[str, str]] = []
        for doc in docs:
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
            state["active"] = str(active or "") or (
                cleaned[0]["key"] if cleaned else ""
            )
            # New sequence → new client tab key, so the requested document is
            # selected on open while later user tab clicks survive refreshes.
            state["sequence"] = int(state.get("sequence", 0)) + 1

        if not self.ctx.ui.has_page(page_id):
            self.ctx.ui.add_page(
                page_id,
                str(title or "阅读"),
                render=lambda pid=page_id: self._render_page(pid),
                icon="📖",
                order=5,
            )
        else:
            self.ctx.ui.set_page_title(page_id, str(title or "阅读"))

        self.ctx.ui.refresh("reader.show")
        if open_page:
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

    # ------------------------------------------------------------------
    # Selection menu registry (per reader page)
    # ------------------------------------------------------------------
    def register_selection_action(
        self,
        page: str,
        item: Mapping[str, Any],
        *,
        owner: str = "",
    ) -> None:
        """Add one selection-menu item to a reader page (or to every page).

        *page* is a page id (``reader:...``), a page key, or empty / ``"*"``
        for an item offered on every reader page.  Per-page items are gated by
        *owner* (they only appear on documents that owner supplied); global
        items are not gated and must validate the document themselves.
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
            "kind": str(item.get("kind", "") or _DEFAULT_SELECTION_KIND),
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

    def unregister_selection_action(
        self, page: str, item_id: str, *, owner: str = ""
    ) -> None:
        text = str(page or "")
        with self._lock:
            if text in ("", "*"):
                self._global_selection.pop(item_id, None)
                return
            page_id = self._as_page_id(text)
            per_owner = self._selection.get(page_id) or {}
            bucket = per_owner.get(str(owner or ""))
            if bucket:
                bucket.pop(item_id, None)

    def selection_actions(
        self, page: str, owner: str = ""
    ) -> list[dict[str, Any]]:
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

    # ------------------------------------------------------------------
    # Q&A blocks: the reader owns the on-disk format, plugins only ask
    # ------------------------------------------------------------------
    def qa_begin(
        self,
        path: str,
        *,
        question: str,
        selection: str = "",
        intensity: str = "",
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
        self, text: str, question: str, selection: str, intensity: str
    ) -> tuple[str, dict[str, Any]]:
        qa_id = qa.new_id()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        block = qa.build_block(
            qa_id,
            ts,
            status="pending",
            question=question,
            answer="",
            intensity=intensity,
        )
        updated, anchored = qa.insert_block(text, block, selection)
        return updated, {"id": qa_id, "ts": ts, "anchored": anchored}

    def qa_finish(
        self,
        path: str,
        qa_id: str,
        *,
        answer: str,
        status: str = "done",
        collapsed: bool = True,
    ) -> bool:
        """Write the agent's answer into an existing pending block."""

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

        result = self._qa_write(path, mutate)
        return bool(result)

    def qa_set_collapsed(self, path: str, qa_id: str, collapsed: bool) -> None:
        """Persist the fold state of one Q&A block in the note file."""

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

        self._qa_write(path, mutate)

    def qa_context(
        self,
        path: str,
        qa_id: str,
        *,
        before_words: int = 500,
        after_words: int = 500,
        history_words: int = 1000,
    ) -> dict[str, Any]:
        """Collect the context the answering agent needs for one question."""

        resolved = self._known_path(path)
        if resolved is None:
            raise ValueError("未知的文稿路径")
        text = resolved.read_text("utf-8")
        blocks = qa.parse_blocks(text)
        current = next((b for b in blocks if b.id == qa_id), None)
        if current is None:
            raise ValueError("未找到这条问答")

        before = qa.tail_words(qa.strip_blocks(text[: current.start]), before_words)
        after = qa.head_words(qa.strip_blocks(text[current.end :]), after_words)

        history: list[dict[str, str]] = []
        used = 0
        for block in reversed(blocks):
            if block.id == qa_id or block.start >= current.start:
                continue
            entry = {"question": block.question, "answer": block.answer}
            size = qa.word_count(entry["question"]) + qa.word_count(entry["answer"])
            if history and used + size > history_words:
                break
            history.append(entry)
            used += size
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

    def _qa_write(self, path: str, mutate: Any) -> dict[str, Any]:
        resolved = self._known_path(path)
        if resolved is None:
            raise ValueError("未知的文稿路径")
        with self._lock:
            text = resolved.read_text("utf-8")
            updated, info = mutate(text)
            resolved.write_text(updated, "utf-8")
        return info

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
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
            text = resolved.read_text("utf-8")
        except OSError as exc:
            return {"ok": False, "message": f"读取失败：{exc}"}
        updated = qa.remove_block(text, qa_id)
        if updated is None:
            return {"ok": False, "message": "未找到这条问答"}
        try:
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
        # The browser already toggled the block; never re-render for this.
        return {"ok": True, "refresh": False}

    def _known_path(self, path: str) -> Path | None:
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

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def _render_page(self, page_id: str) -> list[dict[str, Any]]:
        with self._lock:
            state = dict(self._states.get(page_id) or {})
            docs = [dict(doc) for doc in state.get("docs") or []]
        if not docs:
            return [{"type": "text", "text": "没有可显示的文稿"}]

        title = str(state.get("title") or "")
        active = str(state.get("active") or "")
        sequence = int(state.get("sequence") or 0)

        tabs: list[dict[str, Any]] = []
        for doc in docs:
            text = _read_text(doc["path"])
            owner = str(doc.get("owner", "") or state.get("owner") or "")
            blocks = _document_blocks(text, doc["path"])
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
            *(
                [{"type": "markdown", "text": f"### {title}"}]
                if title
                else []
            ),
            {
                "type": "tabs",
                "id": f"{page_id}:{sequence}",
                "tabs": tabs,
            },
        ]


def _document_blocks(text: str, path: str) -> list[dict[str, Any]]:
    """Render one document: markdown chunks plus custom Q&A components."""

    if not text.strip():
        return [{"type": "text", "text": "（暂无内容）"}]

    blocks: list[dict[str, Any]] = []
    for kind, payload in qa.segments(text):
        if kind == "qa":
            block = payload
            assert isinstance(block, qa.QABlock)
            blocks.append(
                {
                    "type": "qa",
                    "id": block.id,
                    "path": path,
                    "ts": block.ts,
                    "status": block.status,
                    "intensity": block.intensity,
                    "collapsed": block.collapsed,
                    "question": block.question,
                    "answer": block.answer,
                    "question_html": render_markdown(block.question),
                    "answer_html": render_markdown(block.answer)
                    if block.answer
                    else "",
                    "delete_action": DELETE_QA_ACTION,
                    "collapse_action": COLLAPSE_QA_ACTION,
                }
            )
            continue
        chunk_text = str(payload)
        if not chunk_text.strip():
            continue
        blocks.extend(
            {"type": "markdown", "text": chunk}
            for chunk in split_markdown(chunk_text)
        )
    if not blocks:
        return [{"type": "text", "text": "（暂无内容）"}]
    return blocks


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text("utf-8")
    except OSError:
        return ""


def split_markdown(text: str, max_chars: int = 4000) -> list[str]:
    """Split a document into paragraph-aligned chunks for KaTeX rendering."""

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    in_display = False
    in_fence = False
    for para in paragraphs:
        if current and size >= max_chars and not in_display and not in_fence:
            chunks.append("\n\n".join(current))
            current = []
            size = 0
        current.append(para)
        size += len(para) + 2
        if para.count("$$") % 2:
            in_display = not in_display
        if para.count("```") % 2:
            in_fence = not in_fence
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text]