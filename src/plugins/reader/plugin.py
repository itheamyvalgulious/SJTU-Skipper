"""Reader plugin: display markdown documents (lecture notes) in reader pages.

Notes plugins (summarize / instant-summarize) own their files; they hand the
reader a list of ``{key, label, path}`` documents.  Every notes session gets
its **own page** (``reader:notes:<session_id>``), so several courses' handouts
can stay open at the same time instead of sharing one page.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.plugin import Plugin, PluginContext

PAGE_ID = "reader:main"
PAGE_PREFIX = "reader:notes:"
OPEN_ACTION = "reader.open"


class ReaderPlugin(Plugin):
    """One reader page per notes session (plus a fallback ``reader:main``)."""

    id = "reader"

    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self._states: dict[str, dict[str, Any]] = {}

        ctx.register_service("reader", self)

        ctx.ui.add_page(
            PAGE_ID,
            "阅读",
            render=lambda: self._render_page(PAGE_ID),
            icon="📖",
            order=5,
        )
        ctx.ui.register_action(OPEN_ACTION, self._act_open, plugin_id=self.id)

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
    ) -> str:
        """Display *docs* in the page for *page_key*; returns the page id."""

        page_id = self._page_id(page_key)
        cleaned: list[dict[str, str]] = []
        for doc in docs:
            path = str(doc.get("path", "") or "")
            key = str(doc.get("key", "") or Path(path).stem)
            label = str(doc.get("label", "") or key)
            if not path:
                continue
            cleaned.append({"key": key, "label": label, "path": path})

        state = self._states.setdefault(page_id, {})
        state["title"] = str(title or "")
        state["docs"] = cleaned
        state["active"] = str(active or "") or (cleaned[0]["key"] if cleaned else "")
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
        self._states.pop(page_id, None)

    @staticmethod
    def page_id_for(page_key: str) -> str:
        return f"{PAGE_PREFIX}{page_key}" if page_key else PAGE_ID

    def _page_id(self, page_key: str) -> str:
        return self.page_id_for(str(page_key or ""))

    # ------------------------------------------------------------------
    # Action
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
        )
        return {"ok": True, "open_page": page_id}

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def _render_page(self, page_id: str) -> list[dict[str, Any]]:
        state = self._states.get(page_id) or {}
        docs = state.get("docs") or []
        if not docs:
            return [{"type": "text", "text": "没有可显示的文稿"}]

        title = str(state.get("title") or "")
        active = str(state.get("active") or "")
        sequence = int(state.get("sequence") or 0)

        tabs: list[dict[str, Any]] = []
        for doc in docs:
            text = _read_text(doc["path"])
            blocks: list[dict[str, Any]] = []
            if not text.strip():
                blocks.append({"type": "text", "text": "（暂无内容）"})
            else:
                blocks.extend(
                    {"type": "markdown", "text": chunk}
                    for chunk in split_markdown(text)
                )
            tabs.append(
                {
                    "title": doc["label"],
                    "active": doc["key"] == active,
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
