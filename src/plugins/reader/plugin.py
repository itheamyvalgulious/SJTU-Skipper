"""Reader plugin: display markdown documents (lecture notes) in a shared page.

Notes plugins (summarize / instant-summarize) own their files; they hand the
reader a list of ``{key, label, path}`` documents and the reader renders them
with tabs, KaTeX and per-document scrolling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.plugin import Plugin, PluginContext

PAGE_ID = "reader:main"
OPEN_ACTION = "reader.open"


class ReaderPlugin(Plugin):
    """A single reusable document reader page."""

    id = "reader"

    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self._title = ""
        self._docs: list[dict[str, str]] = []
        self._active = ""
        self._sequence = 0

        ctx.register_service("reader", self)

        ctx.ui.add_page(
            PAGE_ID,
            "阅读",
            render=self._render,
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
        open_page: bool = True,
    ) -> None:
        """Display *docs* (``{key,label,path}``) in the reader page."""

        cleaned: list[dict[str, str]] = []
        for doc in docs:
            path = str(doc.get("path", "") or "")
            key = str(doc.get("key", "") or Path(path).stem)
            label = str(doc.get("label", "") or key)
            if not path:
                continue
            cleaned.append({"key": key, "label": label, "path": path})
        self._title = str(title or "")
        self._docs = cleaned
        self._active = str(active or "") or (cleaned[0]["key"] if cleaned else "")
        # New sequence → new client tab key, so the requested document is
        # selected on open while later user tab clicks survive refreshes.
        self._sequence += 1
        self.ctx.ui.refresh("reader.show")
        if open_page:
            self.ctx.ui.open_page(PAGE_ID)

    def clear(self) -> None:
        self._title = ""
        self._docs = []
        self._active = ""

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------
    def _act_open(self, payload: dict[str, Any]) -> dict[str, Any]:
        title = str(payload.get("title", "") or self._title)
        docs = payload.get("docs")
        if not isinstance(docs, list) or not docs:
            return {"ok": False, "message": "没有可显示的文稿"}
        self.show(
            title=title,
            docs=[dict(doc) for doc in docs if isinstance(doc, dict)],
            active=str(payload.get("active", "") or ""),
        )
        return {"ok": True, "open_page": PAGE_ID}

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def _render(self) -> list[dict[str, Any]]:
        if not self._docs:
            return [{"type": "text", "text": "没有可显示的文稿"}]

        tabs: list[dict[str, Any]] = []
        for doc in self._docs:
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
                    "active": doc["key"] == self._active,
                    "blocks": blocks,
                }
            )

        return [
            *(
                [{"type": "markdown", "text": f"### {self._title}"}]
                if self._title
                else []
            ),
            {
                "type": "tabs",
                "id": f"{PAGE_ID}:{self._sequence}",
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
