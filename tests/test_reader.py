from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from config import load_config
from core import note_document as qa
from core.runtime import build_runtime
from core.ui import BlockType
from plugins.reader import ReaderPlugin


def _make_runtime(tmp_path: Path):
    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(cfg, plugins=[ReaderPlugin()])
    runtime.start()
    return runtime


def _write(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_show_opens_reader_page(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    try:
        docs = [
            {
                "key": "knowledge",
                "label": "知识",
                "path": _write(tmp_path, "k.md", "# 知识\n\n## 定义\n\n$a^2$\n"),
            },
            {
                "key": "gossip",
                "label": "闲话",
                "path": _write(tmp_path, "g.md", "# 闲话\n\n聊天。\n"),
            },
        ]
        reader = runtime.services.require("reader")
        reader.show(title="笔记", docs=docs, active="gossip")

        state = runtime.ui.state()
        assert state["active_page"] == "reader:main"
        blocks = state["page_content"]["reader:main"]
        tabs = next(block for block in blocks if block["type"] == BlockType.TABS)
        assert [tab["title"] for tab in tabs["tabs"]] == ["知识", "闲话"]
        assert tabs["tabs"][0]["active"] is False
        assert tabs["tabs"][1]["active"] is True
        markdown = [
            b
            for tab in tabs["tabs"]
            for b in tab["blocks"]
            if b["type"] == BlockType.MARKDOWN
        ]
        assert markdown and "定义" in markdown[0]["html"]
    finally:
        runtime.stop()


def test_document_selection_is_request_scoped(tmp_path: Path, navigation) -> None:
    from core.navigation import Navigation, navigation_scope

    runtime = _make_runtime(tmp_path)
    try:
        reader = runtime.services.require("reader")
        page = reader.show(
            title="Notes",
            docs=[
                {
                    "key": "first",
                    "label": "First",
                    "path": _write(tmp_path, "first.md", "First"),
                },
                {
                    "key": "second",
                    "label": "Second",
                    "path": _write(tmp_path, "second.md", "Second"),
                },
            ],
            active="first",
        )
        assert navigation.doc == "first"
        with navigation_scope(Navigation(page=page, doc="second", pages=[page])):
            tabs = runtime.ui.state()["page_content"][page][0]["tabs"]
            assert [tab["active"] for tab in tabs] == [False, True]
        tabs = runtime.ui.state()["page_content"][page][0]["tabs"]
        assert [tab["active"] for tab in tabs] == [True, False]
    finally:
        runtime.stop()


def test_reader_action_rejects_empty_docs(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    try:
        result = runtime.ui.dispatch_action("reader.open", {"docs": []})
        assert not result.ok
    finally:
        runtime.stop()


def test_reader_action_opens_page(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    try:
        docs = [
            {"key": "k", "label": "知识", "path": _write(tmp_path, "k.md", "# hi\n")}
        ]
        result = runtime.ui.dispatch_action("reader.open", {"title": "T", "docs": docs})
        assert result.ok
        assert result.open_page == "reader:main"
    finally:
        runtime.stop()


class _FootnoteLinks(HTMLParser):
    def __init__(self, html: str) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.targets: list[str] = []
        self.feed(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.append(str(attributes["id"]))
        href = attributes.get("href") or ""
        if tag == "a" and href.startswith("#fn"):
            self.targets.append(href[1:])


def test_document_footnotes_survive_length_and_qa(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    try:
        formula = r"$f_{X_1}(x_1)=e^{-x_1^2/2}$"
        display = "$$\nf(x_1,x_2)=f_1*f_2\n\n+g_1*g_2\n$$"
        text = (
            "# Normal marginal counterexample\n\n"
            f"The marginal is {formula}[^normal-counterexample].\n\n"
            + "Probability context paragraph.\n\n" * 180
            + "The joint density is not normal[^normal-counterexample].\n\n"
            + display
            + "\n\n"
            + "[^normal-counterexample]: Integrate the density with $x_1*x_2$.\n"
            + '    <script>alert(1)</script><b onclick="alert(2)">Checked</b>\n'
        )
        path = _write(tmp_path, "normal.md", text)
        second_path = _write(
            tmp_path,
            "other.md",
            "Another result[^normal-counterexample].\n\n"
            "[^normal-counterexample]: A separate document.\n",
        )
        reader = runtime.services.require("reader")
        reader.show(
            title="Probability",
            docs=[
                {"key": "normal", "label": "Normal", "path": path},
                {"key": "other", "label": "Other", "path": second_path},
            ],
        )

        def rendered() -> tuple[list[dict], str]:
            blocks = runtime.ui.state()["page_content"]["reader:main"]
            tabs = next(block for block in blocks if block["type"] == BlockType.TABS)[
                "tabs"
            ]
            note_blocks = tabs[0]["blocks"]
            html = "".join(
                block["html"]
                for tab in tabs
                for block in tab["blocks"]
                if block["type"] == BlockType.MARKDOWN
            )
            assert "[^normal-counterexample]" not in html
            assert formula in html and "$x_1*x_2$" in html
            assert "$$\nf(x_1,x_2)=f_1*f_2\n+g_1*g_2\n$$" in html
            assert "<em>" not in html
            assert "<script" not in html and "onclick" not in html
            return note_blocks, html

        _, html = rendered()
        links = _FootnoteLinks(html)
        assert len(links.targets) == 6
        assert len(links.ids) == len(set(links.ids))
        assert all(target in links.ids for target in links.targets)

        created = reader.qa_begin(
            path,
            question="Why does this work?[^normal-counterexample]\n\n"
            "[^normal-counterexample]: Question context.",
            selection="The marginal is",
        )
        assert created["anchored"]
        reader.qa_finish(
            path,
            created["id"],
            answer="Integrate $x_1*x_2$[^normal-counterexample].\n\n"
            "[^normal-counterexample]: Answer context.",
        )
        blocks, html = rendered()
        question = next(block for block in blocks if block["type"] == BlockType.QA)
        assert question["path"] == path and question["id"] == created["id"]
        assert question["delete_action"] == "reader.qa_delete"
        assert question["collapse_action"] == "reader.qa_collapse"
        assert [block["type"] for block in blocks] == ["markdown", "qa", "markdown"]
        assert "footnote-ref" in blocks[0]["html"]
        assert "footnote-item" in blocks[2]["html"]
        assert "qa:begin" not in html and "reader-qa-" not in html
        assert qa.parse_blocks(Path(path).read_text("utf-8"))[0].question == (
            "Why does this work?[^normal-counterexample]\n\n"
            "[^normal-counterexample]: Question context."
        )
        links = _FootnoteLinks(html + question["answer_html"])
        assert len(links.targets) == 8
        assert len(links.ids) == len(set(links.ids))
        assert all(target in links.ids for target in links.targets)

        result = runtime.ui.dispatch_action(
            question["delete_action"], {"path": path, "id": created["id"]}
        )
        assert result.ok
        blocks, _ = rendered()
        assert not any(block["type"] == BlockType.QA for block in blocks)
        assert Path(path).read_text("utf-8").strip() == text.strip()
    finally:
        runtime.stop()
