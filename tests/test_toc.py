"""Markdown heading anchors + table of contents helper tests."""

from __future__ import annotations

from core.ui import markdown_toc, render_markdown, render_toc


def test_headings_get_stable_ids() -> None:
    html = render_markdown("# 标题一\n\n## 小节 A\n")
    assert 'id="标题一"' in html
    assert 'id="小节-a"' in html


def test_markdown_toc_lists_h1_to_h3() -> None:
    items = markdown_toc("# 一\n\n## 二\n\n### 三\n\n#### 四\n")
    assert [item["level"] for item in items] == [1, 2, 3]
    assert [item["text"] for item in items] == ["一", "二", "三"]


def test_render_toc_links_and_skip_short_docs() -> None:
    html = render_toc("# 一\n\n## 二\n")
    assert "doc-toc" in html and "doc-toc-link" in html
    assert render_toc("# only one heading\n") == ""
