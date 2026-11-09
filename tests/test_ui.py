from __future__ import annotations

import pytest

from core.ui import ActionResult, UIManager


def test_tabs_items_pages_and_state() -> None:
    changes: list[str] = []
    ui = UIManager(on_change=changes.append)
    ui.add_tab("left", "courses", "课程", order=1)
    ui.add_toolbox_item("courses", "list", "课程列表", render=lambda: [{"type": "text", "text": "hi"}])
    ui.add_page("home", "首页", render=lambda: "# Hello\n\n$E=mc^2$", order=1)
    ui.open_page("home")
    state = ui.state()
    assert [t["id"] for t in state["left_tabs"]] == ["courses"]
    assert state["left_tabs"][0]["toolbox"][0]["id"] == "list"
    assert state["active_page"] == "home"
    assert "home" in state["page_content"]
    html = state["page_content"]["home"][0]["html"]
    assert "<h1>" in html
    assert "$E=mc^2$" in html
    assert changes


def test_action_dispatch_and_open_page() -> None:
    ui = UIManager()
    ui.add_page("home", "首页", render=lambda: "home")
    ui.add_page("detail", "详情", render=lambda: "detail")

    def handler(payload):
        assert payload["value"] == 3
        assert payload["page_id"] == "home"
        return {"ok": True, "message": "ok", "open_page": "detail"}

    ui.register_action("open.detail", handler, plugin_id="test")
    result = ui.dispatch_action("open.detail", {"value": 3}, page_id="home")
    assert isinstance(result, ActionResult)
    assert result.ok
    assert ui.state()["active_page"] == "detail"


def test_action_failure_returns_error() -> None:
    ui = UIManager()

    def bad(payload):
        raise RuntimeError("nope")

    ui.register_action("bad", bad)
    result = ui.dispatch_action("bad")
    assert not result.ok
    assert "nope" in result.message


def test_unknown_action() -> None:
    ui = UIManager()
    result = ui.dispatch_action("missing")
    assert not result.ok
    assert "未知" in result.message


def test_render_error_is_contained() -> None:
    ui = UIManager()
    ui.add_page("broken", "坏页", render=lambda: (_ for _ in ()).throw(RuntimeError("x")))
    ui.open_page("broken")
    state = ui.state()
    block = state["page_content"]["broken"][0]
    assert "渲染失败" in block["html"]


def test_duplicate_tab_page_item_rejected() -> None:
    ui = UIManager()
    ui.add_tab("left", "a", "A")
    with pytest.raises(ValueError):
        ui.add_tab("left", "a", "A")
    ui.add_page("p", "P")
    with pytest.raises(ValueError):
        ui.add_page("p", "P")
    ui.add_toolbox_item("a", "i", "I")
    with pytest.raises(ValueError):
        ui.add_toolbox_item("a", "i", "I")


def test_close_page_and_tab_removal() -> None:
    ui = UIManager()
    ui.add_tab("left", "a", "A")
    ui.add_toolbox_item("a", "i", "I")
    ui.add_page("p", "P")
    ui.open_page("p")
    ui.close_page("p")
    assert ui.state()["active_page"] is None
    ui.remove_tab("a")
    assert ui.state()["left_tabs"] == []
    assert ui.state()["pages"][0]["id"] == "p"


def test_status_provider() -> None:
    ui = UIManager()
    ui.set_status_provider(lambda: {"session": {"id": "x"}})
    assert ui.state()["status"]["session"]["id"] == "x"


def test_tree_block_is_normalized_recursively() -> None:
    ui = UIManager()
    ui.add_page(
        "p",
        "P",
        render=lambda: [
            {
                "type": "tree",
                "id": "courses",
                "nodes": [
                    {
                        "id": "c1",
                        "label": "高等数学",
                        "expanded": True,
                        "toggle": {"action": "t", "value": {"course": "1"}},
                        "children": [
                            {"id": "k1", "label": "11-05 08:00", "action": {"action": "open"}},
                        ],
                    }
                ],
            }
        ],
    )
    ui.open_page("p")
    block = ui.state()["page_content"]["p"][0]
    assert block["type"] == "tree"
    node = block["nodes"][0]
    assert node["depth"] == 0
    assert node["children"][0]["depth"] == 1
    assert node["children"][0]["children"] == []


def test_commands_registry_and_state() -> None:
    ui = UIManager()
    ui.add_page("settings:main", "设置")
    ui.register_command(
        "settings.open",
        "设置",
        category="视图",
        order=10,
        page_id="settings:main",
        keywords="config",
        plugin_id="settings",
    )
    ui.register_command(
        "input.refresh",
        "刷新课程",
        category="课程",
        order=0,
        action="input.refresh_courses",
        plugin_id="input",
    )
    commands = ui.state()["commands"]
    assert [c["id"] for c in commands] == ["input.refresh", "settings.open"]
    assert commands[1]["page"] == "settings:main"
    assert commands[0]["action"] == "input.refresh_courses"
    assert "settings.open" in ui.snapshot()["commands"]

    ui.unregister_command("input.refresh")
    assert [c["id"] for c in ui.state()["commands"]] == ["settings.open"]

    ui.remove_plugin_commands("settings")
    assert ui.state()["commands"] == []


def test_command_requires_page_or_action() -> None:
    ui = UIManager()
    with pytest.raises(ValueError):
        ui.register_command("bad", "Bad")


class TestMathRendering:
    def test_inline_math_survives_markdown_emphasis(self) -> None:
        from core.ui import render_markdown

        html = render_markdown("设 $f_1+f_2$ 与 $g$ 的卷积 $f*g$ 存在。")
        assert "<em>" not in html
        assert "$f_1+f_2$" in html
        assert "$f*g$" in html

    def test_inline_display_math_becomes_block(self) -> None:
        from core.ui import render_markdown

        html = render_markdown("性质 1 $$ a_1 = b_2 $$ 由换元可得。")
        assert "<em>" not in html
        assert "<p>$$\na_1 = b_2\n$$</p>" in html
        assert "由换元可得" in html

    def test_blank_lines_inside_display_math_are_removed(self) -> None:
        from core.ui import render_markdown

        html = render_markdown("$$\nA\n\nB\n$$")
        assert "<p>$$\nA\nB\n$$</p>" in html

    def test_code_spans_are_not_touched(self) -> None:
        from core.ui import render_markdown

        html = render_markdown("代码 `$x$` 不动")
        assert "<code>$x$</code>" in html
