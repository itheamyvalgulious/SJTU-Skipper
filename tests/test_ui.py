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
