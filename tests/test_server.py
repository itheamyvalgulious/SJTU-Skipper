from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient

from config import load_config
from core.plugin import Plugin, PluginContext
from core.runtime import build_runtime
from core.ui import BlockType, Side
from plugins.llm_core.plugin import LLMPlugin
from plugins.settings.plugin import SettingsPlugin
from ui.server import WebSocketMessageType, create_app

from support import FakeLLM


class DemoUI(Plugin):
    id = "test"

    def setup(self, ctx: PluginContext) -> None:
        ctx.ui.add_tab(Side.LEFT, "test", "测试", icon="◈")
        ctx.ui.add_toolbox_item(
            "test",
            "test.item",
            "工具项",
            render=lambda: [{"type": BlockType.TEXT, "text": "tool"}],
        )
        ctx.ui.add_page("test.page", "测试页", render=lambda: "# 页面\n\nhello")
        ctx.ui.open_page("test.page")
        ctx.ui.register_action(
            "test.ping",
            lambda payload: {"ok": True, "message": "pong"},
            plugin_id=self.id,
        )


@pytest.fixture()
def client(tmp_path: Path):
    (tmp_path / "auth.json").write_text(
        json.dumps({"users": {"test": {"password": "test"}}}), "utf-8"
    )
    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(cfg, plugins=[DemoUI()], llm=FakeLLM())
    app = create_app(runtime)
    with TestClient(app) as test_client:
        login = test_client.post(
            "/login",
            data={"username": "test", "password": "test"},
            follow_redirects=False,
        )
        assert login.status_code == 303
        yield test_client, cfg


def test_state_and_action(client) -> None:
    test_client, _cfg = client
    response = test_client.get("/api/ui/state")
    assert response.status_code == 200
    state = response.json()
    assert state["left_tabs"][0]["id"] == "test"
    assert state["active_page"] == "test.page"
    assert "<h1>" in state["page_content"]["test.page"][0]["html"]

    response = test_client.post(
        "/api/ui/action", json={"action": "test.ping", "values": {"x": 1}}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["message"] == "pong"


@pytest.fixture()
def navigation_client(client):
    from types import SimpleNamespace

    test_client, cfg = client
    runtime = test_client.app.state.runtime
    sid = "lecture-test"
    (cfg.data_dir / "sessions" / sid).mkdir(parents=True)
    selected = {"class_id": "", "opened": []}

    def notes(session_id, doc):
        selected.update(session=session_id, doc=doc)
        page = f"reader:notes:{session_id}:notes"
        if not runtime.ui.has_page(page):
            runtime.ui.add_page(page, "Notes")

    def logs(session_id):
        selected["session"] = session_id
        page = f"notes:logs:{session_id}"
        if not runtime.ui.has_page(page):
            runtime.ui.add_page(page, "Logs")

    def chunk_notes(session_id, doc):
        selected.update(session=session_id, doc=doc)
        page = f"reader:notes:{session_id}:chunk-notes"
        if not runtime.ui.has_page(page):
            runtime.ui.add_page(page, "Draft notes")
        return {"ok": True}

    runtime.services.register(
        "notes",
        SimpleNamespace(open_notes=notes, open_logs=logs, open_chunk_notes=chunk_notes),
    )
    runtime.services.register(
        "reader",
        SimpleNamespace(
            navigation=lambda page: selected["doc"],
            select_document=lambda page, key: selected.update(doc=key),
        ),
    )

    def open_class(key):
        selected["class_id"] = key
        selected["opened"].append(key)

    runtime.services.register(
        "input",
        SimpleNamespace(
            open_class=open_class,
            clear_class=lambda: selected.update(class_id=""),
            selected_class=lambda: selected["class_id"],
        ),
    )
    runtime.ui.add_page("input:session", "Class")
    return test_client, sid, selected


def test_navigation_restores_dynamic_pages(navigation_client):
    test_client, sid, selected = navigation_client
    for route in [
        {"page": "input:session", "class": sid},
        {"page": f"reader:notes:{sid}:notes", "class": sid, "doc": "knowledge"},
        {
            "page": f"reader:notes:{sid}:chunk-notes",
            "class": sid,
            "doc": "notice",
        },
        {"page": f"notes:logs:{sid}", "class": sid},
        {"page": "test.page", "class": sid},
        {"page": "", "class": sid},
    ]:
        selected["class_id"] = ""
        response = test_client.post("/api/ui/navigate", json=route)
        assert response.status_code == 200
        assert response.json()["navigation"] == route
        assert test_client.get("/api/ui/state").json()["navigation"] == route
        assert selected["class_id"] == sid
    assert (
        test_client.post("/api/ui/navigate", json={"page": "missing"}).status_code
        == 404
    )
    assert (
        test_client.post(
            "/api/ui/navigate", json={"page": "notes:logs:../secret"}
        ).status_code
        == 400
    )


@pytest.mark.parametrize(
    "page",
    [
        "reader:notes:{sid}:notes",
        "reader:notes:{sid}:chunk-notes",
        "notes:logs:{sid}",
    ],
)
def test_notes_navigation_selects_class(navigation_client, page):
    test_client, sid, selected = navigation_client
    page = page.format(sid=sid)
    selected["class_id"] = "other-class"
    route = {"page": page}
    if page.startswith("reader:"):
        route["doc"] = "notice"
    response = test_client.post("/api/ui/navigate", json=route)
    assert response.status_code == 200
    assert response.json()["navigation"] == {**route, "class": sid}
    assert selected["class_id"] == sid
    assert test_client.get("/api/ui/state").json()["navigation"] == {
        **route,
        "class": sid,
    }


@pytest.mark.parametrize("page", ["", "test.page", "reader:notes:{sid}:notes"])
def test_navigation_clears_explicit_class(navigation_client, page):
    test_client, sid, selected = navigation_client
    selected["class_id"] = sid
    response = test_client.post(
        "/api/ui/navigate", json={"page": page.format(sid=sid), "class": ""}
    )
    assert response.status_code == 200
    assert response.json()["navigation"]["class"] == ""
    assert test_client.get("/api/ui/state").json()["navigation"]["class"] == ""
    assert selected["class_id"] == ""


@pytest.mark.parametrize("page", ["", "test.page"])
def test_navigation_preserves_missing_class(navigation_client, page):
    test_client, sid, selected = navigation_client
    selected["class_id"] = sid
    response = test_client.post("/api/ui/navigate", json={"page": page})
    assert response.status_code == 200
    assert response.json()["navigation"] == {"page": page, "class": sid}
    assert selected["opened"] == []


def test_state_does_not_reopen_class(navigation_client):
    test_client, sid, selected = navigation_client
    selected["class_id"] = sid
    for _ in range(3):
        response = test_client.get("/api/ui/state")
        assert response.status_code == 200
        assert response.json()["navigation"]["class"] == sid
    assert selected["opened"] == []


def test_navigation_restores_notes_after_reset(navigation_client):
    test_client, sid, selected = navigation_client
    route = {
        "page": f"reader:notes:{sid}:notes",
        "class": sid,
        "doc": "notice",
    }
    runtime = test_client.app.state.runtime
    for _ in range(2):
        runtime.ui.show_home()
        selected.update(class_id="", doc="knowledge")
        response = test_client.post("/api/ui/navigate", json=route)
        assert response.status_code == 200
        assert response.json()["navigation"] == route
        assert test_client.get("/api/ui/state").json()["navigation"] == route
        assert selected["session"] == sid


def test_root_navigation_clears_hub_selection(client):
    test_client, _cfg = client
    runtime = test_client.app.state.runtime
    runtime.hub.select_class("stored-class", {"course_name": "Stored course"})
    assert test_client.get("/api/ui/state").json()["navigation"] == {
        "page": "test.page",
        "class": "stored-class",
    }
    response = test_client.post("/api/ui/navigate", json={"page": "", "class": ""})
    assert response.status_code == 200
    assert response.json()["navigation"] == {"page": "", "class": ""}
    assert test_client.get("/api/ui/state").json()["navigation"] == {
        "page": "",
        "class": "",
    }
    assert runtime.hub.selected_class_id == ""


def test_open_close_page(client) -> None:
    test_client, _cfg = client
    response = test_client.post("/api/ui/pages/test.page/close")
    assert response.status_code == 200
    state = test_client.get("/api/ui/state").json()
    assert state["active_page"] is None
    response = test_client.post("/api/ui/pages/test.page/open")
    assert response.status_code == 200
    state = test_client.get("/api/ui/state").json()
    assert state["active_page"] == "test.page"


def test_artifact_is_session_scoped(client) -> None:
    test_client, cfg = client
    session_dir = cfg.sessions_dir / "s1" / "notes"
    session_dir.mkdir(parents=True)
    (session_dir / "knowledge.md").write_text("# K", encoding="utf-8")
    response = test_client.get("/api/artifacts/s1/notes/knowledge.md")
    assert response.status_code == 200
    assert response.text == "# K"

    assert test_client.get("/api/artifacts/../settings.json").status_code == 404
    assert test_client.get("/api/artifacts/s1/../../settings.json").status_code == 404
    assert test_client.get("/api/artifacts/s1/missing.md").status_code == 404


def test_socket_message_serialization() -> None:
    assert json.loads(json.dumps({"type": WebSocketMessageType.REFRESH})) == {
        "type": "refresh"
    }
    assert json.loads(json.dumps({"type": WebSocketMessageType.NOTIFY})) == {
        "type": "notify"
    }


def test_websocket_sends_refresh(client) -> None:
    test_client, _cfg = client
    with test_client.websocket_connect("/ws") as ws:
        data = ws.receive_json()
        assert data["type"] == "refresh"


def test_websocket_forwards_ui_notify(client) -> None:
    test_client, _cfg = client
    runtime = test_client.app.state.runtime
    with test_client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "refresh"
        runtime.ui.notify("qa.live", {"id": "qa-1", "phase": "thinking"})
        data = ws.receive_json()
        assert data == {
            "type": "notify",
            "topic": "qa.live",
            "payload": {"id": "qa-1", "phase": "thinking"},
        }


def test_media_relay_rejects_foreign_hosts(client) -> None:
    test_client, _cfg = client
    assert (
        test_client.get(
            "/api/media", params={"src": "https://example.com/a.mp4"}
        ).status_code
        == 400
    )
    assert (
        test_client.get(
            "/api/media", params={"src": "ftp://live.sjtu.edu.cn/a.mp4"}
        ).status_code
        == 400
    )


def test_media_relay_forwards_range(tmp_path: Path) -> None:

    (tmp_path / "auth.json").write_text(
        json.dumps({"users": {"test": {"password": "test"}}}), "utf-8"
    )
    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(cfg, plugins=[DemoUI()], llm=FakeLLM())
    app = create_app(runtime)
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["range"] = request.headers.get("range")
        seen["url"] = str(request.url)
        return httpx2.Response(
            206,
            headers={
                "content-type": "video/mp4",
                "content-range": "bytes 0-3/100",
                "content-length": "4",
            },
            content=b"JUNK",
        )

    with TestClient(app) as test_client:
        test_client.post(
            "/login",
            data={"username": "test", "password": "test"},
            follow_redirects=False,
        )
        test_client.app.state.media_client = httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler)
        )
        response = test_client.get(
            "/api/media",
            params={"src": "https://live.sjtu.edu.cn/record/a.mp4"},
            headers={"Range": "bytes=0-3"},
        )
    assert response.status_code == 206
    assert response.content == b"JUNK"
    assert response.headers["content-range"] == "bytes 0-3/100"
    assert seen["range"] == "bytes=0-3"
    assert seen["url"] == "https://live.sjtu.edu.cn/record/a.mp4"


def test_health(client) -> None:
    test_client, _cfg = client
    payload = test_client.get("/api/health").json()
    assert payload["ok"] is True
    assert payload["plugin_order"] == ["test"]


def test_settings_save_accepts_http_flat_payload(tmp_path: Path) -> None:
    (tmp_path / "auth.json").write_text(
        json.dumps({"users": {"test": {"password": "test"}}}), "utf-8"
    )
    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(cfg, plugins=[SettingsPlugin(), LLMPlugin()])
    app = create_app(runtime)
    with TestClient(app) as test_client:
        test_client.post(
            "/login",
            data={"username": "test", "password": "test"},
            follow_redirects=False,
        )
        response = test_client.post(
            "/api/ui/action",
            json={
                "action": "settings.save",
                "values": {"llm.model": "deepseek-flash", "llm.temperature": 0.3},
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is True, payload
        stored = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
        assert stored["llm.model"] == "deepseek-flash"
        assert stored["llm.temperature"] == 0.3
