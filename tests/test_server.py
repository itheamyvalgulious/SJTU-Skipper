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
    route = {"page": "test.page", "class": "", "doc": "", "pages": ["test.page"]}
    response = test_client.get("/api/ui/state", params={**route, "pages": json.dumps(route["pages"])})
    assert response.status_code == 200
    state = response.json()
    assert state["left_tabs"][0]["id"] == "test"
    assert state["active_page"] == "test.page"
    assert "<h1>" in state["page_content"]["test.page"][0]["html"]

    response = test_client.post(
        "/api/ui/action", json={"action": "test.ping", "values": {"x": 1}, "navigation": route}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["message"] == "pong"


def state_for(test_client, route):
    return test_client.get(
        "/api/ui/state", params={**route, "pages": json.dumps(route["pages"])}
    )


@pytest.fixture()
def navigation_client(client):
    from types import SimpleNamespace

    test_client, cfg = client
    runtime = test_client.app.state.runtime
    sid = "lecture-test"
    (cfg.data_dir / "sessions" / sid).mkdir(parents=True)
    opened = []

    def notes(session_id, doc):
        page = f"reader:notes:{session_id}:notes"
        opened.append((page, doc))
        runtime.ui.add_page(page, "Notes")

    def logs(session_id):
        runtime.ui.add_page(f"notes:logs:{session_id}", "Logs")

    def chunk_notes(session_id, doc):
        page = f"reader:notes:{session_id}:chunk-notes"
        opened.append((page, doc))
        runtime.ui.add_page(page, "Draft notes")
        return {"ok": True}

    runtime.services.register(
        "notes",
        SimpleNamespace(open_notes=notes, open_logs=logs, open_chunk_notes=chunk_notes),
    )
    runtime.services.register(
        "input", SimpleNamespace(resolve_class=lambda key: {"course_name": key})
    )

    def reader_navigation(page):
        from core.navigation import current_navigation

        navigation = current_navigation()
        if navigation.doc not in {"knowledge", "notice", "gossip"}:
            navigation.doc = "knowledge"
        return navigation.doc

    runtime.services.register("reader", SimpleNamespace(navigation=reader_navigation))
    runtime.ui.add_page("input:session", "Class")
    return test_client, sid, opened


def test_navigation_restores_dynamic_pages(navigation_client):
    test_client, sid, _opened = navigation_client
    for page, doc in [
        ("input:session", ""),
        (f"reader:notes:{sid}:notes", "knowledge"),
        (f"reader:notes:{sid}:chunk-notes", "notice"),
        (f"notes:logs:{sid}", ""),
        ("test.page", ""),
        ("", ""),
    ]:
        route = {"page": page, "class": sid, "doc": doc, "pages": [page] if page else []}
        response = test_client.post("/api/ui/navigate", json=route)
        assert response.status_code == 200
        assert response.json()["navigation"] == route
        state = state_for(test_client, route).json()
        assert "navigation" not in state
        assert state["active_page"] == page
        assert state["open_pages"] == route["pages"]
    assert test_client.post("/api/ui/navigate", json={"page": "missing"}).status_code == 404
    assert test_client.post(
        "/api/ui/navigate", json={"page": "notes:logs:../secret"}
    ).status_code == 400


@pytest.mark.parametrize("page", ["reader:notes:{sid}:notes", "reader:notes:{sid}:chunk-notes", "notes:logs:{sid}"])
def test_notes_navigation_resolves_class(navigation_client, page):
    test_client, sid, _opened = navigation_client
    page = page.format(sid=sid)
    route = {"page": page, "class": "other-class", "doc": "notice" if page.startswith("reader:") else "", "pages": []}
    response = test_client.post("/api/ui/navigate", json=route)
    assert response.status_code == 200
    assert response.json()["navigation"] == {**route, "class": sid, "pages": [page]}


def test_navigation_does_not_retain_request_state(navigation_client):
    test_client, sid, _opened = navigation_client
    route = {"page": "input:session", "class": sid, "doc": "", "pages": ["input:session"]}
    assert test_client.post("/api/ui/navigate", json=route).status_code == 200
    home = {"page": "", "class": "", "doc": "", "pages": []}
    assert test_client.post("/api/ui/navigate", json=home).json()["navigation"] == home
    assert state_for(test_client, route).json()["active_page"] == "input:session"
    assert state_for(test_client, home).json()["open_pages"] == []
    assert state_for(test_client, route).json()["active_page"] == "input:session"


def test_navigation_registers_notes_only_once(navigation_client):
    test_client, sid, opened = navigation_client
    page = f"reader:notes:{sid}:notes"
    route = {"page": page, "class": sid, "doc": "notice", "pages": [page]}
    for _ in range(2):
        assert test_client.post("/api/ui/navigate", json=route).json()["navigation"] == route
        assert state_for(test_client, route).json()["active_page"] == page
    assert opened == [(page, "notice")]


def test_open_close_page(client) -> None:
    test_client, _cfg = client
    route = {"page": "test.page", "class": "", "doc": "", "pages": ["test.page"]}
    response = test_client.post("/api/ui/pages/test.page/close", json={"navigation": route})
    assert response.status_code == 200
    closed = response.json()["navigation"]
    assert closed == {"page": "", "class": "", "doc": "", "pages": []}
    assert state_for(test_client, closed).json()["active_page"] == ""
    assert state_for(test_client, route).json()["active_page"] == "test.page"
    response = test_client.post("/api/ui/pages/test.page/open", json={"navigation": closed})
    assert response.status_code == 200
    assert response.json()["navigation"] == route
    assert state_for(test_client, closed).json()["open_pages"] == []


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
                "navigation": {"page": "", "class": "", "doc": "", "pages": []},
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is True, payload
        stored = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
        assert stored["llm.model"] == "deepseek-flash"
        assert stored["llm.temperature"] == 0.3
