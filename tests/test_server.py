from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from config import load_config
from core.plugin import Plugin, PluginContext
from core.runtime import build_runtime
from plugins.llm_core.plugin import LLMPlugin
from plugins.settings.plugin import SettingsPlugin
from ui.server import create_app

from support import FakeLLM


class DemoUI(Plugin):
    id = "test-ui"

    def setup(self, ctx: PluginContext) -> None:
        ctx.ui.add_tab("left", "test", "测试", icon="◈")
        ctx.ui.add_toolbox_item(
            "test",
            "test.item",
            "工具项",
            render=lambda: [{"type": "text", "text": "tool"}],
        )
        ctx.ui.add_page("test.page", "测试页", render=lambda: "# 页面\n\nhello")
        ctx.ui.open_page("test.page")
        ctx.ui.register_action("test.ping", lambda payload: {"ok": True, "message": "pong"}, plugin_id=self.id)


@pytest.fixture()
def client(tmp_path: Path):
    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(cfg, plugins=[DemoUI()], llm=FakeLLM())
    app = create_app(runtime)
    with TestClient(app) as test_client:
        yield test_client, cfg


def test_state_and_action(client) -> None:
    test_client, _cfg = client
    response = test_client.get("/api/ui/state")
    assert response.status_code == 200
    state = response.json()
    assert state["left_tabs"][0]["id"] == "test"
    assert state["active_page"] == "test.page"
    assert "<h1>" in state["page_content"]["test.page"][0]["html"]

    response = test_client.post("/api/ui/action", json={"action": "test.ping", "values": {"x": 1}})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["message"] == "pong"


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


def test_websocket_sends_refresh(client) -> None:
    test_client, _cfg = client
    with test_client.websocket_connect("/ws") as ws:
        data = ws.receive_json()
        assert data["type"] == "refresh"


def test_media_relay_rejects_foreign_hosts(client) -> None:
    test_client, _cfg = client
    assert test_client.get("/api/media", params={"src": "https://example.com/a.mp4"}).status_code == 400
    assert test_client.get("/api/media", params={"src": "ftp://live.sjtu.edu.cn/a.mp4"}).status_code == 400


def test_media_relay_forwards_range(tmp_path: Path) -> None:
    """The relay streams the upstream 206 and passes the byte range through."""

    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(cfg, plugins=[DemoUI()], llm=FakeLLM())
    app = create_app(runtime)
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["range"] = request.headers.get("range")
        seen["url"] = str(request.url)
        return httpx.Response(
            206,
            headers={
                "content-type": "video/mp4",
                "content-range": "bytes 0-3/100",
                "content-length": "4",
            },
            content=b"JUNK",
        )

    with TestClient(app) as test_client:
        test_client.app.state.media_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
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
    assert payload["plugin_order"] == ["test-ui"]


def test_settings_save_accepts_http_flat_payload(tmp_path: Path) -> None:
    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(cfg, plugins=[SettingsPlugin(), LLMPlugin()])
    app = create_app(runtime)
    with TestClient(app) as test_client:
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
