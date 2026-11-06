"""Tests for the Google Cloud Speech-to-Text v2 STT plugin.

Network access is faked with ``httpx.MockTransport``; no real Google request
is ever made.  Credential tests use a locally generated RSA key.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from core.events import EventBus
from core.meta import MetaRegistry
from core.plugin import PluginContext
from core.services import ServiceRegistry
from core.streams import StreamHub
from core.ui import UIManager
from plugins.stt import STTPlugin, STTService, SttPlugin
from plugins.stt.credentials import GoogleTokenProvider
from plugins.stt.errors import SttError
from plugins.stt.google_cloud import (
    GoogleSpeechClient,
    SpeechConfig,
)
from plugins.stt.plugin import PAGE_ID
from tests.support import FakeSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ctx(
    tmp_path: Path,
    *,
    values: dict[str, Any] | None = None,
) -> PluginContext:
    bus = EventBus()
    bus.start()
    hub = StreamHub(emit=bus.emit, audio_emit_interval=100.0)
    services = ServiceRegistry()
    services.register(
        "settings", FakeSettings(values or {}), plugin_id="<test>"
    )
    return PluginContext(
        plugin_id="stt",
        data_dir=tmp_path,
        events=bus,
        streams=hub,
        llm=None,
        ui=UIManager(),
        services=services,
        meta=MetaRegistry(),
        logger=logging.getLogger("test_stt"),
    )


def disabled_values(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {"stt.enabled": False}
    values.update(overrides)
    return values


def speech_response(*texts: str) -> dict[str, Any]:
    results = []
    for index, text in enumerate(texts, start=1):
        results.append(
            {
                "alternatives": [{"transcript": text, "confidence": 0.9}],
                "resultEndOffset": f"{index}.500s",
                "languageCode": "cmn-hans-cn",
            }
        )
    return {"results": results}


def make_client(handler: Any) -> GoogleSpeechClient:
    return GoogleSpeechClient(
        SpeechConfig(project_id="proj-1"),
        lambda: "TEST-TOKEN",
        transport=httpx.MockTransport(handler),
    )


def plugin_with_project(tmp_path: Path) -> tuple[SttPlugin, PluginContext]:
    """Plugin with project configured but the worker off; caller fakes HTTP."""

    ctx = make_ctx(
        tmp_path,
        values=disabled_values(
            **{
                "stt.project_id": "proj-1",
                "stt.credentials_json": "{}",  # only used for "configured"
            }
        ),
    )
    plugin = SttPlugin()
    plugin.setup(ctx)
    return plugin, ctx


def _rsa_private_key_pem() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")


def _service_account_json(project: str = "proj-1") -> str:
    return json.dumps(
        {
            "type": "service_account",
            "project_id": project,
            "private_key_id": "kid",
            "private_key": _rsa_private_key_pem(),
            "client_email": f"svc@{project}.iam.gserviceaccount.com",
            "client_id": "1234567890",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    )


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

class TestCredentials:
    def test_service_account_token_and_project(self) -> None:
        calls: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(
                {
                    "url": str(request.url),
                    "body": parse_qs(request.content.decode("ascii")),
                }
            )
            return httpx.Response(
                200,
                json={
                    "access_token": "ya29.test",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )

        provider = GoogleTokenProvider(
            credentials_json=_service_account_json("proj-42"),
            transport=httpx.MockTransport(handler),
        )
        try:
            assert provider.project_id == "proj-42"
            assert provider.token() == "ya29.test"
            assert provider.token() == "ya29.test"  # cached
        finally:
            provider.close()

        assert len(calls) == 1
        assert calls[0]["url"] == "https://oauth2.googleapis.com/token"
        body = calls[0]["body"]
        assert body["grant_type"] == [
            "urn:ietf:params:oauth:grant-type:jwt-bearer"
        ]
        assert body["assertion"][0].count(".") == 2  # JWT

    def test_project_override(self) -> None:
        provider = GoogleTokenProvider(
            credentials_json=_service_account_json("from-json"),
            project_id="explicit",
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(500)
            ),
        )
        try:
            assert provider.project_id == "explicit"
        finally:
            provider.close()

    def test_invalid_json(self) -> None:
        provider = GoogleTokenProvider(credentials_json="{not json")
        try:
            with pytest.raises(SttError, match="解析失败"):
                provider.token()
        finally:
            provider.close()

    def test_unsupported_type(self) -> None:
        provider = GoogleTokenProvider(credentials_json='{"type": "magic"}')
        try:
            with pytest.raises(SttError, match="不支持的 Google 凭据类型"):
                provider.token()
        finally:
            provider.close()

    def test_missing_file(self, tmp_path: Path) -> None:
        provider = GoogleTokenProvider(
            credentials_file=str(tmp_path / "missing.json")
        )
        try:
            with pytest.raises(SttError, match="无法读取"):
                provider.token()
        finally:
            provider.close()

    def test_service_account_file(self, tmp_path: Path) -> None:
        path = tmp_path / "sa.json"
        path.write_text(_service_account_json("file-proj"), encoding="utf-8")
        provider = GoogleTokenProvider(
            credentials_file=str(path),
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json={
                        "access_token": "token-from-file",
                        "expires_in": 3600,
                        "token_type": "Bearer",
                    },
                )
            ),
        )
        try:
            assert provider.project_id == "file-proj"
            assert provider.token() == "token-from-file"
        finally:
            provider.close()


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class TestGoogleCloudBackend:
    def test_request_shape_v2(self) -> None:
        seen: dict[str, Any] = {}
        wav = b"RIFF" + b"\x00" * 36

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization")
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=speech_response("你好。"))

        client = make_client(handler)
        try:
            segments = client.recognize(wav, duration=1.0)
        finally:
            client.close()

        assert seen["url"] == (
            "https://us-speech.googleapis.com/v2/projects/proj-1"
            "/locations/us/recognizers/_:recognize"
        )
        assert seen["auth"] == "Bearer TEST-TOKEN"
        config = seen["body"]["config"]
        assert config["autoDecodingConfig"] == {}
        assert config["languageCodes"] == ["cmn-Hans-CN"]
        assert config["model"] == "chirp_3"
        assert config["features"]["enableAutomaticPunctuation"] is True
        assert base64.b64decode(seen["body"]["content"]) == wav

        assert len(segments) == 1
        assert segments[0].text == "你好。"
        assert segments[0].start == pytest.approx(0.0)
        assert segments[0].end == pytest.approx(1.5)
        assert segments[0].confidence == pytest.approx(0.9)
        assert segments[0].language_code == "cmn-hans-cn"

    def test_global_location_uses_global_endpoint(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(200, json={"results": []})

        client = GoogleSpeechClient(
            SpeechConfig(project_id="proj-1", location="global"),
            lambda: "TEST-TOKEN",
            transport=httpx.MockTransport(handler),
        )
        try:
            client.recognize(b"RIFF")
        finally:
            client.close()
        assert seen["url"].startswith(
            "https://speech.googleapis.com/v2/projects/proj-1/locations/global/"
        )

    def test_multiple_results_get_sequential_offsets(self) -> None:
        client = make_client(
            lambda _request: httpx.Response(
                200, json=speech_response("第一句。", "第二句。")
            )
        )
        try:
            segments = client.recognize(b"RIFF")
        finally:
            client.close()
        assert [s.text for s in segments] == ["第一句。", "第二句。"]
        assert segments[0].start == pytest.approx(0.0)
        assert segments[0].end == pytest.approx(1.5)
        assert segments[1].start == pytest.approx(1.5)
        assert segments[1].end == pytest.approx(2.5)

    def test_missing_result_offset_falls_back_to_duration(self) -> None:
        payload = {
            "results": [{"alternatives": [{"transcript": "只有一句。"}]}],
        }
        client = make_client(lambda _request: httpx.Response(200, json=payload))
        try:
            segments = client.recognize(b"RIFF", duration=4.0)
        finally:
            client.close()
        assert segments[0].end == pytest.approx(4.0)

    def test_api_error_is_surfaced(self) -> None:
        payload = {
            "error": {
                "code": 403,
                "message": "Permission denied on recognizer.",
            }
        }
        client = make_client(lambda _request: httpx.Response(403, json=payload))
        try:
            with pytest.raises(SttError) as excinfo:
                client.recognize(b"RIFF")
        finally:
            client.close()
        message = str(excinfo.value)
        assert "HTTP 403" in message
        assert "Permission denied" in message
        assert "speech.recognizers.recognize" in message

    def test_missing_project_raises_before_network(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request expected")

        client = GoogleSpeechClient(
            SpeechConfig(project_id=""),
            lambda: "TEST-TOKEN",
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(SttError, match="project"):
                client.recognize(b"RIFF")
        finally:
            client.close()

    def test_missing_token_raises_before_network(self) -> None:
        client = GoogleSpeechClient(
            SpeechConfig(project_id="proj-1"),
            lambda: "",
            transport=httpx.MockTransport(
                lambda _request: pytest.fail("no request expected")
            ),
        )
        try:
            with pytest.raises(SttError, match="OAuth token"):
                client.recognize(b"RIFF")
        finally:
            client.close()

    def test_empty_audio_returns_nothing(self) -> None:
        client = make_client(lambda _request: httpx.Response(200, json={}))
        try:
            assert client.recognize(b"") == []
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class TestPluginBasics:
    def test_plugin_identity(self) -> None:
        assert SttPlugin.id == "stt"
        assert SttPlugin.requires == ("settings",)
        assert SttPlugin.provides == ("stt",)
        assert STTPlugin is SttPlugin

    def test_setup_registers_service_settings_and_ui(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path, values=disabled_values())
        plugin = SttPlugin()
        plugin.setup(ctx)
        try:
            service = ctx.services.require("stt")
            assert isinstance(service, STTService)

            keys = {spec.key for spec in ctx.services.require("settings").specs()}
            assert {
                "stt.enabled",
                "stt.credentials_json",
                "stt.credentials_file",
                "stt.project_id",
                "stt.location",
                "stt.model",
                "stt.language_code",
                "stt.chunk_seconds",
            } <= keys
            assert "stt.api_key" not in keys

            snap = ctx.ui.snapshot()
            assert PAGE_ID in snap["pages"]
            # The redesigned UI opens pages from the command palette.
            assert "stt.debug" in snap["commands"]
        finally:
            plugin.teardown()

    def test_credentials_json_is_a_masked_secret(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path, values=disabled_values())
        plugin = SttPlugin()
        plugin.setup(ctx)
        try:
            specs = {s.key: s for s in ctx.services.require("settings").specs()}
            assert specs["stt.credentials_json"].secret is True
        finally:
            plugin.teardown()

    def test_status_without_credentials(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path)
        plugin = SttPlugin()
        plugin.setup(ctx)
        try:
            status = ctx.services.require("stt").status()
            assert status["state"] == "needs_credentials"
            assert status["backend"] == "google-cloud-stt-v2"
            assert status["error"] == "未配置 Google 凭据"
            assert status["segments"] == 0
            assert status["chars"] == 0
            assert status["model"] == "chirp_3"
        finally:
            plugin.teardown()

    def test_status_when_disabled(self, tmp_path: Path) -> None:
        ctx = make_ctx(
            tmp_path,
            values=disabled_values(
                **{"stt.credentials_json": _service_account_json()}
            ),
        )
        plugin = SttPlugin()
        plugin.setup(ctx)
        try:
            status = ctx.services.require("stt").status()
            assert status["state"] == "disabled"
            assert status["worker_alive"] is False
        finally:
            plugin.teardown()

    def test_no_text_without_credentials(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path)
        plugin = SttPlugin()
        plugin.setup(ctx)
        try:
            ctx.streams.current.audio.append(b"\x00\x00" * 16000)
            status = ctx.services.require("stt").status()
            assert status["audio_seconds"] == pytest.approx(1.0)
            assert ctx.streams.current.transcript.items_since(0) == []
        finally:
            plugin.teardown()


class TestTranscribeRange:
    def test_appends_transcript_items_and_emits_event(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plugin, ctx = plugin_with_project(tmp_path)
        captured: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "alternatives": [
                                {"transcript": "第一句。", "confidence": 0.8}
                            ],
                            "resultEndOffset": "2.000s",
                        }
                    ]
                },
            )

        monkeypatch.setattr(
            "plugins.stt.plugin.GoogleSpeechClient",
            lambda config, token: GoogleSpeechClient(
                config, lambda: "TEST-TOKEN", transport=httpx.MockTransport(handler)
            ),
        )
        ctx.streams.current.audio.append(b"\x00\x00" * 64000)  # 4 s

        segments: list[dict[str, Any]] = []
        ctx.events.on("stt.segment", lambda event: segments.append(dict(event.payload)))

        try:
            result = plugin.transcribe_range(0.0, 2.0)
            ctx.events.flush(timeout=2.0)
        finally:
            plugin.teardown()
            ctx.events.stop()

        assert result["ok"] is True
        assert result["segments"] == 1
        assert result["chars"] == len("第一句。")

        transcript = ctx.streams.current.transcript
        assert len(transcript) == 1
        item = transcript[0]
        assert item.text == "第一句。"
        assert item.source == "stt"
        assert item.start == pytest.approx(0.0)
        assert item.end == pytest.approx(2.0)
        assert item.meta["model"] == "chirp_3"
        assert item.meta["confidence"] == pytest.approx(0.8)

        body = captured[0]
        assert body["config"]["model"] == "chirp_3"
        assert body["config"]["languageCodes"] == ["cmn-Hans-CN"]
        wav = base64.b64decode(body["content"])
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert segments and segments[0]["segments"] == 1

    def test_skips_when_range_is_empty(self, tmp_path: Path) -> None:
        plugin, ctx = plugin_with_project(tmp_path)
        try:
            result = plugin.transcribe_range(0.0, 0.0)
        finally:
            plugin.teardown()
        assert result["skipped"] is True
        assert len(ctx.streams.current.transcript) == 0

    def test_backend_error_propagates(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plugin, ctx = plugin_with_project(tmp_path)
        payload = {"error": {"code": 400, "message": "bad model"}}
        monkeypatch.setattr(
            "plugins.stt.plugin.GoogleSpeechClient",
            lambda config, token: GoogleSpeechClient(
                config,
                lambda: "TEST-TOKEN",
                transport=httpx.MockTransport(
                    lambda _request: httpx.Response(400, json=payload)
                ),
            ),
        )
        ctx.streams.current.audio.append(b"\x00\x00" * 16000)
        try:
            with pytest.raises(SttError, match="bad model"):
                plugin.transcribe_range(0.0, 1.0)
        finally:
            plugin.teardown()
        assert len(ctx.streams.current.transcript) == 0

    def test_settings_are_used_in_request(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ctx = make_ctx(
            tmp_path,
            values=disabled_values(
                **{
                    "stt.project_id": "proj-9",
                    "stt.location": "asia-northeast1",
                    "stt.model": "long",
                    "stt.language_code": "en-US",
                }
            ),
        )
        plugin = SttPlugin()
        plugin.setup(ctx)
        captured: list[dict[str, Any]] = []
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            seen["url"] = str(request.url)
            return httpx.Response(200, json={"results": []})

        monkeypatch.setattr(
            "plugins.stt.plugin.GoogleSpeechClient",
            lambda config, token: GoogleSpeechClient(
                config, lambda: "TEST-TOKEN", transport=httpx.MockTransport(handler)
            ),
        )
        ctx.streams.current.audio.append(b"\x00\x00" * 16000)
        try:
            plugin.transcribe_range(0.0, 1.0)
        finally:
            plugin.teardown()
        assert captured[0]["config"]["model"] == "long"
        assert captured[0]["config"]["languageCodes"] == ["en-US"]
        assert "asia-northeast1-speech.googleapis.com" in seen["url"]


class TestWorker:
    def test_process_next_window_advances_cursor(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        values = {
            "stt.enabled": False,
            "stt.project_id": "proj-1",
            "stt.credentials_json": "{}",
            "stt.chunk_seconds": 2.0,
        }
        ctx = make_ctx(tmp_path, values=values)
        plugin = SttPlugin()
        plugin.setup(ctx)
        # Enable only the logic under test: the worker thread stays off.
        ctx.services.require("settings").set("stt.enabled", True)
        monkeypatch.setattr(
            "plugins.stt.plugin.GoogleSpeechClient",
            lambda config, token: GoogleSpeechClient(
                config,
                lambda: "TEST-TOKEN",
                transport=httpx.MockTransport(
                    lambda _request: httpx.Response(
                        200, json=speech_response("这是一句话。")
                    )
                ),
            ),
        )
        ctx.streams.current.audio.append(b"\x00\x00" * 64000)  # 4 s
        try:
            plugin._process_next_window()
            status = plugin.status()
            assert len(ctx.streams.current.transcript) == 1
            assert status["segments"] == 1
            assert status["cursor"] == pytest.approx(2.0)

            # The next call consumes the following window, not the same one.
            plugin._process_next_window()
            assert len(ctx.streams.current.transcript) == 2
            assert plugin.status()["cursor"] == pytest.approx(4.0)
        finally:
            plugin.teardown()

    def test_session_switch_resets_cursor(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ctx = make_ctx(
            tmp_path,
            values={
                "stt.enabled": False,
                "stt.project_id": "proj-1",
                "stt.credentials_json": "{}",
            },
        )
        plugin = SttPlugin()
        plugin.setup(ctx)
        ctx.services.require("settings").set("stt.enabled", True)
        monkeypatch.setattr(
            "plugins.stt.plugin.GoogleSpeechClient",
            lambda config, token: GoogleSpeechClient(
                config,
                lambda: "TEST-TOKEN",
                transport=httpx.MockTransport(
                    lambda _request: httpx.Response(
                        200, json=speech_response("旧会话。")
                    )
                ),
            ),
        )
        ctx.streams.current.audio.append(b"\x00\x00" * 16000 * 20)
        try:
            plugin._process_next_window()
            assert plugin.status()["cursor"] > 0

            ctx.streams.start_session(meta={"course": "test"})
            ctx.events.flush(timeout=2.0)
            assert plugin.status()["cursor"] == pytest.approx(0.0)
            assert plugin.status()["segments"] == 0
        finally:
            plugin.teardown()
            ctx.events.stop()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class TestUiState:
    def test_page_render_without_transcript(self, tmp_path: Path) -> None:
        plugin, ctx = plugin_with_project(tmp_path)
        try:
            page = ctx.ui._pages.get(PAGE_ID)
            assert page is not None
            blocks = ctx.ui._render(page.render, "test")
            texts = [b for b in blocks if b.get("type") == "text"]
            assert any("暂无转写结果" in b.get("text", "") for b in texts)
        finally:
            plugin.teardown()

    def test_page_render_shows_transcript(self, tmp_path: Path) -> None:
        plugin, ctx = plugin_with_project(tmp_path)
        try:
            ctx.streams.current.transcript.append(
                "设 $x \\in X$。",
                start=0.0,
                end=2.0,
                source="stt",
            )
            page = ctx.ui._pages.get(PAGE_ID)
            blocks = ctx.ui._render(page.render, "test")
            markdown = [b for b in blocks if b.get("type") == "markdown"]
            assert any("设 $x \\in X$。" in b.get("text", "") for b in markdown)
            assert any("00:00–00:02" in b.get("text", "") for b in markdown)
        finally:
            plugin.teardown()
