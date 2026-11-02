"""Tests for the OpenCode Go OAuth client (no real network)."""

from __future__ import annotations

import httpx

from integrations import opencode_go
from integrations.opencode_go import (
    DeviceFlow,
    OpenCodeGoError,
    poll_device_token,
    refresh_access_token,
    start_device_flow,
)


def test_start_device_flow(monkeypatch) -> None:
    def fake_post(url, *, json, timeout):
        assert url == opencode_go.DEVICE_CODE_URL
        assert json["client_id"] == "opencode-cli"
        return httpx.Response(
            200,
            json={
                "device_code": "dev",
                "user_code": "ABCD",
                "verification_uri_complete": "https://opencode.ai/device?code=ABCD",
                "interval": 2,
                "expires_in": 600,
            },
        )

    monkeypatch.setattr(opencode_go.httpx, "post", fake_post)
    flow = start_device_flow()
    assert flow.device_code == "dev"
    assert flow.user_code == "ABCD"
    assert flow.interval == 2


def test_poll_pending_then_token(monkeypatch) -> None:
    flow = DeviceFlow(
        device_code="dev",
        user_code="ABCD",
        verification_uri_complete="https://opencode.ai/device",
        interval=1,
        expires_in=60,
    )
    calls = {"n": 0}

    def fake_post(url, *, json, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                400, json={"error": "authorization_pending"}
            )
        return httpx.Response(
            200,
            json={
                "access_token": "st_new",
                "refresh_token": "rt_new",
                "expires_in": 3600,
                "org_id": "org_test",
            },
        )

    monkeypatch.setattr(opencode_go.httpx, "post", fake_post)
    assert poll_device_token(flow) is None
    tokens = poll_device_token(flow)
    assert tokens is not None
    assert tokens.access_token == "st_new"
    assert tokens.refresh_token == "rt_new"
    assert tokens.org_id == "org_test"
    assert tokens.expires_at_ms > 0


def test_refresh_token(monkeypatch) -> None:
    def fake_post(url, *, json, timeout):
        assert json["grant_type"] == "refresh_token"
        assert json["refresh_token"] == "rt_old"
        return httpx.Response(
            200,
            json={
                "access_token": "st_new",
                "refresh_token": "rt_new",
                "expires_in": 3600,
                "org_id": "org_test",
            },
        )

    monkeypatch.setattr(opencode_go.httpx, "post", fake_post)
    tokens = refresh_access_token("rt_old")
    assert tokens.access_token == "st_new"
    assert tokens.refresh_token == "rt_new"


def test_refresh_error_is_clear(monkeypatch) -> None:
    def fake_post(url, *, json, timeout):
        return httpx.Response(400, text="invalid_grant")

    monkeypatch.setattr(opencode_go.httpx, "post", fake_post)
    try:
        refresh_access_token("rt_bad")
    except OpenCodeGoError as exc:
        assert "刷新" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected OpenCodeGoError")
