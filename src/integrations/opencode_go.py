"""OpenCode Go OAuth client.

OpenCode Go is an OAuth-protected OpenAI-compatible endpoint.  This module owns
only the OAuth protocol; credentials are stored by the application's settings
service, so SJTU-Skipper does **not** read or depend on a local OpenCode
installation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

OPENCODE_GO_BASE_URL = "https://opencode.ai/inference/go/openai/v1"
"""OpenAI-compatible Go endpoint.

The model catalog declares this exact base URL for ``chat-completions``; the
older ``/zen/go/v1`` path still answers but is not the documented one.
"""
DEVICE_CODE_URL = "https://opencode.ai/console/auth/device/code"
TOKEN_URL = "https://opencode.ai/console/auth/device/token"
OAUTH_CLIENT_ID = "opencode-cli"


class OpenCodeGoError(RuntimeError):
    """OpenCode Go authorization or token refresh failed."""


@dataclass(frozen=True, slots=True)
class DeviceFlow:
    device_code: str
    user_code: str
    verification_uri_complete: str
    interval: int
    expires_in: int


@dataclass(frozen=True, slots=True)
class TokenSet:
    access_token: str
    refresh_token: str
    expires_at_ms: int
    org_id: str


def _parse_token_response(data: dict[str, Any]) -> TokenSet:
    access = str(data.get("access_token") or "")
    if not access:
        raise OpenCodeGoError("OpenCode Go 响应缺少 access_token。")
    refresh = str(data.get("refresh_token") or "")
    expires_in = int(data.get("expires_in") or 0)
    expires_at_ms = (
        int(time.time() * 1000) + expires_in * 1000
        if expires_in
        else int(time.time() * 1000)
    )
    return TokenSet(
        access_token=access,
        refresh_token=refresh,
        expires_at_ms=expires_at_ms,
        org_id=str(data.get("org_id") or ""),
    )


def start_device_flow() -> DeviceFlow:
    """Start OpenCode's device authorization flow."""

    try:
        response = httpx.post(
            DEVICE_CODE_URL,
            json={"client_id": OAUTH_CLIENT_ID, "supports_org_scope": True},
            timeout=30,
        )
    except httpx.HTTPError as exc:
        raise OpenCodeGoError(f"无法连接 OpenCode 授权服务：{exc}") from exc
    if response.status_code != 200:
        raise OpenCodeGoError(
            f"OpenCode 授权请求失败：HTTP {response.status_code} {response.text[:200]}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise OpenCodeGoError("OpenCode 授权响应不是 JSON。") from exc
    device_code = str(data.get("device_code") or "")
    user_code = str(data.get("user_code") or "")
    verification = str(data.get("verification_uri_complete") or data.get("verification_uri") or "")
    if not device_code or not verification:
        raise OpenCodeGoError("OpenCode 授权响应缺少 device_code/verification_uri。")
    return DeviceFlow(
        device_code=device_code,
        user_code=user_code,
        verification_uri_complete=verification,
        interval=max(1, int(data.get("interval") or 5)),
        expires_in=max(30, int(data.get("expires_in") or 600)),
    )


def poll_device_token(flow: DeviceFlow) -> TokenSet | None:
    """Poll once; return ``None`` while authorization is still pending."""

    try:
        response = httpx.post(
            TOKEN_URL,
            json={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": flow.device_code,
                "client_id": OAUTH_CLIENT_ID,
            },
            timeout=30,
        )
    except httpx.HTTPError as exc:
        raise OpenCodeGoError(f"轮询 OpenCode 授权状态失败：{exc}") from exc
    if response.status_code == 400:
        try:
            data = response.json()
        except ValueError:
            data = {}
        if data.get("error") == "authorization_pending":
            return None
        raise OpenCodeGoError(
            f"OpenCode 授权失败：{data.get('error_description') or data.get('error') or response.text[:200]}"
        )
    if response.status_code != 200:
        raise OpenCodeGoError(
            f"OpenCode 授权轮询失败：HTTP {response.status_code} {response.text[:200]}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise OpenCodeGoError("OpenCode 授权轮询响应不是 JSON。") from exc
    if data.get("error") == "authorization_pending":
        return None
    return _parse_token_response(data)


def refresh_access_token(refresh_token: str) -> TokenSet:
    """Exchange a refresh token for a new access/refresh token pair."""

    if not refresh_token:
        raise OpenCodeGoError("OpenCode Go refresh token 不存在，请重新关联套餐。")
    try:
        response = httpx.post(
            TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": OAUTH_CLIENT_ID,
            },
            timeout=30,
        )
    except httpx.HTTPError as exc:
        raise OpenCodeGoError(f"刷新 OpenCode Go 凭据失败：{exc}") from exc
    if response.status_code != 200:
        raise OpenCodeGoError(
            f"刷新 OpenCode Go 凭据失败：HTTP {response.status_code} {response.text[:200]}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise OpenCodeGoError("OpenCode Go 刷新响应不是 JSON。") from exc
    return _parse_token_response(data)


__all__ = [
    "DEVICE_CODE_URL",
    "OPENCODE_GO_BASE_URL",
    "OAUTH_CLIENT_ID",
    "TOKEN_URL",
    "DeviceFlow",
    "OpenCodeGoError",
    "TokenSet",
    "poll_device_token",
    "refresh_access_token",
    "start_device_flow",
]
