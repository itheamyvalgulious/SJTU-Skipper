"""OpenCode Go credential integration.

OpenCode Go is not an OpenAI-compatible service with a static API key.  It
uses the OAuth credential stored by the local OpenCode CLI and requires the
``x-opencode-org-id`` / ``x-opencode-session`` headers in addition to a bearer
token.

This module reads that credential from OpenCode's SQLite database, refreshes it
through OpenCode's device-token endpoint when it is close to expiry, writes the
rotated credential back to the same row, and returns the values needed by the
LLM provider.  No credential value is exposed to the browser.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger("lecture_agent.integrations.opencode_go")

OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_CONSOLE_URL = "https://opencode.ai/console"
OAUTH_CLIENT_ID = "opencode-cli"
REFRESH_SKEW_MS = 60_000
SESSION_ID = uuid.uuid4().hex

_LOCK = threading.RLock()


class OpenCodeGoError(RuntimeError):
    """OpenCode Go credentials are missing or cannot be refreshed."""


@dataclass(slots=True)
class OpenCodeCredential:
    row_id: str
    access: str
    refresh: str
    expires_ms: int
    org_id: str
    server: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OpenCodeGoCredentials:
    base_url: str
    access_token: str
    headers: dict[str, str]


def _db_path() -> Path:
    override = os.environ.get("OPENCODE_DB")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".local/share/opencode/opencode.db"


def _load_credential() -> OpenCodeCredential:
    path = _db_path()
    if not path.exists():
        raise OpenCodeGoError(
            f"找不到 OpenCode 凭据数据库：{path}；请先在本机登录 OpenCode Go。"
        )
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        try:
            row = con.execute(
                "select id, value from credential "
                "where integration_id = 'opencode' "
                "order by time_updated desc limit 1"
            ).fetchone()
        finally:
            con.close()
    except sqlite3.Error as exc:
        raise OpenCodeGoError(f"读取 OpenCode 凭据失败：{exc}") from exc

    if row is None:
        raise OpenCodeGoError("OpenCode 中没有找到 opencode 凭据，请先登录 OpenCode Go。")
    row_id, raw_text = row
    try:
        raw = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise OpenCodeGoError("OpenCode 凭据格式无法解析。") from exc
    metadata = raw.get("metadata") or {}
    access = str(raw.get("access") or raw.get("key") or "")
    refresh = str(raw.get("refresh") or "")
    org_id = str(metadata.get("orgID") or "")
    server = str(metadata.get("server") or DEFAULT_CONSOLE_URL).rstrip("/")
    if not access:
        raise OpenCodeGoError("OpenCode 凭据中没有 access token，请重新登录 OpenCode Go。")
    return OpenCodeCredential(
        row_id=str(row_id),
        access=access,
        refresh=refresh,
        expires_ms=int(raw.get("expires") or 0),
        org_id=org_id,
        server=server,
        raw=raw,
    )


def _expired(cred: OpenCodeCredential) -> bool:
    return bool(cred.expires_ms and cred.expires_ms <= int(time.time() * 1000) + REFRESH_SKEW_MS)


def _save_credential(cred: OpenCodeCredential, *, access: str, refresh: str, expires_ms: int, org_id: str) -> None:
    raw = dict(cred.raw)
    raw["access"] = access
    raw["refresh"] = refresh
    raw["expires"] = expires_ms
    metadata = dict(raw.get("metadata") or {})
    if org_id:
        metadata["orgID"] = org_id
        raw["metadata"] = metadata
    path = _db_path()
    try:
        con = sqlite3.connect(path, timeout=10)
        try:
            con.execute(
                "update credential set value = ?, time_updated = ? where id = ?",
                (json.dumps(raw, ensure_ascii=False), int(time.time() * 1000), cred.row_id),
            )
            con.commit()
        finally:
            con.close()
    except sqlite3.Error as exc:
        logger.warning("failed to write refreshed OpenCode credential: %s", exc)


def _refresh(cred: OpenCodeCredential) -> OpenCodeCredential:
    if not cred.refresh:
        raise OpenCodeGoError("OpenCode Go refresh token 不存在，请重新登录 OpenCode Go。")
    url = f"{cred.server.rstrip('/')}/auth/device/token"
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": cred.refresh,
        "client_id": OAUTH_CLIENT_ID,
    }
    try:
        response = httpx.post(url, json=payload, timeout=30)
    except httpx.HTTPError as exc:
        raise OpenCodeGoError(f"刷新 OpenCode Go 凭据失败：{exc}") from exc
    if response.status_code != 200:
        raise OpenCodeGoError(
            f"刷新 OpenCode Go 凭据失败：HTTP {response.status_code} "
            f"{response.text[:200]}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise OpenCodeGoError("OpenCode Go 刷新响应不是 JSON。") from exc
    access = str(data.get("access_token") or "")
    refresh = str(data.get("refresh_token") or cred.refresh)
    if not access:
        raise OpenCodeGoError("OpenCode Go 刷新响应缺少 access_token。")
    expires_in = int(data.get("expires_in") or 0)
    expires_ms = int(time.time() * 1000) + expires_in * 1000 if expires_in else cred.expires_ms
    org_id = str(data.get("org_id") or cred.org_id)
    refreshed = OpenCodeCredential(
        row_id=cred.row_id,
        access=access,
        refresh=refresh,
        expires_ms=expires_ms,
        org_id=org_id,
        server=cred.server,
        raw=cred.raw,
    )
    _save_credential(
        cred,
        access=access,
        refresh=refresh,
        expires_ms=expires_ms,
        org_id=org_id,
    )
    return refreshed


def resolve_go_credentials() -> OpenCodeGoCredentials:
    """Return the current OpenCode Go token and routing headers.

    The database is read on every call (cheap SQLite read).  A refresh is only
    attempted near expiry, under a process-wide lock so concurrent VLM/Agent
    requests cannot rotate the refresh token twice.
    """

    with _LOCK:
        cred = _load_credential()
        if _expired(cred):
            try:
                cred = _refresh(cred)
            except OpenCodeGoError:
                # OpenCode itself may have refreshed the credential while we
                # were working; prefer the newer DB value if it is valid.
                latest = _load_credential()
                if latest.access != cred.access and not _expired(latest):
                    cred = latest
                else:
                    raise
        if not cred.org_id:
            raise OpenCodeGoError("OpenCode 凭据缺少 orgID，无法调用 OpenCode Go。")
        return OpenCodeGoCredentials(
            base_url=OPENCODE_GO_BASE_URL,
            access_token=cred.access,
            headers={
                "x-opencode-org-id": cred.org_id,
                "x-opencode-session": SESSION_ID,
            },
        )


def is_opencode_go_url(base_url: str) -> bool:
    """Return True for the OpenCode Go OpenAI-compatible endpoint."""

    try:
        parsed = urlsplit(str(base_url or "").strip())
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return (host == "opencode.ai" or host.endswith(".opencode.ai")) and "/zen/go" in (parsed.path or "")


__all__ = [
    "OPENCODE_GO_BASE_URL",
    "OpenCodeCredential",
    "OpenCodeGoCredentials",
    "OpenCodeGoError",
    "is_opencode_go_url",
    "resolve_go_credentials",
]
