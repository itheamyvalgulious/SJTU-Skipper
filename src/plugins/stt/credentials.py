"""Google OAuth credentials for Speech-to-Text v2.

Speech-to-Text v2 rejects API keys (HTTP 401: "API keys are not supported by
this API"), so requests must carry an OAuth access token.  Credentials can be:

* a pasted service-account JSON (``"type": "service_account"``),
* a pasted/file authorized-user JSON (``"type": "authorized_user"``, the shape
  produced by ``gcloud auth application-default login``),
* Application Default Credentials when neither is configured.

Tokens are refreshed with ``google-auth``; the token endpoint is called through
the same ``httpx`` stack as the rest of the project.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import timezone
from pathlib import Path
from typing import Any

import httpx

from .errors import SttError

SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)
_REFRESH_SKEW_SECONDS = 60


class _AuthResponse:
    """Concrete implementation of ``google.auth.transport.Response``."""

    def __init__(self, status: int, headers: dict[str, str], data: bytes) -> None:
        self.status = status
        self.headers = headers
        self.data = data


class _HttpxRequest:
    """``google.auth.transport.Request`` adapter backed by ``httpx``."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def __call__(
        self,
        url: str,
        method: str = "GET",
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = 60.0,
        **_kwargs: Any,
    ) -> _AuthResponse:
        try:
            response = self._client.request(
                method,
                url,
                content=body,
                headers=headers,
                timeout=timeout or 60.0,
            )
        except httpx.HTTPError as exc:
            raise SttError(f"获取 Google OAuth token 失败：{exc}") from exc
        return _AuthResponse(
            response.status_code,
            dict(response.headers),
            response.content,
        )


class GoogleTokenProvider:
    """Loads credentials once and returns a valid OAuth access token."""

    def __init__(
        self,
        *,
        credentials_json: str = "",
        credentials_file: str = "",
        project_id: str = "",
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._credentials_json = str(credentials_json or "").strip()
        self._credentials_file = str(credentials_file or "").strip()
        self._project_id = str(project_id or "").strip()
        self._credentials: Any | None = None
        self._client = httpx.Client(
            timeout=max(1.0, float(timeout)),
            transport=transport,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def project_id(self) -> str:
        with self._lock:
            self._ensure_loaded()
            return self._project_id

    def token(self) -> str:
        with self._lock:
            self._ensure_loaded()
            credentials = self._credentials
            expiry = getattr(credentials, "expiry", None)
            expiry_ts = None
            if expiry is not None:
                # google-auth stores naive UTC datetimes.
                if expiry.tzinfo is None:
                    expiry_ts = expiry.replace(tzinfo=timezone.utc).timestamp()
                else:
                    expiry_ts = expiry.timestamp()
            expired = bool(
                expiry_ts is not None
                and expiry_ts - _REFRESH_SKEW_SECONDS <= time.time()
            )
            if not credentials.valid or expired:
                credentials.refresh(_HttpxRequest(self._client))
            token = getattr(credentials, "token", None)
            if not token:
                raise SttError("未能获取 Google OAuth access token")
            return str(token)

    def close(self) -> None:
        with self._lock:
            self._client.close()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._credentials is not None:
            return
        try:
            self._load()
        except SttError:
            raise
        except Exception as exc:  # noqa: BLE001 - google-auth raises many types
            raise SttError(f"Google 凭据加载失败：{type(exc).__name__}: {exc}") from exc

    def _load(self) -> None:
        if self._credentials_json:
            self._from_info(self._parse_json(self._credentials_json, "<设置项>"))
            return
        if self._credentials_file:
            path = Path(self._credentials_file).expanduser()
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise SttError(f"无法读取 Google 凭据文件 {path}：{exc}") from exc
            self._from_info(self._parse_json(raw, str(path)))
            return
        self._load_adc()

    @staticmethod
    def _parse_json(text: str, source: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise SttError(
                f"Google 凭据 JSON 解析失败（{source}）："
                f"{exc.msg}（第 {exc.lineno} 行）"
            ) from exc

    def _load_adc(self) -> None:
        from google.auth import default as google_auth_default

        credentials, project = google_auth_default(scopes=list(SCOPES))
        self._credentials = credentials
        if project and not self._project_id:
            self._project_id = str(project).strip()

    def _from_info(self, info: Any) -> None:
        if not isinstance(info, dict):
            raise SttError("Google 凭据 JSON 必须是对象")
        credential_type = str(info.get("type") or "")
        if credential_type == "service_account":
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_info(
                info, scopes=list(SCOPES)
            )
            project = str(info.get("project_id") or "").strip()
        elif credential_type == "authorized_user":
            from google.oauth2.credentials import Credentials

            credentials = Credentials.from_authorized_user_info(
                info, scopes=list(SCOPES)
            )
            project = str(
                info.get("quota_project_id") or info.get("project_id") or ""
            ).strip()
        else:
            raise SttError(
                "不支持的 Google 凭据类型 "
                f"{credential_type!r}；需要 service_account 或 authorized_user"
            )
        self._credentials = credentials
        if project and not self._project_id:
            self._project_id = project


__all__ = ["GoogleTokenProvider", "SCOPES"]
