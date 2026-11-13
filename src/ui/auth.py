"""Single-file password authentication for the web UI.

The app may be reachable from the campus network or the internet, so every
route is gated behind a signed session cookie.  Credentials live in
``<data_dir>/auth.json``:

.. code-block:: json

    {
      "users": {"alice": {"password": "choose-a-strong-one"}},
      "session_secret": "<generated on first start>",
      "session_ttl_hours": 168
    }

``password`` may be plaintext (hashed in memory on load) or a precomputed
``password_hash`` of the form ``scrypt$<salt_hex>$<hash_hex>``.  The file is
created with a random password on first start and logged once.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

logger = logging.getLogger("lecture_agent.ui.auth")

COOKIE_NAME = "skipper_session"
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_MAX_FAILURES = 5
_LOCKOUT_SECONDS = 300.0


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """Return ``scrypt$<salt>$<hash>`` for *password*."""

    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=32,
    )
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = encoded.split("$", 2)
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except ValueError:
        return False
    actual = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=len(expected),
    )
    return hmac.compare_digest(actual, expected)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class AuthStore:
    """Users, session signing and login throttling."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.users: dict[str, str] = {}
        self.secret = ""
        self.ttl_seconds = 7 * 24 * 3600
        self._lock = threading.Lock()
        self._failures: dict[str, tuple[int, float]] = {}
        self._load()

    # -- io ------------------------------------------------------------
    def _load(self) -> None:
        raw: dict[str, Any] = {}
        created = False
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text("utf-8"))
            except (OSError, ValueError):
                logger.error("auth file %s is unreadable; ignoring it", self.path)
                raw = {}
        if not isinstance(raw, dict):
            raw = {}
        users = raw.get("users")
        if not isinstance(users, dict) or not users:
            # First start (or an emptied file): create a usable account so the
            # UI is never left wide open.
            password = secrets.token_urlsafe(12)
            users = {"admin": {"password": password}}
            raw["users"] = users
            created = True
            logger.warning(
                "created %s with username 'admin' and password %s — change it",
                self.path,
                password,
            )
        hashes: dict[str, str] = {}
        for name, spec in users.items():
            if not isinstance(spec, dict):
                continue
            encoded = str(spec.get("password_hash") or "")
            if not encoded and spec.get("password") is not None:
                encoded = hash_password(str(spec.get("password")))
            if encoded:
                hashes[str(name)] = encoded
        self.users = hashes
        self.secret = str(raw.get("session_secret") or "")
        if not self.secret:
            self.secret = secrets.token_hex(32)
            raw["session_secret"] = self.secret
            created = True
        raw_ttl = raw.get("session_ttl_hours")
        if raw_ttl is None:
            self.ttl_seconds = 7 * 24 * 3600
        else:
            try:
                self.ttl_seconds = max(0, int(raw_ttl)) * 3600
            except (TypeError, ValueError):
                self.ttl_seconds = 7 * 24 * 3600
        if created:
            self._save(raw)

    def _save(self, raw: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(raw, indent=2), "utf-8")
            os.chmod(tmp, 0o600)
            tmp.replace(self.path)
        except OSError:
            logger.exception("failed to write %s", self.path)

    # -- auth ----------------------------------------------------------
    def verify(self, username: str, password: str, *, key: str = "") -> bool:
        username = (username or "").strip()
        throttle_key = key or username
        now = time.monotonic()
        with self._lock:
            failures, until = self._failures.get(throttle_key, (0, 0.0))
            if until and now < until:
                return False
        encoded = self.users.get(username)
        ok = bool(encoded) and verify_password(password, encoded)
        with self._lock:
            if ok:
                self._failures.pop(throttle_key, None)
            else:
                failures += 1
                until = now + _LOCKOUT_SECONDS if failures >= _MAX_FAILURES else 0.0
                self._failures[throttle_key] = (failures, until)
        return ok

    def locked(self, key: str) -> bool:
        with self._lock:
            _failures, until = self._failures.get(key, (0, 0.0))
            return bool(until) and time.monotonic() < until

    # -- sessions ------------------------------------------------------
    def _sign(self, payload: str) -> str:
        digest = hmac.new(
            self.secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
        ).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    def issue(self, username: str) -> str:
        payload = json.dumps(
            {"u": username, "exp": int(time.time()) + self.ttl_seconds},
            separators=(",", ":"),
        ).encode("utf-8")
        body = base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
        return f"{body}.{self._sign(body)}"

    def check(self, token: str) -> str | None:
        if not token or "." not in token:
            return None
        body, signature = token.rsplit(".", 1)
        if not hmac.compare_digest(signature, self._sign(body)):
            return None
        try:
            padded = body + "=" * (-len(body) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
        except (ValueError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        if int(payload.get("exp") or 0) <= int(time.time()):
            return None
        username = str(payload.get("u") or "")
        return username if username in self.users else None


# ---------------------------------------------------------------------------
# HTTP wiring
# ---------------------------------------------------------------------------


_LOGIN_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>登录 · SJTU-Skipper</title>
<style>
  :root { --blue:#1a73e8; --border:#dadce0; --text:#202124; --muted:#5f6368; }
  * { box-sizing:border-box; }
  body { margin:0; min-height:100vh; display:flex; align-items:center;
         justify-content:center; background:#f8f9fa; color:var(--text);
         font-family:"Google Sans",Roboto,"Helvetica Neue",Arial,"PingFang SC",
         "Microsoft YaHei",sans-serif; }
  .card { width:min(400px,92vw); background:#fff; border:1px solid var(--border);
          border-radius:8px; padding:40px 40px 32px; }
  h1 { margin:0 0 8px; font-size:22px; font-weight:500; }
  p.sub { margin:0 0 28px; color:var(--muted); font-size:14px; }
  label { display:block; font-size:13px; color:var(--muted); margin:0 0 6px; }
  input { width:100%; padding:12px 14px; font-size:15px; border:1px solid var(--border);
          border-radius:4px; outline:none; margin-bottom:20px; }
  input:focus { border-color:var(--blue); box-shadow:0 0 0 1px var(--blue); }
  button { width:100%; padding:11px; font-size:15px; font-weight:500; color:#fff;
           background:var(--blue); border:0; border-radius:4px; cursor:pointer; }
  button:hover { background:#1765cc; }
  .error { color:#d93025; font-size:13px; margin:0 0 16px; }
</style>
</head>
<body>
  <form class="card" method="post" action="/login">
    <h1>SJTU-Skipper</h1>
    <p class="sub">请使用本地 auth.json 中配置的账号登录</p>
    {error}
    <label for="username">用户名</label>
    <input id="username" name="username" autocomplete="username" autofocus required>
    <label for="password">密码</label>
    <input id="password" name="password" type="password" autocomplete="current-password" required>
    <button type="submit">登录</button>
  </form>
</body>
</html>
"""


def _cookie_secure(request: Request) -> bool:
    proto = request.headers.get("x-forwarded-proto", "")
    return request.url.scheme == "https" or proto.split(",")[0].strip() == "https"


def install(app: Any, data_dir: Path) -> AuthStore:
    """Gate *app* behind a login form + signed cookie."""

    store = AuthStore(Path(data_dir) / "auth.json")

    @app.get("/login")
    def login_page(request: Request) -> Any:
        if store.check(request.cookies.get(COOKIE_NAME, "")):
            return RedirectResponse("/", status_code=303)
        return HTMLResponse(_LOGIN_HTML.replace("{error}", ""))

    @app.post("/login")
    async def login_submit(request: Request) -> Any:
        body = await request.body()
        form = {
            key: values[0]
            for key, values in urllib.parse.parse_qs(
                body.decode("utf-8", "replace")
            ).items()
            if values
        }
        username = str(form.get("username") or "")
        password = str(form.get("password") or "")
        if store.locked(username):
            error = '<p class="error">尝试次数过多，请五分钟后再试。</p>'
            return HTMLResponse(
                _LOGIN_HTML.replace("{error}", error), status_code=429
            )
        if not store.verify(username, password, key=username):
            logger.warning("failed login for %r", username)
            error = '<p class="error">用户名或密码不正确。</p>'
            return HTMLResponse(
                _LOGIN_HTML.replace("{error}", error), status_code=401
            )
        token = store.issue(username)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            COOKIE_NAME,
            token,
            max_age=store.ttl_seconds,
            httponly=True,
            samesite="lax",
            secure=_cookie_secure(request),
            path="/",
        )
        logger.info("login ok for %r", username)
        return response

    @app.post("/logout")
    def logout() -> Any:
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE_NAME, path="/")
        return response

    public = {"/login", "/favicon.ico"}

    @app.middleware("http")
    async def auth_gate(request: Request, call_next: Any) -> Any:
        path = request.url.path
        if path in public:
            return await call_next(request)
        user = store.check(request.cookies.get(COOKIE_NAME, ""))
        if user is None:
            if path.startswith("/api/"):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return RedirectResponse("/login", status_code=303)
        request.state.user = user
        return await call_next(request)

    return store
