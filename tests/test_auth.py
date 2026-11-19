from __future__ import annotations

import html
import json
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ui.auth import AuthStore, COOKIE_NAME, hash_password, install, verify_password


def _write_auth(tmp_path: Path, password: str = "p") -> Path:
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"users": {"u": {"password": password}}}), "utf-8")
    return path


def test_password_hash_roundtrip() -> None:
    encoded = hash_password("correct horse")
    assert verify_password("correct horse", encoded)
    assert not verify_password("wrong", encoded)
    assert not verify_password("correct horse", "garbage")


def test_store_ignores_plaintext_on_disk(tmp_path: Path) -> None:
    path = _write_auth(tmp_path)
    store = AuthStore(path)
    assert store.verify("u", "p")
    assert not store.verify("u", "wrong")
    assert store.verify("u", "p")  # failure does not lock a <5 failure streak


def test_store_creates_an_account_when_missing(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / "auth.json")
    raw = json.loads((tmp_path / "auth.json").read_text("utf-8"))
    assert raw["users"]
    username, account = next(iter(raw["users"].items()))
    assert store.verify(username, account["password"])
    token = store.issue(username)
    reloaded = AuthStore(tmp_path / "auth.json")
    assert reloaded.verify(username, account["password"])
    assert reloaded.check(token) == username


def test_session_token_roundtrip_and_tampering(tmp_path: Path) -> None:
    store = AuthStore(_write_auth(tmp_path))
    token = store.issue("u")
    assert store.check(token) == "u"
    assert store.check(token + "x") is None
    body, signature = token.split(".")
    assert store.check(f"{body}.{'A' * len(signature)}") is None


def test_session_expiry(tmp_path: Path) -> None:
    path = _write_auth(tmp_path)
    raw = json.loads(path.read_text("utf-8"))
    raw["session_ttl_hours"] = 0
    path.write_text(json.dumps(raw), "utf-8")
    store = AuthStore(path)
    time.sleep(0.01)
    assert store.check(store.issue("u")) is None


def test_login_flow(tmp_path: Path) -> None:
    _write_auth(tmp_path)
    app = FastAPI()

    @app.get("/")
    def home() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/thing")
    def thing() -> dict[str, bool]:
        return {"ok": True}

    install(app, tmp_path)

    with TestClient(app) as client:
        assert client.get("/", follow_redirects=False).status_code == 303
        assert client.get("/api/thing").status_code == 401

        bad = client.post(
            "/login",
            data={"username": "u", "password": "nope"},
            follow_redirects=False,
        )
        assert bad.status_code == 401
        assert COOKIE_NAME not in bad.cookies

        good = client.post(
            "/login",
            data={"username": "u", "password": "p"},
            follow_redirects=False,
        )
        assert good.status_code == 303
        assert COOKIE_NAME in good.cookies
        assert client.get("/api/thing").status_code == 200
        assert client.get("/", follow_redirects=False).status_code == 200


def test_login_is_rate_limited(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("ui.auth._MAX_FAILURES", 2)
    _write_auth(tmp_path)
    app = FastAPI()
    install(app, tmp_path)
    with TestClient(app) as client:
        for _ in range(2):
            assert (
                client.post(
                    "/login",
                    data={"username": "u", "password": "bad"},
                    follow_redirects=False,
                ).status_code
                == 401
            )
        blocked = client.post(
            "/login",
            data={"username": "u", "password": "p"},
            follow_redirects=False,
        )
        assert blocked.status_code == 429


def test_login_returns_to_requested_route(tmp_path: Path) -> None:
    _write_auth(tmp_path)
    app = FastAPI()

    @app.get("/reader")
    def reader() -> dict[str, bool]:
        return {"ok": True}

    install(app, tmp_path)
    target = "/reader?page=reader:notes:session:notes&doc=notes%2Flecture.md"
    hidden = f'name="next" value="{html.escape(target, quote=True)}"'
    with TestClient(app) as client:
        gate = client.get(target, follow_redirects=False)
        assert gate.status_code == 303
        login_url = gate.headers["location"]
        assert urlsplit(login_url).path == "/login"
        assert parse_qs(urlsplit(login_url).query) == {"next": [target]}
        assert hidden in client.get(login_url).text

        failed = client.post(
            "/login",
            data={"username": "u", "password": "bad", "next": target},
            follow_redirects=False,
        )
        assert failed.status_code == 401
        assert hidden in failed.text
        assert COOKIE_NAME not in failed.cookies

        success = client.post(
            "/login",
            data={"username": "u", "password": "p", "next": target},
            follow_redirects=False,
        )
        assert success.status_code == 303
        assert success.headers["location"] == target
        assert client.get(target).status_code == 200
        authenticated = client.get(login_url, follow_redirects=False)
        assert authenticated.status_code == 303
        assert authenticated.headers["location"] == target


def test_locked_login_preserves_next(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("ui.auth._MAX_FAILURES", 1)
    _write_auth(tmp_path)
    app = FastAPI()
    install(app, tmp_path)
    target = "/?page=input:session&class=course"
    with TestClient(app) as client:
        client.post("/login", data={"username": "u", "password": "bad"})
        blocked = client.post(
            "/login", data={"username": "u", "password": "p", "next": target}
        )
        assert blocked.status_code == 429
        assert f'name="next" value="{html.escape(target, quote=True)}"' in blocked.text
        assert COOKIE_NAME not in blocked.cookies


@pytest.mark.parametrize(
    "target",
    [
        "https://example.com/",
        "//example.com/",
        "///example.com/",
        "javascript:alert(1)",
        "relative/path",
        "\\\\example.com/",
        "/\\example.com/",
        "/%2fexample.com/",
        "/%5cexample.com/",
        "%2f%2fexample.com/",
        "/\n/example.com/",
        "/%0d%0aLocation:https://example.com/",
    ],
)
def test_login_rejects_external_next(tmp_path: Path, target: str) -> None:
    _write_auth(tmp_path)
    app = FastAPI()
    install(app, tmp_path)
    login_url = "/login?" + urlencode({"next": target})
    with TestClient(app) as client:
        page = client.get(login_url)
        assert 'name="next" value="/"' in page.text
        success = client.post(
            "/login",
            data={"username": "u", "password": "p", "next": target},
            follow_redirects=False,
        )
        assert success.status_code == 303
        assert success.headers["location"] == "/"
        authenticated = client.get(login_url, follow_redirects=False)
        assert authenticated.headers["location"] == "/"


def test_login_escapes_next(tmp_path: Path) -> None:
    _write_auth(tmp_path)
    app = FastAPI()
    install(app, tmp_path)
    target = '/?page=settings&label="<example>"#section'
    with TestClient(app) as client:
        page = client.get("/login", params={"next": target})
        assert f'name="next" value="{html.escape(target, quote=True)}"' in page.text
        assert target not in page.text
