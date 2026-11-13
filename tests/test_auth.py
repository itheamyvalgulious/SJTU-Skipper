"""Tests for the single-file web login."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ui.auth import AuthStore, COOKIE_NAME, hash_password, install, verify_password


def _write_auth(tmp_path: Path, password: str = "p") -> Path:
    path = tmp_path / "auth.json"
    path.write_text(
        json.dumps({"users": {"u": {"password": password}}}), "utf-8"
    )
    return path


def test_password_hash_roundtrip() -> None:
    encoded = hash_password("correct horse")
    assert encoded.startswith("scrypt$")
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
    assert list(store.users) == ["admin"]
    raw = json.loads((tmp_path / "auth.json").read_text("utf-8"))
    assert raw["users"]["admin"]["password"]
    assert raw["session_secret"]


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
            "/login", data={"username": "u", "password": "nope"},
            follow_redirects=False,
        )
        assert bad.status_code == 401
        assert COOKIE_NAME not in bad.cookies

        good = client.post(
            "/login", data={"username": "u", "password": "p"},
            follow_redirects=False,
        )
        assert good.status_code == 303
        assert COOKIE_NAME in good.cookies
        assert client.get("/api/thing").status_code == 200
        assert client.get("/", follow_redirects=False).status_code == 200


def test_login_is_rate_limited(tmp_path: Path) -> None:
    _write_auth(tmp_path)
    app = FastAPI()
    install(app, tmp_path)
    with TestClient(app) as client:
        for _ in range(5):
            assert client.post(
                "/login", data={"username": "u", "password": "bad"},
                follow_redirects=False,
            ).status_code == 401
        blocked = client.post(
            "/login", data={"username": "u", "password": "p"},
            follow_redirects=False,
        )
        assert blocked.status_code == 429
