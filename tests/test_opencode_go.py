"""Tests for the OpenCode Go credential bridge (no real network)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import httpx

from integrations import opencode_go
from integrations.opencode_go import (
    OpenCodeGoError,
    is_opencode_go_url,
    resolve_go_credentials,
)


def make_db(path: Path, *, expires_ms: int, access: str = "st_old") -> None:
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            create table credential (
              id text primary key,
              integration_id text not null,
              label text not null,
              value text not null,
              connector_id text,
              method_id text,
              active integer,
              time_created integer not null,
              time_updated integer not null
            )
            """
        )
        value = {
            "type": "oauth",
            "methodID": "device",
            "refresh": "rt_old",
            "access": access,
            "expires": expires_ms,
            "metadata": {
                "server": "https://opencode.ai/console",
                "orgID": "org_test",
            },
        }
        con.execute(
            "insert into credential values (?,?,?,?,?,?,?,?,?)",
            (
                "cred_test",
                "opencode",
                "Personal",
                json.dumps(value),
                None,
                None,
                1,
                int(time.time() * 1000),
                int(time.time() * 1000),
            ),
        )
        con.commit()
    finally:
        con.close()


def test_reading_go_credentials(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "opencode.db"
    make_db(db, expires_ms=int(time.time() * 1000) + 3600_000)
    monkeypatch.setenv("OPENCODE_DB", str(db))

    creds = resolve_go_credentials()
    assert creds.base_url == "https://opencode.ai/zen/go/v1"
    assert creds.access_token == "st_old"
    assert creds.headers["x-opencode-org-id"] == "org_test"
    assert creds.headers["x-opencode-session"]


def test_refresh_updates_db_and_returns_new_token(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "opencode.db"
    make_db(db, expires_ms=int(time.time() * 1000) - 1000)
    monkeypatch.setenv("OPENCODE_DB", str(db))

    def fake_post(url, *, json, timeout):
        assert url.endswith("/auth/device/token")
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
    creds = resolve_go_credentials()
    assert creds.access_token == "st_new"

    con = sqlite3.connect(db)
    try:
        raw = json.loads(
            con.execute("select value from credential where id='cred_test'").fetchone()[0]
        )
    finally:
        con.close()
    assert raw["access"] == "st_new"
    assert raw["refresh"] == "rt_new"
    assert raw["expires"] > int(time.time() * 1000)


def test_missing_db_is_clear(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENCODE_DB", str(tmp_path / "missing.db"))
    try:
        resolve_go_credentials()
    except OpenCodeGoError as exc:
        assert "OpenCode" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected OpenCodeGoError")


def test_url_detection() -> None:
    assert is_opencode_go_url("https://opencode.ai/zen/go/v1")
    assert not is_opencode_go_url("https://opencode.ai/zen/v1")
    assert not is_opencode_go_url("https://evilopencode.ai/zen/go/v1")
    assert not is_opencode_go_url("https://api.example.com/v1")
