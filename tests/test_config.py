from __future__ import annotations

from pathlib import Path

from config import load_config


def test_load_config_defaults_and_dirs(tmp_path: Path) -> None:
    cfg = load_config(data_dir=tmp_path / "agent")
    assert cfg.data_dir == (tmp_path / "agent").resolve()
    assert cfg.settings_path == cfg.data_dir / "settings.json"
    assert cfg.sessions_dir.exists()
    assert cfg.cache_dir.exists()


def test_load_config_toml(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "agent.toml"
    config_file.write_text(
        "\n".join(
            [
                "[agent]",
                f'data_dir = "{tmp_path / "from-toml"}"',
                'host = "0.0.0.0"',
                "port = 9999",
                "enable_stt = false",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(config_file=config_file)
    assert cfg.data_dir == (tmp_path / "from-toml").resolve()
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 9999
    assert cfg.enable_stt is False


def test_env_overrides(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LECTURE_AGENT_DATA_DIR", str(tmp_path / "env"))
    monkeypatch.setenv("LECTURE_AGENT_PORT", "8123")
    cfg = load_config()
    assert cfg.data_dir == (tmp_path / "env").resolve()
    assert cfg.port == 8123
