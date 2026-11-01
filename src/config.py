"""Application configuration and path resolution."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "agent"


def _load_env() -> None:
    """Load project and submodule .env files without overriding real env."""

    for candidate in (PROJECT_ROOT / ".env",):
        if candidate.exists():
            load_dotenv(candidate, override=False)


@dataclass(slots=True)
class AppConfig:
    """Runtime configuration.

    Values here are process-level; user-editable runtime settings live in
    ``data_dir/settings.json`` and are managed by the settings plugin.
    """

    data_dir: Path = DATA_DIR
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "INFO"
    plugins: list[str] = field(default_factory=list)
    enable_input: bool = True
    enable_stt: bool = True
    enable_vlm: bool = True
    enable_summarize: bool = True
    demo_mode: bool = False
    config_file: Path | None = None

    def resolve(self) -> "AppConfig":
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        return self

    @property
    def settings_path(self) -> Path:
        return self.data_dir / "settings.json"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.sessions_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def load_config(
    *,
    data_dir: str | Path | None = None,
    host: str | None = None,
    port: int | None = None,
    config_file: str | Path | None = None,
) -> AppConfig:
    """Build an :class:`AppConfig` from env vars and an optional TOML file."""

    _load_env()
    cfg = AppConfig()
    candidate = Path(config_file) if config_file else PROJECT_ROOT / "agent.toml"
    if candidate.exists():
        with candidate.open("rb") as fh:
            raw = tomllib.load(fh)
        section = raw.get("agent", raw)
        cfg.data_dir = Path(section.get("data_dir", cfg.data_dir))
        cfg.host = str(section.get("host", cfg.host))
        cfg.port = int(section.get("port", cfg.port))
        cfg.log_level = str(section.get("log_level", cfg.log_level))
        cfg.plugins = list(section.get("plugins", cfg.plugins))
        cfg.enable_input = _coerce_bool(section.get("enable_input"), cfg.enable_input)
        cfg.enable_stt = _coerce_bool(section.get("enable_stt"), cfg.enable_stt)
        cfg.enable_vlm = _coerce_bool(section.get("enable_vlm"), cfg.enable_vlm)
        cfg.enable_summarize = _coerce_bool(
            section.get("enable_summarize"), cfg.enable_summarize
        )
        cfg.demo_mode = _coerce_bool(section.get("demo_mode"), cfg.demo_mode)
        cfg.config_file = candidate

    if data_dir is not None:
        cfg.data_dir = Path(data_dir)
    if host is not None:
        cfg.host = host
    if port is not None:
        cfg.port = port
    cfg.data_dir = Path(os.environ.get("LECTURE_AGENT_DATA_DIR", cfg.data_dir))
    cfg.host = os.environ.get("LECTURE_AGENT_HOST", cfg.host)
    cfg.port = int(os.environ.get("LECTURE_AGENT_PORT", cfg.port))
    cfg.log_level = os.environ.get("LECTURE_AGENT_LOG_LEVEL", cfg.log_level)
    cfg.resolve()
    cfg.ensure_dirs()
    return cfg
