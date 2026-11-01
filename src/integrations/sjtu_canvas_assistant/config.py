"""Configuration loading for the CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .exceptions import ConfigError

DEFAULT_BASE_URL = "https://oc.sjtu.edu.cn"
DEFAULT_DATA_DIR = "data"


@dataclass(frozen=True)
class Settings:
    base_url: str
    token: str
    data_dir: Path
    timeout_seconds: float = 30.0
    video_token: str = ""
    video_course_id: str = ""


def _clean_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        raise ConfigError("CANVAS_BASE_URL must start with http:// or https://.")
    return value


def load_settings(
    *,
    token: Optional[str] = None,
    base_url: Optional[str] = None,
    data_dir: Optional[Path] = None,
    require_token: bool = True,
) -> Settings:
    """Load settings from .env/environment with optional CLI overrides."""

    load_dotenv()
    resolved_token = token or os.getenv("CANVAS_TOKEN", "")
    resolved_base_url = base_url or os.getenv("CANVAS_BASE_URL", DEFAULT_BASE_URL)
    resolved_data_dir = data_dir or Path(os.getenv("CANVAS_DATA_DIR", DEFAULT_DATA_DIR))
    resolved_video_token = os.getenv("SJTU_VIDEO_TOKEN", "").strip()
    resolved_video_course_id = os.getenv("SJTU_VIDEO_COURSE_ID", "").strip()

    if require_token and not resolved_token.strip():
        raise ConfigError(
            "Canvas token is missing. Set CANVAS_TOKEN in .env or pass --token."
        )

    timeout_text = os.getenv("CANVAS_TIMEOUT_SECONDS", "30")
    try:
        timeout_seconds = float(timeout_text)
    except ValueError as exc:
        raise ConfigError("CANVAS_TIMEOUT_SECONDS must be a number.") from exc

    return Settings(
        base_url=_clean_base_url(resolved_base_url),
        token=resolved_token.strip(),
        data_dir=Path(resolved_data_dir).expanduser(),
        timeout_seconds=timeout_seconds,
        video_token=resolved_video_token,
        video_course_id=resolved_video_course_id,
    )
