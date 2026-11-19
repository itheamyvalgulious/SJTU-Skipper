"""Stable identities for scheduled lectures and manual media sources."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256


def lecture_id(course_id: str, begin: str | datetime) -> str:
    """Identify a lecture by its course and first scheduled period."""
    if isinstance(begin, str):
        begin = datetime.strptime(begin, "%Y-%m-%d %H:%M:%S")
    return f"{course_id}-{begin:%Y%m%d}-{begin:%H%M}"


def source_id(url: str) -> str:
    """Identify a manual source by its exact URL."""
    return "source-" + sha256(url.encode("utf-8")).hexdigest()[:20]
