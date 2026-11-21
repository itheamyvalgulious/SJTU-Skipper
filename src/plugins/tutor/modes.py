"""User-editable note modes (abstract / interactive / detailed / summary …).

Modes live in ``<data_dir>/tutor/modes.json``.  The four built-in modes are
only defaults shipped with the plugin; users can edit their prompts, add new
modes or restore the defaults from the settings page.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

MAX_NAME = 40
MAX_DESC = 200
MAX_PROMPT = 20000

_SLUG = re.compile(r"[^a-z0-9_-]+")


class ModeStore:
    def __init__(self, path: Path, defaults: list[dict[str, str]]) -> None:
        self.path = Path(path)
        self.defaults = [dict(item) for item in defaults]
        self._lock = threading.RLock()
        self._modes: list[dict[str, str]] = []

    def load(self) -> None:
        with self._lock:
            raw = self._read()
            if raw is None:
                self._modes = [dict(item) for item in self.defaults]
                self._save()
                return
            modes: list[dict[str, str]] = []
            for item in raw.get("modes") or []:
                mode = self._clean(item)
                if mode:
                    modes.append(mode)
            self._modes = modes or [dict(item) for item in self.defaults]

    def _read(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _save(self) -> None:
        payload = {"version": 1, "modes": self._modes}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=f".{self.path.name}.", suffix=".tmp"
        )
        tmp = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def _clean(self, raw: Any) -> dict[str, str] | None:
        if not isinstance(raw, dict):
            return None
        prompt = str(raw.get("prompt") or "").strip()
        name = str(raw.get("name") or "").strip()
        if not prompt or not name:
            return None
        mode_id = _SLUG.sub("-", str(raw.get("id") or "")).strip("-") or uuid.uuid4().hex[:8]
        return {
            "id": mode_id,
            "name": name[:MAX_NAME],
            "description": str(raw.get("description") or "").strip()[:MAX_DESC],
            "prompt": prompt[:MAX_PROMPT],
        }

    def items(self) -> list[dict[str, str]]:
        with self._lock:
            return [dict(item) for item in self._modes]

    def get(self, mode_id: str) -> dict[str, str] | None:
        with self._lock:
            return next(
                (dict(item) for item in self._modes if item["id"] == mode_id), None
            )

    def add(self, name: str, prompt: str, description: str = "") -> tuple[bool, str]:
        name = str(name or "").strip()
        prompt = str(prompt or "").strip()
        if not name or not prompt:
            return False, "名称和提示词不能为空。"
        mode_id = _SLUG.sub("-", name.lower()) or uuid.uuid4().hex[:8]
        with self._lock:
            base = mode_id
            index = 2
            while any(item["id"] == mode_id for item in self._modes):
                mode_id = f"{base}-{index}"
                index += 1
            self._modes.append(
                {
                    "id": mode_id,
                    "name": name[:MAX_NAME],
                    "description": str(description or "").strip()[:MAX_DESC],
                    "prompt": prompt[:MAX_PROMPT],
                }
            )
            self._save()
            return True, f"已添加模式「{name[:MAX_NAME]}」"

    def update(self, mode_id: str, name: str, prompt: str, description: str = "") -> tuple[bool, str]:
        name = str(name or "").strip()
        prompt = str(prompt or "").strip()
        if not name or not prompt:
            return False, "名称和提示词不能为空。"
        with self._lock:
            mode = next((item for item in self._modes if item["id"] == mode_id), None)
            if mode is None:
                return False, "没有找到该模式。"
            mode.update(
                name=name[:MAX_NAME],
                description=str(description or "").strip()[:MAX_DESC],
                prompt=prompt[:MAX_PROMPT],
            )
            self._save()
            return True, f"已保存模式「{name[:MAX_NAME]}」"

    def delete(self, mode_id: str) -> bool:
        with self._lock:
            before = len(self._modes)
            self._modes = [item for item in self._modes if item["id"] != mode_id]
            if len(self._modes) == before or not self._modes:
                return False
            self._save()
            return True

    def reset_to_defaults(self) -> int:
        with self._lock:
            self._modes = [dict(item) for item in self.defaults]
            self._save()
            return len(self._modes)

    def options(self) -> list[tuple[str, str]]:
        return [(item["id"], item["name"]) for item in self.items()]


__all__ = ["ModeStore", "MAX_NAME", "MAX_DESC", "MAX_PROMPT"]
