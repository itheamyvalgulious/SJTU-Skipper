"""Settings persistence (``settings.json``) and the ``settings`` service.

``SettingsStore`` owns durability only: load, atomic save (0600, tmp +
``os.replace``), corrupt-file recovery.  ``SettingsService`` layers specs,
validation, defaults, secret masking and change notification on top.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .spec import SECRET_MASK, SettingsSpec

logger = logging.getLogger("lecture_agent.plugins.settings")

CHANGED_EVENT = "settings.changed"

Listener = Callable[[str], None]


class SettingsStore:
    """Thread-safe JSON file holding configured values.

    Defaults are *not* written to disk until a value is actually set.  A
    corrupt file is moved aside (best effort) and the store falls back to an
    empty mapping so a bad byte never blocks startup.
    """

    def __init__(self, path: str | Path, *, logger_: logging.Logger | None = None) -> None:
        self.path = Path(path)
        self.logger = logger_ or logger
        self._lock = threading.RLock()
        self._values: dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._values:
                return self._values[key]
            return default

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._values

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._values)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.has(key)

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def set(self, key: str, value: Any) -> Any:
        """Persist one value and return it."""

        with self._lock:
            if key in self._values and self._values[key] == value:
                return value
            updated = dict(self._values)
            updated[key] = value
            self._write(updated)
            self._values = updated
            return value

    def update(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Persist several values in one atomic write."""

        with self._lock:
            updated = dict(self._values)
            updated.update(values)
            if updated == self._values:
                return dict(values)
            self._write(updated)
            self._values = updated
            return dict(values)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key not in self._values:
                return False
            updated = dict(self._values)
            updated.pop(key, None)
            self._write(updated)
            self._values = updated
            return True

    # ------------------------------------------------------------------
    # Loading / saving
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            self._values = {}
            return
        try:
            text = self.path.read_text(encoding="utf-8")
            data = json.loads(text) if text.strip() else {}
            if not isinstance(data, dict):
                raise ValueError("settings.json must contain a JSON object")
        except Exception as exc:  # noqa: BLE001 - any damage falls back
            backup = self._backup_corrupt()
            self.logger.warning(
                "settings file %s is unreadable (%s); using defaults (backup: %s)",
                self.path,
                exc,
                backup,
            )
            self._values = {}
            return
        self._values = dict(data)

    def _backup_corrupt(self) -> Path | None:
        """Move the damaged file aside; never raise."""

        stem = self.path.name
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for attempt in range(100):
            suffix = "" if attempt == 0 else f"-{attempt}"
            candidate = self.path.with_name(f"{stem}.corrupt-{stamp}{suffix}.bak")
            if candidate.exists():
                continue
            try:
                shutil.copy2(self.path, candidate)
                try:
                    os.chmod(candidate, 0o600)
                except OSError:  # pragma: no cover - exotic filesystems
                    pass
                return candidate
            except OSError:
                self.logger.exception("failed to back up corrupt settings file")
                return None
        return None

    def _write(self, values: Mapping[str, Any]) -> None:
        """Atomically replace the settings file with 0600 permissions."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            dict(values), ensure_ascii=False, indent=2, sort_keys=True
        )
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=f".{self.path.name}.", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise


class SettingsService:
    """Spec registry + typed access to :class:`SettingsStore`.

    ``get`` returns real values (including secrets); ``all(public=True)`` and
    ``public_all()`` mask secrets.  Every accepted change emits
    ``settings.changed`` with ``key`` / ``old`` / ``new`` when an ``emit``
    callable was supplied.
    """

    def __init__(
        self,
        store: SettingsStore,
        *,
        emit: Callable[..., Any] | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        self._store = store
        self._emit = emit
        self.logger = logger_ or logger
        self._specs: dict[str, SettingsSpec] = {}
        self._listeners: list[Listener] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def store(self) -> SettingsStore:
        return self._store

    def specs(self) -> list[SettingsSpec]:
        with self._lock:
            return sorted(
                self._specs.values(), key=lambda s: (s.group, s.order, s.key)
            )

    def groups(self) -> list[str]:
        seen: dict[str, None] = {}
        for spec in self.specs():
            seen.setdefault(spec.group, None)
        return list(seen)

    def get_spec(self, key: str) -> SettingsSpec | None:
        with self._lock:
            return self._specs.get(key)

    def keys(self) -> list[str]:
        with self._lock:
            declared = list(self._specs)
        return list(dict.fromkeys([*sorted(declared), *self._store.keys()]))

    def has(self, key: str) -> bool:
        return self._store.has(key)

    # ------------------------------------------------------------------
    # Listener plumbing (UI refresh, toolbox sync, ...)
    # ------------------------------------------------------------------
    def add_listener(self, listener: Listener) -> Listener:
        with self._lock:
            self._listeners.append(listener)
        return listener

    def remove_listener(self, listener: Listener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _notify(self, reason: str) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(reason)
            except Exception:  # noqa: BLE001 - listeners are plugin code
                self.logger.exception("settings listener failed: %s", reason)

    def _emit_changed(self, key: str, old: Any, new: Any) -> None:
        if self._emit is None:
            return
        try:
            self._emit(CHANGED_EVENT, key=key, old=old, new=new)
        except Exception:  # noqa: BLE001 - bus must not break saving
            self.logger.exception("failed to emit %s", CHANGED_EVENT)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, spec: SettingsSpec) -> SettingsSpec:
        """Declare a setting (idempotent per key) and return the spec."""

        if not isinstance(spec, SettingsSpec):
            raise TypeError("register() expects a SettingsSpec")
        with self._lock:
            self._specs[spec.key] = spec
        self._notify(f"register:{spec.key}")
        return spec

    def register_many(self, specs: Iterable[SettingsSpec]) -> list[SettingsSpec]:
        return [self.register(spec) for spec in specs]

    # ------------------------------------------------------------------
    # Values
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Return the effective value (stored > spec default > ``default``)."""

        if self._store.has(key):
            return self._store.get(key)
        spec = self.get_spec(key)
        if spec is not None:
            if spec.default is None:
                return default
            return spec.default
        return default

    def public_value(self, key: str) -> Any:
        """Return the value safe for the UI (secrets masked)."""

        value = self.get(key)
        spec = self.get_spec(key)
        if spec is None:
            return value
        return spec.mask(value)

    def is_set(self, key: str) -> bool:
        """True when a non-empty value is configured or defaulted."""

        value = self.get(key)
        return value not in (None, "")

    def set(self, key: str, value: Any) -> Any:
        """Validate, persist and announce one value; returns the stored value."""

        spec = self.get_spec(key)
        coerced = spec.coerce(value) if spec is not None else _ensure_jsonable(key, value)
        old = self.get(key)
        self._store.set(key, coerced)
        if old != coerced:
            self._emit_changed(key, old, coerced)
        self._notify(f"set:{key}")
        return coerced

    def set_many(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Validate every value first, then persist them in one write.

        Secret entries that are empty or equal to the mask are skipped, which
        lets a form round-trip without leaking or clobbering the real secret.
        """

        if not isinstance(values, Mapping):
            raise TypeError("set_many() expects a mapping")
        pending: dict[str, Any] = {}
        old_values: dict[str, Any] = {}
        with self._lock:
            specs = dict(self._specs)
        for key, value in values.items():
            spec = specs.get(key)
            if spec is not None and spec.is_secret_value(value):
                continue
            coerced = spec.coerce(value) if spec is not None else _ensure_jsonable(key, value)
            pending[key] = coerced
            old_values[key] = self.get(key)
        if not pending:
            return {}
        self._store.update(pending)
        for key, new in pending.items():
            old = old_values.get(key)
            if old != new:
                self._emit_changed(key, old, new)
            self._notify(f"set:{key}")
        return dict(pending)

    def reset(self, key: str) -> Any:
        """Remove the stored value (back to the spec default) and announce it."""

        old = self.get(key)
        existed = self._store.delete(key)
        new = self.get(key)
        if existed or old != new:
            self._emit_changed(key, old, new)
            self._notify(f"reset:{key}")
        return new

    def all(self, *, public: bool = False) -> dict[str, Any]:
        """Return every declared or stored setting as a plain dict."""

        out: dict[str, Any] = {}
        for key in self.keys():
            out[key] = self.public_value(key) if public else self.get(key)
        return out

    def public_all(self) -> dict[str, Any]:
        """``all(public=True)`` shorthand."""

        return self.all(public=True)

    def describe(self) -> list[dict[str, Any]]:
        """UI-friendly spec list including masked values and ``is_set``."""

        out: list[dict[str, Any]] = []
        for spec in self.specs():
            value = self.get(spec.key)
            out.append(
                {
                    "key": spec.key,
                    "label": spec.label,
                    "type": spec.type,
                    "group": spec.group,
                    "description": spec.description,
                    "options": list(spec.options),
                    "order": spec.order,
                    "secret": spec.secret,
                    "placeholder": spec.placeholder,
                    "value": spec.mask(value),
                    "is_set": value not in (None, ""),
                }
            )
        return out

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.get_spec(key) is not None


def _ensure_jsonable(key: str, value: Any) -> Any:
    """Store undeclared settings only when they survive a JSON round trip."""

    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"设置项 {key!r} 的值无法序列化为 JSON：{value!r}") from exc


__all__ = [
    "CHANGED_EVENT",
    "SECRET_MASK",
    "SettingsService",
    "SettingsStore",
]
