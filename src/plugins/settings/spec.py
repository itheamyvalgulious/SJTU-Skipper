"""Shared settings specification contract.

The settings plugin implements persistence and UI; every other plugin only
needs this dataclass to declare a setting.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

VALID_TYPES = {
    "string",
    "text",
    "int",
    "float",
    "bool",
    "select",
    "secret",
    "json",
}

SECRET_MASK = "******"
"""Placeholder shown in public (UI/HTTP) output for a configured secret."""

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}
_FALSY = {"0", "false", "no", "off", "n", "f", ""}


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or value != int(value):
            raise ValueError(f"需要整数，收到 {value!r}")
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("需要整数，收到空字符串")
        try:
            return int(text, 10)
        except ValueError:
            try:
                number = float(text)
            except ValueError as exc:
                raise ValueError(f"需要整数，收到 {value!r}") from exc
            if not math.isfinite(number) or number != int(number):
                raise ValueError(f"需要整数，收到 {value!r}") from None
            return int(number)
    raise ValueError(f"需要整数，收到 {value!r}")


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("需要数字，收到空字符串")
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(f"需要数字，收到 {value!r}") from exc
    else:
        raise ValueError(f"需要数字，收到 {value!r}")
    if not math.isfinite(number):
        raise ValueError(f"需要有限数字，收到 {value!r}")
    return number


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUTHY:
            return True
        if text in _FALSY:
            return False
        raise ValueError(f"需要布尔值（true/false），收到 {value!r}")
    raise ValueError(f"需要布尔值，收到 {value!r}")


def _coerce_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 解析失败：{exc.msg}（第 {exc.lineno} 行）") from exc
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"值无法序列化为 JSON：{value!r}") from exc


@dataclass(frozen=True, slots=True)
class SettingsSpec:
    key: str
    label: str
    type: str = "string"
    default: Any = None
    group: str = "General"
    description: str = ""
    options: tuple[Any, ...] = ()
    order: float = 0
    secret: bool = False
    placeholder: str = ""
    # Environment variables used to seed this setting on first registration.
    # Seeding writes the value into settings.json so the UI shows the effective
    # value instead of silently reading a hidden fallback at use time.
    env: tuple[str, ...] = ()
    # Only show this field when another setting has one of the given values.
    # Example: ("input.mode", ("advanced",)).
    visible_when: tuple[str, tuple[Any, ...]] | None = None
    # Optional explanatory text shown above the group's form.
    group_note: str = ""

    def __post_init__(self) -> None:
        if self.type not in VALID_TYPES:
            raise ValueError(f"unsupported setting type {self.type!r}")
        if not self.key:
            raise ValueError("setting key is required")
        if self.type == "secret" and not self.secret:
            object.__setattr__(self, "secret", True)

    # ------------------------------------------------------------------
    # Helpers (contract stays: key/label/type/default/group/description/
    # options/order/secret/placeholder)
    # ------------------------------------------------------------------
    def option_pairs(self) -> tuple[tuple[Any, str], ...]:
        """Return ``(value, label)`` pairs for ``select`` options.

        Options may be plain values, or ``(value, label)`` pairs.
        """

        pairs: list[tuple[Any, str]] = []
        for option in self.options:
            if isinstance(option, (tuple, list)) and len(option) == 2:
                value, label = option
                pairs.append((value, str(label)))
            else:
                pairs.append((option, str(option)))
        return tuple(pairs)

    def coerce(self, value: Any) -> Any:
        """Validate and convert an incoming value to this setting's type.

        Raises :class:`ValueError` for values that cannot be converted or that
        are not allowed by this specification.
        """

        if self.type in ("string", "text", "secret"):
            if value is None:
                return ""
            return value if isinstance(value, str) else str(value)
        if self.type == "int":
            return _coerce_int(value)
        if self.type == "float":
            return _coerce_float(value)
        if self.type == "bool":
            return _coerce_bool(value)
        if self.type == "json":
            return _coerce_json(value)
        if self.type == "select":
            return self._coerce_select(value)
        # Unreachable: __post_init__ validates the type.
        raise ValueError(f"unsupported setting type {self.type!r}")

    def _coerce_select(self, value: Any) -> Any:
        pairs = self.option_pairs()
        if not pairs:
            return value
        for option, _label in pairs:
            if value is option or value == option:
                return option
        if isinstance(value, str):
            text = value.strip()
            for option, _label in pairs:
                if str(option) == text:
                    return option
        allowed = ", ".join(repr(option) for option, _ in pairs)
        raise ValueError(f"取值必须是 {allowed} 之一，收到 {value!r}")

    def mask(self, value: Any) -> Any:
        """Return the public form of ``value`` (secrets are masked)."""

        if not self.secret:
            return value
        if value is None or value == "":
            return ""
        return SECRET_MASK

    def is_secret_value(self, value: Any) -> bool:
        """True when ``value`` is empty or the mask (i.e. "keep current")."""

        return self.secret and (value is None or value == "" or value == SECRET_MASK)
