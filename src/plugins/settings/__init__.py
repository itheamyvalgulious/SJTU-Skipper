"""Settings plugin package: spec contract, store, service and plugin."""

from __future__ import annotations

from .plugin import PAGE_ID, SAVE_ACTION, TAB_ID, SettingsPlugin
from .spec import SECRET_MASK, VALID_TYPES, SettingsSpec
from .store import CHANGED_EVENT, SettingsService, SettingsStore

__all__ = [
    "CHANGED_EVENT",
    "PAGE_ID",
    "SAVE_ACTION",
    "SECRET_MASK",
    "TAB_ID",
    "VALID_TYPES",
    "SettingsPlugin",
    "SettingsService",
    "SettingsSpec",
    "SettingsStore",
]
