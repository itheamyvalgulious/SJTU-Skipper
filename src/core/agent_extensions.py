"""Role-scoped agent declarations and callback snapshots."""

from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping

from core.agent_types import AgentToolName

ExtensionHandler = Callable[[dict[str, Any]], Any]
_RESERVED_TOOLS = frozenset(AgentToolName) | {"review_notes"}


@dataclass(frozen=True, slots=True)
class AgentExtensionTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ExtensionHandler
    label: str | None = None

    def descriptor(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description,
            "parameters": deepcopy(self.parameters),
        }
        if self.label is not None:
            result["label"] = self.label
        return result


@dataclass(frozen=True, slots=True)
class AgentCompletionHook:
    id: str
    handler: ExtensionHandler
    priority: int = 0


@dataclass(frozen=True, slots=True)
class AgentExtension:
    id: str
    roles: tuple[str, ...]
    priority: int = 0
    prompt: str = ""
    tools: tuple[AgentExtensionTool, ...] = ()
    completion_hooks: tuple[AgentCompletionHook, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtensionSnapshot:
    role: str
    contributions: tuple[tuple[str, AgentExtension], ...]
    tools: Mapping[str, AgentExtensionTool]
    hooks: Mapping[str, AgentCompletionHook]
    hook_order: tuple[tuple[str, AgentCompletionHook], ...]

    @property
    def needs_callbacks(self) -> bool:
        return bool(self.tools or self.hooks)

    def payload(self) -> dict[str, Any]:
        return {
            "prompts": [item.prompt for _, item in self.contributions if item.prompt],
            "tools": [tool.descriptor() for tool in self.tools.values()],
            "completion_hooks": [
                {"owner": owner, "id": hook.id, "priority": hook.priority}
                for owner, hook in self.hook_order
            ],
        }


class AgentExtensionRegistry:
    def __init__(self, registered_roles: Callable[[], list[dict[str, Any]]]) -> None:
        self._lock = threading.RLock()
        self._entries: dict[tuple[str, str], AgentExtension] = {}
        self._registered_roles = registered_roles

    def register(self, owner: str, extension: AgentExtension) -> None:
        if not isinstance(owner, str) or not owner or "/" in owner:
            raise ValueError(
                "Extension owner must be non-empty and must not contain '/'"
            )
        if not isinstance(extension.id, str) or not extension.id or not extension.roles:
            raise ValueError("Extension ID and roles must not be empty")
        roles = tuple(extension.roles)
        self._validate_roles(roles)
        if type(extension.priority) is not int:
            raise TypeError("Extension priority must be an integer")
        tools: dict[str, AgentExtensionTool] = {}
        hooks: dict[str, AgentCompletionHook] = {}
        for tool in extension.tools:
            if (
                not isinstance(tool.name, str)
                or not tool.name
                or tool.name in _RESERVED_TOOLS
                or tool.name in tools
            ):
                raise ValueError(f"Duplicate or reserved tool name: {tool.name!r}")
            if not isinstance(tool.parameters, dict) or not callable(tool.handler):
                raise TypeError(
                    "Extension tools require a schema object and a callable handler"
                )
            tools[tool.name] = replace(tool, parameters=deepcopy(tool.parameters))
        for hook in extension.completion_hooks:
            if (
                not isinstance(hook.id, str)
                or not hook.id
                or "/" in hook.id
                or hook.id in hooks
            ):
                raise ValueError(
                    f"Duplicate or invalid completion hook ID: {hook.id!r}"
                )
            if type(hook.priority) is not int or not callable(hook.handler):
                raise TypeError(
                    "Completion hooks require an integer priority and a callable handler"
                )
            hooks[hook.id] = hook
        frozen = replace(
            extension,
            roles=roles,
            tools=tuple(tools.values()),
            completion_hooks=tuple(hooks.values()),
        )
        key = (owner, extension.id)
        with self._lock:
            if key in self._entries:
                raise ValueError(
                    f"Extension is already registered: {owner}/{extension.id}"
                )
            for (other_owner, _), other in self._entries.items():
                if not set(frozen.roles).intersection(other.roles):
                    continue
                duplicate_tools = set(tools).intersection(
                    tool.name for tool in other.tools
                )
                if duplicate_tools:
                    raise ValueError(
                        f"Duplicate tool name: {sorted(duplicate_tools)[0]}"
                    )
                if owner == other_owner and set(hooks).intersection(
                    hook.id for hook in other.completion_hooks
                ):
                    raise ValueError(f"Duplicate completion hook ID for owner: {owner}")
            self._entries[key] = frozen

    def unregister(self, owner: str) -> None:
        with self._lock:
            self._entries = {
                key: item for key, item in self._entries.items() if key[0] != owner
            }

    def _validate_roles(self, roles: tuple[str, ...]) -> None:
        registered = {item["id"] for item in self._registered_roles()}
        for role in roles:
            if role == "*":
                continue
            if not isinstance(role, str) or not role or role not in registered:
                raise ValueError(f"Unregistered agent role: {role!r}")

    def snapshots(self, *roles: str) -> tuple[ExtensionSnapshot, ...]:
        self._validate_roles(roles)
        with self._lock:
            return tuple(self._snapshot(role) for role in roles)

    def _snapshot(self, role: str) -> ExtensionSnapshot:
        contributions = tuple(
            sorted(
                (
                    (owner, item)
                    for (owner, _), item in self._entries.items()
                    if role in item.roles or "*" in item.roles
                ),
                key=lambda entry: (entry[1].priority, entry[0], entry[1].id),
            )
        )
        tools = {tool.name: tool for _, item in contributions for tool in item.tools}
        hook_order = tuple(
            sorted(
                (
                    (owner, hook)
                    for owner, item in contributions
                    for hook in item.completion_hooks
                ),
                key=lambda entry: (entry[1].priority, entry[0], entry[1].id),
            )
        )
        return ExtensionSnapshot(
            role=role,
            contributions=contributions,
            tools=MappingProxyType(tools),
            hooks=MappingProxyType(
                {f"{owner}/{hook.id}": hook for owner, hook in hook_order}
            ),
            hook_order=hook_order,
        )


__all__ = ["AgentExtension", "AgentExtensionTool", "AgentCompletionHook"]
