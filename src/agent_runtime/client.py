"""Multiplex requests over the private Bun process pipes."""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from core.agent_types import IpcMethod, RuntimeFrameType


class RuntimeClient:
    def __init__(self, data_dir: Path, *, settings=None):
        self.data_dir = Path(data_dir).resolve()
        self._settings = settings
        self._runtime_dir: Path | None = None
        self._reader: threading.Thread | None = None
        self._process = None
        self._lock = threading.RLock()
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[str, queue.Queue] = {}
        self._reader_error: str | None = None
        self._closed = False
        self._callback_handler: Callable[[dict[str, Any]], Any] | None = None
        self._roles_lock = threading.RLock()
        self._roles_sync_lock = threading.Lock()
        self._roles = {
            "default": {"id": "default", "title": "默认", "input": ["text"]},
            "compaction": {
                "id": "compaction",
                "title": "上下文压缩",
                "input": ["text"],
            },
        }
        self._role_owners: dict[str, str] = {}
        self._saved_role_selectors: dict[str, Any] | None = None
        self._deferred_role_selectors: dict[str, str] = {}
        self._registry_persistence = False
        self._roles_ready = threading.Event()
        self._roles_ready.set()

    def _load_role_selectors(self) -> dict[str, Any]:
        if self._saved_role_selectors is None:
            settings = self._settings or self._settings_store()
            self._saved_role_selectors = dict(settings.get("llm.roles") or {})
        return self._saved_role_selectors

    @staticmethod
    def _selector_alias(pattern: str) -> tuple[str | None, str]:
        target, separator, level = pattern.strip().rpartition(":")
        suffix = ""
        if separator and level in (
            "off",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
            "auto",
        ):
            pattern = target
            suffix = f":{level}"
        else:
            pattern = pattern.strip()
        if pattern == "*":
            return "default", suffix
        if pattern.startswith("@"):
            return pattern[1:], suffix
        if pattern.startswith("pi/"):
            return pattern[3:], suffix
        return None, suffix

    def _default_selector(self, saved: dict[str, Any]) -> str:
        def expand(role: str, seen: set[str]) -> str:
            if role in seen or not isinstance(saved.get(role), str):
                raise ValueError("Default role has no model selector")
            patterns = []
            for pattern in saved[role].split(","):
                target, suffix = self._selector_alias(pattern)
                value = (
                    expand(target, seen | {role})
                    if target is not None
                    else pattern.strip()
                )
                if target is not None and suffix:
                    candidates = []
                    for item in value.split(","):
                        _, previous_suffix = self._selector_alias(item)
                        candidates.append(
                            (
                                item.removesuffix(previous_suffix)
                                if previous_suffix
                                else item
                            )
                            + suffix
                        )
                    value = ",".join(candidates)
                patterns.append(value)
            return ",".join(patterns)

        return expand("default", set())

    def _role_selectors(self) -> dict[str, Any]:
        saved = self._load_role_selectors()
        selectors = {role: saved.get(role, "@default") for role in self._roles}
        self._deferred_role_selectors = {}
        deferred = {
            role
            for role, selector in selectors.items()
            if isinstance(selector, str)
            and any(
                target is not None and target not in self._roles
                for part in selector.split(",")
                for target, _ in (self._selector_alias(part),)
            )
        }
        while True:
            dependent = {
                role
                for role, selector in selectors.items()
                if isinstance(selector, str)
                and any(
                    self._selector_alias(part)[0] in deferred
                    for part in selector.split(",")
                )
            }
            if dependent <= deferred:
                break
            deferred.update(dependent)
        for role in deferred:
            replacement = (
                self._default_selector(saved) if role == "default" else "@default"
            )
            selectors[role] = replacement
            self._deferred_role_selectors[role] = replacement
        return selectors

    def list_roles(self) -> list[dict[str, Any]]:
        with self._roles_lock:
            return [
                dict(role, input=list(role["input"])) for role in self._roles.values()
            ]

    def _sync_roles(self) -> None:
        with self._lock:
            running = (
                not self._closed
                and self._process is not None
                and self._process.poll() is None
            )
        if not running:
            return
        self._roles_ready.wait()
        if self._reader_error is not None:
            raise RuntimeError(self._reader_error)
        with self._roles_lock:
            selectors = self._role_selectors()
            roles = self.list_roles()
            self._registry_persistence = True
        # The reader persists settings before it delivers the result.
        # Do not hold either state lock while this call waits for the reader.
        try:
            self.call(IpcMethod.ROLES_SYNC, roles=roles, selectors=selectors)
        finally:
            with self._roles_lock:
                self._registry_persistence = False

    def register_role(
        self, owner: str, role: str, title: str, *, input: tuple[str, ...] = ("text",)
    ) -> None:
        for name, value in (("owner", owner), ("role", role), ("title", title)):
            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
            ):
                raise ValueError(f"Role {name} must be a nonempty trimmed string")
        if (
            not isinstance(input, tuple)
            or not input
            or any(value not in ("text", "image") for value in input)
            or len(set(input)) != len(input)
        ):
            raise ValueError("Role input must contain unique text or image formats")
        with self._roles_sync_lock:
            if self._closed:
                raise RuntimeError("Agent runtime is closed")
            with self._roles_lock:
                if role in self._roles:
                    raise ValueError(f"Role {role!r} is already registered")
                self._load_role_selectors()
                self._roles[role] = {"id": role, "title": title, "input": list(input)}
                self._role_owners[role] = owner
            try:
                self._sync_roles()
            except Exception:
                with self._roles_lock:
                    del self._roles[role]
                    del self._role_owners[role]
                raise

    def unregister_roles(self, owner: str) -> None:
        with self._roles_sync_lock:
            with self._roles_lock:
                removed = {
                    role: self._roles[role]
                    for role, registered_owner in self._role_owners.items()
                    if registered_owner == owner
                }
                if not removed:
                    return
                saved = self._load_role_selectors()
                previous = {role: saved[role] for role in removed if role in saved}
                affected = set(removed)
                while True:
                    dependent = {
                        role
                        for role, selector in saved.items()
                        if isinstance(selector, str)
                        and any(
                            self._selector_alias(part)[0] in affected
                            for part in selector.split(",")
                        )
                    }
                    if dependent <= affected:
                        break
                    affected.update(dependent)
                default_selector = (
                    self._default_selector(saved) if "default" in affected else None
                )
                for role in removed:
                    del self._roles[role]
                    del self._role_owners[role]
                    saved.pop(role, None)
                # References to removed roles must not survive plugin unload.
                changed = {
                    role: selector
                    for role, selector in saved.items()
                    if role in affected
                }
                for role in changed:
                    saved[role] = default_selector if role == "default" else "@default"
            try:
                self._sync_roles()
            except Exception:
                with self._roles_lock:
                    self._roles.update(removed)
                    self._role_owners.update({role: owner for role in removed})
                    saved.update(previous)
                    saved.update(changed)
                raise

    @property
    def runtime_dir(self) -> Path:
        if self._runtime_dir is None:
            raise RuntimeError("Agent runtime has not started")
        return self._runtime_dir

    def _settings_store(self):
        from plugins.settings.store import SettingsStore

        return SettingsStore(self.data_dir / "settings.json")

    def _bootstrap(self):
        settings = self._settings or self._settings_store()
        auth_path = self.data_dir / "auth.json"
        auth = json.loads(auth_path.read_text("utf-8")) if auth_path.exists() else {}
        with self._roles_lock:
            saved = self._load_role_selectors()
            bootstrap = {
                "credentials": auth.get("llm", {}),
                "registered_roles": self.list_roles(),
            }
            if saved:
                bootstrap["roles"] = {**saved, **self._role_selectors()}
            self._registry_persistence = True
        models = settings.get("llm.models")
        if models is not None:
            bootstrap["models"] = models
        return bootstrap

    def _persist(self, message):
        if "settings" in message:
            if "roles" in message["settings"]:
                with self._roles_lock:
                    # Retain saved selectors for plugins that register later.
                    saved = self._load_role_selectors()
                    for role, selector in message["settings"]["roles"].items():
                        if (
                            self._registry_persistence
                            and self._deferred_role_selectors.get(role) == selector
                        ):
                            continue
                        saved[role] = selector
                        self._deferred_role_selectors.pop(role, None)
                    self._registry_persistence = False
            values = {f"llm.{key}": value for key, value in message["settings"].items()}
            if self._settings is not None:
                self._settings.set_many(values)
            else:
                self._settings_store().update(values)
        if "credentials" in message:
            auth_path = self.data_dir / "auth.json"
            auth = (
                json.loads(auth_path.read_text("utf-8")) if auth_path.exists() else {}
            )
            auth["llm"] = message["credentials"]
            descriptor, name = tempfile.mkstemp(
                prefix="auth-", suffix=".tmp", dir=self.data_dir
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                    json.dump(auth, output, ensure_ascii=False, indent=2)
                os.replace(name, auth_path)
            finally:
                Path(name).unlink(missing_ok=True)

    def _start(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("Agent runtime is closed")
            if self._process is not None and self._process.poll() is None:
                return
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if self._reader is not None:
                self._reader.join()
            # Agent transcripts and session pointers must survive backend
            # restarts so reopen_session can restore full agent context.
            self._runtime_dir = self.data_dir / "llm_runtime"
            self._runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            bootstrap = self.runtime_dir / "bootstrap.json"
            bootstrap.write_text(
                json.dumps(self._bootstrap(), ensure_ascii=False), "utf-8"
            )
            bootstrap.chmod(0o600)
            self._reader_error = None
            entry = Path(__file__).with_name("main.ts")
            self._roles_ready.clear()
            self._process = subprocess.Popen(
                ["bun", str(entry), str(self.runtime_dir)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            process = self._process
            self._reader = threading.Thread(
                target=self._read, args=(process,), daemon=True
            )
            self._reader.start()
            threading.Thread(target=self._logs, args=(process,), daemon=True).start()

    def _logs(self, process):
        for line in process.stderr:
            logging.getLogger("lecture_agent.runtime.bun").info("%s", line.rstrip())

    def set_callback_handler(
        self, handler: Callable[[dict[str, Any]], Any] | None
    ) -> None:
        with self._lock:
            self._callback_handler = handler

    def _callback(self, process, message, handler):
        if self._closed or self._process is not process:
            return
        callback_id = message["id"]
        try:
            request = {
                key: message[key]
                for key in ("session_id", "role", "kind", "name", "arguments")
            }
            if handler is None:
                raise RuntimeError("Agent callback handler is not registered")
            response = json.dumps(
                {"callback_id": callback_id, "result": handler(request)},
                ensure_ascii=False,
            )
        except Exception as exc:
            response = json.dumps(
                {"callback_id": callback_id, "error": str(exc) or type(exc).__name__}
            )
        with self._write_lock:
            if (
                self._closed
                or self._process is not process
                or process.poll() is not None
                or self._reader_error is not None
            ):
                return
            try:
                process.stdin.write(response + "\n")
                process.stdin.flush()
            except (OSError, ValueError):
                # EOF can close the pipe while a handler is still running.
                return

    def _read(self, process):
        try:
            for line in process.stdout:
                message = json.loads(line)
                if message.get("type") == "persistence":
                    self._persist(message)
                    continue
                if message.get("type") == "callback":
                    threading.Thread(
                        target=self._callback,
                        args=(process, message, self._callback_handler),
                        daemon=True,
                    ).start()
                    continue
                message["type"] = RuntimeFrameType(message["type"])
                with self._roles_lock:
                    self._registry_persistence = False
                self._roles_ready.set()
                with self._pending_lock:
                    target = self._pending.get(str(message.get("id", "")))
                if target is not None:
                    target.put(message)
        except Exception as exc:
            reason = str(exc)
            if process.poll() is None:
                process.kill()
                process.wait()
        else:
            reason = f"Agent runtime exited ({process.wait()})"
        with self._pending_lock:
            if self._process is process:
                self._reader_error = reason
                for target in self._pending.values():
                    target.put({"type": RuntimeFrameType.ERROR, "error": reason})
        self._roles_ready.set()

    def stream(self, method: IpcMethod, *, on_sent=None, **params):
        if not isinstance(method, IpcMethod):
            raise TypeError("Runtime method must be an IpcMethod")
        with self._lock:
            if method in (IpcMethod.CANCEL, IpcMethod.CLOSE_SESSION, IpcMethod.MESSAGE):
                if (
                    self._closed
                    or self._process is None
                    or self._process.poll() is not None
                ):
                    if method == IpcMethod.MESSAGE:
                        raise RuntimeError("Agent runtime is closed")
                    return
            else:
                self._start()
        request_id = uuid.uuid4().hex
        target = queue.Queue()
        with self._pending_lock:
            if self._reader_error is not None:
                raise RuntimeError(self._reader_error)
            self._pending[request_id] = target
        try:
            with self._write_lock:
                process = self._process
                assert process is not None and process.stdin is not None
                process.stdin.write(
                    json.dumps(
                        {"id": request_id, "method": method, "params": params},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                process.stdin.flush()
            if on_sent is not None:
                on_sent()
            while True:
                message = target.get()
                if message["type"] == RuntimeFrameType.ERROR:
                    raise RuntimeError(message["error"])
                yield message
                if message["type"] == RuntimeFrameType.RESULT:
                    break
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

    def call(self, method: IpcMethod, **params) -> dict[str, Any]:
        for message in self.stream(method, **params):
            if message["type"] == RuntimeFrameType.RESULT:
                return message["result"]
        if method in (IpcMethod.CANCEL, IpcMethod.CLOSE_SESSION):
            return {}
        raise RuntimeError("Agent runtime returned no result")

    def cancel(self, session_id: str):
        return self.call(IpcMethod.CANCEL, session_id=session_id)

    def session_transcript(self, session_id: str) -> Path | None:
        """Return the transcript path if the runtime can reopen the session."""

        result = self.call(IpcMethod.SESSION_TRANSCRIPT, session_id=session_id)
        path = result["path"]
        return Path(path) if path is not None else None

    def close(self):
        with self._lock:
            process = self._process
            self._closed = True
        if process is None:
            self._runtime_dir = None
            return
        if process.poll() is None:
            assert process.stdin is not None
            with self._write_lock:
                process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if self._reader is not None:
            self._reader.join()
        self._runtime_dir = None
