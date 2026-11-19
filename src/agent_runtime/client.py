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
from typing import Any

from core.agent_types import IpcMethod, RuntimeFrameType


class RuntimeClient:
    def __init__(self, data_dir: Path, *, settings=None):
        self.data_dir = Path(data_dir).resolve()
        self._settings = settings
        self._temporary: tempfile.TemporaryDirectory | None = None
        self._reader: threading.Thread | None = None
        self._process = None
        self._lock = threading.RLock()
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[str, queue.Queue] = {}
        self._reader_error: str | None = None
        self._closed = False

    @property
    def runtime_dir(self) -> Path:
        if self._temporary is None:
            raise RuntimeError("Agent runtime has not started")
        return Path(self._temporary.name)

    def _settings_store(self):
        from plugins.settings.store import SettingsStore

        return SettingsStore(self.data_dir / "settings.json")

    def _bootstrap(self):
        settings = self._settings or self._settings_store()
        auth_path = self.data_dir / "auth.json"
        auth = json.loads(auth_path.read_text("utf-8")) if auth_path.exists() else {}
        bootstrap = {"credentials": auth.get("llm", {})}
        for key in ("models", "roles"):
            value = settings.get(f"llm.{key}")
            if value is not None:
                bootstrap[key] = value
        return bootstrap

    def _persist(self, message):
        if "settings" in message:
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
            if self._temporary is not None:
                self._temporary.cleanup()
            self._temporary = tempfile.TemporaryDirectory(prefix="skipper-llm-")
            bootstrap = self.runtime_dir / "bootstrap.json"
            bootstrap.write_text(
                json.dumps(self._bootstrap(), ensure_ascii=False), "utf-8"
            )
            bootstrap.chmod(0o600)
            self._reader_error = None
            entry = Path(__file__).with_name("main.ts")
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

    def _read(self, process):
        try:
            for line in process.stdout:
                message = json.loads(line)
                if message.get("type") == "persistence":
                    self._persist(message)
                    continue
                message["type"] = RuntimeFrameType(message["type"])
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
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None
            return
        if process.poll() is None:
            assert process.stdin is not None
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
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
