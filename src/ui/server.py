"""FastAPI web adapter for the declarative UI and runtime."""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import SHUTDOWN_TIMEOUT
from core.runtime import Runtime

logger = logging.getLogger("lecture_agent.server")
STATIC_DIR = Path(__file__).with_name("static")


class ActionRequest(BaseModel):
    action: str
    values: dict[str, Any] = Field(default_factory=dict)
    page_id: str | None = None
    item_id: str | None = None


class _Broadcaster:
    """Send UI refresh notifications to connected WebSockets."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def add(self, ws: WebSocket) -> None:
        with self._lock:
            self._clients.add(ws)

    def remove(self, ws: WebSocket) -> None:
        with self._lock:
            self._clients.discard(ws)

    def broadcast(self, reason: str = "") -> None:
        if self._loop is None:
            return
        with self._lock:
            if not self._clients:
                return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(reason), self._loop)
        except RuntimeError:
            pass

    async def _broadcast(self, reason: str) -> None:
        with self._lock:
            clients = list(self._clients)
        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json({"type": "refresh", "reason": reason})
            except Exception:  # noqa: BLE001 - client disconnected
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    self._clients.discard(ws)


def _safe_artifact(root: Path, session_id: str, subpath: str) -> Path:
    sessions = (root / "sessions").resolve()
    candidate = (sessions / session_id / subpath).resolve()
    if not str(candidate).startswith(str(sessions) + "/"):
        raise HTTPException(status_code=404, detail="artifact not found")
    if candidate.name in {"settings.json", ".env"}:
        raise HTTPException(status_code=404, detail="artifact not found")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return candidate


def _stop_runtime(runtime: Runtime, timeout: float = SHUTDOWN_TIMEOUT) -> None:
    """Tear the runtime down, but never let a wedged plugin block Ctrl+C.

    ``runtime.stop()`` talks to the network (STT/LLM teardown), so it can block
    for a long time. Running it on a daemon thread means the interpreter can
    still exit once ``timeout`` expires.
    """

    done = threading.Event()

    def _run() -> None:
        try:
            runtime.stop()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            logger.exception("runtime stop failed")
        finally:
            done.set()

    threading.Thread(target=_run, name="runtime-stop", daemon=True).start()
    if not done.wait(timeout):
        logger.warning("runtime stop still running after %ss; abandoning it", timeout)


def create_app(runtime: Runtime) -> FastAPI:
    broadcaster = _Broadcaster()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_runtime = not runtime.started
        broadcaster.bind_loop(asyncio.get_running_loop())
        subscription = runtime.bus.on(
            "ui.changed",
            lambda event: broadcaster.broadcast(event.get("reason", "")),
            plugin_id="server",
        )
        if owns_runtime:
            runtime.start()
        app.state.runtime = runtime
        app.state.ui_subscription = subscription
        try:
            yield
        finally:
            subscription.cancel()
            if owns_runtime:
                _stop_runtime(runtime)

    app = FastAPI(title="Lecture Agent", version="0.1.0", lifespan=lifespan)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    if not runtime.ui.has_action("ui.open_page"):
        runtime.ui.register_action(
            "ui.open_page",
            lambda payload: {
                "ok": True,
                "open_page": payload.get("page_id") or payload.get("value"),
            },
            plugin_id="server",
        )

    @app.get("/")
    def index() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=503, detail="UI static files are missing")
        return FileResponse(index_path)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "started": runtime.started,
            "session": runtime.session_id,
            "plugin_order": runtime.plugins.order,
        }

    @app.get("/api/ui/state")
    def ui_state() -> JSONResponse:
        return JSONResponse(runtime.ui.state())

    @app.get("/api/runtime")
    def runtime_status() -> JSONResponse:
        return JSONResponse(runtime.status())

    @app.post("/api/ui/action")
    def ui_action(request: ActionRequest) -> JSONResponse:
        result = runtime.ui.dispatch_action(
            request.action,
            request.values,
            page_id=request.page_id,
            item_id=request.item_id,
        )
        return JSONResponse(
            {
                "ok": result.ok,
                "message": result.message,
                "refresh": result.refresh,
                "open_page": result.open_page,
                "data": result.data,
            }
        )

    @app.post("/api/ui/pages/{page_id}/open")
    def open_page(page_id: str) -> JSONResponse:
        try:
            runtime.ui.open_page(page_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="page not found")
        return JSONResponse({"ok": True, "active_page": page_id})

    @app.post("/api/ui/pages/{page_id}/close")
    def close_page(page_id: str) -> JSONResponse:
        runtime.ui.close_page(page_id)
        return JSONResponse({"ok": True})

    @app.get("/api/artifacts/{session_id}/{subpath:path}")
    def artifact(session_id: str, subpath: str, request: Request) -> FileResponse:
        path = _safe_artifact(runtime.data_dir, session_id, subpath)
        return FileResponse(path)

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        broadcaster.add(ws)
        try:
            await ws.send_json({"type": "refresh", "reason": "connected"})
            while True:
                # Keep the connection alive; clients may send pings.
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            pass
        finally:
            broadcaster.remove(ws)

    return app
