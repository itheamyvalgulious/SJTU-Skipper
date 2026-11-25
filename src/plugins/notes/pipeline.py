from __future__ import annotations

import json
import math
import os
import shutil
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core.agent_types import (
    AgentEventType,
    AgentToolName,
    ExecutionRole,
    ExecutionStatus,
    ExecutionBlockKind,
    ExecutionBlockStatus,
    GenerationErrorCode,
)
from core.file_access import publication_lock
from core.events import Event
from core.navigation import current_navigation
from core.streams import StreamKind
from core.ui import StatusLabel, TaskState
from core.settings_spec import SettingsSpec, SettingType
from core.workspace import FileWorkspace
from core.note_document import parse_blocks, strip_blocks

from .material import chunk_material, collect_material, digest, read_log, render
from .types import (
    ChunkState,
    ContinuationStage,
    GenerationState,
    NotePhase,
    NotePublication,
    NoteTaskKind,
)
from .prompts import (
    CHUNK,
    REVIEW,
    KINDS,
    LABELS,
    MODE_LABELS,
    NOTE_HEADERS,
    REVIEW_SYSTEM,
    SOURCE_FILENAME,
    SYSTEM,
    WRITER,
    system_prompt_for_mode,
)


class NotesPipeline:
    id = "notes"

    def setup(self, ctx):
        self.ctx = ctx
        for role, title in (
            ("note_chunk", "分段笔记"),
            ("note_writer", "笔记写作"),
            ("note_review", "笔记审阅"),
        ):
            ctx.llm.register_role(ctx.plugin_id, role, title)
        self.settings = ctx.require("settings")
        self.agent = ctx.require("agent")
        for key, default, kind, label, section, description, order, *extra in (
            (
                "chunks_enabled",
                True,
                SettingType.BOOL,
                "自动生成分段笔记",
                "分段笔记",
                "",
                10,
            ),
            (
                "chunk_seconds",
                600,
                SettingType.INT,
                "分段时长（秒）",
                "分段笔记",
                "每段笔记覆盖的课次时长，至少为 1 秒．",
                20,
            ),
            (
                "overlap_seconds",
                60,
                SettingType.INT,
                "前文时长（秒）",
                "分段笔记",
                "生成分段笔记时额外提供的前文范围．",
                30,
            ),
            (
                "concurrency",
                8,
                SettingType.INT,
                "笔记任务并发数",
                "任务并发",
                "修改后重启服务生效．",
                70,
            ),
            (
                "review_enabled",
                True,
                SettingType.BOOL,
                "审阅完整笔记",
                "笔记审阅",
                "",
                80,
            ),
            (
                "review_max_rounds",
                0,
                SettingType.INT,
                "审阅轮数上限（轮）",
                "笔记审阅",
                "0 表示不限制审阅轮数．",
                90,
            ),
            (
                "generation_mode",
                "default",
                SettingType.SELECT,
                "笔记生成模式",
                "提示词",
                "选择完整笔记使用的提示词；默认保持完整笔记提示词不变．",
                190,
                tuple((key, label) for key, label in MODE_LABELS.items()),
            ),
            (
                "chunk_system_prompt",
                SYSTEM,
                SettingType.TEXT,
                "分段笔记提示词",
                "提示词",
                "留空时使用默认提示词．",
                200,
            ),
            (
                "system_prompt",
                SYSTEM,
                SettingType.TEXT,
                "完整笔记提示词",
                "提示词",
                "留空时使用默认提示词．",
                210,
            ),
            (
                "reviewer_prompt",
                REVIEW_SYSTEM,
                SettingType.TEXT,
                "笔记审阅提示词",
                "提示词",
                "留空时使用默认提示词．",
                220,
            ),
        ):
            self.settings.register(
                SettingsSpec(
                    key=f"notes.{key}",
                    default=default,
                    type=kind,
                    label=label,
                    group="笔记",
                    section=section,
                    order=order,
                    description=description,
                ),
                plugin_id=self.id,
            )
        self.condition = threading.Condition(threading.RLock())
        self.sessions = {}
        self.queue = deque()
        self.active = {}
        self.agent_tasks = {}
        self.writer_messages = {}
        self.continuations = {}
        self.closed = False
        self.limit = max(1, int(self.setting("notes.concurrency", 8)))
        self.pool = ThreadPoolExecutor(
            max_workers=self.limit, thread_name_prefix="notes"
        )
        self.agent_running = 0
        for path in ctx.data_dir.glob("sessions/*/note_work/state.json"):
            sid = path.parent.parent.name
            if sid != "boot":
                self._state(sid)
        ctx.ui.register_task_provider(self.running_tasks, plugin_id=self.id)
        ctx.ui.register_action("notes.message", self.message, plugin_id=self.id)
        for event in (
            "session.started",
            "stream.sealed",
            "stream.appended",
            "session.stopped",
        ):
            ctx.events.on(event, self._event, plugin_id=self.id)
        for event in (AgentEventType.RUN_READY,):
            ctx.events.on(event, self._agent_event, plugin_id=self.id)
        self.coordinator = threading.Thread(
            target=self._coordinate, name="notes-coordinator", daemon=True
        )
        self.coordinator.start()

    def setting(self, key, default):
        return self.settings.get(key, default)

    def _sid(self, sid=None):
        sid = current_navigation().class_id if sid is None else sid
        return "" if not sid or sid == "boot" else str(sid)

    def _directory(self, sid):
        return self.ctx.data_dir / "sessions" / sid / "note_work"

    def _state(self, sid):
        if not sid or sid == "boot":
            raise ValueError("A class ID is required")
        if sid not in self.sessions:
            path = self._directory(sid) / "state.json"
            state: dict[str, Any] = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.exists()
                else {
                    "session_id": sid,
                    "source_digest": "",
                    "chunks": [],
                    "published_end": 0.0,
                    "finalize_requested": False,
                    "phase": NotePhase.COLLECTING,
                    "error": "",
                }
            )
            restored = path.exists()
            state["phase"] = NotePhase(state["phase"])
            state.setdefault("review_passed", False)
            for chunk in state["chunks"]:
                chunk["state"] = ChunkState(chunk["state"])
                if (
                    chunk["state"] in (ChunkState.RUNNING, ChunkState.PENDING)
                    and restored
                ):
                    chunk["state"] = ChunkState.CANCELLED
                if chunk["state"] == ChunkState.DONE:
                    root = self._directory(sid) / "chunks" / str(chunk["index"])
                    if not all((root / name).exists() for name in NOTE_HEADERS):
                        chunk["state"] = ChunkState.PENDING
            state.setdefault("writer", None)
            state.setdefault("deleted", False)
            state["errors"] = [
                {"chunk": chunk["index"], "error": chunk["error"]}
                for chunk in state["chunks"]
                if chunk["state"] == ChunkState.FAILED
            ]
            for record in read_log(
                self.ctx.data_dir / "sessions" / sid / "visual.jsonl"
            ):
                if detail := record.get("error"):
                    state["errors"].append(
                        {
                            "source": StreamKind.VISUAL,
                            "start": record["start"],
                            "end": record["end"],
                            "error": detail["message"],
                            "code": detail["code"],
                            "attempts": detail["attempts"],
                        }
                    )
            if restored:
                state["finalize_requested"] = False
                if state["phase"] not in (NotePhase.PUBLISHED, NotePhase.DELETED):
                    state["phase"] = NotePhase.CANCELLED
                state["restart_cancelled"] = True
            self.sessions[sid] = state
        return self.sessions[sid]

    def _generation_state(self, sid):
        state = self._state(sid)
        if any(task[0] == sid for task in self.queue) or any(
            task["session_id"] == sid
            for task in (*self.active.values(), *self.agent_tasks.values())
        ):
            return GenerationState.GENERATING
        if state.get("deleted"):
            return GenerationState.NOT_STARTED
        session = self.ctx.streams.get_session(sid)
        cancelled = session is not None and (
            session.meta.get("input_failed") or session.meta.get("processing_cancelled")
        )
        if (
            state.get("restart_cancelled")
            or cancelled
            or state["phase"]
            in (NotePhase.PUBLISHED, NotePhase.FAILED, NotePhase.CANCELLED)
        ):
            return GenerationState.FINISHED
        if any(
            chunk["state"] in (ChunkState.PENDING, ChunkState.RUNNING)
            for chunk in state["chunks"]
        ):
            return GenerationState.GENERATING
        _, final = self._readiness(sid, [])
        if state["finalize_requested"] or (
            not final
            and self.setting("notes.chunks_enabled", True)
            and session is not None
        ):
            return GenerationState.GENERATING
        return (
            GenerationState.FINISHED if state["chunks"] else GenerationState.NOT_STARTED
        )

    def _save(self, state):
        directory = self._directory(state["session_id"])
        directory.mkdir(parents=True, exist_ok=True)
        fields = (
            "session_id",
            "source_digest",
            "chunks",
            "published_end",
            "finalize_requested",
            "phase",
            "error",
            "review_passed",
            "deleted",
            "writer",
        )
        path = directory / "state.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({key: state[key] for key in fields}, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _event(self, event):
        sid = str(event.get("session_id", ""))
        if not sid or sid == "boot":
            return
        session = self.ctx.streams.get_session(sid)
        if getattr(event, "type", "") == "session.started":
            if session is None or event.get("started_at") != session.started_at:
                return
            with self.condition:
                self._state(sid).pop("restart_cancelled", None)
        with self.condition:
            state = self._state(sid)
            if event.get("error"):
                failure = {"source": event.get("stream"), **event.get("error")}
                key = lambda row: tuple(
                    row.get(name) for name in ("source", "start", "end", "code")
                )
                if failure.get("code") != GenerationErrorCode.THINKING_LOOP or not any(
                    key(row) == key(failure) for row in state["errors"]
                ):
                    state["errors"].append(failure)
            self.condition.notify_all()

    def start(self, session_id=None) -> bool:

        sid = self._sid(session_id)
        if not sid or sid == "boot":
            return False
        records = collect_material(self.ctx, sid)
        session = self.ctx.streams.get_session(sid)
        if not records and (session is None or session.ended_at is not None):
            return False
        with self.condition:
            if self.closed:
                return False
            state = self._state(sid)
            if self._generation_state(sid) != GenerationState.NOT_STARTED:
                return False
            state["deleted"] = False
            state.pop("restart_cancelled", None)
            state["phase"] = NotePhase.COLLECTING
            state["finalize_requested"] = True
            self._save(state)
            self.condition.notify_all()
        return True

    def finalize(self, session_id=None):
        sid = self._sid(session_id)
        if not sid:
            return False
        with self.condition:
            state = self._state(sid)
            if (
                state.get("restart_cancelled")
                or state["phase"]
                in (NotePhase.PUBLISHED, NotePhase.FAILED, NotePhase.CANCELLED)
                or state.get("deleted")
            ):
                return False
            state["finalize_requested"] = True
            self.condition.notify_all()
        return True

    def _writer_inputs(self, sid, writer):
        root = self._directory(sid)
        files = {name: root / relative for name, relative in writer["files"].items()}
        for path in files.values():
            path.read_text(encoding="utf-8")
        notes = self.ctx.data_dir / "sessions" / sid / "notes"
        for name in NOTE_HEADERS:
            (notes / name).read_text(encoding="utf-8")
        return files

    def message(self, payload):
        sid = str(payload.get("session_id") or "")
        target = str(payload.get("agent_session_id") or "")
        text = str(payload.get("message") or "")

        def reject(reason):
            return {"ok": False, "message": reason, "refresh": False}

        if not sid or not target or not text.strip():
            return reject("消息或会话记录不可用")
        with self.condition:
            state = self._state(sid)
            if self.closed or state.get("deleted"):
                return reject("笔记任务已关闭")
            writer = state.get("writer")
            is_writer = writer is not None and target == writer["session_id"]
            stage = self.continuations.get(sid) if is_writer else None
            if stage == ContinuationStage.QUEUED:
                self.writer_messages[sid].append(text)
                return {"ok": True, "refresh": True}
            if stage == ContinuationStage.STARTING:
                return reject("笔记写作正在恢复，请稍后发送")
            if stage == ContinuationStage.FINISHING:
                return reject("笔记写作正在发布，请稍后发送")
            live = any(
                task["session_id"] == sid
                and any(
                    agent.get("session_id") == target
                    for agent in task["agents"].values()
                )
                for task in self.agent_tasks.values()
            )
            if not live:
                if not writer:
                    return reject("笔记写作会话记录不可用")
                if not is_writer or not (
                    state["phase"] == NotePhase.PUBLISHED
                    or (
                        state["phase"] == NotePhase.FAILED
                        and not state["finalize_requested"]
                    )
                ):
                    return reject("会话已完成，不能发送消息")
                if any(task[0] == sid for task in self.queue) or any(
                    task["session_id"] == sid
                    for task in (*self.active.values(), *self.agent_tasks.values())
                ):
                    return reject("笔记任务尚未结束，请稍后发送")
                try:
                    self._writer_inputs(sid, writer)
                except (OSError, ValueError, RuntimeError) as exc:
                    return reject(f"笔记写作会话记录不可用：{exc}")
                self.writer_messages[sid] = [text]
                self.continuations[sid] = ContinuationStage.QUEUED
                # Only initial publication keeps this request set after a failure.
                state.update(
                    phase=NotePhase.WRITING,
                    review_passed=False,
                    error="",
                    finalize_requested=False,
                )
                state.pop("restart_cancelled", None)
                self.queue.append((sid, NoteTaskKind.FINAL, -1))
                self._save(state)
                self.condition.notify_all()
                return {"ok": True, "refresh": True}
            if state.get("restart_cancelled"):
                return reject("笔记任务正在停止")
        result = self.agent.message(target, text)
        return {
            "ok": result["accepted"],
            "message": result.get("reason", ""),
            "refresh": False,
        }

    def stop(self, session_id=None):
        sid = self._sid(session_id)
        if not sid:
            return False
        with self.condition:
            state = self._state(sid)
            if self._generation_state(sid) != GenerationState.GENERATING:
                return False
            state["restart_cancelled"] = True
            state["finalize_requested"] = False
            self.queue = deque(task for task in self.queue if task[0] != sid)
            self.writer_messages.pop(sid, None)
            self.continuations.pop(sid, None)
            for chunk in state["chunks"]:
                if chunk["state"] in (ChunkState.PENDING, ChunkState.RUNNING):
                    chunk["state"] = ChunkState.CANCELLED
            running = [
                key
                for key, task in self.agent_tasks.items()
                if task["session_id"] == sid
            ]
            self.condition.notify_all()
        for key in running:
            self.agent.cancel(key)
        with self.condition:
            self.condition.wait_for(
                lambda: (
                    not any(task["session_id"] == sid for task in self.active.values())
                    and not any(
                        task["session_id"] == sid for task in self.agent_tasks.values()
                    )
                )
            )
            state["phase"] = NotePhase.CANCELLED
            self._save(state)
        self.ctx.ui.refresh("notes.stopped")
        return True

    def reset(self, session_id=None):
        sid = self._sid(session_id)
        if not sid or sid == "boot":
            return False
        self.stop(sid)
        with self.condition:
            self.queue = deque(task for task in self.queue if task[0] != sid)
            self.writer_messages.pop(sid, None)
            self.continuations.pop(sid, None)
            root = self.ctx.data_dir / "sessions" / sid
            with publication_lock(root / "notes"):
                for name in ("note_work", "notes"):
                    path = root / name
                    if path.exists():
                        shutil.rmtree(path)
                (root / SOURCE_FILENAME).unlink(missing_ok=True)
            self.sessions.pop(sid, None)
            state = self._state(sid)
            state["restart_cancelled"] = True
            state["errors"] = []
            self.condition.notify_all()
        self.ctx.ui.refresh("notes.reset")
        return True

    def delete(self, session_id=None):
        sid = self._sid(session_id)
        if not sid:
            return False
        with self.condition:
            state = self._state(sid)
            if self._generation_state(sid) != GenerationState.FINISHED:
                return False
            notes = self.ctx.data_dir / "sessions" / sid / "notes"
            with publication_lock(notes):
                for kind in NotePublication:
                    for name, _ in self.note_files(kind).values():
                        (notes / name).unlink(missing_ok=True)
                (notes.parent / SOURCE_FILENAME).unlink(missing_ok=True)
            state.update(
                chunks=[],
                source_digest="",
                published_end=0.0,
                phase=NotePhase.DELETED,
                error="",
                errors=[],
                review_passed=False,
                deleted=True,
                finalize_requested=False,
                last_agent_tasks=[],
                writer=None,
            )
            self.writer_messages.pop(sid, None)
            self.continuations.pop(sid, None)
            self._save(state)
        self.ctx.ui.refresh("notes.deleted")
        return True

    def status(self, session_id=None):
        sid = self._sid(session_id)
        if not sid:
            return {
                "session_id": "",
                "generation_state": GenerationState.NOT_STARTED,
                "phase": NotePhase.COLLECTING,
                "pending_chunks": 0,
                "running_chunks": 0,
                "published_end": 0.0,
                "review_passed": False,
                "completed_chunks": 0,
                "total_chunks": 0,
                "expected_chunks": None,
                "chunk_enabled": bool(self.setting("notes.chunks_enabled", True)),
                "tasks": [],
                "error": "",
                "errors": [],
            }
        with self.condition:
            state = self._state(sid)
            session = self.ctx.streams.get_session(sid)
            duration = (
                float(session.meta.get("total_media_seconds") or 0) if session else 0
            )
            chunk_enabled = bool(self.setting("notes.chunks_enabled", True))
            expected_chunks = (
                math.ceil(
                    duration / max(1, float(self.setting("notes.chunk_seconds", 600)))
                )
                if duration > 0 and chunk_enabled
                else (0 if not chunk_enabled else None)
            )
            history = list(state.get("last_agent_tasks", []))
            writer = state.get("writer")
            if (
                writer
                and state["phase"] == NotePhase.PUBLISHED
                and not any(
                    agent.get("session_id") == writer["session_id"]
                    for task in (*history, *self.agent_tasks.values())
                    for agent in task["agents"].values()
                )
            ):
                main = self._execution_agent(
                    "main",
                    ExecutionRole.WRITER,
                    writer["name"],
                    session_id=writer["session_id"],
                )
                main["status"] = ExecutionStatus.COMPLETED
                history.append(
                    {
                        "id": f"notes-agent:{writer['session_id']}",
                        "session_id": sid,
                        "started": 0,
                        "agents": {"main": main},
                    }
                )
            return {
                "session_id": sid,
                "generation_state": self._generation_state(sid),
                "phase": state["phase"],
                "pending_chunks": sum(
                    c["state"] == ChunkState.PENDING for c in state["chunks"]
                ),
                "running_chunks": sum(
                    c["state"] == ChunkState.RUNNING for c in state["chunks"]
                ),
                "published_end": state["published_end"],
                "review_passed": state["review_passed"],
                "completed_chunks": sum(
                    c["state"] in (ChunkState.DONE, ChunkState.EMPTY)
                    for c in state["chunks"]
                ),
                "total_chunks": len(state["chunks"]),
                "expected_chunks": expected_chunks,
                "chunk_enabled": chunk_enabled,
                "tasks": sorted(
                    [
                        *history,
                        *(
                            dict(task)
                            for task in self.agent_tasks.values()
                            if sid == task["session_id"]
                        ),
                    ],
                    key=lambda task: task["started"],
                ),
                "error": state["error"],
                "errors": list(state["errors"]),
            }

    def note_files(self, kind: NotePublication = NotePublication.FINAL):
        kind = NotePublication(kind)
        prefix = "chunk_note" if kind == NotePublication.CHUNK else "note"
        return {key: (f"{prefix}_{key}.md", LABELS[key]) for key in KINDS}

    def draft_paths(self, session_id=None):
        sid = self._sid(session_id)
        if not sid:
            return {}
        directory = self.ctx.data_dir / "sessions" / sid / "notes"
        return {
            key: directory / name
            for key, (name, _) in self.note_files(NotePublication.CHUNK).items()
            if (directory / name).exists()
        }

    def source_path(self, session_id=None):
        sid = self._sid(session_id)
        if not sid:
            return None
        path = self.ctx.data_dir / "sessions" / sid / SOURCE_FILENAME
        return path if path.exists() else None

    def running_tasks(self):
        with self.condition:
            tasks = []
            for task in self.active.values():
                sid = task["session_id"]
                phase = (
                    NotePhase.CHUNKING
                    if task["phase"] == NotePhase.CHUNKING
                    else self._state(sid)["phase"]
                )
                elapsed = max(0, int(time.monotonic() - task["started"]))
                tasks.append(
                    {
                        **task,
                        "phase": phase,
                        "state": TaskState.RUNNING,
                        "cancel": {
                            "action": "notes.stop",
                            "payload": {"session_id": sid},
                        },
                        "label": f"笔记 {sid}",
                        "detail": (
                            f"分段笔记；第 {task['chunk'] + 1} 段；{elapsed} 秒"
                            if phase == NotePhase.CHUNKING
                            else f"笔记写作；{elapsed} 秒"
                        ),
                    }
                )
            tasks.extend(
                {
                    "id": f"notes:{sid}:{kind}:{index}",
                    "session_id": sid,
                    "label": f"笔记 {sid}",
                    "detail": (
                        f"{StatusLabel.QUEUED}；分段笔记；第 {index + 1} 段"
                        if kind == NoteTaskKind.CHUNK
                        else f"{StatusLabel.QUEUED}；笔记写作"
                    ),
                    "phase": kind,
                    "state": TaskState.PENDING,
                    "cancel": {
                        "action": "notes.stop",
                        "payload": {"session_id": sid},
                    },
                    "chunk": index,
                }
                for sid, kind, index in self.queue
            )
            return tasks

    def _readiness(self, sid, records):
        session = self.ctx.streams.get_session(sid)
        if session is None:
            return math.inf, True
        states = []
        for service, stream in (
            ("stt", StreamKind.TRANSCRIPT),
            ("vlm", StreamKind.VISUAL),
        ):
            if self.ctx.service(service) is not None:
                progress = self.ctx.streams.text_progress(sid, stream)
                if progress.enabled:
                    states.append(progress)
        if not states:
            return (
                (math.inf, True)
                if session.input_done or session.ended_at is not None
                else (0.0, False)
            )
        return min(
            (math.inf if state.final else state.before) for state in states
        ), all(state.final for state in states) and (
            session.input_done or session.ended_at is not None
        )

    def _schedule(self, sid):
        state = self._state(sid)
        if sid in self.continuations:
            return
        if (
            state.get("deleted")
            or state.get("restart_cancelled")
            or state["phase"]
            in (NotePhase.PUBLISHED, NotePhase.FAILED, NotePhase.CANCELLED)
        ):
            return
        session = self.ctx.streams.get_session(sid)
        if session is not None and (
            session.meta.get("input_failed") or session.meta.get("processing_cancelled")
        ):
            return
        records = collect_material(self.ctx, sid)
        before, final = self._readiness(sid, records)
        if final and not records:
            state["finalize_requested"] = False
            self._save(state)
            return
        seconds = max(1, float(self.setting("notes.chunk_seconds", 600)))
        end = max(
            (
                float(record.get("end") or record.get("start") or 0)
                for record in records
            ),
            default=0,
        )
        if session is not None:
            end = max(end, session.duration)
        enabled = bool(self.setting("notes.chunks_enabled", True))
        manual = [record for record in records if record.get("start") is None]
        manual_before = max(
            (chunk.get("manual_end", 0) for chunk in state["chunks"]), default=0
        )
        count = int(end // seconds)
        if final and (enabled or state["finalize_requested"]) and end > count * seconds:
            count += 1
        if final and records and not count:
            count = 1
        if manual and not end:
            count = len(state["chunks"]) + int(len(manual) > manual_before)
        if enabled:
            for index in range(count):
                start, finish = index * seconds, min((index + 1) * seconds, end)
                if manual and not end:
                    start = finish = 0.0
                if not final and before < finish and end:
                    break
                core, context = chunk_material(
                    records,
                    start,
                    finish,
                    float(self.setting("notes.overlap_seconds", 60)),
                )
                existing = next(
                    (c for c in state["chunks"] if c["index"] == index), None
                )
                manual_start = (
                    existing.get("manual_start", 0) if existing else manual_before
                )
                manual_end = existing.get("manual_end", 0) if existing else len(manual)
                core += manual[manual_start:manual_end]
                input_digest = digest(core + context)
                if (
                    existing
                    and existing["input_digest"] != input_digest
                    and existing["state"] != ChunkState.RUNNING
                ):
                    existing.update(
                        state=ChunkState.PENDING,
                        input_digest=input_digest,
                        error="",
                        end=finish,
                    )
                if existing is None:
                    existing = {
                        "index": index,
                        "start": start,
                        "end": finish,
                        "input_digest": input_digest,
                        "state": ChunkState.PENDING,
                        "error": "",
                        "manual_start": manual_start,
                        "manual_end": manual_end,
                    }
                    state["chunks"].append(existing)
                    manual_before = manual_end
                task = (sid, NoteTaskKind.CHUNK, index)
                if existing["state"] == ChunkState.PENDING and task not in self.queue:
                    self.queue.append(task)
        state["source_digest"] = digest(records)
        if (
            final
            and state["finalize_requested"]
            and all(
                c["state"] in (ChunkState.DONE, ChunkState.EMPTY, ChunkState.FAILED)
                for c in state["chunks"]
            )
        ):
            task = (sid, NoteTaskKind.FINAL, -1)
            if (
                records
                and task not in self.queue
                and not any(t["session_id"] == sid for t in self.active.values())
                and state["phase"] not in (NotePhase.PUBLISHED, NotePhase.FAILED)
            ):
                self.queue.appendleft(task)
        self._save(state)

    def _coordinate(self):
        while True:
            with self.condition:
                if self.closed:
                    return
                ids = list(self.sessions)
            for sid in ids:
                try:
                    with self.condition:
                        self._schedule(sid)
                except Exception as exc:
                    with self.condition:
                        self._state(sid)["errors"].append({"error": str(exc)})
            with self.condition:
                while self.queue and len(self.active) < self.limit:
                    sid, kind, index = self.queue.popleft()
                    token = f"{sid}:{kind}:{index}"
                    state = self._state(sid)
                    if state.get("deleted") or state.get("restart_cancelled"):
                        continue
                    messages = None
                    if self.continuations.get(sid) == ContinuationStage.QUEUED:
                        messages = self.writer_messages.pop(sid)
                        self.continuations[sid] = ContinuationStage.STARTING
                    if kind == NoteTaskKind.CHUNK:
                        state["chunks"][index]["state"] = ChunkState.RUNNING
                        state["phase"] = NotePhase.CHUNKING
                    else:
                        state["phase"] = NotePhase.WRITING
                    self.active[token] = {
                        "id": token,
                        "session_id": sid,
                        "phase": NotePhase.CHUNKING
                        if kind == NoteTaskKind.CHUNK
                        else NotePhase.WRITING,
                        "chunk": index,
                        "started": time.monotonic(),
                        "agent_step": 0,
                    }
                    records = collect_material(self.ctx, sid)
                    self.pool.submit(
                        self._execute, token, sid, kind, index, records, messages
                    )
                self.condition.wait(0.1)

    def _execute(self, token, sid, kind, index, records, messages=None):
        try:
            if kind == NoteTaskKind.CHUNK:
                self._chunk(sid, index, records)
            else:
                self._final(sid, records, messages)
        except Exception as exc:
            with self.condition:
                if self._state(sid).get("deleted") or self._state(sid).get(
                    "restart_cancelled"
                ):
                    return
                if self.closed:
                    if kind == NoteTaskKind.CHUNK:
                        self._state(sid)["chunks"][index]["state"] = ChunkState.PENDING
                    self._save(self._state(sid))
                    return
                state = self._state(sid)
                state["errors"].append({"chunk": index, "error": str(exc)})
                if kind == NoteTaskKind.CHUNK:
                    state["chunks"][index].update(
                        state=ChunkState.FAILED, error=str(exc)
                    )
                else:
                    state["error"] = str(exc)
                    state["phase"] = NotePhase.FAILED
                self._save(state)
        finally:
            with self.condition:
                self.active.pop(token, None)
                if kind == NoteTaskKind.FINAL:
                    self.writer_messages.pop(sid, None)
                    self.continuations.pop(sid, None)
                self.condition.notify_all()

    @staticmethod
    def _execution_agent(key, role: ExecutionRole, label, model="", *, session_id=""):
        return {
            "id": key,
            "session_id": session_id,
            "role": ExecutionRole(role),
            "label": label,
            "status": ExecutionStatus.RUNNING,
            "review_passed": False,
            "model": model,
            "thinking": "",
            "context_tokens": 0,
            "context_window": 0,
            "llm_seconds": 0,
            "tool_seconds": 0,
            "output_tokens": 0,
            "tokens_per_second": 0,
            "blocks": [],
        }

    @staticmethod
    def _merge_execution_block(agent, block):
        block = {
            **block,
            "kind": ExecutionBlockKind(block["kind"]),
            "status": ExecutionBlockStatus(block["status"]),
        }
        for index, existing in enumerate(agent["blocks"]):
            if existing["id"] == block["id"]:
                agent["blocks"][index] = dict(block)
                return
        agent["blocks"].append(dict(block))

    def _review_execution_block(self, agent, event, **fields):
        block = {
            "id": f"{agent['id']}:review:{event.get('round')}",
            "kind": ExecutionBlockKind.REVIEW,
            "round": event.get("round"),
            **fields,
        }
        source_ids = set(event.get("block_ids") or ())
        blocks = []
        inserted = False
        for existing in agent["blocks"]:
            if existing["id"] in source_ids:
                if not inserted:
                    blocks.append(block)
                    inserted = True
            else:
                blocks.append(existing)
        agent["blocks"] = blocks
        self._merge_execution_block(agent, block)

    def _agent_event(self, event):
        with self.condition:
            task = self.agent_tasks.get(str(event.get("session_id", "")))
            if task is None:
                return
            if event.type == AgentEventType.RUN_READY:
                if (
                    task.get("ready")
                    and self.continuations.get(task["session_id"])
                    == ContinuationStage.STARTING
                ):
                    self.continuations[task["session_id"]] = ContinuationStage.LIVE
                return
            task["agent_step"] = int(event.get("step", task.get("agent_step", 0)))
            agents = task["agents"]
            reviewer = event.get("reviewer_id")
            key = reviewer or "main"
            name = event.get("event") if reviewer else event.type
            finished = event.get("finished") if reviewer else None
            if key not in agents:
                agents[key] = self._execution_agent(
                    key,
                    ExecutionRole.REVIEWER if reviewer else ExecutionRole.WRITER,
                    {
                        "note_review": "笔记审阅",
                    }.get(event.get("role"), task.get("label", "笔记写作")),
                    session_id=str(reviewer or event.get("session_id", "")),
                )
            agent = agents[key]
            now = time.monotonic()
            if name == AgentEventType.EXECUTION:
                agent.update(event.get("metrics") or {})
                if event.get("block") is not None:
                    self._merge_execution_block(agent, event.get("block"))
            elif name == AgentEventType.FILE:
                for child in agents.values():
                    child["review_passed"] = False
            elif finished is False:
                agent["status"] = ExecutionStatus.RUNNING
                agent["review_passed"] = False
            elif finished is True:
                agent["status"] = ExecutionStatus.COMPLETED
                agent["review_passed"] = False
                try:
                    report = json.loads(event.get("content"))
                    if not isinstance(report["correct"], bool):
                        raise ValueError("Review correctness must be a boolean")
                except (KeyError, ValueError, TypeError):
                    pass
                else:
                    agent["review_passed"] = report["correct"] and not event.get(
                        "stale"
                    )
                    self._review_execution_block(
                        agent,
                        event,
                        text=event.get("content"),
                        report=report,
                        status=ExecutionBlockStatus.COMPLETED,
                    )
            refresh = (
                now - task.get("last_refresh", 0) >= 1
                or finished is not None
                or name == AgentEventType.FILE
            )
            if refresh:
                task["last_refresh"] = now
                self.ctx.ui.refresh("notes.activity")

    def _check_running(self, sid):
        if (
            self.closed
            or self._state(sid).get("deleted")
            or self._state(sid).get("restart_cancelled")
        ):
            raise RuntimeError("Note generation cancelled")

    @contextmanager
    def _agent_slot(self, sid):
        with self.condition:
            self.condition.wait_for(
                lambda: (
                    self.closed
                    or self._state(sid).get("deleted")
                    or self._state(sid).get("restart_cancelled")
                    or self.agent_running < self.limit
                )
            )
            self._check_running(sid)
            self.agent_running += 1
        try:
            yield
        finally:
            with self.condition:
                self.agent_running -= 1
                self.condition.notify_all()

    def _lecture_context(self, sid):
        session = self.ctx.streams.get_session(sid)
        if session is not None:
            meta = dict(session.meta)
        else:
            path = self.ctx.data_dir / "sessions" / sid / "meta.json"
            saved = (
                json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            )
            meta = saved.get("meta", saved)
        return (
            "## 目标课次\n会话："
            + sid
            + "\n课次元信息："
            + json.dumps(meta, ensure_ascii=False)
            + "\n"
        )

    def _agent_run(
        self, sid, session_id, instruction, *, follow_ups=(), contextualize=True
    ):
        with self._agent_slot(sid):
            info = self.agent.get_session(session_id)
            if info is None:
                raise KeyError(f"Agent session {session_id!r} not found")
            is_writer = info.role == "note_writer"
            with self.condition:
                self._check_running(sid)
                history = self._state(sid).setdefault("last_agent_tasks", [])
                previous = next(
                    (
                        task
                        for task in history
                        if task["id"] == f"notes-agent:{session_id}"
                    ),
                    None,
                )
                if previous is not None:
                    history.remove(previous)
                self.agent_tasks[session_id] = {
                    "id": f"notes-agent:{session_id}",
                    "session_id": sid,
                    "label": info.name,
                    "detail": "执行中",
                    "agent_step": 0,
                    "started": time.monotonic(),
                    "agents": {
                        "main": self._execution_agent(
                            "main",
                            ExecutionRole.WRITER
                            if is_writer
                            else ExecutionRole.CHUNK_WRITER,
                            info.name,
                            info.model,
                            session_id=session_id,
                        )
                    },
                }
                if previous is not None:
                    self.agent_tasks[session_id]["agents"]["main"]["blocks"] = list(
                        previous["agents"]["main"]["blocks"]
                    )
                    self.agent_tasks[session_id]["agents"].update(
                        {
                            key: value
                            for key, value in previous["agents"].items()
                            if key != "main"
                        }
                    )

            def on_event(event):
                if event.type == AgentEventType.RUN_READY:
                    with self.condition:
                        self.agent_tasks[session_id]["ready"] = True
                if event.type in (
                    AgentEventType.EXECUTION,
                    AgentEventType.FILE,
                    AgentEventType.REVIEWER,
                    AgentEventType.RUN_READY,
                ):
                    self._agent_event(
                        Event(
                            event.type,
                            {
                                "session_id": session_id,
                                "step": event.step,
                                **event.data,
                            },
                            0,
                            0,
                        )
                    )
                if event.type == AgentEventType.FILE:
                    workspace = (
                        self.agent.session_workspace(session_id) if is_writer else None
                    )
                    with self.condition:
                        self._check_running(sid)
                        if workspace is not None:
                            self._publish_lecture(sid, workspace, draft=True)
                            self.ctx.ui.refresh("notes.preview")

            try:
                result = self.agent.run(
                    session_id,
                    self._lecture_context(sid) + "\n" + instruction
                    if contextualize
                    else instruction,
                    follow_ups=follow_ups,
                    on_event=on_event,
                    write_guard=lambda path: publication_lock(
                        self.ctx.data_dir / "sessions" / sid / "notes"
                    ),
                )
                with self.condition:
                    if is_writer:
                        self.continuations[sid] = ContinuationStage.FINISHING
                    self._check_running(sid)
                    task = self.agent_tasks[session_id]
                    task["agents"]["main"]["status"] = (
                        ExecutionStatus.FAILED
                        if result.error
                        else ExecutionStatus.COMPLETED
                    )
                return result
            finally:
                with self.condition:
                    if is_writer:
                        self.continuations[sid] = ContinuationStage.FINISHING
                    task = self.agent_tasks.pop(session_id, None)
                    if task is not None:
                        if task["agents"]["main"]["status"] == ExecutionStatus.RUNNING:
                            task["agents"]["main"]["status"] = ExecutionStatus.FAILED
                        self._state(sid).setdefault("last_agent_tasks", []).append(task)
                    self.condition.notify_all()

    def _run_agent(self, sid, name, workspace, instruction):
        session = self.agent.create_session(
            name,
            workspace=workspace,
            system_prompt=self.setting("notes.chunk_system_prompt", "") or SYSTEM,
            role="note_chunk",
            tools=[
                AgentToolName.READ,
                AgentToolName.GREP,
                AgentToolName.GLOB,
                AgentToolName.EDIT,
                AgentToolName.WRITE,
            ],
        )
        try:
            result = self._agent_run(sid, session.id, instruction)
            if result.error:
                raise RuntimeError(result.error)
            return result
        finally:
            self.agent.close(session.id)

    def _chunk(self, sid, index, records):
        with self.condition:
            chunk = dict(self._state(sid)["chunks"][index])
            self._check_running(sid)
        core, context = chunk_material(
            records,
            chunk["start"],
            chunk["end"],
            float(self.setting("notes.overlap_seconds", 60)),
        )
        manual = [record for record in records if record.get("start") is None]
        core += manual[chunk.get("manual_start", 0) : chunk.get("manual_end", 0)]
        if not core:
            with self.condition:
                self._state(sid)["chunks"][index]["state"] = ChunkState.EMPTY
                self._publish_preview(sid)
                self._save(self._state(sid))
            return
        root = self._directory(sid) / "chunks" / str(index)
        root.mkdir(parents=True, exist_ok=True)
        (root / "source.md").write_text(render(core), encoding="utf-8")
        (root / "context.md").write_text(render(context), encoding="utf-8")
        (root / SOURCE_FILENAME).write_text(render(records), encoding="utf-8")
        (root / "visual_history.md").write_text(
            render(
                [
                    r
                    for r in records
                    if r["stream"] == StreamKind.VISUAL
                    and float(r.get("start") or 0) < chunk["start"]
                    and not r.get("error")
                ]
            ),
            encoding="utf-8",
        )
        files = {
            name: root / name
            for name in (
                "source.md",
                "context.md",
                "visual_history.md",
                SOURCE_FILENAME,
            )
        }
        for name, header in NOTE_HEADERS.items():
            files[name] = root / name
            files[name].write_text(header, encoding="utf-8")
        workspace = FileWorkspace(
            files,
            readonly=("source.md", "context.md", "visual_history.md", SOURCE_FILENAME),
        )
        self._run_agent(sid, f"{sid} 分段笔记 第 {index + 1} 段", workspace, CHUNK)
        self._validate_outputs(root)
        with self.condition:
            self._check_running(sid)
            self._state(sid)["chunks"][index]["state"] = ChunkState.DONE
            self._publish_preview(sid)
            self._save(self._state(sid))

    def _validate_outputs(self, directory):
        bodies = []
        for name in NOTE_HEADERS:
            text = (directory / name).read_text(encoding="utf-8")
            bodies.append(
                any(
                    line.strip() and not line.lstrip().startswith("#")
                    for line in text.splitlines()
                )
            )
        if not any(bodies):
            raise RuntimeError("The note agent produced no body text")

    def _publish_preview(self, sid):
        if (
            self.closed
            or self._state(sid).get("deleted")
            or not self.setting("instant-notes.enabled", True)
        ):
            return
        state = self._state(sid)
        chunks = []
        for chunk in sorted(state["chunks"], key=lambda c: c["index"]):
            if chunk["state"] not in (ChunkState.DONE, ChunkState.EMPTY):
                break
            chunks.append(chunk)
        notes = self.ctx.data_dir / "sessions" / sid / "notes"
        notes.mkdir(parents=True, exist_ok=True)
        with publication_lock(notes):
            for kind in KINDS:
                body = []
                for chunk in chunks:
                    path = (
                        self._directory(sid)
                        / "chunks"
                        / str(chunk["index"])
                        / f"note_{kind}.md"
                    )
                    if path.exists():
                        text = strip_blocks(path.read_text(encoding="utf-8")).strip()
                        heading = NOTE_HEADERS[f"note_{kind}.md"].strip()
                        lines = text.splitlines()
                        if lines and lines[0].strip() == heading:
                            text = "\n".join(lines[1:]).strip()
                        if text:
                            body.append(text)
                path = notes / f"chunk_note_{kind}.md"
                previous = path.read_text(encoding="utf-8") if path.exists() else ""
                seen = set()
                qa_blocks = []
                for block in parse_blocks(previous):
                    if block.id not in seen:
                        seen.add(block.id)
                        qa_blocks.append(previous[block.start : block.end])
                content = f"# {LABELS[kind]}\n\n> 未审阅\n\n" + "\n\n".join(body)
                if qa_blocks:
                    content += "\n\n## 问答\n\n" + "\n\n".join(qa_blocks)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(
                    content,
                    encoding="utf-8",
                )
                os.replace(temporary, path)
        state["published_end"] = chunks[-1]["end"] if chunks else 0
        self.ctx.ui.refresh("notes.preview")

    def _publish_lecture(self, sid, workspace, *, draft=False):
        notes = self.ctx.data_dir / "sessions" / sid / "notes"
        notes.mkdir(parents=True, exist_ok=True)
        with self.condition, publication_lock(notes):
            self._check_running(sid)
            for name in NOTE_HEADERS:
                destination = notes / name
                latest = (
                    destination.read_text(encoding="utf-8")
                    if destination.exists()
                    else ""
                )
                seen = set()
                blocks = []
                for block in parse_blocks(latest):
                    if block.id not in seen:
                        seen.add(block.id)
                        blocks.append(latest[block.start : block.end])
                body = strip_blocks(workspace.path(name).read_text(encoding="utf-8"))
                if draft:
                    body = "> 生成中，未审阅\n\n" + body
                if blocks:
                    body += "\n\n## 问答\n\n" + "\n\n".join(blocks)
                temporary = destination.with_suffix(".tmp")
                temporary.write_text(body, encoding="utf-8")
                os.replace(temporary, destination)
            source = notes.parent / SOURCE_FILENAME
            temporary = source.with_suffix(".tmp")
            temporary.write_bytes(workspace.path(SOURCE_FILENAME).read_bytes())
            os.replace(temporary, source)

    def _writer_system_prompt(self):
        mode = str(self.setting("notes.generation_mode", "default") or "default")
        original = self.setting("notes.system_prompt", "") or SYSTEM
        return system_prompt_for_mode(mode, original)

    def _prepare_writer(self, sid, records):
        with self.condition:
            self._check_running(sid)
        generation = digest(records)
        directory = self._directory(sid) / "final" / generation
        directory.mkdir(parents=True, exist_ok=True)
        source = directory / SOURCE_FILENAME
        errors = self._state(sid)["errors"]
        gap_notice = (
            "# 来源缺口\n\n" + json.dumps(errors, ensure_ascii=False) + "\n\n"
            if errors
            else ""
        )
        source.write_text(gap_notice + render(records), encoding="utf-8")
        files = {SOURCE_FILENAME: source}
        index_lines = []
        for chunk in self._state(sid)["chunks"]:
            if chunk["state"] != ChunkState.DONE:
                index_lines.append(
                    f"Chunk {chunk['index']}: {chunk['state']} {chunk['error']}"
                )
                continue
            for kind in KINDS:
                name = f"chunk_{chunk['index']}_{kind}.md"
                original = (
                    self._directory(sid)
                    / "chunks"
                    / str(chunk["index"])
                    / f"note_{kind}.md"
                )
                frozen = directory / name
                frozen.write_text(
                    strip_blocks(original.read_text(encoding="utf-8")), encoding="utf-8"
                )
                files[name] = frozen
                index_lines.append(name)
        index_path = directory / "index.md"
        index_path.write_text("\n".join(index_lines), encoding="utf-8")
        files["index.md"] = index_path
        readonly = list(files)
        for name, header in NOTE_HEADERS.items():
            files[name] = directory / name
            if not files[name].exists():
                files[name].write_text(header, encoding="utf-8")
        workspace = FileWorkspace(files, readonly=readonly)
        review_enabled = bool(self.setting("notes.review_enabled", True))
        review = {
            "prompt": REVIEW,
            "role": "note_review",
            "system": self.setting("notes.reviewer_prompt", "") or REVIEW_SYSTEM,
            "maxRounds": int(self.setting("notes.review_max_rounds", 0)),
            "targets": list(NOTE_HEADERS),
            "sources": [SOURCE_FILENAME],
        }
        writer = self.agent.create_session(
            f"{sid} 笔记写作",
            workspace=workspace,
            system_prompt=self._writer_system_prompt(),
            role="note_writer",
            tools=[
                AgentToolName.READ,
                AgentToolName.GREP,
                AgentToolName.GLOB,
                AgentToolName.EDIT,
                AgentToolName.WRITE,
            ],
            meta={"review": review} if review_enabled else {},
        )
        with self.condition:
            state = self._state(sid)
            state["writer"] = {
                "session_id": writer.id,
                "directory": str(directory.relative_to(self._directory(sid))),
                "name": writer.name,
                "system_prompt": self._writer_system_prompt(),
                "role": "note_writer",
                "tools": [
                    AgentToolName.READ,
                    AgentToolName.GREP,
                    AgentToolName.GLOB,
                    AgentToolName.EDIT,
                    AgentToolName.WRITE,
                ],
                "files": {
                    name: str(path.relative_to(self._directory(sid)))
                    for name, path in files.items()
                },
                "readonly": readonly,
                "review": review if review_enabled else None,
            }
            self.continuations[sid] = ContinuationStage.STARTING
            self._save(state)
        return writer, workspace, directory, review_enabled

    def _final(self, sid, records, messages=None):
        if messages is None:
            writer, workspace, directory, review_enabled = self._prepare_writer(
                sid, records
            )
            instruction = WRITER
        else:
            with self.condition:
                self._check_running(sid)
                saved = dict(self._state(sid)["writer"])
                files = self._writer_inputs(sid, saved)
                notes = self.ctx.data_dir / "sessions" / sid / "notes"
                with publication_lock(notes):
                    for name in NOTE_HEADERS:
                        files[name].write_text(
                            strip_blocks((notes / name).read_text(encoding="utf-8")),
                            encoding="utf-8",
                        )
            workspace = FileWorkspace(files, readonly=saved["readonly"])
            review = (
                {
                    **saved["review"],
                    "role": "note_review",
                    "targets": list(NOTE_HEADERS),
                    "sources": [SOURCE_FILENAME],
                }
                if saved["review"]
                else None
            )
            parameters = dict(
                name=saved["name"],
                workspace=workspace,
                system_prompt=saved["system_prompt"],
                role=saved["role"],
                tools=saved["tools"],
                meta={"review": review} if review else {},
            )
            if self.agent.can_resume(saved["session_id"])[0]:
                writer = self.agent.reopen(saved["session_id"], **parameters)
            else:
                writer = self.agent.create_session(**parameters)
                with self.condition:
                    self._state(sid)["writer"]["session_id"] = writer.id
                    self._save(self._state(sid))
            directory = self._directory(sid) / saved["directory"]
            review_enabled = review is not None
            instruction = messages[0]
        try:
            if messages is None and review_enabled:
                instruction += "\n保存文稿后调用 review_notes．核对审阅依据后修订，每次修改后必须再次调用，直到审阅通过．"
            result = self._agent_run(
                sid,
                writer.id,
                instruction,
                follow_ups=messages[1:] if messages is not None else (),
                contextualize=messages is None,
            )
            if result.error:
                raise RuntimeError(result.error)
            if review_enabled and not result.review_passed:
                raise RuntimeError("The saved notes do not have a passing review")
            self._validate_outputs(directory)
            self._publish_lecture(sid, workspace)
            with self.condition:
                self._check_running(sid)
                state = self._state(sid)
                state.update(
                    phase=NotePhase.PUBLISHED,
                    review_passed=True,
                    finalize_requested=False,
                )
                self._save(state)
                self.ctx.ui.refresh("notes.published")
        finally:
            self.agent.close(writer.id)

    def teardown(self):
        with self.condition:
            self.closed = True
            self.queue.clear()
            self.writer_messages.clear()
            self.continuations.clear()
            running = tuple(self.agent_tasks)
            self.condition.notify_all()
        self.ctx.ui.unregister_action("notes.message")
        for session_id in running:
            self.agent.cancel(session_id)
        self.coordinator.join()
        self.pool.shutdown(wait=True, cancel_futures=True)
