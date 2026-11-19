import json
import threading
from enum import StrEnum
from html import escape
from core.ui import BlockType, Side
from core.settings_spec import SettingsSpec, SettingType
from .types import GenerationState, NoteKind, NotePublication


class OnStopAction(StrEnum):
    DIRECT = "direct"
    FILL = "fill"
    SKIP = "skip"


def note_log_blocks(state, *, message_channel: bool = True):
    agents = [agent for task in state["tasks"] for agent in task["agents"].values()]
    blocks = (
        [
            {
                "type": BlockType.AGENT_ACTIVITY,
                "id": state["session_id"],
                "agents": agents,
                **(
                    {
                        "send_action": "notes.message",
                        "send_values": {"session_id": state["session_id"]},
                    }
                    if message_channel
                    else {"send_action": "", "send_values": {}}
                ),
            }
        ]
        if agents
        else []
    )
    return blocks


class NotesView:
    id = "notes"

    def open_notes(self, session_id, doc=NoteKind.KNOWLEDGE):
        return self._read({"session_id": session_id, "doc": doc})

    def open_logs(self, session_id):
        return self._logs({"session_id": session_id})

    def open_chunk_notes(self, session_id, doc=NoteKind.KNOWLEDGE):
        return self._chunk_read({"session_id": session_id, "doc": doc})

    ON_STOP_LABELS = (
        (OnStopAction.DIRECT, "直接生成笔记"),
        (OnStopAction.FILL, "先补全再生成"),
        (OnStopAction.SKIP, "不自动生成笔记"),
    )

    def setup(self, ctx, pipeline):
        self.ctx = ctx
        self.settings = ctx.require("settings")
        self.pipeline = pipeline
        self.stop_event = threading.Event()
        self.notices = {}
        self.fill_threads = []
        self.log_pages = set()
        for key, default, label, order in (
            ("fill_wait_minutes", 20, "补录重试间隔（分钟）", 110),
            ("fill_max_attempts", 6, "补录尝试次数上限", 120),
        ):
            self.settings.register(
                SettingsSpec(
                    key=f"notes.{key}",
                    default=default,
                    type=SettingType.INT,
                    label=label,
                    group="笔记",
                    order=order,
                ),
                plugin_id=self.id,
            )
        self.settings.register(
            SettingsSpec(
                key="notes.on_stop",
                default=OnStopAction.DIRECT,
                type=SettingType.SELECT,
                label="课后处理",
                options=list(self.ON_STOP_LABELS),
                group="笔记",
                order=100,
            ),
            plugin_id=self.id,
        )
        if not ctx.ui.has_tab("note"):
            ctx.ui.add_tab(Side.RIGHT, "note", "笔记", order=20, plugin_id=self.id)
        ctx.ui.add_toolbox_item(
            "note",
            "notes.final",
            "",
            order=10,
            render=self._render,
            plugin_id=self.id,
        )
        ctx.ui.add_toolbox_item(
            "note",
            "notes.chunks",
            "",
            order=11,
            render=self._render_chunks,
            plugin_id=self.id,
        )
        for action, handler in (
            ("read", self._read),
            ("logs", self._logs),
            ("set_mode", self._set_mode),
            ("fill_again", self._fill_again),
            ("delete", self._delete),
            ("generate", self._generate),
            ("stop", self._stop),
            ("chunk_toggle", self._chunk_toggle),
            ("chunk_read", self._chunk_read),
        ):
            ctx.ui.register_action(f"notes.{action}", handler, plugin_id=self.id)
        ctx.events.on("session.stopped", self._stopped, plugin_id=self.id)

    def _render(self):
        sid = self.ctx.streams.selected_class_id
        if not sid or sid == "boot":
            return []
        state = self.pipeline.status(sid)["generation_state"]
        state_label = {
            GenerationState.NOT_STARTED: "未开始",
            GenerationState.GENERATING: "正在生成",
            GenerationState.FINISHED: "生成结束",
        }[state]
        generation_action = {
            GenerationState.NOT_STARTED: (
                "generate",
                "生成笔记",
                "btn-solid theme-primary",
            ),
            GenerationState.GENERATING: (
                "stop",
                "停止生成",
                "btn-outline theme-danger",
            ),
            GenerationState.FINISHED: ("delete", "删除笔记", "btn-text theme-danger"),
        }[state]
        action, label, style = generation_action
        mode = OnStopAction(self.settings.get("notes.on_stop", OnStopAction.DIRECT))
        options = "".join(
            f'<option value="{key}"{" selected" if key == mode else ""}>{text}</option>'
            for key, text in self.ON_STOP_LABELS
        )
        notice = (
            f'<p class="fg-2 mt-2 mb-0">{escape(self.notices[sid])}</p>'
            if sid in self.notices
            else ""
        )
        return [
            {
                "type": BlockType.HTML,
                "html": f'''<section class="p-3 border-bottom" aria-label="课后笔记">
              <div class="d-flex align-items-center justify-content-between gap-2 mb-3">
                <h3 class="fs-md fw-semibold mb-0">课后笔记</h3>
                <span class="fg-2" role="status">{state_label}</span>
              </div>
              <div class="d-flex flex-wrap align-items-center gap-2">
                <button type="button" class="btn-outline theme-primary" data-action="notes.read">打开笔记</button>
                <button type="button" class="btn-text" data-action="notes.logs">查看日志</button>
                <button type="button" class="{style}" data-action="notes.{action}">{label}</button>
              </div>
              {notice}
              <div class="mt-3 pt-3 border-top">
                <label class="d-block fw-semibold mb-2" for="notes-on-stop">课程结束后</label>
                <select id="notes-on-stop" name="mode" class="form-control min-w-0"
                  data-change-action="notes.set_mode">{options}</select>
              </div>
            </section>''',
            }
        ]

    def _action_sid(self, payload):
        sid = str(payload.get("session_id", self.ctx.streams.selected_class_id) or "")
        return "" if sid == "boot" else sid

    def _delete(self, payload):
        sid = self._action_sid(payload)
        if not sid:
            return {"ok": False, "message": "请先选择课程"}
        deleted = self.pipeline.delete(sid)
        if not deleted:
            return {"ok": False, "refresh": True, "message": "仅生成结束的笔记可删除"}
        return {"ok": True, "refresh": True, "message": "已删除笔记"}

    def _generate(self, payload):
        sid = self._action_sid(payload)
        if not sid:
            return {"ok": False, "message": "请先选择课程"}
        started = self.pipeline.start(sid)
        if not started:
            return {
                "ok": False,
                "refresh": True,
                "message": "笔记未开始生成，请检查状态和来源",
            }
        return {"ok": True, "refresh": True, "message": "正在生成笔记"}

    def _stop(self, payload):
        sid = self._action_sid(payload)
        if not sid:
            return {"ok": False, "message": "请先选择课程"}
        stopped = self.pipeline.stop(sid)
        if not stopped:
            return {"ok": False, "refresh": True, "message": "仅正在生成的笔记可停止"}
        return {"ok": True, "refresh": True, "message": "已停止生成笔记"}

    def _read(self, payload):
        sid = self._action_sid(payload)
        if not sid:
            return {"ok": False, "message": "请先选择课程"}
        docs = [
            {
                "key": key,
                "label": label,
                "path": str(self.ctx.data_dir / "sessions" / sid / "notes" / name),
            }
            for key, (name, label) in self.pipeline.note_files(
                NotePublication.FINAL
            ).items()
        ]
        page = self.ctx.require("reader").show(
            title="笔记",
            docs=docs,
            active=NoteKind(payload.get("doc", NoteKind.KNOWLEDGE)),
            page_key=f"{sid}:notes",
            owner=self.id,
        )
        return {"ok": True, "open_page": page}

    def _logs(self, payload):
        sid = self._action_sid(payload)
        if not sid:
            return {"ok": False, "message": "请先选择课程"}
        page = f"notes:logs:{sid}"
        if not self.ctx.ui.has_page(page):
            self.ctx.ui.add_page(
                page,
                "Agent 日志",
                render=lambda sid=sid: self._render_logs(sid),
                order=6,
                plugin_id=self.id,
            )
        self.log_pages.add(page)
        self.ctx.ui.open_page(page)
        return {"ok": True, "open_page": page}

    def _render_logs(self, sid):
        blocks = note_log_blocks(
            self.pipeline.status(sid),
            message_channel=self.ctx.ui.has_action("notes.message"),
        )
        return blocks or [
            {
                "type": BlockType.TEXT,
                "text": "还没有 Agent 日志．生成笔记后，日志会显示在这里．",
            }
        ]

    def _set_mode(self, payload):
        try:
            mode = OnStopAction(payload.get("mode"))
        except ValueError:
            return {"ok": False, "message": "未知的课后处理方式"}
        self.settings.set("notes.on_stop", mode)
        return {"ok": True, "refresh": True}

    def _meta(self, sid):
        session = self.ctx.streams.get_session(sid)
        if session is not None:
            return session.meta
        path = self.ctx.data_dir / "sessions" / sid / "meta.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return raw.get("meta", raw)
        return {}

    def _stopped(self, event):
        sid = str(event.get("session_id", ""))
        if not sid or sid == "boot":
            return
        meta = event.get("meta")
        if meta is None:
            meta = self._meta(sid)
        session = self.ctx.streams.get_session(sid)
        if (
            event.get("started_at") is not None
            and session is not None
            and session.started_at != event.get("started_at")
        ):
            return
        if meta.get("input_failed") or meta.get("processing_cancelled"):
            return
        if meta.get("fill_for"):
            self.pipeline.finalize(str(meta["fill_for"]))
            return
        mode = OnStopAction(self.settings.get("notes.on_stop", OnStopAction.DIRECT))
        if mode == OnStopAction.DIRECT:
            self.pipeline.finalize(sid)
        elif mode == OnStopAction.FILL:
            self._request_fill(sid)

    def _fill_again(self, payload):
        sid = self._action_sid(payload)
        if not sid:
            return {"ok": False, "message": "请先选择课程"}
        self._request_fill(sid)
        return {"ok": True, "message": "正在查找可补录的内容", "refresh": True}

    def _request_fill(self, sid):
        if not sid or sid == "boot":
            return
        thread = threading.Thread(
            target=self._fill, args=(sid,), name=f"notes-fill-{sid}", daemon=True
        )
        self.fill_threads.append(thread)
        thread.start()

    def _fill(self, sid):
        service = self.ctx.service("input")
        attempts = max(1, int(self.settings.get("notes.fill_max_attempts", 6)))
        for attempt in range(attempts):
            if self.stop_event.is_set():
                return
            try:
                if service is not None and service.fill_gaps(sid):
                    return
            except Exception as exc:
                self.notices[sid] = f"补录查询失败：{exc}"
            if attempt + 1 < attempts and self.stop_event.wait(
                max(1, int(self.settings.get("notes.fill_wait_minutes", 20))) * 60
            ):
                return
        self.notices[sid] = "未取得录播，笔记使用已有来源，并保留来源缺口标记"
        self.pipeline.finalize(sid)
        self.ctx.ui.refresh("notes.fill")

    def teardown(self):
        self.stop_event.set()
        for thread in self.fill_threads:
            thread.join()
        for action in (
            "read",
            "logs",
            "set_mode",
            "fill_again",
            "delete",
            "generate",
            "stop",
            "chunk_toggle",
            "chunk_read",
        ):
            self.ctx.ui.unregister_action(f"notes.{action}")
        self.ctx.ui.remove_toolbox_item("notes.final")
        self.ctx.ui.remove_toolbox_item("notes.chunks")
        for page in self.log_pages:
            self.ctx.ui.remove_page(page)

    def _render_chunks(self):
        sid = self.ctx.streams.selected_class_id
        if not sid or sid == "boot":
            return []
        status = self.pipeline.status(sid)
        enabled = bool(self.settings.get("notes.chunks_enabled", True))
        errors = len(status["errors"])
        gaps = f"，缺口 {errors} 项" if errors else ""
        return [
            {
                "type": BlockType.HTML,
                "html": f'''<section class="px-3 pb-3" aria-label="分段笔记">
              <div class="d-flex align-items-center justify-content-between gap-2 mb-2">
                <h3 class="fs-md fw-semibold mb-0">分段笔记</h3>
                <div class="d-flex align-items-center gap-2">
                  <div class="switch">
                    <input id="chunk-notes-enabled" type="checkbox" role="switch" switch
                      data-change-action="notes.chunk_toggle"{" checked" if enabled else ""}>
                  </div>
                  <label for="chunk-notes-enabled">启用</label>
                </div>
              </div>
              <p class="fg-2 mb-2" role="status">已发布至 {status["published_end"]:.0f} 秒{gaps}</p>
              <button type="button" class="btn-outline theme-primary" data-action="notes.chunk_read">打开分段笔记</button>
            </section>''',
            }
        ]

    def _chunk_toggle(self, payload):
        enabled = payload.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.lower() in ("true", "1", "yes", "on")
        self.settings.set("notes.chunks_enabled", bool(enabled))
        self.ctx.ui.refresh("notes.toggle")
        return {"ok": True, "refresh": True}

    def _chunk_read(self, payload):
        sid = str(payload.get("session_id", self.ctx.streams.selected_class_id) or "")
        if not sid or sid == "boot":
            return {"ok": False, "message": "请先选择课程"}
        docs = [
            {
                "key": key,
                "label": label,
                "path": str(self.ctx.data_dir / "sessions" / sid / "notes" / name),
            }
            for key, (name, label) in self.pipeline.note_files(
                NotePublication.CHUNK
            ).items()
        ]
        page = self.ctx.require("reader").show(
            title="分段笔记",
            docs=docs,
            active=NoteKind(payload.get("doc", NoteKind.KNOWLEDGE)),
            page_key=f"{sid}:chunk-notes",
            owner=self.id,
        )
        return {"ok": True, "open_page": page}
