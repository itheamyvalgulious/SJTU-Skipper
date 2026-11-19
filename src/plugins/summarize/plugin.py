import json
import threading
from enum import StrEnum
from html import escape
from core.plugin import Plugin
from core.ui import BlockType, Side
from core.settings_spec import SettingsSpec, SettingType
from plugins.notes_pipeline.types import GenerationState, NoteKind, NotePublication


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
                        "send_action": "notes-pipeline.message",
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


class SummarizeService:
    def __init__(self, plugin):
        self.plugin = plugin

    @property
    def busy(self):
        state = self.plugin.pipeline.status()
        return state["generation_state"] == GenerationState.GENERATING

    def start(self, session_id=""):
        return self.plugin.pipeline.start(session_id)

    def source_path(self, session_id=""):
        return self.plugin.pipeline.source_path(session_id)

    def note_files(self):
        return self.plugin.pipeline.note_files(NotePublication.FINAL)

    def running_tasks(self):
        return self.plugin.pipeline.running_tasks()

    def open_notes(self, session_id, doc: NoteKind = NoteKind.KNOWLEDGE):
        return self.plugin._read({"session_id": session_id, "doc": doc})

    def open_logs(self, session_id):
        return self.plugin._logs({"session_id": session_id})


class SummarizePlugin(Plugin):
    id = "summarize"
    requires = ("settings", "notes-pipeline", "reader")
    optional_requires = ("input",)
    provides = ("summarize",)
    uses_actions = ("notes-pipeline.message",)
    ON_STOP_LABELS = (
        (OnStopAction.DIRECT, "直接生成笔记"),
        (OnStopAction.FILL, "先补全再生成"),
        (OnStopAction.SKIP, "不自动生成笔记"),
    )

    def setup(self, ctx):
        self.ctx = ctx
        self.settings = ctx.require("settings")
        self.pipeline = ctx.require("notes-pipeline")
        self.stop_event = threading.Event()
        self.notices = {}
        self.fill_threads = []
        self.log_pages = set()
        for key, default, kind in (
            ("fill_wait_minutes", 20, SettingType.INT),
            ("fill_max_attempts", 6, SettingType.INT),
        ):
            self.settings.register(
                SettingsSpec(
                    key=f"summarize.{key}",
                    default=default,
                    type=kind,
                    label=key,
                    group="Summarize",
                ),
                plugin_id=self.id,
            )
        self.settings.register(
            SettingsSpec(
                key="summarize.on_stop",
                default=OnStopAction.DIRECT,
                type=SettingType.SELECT,
                label="课后处理",
                options=list(self.ON_STOP_LABELS),
                group="Summarize",
            ),
            plugin_id=self.id,
        )
        ctx.register_service(self.id, SummarizeService(self))
        if not ctx.ui.has_tab("note"):
            ctx.ui.add_tab(Side.RIGHT, "note", "笔记", order=20, plugin_id=self.id)
        ctx.ui.add_toolbox_item(
            "note",
            "summarize.notes",
            "",
            order=10,
            render=self._render,
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
        ):
            ctx.ui.register_action(f"summarize.{action}", handler, plugin_id=self.id)
        ctx.events.on("session.stopped", self._stopped, plugin_id=self.id)

    def _render(self):
        sid = self.ctx.streams.notes_session_id
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
        mode = OnStopAction(self.settings.get("summarize.on_stop", OnStopAction.DIRECT))
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
                <button type="button" class="btn-outline theme-primary" data-action="summarize.read">打开笔记</button>
                <button type="button" class="btn-text" data-action="summarize.logs">查看日志</button>
                <button type="button" class="{style}" data-action="summarize.{action}">{label}</button>
              </div>
              {notice}
              <div class="mt-3 pt-3 border-top">
                <label class="d-block fw-semibold mb-2" for="notes-on-stop">课程结束后</label>
                <select id="notes-on-stop" name="mode" class="form-control min-w-0"
                  data-change-action="summarize.set_mode">{options}</select>
              </div>
            </section>''',
            }
        ]

    def _delete(self, payload):
        deleted = self.pipeline.delete(
            str(payload.get("session_id") or self.ctx.streams.notes_session_id)
        )
        if not deleted:
            return {"ok": False, "refresh": True, "message": "仅生成结束的笔记可删除"}
        return {"ok": True, "refresh": True, "message": "已删除笔记"}

    def _generate(self, payload):
        started = self.pipeline.start(
            str(payload.get("session_id") or self.ctx.streams.notes_session_id)
        )
        if not started:
            return {
                "ok": False,
                "refresh": True,
                "message": "笔记未开始生成，请检查状态和来源",
            }
        return {"ok": True, "refresh": True, "message": "正在生成笔记"}

    def _stop(self, payload):
        stopped = self.pipeline.stop(
            str(payload.get("session_id") or self.ctx.streams.notes_session_id)
        )
        if not stopped:
            return {"ok": False, "refresh": True, "message": "仅正在生成的笔记可停止"}
        return {"ok": True, "refresh": True, "message": "已停止生成笔记"}

    def _read(self, payload):
        sid = str(payload.get("session_id") or self.ctx.streams.notes_session_id)
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
            page_key=f"{sid}:summarize",
            owner=self.id,
        )
        return {"ok": True, "open_page": page}

    def _logs(self, payload):
        sid = str(payload.get("session_id") or self.ctx.streams.notes_session_id)
        page = f"summarize:logs:{sid}"
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
            message_channel=self.ctx.ui.has_action("notes-pipeline.message"),
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
        self.settings.set("summarize.on_stop", mode)
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
        meta = self._meta(sid)
        if meta.get("input_failed") or meta.get("processing_cancelled"):
            return
        if meta.get("fill_for"):
            self.pipeline.finalize(str(meta["fill_for"]))
            return
        mode = OnStopAction(self.settings.get("summarize.on_stop", OnStopAction.DIRECT))
        if mode == OnStopAction.DIRECT:
            self.pipeline.finalize(sid)
        elif mode == OnStopAction.FILL:
            self._request_fill(sid)

    def _fill_again(self, payload):
        self._request_fill(
            str(payload.get("session_id") or self.ctx.streams.notes_session_id)
        )
        return {"ok": True, "message": "正在查找可补录的内容", "refresh": True}

    def _request_fill(self, sid):
        thread = threading.Thread(
            target=self._fill, args=(sid,), name=f"notes-fill-{sid}", daemon=True
        )
        self.fill_threads.append(thread)
        thread.start()

    def _fill(self, sid):
        service = self.ctx.service("input")
        attempts = max(1, int(self.settings.get("summarize.fill_max_attempts", 6)))
        for attempt in range(attempts):
            if self.stop_event.is_set():
                return
            try:
                if service is not None and service.fill_gaps(sid):
                    return
            except Exception as exc:
                self.notices[sid] = f"补录查询失败：{exc}"
            if attempt + 1 < attempts and self.stop_event.wait(
                max(1, int(self.settings.get("summarize.fill_wait_minutes", 20))) * 60
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
        ):
            self.ctx.ui.unregister_action(f"summarize.{action}")
        self.ctx.ui.remove_toolbox_item("summarize.notes")
        for page in self.log_pages:
            self.ctx.ui.remove_page(page)
