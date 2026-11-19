from core.plugin import Plugin
from core.ui import BlockType, Side
from plugins.notes_pipeline.types import NoteKind, NotePublication


class InstantSummarizeService:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def draft_paths(self, session_id=""):
        return self.pipeline.draft_paths(session_id)

    def note_files(self):
        return self.pipeline.note_files(NotePublication.INSTANT)

    def running_tasks(self):
        return self.pipeline.running_tasks()


class InstantSummarizePlugin(Plugin):
    id = "instant-summarize"
    requires = ("settings", "notes-pipeline", "reader")
    provides = ("instant-summarize",)

    def setup(self, ctx):
        self.ctx = ctx
        self.settings = ctx.require("settings")
        self.pipeline = ctx.require("notes-pipeline")
        ctx.register_service(self.id, InstantSummarizeService(self.pipeline))
        if not ctx.ui.has_tab("note"):
            ctx.ui.add_tab(Side.RIGHT, "note", "笔记", order=20, plugin_id=self.id)
        ctx.ui.add_toolbox_item(
            "note",
            "instant-summarize.notes",
            "",
            order=11,
            render=self._render,
            plugin_id=self.id,
        )
        ctx.ui.register_action(
            "instant-summarize.toggle", self._toggle, plugin_id=self.id
        )
        ctx.ui.register_action("instant-summarize.read", self._read, plugin_id=self.id)

    def _render(self):
        status = self.pipeline.status()
        enabled = bool(self.settings.get("instant-summarize.enabled", True))
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
                    <input id="instant-notes-enabled" type="checkbox" role="switch" switch
                      data-change-action="instant-summarize.toggle"{" checked" if enabled else ""}>
                  </div>
                  <label for="instant-notes-enabled">启用</label>
                </div>
              </div>
              <p class="fg-2 mb-2" role="status">已发布至 {status["published_end"]:.0f} 秒{gaps}</p>
              <button type="button" class="btn-outline theme-primary" data-action="instant-summarize.read">打开分段笔记</button>
            </section>''',
            }
        ]

    def _toggle(self, payload):
        enabled = payload.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.lower() in ("true", "1", "yes", "on")
        self.settings.set("instant-summarize.enabled", bool(enabled))
        self.ctx.ui.refresh("notes.toggle")
        return {"ok": True, "refresh": True}

    def _read(self, payload):
        sid = str(payload.get("session_id") or self.ctx.streams.notes_session_id)
        docs = [
            {
                "key": key,
                "label": label,
                "path": str(self.ctx.data_dir / "sessions" / sid / "notes" / name),
            }
            for key, (name, label) in self.pipeline.note_files(
                NotePublication.INSTANT
            ).items()
        ]
        page = self.ctx.require("reader").show(
            title="分段笔记",
            docs=docs,
            active=NoteKind(payload.get("doc", NoteKind.KNOWLEDGE)),
            page_key=f"{sid}:instant-summarize",
            owner=self.id,
        )
        return {"ok": True, "open_page": page}

    def teardown(self):
        self.ctx.ui.unregister_action("instant-summarize.toggle")
        self.ctx.ui.unregister_action("instant-summarize.read")
        self.ctx.ui.remove_toolbox_item("instant-summarize.notes")
