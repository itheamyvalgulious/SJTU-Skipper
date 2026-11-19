from core.plugin import Plugin

from .pipeline import NotesPipeline
from .types import GenerationState, NoteKind, NotePublication
from .view import NotesView


class NotesService:
    def __init__(self, pipeline, view):
        self.pipeline = pipeline
        self.view = view

    @property
    def busy(self):
        return self.pipeline.status()["generation_state"] == GenerationState.GENERATING

    def start(self, session_id=None):
        return self.pipeline.start(session_id)

    def finalize(self, session_id=None):
        return self.pipeline.finalize(session_id)

    def status(self, session_id=None):
        return self.pipeline.status(session_id)

    def stop(self, session_id=None):
        return self.pipeline.stop(session_id)

    def reset(self, session_id=None):
        return self.pipeline.reset(session_id)

    def delete(self, session_id=None):
        return self.pipeline.delete(session_id)

    def draft_paths(self, session_id=None):
        return self.pipeline.draft_paths(session_id)

    def source_path(self, session_id=None):
        return self.pipeline.source_path(session_id)

    def note_files(self, kind=NotePublication.FINAL):
        return self.pipeline.note_files(kind)

    def running_tasks(self):
        return self.pipeline.running_tasks()

    def open_notes(self, session_id, doc=NoteKind.KNOWLEDGE):
        return self.view.open_notes(session_id, doc)

    def open_logs(self, session_id):
        return self.view.open_logs(session_id)

    def open_chunk_notes(self, session_id, doc=NoteKind.KNOWLEDGE):
        return self.view.open_chunk_notes(session_id, doc)


class NotesPlugin(Plugin):
    id = "notes"
    requires = ("settings", "llm-agent", "reader")
    optional_requires = ("stt", "vlm")
    provides = ("notes",)

    def setup(self, ctx):
        self.pipeline = NotesPipeline()
        self.pipeline.setup(ctx)
        self.view = NotesView()
        self.view.setup(ctx, self.pipeline)
        ctx.register_service(self.id, NotesService(self.pipeline, self.view))

    def teardown(self):
        self.view.teardown()
        self.pipeline.teardown()
