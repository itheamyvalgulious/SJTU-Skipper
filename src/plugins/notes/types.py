from enum import StrEnum


class NoteKind(StrEnum):
    KNOWLEDGE = "knowledge"
    GOSSIP = "gossip"
    NOTICE = "notice"


class NotePublication(StrEnum):
    CHUNK = "chunk"
    FINAL = "final"


class NotePhase(StrEnum):
    COLLECTING = "collecting"
    CHUNKING = "chunking"
    WRITING = "writing"
    PUBLISHED = "published"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DELETED = "deleted"


class ChunkState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    CANCELLED = "cancelled"
    DONE = "done"
    FAILED = "failed"
    EMPTY = "empty"


class GenerationState(StrEnum):
    NOT_STARTED = "not_started"
    GENERATING = "generating"
    FINISHED = "finished"


class ContinuationStage(StrEnum):
    QUEUED = "queued"
    STARTING = "starting"
    LIVE = "live"
    FINISHING = "finishing"


class NoteTaskKind(StrEnum):
    CHUNK = "chunk"
    FINAL = "final"
