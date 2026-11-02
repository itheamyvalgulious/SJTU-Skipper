"""Speech-to-text backends.

Two implementations share one tiny protocol:

* :class:`OpenAITranscriptionBackend` — any OpenAI-compatible
  ``/audio/transcriptions`` endpoint (multipart upload, ``httpx``).
* :class:`FasterWhisperBackend` — optional local inference; the dependency is
  imported lazily so the core install stays light.

Backends only translate WAV bytes into :class:`STTResult`; they never touch
streams, settings or the event bus.  That keeps them usable from
``transcribe_once`` without a running worker.
"""

from __future__ import annotations

import io
import json
import threading
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import httpx


DEFAULT_MODEL = "whisper-1"
DEFAULT_FASTER_WHISPER_MODEL = "base"
PROVIDERS: tuple[str, ...] = ("none", "openai", "faster-whisper")

TRANSCRIPTION_PATH = "/audio/transcriptions"


class STTError(RuntimeError):
    """Base error for speech recognition failures."""


class STTBackendError(STTError):
    """A backend failed to produce a transcription."""


class STTUnavailableError(STTError):
    """A backend cannot be used (missing dependency, missing configuration)."""


@dataclass(frozen=True, slots=True)
class STTSegment:
    """One timed piece of a transcription, relative to the submitted audio."""

    start: float
    end: float
    text: str


@dataclass(slots=True)
class STTResult:
    """Result of one ``transcribe`` call."""

    text: str = ""
    segments: list[STTSegment] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class STTBackend(Protocol):
    """Minimal backend contract used by the worker and the service."""

    name: str

    def transcribe(
        self,
        wav_bytes: bytes,
        *,
        language: str | None = None,
        model: str | None = None,
    ) -> STTResult:
        """Transcribe a complete WAV container."""


def transcription_endpoint(base_url: str) -> str:
    """Derive the ``/audio/transcriptions`` endpoint from a chat base URL.

    Accepts the shapes users actually paste into settings::

        https://api.openai.com/v1                  -> .../v1/audio/transcriptions
        https://api.openai.com/v1/chat/completions -> .../v1/audio/transcriptions
        https://api.openai.com                    -> .../v1/audio/transcriptions
        https://host/v1/audio/transcriptions      -> unchanged
    """

    url = (base_url or "").strip()
    if not url:
        raise STTUnavailableError("未配置 STT base_url，无法定位转写接口")
    url = url.rstrip("/")

    if url.endswith(TRANSCRIPTION_PATH):
        return url
    for suffix in ("/chat/completions", "/completions"):
        if url.endswith(suffix):
            root = url[: -len(suffix)].rstrip("/")
            return root + TRANSCRIPTION_PATH

    parsed = urllib.parse.urlsplit(url)
    if parsed.path in ("", "/"):
        return url + "/v1" + TRANSCRIPTION_PATH
    return url + TRANSCRIPTION_PATH


def parse_transcription_payload(payload: Mapping[str, Any]) -> STTResult:
    """Normalise a ``text`` or ``verbose_json`` transcription response."""

    if not isinstance(payload, Mapping):
        raise STTBackendError("转写响应不是 JSON 对象")

    segments: list[STTSegment] = []
    raw_segments = payload.get("segments") or []
    if isinstance(raw_segments, Sequence):
        for entry in raw_segments:
            if not isinstance(entry, Mapping):
                continue
            text = str(entry.get("text", "") or "").strip()
            if not text:
                continue
            segments.append(
                STTSegment(
                    start=float(entry.get("start", 0.0) or 0.0),
                    end=float(entry.get("end", 0.0) or 0.0),
                    text=text,
                )
            )

    text = payload.get("text")
    if text is None:
        text = "".join(segment.text for segment in segments)
    text = str(text).strip()
    if not text and segments:
        text = "".join(segment.text for segment in segments).strip()
    return STTResult(text=text, segments=segments, raw=dict(payload))


class OpenAITranscriptionBackend:
    """OpenAI-compatible ``POST {base}/audio/transcriptions`` backend."""

    name = "openai"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = float(timeout)
        self._client = client
        self._owns_client = client is None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_client(self) -> httpx.Client:
        with self._lock:
            if self._client is None:
                self._client = httpx.Client(timeout=self.timeout)
                self._owns_client = True
            return self._client

    def close(self) -> None:
        with self._lock:
            client = self._client
            owns = self._owns_client
            self._client = None
        if client is not None and owns:
            try:
                client.close()
            except Exception:  # pragma: no cover - best effort
                pass

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def transcribe(
        self,
        wav_bytes: bytes,
        *,
        language: str | None = None,
        model: str | None = None,
    ) -> STTResult:
        """Upload one WAV chunk and parse the transcription response."""

        if not wav_bytes:
            raise STTBackendError("空音频无法转写")
        endpoint = transcription_endpoint(self.base_url)
        client = self._get_client()
        files = {"file": ("chunk.wav", bytes(wav_bytes), "audio/wav")}

        last_error = ""
        for response_format in ("verbose_json", "json"):
            data: dict[str, Any] = {
                "model": model or self.model,
                "response_format": response_format,
            }
            if language:
                data["language"] = language
            try:
                response = client.post(
                    endpoint,
                    headers=self._headers(),
                    data=data,
                    files=files,
                )
            except httpx.HTTPError as exc:
                raise STTBackendError(f"转写请求失败：{exc}") from exc

            if response.status_code < 400:
                try:
                    payload = response.json()
                except (json.JSONDecodeError, ValueError) as exc:
                    raise STTBackendError("转写响应不是合法 JSON") from exc
                return parse_transcription_payload(payload)

            body = response.text[:200].strip()
            last_error = f"转写接口返回 {response.status_code}: {body}"
            # Some compatible servers only implement plain JSON output.
            if response_format == "verbose_json" and response.status_code in (400, 404, 415, 422):
                continue
            raise STTBackendError(last_error)

        raise STTBackendError(last_error or "转写接口不可用")


def _import_faster_whisper() -> Any:
    """Import ``faster_whisper`` lazily with an actionable error message."""

    try:
        import faster_whisper  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency is optional
        raise STTUnavailableError(
            "未安装 faster-whisper，无法本地转写；请执行 "
            "`.venv/bin/pip install 'sjtu-skipper[stt]'` 或把 stt.provider 改为 none/openai"
        ) from exc
    return faster_whisper


class FasterWhisperBackend:
    """Local transcription through the optional ``faster-whisper`` package."""

    name = "faster-whisper"

    def __init__(
        self,
        *,
        model: str = DEFAULT_FASTER_WHISPER_MODEL,
        device: str = "cpu",
        compute_type: str = "int8",
        vad_filter: bool = True,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.vad_filter = vad_filter
        self._model_factory = model_factory
        self._models: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _load_model(self, name: str) -> Any:
        with self._lock:
            cached = self._models.get(name)
            if cached is not None:
                return cached
            if self._model_factory is not None:
                engine = self._model_factory(name)
            else:
                module = _import_faster_whisper()
                factory = getattr(module, "WhisperModel", None)
                if factory is None:
                    raise STTUnavailableError("faster-whisper 缺少 WhisperModel")
                engine = factory(name, device=self.device, compute_type=self.compute_type)
            self._models[name] = engine
            return engine

    def transcribe(
        self,
        wav_bytes: bytes,
        *,
        language: str | None = None,
        model: str | None = None,
    ) -> STTResult:
        """Run local inference on the submitted WAV bytes."""

        if not wav_bytes:
            raise STTBackendError("空音频无法转写")
        name = model or self.model
        engine = self._load_model(name)
        segments: list[STTSegment] = []
        info: Any = None
        try:
            raw_segments, info = engine.transcribe(
                io.BytesIO(bytes(wav_bytes)),
                language=language or None,
                vad_filter=self.vad_filter,
            )
            for entry in raw_segments:
                text = str(getattr(entry, "text", "") or "").strip()
                if not text:
                    continue
                segments.append(
                    STTSegment(
                        start=float(getattr(entry, "start", 0.0) or 0.0),
                        end=float(getattr(entry, "end", 0.0) or 0.0),
                        text=text,
                    )
                )
        except STTError:
            raise
        except Exception as exc:  # noqa: BLE001 - wrap engine errors
            raise STTBackendError(f"faster-whisper 转写失败：{exc}") from exc

        text = "".join(segment.text for segment in segments).strip()
        duration = getattr(info, "duration", None) if info is not None else None
        return STTResult(
            text=text,
            segments=segments,
            raw={
                "provider": self.name,
                "model": name,
                "language": language,
                "duration": duration,
            },
        )


def create_backend(
    provider: str,
    *,
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    timeout: float = 60.0,
) -> STTBackend:
    """Build a backend by name, raising a clear error for unusable choices."""

    name = (provider or "").strip().lower()
    if name == "openai":
        if not base_url:
            raise STTUnavailableError("未配置 STT base_url，无法调用转写接口")
        if not api_key:
            raise STTUnavailableError("未配置 STT API Key，无法调用转写接口")
        return OpenAITranscriptionBackend(
            base_url=base_url,
            api_key=api_key,
            model=model or DEFAULT_MODEL,
            timeout=timeout,
        )
    if name in ("faster-whisper", "faster_whisper", "whisper"):
        return FasterWhisperBackend(model=model or DEFAULT_FASTER_WHISPER_MODEL)
    if name == "none":
        raise STTUnavailableError("STT provider 为 none，未接入转写后端")
    raise STTUnavailableError(
        f"未知 STT provider：{provider!r}（可选：{', '.join(PROVIDERS)}）"
    )
