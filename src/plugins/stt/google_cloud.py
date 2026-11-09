"""Google Cloud Speech-to-Text **v2** REST backend.

Only the synchronous ``recognize`` method is used; v2 accepts one audio payload
of at most about one minute, which matches the STT plugin's fixed-length chunked
worker.  Audio is sent as a WAV container with ``autoDecodingConfig`` so the
newer ``chirp_*`` models work as well as the classic ``long``/``short`` models.

v2 rejects API keys, so :class:`GoogleSpeechClient` takes a callable returning
an OAuth access token (see :mod:`plugins.stt.credentials`).  Google's optional
"model adaptation" (uploading phrase sets / custom classes of specialized
terms) is not used.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, replace
from typing import Any, Callable

import httpx

from .errors import SttError

logger = logging.getLogger("lecture_agent.plugin.stt")

DEFAULT_LOCATION = "us"
DEFAULT_MODEL = "chirp_3"
# ``chirp_*`` models are not served from the ``global`` endpoint; they live in
# regional ones.  When a chirp model is paired with ``global`` we retry against
# this region before giving up.
CHIRP_FALLBACK_LOCATION = "us"
# Last-resort model: it exists in every location and supports the languages we
# care about (including cmn-Hans-CN), so transcripts keep flowing even when the
# requested model/region pair is rejected.
FALLBACK_MODEL = "long"

MODEL_OPTIONS: tuple[tuple[str, str], ...] = (
    ("chirp_3", "chirp_3（最新多语言模型，推荐）"),
    ("chirp_2", "chirp_2（上一代多语言模型）"),
    ("long", "long（长音频）"),
    ("short", "short（短音频）"),
    ("telephony", "telephony（电话音频）"),
    ("telephony_short", "telephony_short（电话短音频）"),
)


@dataclass(frozen=True, slots=True)
class SpeechConfig:
    project_id: str
    location: str = DEFAULT_LOCATION
    model: str = DEFAULT_MODEL
    language_code: str = "cmn-Hans-CN"
    punctuation: bool = True
    timeout: float = 60.0


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """One recognized span, with offsets relative to the audio payload."""

    text: str
    start: float
    end: float
    confidence: float = 0.0
    language_code: str = ""


class GoogleSpeechClient:
    """Thin wrapper around v2 ``recognizers.recognize``."""

    def __init__(
        self,
        config: SpeechConfig,
        access_token: Callable[[], str],
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._access_token = access_token
        self._used_config: SpeechConfig = config
        self._client = httpx.Client(
            timeout=max(1.0, float(config.timeout)),
            transport=transport,
        )

    @property
    def used_config(self) -> SpeechConfig:
        """The model/location actually accepted by the last request."""

        return self._used_config

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GoogleSpeechClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # URLs
    # ------------------------------------------------------------------
    @property
    def location(self) -> str:
        return self._config.location.strip() or DEFAULT_LOCATION

    def endpoint_base(self, location: str | None = None) -> str:
        location = (location or self.location).strip().lower() or DEFAULT_LOCATION
        if location == "global":
            return "https://speech.googleapis.com"
        return f"https://{location}-speech.googleapis.com"

    def recognize_url(self, config: SpeechConfig | None = None) -> str:
        config = config or self._config
        project = config.project_id.strip()
        if not project:
            raise SttError("缺少 Google Cloud project ID（凭据里没有或设置项为空）")
        location = config.location.strip() or DEFAULT_LOCATION
        return (
            f"{self.endpoint_base(location)}/v2/projects/{project}"
            f"/locations/{location}/recognizers/_:recognize"
        )

    def _candidates(self) -> list[SpeechConfig]:
        """Configurations to try, in order.

        Google rejects a model that is not served in the configured location
        with HTTP 400.  ``chirp_*`` models are regional, so a stored
        ``global`` location silently breaks transcription.  Retry with a
        regional endpoint and finally with the always-available ``long``
        model so audio is still transcribed.
        """

        config = self._config
        model = config.model.strip() or DEFAULT_MODEL
        location = config.location.strip() or DEFAULT_LOCATION
        out: list[SpeechConfig] = [replace(config, model=model, location=location)]
        if model.lower().startswith("chirp") and location.lower() == "global":
            out.append(
                replace(config, model=model, location=CHIRP_FALLBACK_LOCATION)
            )
        if model.lower() != FALLBACK_MODEL:
            out.append(replace(config, model=FALLBACK_MODEL, location=location))
        seen: set[tuple[str, str]] = set()
        unique: list[SpeechConfig] = []
        for candidate in out:
            key = (candidate.location.lower(), candidate.model.lower())
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    # ------------------------------------------------------------------
    # Request
    # ------------------------------------------------------------------
    def recognize(
        self,
        wav: bytes,
        *,
        duration: float | None = None,
    ) -> list[SpeechSegment]:
        """Transcribe a WAV container."""

        if not wav:
            return []
        token = str(self._access_token() or "").strip()
        if not token:
            raise SttError("未获取到 Google OAuth token")

        candidates = self._candidates()
        last_error: SttError | None = None
        for index, config in enumerate(candidates):
            try:
                data = self._post(config, token, wav)
            except SttError as exc:
                if not _is_model_location_error(str(exc)) or index + 1 >= len(candidates):
                    raise
                last_error = exc
                following = candidates[index + 1]
                logger.warning(
                    "STT %s@%s unavailable (%s); retrying with %s@%s",
                    config.model,
                    config.location,
                    exc,
                    following.model,
                    following.location,
                )
                continue
            self._used_config = config
            return _parse_results(data, duration)
        if last_error is not None:  # pragma: no cover - loop always raises
            raise last_error
        return []

    def _post(self, config: SpeechConfig, token: str, wav: bytes) -> Any:
        payload = {
            "config": {
                "autoDecodingConfig": {},
                "languageCodes": [config.language_code],
                "model": config.model,
                "features": {
                    "enableAutomaticPunctuation": bool(config.punctuation),
                },
            },
            "content": base64.b64encode(wav).decode("ascii"),
        }
        try:
            response = self._client.post(
                self.recognize_url(config),
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            raise SttError(f"请求 Google STT v2 失败：{exc}") from exc

        data = _parse_json(response)
        if response.status_code >= 400:
            raise SttError(_error_message(response.status_code, data))
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise SttError(
            f"Google STT v2 返回了非 JSON 响应（HTTP {response.status_code}）"
        ) from exc


def _is_model_location_error(text: str) -> bool:
    """True when Google rejected the model/region combination."""

    message = str(text).lower()
    markers = (
        "does not exist in the location",
        "not supported in the location",
        "is not available in",
        "unsupported model",
        "unknown model",
    )
    return any(marker in message for marker in markers)


def _error_message(status: int, data: Any) -> str:
    message = ""
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "")
        elif error is not None:
            message = str(error)
    if not message:
        message = "未知错误"
    hint = ""
    if status == 401:
        hint = "；请检查服务账号 / OAuth 凭据是否有效且未过期"
    elif status == 403:
        hint = (
            "；请确认已启用 Speech-to-Text API，且账号有 "
            "speech.recognizers.recognize 权限"
        )
    elif status == 404:
        hint = "；请确认 project ID / location / recognizer 是否正确"
    elif status == 429:
        hint = "；配额或速率超限"
    elif status == 400 and _is_model_location_error(message):
        hint = (
            "；该模型在当前区域不可用（chirp_* 不支持 global），"
            "请把区域改成 us 或 asia-southeast1 等区域"
        )
    return f"Google STT v2 HTTP {status}：{message}{hint}"



def _parse_duration(value: Any) -> float:
    """Parse a protobuf duration string such as ``"3.500s"``."""

    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    if text.endswith("s"):
        text = text[:-1]
    try:
        return max(0.0, float(text))
    except ValueError:
        return 0.0


def _parse_results(data: Any, duration: float | None) -> list[SpeechSegment]:
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return []
    chunk_end = float(duration) if duration and duration > 0 else 0.0
    segments: list[SpeechSegment] = []
    previous_end = 0.0
    for result in results:
        if not isinstance(result, dict):
            continue
        # v2 uses resultEndOffset; keep resultEndTime as a defensive fallback.
        end = _parse_duration(
            result.get("resultEndOffset") or result.get("resultEndTime")
        ) or chunk_end
        if end < previous_end:
            end = previous_end
        start = previous_end
        previous_end = end
        alternatives = result.get("alternatives")
        if not isinstance(alternatives, list) or not alternatives:
            continue
        top = alternatives[0]
        if not isinstance(top, dict):
            continue
        text = str(top.get("transcript") or "").strip()
        if not text:
            continue
        try:
            confidence = float(top.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        segments.append(
            SpeechSegment(
                text=text,
                start=start,
                end=end,
                confidence=confidence,
                language_code=str(result.get("languageCode") or ""),
            )
        )
    return segments


__all__ = [
    "DEFAULT_LOCATION",
    "DEFAULT_MODEL",
    "GoogleSpeechClient",
    "MODEL_OPTIONS",
    "SpeechConfig",
    "SpeechSegment",
    "SttError",
]
