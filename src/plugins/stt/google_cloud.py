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
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from .errors import SttError

DEFAULT_LOCATION = "us"
DEFAULT_MODEL = "chirp_3"

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
        self._client = httpx.Client(
            timeout=max(1.0, float(config.timeout)),
            transport=transport,
        )

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

    def endpoint_base(self) -> str:
        location = self.location.lower()
        if location == "global":
            return "https://speech.googleapis.com"
        return f"https://{location}-speech.googleapis.com"

    def recognize_url(self) -> str:
        project = self._config.project_id.strip()
        if not project:
            raise SttError("缺少 Google Cloud project ID（凭据里没有或设置项为空）")
        return (
            f"{self.endpoint_base()}/v2/projects/{project}"
            f"/locations/{self.location}/recognizers/_:recognize"
        )

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

        payload = {
            "config": {
                "autoDecodingConfig": {},
                "languageCodes": [self._config.language_code],
                "model": self._config.model,
                "features": {
                    "enableAutomaticPunctuation": bool(self._config.punctuation),
                },
            },
            "content": base64.b64encode(wav).decode("ascii"),
        }
        try:
            response = self._client.post(
                self.recognize_url(),
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
        return _parse_results(data, duration)


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
    elif status == 429:
        hint = "；配额或速率超限"
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
