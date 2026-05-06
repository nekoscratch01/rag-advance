from __future__ import annotations

from typing import Any

from openai import OpenAI

from atlas.core.config import Settings
from atlas.core.errors import AtlasError, ErrorCode
from atlas.llm.clients.base import LLMResponse


class OpenAIClient:
    def __init__(self, settings: Settings) -> None:
        if settings.openai_api_key is None:
            raise AtlasError(
                ErrorCode.CONFIGURATION_ERROR,
                "OPENAI_API_KEY is required for OpenAI LLM calls.",
                status_code=500,
            )
        self.settings = settings
        self._client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
        )

    def create_response(self, request: dict[str, Any]) -> LLMResponse:
        payload = dict(request)
        try:
            try:
                response = self._client.responses.create(**payload)
            except TypeError:
                payload.pop("store", None)
                try:
                    response = self._client.responses.create(**payload)
                except TypeError:
                    payload.pop("text", None)
                    response = self._client.responses.create(**payload)
        except Exception as exc:
            raise AtlasError(
                ErrorCode.UPSTREAM_LLM_UNAVAILABLE,
                "OpenAI response generation failed.",
                status_code=502,
                details={"type": exc.__class__.__name__},
            ) from exc
        return LLMResponse(
            output_text=_extract_output_text(response),
            raw=response,
            usage=getattr(response, "usage", None),
        )


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()
