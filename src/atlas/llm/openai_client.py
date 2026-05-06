import json
import re
import time
from typing import Any

from openai import OpenAI

from atlas.core.config import Settings
from atlas.core.errors import AtlasError, ErrorCode
from atlas.llm.base import GeneratedAnswer, LLMUsage
from atlas.llm.prompts import ANSWER_INSTRUCTIONS, build_answer_input
from atlas.retrieval.models.evidence import Evidence


class OpenAIAnswerGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.llm_model

    def generate(self, *, query: str, evidence: list[Evidence]) -> GeneratedAnswer:
        if self.settings.openai_api_key is None:
            raise AtlasError(
                ErrorCode.CONFIGURATION_ERROR,
                "OPENAI_API_KEY is required for answer generation.",
                status_code=500,
            )

        client = OpenAI(
            api_key=self.settings.openai_api_key.get_secret_value(),
            timeout=self.settings.llm_timeout_seconds,
        )
        request: dict[str, Any] = {
            "model": self.settings.llm_model,
            "instructions": ANSWER_INSTRUCTIONS,
            "input": build_answer_input(query=query, evidence=evidence),
            "max_output_tokens": self.settings.llm_max_output_tokens,
            "reasoning": {"effort": self.settings.llm_reasoning_effort},
            "store": False,
        }

        started = time.perf_counter()
        try:
            try:
                response = client.responses.create(**request)
            except TypeError:
                request.pop("store", None)
                response = client.responses.create(**request)
        except Exception as exc:
            raise AtlasError(
                ErrorCode.UPSTREAM_LLM_UNAVAILABLE,
                "OpenAI response generation failed.",
                status_code=502,
                details={"type": exc.__class__.__name__},
            ) from exc

        _ = int((time.perf_counter() - started) * 1000)
        raw_output = _extract_output_text(response)
        parsed = _parse_json_output(raw_output)
        usage = getattr(response, "usage", None)
        return GeneratedAnswer(
            answer=parsed.get("answer") or raw_output,
            confidence=_normalize_confidence(parsed.get("confidence")),
            usage=LLMUsage(
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
            ),
            raw_output=raw_output,
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


def _parse_json_output(raw_output: str) -> dict[str, str]:
    try:
        value = json.loads(raw_output)
        if isinstance(value, dict):
            return {str(key): str(val) for key, val in value.items()}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return {str(key): str(val) for key, val in value.items()}
        except json.JSONDecodeError:
            pass

    return {"confidence": "unknown", "answer": raw_output}


def _normalize_confidence(value: str | None) -> str:
    if value in {"supported", "insufficient", "conflicted"}:
        return value
    return "unknown"
