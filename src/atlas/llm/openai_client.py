import json
import re
import time
from typing import Any

from atlas.core.config import Settings
from atlas.llm.clients import LLMClient, OpenAIClient
from atlas.llm.base import GeneratedAnswer, LLMUsage
from atlas.llm.prompts import ANSWER_INSTRUCTIONS, build_answer_input
from atlas.retrieval.models.evidence import Evidence


class OpenAIAnswerGenerator:
    def __init__(self, settings: Settings, *, client: LLMClient | None = None) -> None:
        self.settings = settings
        self.model_name = settings.llm_model
        self.client = client

    def generate(self, *, query: str, evidence: list[Evidence]) -> GeneratedAnswer:
        client = self.client or OpenAIClient(self.settings)
        request: dict[str, Any] = {
            "model": self.settings.llm_model,
            "instructions": ANSWER_INSTRUCTIONS,
            "input": build_answer_input(query=query, evidence=evidence),
            "max_output_tokens": self.settings.llm_max_output_tokens,
            "reasoning": {"effort": self.settings.llm_reasoning_effort},
            "store": False,
        }

        started = time.perf_counter()
        response = client.create_response(request)

        _ = int((time.perf_counter() - started) * 1000)
        raw_output = response.output_text
        parsed = _parse_json_output(raw_output)
        usage = response.usage
        return GeneratedAnswer(
            answer=parsed.get("answer") or raw_output,
            confidence=_normalize_confidence(parsed.get("confidence")),
            usage=LLMUsage(
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
            ),
            raw_output=raw_output,
        )


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
