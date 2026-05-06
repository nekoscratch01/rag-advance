from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMResponse:
    output_text: str
    raw: Any
    usage: Any | None = None


class LLMClient(Protocol):
    def create_response(self, request: dict[str, Any]) -> LLMResponse:
        ...

