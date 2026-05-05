from dataclasses import dataclass

from atlas.retrieval.evidence import Evidence


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    confidence: str
    usage: LLMUsage
    raw_output: str


class AnswerGenerator:
    model_name: str

    def generate(self, *, query: str, evidence: list[Evidence]) -> GeneratedAnswer:
        raise NotImplementedError
