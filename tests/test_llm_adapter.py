from atlas.core.config import Settings
from atlas.llm.clients.base import LLMResponse
from atlas.llm.openai_client import OpenAIAnswerGenerator
from atlas.retrieval.models.evidence import Evidence


class _FakeClient:
    def __init__(self) -> None:
        self.requests = []

    def create_response(self, request):
        self.requests.append(request)
        return LLMResponse(
            output_text='{"confidence":"supported","answer":"3M capex was 1,577 [c1]."}',
            raw={"ok": True},
            usage=type("Usage", (), {"input_tokens": 12, "output_tokens": 9})(),
        )


def test_answer_generator_uses_llm_client_adapter() -> None:
    client = _FakeClient()
    generator = OpenAIAnswerGenerator(Settings(openai_api_key=None), client=client)
    evidence = [
        Evidence(
            evidence_id="c1",
            document_id="doc_1",
            chunk_id="chk_1",
            text="3M capital expenditures were 1,577.",
            source_title="3M 10-K",
            source_uri=None,
            section_title=None,
            page_start=1,
            page_end=1,
            retrieval_score=1.0,
            rank=1,
            token_count=8,
        )
    ]

    answer = generator.generate(query="What was 3M capex?", evidence=evidence)

    assert client.requests
    assert answer.confidence == "supported"
    assert answer.usage.input_tokens == 12
