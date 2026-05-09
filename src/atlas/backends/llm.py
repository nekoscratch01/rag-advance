from __future__ import annotations

from abc import ABC

from atlas.backends.base import Backend, BackendBuildContext, BackendRegistry
from atlas.llm.base import AnswerGenerator
from atlas.llm.clients import LLMClient, OpenAIClient
from atlas.llm.openai_client import OpenAIAnswerGenerator


class LLMClientBackend(Backend[LLMClient], ABC):
    pass


class AnswerGeneratorBackend(Backend[AnswerGenerator], ABC):
    pass


class OpenAILLMClientBackend(LLMClientBackend):
    def build(self, context: BackendBuildContext) -> LLMClient:
        return OpenAIClient(context.settings)


class OpenAIAnswerGeneratorBackend(AnswerGeneratorBackend):
    def build(self, context: BackendBuildContext) -> AnswerGenerator:
        return OpenAIAnswerGenerator(context.settings)


llm_client_backends: BackendRegistry[LLMClient] = BackendRegistry(
    namespace="llm_client",
    backend_type=LLMClientBackend,
)
answer_generator_backends: BackendRegistry[AnswerGenerator] = BackendRegistry(
    namespace="answer_generator",
    backend_type=AnswerGeneratorBackend,
)


def build_llm_client(name: str, context: BackendBuildContext) -> LLMClient:
    return llm_client_backends.build(name, context)


def build_answer_generator(name: str, context: BackendBuildContext) -> AnswerGenerator:
    return answer_generator_backends.build(name, context)


def _register_builtin_backends() -> None:
    if "openai" not in llm_client_backends.names:
        llm_client_backends.register("openai", OpenAILLMClientBackend())
    if "openai" not in answer_generator_backends.names:
        answer_generator_backends.register("openai", OpenAIAnswerGeneratorBackend())


_register_builtin_backends()
