from __future__ import annotations

from abc import ABC

from atlas.backends.base import Backend, BackendBuildContext, BackendRegistry
from atlas.retrieval.ranking.reranker import CrossEncoderReranker, Reranker


class RerankerBackend(Backend[Reranker], ABC):
    pass


class CrossEncoderBackend(RerankerBackend):
    def build(self, context: BackendBuildContext) -> Reranker:
        return CrossEncoderReranker(context.settings.reranker_model)


reranker_backends: BackendRegistry[Reranker] = BackendRegistry(
    namespace="reranker",
    backend_type=RerankerBackend,
)


def build_reranker(name: str, context: BackendBuildContext) -> Reranker:
    return reranker_backends.build(name, context)


def _register_builtin_backends() -> None:
    if "cross_encoder" not in reranker_backends.names:
        reranker_backends.register("cross_encoder", CrossEncoderBackend())


_register_builtin_backends()
