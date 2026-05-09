from __future__ import annotations

from abc import ABC

from atlas.backends.base import Backend, BackendBuildContext, BackendRegistry
from atlas.embeddings.base import Embedder
from atlas.embeddings.bge_local import LocalBGEEmbedder


class EmbeddingBackend(Backend[Embedder], ABC):
    pass


class LocalBGEBackend(EmbeddingBackend):
    def build(self, context: BackendBuildContext) -> Embedder:
        return LocalBGEEmbedder(context.settings)


embedding_backends: BackendRegistry[Embedder] = BackendRegistry(
    namespace="embedding",
    backend_type=EmbeddingBackend,
)


def build_embedder(name: str, context: BackendBuildContext) -> Embedder:
    return embedding_backends.build(name, context)


def _register_builtin_backends() -> None:
    if "local_bge" not in embedding_backends.names:
        embedding_backends.register("local_bge", LocalBGEBackend())


_register_builtin_backends()
