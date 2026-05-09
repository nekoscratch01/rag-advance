from __future__ import annotations

from abc import ABC

from qdrant_client import QdrantClient

from atlas.backends.base import Backend, BackendBuildContext, BackendRegistry


class VectorStoreBackend(Backend[QdrantClient], ABC):
    pass


class QdrantVectorStoreBackend(VectorStoreBackend):
    def build(self, context: BackendBuildContext) -> QdrantClient:
        return QdrantClient(url=context.settings.qdrant_url)


vector_store_backends: BackendRegistry[QdrantClient] = BackendRegistry(
    namespace="vector_store",
    backend_type=VectorStoreBackend,
)


def build_vector_store(name: str, context: BackendBuildContext) -> QdrantClient:
    return vector_store_backends.build(name, context)


def _register_builtin_backends() -> None:
    if "qdrant" not in vector_store_backends.names:
        vector_store_backends.register("qdrant", QdrantVectorStoreBackend())


_register_builtin_backends()
