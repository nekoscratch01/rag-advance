from __future__ import annotations

from abc import ABC
from typing import Any, Protocol

from atlas.backends.base import Backend, BackendBuildContext, BackendRegistry
from atlas.embeddings.bm25_sparse import BM25SparseEncoder


class SparseEncoder(Protocol):
    model_name: str

    def embed_texts(self, texts: list[str]) -> list[Any]:
        ...

    def embed_query(self, query: str) -> Any:
        ...


class SparseBackend(Backend[SparseEncoder], ABC):
    pass


class FastEmbedBM25Backend(SparseBackend):
    def build(self, context: BackendBuildContext) -> SparseEncoder:
        return BM25SparseEncoder(context.settings)


sparse_backends: BackendRegistry[SparseEncoder] = BackendRegistry(
    namespace="sparse",
    backend_type=SparseBackend,
)


def build_sparse_encoder(name: str, context: BackendBuildContext) -> SparseEncoder:
    return sparse_backends.build(name, context)


def _register_builtin_backends() -> None:
    if "fastembed_bm25" not in sparse_backends.names:
        sparse_backends.register("fastembed_bm25", FastEmbedBM25Backend())


_register_builtin_backends()
