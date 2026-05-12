from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Callable

from qdrant_client import QdrantClient

from atlas.core.config import Settings
from atlas.core.registry import ComponentRegistry
from atlas.embeddings.base import Embedder
from atlas.embeddings.bm25_sparse import BM25SparseEncoder
from atlas.retrieval.providers.base import RetrievalProvider
from atlas.retrieval.providers.graph import GraphProvider, GraphStore
from atlas.retrieval.providers.sql import SQLProvider
from atlas.retrieval.providers.text_hybrid import TextHybridProvider
from atlas.retrieval.providers.text_hybrid.adapters.bm25 import BM25Retriever
from atlas.retrieval.providers.text_hybrid.adapters.dense import DenseRetriever
from atlas.retrieval.providers.text_hybrid.adapters.hybrid import HybridRetriever
from atlas.retrieval.ranking.reranker import Reranker


ProviderFactory = Callable[["ProviderBuildContext"], RetrievalProvider]


@dataclass(frozen=True)
class ProviderBuildContext:
    settings: Settings
    qdrant_factory: Callable[[], QdrantClient]
    embedder_factory: Callable[[], Embedder]
    sparse_encoder_factory: Callable[[], BM25SparseEncoder]
    reranker_factory: Callable[[], Reranker | None]
    graph_store_factory: Callable[[], GraphStore]

    @cached_property
    def qdrant(self) -> QdrantClient:
        return self.qdrant_factory()

    @cached_property
    def embedder(self) -> Embedder:
        return self.embedder_factory()

    @cached_property
    def sparse_encoder(self) -> BM25SparseEncoder:
        return self.sparse_encoder_factory()

    @cached_property
    def reranker(self) -> Reranker | None:
        return self.reranker_factory()

    @cached_property
    def graph_store(self) -> GraphStore:
        return self.graph_store_factory()


provider_registry: ComponentRegistry[ProviderFactory] = ComponentRegistry(
    namespace="retrieval_provider"
)


def build_provider(name: str, context: ProviderBuildContext) -> RetrievalProvider:
    return provider_registry.build(name, context)


def build_providers(
    names: tuple[str, ...],
    context: ProviderBuildContext,
) -> dict[str, RetrievalProvider]:
    return {name: build_provider(name, context) for name in names}


def _build_text_hybrid_provider(context: ProviderBuildContext) -> TextHybridProvider:
    settings = context.settings
    dense = DenseRetriever(
        settings=settings,
        embedder=context.embedder,
        qdrant=context.qdrant,
    )
    bm25 = BM25Retriever(
        settings=settings,
        sparse_encoder=context.sparse_encoder,
        qdrant=context.qdrant,
    )
    hybrid_rerank = HybridRetriever(
        dense,
        bm25,
        rrf_k=settings.hybrid_rrf_k,
        rrf_top_k=settings.rrf_top_k,
        reranker=context.reranker,
        reranker_enabled=True,
        reranker_top_k=settings.reranker_top_k,
        reranker_output_k=settings.reranker_output_k,
        dense_top_k=settings.hybrid_dense_top_k,
        lexical_top_k=settings.hybrid_lexical_top_k,
        max_context_tokens=settings.max_context_tokens,
    )
    hybrid_rrf = HybridRetriever(
        dense,
        bm25,
        rrf_k=settings.hybrid_rrf_k,
        rrf_top_k=settings.rrf_top_k,
        reranker=None,
        reranker_enabled=False,
        reranker_top_k=settings.reranker_top_k,
        reranker_output_k=settings.reranker_output_k,
        dense_top_k=settings.hybrid_dense_top_k,
        lexical_top_k=settings.hybrid_lexical_top_k,
        max_context_tokens=settings.max_context_tokens,
    )
    default_mode = settings.retrieval_mode.strip().lower()
    if default_mode == "hybrid" and not settings.reranker_enabled:
        default_mode = "hybrid_rrf"
    return TextHybridProvider(
        dense_retriever=dense,
        bm25_retriever=bm25,
        hybrid_rrf_retriever=hybrid_rrf,
        hybrid_rerank_retriever=hybrid_rerank,
        default_mode=default_mode,
        rrf_k=settings.hybrid_rrf_k,
        rrf_top_k=settings.rrf_top_k,
        reranker=context.reranker,
        reranker_enabled=settings.reranker_enabled,
        reranker_top_k=settings.reranker_top_k,
        reranker_output_k=settings.reranker_output_k,
        dense_top_k=settings.hybrid_dense_top_k,
        lexical_top_k=settings.hybrid_lexical_top_k,
        max_context_tokens=settings.max_context_tokens,
    )


def _build_graph_provider(context: ProviderBuildContext) -> GraphProvider:
    return GraphProvider(
        store=context.graph_store,
        max_context_tokens=context.settings.max_context_tokens,
    )


def _build_sql_provider(context: ProviderBuildContext) -> SQLProvider:
    return SQLProvider(settings=context.settings)


def _register_builtin_providers() -> None:
    for name, factory in {
        "hybrid": _build_text_hybrid_provider,
        "graph": _build_graph_provider,
        "sql": _build_sql_provider,
    }.items():
        if name not in provider_registry.names:
            provider_registry.register(name, factory)


_register_builtin_providers()
