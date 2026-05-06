from functools import lru_cache

from atlas.core.config import Settings, bm25_sparse_enabled, get_settings
from atlas.embeddings.bm25_sparse import BM25SparseEncoder
from atlas.embeddings.bge_local import LocalBGEEmbedder
from atlas.ingestion.service import IngestionService
from atlas.eval.service import EvalService
from atlas.llm.openai_client import OpenAIAnswerGenerator
from atlas.query_orchestrator.service import QueryOrchestrator
from atlas.query_runtime.service import QueryRuntime
from atlas.retrieval.retrievers.bm25 import BM25Retriever
from atlas.retrieval.retrievers.dense import DenseRetriever
from atlas.retrieval.retrievers.hybrid import HybridRetriever
from atlas.retrieval.providers.text_hybrid import TextHybridProvider
from atlas.retrieval.ranking.reranker import CrossEncoderReranker
from atlas.vector.qdrant_client import get_qdrant_client


@lru_cache
def get_embedder() -> LocalBGEEmbedder:
    return LocalBGEEmbedder(get_settings())


@lru_cache
def get_sparse_encoder() -> BM25SparseEncoder:
    return BM25SparseEncoder(get_settings())


@lru_cache
def get_ingestion_service() -> IngestionService:
    settings = get_settings()
    return IngestionService(
        settings=settings,
        embedder=get_embedder(),
        qdrant=get_qdrant_client(),
        sparse_encoder=get_sparse_encoder() if bm25_sparse_enabled(settings) else None,
    )


@lru_cache
def get_dense_retriever() -> DenseRetriever:
    settings = get_settings()
    return DenseRetriever(
        settings=settings,
        embedder=get_embedder(),
        qdrant=get_qdrant_client(),
    )


@lru_cache
def get_bm25_retriever() -> BM25Retriever:
    settings = get_settings()
    return BM25Retriever(
        settings=settings,
        sparse_encoder=get_sparse_encoder(),
        qdrant=get_qdrant_client(),
    )


@lru_cache
def get_reranker() -> CrossEncoderReranker | None:
    settings = get_settings()
    return CrossEncoderReranker(settings.reranker_model)


@lru_cache
def get_retriever():
    settings = get_settings()
    hybrid_rerank = HybridRetriever(
        get_dense_retriever(),
        get_bm25_retriever(),
        rrf_k=settings.hybrid_rrf_k,
        rrf_top_k=settings.rrf_top_k,
        reranker=get_reranker(),
        reranker_enabled=True,
        reranker_top_k=settings.reranker_top_k,
        reranker_output_k=settings.reranker_output_k,
        dense_top_k=settings.hybrid_dense_top_k,
        lexical_top_k=settings.hybrid_lexical_top_k,
        max_context_tokens=settings.max_context_tokens,
    )
    hybrid_rrf = HybridRetriever(
        get_dense_retriever(),
        get_bm25_retriever(),
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
        dense_retriever=get_dense_retriever(),
        bm25_retriever=get_bm25_retriever(),
        hybrid_rrf_retriever=hybrid_rrf,
        hybrid_rerank_retriever=hybrid_rerank,
        default_mode=default_mode,
        rrf_k=settings.hybrid_rrf_k,
        rrf_top_k=settings.rrf_top_k,
        reranker=get_reranker(),
        reranker_enabled=settings.reranker_enabled,
        reranker_top_k=settings.reranker_top_k,
        reranker_output_k=settings.reranker_output_k,
        dense_top_k=settings.hybrid_dense_top_k,
        lexical_top_k=settings.hybrid_lexical_top_k,
        max_context_tokens=settings.max_context_tokens,
    )


@lru_cache
def get_answer_generator() -> OpenAIAnswerGenerator:
    return OpenAIAnswerGenerator(get_settings())


@lru_cache
def get_query_orchestrator() -> QueryOrchestrator:
    return QueryOrchestrator(settings=get_settings())


@lru_cache
def get_query_runtime() -> QueryRuntime:
    settings = get_settings()
    return QueryRuntime(
        settings=settings,
        retriever=get_retriever(),
        generator=get_answer_generator(),
        orchestrator=get_query_orchestrator(),
    )


@lru_cache
def get_eval_service() -> EvalService:
    return EvalService(get_query_runtime())


def settings_dependency() -> Settings:
    return get_settings()
