from functools import lru_cache

from qdrant_client import QdrantClient

from atlas.backends import (
    BackendBuildContext,
    SparseEncoder,
    build_answer_generator,
    build_embedder,
    build_graph_store,
    build_llm_client,
    build_reranker,
    build_sparse_encoder,
    build_vector_store,
)
from atlas.core.config import (
    Settings,
    bm25_sparse_enabled,
    executable_query_providers,
    get_settings,
    known_query_providers,
)
from atlas.db.session import SessionLocal
from atlas.embeddings.base import Embedder
from atlas.eval.service import EvalService
from atlas.ingestion.service import IngestionService
from atlas.llm.base import AnswerGenerator
from atlas.llm.clients import LLMClient
from atlas.query_orchestrator.llm_planner import LLMQueryPlanner
from atlas.query_orchestrator.ontology import FinanceMetricOntology
from atlas.query_orchestrator.service import QueryOrchestrator
from atlas.query_runtime.service import QueryRuntime
from atlas.retrieval.providers.graph import GraphProvider, GraphStore
from atlas.retrieval.providers.registry import (
    ProviderBuildContext,
    build_provider,
)
from atlas.retrieval.providers.text_hybrid import TextHybridProvider
from atlas.retrieval.ranking.reranker import Reranker
from atlas.retrieval.router import ProviderRouter


def _backend_build_context() -> BackendBuildContext:
    return BackendBuildContext(settings=get_settings())


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    return build_embedder(settings.embedding_backend, _backend_build_context())


@lru_cache
def get_sparse_encoder() -> SparseEncoder:
    settings = get_settings()
    return build_sparse_encoder(settings.sparse_backend, _backend_build_context())


@lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return build_vector_store(settings.vector_store_backend, _backend_build_context())


@lru_cache
def get_graph_store() -> GraphStore:
    settings = get_settings()
    return build_graph_store(settings.graph_store_backend, _backend_build_context())


@lru_cache
def get_llm_client() -> LLMClient:
    settings = get_settings()
    return build_llm_client(settings.llm_client_backend, _backend_build_context())


@lru_cache
def get_ingestion_service() -> IngestionService:
    settings = get_settings()
    return IngestionService(
        settings=settings,
        embedder=get_embedder(),
        qdrant=get_qdrant_client(),
        sparse_encoder=get_sparse_encoder() if bm25_sparse_enabled(settings) else None,
    )


def _provider_build_context() -> ProviderBuildContext:
    settings = get_settings()
    return ProviderBuildContext(
        settings=settings,
        qdrant_factory=get_qdrant_client,
        embedder_factory=get_embedder,
        sparse_encoder_factory=get_sparse_encoder,
        reranker_factory=get_reranker,
        graph_store_factory=get_graph_store,
    )


@lru_cache
def get_reranker() -> Reranker | None:
    settings = get_settings()
    return build_reranker(settings.reranker_backend, _backend_build_context())


@lru_cache
def get_retriever():
    return get_text_hybrid_provider()


@lru_cache
def get_text_hybrid_provider() -> TextHybridProvider:
    return build_provider("hybrid", _provider_build_context())


@lru_cache
def get_graph_provider() -> GraphProvider:
    return build_provider("graph", _provider_build_context())


@lru_cache
def get_provider_router() -> ProviderRouter:
    settings = get_settings()
    executable_providers = executable_query_providers(settings)
    providers = {
        name: _get_provider_by_name(name)
        for name in executable_providers
    }
    return ProviderRouter(
        providers,
        known_providers=known_query_providers(settings),
        session_factory=SessionLocal,
        reranker=get_reranker(),
        reranker_enabled=settings.reranker_enabled,
        reranker_top_k=settings.reranker_top_k,
        reranker_output_k=settings.reranker_output_k,
        max_context_tokens=settings.max_context_tokens,
    )


def _get_provider_by_name(name: str):
    if name == "hybrid":
        return get_text_hybrid_provider()
    if name == "graph":
        return get_graph_provider()
    return build_provider(name, _provider_build_context())


@lru_cache
def get_answer_generator() -> AnswerGenerator:
    settings = get_settings()
    return build_answer_generator(settings.answer_generator_backend, _backend_build_context())


@lru_cache
def get_query_orchestrator() -> QueryOrchestrator:
    settings = get_settings()
    if settings.openai_api_key is None:
        return QueryOrchestrator(settings=settings)
    llm_client = get_llm_client()
    ontology = FinanceMetricOntology.load(settings.finance_metric_ontology_path)
    return QueryOrchestrator(
        settings=settings,
        ontology=ontology,
        llm_planner=LLMQueryPlanner(
            settings=settings,
            ontology=ontology,
            client=llm_client,
        ),
    )


@lru_cache
def get_query_runtime() -> QueryRuntime:
    settings = get_settings()
    return QueryRuntime(
        settings=settings,
        retriever=get_text_hybrid_provider(),
        provider_router=get_provider_router(),
        generator=get_answer_generator(),
        orchestrator=get_query_orchestrator(),
    )


@lru_cache
def get_eval_service() -> EvalService:
    return EvalService(get_query_runtime())


def settings_dependency() -> Settings:
    return get_settings()
