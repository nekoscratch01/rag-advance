import pytest
from qdrant_client import QdrantClient

from atlas.api import dependencies as dependency_module
from atlas.backends import (
    BackendBuildContext,
    answer_generator_backends,
    build_answer_generator,
    build_embedder,
    build_graph_store,
    build_llm_client,
    build_reranker,
    build_sparse_encoder,
    build_vector_store,
    embedding_backends,
    graph_store_backends,
    llm_client_backends,
    reranker_backends,
    sparse_backends,
    vector_store_backends,
)
from atlas.core.config import Settings
from atlas.core.errors import AtlasError, ErrorCode
from atlas.embeddings.bge_local import LocalBGEEmbedder
from atlas.embeddings.bm25_sparse import BM25SparseEncoder
from atlas.llm.clients import OpenAIClient
from atlas.llm.openai_client import OpenAIAnswerGenerator
from atlas.query_runtime.service import QueryRuntime
from atlas.retrieval.providers.graph import PostgresGraphStore
from atlas.retrieval.providers.registry import ProviderBuildContext, build_provider
from atlas.retrieval.ranking.reranker import CrossEncoderReranker
from atlas.vector import qdrant_client as legacy_qdrant_module


def _context(**overrides) -> BackendBuildContext:
    return BackendBuildContext(settings=Settings(openai_api_key="sk-test", **overrides))


def test_current_backend_implementations_are_registered_under_named_defaults() -> None:
    assert embedding_backends.names == ("local_bge",)
    assert sparse_backends.names == ("fastembed_bm25",)
    assert reranker_backends.names == ("cross_encoder",)
    assert llm_client_backends.names == ("openai",)
    assert answer_generator_backends.names == ("openai",)
    assert vector_store_backends.names == ("qdrant",)
    assert graph_store_backends.names == ("postgres_graph",)


def test_backend_settings_default_to_registered_backend_names() -> None:
    settings = Settings(openai_api_key=None)

    assert settings.embedding_backend == "local_bge"
    assert settings.sparse_backend == "fastembed_bm25"
    assert settings.reranker_backend == "cross_encoder"
    assert settings.llm_client_backend == "openai"
    assert settings.answer_generator_backend == "openai"
    assert settings.vector_store_backend == "qdrant"
    assert settings.graph_store_backend == "postgres_graph"


def test_backend_registry_builds_current_default_implementations() -> None:
    context = _context()

    assert isinstance(build_embedder("local_bge", context), LocalBGEEmbedder)
    assert isinstance(build_sparse_encoder("fastembed_bm25", context), BM25SparseEncoder)
    assert isinstance(build_reranker("cross_encoder", context), CrossEncoderReranker)
    assert isinstance(build_llm_client("openai", context), OpenAIClient)
    assert isinstance(build_answer_generator("openai", context), OpenAIAnswerGenerator)
    assert isinstance(build_vector_store("qdrant", context), QdrantClient)
    assert isinstance(build_graph_store("postgres_graph", context), PostgresGraphStore)


@pytest.mark.parametrize(
    ("builder", "namespace"),
    [
        (build_embedder, "embedding"),
        (build_sparse_encoder, "sparse"),
        (build_reranker, "reranker"),
        (build_llm_client, "llm_client"),
        (build_answer_generator, "answer_generator"),
        (build_vector_store, "vector_store"),
        (build_graph_store, "graph_store"),
    ],
)
def test_invalid_backend_name_raises_clear_configuration_error(builder, namespace) -> None:
    with pytest.raises(AtlasError) as exc_info:
        builder("missing_backend", _context())

    error = exc_info.value
    assert error.error_code == ErrorCode.CONFIGURATION_ERROR
    assert f"Unknown {namespace} backend 'missing_backend'" in error.error_message
    assert error.details["backend_type"] == namespace
    assert error.details["backend"] == "missing_backend"


def test_dependencies_build_vector_store_from_backend_setting(monkeypatch) -> None:
    monkeypatch.setattr(
        dependency_module,
        "get_settings",
        lambda: Settings(openai_api_key=None, vector_store_backend="missing_backend"),
    )
    dependency_module.get_qdrant_client.cache_clear()

    with pytest.raises(AtlasError) as exc_info:
        dependency_module.get_qdrant_client()

    assert exc_info.value.error_code == ErrorCode.CONFIGURATION_ERROR
    assert "Unknown vector_store backend 'missing_backend'" in exc_info.value.error_message
    dependency_module.get_qdrant_client.cache_clear()


def test_dependencies_build_graph_store_from_backend_setting(monkeypatch) -> None:
    monkeypatch.setattr(
        dependency_module,
        "get_settings",
        lambda: Settings(openai_api_key=None, graph_store_backend="missing_backend"),
    )
    dependency_module.get_graph_store.cache_clear()
    dependency_module.get_graph_provider.cache_clear()

    with pytest.raises(AtlasError) as exc_info:
        dependency_module.get_graph_provider()

    assert exc_info.value.error_code == ErrorCode.CONFIGURATION_ERROR
    assert "Unknown graph_store backend 'missing_backend'" in exc_info.value.error_message
    dependency_module.get_graph_store.cache_clear()
    dependency_module.get_graph_provider.cache_clear()


def test_legacy_qdrant_helper_builds_vector_store_from_backend_setting(monkeypatch) -> None:
    monkeypatch.setattr(
        legacy_qdrant_module,
        "get_settings",
        lambda: Settings(openai_api_key=None, vector_store_backend="missing_backend"),
    )
    legacy_qdrant_module.get_qdrant_client.cache_clear()

    with pytest.raises(AtlasError) as exc_info:
        legacy_qdrant_module.get_qdrant_client()

    assert exc_info.value.error_code == ErrorCode.CONFIGURATION_ERROR
    assert "Unknown vector_store backend 'missing_backend'" in exc_info.value.error_message
    legacy_qdrant_module.get_qdrant_client.cache_clear()


def test_query_runtime_auto_wire_uses_backend_registry() -> None:
    with pytest.raises(AtlasError) as exc_info:
        QueryRuntime(
            settings=Settings(
                openai_api_key=None,
                embedding_backend="missing_backend",
                query_runtime_executable_providers="hybrid",
            ),
            generator=object(),
        )

    assert exc_info.value.error_code == ErrorCode.CONFIGURATION_ERROR
    assert "Unknown embedding backend 'missing_backend'" in exc_info.value.error_message


def test_query_runtime_router_reranker_uses_backend_registry() -> None:
    with pytest.raises(AtlasError) as exc_info:
        QueryRuntime(
            settings=Settings(
                openai_api_key=None,
                reranker_backend="missing_backend",
                query_runtime_executable_providers="graph",
            ),
            generator=object(),
        )

    assert exc_info.value.error_code == ErrorCode.CONFIGURATION_ERROR
    assert "Unknown reranker backend 'missing_backend'" in exc_info.value.error_message


def test_query_orchestrator_dependency_builds_llm_client_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        dependency_module,
        "get_settings",
        lambda: Settings(openai_api_key="sk-test", llm_client_backend="missing_backend"),
    )
    dependency_module.get_llm_client.cache_clear()
    dependency_module.get_query_orchestrator.cache_clear()

    with pytest.raises(AtlasError) as exc_info:
        dependency_module.get_query_orchestrator()

    assert exc_info.value.error_code == ErrorCode.CONFIGURATION_ERROR
    assert "Unknown llm_client backend 'missing_backend'" in exc_info.value.error_message
    dependency_module.get_llm_client.cache_clear()
    dependency_module.get_query_orchestrator.cache_clear()


def test_answer_generator_dependency_builds_answer_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        dependency_module,
        "get_settings",
        lambda: Settings(openai_api_key="sk-test", answer_generator_backend="missing_backend"),
    )
    dependency_module.get_answer_generator.cache_clear()

    with pytest.raises(AtlasError) as exc_info:
        dependency_module.get_answer_generator()

    assert exc_info.value.error_code == ErrorCode.CONFIGURATION_ERROR
    assert "Unknown answer_generator backend 'missing_backend'" in exc_info.value.error_message
    dependency_module.get_answer_generator.cache_clear()


def test_provider_registry_uses_graph_store_from_build_context() -> None:
    graph_store = PostgresGraphStore()
    context = ProviderBuildContext(
        settings=Settings(openai_api_key=None, max_context_tokens=1234),
        qdrant_factory=lambda: object(),
        embedder_factory=lambda: object(),
        sparse_encoder_factory=lambda: object(),
        reranker_factory=lambda: None,
        graph_store_factory=lambda: graph_store,
    )

    provider = build_provider("graph", context)

    assert provider.store is graph_store
    assert provider.max_context_tokens == 1234


def test_provider_registry_requires_explicit_graph_store_factory() -> None:
    with pytest.raises(TypeError):
        ProviderBuildContext(
            settings=Settings(openai_api_key=None),
            qdrant_factory=lambda: object(),
            embedder_factory=lambda: object(),
            sparse_encoder_factory=lambda: object(),
            reranker_factory=lambda: None,
        )
