from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from atlas.core.errors import AtlasError, ErrorCode


IMPLEMENTED_RUNTIME_PROVIDERS = ("hybrid", "graph", "sql")
RESERVED_INTERNAL_PROVIDER_NAMES = (
    "dense",
    "bm25",
    "sparse",
    "table",
    "section",
    "metric_alias",
)
NON_EXECUTABLE_QUERY_PROVIDERS = ("sql",)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ATLAS_",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "Atlas Kernel"
    api_prefix: str = "/v1"

    database_url: str = "postgresql+psycopg://atlas:atlas@localhost:15432/atlas"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "atlas_chunks_bge_small_zh_v1_5"
    v4_qdrant_collection: str = "atlas_v4_chunks_bge_small_zh_v1_5"
    qdrant_dense_vector_name: str = "dense"
    qdrant_sparse_vector_name: str = "bm25"
    vector_store_backend: str = "qdrant"
    graph_store_backend: str = "postgres_graph"

    embedding_backend: str = "local_bge"
    # Deprecated compatibility setting; backend construction uses embedding_backend.
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    embedding_batch_size: int = 16

    retrieval_mode: str = "hybrid"
    sparse_backend: str = "fastembed_bm25"
    bm25_enabled: bool = True
    bm25_model: str = "Qdrant/bm25"
    bm25_language: str = "english"
    bm25_k: float = 1.2
    bm25_b: float = 0.75
    bm25_avg_len: float = 256.0
    hybrid_dense_top_k: int = 24
    hybrid_lexical_top_k: int = 24
    hybrid_rrf_k: int = 60
    rrf_top_k: int = 40
    reranker_backend: str = "cross_encoder"
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_top_k: int = 30
    reranker_output_k: int = 8

    llm_client_backend: str = "openai"
    answer_generator_backend: str = "openai"
    # Deprecated compatibility setting; backend construction uses llm_client_backend
    # and answer_generator_backend.
    llm_provider: str = "openai"
    llm_model: str = "gpt-5-nano"
    llm_timeout_seconds: int = 60
    llm_max_output_tokens: int = 2000
    llm_reasoning_effort: str = "low"
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    query_planner_model: str = "gpt-5-nano"
    query_planner_version: str = "query_planner_v1"
    finance_metric_ontology_path: str = "configs/finance_metric_ontology.yaml"
    query_planner_max_units: int = 6
    query_planner_known_providers: str = "hybrid,sql,graph"
    query_runtime_executable_providers: str = "hybrid,graph"
    sql_provider_enabled: bool = False
    structured_sql_duckdb_dir: str = "artifacts/structured_sql_duckdb"
    structured_sql_timeout_ms: int = 1000
    structured_sql_max_rows: int = 100
    structured_sql_max_result_bytes: int = 65536
    structured_sql_memory_limit: str = "128MB"
    structured_sql_compiler_mode: str = "heuristic"
    structured_sql_min_table_score: float = 0.15
    structured_sql_min_score_margin: float = 0.10
    structured_sql_max_candidate_tables: int = 1
    # Deprecated: retained only so older .env files do not fail settings parsing.
    query_planner_enabled_providers: str = "hybrid"
    query_planner_retry_count: int = 2

    prompt_version: str = "v1-evidence-answer-2026-05-06"
    default_top_k: int = 8
    max_top_k: int = 12
    max_context_tokens: int = 6000
    chunk_target_tokens: int = 600
    chunk_overlap_tokens: int = 80
    corpus_version: str | None = None
    evidence_builder_version: str = "parent_child_v1"
    critic_version: str = "critic_lite_v1"
    document_roots: str = "samples,corpus"
    v4_structured_artifact_output_dir: str = "artifacts/v4_structured_artifacts"

    cache_enabled: bool = False
    cache_backend: str = "local"
    cache_ttl_seconds: int = 3600
    trace_include_raw_llm_io_default: bool = False

    @field_validator("structured_sql_compiler_mode")
    @classmethod
    def _validate_structured_sql_compiler_mode(cls, value: str) -> str:
        mode = str(value or "").strip().lower()
        if mode not in {"heuristic", "llm"}:
            raise ValueError("structured_sql_compiler_mode must be 'heuristic' or 'llm'")
        return mode


@lru_cache
def get_settings() -> Settings:
    return Settings()


def bm25_sparse_enabled(settings: Settings) -> bool:
    retrieval_mode = settings.retrieval_mode.strip().lower()
    return settings.bm25_enabled or retrieval_mode in {"bm25", "hybrid", "lexical"}


def enabled_query_providers(settings: Settings) -> tuple[str, ...]:
    """Deprecated compatibility alias for executable runtime providers."""
    return executable_query_providers(settings)


def known_query_providers(settings: Settings) -> tuple[str, ...]:
    reserved = set(RESERVED_INTERNAL_PROVIDER_NAMES)
    providers = tuple(
        provider
        for provider in _provider_list(settings.query_planner_known_providers)
        if provider not in reserved
    )
    return providers or ("hybrid", "sql", "graph")


def executable_query_providers(settings: Settings) -> tuple[str, ...]:
    requested = _provider_list(settings.query_runtime_executable_providers)
    known_providers = known_query_providers(settings)
    known = set(known_providers)
    registered = _registered_runtime_providers()
    non_executable = set(non_executable_query_providers(settings))
    reserved = set(RESERVED_INTERNAL_PROVIDER_NAMES)
    reserved_requested = tuple(provider for provider in requested if provider in reserved)
    unknown_requested = tuple(
        provider
        for provider in requested
        if provider not in known and provider not in registered and provider not in non_executable
    )
    unregistered_requested = tuple(
        provider
        for provider in requested
        if provider in known
        and provider not in registered
        and provider not in non_executable
    )
    if reserved_requested or unknown_requested or unregistered_requested:
        problems = []
        if reserved_requested:
            problems.append(f"reserved lanes: {', '.join(reserved_requested)}")
        if unknown_requested:
            problems.append(f"unknown providers: {', '.join(unknown_requested)}")
        if unregistered_requested:
            problems.append(f"not registered: {', '.join(unregistered_requested)}")
        raise AtlasError(
            ErrorCode.CONFIGURATION_ERROR,
            "Invalid executable query provider configuration: " + "; ".join(problems) + ".",
            status_code=500,
            details={
                "requested": list(requested),
                "known_providers": list(known_providers),
                "registered_providers": sorted(registered),
                "non_executable_providers": list(non_executable),
                "reserved_providers": list(reserved_requested),
                "unknown_providers": list(unknown_requested),
                "unregistered_providers": list(unregistered_requested),
            },
        )
    return tuple(
        provider
        for provider in requested
        if provider in registered and provider in known and provider not in non_executable
    )


def non_executable_query_providers(settings: Settings) -> tuple[str, ...]:
    providers = []
    requested = set(_provider_list(settings.query_runtime_executable_providers))
    if not settings.sql_provider_enabled or "sql" not in requested:
        providers.append("sql")
    return tuple(providers)


def sql_provider_runtime_enabled(settings: Settings) -> bool:
    requested = set(_provider_list(settings.query_runtime_executable_providers))
    return settings.sql_provider_enabled and "sql" in requested


def _registered_runtime_providers() -> set[str]:
    try:
        from atlas.retrieval.providers.registry import provider_registry
    except ImportError:
        return set(IMPLEMENTED_RUNTIME_PROVIDERS)
    return set(provider_registry.names)


def _provider_list(value: str) -> tuple[str, ...]:
    return tuple(
        provider.strip().lower()
        for provider in value.split(",")
        if provider.strip()
    )


def legacy_enabled_query_providers(settings: Settings) -> tuple[str, ...]:
    """Deprecated parser for older ATLAS_QUERY_PLANNER_ENABLED_PROVIDERS env files."""
    providers = tuple(
        provider.strip().lower()
        for provider in settings.query_planner_enabled_providers.split(",")
        if provider.strip()
    )
    return providers or ("hybrid",)
