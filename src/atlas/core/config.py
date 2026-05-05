from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    qdrant_dense_vector_name: str = "dense"
    qdrant_sparse_vector_name: str = "bm25"

    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    embedding_batch_size: int = 16

    retrieval_mode: str = "dense"
    bm25_enabled: bool = False
    bm25_model: str = "Qdrant/bm25"
    bm25_language: str = "english"
    bm25_k: float = 1.2
    bm25_b: float = 0.75
    bm25_avg_len: float = 256.0
    hybrid_dense_top_k: int = 24
    hybrid_lexical_top_k: int = 24
    hybrid_rrf_k: int = 60
    rrf_top_k: int = 40
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_top_k: int = 30
    reranker_output_k: int = 8

    llm_provider: str = "openai"
    llm_model: str = "gpt-5-nano"
    llm_timeout_seconds: int = 60
    llm_max_output_tokens: int = 2000
    llm_reasoning_effort: str = "low"
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")

    prompt_version: str = "v0.0-rag-answer-2026-05-03"
    default_top_k: int = 8
    max_top_k: int = 12
    max_context_tokens: int = 6000
    chunk_target_tokens: int = 600
    chunk_overlap_tokens: int = 80
    corpus_version: str | None = None
    evidence_builder_version: str = "parent_child_v1"
    critic_version: str = "critic_lite_v1"
    document_roots: str = "samples,corpus"

    cache_enabled: bool = False
    cache_backend: str = "local"
    cache_ttl_seconds: int = 3600


@lru_cache
def get_settings() -> Settings:
    return Settings()


def bm25_sparse_enabled(settings: Settings) -> bool:
    retrieval_mode = settings.retrieval_mode.strip().lower()
    return settings.bm25_enabled or retrieval_mode in {"bm25", "hybrid", "lexical"}
