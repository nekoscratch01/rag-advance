from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from atlas.core.config import Settings
from atlas.db.models import QueryCache, utcnow

CACHE_KEY_SCHEMA = "atlas-query-cache-v2"


def make_cache_key(
    query: str,
    filters: Mapping[str, Any] | None,
    settings: Settings,
    retrieval_mode: str | Mapping[str, Any] | None = None,
    top_k: int | None = None,
    options: Mapping[str, Any] | None = None,
) -> str:
    merged_options = _merged_options(retrieval_mode, options)
    explicit_retrieval_mode = retrieval_mode if isinstance(retrieval_mode, str) else None

    key_payload = {
        "schema": CACHE_KEY_SCHEMA,
        "normalized_query": _normalize_query(query),
        "filters": _canonicalize(filters or {}),
        "corpus": _corpus_cache_config(settings, merged_options),
        "retrieval": _retrieval_cache_config(
            settings,
            merged_options,
            explicit_retrieval_mode=explicit_retrieval_mode,
            top_k=top_k,
        ),
        "reranker": _reranker_cache_config(settings, merged_options, top_k),
        "evidence": _evidence_cache_config(settings, merged_options),
        "critic": _critic_cache_config(settings, merged_options),
        "prompt": _prompt_cache_config(settings, merged_options),
        "llm": _llm_cache_config(settings, merged_options),
    }
    return hashlib.sha256(_stable_json(key_payload).encode("utf-8")).hexdigest()


class QueryCacheStore:
    @staticmethod
    def get(db: Session, key: str) -> dict[str, Any] | None:
        record = db.get(QueryCache, key)
        if record is None or _is_expired(record.expires_at):
            return None

        record.hit_count += 1
        record.updated_at = utcnow()
        db.add(record)
        return {
            "answer": record.answer,
            "confidence": record.confidence,
            "citations": record.citations_json,
            "metadata": record.metadata_json,
        }

    @staticmethod
    def set(
        db: Session,
        key: str,
        payload: Mapping[str, Any],
        metadata: Mapping[str, Any] | None,
        ttl_seconds: int | None,
    ) -> QueryCache:
        now = utcnow()
        record = db.get(QueryCache, key)
        if record is None:
            record = QueryCache(key=key, created_at=now, hit_count=0)

        record.answer = str(payload.get("answer", ""))
        confidence = payload.get("confidence")
        record.confidence = str(confidence) if confidence is not None else None
        record.citations_json = _json_list(payload.get("citations", []))
        record.metadata_json = dict(metadata or {})
        record.updated_at = now
        record.expires_at = _expires_at(now, ttl_seconds)
        db.add(record)
        return record


def _merged_options(
    retrieval_mode: str | Mapping[str, Any] | None,
    options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(retrieval_mode, Mapping):
        merged.update(retrieval_mode)
    if options:
        merged.update(options)
    return merged


def _normalize_query(query: str) -> str:
    return " ".join(query.split())


def _corpus_cache_config(settings: Settings, options: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": _option(
            options,
            "corpus_version",
            "corpus_version_id",
            default=_setting(settings, "corpus_version", None),
        ),
        "collection": _option(
            options,
            "collection",
            "qdrant_collection",
            default=settings.qdrant_collection,
        ),
        "chunk_target_tokens": _option(
            options,
            "chunk_target_tokens",
            default=settings.chunk_target_tokens,
        ),
        "chunk_overlap_tokens": _option(
            options,
            "chunk_overlap_tokens",
            default=settings.chunk_overlap_tokens,
        ),
    }


def _retrieval_cache_config(
    settings: Settings,
    options: Mapping[str, Any],
    *,
    explicit_retrieval_mode: str | None,
    top_k: int | None,
) -> dict[str, Any]:
    return {
        "mode": explicit_retrieval_mode
        or _option(options, "retrieval_mode", default=settings.retrieval_mode),
        "top_k": _effective_top_k(settings, top_k, options),
        "embedding_provider": _option(
            options,
            "embedding_provider",
            default=settings.embedding_provider,
        ),
        "embedding_model": _option(
            options,
            "embedding_model",
            default=settings.embedding_model,
        ),
        "embedding_dim": _option(options, "embedding_dim", default=settings.embedding_dim),
        "bm25": _bm25_cache_config(settings, options),
        "hybrid_dense_top_k": _option(
            options,
            "hybrid_dense_top_k",
            default=settings.hybrid_dense_top_k,
        ),
        "hybrid_lexical_top_k": _option(
            options,
            "hybrid_lexical_top_k",
            default=settings.hybrid_lexical_top_k,
        ),
        "rrf_k": _option(options, "rrf_k", "hybrid_rrf_k", default=settings.hybrid_rrf_k),
    }


def _bm25_cache_config(settings: Settings, options: Mapping[str, Any]) -> dict[str, Any]:
    nested = _mapping_option(options, "bm25", "bm25_config")
    return {
        "enabled": _option(options, "bm25_enabled", default=settings.bm25_enabled),
        "model": _option(options, "bm25_model", default=nested.get("model", settings.bm25_model)),
        "language": _option(
            options,
            "bm25_language",
            default=nested.get("language", settings.bm25_language),
        ),
        "k": _option(options, "bm25_k", default=nested.get("k", settings.bm25_k)),
        "b": _option(options, "bm25_b", default=nested.get("b", settings.bm25_b)),
        "avg_len": _option(
            options,
            "bm25_avg_len",
            default=nested.get("avg_len", settings.bm25_avg_len),
        ),
    }


def _reranker_cache_config(
    settings: Settings,
    options: Mapping[str, Any],
    top_k: int | None,
) -> dict[str, Any]:
    nested = _mapping_option(options, "reranker", "reranker_config")
    effective_top_k = _effective_top_k(settings, top_k, options)
    return {
        "enabled": _option(
            options,
            "reranker_enabled",
            "use_reranker",
            default=nested.get("enabled", _setting(settings, "reranker_enabled", False)),
        ),
        "model": _option(
            options,
            "reranker_model",
            default=nested.get("model", _setting(settings, "reranker_model", None)),
        ),
        "top_k": _option(
            options,
            "reranker_top_k",
            default=nested.get("top_k", _setting(settings, "reranker_top_k", effective_top_k)),
        ),
        "output_k": _option(
            options,
            "reranker_output_k",
            "output_k",
            default=nested.get(
                "output_k",
                _setting(settings, "reranker_output_k", effective_top_k),
            ),
        ),
    }


def _evidence_cache_config(settings: Settings, options: Mapping[str, Any]) -> dict[str, Any]:
    nested = _mapping_option(options, "evidence", "evidence_config")
    return {
        "builder_version": _option(
            options,
            "evidence_builder_version",
            default=nested.get(
                "builder_version",
                _setting(settings, "evidence_builder_version", "v1"),
            ),
        ),
        "max_context_tokens": _option(
            options,
            "max_context_tokens",
            default=nested.get("max_context_tokens", settings.max_context_tokens),
        ),
        "output_k": _option(
            options,
            "evidence_output_k",
            "evidence_top_k",
            default=nested.get("output_k", _setting(settings, "evidence_output_k", None)),
        ),
    }


def _critic_cache_config(settings: Settings, options: Mapping[str, Any]) -> dict[str, Any]:
    nested = _mapping_option(options, "critic", "critic_config")
    return {
        "enabled": _option(
            options,
            "critic_enabled",
            default=nested.get("enabled", _setting(settings, "critic_enabled", True)),
        ),
        "version": _option(
            options,
            "critic_version",
            default=nested.get("version", _setting(settings, "critic_version", "critic_lite_v1")),
        ),
    }


def _prompt_cache_config(settings: Settings, options: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": _option(options, "prompt_version", default=settings.prompt_version),
    }


def _llm_cache_config(settings: Settings, options: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "provider": _option(options, "llm_provider", default=settings.llm_provider),
        "model": _option(
            options,
            "answer_model",
            "llm_model",
            "model_name",
            default=settings.llm_model,
        ),
        "max_output_tokens": _option(
            options,
            "llm_max_output_tokens",
            default=settings.llm_max_output_tokens,
        ),
        "reasoning_effort": _option(
            options,
            "llm_reasoning_effort",
            default=settings.llm_reasoning_effort,
        ),
    }


def _effective_top_k(
    settings: Settings,
    top_k: int | None,
    options: Mapping[str, Any],
) -> int:
    requested = (
        top_k
        if top_k is not None
        else _option(options, "top_k", default=settings.default_top_k)
    )
    return min(int(requested), settings.max_top_k)


def _option(options: Mapping[str, Any], *names: str, default: Any) -> Any:
    for name in names:
        if name in options and options[name] is not None:
            return options[name]
    return default


def _mapping_option(options: Mapping[str, Any], *names: str) -> dict[str, Any]:
    for name in names:
        value = options.get(name)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _setting(settings: Settings, name: str, default: Any) -> Any:
    return getattr(settings, name, default)


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(value)
    return [value]


def _expires_at(now: datetime, ttl_seconds: int | None) -> datetime | None:
    if ttl_seconds is None or ttl_seconds <= 0:
        return None
    return now + timedelta(seconds=ttl_seconds)


def _is_expired(expires_at: datetime | None) -> bool:
    if expires_at is None:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at <= utcnow()


def _stable_json(value: Any) -> str:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_canonicalize(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(_canonicalize(item) for item in value)
    return value
