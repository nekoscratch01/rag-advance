from copy import deepcopy
from typing import Any

from atlas.core.config import Settings
from atlas.core.ids import new_id
from atlas.db.models import GenerationEvent, QueryRun, RetrievalEvent
from atlas.llm.base import GeneratedAnswer
from atlas.retrieval.evidence import Evidence

_TRACE_METADATA_LIMIT = 2048
_TRACE_METADATA_BY_QUERY_ID: dict[str, dict[str, Any]] = {}


def make_query_run(
    *,
    query_id: str,
    trace_id: str,
    user_query: str,
    normalized_query: str,
    answer: str,
    confidence: str,
    citations: list[dict],
    settings: Settings,
    latency_ms: int,
    error_message: str | None = None,
) -> QueryRun:
    return QueryRun(
        query_id=query_id,
        trace_id=trace_id,
        user_query=user_query,
        normalized_query=normalized_query,
        answer=answer,
        confidence=confidence,
        citations_json=citations,
        model_name=settings.llm_model,
        prompt_version=settings.prompt_version,
        latency_ms=latency_ms,
        error_message=error_message,
    )


def make_retrieval_events(*, query_id: str, evidence: list[Evidence]) -> list[RetrievalEvent]:
    return [
        RetrievalEvent(
            event_id=new_id("ret"),
            query_id=query_id,
            chunk_id=item.chunk_id,
            rank=item.rank,
            retrieval_score=item.retrieval_score,
            retriever_type=_retriever_type(item),
        )
        for item in evidence
    ]


def make_generation_event(
    *,
    query_id: str,
    settings: Settings,
    generated: GeneratedAnswer | None,
    latency_ms: int | None,
    status: str,
    error_message: str | None = None,
) -> GenerationEvent:
    return GenerationEvent(
        event_id=new_id("gen"),
        query_id=query_id,
        model_name=settings.llm_model,
        prompt_version=settings.prompt_version,
        input_tokens=generated.usage.input_tokens if generated else None,
        output_tokens=generated.usage.output_tokens if generated else None,
        latency_ms=latency_ms,
        status=status,
        error_message=error_message,
    )


def record_query_trace_metadata(
    *,
    query_id: str,
    cache_hit: bool,
    cache_key: str | None,
    cache_latency_ms: int | None,
    retrieval_latency_ms: int | None,
    generation_latency_ms: int | None,
    cache_status: str,
    retrieval_status: str,
    generation_status: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    if len(_TRACE_METADATA_BY_QUERY_ID) >= _TRACE_METADATA_LIMIT:
        oldest_query_id = next(iter(_TRACE_METADATA_BY_QUERY_ID))
        _TRACE_METADATA_BY_QUERY_ID.pop(oldest_query_id, None)

    _TRACE_METADATA_BY_QUERY_ID[query_id] = {
        "cache": {
            "cache_hit": cache_hit,
            "cache_key": cache_key,
            "cache_latency_ms": cache_latency_ms,
        },
        "stages": [
            *(
                [
                    {
                        "name": "planning",
                        "status": "completed",
                        "latency_ms": metadata.get("plan_latency_ms"),
                    }
                ]
                if isinstance(metadata, dict) and metadata.get("plan_latency_ms") is not None
                else []
            ),
            {
                "name": "cache",
                "status": cache_status,
                "latency_ms": cache_latency_ms,
            },
            {
                "name": "retrieval",
                "status": retrieval_status,
                "latency_ms": retrieval_latency_ms,
            },
            {
                "name": "generation",
                "status": generation_status,
                "latency_ms": generation_latency_ms,
            },
        ],
        "metadata": dict(metadata or {}),
    }


def get_query_trace_metadata(query_id: str) -> dict[str, Any]:
    return deepcopy(_TRACE_METADATA_BY_QUERY_ID.get(query_id, {}))


def _retriever_type(evidence: Evidence) -> str:
    metadata = evidence.metadata or {}
    sources = metadata.get("retrieved_by") or metadata.get("sources")
    if isinstance(sources, str):
        sources = [sources]
    if isinstance(sources, list | tuple | set):
        normalized = {str(source) for source in sources}
        if {"dense", "lexical"} <= normalized or {"dense", "bm25"} <= normalized:
            return "hybrid"
        if "lexical" in normalized or "bm25" in normalized:
            return "bm25"
        if "dense" in normalized:
            return "dense"
    if metadata.get("fusion_score") is not None or metadata.get("best_fusion_score") is not None:
        return "hybrid"
    if metadata.get("lexical_score") is not None or metadata.get("best_lexical_score") is not None:
        return "bm25"
    return "dense"
