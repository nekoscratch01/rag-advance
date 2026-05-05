from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from atlas.api.dependencies import get_query_runtime
from atlas.db.repositories import get_chunks_by_ids, get_query_run
from atlas.db.session import get_db
from atlas.query_runtime.service import QueryRuntime
from atlas.query_runtime.trace_logger import get_query_trace_metadata

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    filters: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


@router.post("")
def run_query(
    request: QueryRequest,
    db: Session = Depends(get_db),
    runtime: QueryRuntime = Depends(get_query_runtime),
) -> dict[str, Any]:
    result = runtime.run(
        db,
        query=request.query,
        top_k=request.top_k,
        filters=request.filters,
        options=request.options,
    )
    return {
        "query_id": result.query_id,
        "trace_id": result.trace_id,
        "answer": result.answer,
        "confidence": result.confidence,
        "citations": result.citations,
    }


@router.get("/{query_id}")
def get_query(query_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    query_run = get_query_run(db, query_id)
    if query_run is None:
        raise HTTPException(status_code=404, detail="Query run not found")

    return _serialize_query_run(query_run)


@router.get("/{query_id}/trace")
def get_query_trace(query_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    query_run = get_query_run(db, query_id)
    if query_run is None:
        raise HTTPException(status_code=404, detail="Query run not found")

    retrieval_events = sorted(query_run.retrieval_events, key=lambda event: event.rank)
    generation_events = sorted(query_run.generation_events, key=lambda event: event.created_at)
    chunk_map = get_chunks_by_ids(db, [event.chunk_id for event in retrieval_events])
    trace_metadata = get_query_trace_metadata(query_run.query_id)
    if not trace_metadata:
        details = query_run.details_json or {}
        trace = details.get("trace") if isinstance(details, dict) else None
        trace_metadata = trace if isinstance(trace, dict) else {}
    generation_latency_ms = _generation_latency_ms(generation_events)
    if generation_latency_ms is None:
        generation_latency_ms = _stage_latency(trace_metadata, "generation")

    return {
        "query": {
            "query_id": query_run.query_id,
            "trace_id": query_run.trace_id,
            "user_query": query_run.user_query,
            "normalized_query": query_run.normalized_query,
            "created_at": query_run.created_at.isoformat(),
        },
        "result": {
            "answer": query_run.answer,
            "confidence": query_run.confidence,
            "citations": query_run.citations_json,
            "details": query_run.details_json,
            "error_message": query_run.error_message,
        },
        "retrieval": {
            "retriever_type": _retriever_type(retrieval_events),
            "event_count": len(retrieval_events),
            "top_k": [
                _serialize_retrieval_event(event, chunk_map.get(event.chunk_id))
                for event in retrieval_events
            ],
        },
        "generation": [
            {
                "model_name": event.model_name,
                "prompt_version": event.prompt_version,
                "input_tokens": event.input_tokens,
                "output_tokens": event.output_tokens,
                "latency_ms": event.latency_ms,
                "status": event.status,
                "error_message": event.error_message,
                "created_at": event.created_at.isoformat(),
            }
                for event in generation_events
            ],
        "latency": {
            "total_latency_ms": query_run.latency_ms,
            "cache_latency_ms": _trace_cache(trace_metadata)["cache_latency_ms"],
            "retrieval_latency_ms": _stage_latency(trace_metadata, "retrieval"),
            "generation_latency_ms": generation_latency_ms,
        },
        "cache": _trace_cache(trace_metadata),
        "stages": _trace_stages(trace_metadata, retrieval_events, generation_events),
        "metadata": _trace_route_metadata(trace_metadata),
        "model": {
            "model_name": query_run.model_name,
            "prompt_version": query_run.prompt_version,
        },
    }


def _serialize_query_run(query_run) -> dict[str, Any]:
    return {
        "query_id": query_run.query_id,
        "trace_id": query_run.trace_id,
        "user_query": query_run.user_query,
        "normalized_query": query_run.normalized_query,
        "answer": query_run.answer,
        "confidence": query_run.confidence,
        "citations": query_run.citations_json,
        "details": query_run.details_json,
        "model_name": query_run.model_name,
        "prompt_version": query_run.prompt_version,
        "latency_ms": query_run.latency_ms,
        "error_message": query_run.error_message,
        "created_at": query_run.created_at.isoformat(),
        "retrieval_events": [
            {
                "chunk_id": event.chunk_id,
                "rank": event.rank,
                "retrieval_score": event.retrieval_score,
                "retriever_type": event.retriever_type,
            }
            for event in query_run.retrieval_events
        ],
        "generation_events": [
            {
                "model_name": event.model_name,
                "prompt_version": event.prompt_version,
                "input_tokens": event.input_tokens,
                "output_tokens": event.output_tokens,
                "latency_ms": event.latency_ms,
                "status": event.status,
                "error_message": event.error_message,
            }
            for event in query_run.generation_events
        ],
    }


def _serialize_retrieval_event(event, chunk) -> dict[str, Any]:
    payload = {
        "rank": event.rank,
        "chunk_id": event.chunk_id,
        "retrieval_score": event.retrieval_score,
        "retriever_type": event.retriever_type,
    }
    if chunk is None:
        payload["chunk_found"] = False
        return payload

    payload.update(
        {
            "chunk_found": True,
            "document_id": chunk.document_id,
            "source_title": chunk.document.title,
            "source_uri": chunk.document.source_uri,
            "section_title": chunk.section_title,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "token_count": chunk.token_count,
            "preview": _preview(chunk.text),
        }
    )
    return payload


def _preview(text: str, limit: int = 240) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _retriever_type(events) -> str:
    types = {event.retriever_type for event in events}
    if not types:
        return "unknown"
    if len(types) == 1:
        return next(iter(types))
    if "hybrid" in types or {"dense", "bm25"} <= types:
        return "hybrid"
    return "mixed"


def _trace_cache(trace_metadata: dict[str, Any]) -> dict[str, Any]:
    cache = trace_metadata.get("cache")
    if isinstance(cache, dict):
        return {
            "cache_hit": bool(cache.get("cache_hit")),
            "cache_key": cache.get("cache_key"),
            "cache_latency_ms": cache.get("cache_latency_ms"),
        }
    return {
        "cache_hit": False,
        "cache_key": None,
        "cache_latency_ms": None,
    }


def _trace_stages(trace_metadata, retrieval_events, generation_events) -> list[dict[str, Any]]:
    stages = trace_metadata.get("stages")
    if isinstance(stages, list) and stages:
        return stages
    return [
        {
            "name": "cache",
            "status": "unknown",
            "latency_ms": None,
        },
        {
            "name": "retrieval",
            "status": "completed" if retrieval_events else "unknown",
            "latency_ms": None,
        },
        {
            "name": "generation",
            "status": _generation_status(generation_events),
            "latency_ms": _generation_latency_ms(generation_events),
        },
    ]


def _trace_route_metadata(trace_metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = trace_metadata.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _stage_latency(trace_metadata: dict[str, Any], stage_name: str) -> int | None:
    stages = trace_metadata.get("stages")
    if not isinstance(stages, list):
        return None
    for stage in stages:
        if isinstance(stage, dict) and stage.get("name") == stage_name:
            latency = stage.get("latency_ms")
            return latency if isinstance(latency, int) else None
    return None


def _generation_latency_ms(generation_events) -> int | None:
    latency_ms = sum(
        event.latency_ms or 0 for event in generation_events if event.status == "completed"
    )
    return latency_ms or _stage_latency_from_events(generation_events)


def _stage_latency_from_events(generation_events) -> int | None:
    latency_ms = sum(event.latency_ms or 0 for event in generation_events)
    return latency_ms or None


def _generation_status(generation_events) -> str:
    if not generation_events:
        return "skipped"
    if any(event.status == "failed" for event in generation_events):
        return "failed"
    if any(event.status == "completed" for event in generation_events):
        return "completed"
    return "unknown"
