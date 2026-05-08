from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from atlas.api.dependencies import (
    get_provider_router,
    get_query_orchestrator,
    get_query_runtime,
    settings_dependency,
)
from atlas.core.config import Settings, executable_query_providers
from atlas.db.repositories import get_chunks_by_ids, get_query_run, get_v1_trace_family
from atlas.db.session import get_db
from atlas.query_orchestrator.schema import serialize_query_plan
from atlas.query_orchestrator.service import QueryOrchestrator
from atlas.query_runtime.service import QueryRuntime, _retrieve_with_router
from atlas.query_runtime.trace_logger import get_query_trace_metadata
from atlas.retrieval.models.retrieval_task import serialize_retrieval_task, tasks_from_plan
from atlas.retrieval.router import ProviderRouter

router = APIRouter(prefix="/query", tags=["query"])
retrieve_router = APIRouter(prefix="/retrieve", tags=["retrieval"])
_REDACTED = "[redacted]"
_LLM_CALL_RAW_FIELDS = (
    "request",
    "response",
    "instructions_text",
    "input_text",
    "raw_output_text",
    "parsed_answer_text",
)
_LLM_CALL_EVIDENCE_RAW_FIELDS = ("text_snapshot",)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    filters: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


@router.post("/plan")
def plan_query(
    request: QueryRequest,
    orchestrator: QueryOrchestrator = Depends(get_query_orchestrator),
    provider_router: ProviderRouter = Depends(get_provider_router),
) -> dict[str, Any]:
    use_llm = not _truthy(request.options.get("query_plan_fallback_only"))
    plan = orchestrator.plan(request.query, use_llm=use_llm)
    tasks = tasks_from_plan(
        plan,
        executable_providers=provider_router.executable_providers,
    )
    return {
        "query_plan": serialize_query_plan(plan),
        "retrieval_tasks": [serialize_retrieval_task(task) for task in tasks],
    }


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
        "details": result.details if request.options.get("return_trace") else {},
    }


@retrieve_router.post("")
def retrieve_only(
    request: QueryRequest,
    db: Session = Depends(get_db),
    runtime: QueryRuntime = Depends(get_query_runtime),
    orchestrator: QueryOrchestrator = Depends(get_query_orchestrator),
) -> dict[str, Any]:
    use_llm = not _truthy(request.options.get("query_plan_fallback_only"))
    plan = orchestrator.plan(request.query, use_llm=use_llm)
    executable_providers = _runtime_executable_providers(runtime)
    tasks = tasks_from_plan(plan, executable_providers=executable_providers)
    top_k = request.top_k or runtime.settings.default_top_k
    runtime_options = {
        **request.options,
        "query_plan": serialize_query_plan(plan),
        "retrieval_tasks": [serialize_retrieval_task(task) for task in tasks],
    }
    provider_results = []
    if hasattr(runtime, "provider_router"):
        router_result = _retrieve_with_router(
            runtime.provider_router,
            db,
            query=request.query,
            top_k=min(top_k, runtime.settings.max_top_k),
            filters=request.filters,
            options=runtime_options,
            query_plan=plan,
            retrieval_tasks=tasks,
        )
        evidence = list(router_result.evidence)
        provider_results = [
            {
                "provider": result.provider,
                "task_id": result.task_id,
                "unit_id": result.unit_id,
                "status": result.status,
                "reason": result.reason,
                "candidate_count": len(result.candidates),
                "evidence_count": len(result.evidence),
            }
            for result in router_result.provider_results
        ]
    else:
        from atlas.query_runtime.service import _retrieve_evidence

        evidence = _retrieve_evidence(
            runtime.retriever,
            db,
            query=request.query,
            top_k=min(top_k, runtime.settings.max_top_k),
            filters=request.filters,
            options=runtime_options,
            query_plan=plan,
            retrieval_tasks=tasks,
        )
    return {
        "query_plan": serialize_query_plan(plan),
        "retrieval_tasks": [serialize_retrieval_task(task) for task in tasks],
        "provider_results": provider_results,
        "evidence": [_serialize_evidence(item) for item in evidence],
    }


@router.get("/{query_id}")
def get_query(query_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    query_run = get_query_run(db, query_id)
    if query_run is None:
        raise HTTPException(status_code=404, detail="Query run not found")

    return _serialize_query_run(query_run)


@router.get("/{query_id}/trace")
def get_query_trace(
    query_id: str,
    include_raw_llm_io: bool = False,
    x_atlas_include_raw_llm_io: str | None = Header(default=None),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dependency),
) -> dict[str, Any]:
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
    raw_llm_io_included = _include_raw_llm_io(
        include_raw_llm_io,
        x_atlas_include_raw_llm_io,
        settings,
    )
    v1_trace = get_v1_trace_family(db, query_run.query_id)

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
        "v1_trace": (
            v1_trace if raw_llm_io_included else _redact_v1_trace_llm_io(v1_trace)
        ),
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


def _include_raw_llm_io(
    include_raw_llm_io: bool,
    header_value: str | None,
    settings: Settings,
) -> bool:
    return (
        include_raw_llm_io
        or _truthy(header_value)
        or bool(settings.trace_include_raw_llm_io_default)
    )


def _redact_v1_trace_llm_io(v1_trace: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(v1_trace)
    redacted["llm_calls"] = [
        _redact_fields(item, _LLM_CALL_RAW_FIELDS)
        for item in _list_of_mappings(v1_trace.get("llm_calls"))
    ]
    redacted["llm_call_evidence"] = [
        _redact_fields(item, _LLM_CALL_EVIDENCE_RAW_FIELDS)
        for item in _list_of_mappings(v1_trace.get("llm_call_evidence"))
    ]
    return redacted


def _redact_fields(item: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    redacted = dict(item)
    for field in fields:
        if field in redacted and redacted[field] is not None:
            redacted[field] = _REDACTED
    return redacted


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _runtime_executable_providers(runtime: QueryRuntime) -> tuple[str, ...]:
    provider_router = getattr(runtime, "provider_router", None)
    executable = getattr(provider_router, "executable_providers", None)
    if executable is not None:
        return tuple(executable)
    settings = getattr(runtime, "settings", None)
    if isinstance(settings, Settings):
        return executable_query_providers(settings)
    return ("hybrid",)


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


def _serialize_evidence(item) -> dict[str, Any]:
    return {
        "evidence_id": item.evidence_id,
        "document_id": item.document_id,
        "chunk_id": item.chunk_id,
        "text": item.text,
        "source_title": item.source_title,
        "source_uri": item.source_uri,
        "section_title": item.section_title,
        "page_start": item.page_start,
        "page_end": item.page_end,
        "retrieval_score": item.retrieval_score,
        "rank": item.rank,
        "token_count": item.token_count,
        "metadata": item.metadata,
        "parent_id": item.parent_id,
        "child_ids": list(item.child_ids),
        "retrieved_by": list(item.retrieved_by),
    }


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


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled", "enable"}


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
