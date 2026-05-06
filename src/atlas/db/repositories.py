from collections.abc import Iterable
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from atlas.core.ids import new_id
from atlas.db.models import (
    AnswerRecord,
    CandidateRecord,
    CitationRecord,
    CitationVerificationRecord,
    Chunk,
    Document,
    EvidenceBlockRecord,
    EvidenceEvaluationRecord,
    EvidencePackRecord,
    GenerationEvent,
    ParentBlock,
    QueryPlanRecord,
    QueryRun,
    RetrievalEvent,
    RetrievalResultRecord,
    RetrievalTaskRecord,
)


def get_document_by_hash(db: Session, content_hash: str) -> Document | None:
    return db.scalar(select(Document).where(Document.content_hash == content_hash))


def get_chunks_for_document(db: Session, document_id: str) -> list[Chunk]:
    return list(
        db.scalars(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index.asc())
        )
    )


def get_chunks_by_ids(db: Session, chunk_ids: Iterable[str]) -> dict[str, Chunk]:
    ids = list(chunk_ids)
    if not ids:
        return {}
    chunks = db.scalars(select(Chunk).where(Chunk.chunk_id.in_(ids))).all()
    return {chunk.chunk_id: chunk for chunk in chunks}


def get_parent_blocks_by_ids(db: Session, parent_ids: Iterable[str]) -> dict[str, ParentBlock]:
    ids = [parent_id for parent_id in parent_ids if parent_id]
    if not ids:
        return {}
    parent_blocks = db.scalars(select(ParentBlock).where(ParentBlock.parent_id.in_(ids))).all()
    return {parent.parent_id: parent for parent in parent_blocks}


def get_parent_blocks_for_chunks(db: Session, chunks: Iterable[Chunk]) -> dict[str, ParentBlock]:
    chunk_list = list(chunks)
    parent_blocks = get_parent_blocks_by_ids(
        db,
        [chunk.parent_id for chunk in chunk_list if chunk.parent_id],
    )
    return {
        chunk.chunk_id: parent_blocks[chunk.parent_id]
        for chunk in chunk_list
        if chunk.parent_id and chunk.parent_id in parent_blocks
    }


def get_query_run(db: Session, query_id: str) -> QueryRun | None:
    return db.get(QueryRun, query_id)


def add_query_trace(
    db: Session,
    query_run: QueryRun,
    retrieval_events: list[RetrievalEvent],
    generation_event: GenerationEvent | None,
) -> None:
    db.add(query_run)
    db.flush()
    for event in retrieval_events:
        db.add(event)
    if generation_event is not None:
        db.add(generation_event)
    record_v1_trace_family(db, query_run, retrieval_events, generation_event)


def record_v1_trace_family(
    db: Session,
    query_run: QueryRun,
    retrieval_events: list[RetrievalEvent],
    generation_event: GenerationEvent | None,
) -> None:
    details = query_run.details_json if isinstance(query_run.details_json, dict) else {}
    trace = details.get("trace") if isinstance(details.get("trace"), dict) else {}
    trace_metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    query_plan = _first_mapping(details.get("query_plan"), trace_metadata.get("query_plan"))
    retrieval_tasks = _list_mapping(details.get("retrieval_tasks") or trace_metadata.get("retrieval_tasks"))
    retrieval_trace = _mapping(details.get("retrieval_trace"))
    provider_router_trace = _mapping(details.get("provider_router_trace"))
    provider_results = _list_mapping(details.get("provider_results"))
    critic = _mapping(details.get("critic"))
    evidence_pack = _mapping(details.get("evidence_pack"))

    if query_plan:
        db.add(
            QueryPlanRecord(
                record_id=new_id("qpr"),
                query_id=query_run.query_id,
                plan_id=_optional_str(query_plan.get("plan_id")),
                planner=_optional_str(query_plan.get("planner")),
                payload_json=query_plan,
            )
        )
    for task in retrieval_tasks:
        db.add(
            RetrievalTaskRecord(
                record_id=new_id("rtr"),
                query_id=query_run.query_id,
                task_id=_optional_str(task.get("task_id")),
                unit_id=_optional_str(task.get("unit_id")),
                payload_json=task,
            )
        )

    db.add(
        RetrievalResultRecord(
            record_id=new_id("rr"),
            query_id=query_run.query_id,
            status=_trace_stage_status(trace, "retrieval")
            or _optional_str(trace_metadata.get("retrieval_status")),
            payload_json={
                "retrieval_trace": retrieval_trace,
                "provider_router_trace": provider_router_trace,
                "provider_results": provider_results,
                "event_count": len(retrieval_events),
                "events": [_retrieval_event_payload(event) for event in retrieval_events],
            },
        )
    )
    seen_pack_ids: set[str] = set()
    seen_candidate_keys: set[tuple[Any, ...]] = set()
    for item in _provider_candidate_payloads(provider_results):
        _add_candidate_record(db, query_run.query_id, item, seen_candidate_keys)
    for item in _list_mapping(retrieval_trace.get("top_k")):
        _add_candidate_record(db, query_run.query_id, item, seen_candidate_keys)
        db.add(
            EvidenceBlockRecord(
                record_id=new_id("eb"),
                query_id=query_run.query_id,
                evidence_id=_optional_str(item.get("evidence_id")),
                chunk_id=_optional_str(item.get("chunk_id")),
                rank=_optional_int(item.get("rank")),
                payload_json=item,
            )
        )
        pack = _mapping(item.get("evidence_pack"))
        pack_id = _optional_str(pack.get("pack_id")) if pack else None
        if pack and (pack_id or "") not in seen_pack_ids:
            seen_pack_ids.add(pack_id or "")
            db.add(
                EvidencePackRecord(
                    record_id=new_id("eprec"),
                    query_id=query_run.query_id,
                    pack_id=pack_id,
                    payload_json=pack,
                )
            )
    if evidence_pack:
        pack_id = _optional_str(evidence_pack.get("pack_id"))
        if (pack_id or "") in seen_pack_ids:
            evidence_pack = {}
    if evidence_pack:
        db.add(
            EvidencePackRecord(
                record_id=new_id("eprec"),
                query_id=query_run.query_id,
                pack_id=_optional_str(evidence_pack.get("pack_id")),
                payload_json=evidence_pack,
            )
        )

    evidence_evaluation = _mapping(critic.get("evidence_evaluation"))
    if evidence_evaluation:
        db.add(
            EvidenceEvaluationRecord(
                record_id=new_id("ee"),
                query_id=query_run.query_id,
                status=_optional_str(evidence_evaluation.get("status")),
                payload_json=evidence_evaluation,
            )
        )
    citation_verification = _mapping(critic.get("citation_verification"))
    if citation_verification:
        db.add(
            CitationVerificationRecord(
                record_id=new_id("cvrec"),
                query_id=query_run.query_id,
                status=_optional_str(citation_verification.get("status")),
                payload_json=citation_verification,
            )
        )

    db.add(
        AnswerRecord(
            record_id=new_id("ans"),
            query_id=query_run.query_id,
            confidence=query_run.confidence,
            payload_json={
                "answer": query_run.answer,
                "confidence": query_run.confidence,
                "model_name": query_run.model_name,
                "prompt_version": query_run.prompt_version,
                "generation_event": _generation_event_payload(generation_event),
            },
        )
    )
    for citation in _list_mapping(query_run.citations_json):
        db.add(
            CitationRecord(
                record_id=new_id("citrec"),
                query_id=query_run.query_id,
                citation_id=_optional_str(citation.get("citation_id")),
                evidence_id=_optional_str(citation.get("evidence_id") or citation.get("citation_id")),
                payload_json=citation,
            )
        )


def _add_candidate_record(
    db: Session,
    query_id: str,
    item: dict[str, Any],
    seen_candidate_keys: set[tuple[Any, ...]],
) -> None:
    key = (
        item.get("candidate_id"),
        item.get("chunk_id"),
        item.get("rank"),
        item.get("retrieval_task_id"),
        item.get("retrieval_unit_id"),
    )
    if key in seen_candidate_keys:
        return
    seen_candidate_keys.add(key)
    db.add(
        CandidateRecord(
            record_id=new_id("cand"),
            query_id=query_id,
            chunk_id=_optional_str(item.get("chunk_id")),
            rank=_optional_int(
                item.get("rank")
                or item.get("final_rank")
                or item.get("rerank_rank")
                or item.get("fusion_rank")
            ),
            payload_json=item,
        )
    )


def _provider_candidate_payloads(provider_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for result in provider_results:
        provider = result.get("provider")
        task_id = result.get("task_id")
        unit_id = result.get("unit_id")
        for item in _list_mapping(result.get("candidates")):
            payload = {
                "provider": provider,
                "retrieval_task_id": task_id,
                "retrieval_unit_id": unit_id,
                **item,
            }
            candidates.append(payload)
    return candidates


def get_v1_trace_family(db: Session, query_id: str) -> dict[str, Any]:
    return {
        "query_plans": _records(db, QueryPlanRecord, query_id),
        "retrieval_tasks": _records(db, RetrievalTaskRecord, query_id),
        "retrieval_results": _records(db, RetrievalResultRecord, query_id),
        "candidates": _records(db, CandidateRecord, query_id),
        "evidence_blocks": _records(db, EvidenceBlockRecord, query_id),
        "evidence_packs": _records(db, EvidencePackRecord, query_id),
        "evidence_evaluations": _records(db, EvidenceEvaluationRecord, query_id),
        "answers": _records(db, AnswerRecord, query_id),
        "citations": _records(db, CitationRecord, query_id),
        "citation_verifications": _records(db, CitationVerificationRecord, query_id),
    }


def _records(db: Session, model, query_id: str) -> list[dict[str, Any]]:
    rows = db.scalars(select(model).where(model.query_id == query_id)).all()
    return [
        {
            "record_id": row.record_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "payload": row.payload_json,
        }
        for row in rows
    ]


def _retrieval_event_payload(event: RetrievalEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "chunk_id": event.chunk_id,
        "rank": event.rank,
        "retrieval_score": event.retrieval_score,
        "retriever_type": event.retriever_type,
    }


def _generation_event_payload(event: GenerationEvent | None) -> dict[str, Any] | None:
    if event is None:
        return None
    return {
        "event_id": event.event_id,
        "model_name": event.model_name,
        "prompt_version": event.prompt_version,
        "input_tokens": event.input_tokens,
        "output_tokens": event.output_tokens,
        "latency_ms": event.latency_ms,
        "status": event.status,
        "error_message": event.error_message,
    }


def _trace_stage_status(trace: dict[str, Any], stage_name: str) -> str | None:
    stages = trace.get("stages")
    if not isinstance(stages, list):
        return None
    for stage in stages:
        if isinstance(stage, dict) and stage.get("name") == stage_name:
            return _optional_str(stage.get("status"))
    return None


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        mapped = _mapping(value)
        if mapped:
            return mapped
    return {}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list_mapping(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
