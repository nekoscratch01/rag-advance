from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
import hashlib
import json
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
    LLMCallEvidenceRecord,
    LLMCallRecord,
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
    *,
    observability: dict[str, Any] | None = None,
) -> None:
    db.add(query_run)
    for event in retrieval_events:
        db.add(event)
    if generation_event is not None:
        db.add(generation_event)
    record_v1_trace_family(
        db,
        query_run,
        retrieval_events,
        generation_event,
        observability=observability,
    )


def record_v1_trace_family(
    db: Session,
    query_run: QueryRun,
    retrieval_events: list[RetrievalEvent],
    generation_event: GenerationEvent | None,
    *,
    observability: dict[str, Any] | None = None,
) -> None:
    details = _json_mapping(query_run.details_json)
    planner_observability = _planner_observability(observability)
    planner_call_id = None
    if planner_observability:
        planner_call_id = _add_planner_llm_observability_records(
            db,
            query_run=query_run,
            observability=planner_observability,
        )
        details = _replace_planner_observability_with_pointer(
            details,
            planner_observability,
            planner_call_id=planner_call_id,
        )
    answer_observability = _answer_observability(details, observability)
    if answer_observability:
        details = _replace_llm_io_with_pointer(details, answer_observability)
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
        query_plan = _query_plan_payload_with_planner_pointer(
            query_plan,
            planner_call_id=planner_call_id,
        )
        details = _json_mapping(details)
        details["query_plan"] = query_plan
        db.add(
            QueryPlanRecord(
                record_id=new_id("qpr"),
                query_id=query_run.query_id,
                planner_call_id=planner_call_id,
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
    evidence_block_record_ids_by_evidence_id: dict[str, str] = {}
    evidence_block_record_ids_by_chunk_id: dict[str, str] = {}
    for item in _provider_candidate_payloads(provider_results):
        _add_candidate_record(db, query_run.query_id, item, seen_candidate_keys)
    for item in _list_mapping(retrieval_trace.get("top_k")):
        _add_candidate_record(db, query_run.query_id, item, seen_candidate_keys)
        evidence_block_record_id = new_id("eb")
        db.add(
            EvidenceBlockRecord(
                record_id=evidence_block_record_id,
                query_id=query_run.query_id,
                evidence_id=_optional_str(item.get("evidence_id")),
                chunk_id=_optional_str(item.get("chunk_id")),
                rank=_optional_int(item.get("rank")),
                payload_json=item,
            )
        )
        evidence_id = _optional_str(item.get("evidence_id"))
        chunk_id = _optional_str(item.get("chunk_id"))
        if evidence_id:
            evidence_block_record_ids_by_evidence_id.setdefault(
                evidence_id,
                evidence_block_record_id,
            )
        if chunk_id:
            evidence_block_record_ids_by_chunk_id.setdefault(
                chunk_id,
                evidence_block_record_id,
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

    answer_call_id = None
    if answer_observability:
        answer_call_id = _add_answer_llm_observability_records(
            db,
            query_run=query_run,
            observability=answer_observability,
            evidence_block_record_ids_by_evidence_id=evidence_block_record_ids_by_evidence_id,
            evidence_block_record_ids_by_chunk_id=evidence_block_record_ids_by_chunk_id,
        )

    answer_payload = {
        "answer": query_run.answer,
        "confidence": query_run.confidence,
        "model": query_run.model_name,
        "prompt_version": query_run.prompt_version,
        "generation_event": _generation_event_payload(generation_event),
        "answer_llm_call_id": answer_call_id
        or _optional_str(_mapping(details.get("llm_io")).get("answer_llm_call_id")),
    }
    db.add(
        AnswerRecord(
            record_id=new_id("ans"),
            query_id=query_run.query_id,
            answer_call_id=_optional_str(answer_payload.get("answer_llm_call_id")),
            confidence=query_run.confidence,
            payload_json=answer_payload,
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
    query_run.details_json = details


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


def _planner_observability(observability: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(observability, dict):
        return {}
    calls = _list_mapping(observability.get("planner_llm_calls"))
    if not calls:
        call = _mapping(observability.get("planner_llm_call"))
        if call:
            calls = [call]
    if not calls:
        return {}
    for index, call in enumerate(calls, start=1):
        if not _optional_str(call.get("call_id")):
            call["call_id"] = new_id("llmc")
        call.setdefault("stage", "planner")
        call.setdefault("attempt_index", index)
        call.setdefault("sequence_index", index)
    return {
        "planner_llm_calls": calls,
        "planner_llm_call": dict(calls[-1]),
    }


def _add_planner_llm_observability_records(
    db: Session,
    *,
    query_run: QueryRun,
    observability: dict[str, Any],
) -> str | None:
    call_ids: list[str] = []
    for index, call in enumerate(
        _list_mapping(observability.get("planner_llm_calls")),
        start=1,
    ):
        call.setdefault("stage", "planner")
        call.setdefault("attempt_index", index)
        call.setdefault("sequence_index", index)
        call_ids.append(
            _add_llm_call_record(
                db,
                query_run=query_run,
                call=call,
                default_stage="planner",
            )
        )
    return call_ids[-1] if call_ids else None


def _replace_planner_observability_with_pointer(
    details: dict[str, Any],
    observability: dict[str, Any],
    *,
    planner_call_id: str | None,
) -> dict[str, Any]:
    payload = _json_mapping(details)
    pointer = _planner_pointer_payload(observability, planner_call_id=planner_call_id)
    if not pointer:
        return payload
    payload["planner_llm"] = pointer
    query_plan = _mapping(payload.get("query_plan"))
    if query_plan:
        payload["query_plan"] = _query_plan_payload_with_planner_pointer(
            query_plan,
            planner_call_id=planner_call_id,
            pointer=pointer,
        )
    return payload


def _query_plan_payload_with_planner_pointer(
    query_plan: dict[str, Any],
    *,
    planner_call_id: str | None,
    pointer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _json_mapping(query_plan)
    metadata = _mapping(payload.get("metadata"))
    for key in (
        "planner_llm_call",
        "planner_llm_calls",
        "planner_request",
        "planner_response",
        "raw_planner_prompt",
        "raw_planner_response",
        "request",
        "response",
        "raw_output",
        "raw_output_text",
        "instructions",
        "input",
    ):
        metadata.pop(key, None)
    if pointer is None:
        pointer = {}
        if planner_call_id:
            pointer["planner_llm_call_id"] = planner_call_id
    metadata.update(pointer)
    payload["metadata"] = metadata
    return payload


def _planner_pointer_payload(
    observability: dict[str, Any],
    *,
    planner_call_id: str | None,
) -> dict[str, Any]:
    call = _mapping(observability.get("planner_llm_call"))
    pointer: dict[str, Any] = {}
    call_id = planner_call_id or _optional_str(call.get("call_id"))
    if call_id:
        pointer["planner_llm_call_id"] = call_id
    status = _optional_str(call.get("status"))
    if status:
        pointer["planner_llm_status"] = status
    validation_status = _optional_str(call.get("validation_status"))
    if validation_status:
        pointer["planner_validation_status"] = validation_status
    error_message = _optional_str(call.get("error_message"))
    if status in {"failed", "invalid"} and error_message:
        pointer["error_message"] = error_message[:500]
    return pointer


def _answer_observability(
    details: dict[str, Any],
    observability: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(observability, dict):
        call = _mapping(observability.get("answer_llm_call"))
        if call:
            if not _optional_str(call.get("call_id")):
                call["call_id"] = new_id("llmc")
            return {
                "answer_llm_call": call,
                "answer_prompt_evidence": _list_mapping(
                    observability.get("answer_prompt_evidence")
                ),
            }

    llm_io = _mapping(details.get("llm_io"))
    if not llm_io or ("request" not in llm_io and "response" not in llm_io):
        return {}

    response = llm_io.get("response")
    response_json = _json_mapping(response)
    usage = _mapping(response_json.get("usage"))
    call_id = _optional_str(llm_io.get("answer_llm_call_id")) or new_id("llmc")
    return {
        "answer_llm_call": {
            "call_id": call_id,
            "stage": "answer",
            "attempt_index": 1,
            "sequence_index": 100,
            "status": _optional_str(llm_io.get("status")) or "completed",
            "error_message": _optional_str(llm_io.get("error_message")),
            "request": _mapping(llm_io.get("request")),
            "request_metadata": _mapping(llm_io.get("request_metadata")),
            "response": response if isinstance(response, dict) else None,
            "usage": usage,
            "raw_output": response_json.get("raw_output"),
            "parsed_answer": response_json.get("parsed_answer"),
            "parsed_confidence": response_json.get("parsed_confidence"),
        },
        "answer_prompt_evidence": [],
    }


def _replace_llm_io_with_pointer(
    details: dict[str, Any],
    observability: dict[str, Any],
) -> dict[str, Any]:
    payload = _json_mapping(details)
    call = _mapping(observability.get("answer_llm_call"))
    status = _optional_str(call.get("status")) or "completed"
    llm_io: dict[str, Any] = {"status": status}
    call_id = _optional_str(call.get("call_id"))
    if call_id:
        llm_io["answer_llm_call_id"] = call_id
    if status == "failed":
        error_message = _optional_str(call.get("error_message"))
        if error_message:
            llm_io["error_message"] = error_message
    elif status == "skipped":
        reason = _optional_str(call.get("reason"))
        if reason:
            llm_io["reason"] = reason
    payload["llm_io"] = llm_io
    return payload


def _add_llm_call_record(
    db: Session,
    *,
    query_run: QueryRun,
    call: dict[str, Any],
    default_stage: str,
) -> str:
    call_id = _optional_str(call.get("call_id")) or new_id("llmc")
    request_json = _json_mapping(call.get("request"))
    response_json = _json_mapping(call.get("response"))
    usage_json = _mapping(call.get("usage")) or _mapping(response_json.get("usage"))
    request_metadata = _mapping(call.get("request_metadata"))
    raw_output = _optional_str(call.get("raw_output") or response_json.get("raw_output"))
    parsed_answer = _optional_str(
        call.get("parsed_answer") or response_json.get("parsed_answer")
    )
    parsed_confidence = _optional_str(
        call.get("parsed_confidence") or response_json.get("parsed_confidence")
    )
    parsed_plan_id = _optional_str(
        call.get("parsed_plan_id") or response_json.get("parsed_plan_id")
    )
    validation_status = _optional_str(
        call.get("validation_status") or response_json.get("validation_status")
    )
    instructions_text = _text_or_json(request_json.get("instructions"))
    input_text = _text_or_json(request_json.get("input"))
    raw_payload_hash = _optional_str(call.get("raw_payload_hash")) or _hash_payload(
        {
            "request": request_json,
            "response": response_json,
            "instructions_text": instructions_text,
            "input_text": input_text,
            "raw_output_text": raw_output,
            "parsed_answer_text": parsed_answer,
            "parsed_plan_id": parsed_plan_id,
            "validation_status": validation_status,
        }
    )
    metadata_json = _mapping(call.get("metadata"))
    if request_metadata:
        metadata_json["request_metadata"] = request_metadata

    db.add(
        LLMCallRecord(
            call_id=call_id,
            query_id=query_run.query_id,
            stage=_optional_str(call.get("stage")) or default_stage,
            attempt_index=_optional_int(call.get("attempt_index")),
            sequence_index=_optional_int(call.get("sequence_index")),
            model_name=_optional_str(call.get("model_name"))
            or _optional_str(request_json.get("model"))
            or query_run.model_name
            or "unknown",
            prompt_version=_optional_str(call.get("prompt_version"))
            or _optional_str(request_metadata.get("prompt_version"))
            or (query_run.prompt_version if default_stage == "answer" else None),
            planner_version=_optional_str(call.get("planner_version")),
            status=_optional_str(call.get("status")) or "completed",
            error_message=_optional_str(call.get("error_message")),
            latency_ms=_optional_int(call.get("latency_ms")),
            input_tokens=_optional_int(usage_json.get("input_tokens")),
            output_tokens=_optional_int(usage_json.get("output_tokens")),
            request_json=request_json,
            response_json=response_json,
            usage_json=usage_json,
            metadata_json=metadata_json,
            instructions_text=instructions_text,
            input_text=input_text,
            raw_output_text=raw_output,
            parsed_answer_text=parsed_answer,
            parsed_confidence=parsed_confidence,
            parsed_plan_id=parsed_plan_id,
            validation_status=validation_status,
            max_output_tokens=_optional_int(request_json.get("max_output_tokens")),
            reasoning_effort=_reasoning_effort(request_json),
            store=_optional_bool(request_json.get("store")),
            raw_payload_hash=raw_payload_hash,
            raw_redaction_status=_optional_str(call.get("raw_redaction_status"))
            or "unredacted",
            raw_encryption_status=_optional_str(call.get("raw_encryption_status"))
            or "plaintext",
            raw_retention_expires_at=_datetime_or_default(
                call.get("raw_retention_expires_at")
            ),
        )
    )
    db.flush()
    return call_id


def _add_answer_llm_observability_records(
    db: Session,
    *,
    query_run: QueryRun,
    observability: dict[str, Any],
    evidence_block_record_ids_by_evidence_id: dict[str, str],
    evidence_block_record_ids_by_chunk_id: dict[str, str],
) -> str | None:
    call = _mapping(observability.get("answer_llm_call"))
    if not call:
        return None

    call_id = _add_llm_call_record(
        db,
        query_run=query_run,
        call=call,
        default_stage="answer",
    )

    used_ranks: set[int] = set()
    for index, item in enumerate(
        _list_mapping(observability.get("answer_prompt_evidence")),
        start=1,
    ):
        rank = _optional_int(item.get("rank")) or index
        while rank in used_ranks:
            rank += 1
        used_ranks.add(rank)
        text_snapshot = _optional_str(item.get("text_snapshot"))
        evidence_id = (
            _optional_str(item.get("evidence_id"))
            or _optional_str(item.get("chunk_id"))
            or new_id("evref")
        )
        evidence_block_record_id = _optional_str(item.get("evidence_block_record_id"))
        if evidence_block_record_id is None:
            evidence_block_record_id = evidence_block_record_ids_by_evidence_id.get(evidence_id)
        if evidence_block_record_id is None:
            chunk_id = _optional_str(item.get("chunk_id"))
            evidence_block_record_id = (
                evidence_block_record_ids_by_chunk_id.get(chunk_id) if chunk_id else None
            )
        db.add(
            LLMCallEvidenceRecord(
                record_id=new_id("llmev"),
                call_id=call_id,
                query_id=query_run.query_id,
                evidence_id=evidence_id,
                evidence_block_record_id=evidence_block_record_id,
                rank=rank,
                provider=_optional_str(item.get("provider")),
                chunk_id=_optional_str(item.get("chunk_id")),
                document_id=_optional_str(item.get("document_id")),
                page_start=_optional_int(item.get("page_start")),
                page_end=_optional_int(item.get("page_end")),
                retrieval_score=_optional_float(item.get("retrieval_score")),
                token_count=_optional_int(item.get("token_count")),
                text_snapshot=text_snapshot,
                text_hash=_optional_str(item.get("text_hash"))
                or (_hash_text(text_snapshot) if text_snapshot else None),
                snapshot_redaction_status=_optional_str(
                    item.get("snapshot_redaction_status")
                )
                or "unredacted",
                snapshot_encryption_status=_optional_str(
                    item.get("snapshot_encryption_status")
                )
                or "plaintext",
                snapshot_retention_expires_at=_datetime_or_default(
                    item.get("snapshot_retention_expires_at"),
                    required=bool(text_snapshot),
                ),
            )
        )
    return call_id


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
        "llm_calls": _llm_call_records(db, query_id),
        "llm_call_evidence": _llm_call_evidence_records(db, query_id),
        "citations": _records(db, CitationRecord, query_id),
        "citation_verifications": _records(db, CitationVerificationRecord, query_id),
    }


def _records(db: Session, model, query_id: str) -> list[dict[str, Any]]:
    rows = db.scalars(select(model).where(model.query_id == query_id)).all()
    records: list[dict[str, Any]] = []
    for row in rows:
        record = {
            "record_id": row.record_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "payload": row.payload_json,
        }
        for attr in ("answer_call_id", "planner_call_id"):
            value = getattr(row, attr, None)
            if value:
                record[attr] = value
        records.append(record)
    return records


def _llm_call_records(db: Session, query_id: str) -> list[dict[str, Any]]:
    rows = db.scalars(
        select(LLMCallRecord)
        .where(LLMCallRecord.query_id == query_id)
        .order_by(LLMCallRecord.sequence_index.asc(), LLMCallRecord.created_at.asc())
    ).all()
    return [
        {
            "call_id": row.call_id,
            "query_id": row.query_id,
            "stage": row.stage,
            "attempt_index": row.attempt_index,
            "sequence_index": row.sequence_index,
            "model_name": row.model_name,
            "prompt_version": row.prompt_version,
            "planner_version": row.planner_version,
            "status": row.status,
            "error_message": row.error_message,
            "latency_ms": row.latency_ms,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "request": row.request_json,
            "response": row.response_json,
            "usage": row.usage_json,
            "metadata": row.metadata_json,
            "instructions_text": row.instructions_text,
            "input_text": row.input_text,
            "raw_output_text": row.raw_output_text,
            "parsed_answer_text": row.parsed_answer_text,
            "parsed_confidence": row.parsed_confidence,
            "parsed_plan_id": row.parsed_plan_id,
            "validation_status": row.validation_status,
            "max_output_tokens": row.max_output_tokens,
            "reasoning_effort": row.reasoning_effort,
            "store": row.store,
            "raw_payload_hash": row.raw_payload_hash,
            "raw_redaction_status": row.raw_redaction_status,
            "raw_encryption_status": row.raw_encryption_status,
            "raw_retention_expires_at": (
                row.raw_retention_expires_at.isoformat()
                if row.raw_retention_expires_at
                else None
            ),
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


def _llm_call_evidence_records(db: Session, query_id: str) -> list[dict[str, Any]]:
    rows = db.scalars(
        select(LLMCallEvidenceRecord)
        .where(LLMCallEvidenceRecord.query_id == query_id)
        .order_by(
            LLMCallEvidenceRecord.call_id.asc(),
            LLMCallEvidenceRecord.rank.asc(),
            LLMCallEvidenceRecord.created_at.asc(),
        )
    ).all()
    return [
        {
            "record_id": row.record_id,
            "call_id": row.call_id,
            "query_id": row.query_id,
            "evidence_id": row.evidence_id,
            "evidence_block_record_id": row.evidence_block_record_id,
            "rank": row.rank,
            "provider": row.provider,
            "chunk_id": row.chunk_id,
            "document_id": row.document_id,
            "page_start": row.page_start,
            "page_end": row.page_end,
            "retrieval_score": row.retrieval_score,
            "token_count": row.token_count,
            "text_snapshot": row.text_snapshot,
            "text_hash": row.text_hash,
            "snapshot_redaction_status": row.snapshot_redaction_status,
            "snapshot_encryption_status": row.snapshot_encryption_status,
            "snapshot_retention_expires_at": (
                row.snapshot_retention_expires_at.isoformat()
                if row.snapshot_retention_expires_at
                else None
            ),
            "created_at": row.created_at.isoformat() if row.created_at else None,
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


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _json_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return _json_safe(value)
    return {"value": _json_safe(value)}


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _text_or_json(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _reasoning_effort(request_json: dict[str, Any]) -> str | None:
    reasoning = request_json.get("reasoning")
    if isinstance(reasoning, dict):
        return _optional_str(reasoning.get("effort"))
    return None


def _hash_payload(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _datetime_or_default(value: Any, *, required: bool = True) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        except ValueError:
            pass
    return _default_retention_expires_at() if required else None


def _default_retention_expires_at() -> datetime:
    return datetime.now(UTC) + timedelta(days=30)
