import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from atlas.core.ids import new_id
from atlas.db.models import EvalResult, EvalRun, utcnow
from atlas.db.repositories import get_chunks_by_ids, get_query_run
from atlas.eval.metrics import (
    answer_gold_contains,
    answer_metric_details,
    answer_numeric_match,
    critic_metric_details,
    dense_retrieval_metrics,
    expected_confidence_hit,
    keyword_hit,
    source_hit,
)
from atlas.query_runtime.service import QueryRuntime


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expected_sources: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    expected_confidence: str | None = None
    expected_answer: str | None = None
    expected_evidence: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalService:
    def __init__(self, runtime: QueryRuntime) -> None:
        self.runtime = runtime

    def run_cases_file(
        self,
        db: Session,
        *,
        cases_path: str,
        top_k: int,
    ) -> EvalRun:
        cases = load_cases(cases_path)
        eval_run = EvalRun(
            eval_run_id=new_id("eval"),
            status="running",
            cases_path=cases_path,
            total_cases=len(cases),
        )
        db.add(eval_run)
        db.commit()

        results: list[EvalResult] = []
        for case in cases:
            results.append(self._run_case(db, eval_run.eval_run_id, case, top_k=top_k))
            db.commit()

        _finalize_eval_run(
            eval_run,
            results,
            top_k=top_k,
            retrieval_mode=self.runtime.settings.retrieval_mode,
        )
        db.add(eval_run)
        db.commit()
        db.refresh(eval_run)
        return eval_run

    def get_eval_run(self, db: Session, eval_run_id: str) -> EvalRun | None:
        return db.get(EvalRun, eval_run_id)

    def _run_case(self, db: Session, eval_run_id: str, case: EvalCase, *, top_k: int) -> EvalResult:
        started = time.perf_counter()
        try:
            query_result = self.runtime.run(db, query=case.question, top_k=top_k, filters={})
            latency_ms = int((time.perf_counter() - started) * 1000)
            query_run = get_query_run(db, query_result.query_id)
            generation = (
                query_run.generation_events[0]
                if query_run and query_run.generation_events
                else None
            )
            retrieved_top_k = _retrieved_top_k(db, query_run)
            retrieval_details = dense_retrieval_metrics(
                retrieved_top_k,
                case.expected_evidence,
                case.expected_sources,
            )
            answer_details = answer_metric_details(query_result.answer, case.expected_answer)
            query_details = _query_details(query_run, query_result.details)
            critic_details = critic_metric_details(
                actual_confidence=query_result.confidence,
                expected_confidence=case.expected_confidence,
                expected_answer=case.expected_answer,
                expected_evidence=case.expected_evidence,
                expected_sources=case.expected_sources,
                expected_keywords=case.expected_keywords,
                details=query_details,
            )
            result = EvalResult(
                eval_result_id=new_id("evr"),
                eval_run_id=eval_run_id,
                case_id=case.id,
                question=case.question,
                query_id=query_result.query_id,
                trace_id=query_result.trace_id,
                expected_confidence=case.expected_confidence,
                actual_confidence=query_result.confidence,
                source_hit=source_hit(query_result.citations, case.expected_sources),
                confidence_hit=expected_confidence_hit(
                    query_result.confidence, case.expected_confidence
                ),
                keyword_score=keyword_hit(query_result.answer, case.expected_keywords),
                latency_ms=latency_ms,
                input_tokens=generation.input_tokens if generation else None,
                output_tokens=generation.output_tokens if generation else None,
                details_json={
                    "expected_sources": case.expected_sources,
                    "expected_keywords": case.expected_keywords,
                    "expected_answer": case.expected_answer,
                    "expected_evidence": case.expected_evidence,
                    "metadata": case.metadata,
                    "citations": query_result.citations,
                    "query_details": query_details,
                    "retrieved_top_k": retrieved_top_k,
                    **retrieval_details,
                    **answer_details,
                    **critic_details,
                },
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            result = EvalResult(
                eval_result_id=new_id("evr"),
                eval_run_id=eval_run_id,
                case_id=case.id,
                question=case.question,
                expected_confidence=case.expected_confidence,
                actual_confidence=None,
                source_hit=False,
                confidence_hit=False,
                keyword_score=0.0,
                latency_ms=latency_ms,
                error_message=str(exc),
                details_json={
                    "expected_sources": case.expected_sources,
                    "expected_keywords": case.expected_keywords,
                    "expected_answer": case.expected_answer,
                    "expected_evidence": case.expected_evidence,
                    "metadata": case.metadata,
                    "retrieved_top_k": [],
                    "retrieval_doc_hit": False
                    if case.expected_evidence or case.expected_sources
                    else None,
                    "retrieval_page_hit": False if case.expected_evidence else None,
                    "retrieval_doc_mrr": 0.0
                    if case.expected_evidence or case.expected_sources
                    else None,
                    "retrieval_page_mrr": 0.0 if case.expected_evidence else None,
                    "first_doc_match_rank": None,
                    "first_page_match_rank": None,
                    "answer_gold_contains": answer_gold_contains(None, case.expected_answer),
                    "answer_numeric_match": answer_numeric_match(None, case.expected_answer),
                    **critic_metric_details(
                        actual_confidence=None,
                        expected_confidence=case.expected_confidence,
                        expected_answer=case.expected_answer,
                        expected_evidence=case.expected_evidence,
                        expected_sources=case.expected_sources,
                        expected_keywords=case.expected_keywords,
                        details=None,
                    ),
                },
            )

        db.add(result)
        return result


def load_cases(cases_path: str) -> list[EvalCase]:
    data = yaml.safe_load(Path(cases_path).read_text(encoding="utf-8"))
    cases = data.get("cases", []) if isinstance(data, dict) else data
    return [_load_case(case) for case in cases or []]


def serialize_eval_run(eval_run: EvalRun) -> dict[str, Any]:
    return {
        "eval_run_id": eval_run.eval_run_id,
        "status": eval_run.status,
        "cases_path": eval_run.cases_path,
        "total_cases": eval_run.total_cases,
        "source_hits": eval_run.source_hits,
        "confidence_hits": eval_run.confidence_hits,
        "average_keyword_score": eval_run.average_keyword_score,
        "average_latency_ms": eval_run.average_latency_ms,
        "input_tokens": eval_run.input_tokens,
        "output_tokens": eval_run.output_tokens,
        "summary": eval_run.summary_json,
        "error_message": eval_run.error_message,
        "created_at": eval_run.created_at.isoformat(),
        "finished_at": eval_run.finished_at.isoformat() if eval_run.finished_at else None,
        "results": [
            {
                "case_id": result.case_id,
                "question": result.question,
                "query_id": result.query_id,
                "trace_id": result.trace_id,
                "expected_confidence": result.expected_confidence,
                "actual_confidence": result.actual_confidence,
                "source_hit": result.source_hit,
                "confidence_hit": result.confidence_hit,
                "keyword_score": result.keyword_score,
                "latency_ms": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "details": result.details_json,
                "error_message": result.error_message,
            }
            for result in sorted(eval_run.results, key=lambda item: item.created_at)
        ],
    }


def list_eval_runs(db: Session, limit: int = 20) -> list[EvalRun]:
    return list(db.scalars(select(EvalRun).order_by(EvalRun.created_at.desc()).limit(limit)))


def _finalize_eval_run(
    eval_run: EvalRun,
    results: list[EvalResult],
    *,
    top_k: int,
    retrieval_mode: str,
) -> None:
    total = len(results)
    source_hits = sum(1 for result in results if result.source_hit)
    confidence_hits = sum(1 for result in results if result.confidence_hit)
    failures = [
        result
        for result in results
        if result.error_message
        or not result.source_hit
        or not result.confidence_hit
        or result.keyword_score < 0.5
        or _detail_metric(result, "retrieval_doc_hit") is False
        or _detail_metric(result, "retrieval_page_hit") is False
        or _detail_metric(result, "answer_gold_contains") is False
        or _detail_metric(result, "answer_numeric_match") is False
        or _detail_metric(result, "unsupported_answer") is True
        or _detail_metric(result, "false_insufficient") is True
    ]

    eval_run.status = "completed" if not failures else "partial_failed"
    eval_run.total_cases = total
    eval_run.source_hits = source_hits
    eval_run.confidence_hits = confidence_hits
    eval_run.average_keyword_score = (
        sum(result.keyword_score for result in results) / total if total else None
    )
    eval_run.average_latency_ms = (
        sum(result.latency_ms or 0 for result in results) / total if total else None
    )
    eval_run.input_tokens = sum(result.input_tokens or 0 for result in results)
    eval_run.output_tokens = sum(result.output_tokens or 0 for result in results)
    eval_run.summary_json = {
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
        "dense_retrieval": {
            "retrieval_doc_hit": _aggregate_bool_detail(results, "retrieval_doc_hit"),
            "retrieval_page_hit": _aggregate_bool_detail(results, "retrieval_page_hit"),
            "retrieval_doc_mrr": _average_detail(results, "retrieval_doc_mrr"),
            "retrieval_page_mrr": _average_detail(results, "retrieval_page_mrr"),
        },
        "retrieval": {
            "retrieval_doc_hit": _aggregate_bool_detail(results, "retrieval_doc_hit"),
            "retrieval_page_hit": _aggregate_bool_detail(results, "retrieval_page_hit"),
            "retrieval_doc_mrr": _average_detail(results, "retrieval_doc_mrr"),
            "retrieval_page_mrr": _average_detail(results, "retrieval_page_mrr"),
        },
        "answer": {
            "answer_gold_contains": _aggregate_bool_detail(results, "answer_gold_contains"),
            "answer_numeric_match": _aggregate_bool_detail(results, "answer_numeric_match"),
        },
        "critic": {
            "unsupported_answer_rate": _rate_detail(results, "unsupported_answer"),
            "false_insufficient_rate": _rate_detail(results, "false_insufficient"),
        },
        "failures": [
            {
                "case_id": result.case_id,
                "query_id": result.query_id,
                "reason": _failure_reasons(result),
                "error_message": result.error_message,
            }
            for result in failures
        ]
    }
    eval_run.error_message = f"{len(failures)} case(s) failed" if failures else None
    eval_run.finished_at = utcnow()


def _failure_reasons(result: EvalResult) -> list[str]:
    reasons: list[str] = []
    if result.error_message:
        reasons.append("error")
    if not result.source_hit:
        reasons.append("source_miss")
    if not result.confidence_hit:
        reasons.append("confidence_miss")
    if result.keyword_score < 0.5:
        reasons.append("keyword_low")
    if _detail_metric(result, "retrieval_doc_hit") is False:
        reasons.append("retrieval_doc_miss")
    if _detail_metric(result, "retrieval_page_hit") is False:
        reasons.append("retrieval_page_miss")
    if _detail_metric(result, "answer_gold_contains") is False:
        reasons.append("answer_gold_missing")
    if _detail_metric(result, "answer_numeric_match") is False:
        reasons.append("answer_numeric_miss")
    if _detail_metric(result, "unsupported_answer") is True:
        reasons.append("unsupported_answer")
    if _detail_metric(result, "false_insufficient") is True:
        reasons.append("false_insufficient")
    return reasons


def _load_case(case: dict[str, Any]) -> EvalCase:
    metadata = dict(case.get("metadata") or {})
    for key in [
        "financebench_id",
        "company",
        "doc_name",
        "doc_type",
        "doc_period",
        "doc_link",
    ]:
        if key in case and key not in metadata:
            metadata[key] = case[key]

    expected_evidence = _expected_evidence(case)
    return EvalCase(
        id=str(case["id"]),
        question=str(case["question"]),
        expected_sources=_string_list(case.get("expected_sources", [])),
        expected_keywords=_string_list(case.get("expected_keywords", [])),
        expected_confidence=case.get("expected_confidence"),
        expected_answer=case.get("expected_answer", case.get("answer")),
        expected_evidence=expected_evidence,
        metadata=metadata,
    )


def _expected_evidence(case: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_keys = [
        "doc_name",
        "canonical_doc_name",
        "document_id",
        "source_title",
        "source_uri",
        "evidence_text",
        "evidence_text_full_page",
        "evidence_page_num_raw",
        "page_num_normalized",
        "page_start",
        "page_end",
        "page",
        "page_number",
    ]
    base = {key: case[key] for key in evidence_keys if key in case}
    raw = case.get("expected_evidence", case.get("evidence"))
    if raw is None:
        if base:
            raw = [base]

    if raw is None:
        return []
    if isinstance(raw, dict):
        return [{**base, **raw}]
    if isinstance(raw, list):
        return [
            {**base, **item} if isinstance(item, dict) else {**base, "value": item}
            for item in raw
        ]
    return [{**base, "value": raw}]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def _retrieved_top_k(db: Session, query_run) -> list[dict[str, Any]]:
    if query_run is None:
        return []

    retrieval_events = sorted(query_run.retrieval_events, key=lambda event: event.rank)
    chunk_map = get_chunks_by_ids(db, [event.chunk_id for event in retrieval_events])
    retrieved: list[dict[str, Any]] = []
    for event in retrieval_events:
        chunk = chunk_map.get(event.chunk_id)
        item = {
            "rank": event.rank,
            "chunk_id": event.chunk_id,
            "score": event.retrieval_score,
            "retrieval_score": event.retrieval_score,
            "retriever_type": event.retriever_type,
        }
        if chunk is not None:
            item.update(
                {
                    "document_id": chunk.document_id,
                    "source_title": chunk.document.title,
                    "source_uri": chunk.document.source_uri,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "chunk_index": chunk.chunk_index,
                    "section_title": chunk.section_title,
                }
            )
        retrieved.append(item)
    return retrieved


def _detail_metric(result: EvalResult, key: str) -> Any:
    return (result.details_json or {}).get(key)


def _query_details(query_run, result_details: dict[str, Any] | None) -> dict[str, Any]:
    details: dict[str, Any] = {}
    if isinstance(result_details, dict):
        details.update(result_details)
    if query_run is not None:
        persisted = getattr(query_run, "details_json", None)
        if isinstance(persisted, dict):
            details.update(persisted)
    return details


def _aggregate_bool_detail(results: list[EvalResult], key: str) -> dict[str, Any]:
    values = [_detail_metric(result, key) for result in results]
    evaluated = [value for value in values if value is not None]
    hits = sum(1 for value in evaluated if value is True)
    total = len(evaluated)
    return {
        "hits": hits,
        "total": total,
        "rate": hits / total if total else None,
    }


def _rate_detail(results: list[EvalResult], key: str) -> dict[str, Any]:
    values = [_detail_metric(result, key) for result in results]
    evaluated = [value for value in values if value is not None]
    count = sum(1 for value in evaluated if value is True)
    total = len(evaluated)
    return {
        "count": count,
        "total": total,
        "rate": count / total if total else None,
    }


def _average_detail(results: list[EvalResult], key: str) -> dict[str, Any]:
    values = [
        float(value)
        for result in results
        if (value := _detail_metric(result, key)) is not None
    ]
    return {
        "total": len(values),
        "average": sum(values) / len(values) if values else None,
    }
