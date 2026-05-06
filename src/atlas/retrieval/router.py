from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any, Mapping

from sqlalchemy.orm import Session

from atlas.query_orchestrator.schema import QueryPlan
from atlas.retrieval.contracts import (
    ProviderResult,
    ProviderRouterResult,
    source_anchor_from_candidate,
)
from atlas.retrieval.models.retrieval_task import RetrievalTask


RESERVED_INTERNAL_PROVIDER_NAMES = frozenset(
    {"dense", "bm25", "sparse", "table", "section", "metric_alias"}
)


class ProviderRouter:
    """Thin runtime boundary between semantic retrieval tasks and executable providers."""

    def __init__(
        self,
        providers: Mapping[str, Any],
        *,
        known_providers: tuple[str, ...] = ("hybrid", "sql", "graph"),
    ) -> None:
        self.known_providers = tuple(str(provider).strip().lower() for provider in known_providers)
        self.providers = {}
        for name, provider in providers.items():
            if not name or provider is None:
                continue
            provider_name = str(name).strip().lower()
            if provider_name in RESERVED_INTERNAL_PROVIDER_NAMES:
                raise ValueError(
                    f"internal_lane_registered_as_provider:{provider_name}"
                )
            if provider_name not in self.known_providers:
                raise ValueError(f"unknown_provider_registered:{provider_name}")
            self.providers[provider_name] = provider

    @property
    def executable_providers(self) -> tuple[str, ...]:
        return tuple(self.providers)

    def retrieve(
        self,
        db: Session,
        *,
        query: str,
        top_k: int,
        filters: dict | None,
        options: dict,
        query_plan: QueryPlan,
        retrieval_tasks: list[RetrievalTask],
    ) -> ProviderRouterResult:
        started = time.perf_counter()
        provider_results: list[ProviderResult] = []
        evidence: list[Any] = []
        for provider_name, tasks in _group_tasks(retrieval_tasks).items():
            executable_tasks: list[RetrievalTask] = []
            for task in tasks:
                if _task_is_non_executable(task):
                    provider_results.append(
                        _skipped_result(task, known=provider_name in self.known_providers)
                    )
                    continue
                executable_tasks.append(task)
            if not executable_tasks:
                continue
            provider = self.providers.get(provider_name)
            if provider is None:
                provider_results.extend(
                    _skipped_result(task, known=provider_name in self.known_providers)
                    for task in executable_tasks
                )
                continue
            result = _execute_provider(
                provider,
                db,
                provider_name=provider_name,
                query=query,
                top_k=top_k,
                filters=filters,
                options=options,
                query_plan=query_plan,
                retrieval_tasks=executable_tasks,
            )
            provider_results.append(result)
            evidence.extend(result.evidence)

        latency_ms = int((time.perf_counter() - started) * 1000)
        return ProviderRouterResult(
            evidence=tuple(evidence[:top_k]),
            provider_results=tuple(provider_results),
            trace=_router_trace(
                query_plan=query_plan,
                provider_results=provider_results,
                latency_ms=latency_ms,
                known_providers=self.known_providers,
                executable_providers=self.executable_providers,
            ),
        )


def _group_tasks(tasks: list[RetrievalTask]) -> dict[str, list[RetrievalTask]]:
    grouped: dict[str, list[RetrievalTask]] = {}
    for task in tasks:
        grouped.setdefault(str(task.provider).strip().lower(), []).append(task)
    return grouped


def _task_is_non_executable(task: RetrievalTask) -> bool:
    if task.provider_status == "skipped_non_executable":
        return True
    reason = task.unsupported_reason or ""
    return reason.startswith("provider_not_executable_in_v1:")


def _skipped_result(task: RetrievalTask, *, known: bool) -> ProviderResult:
    reason = (
        task.unsupported_reason
        or (
            f"provider_not_executable_in_v1:{task.provider}"
            if known
            else f"unknown_provider:{task.provider}"
        )
    )
    trace = {
        "provider": task.provider,
        "task_id": task.task_id,
        "unit_id": task.unit_id,
        "status": "skipped_non_executable",
        "reason": reason,
        "planned_text": task.query_text,
        "metadata_filter": dict(task.metadata_filter),
    }
    return ProviderResult(
        provider=task.provider,
        task_id=task.task_id,
        unit_id=task.unit_id,
        status="skipped_non_executable",
        candidates=(),
        latency_ms=0,
        reason=reason,
        trace=trace,
    )


def _execute_provider(
    provider: Any,
    db: Session,
    *,
    provider_name: str,
    query: str,
    top_k: int,
    filters: dict | None,
    options: dict,
    query_plan: QueryPlan,
    retrieval_tasks: list[RetrievalTask],
) -> ProviderResult:
    retrieve_provider_result = getattr(provider, "retrieve_provider_result", None)
    owns_provider_result = "retrieve_provider_result" in type(provider).__dict__
    if callable(retrieve_provider_result) and (owns_provider_result or hasattr(provider, "default_mode")):
        return retrieve_provider_result(
            db,
            query=query,
            top_k=top_k,
            filters=filters,
            options=options,
            query_plan=query_plan,
            retrieval_tasks=retrieval_tasks,
        )
    started = time.perf_counter()
    retrieve_with_plan = getattr(provider, "retrieve_with_plan", None)
    if callable(retrieve_with_plan):
        evidence = tuple(
            retrieve_with_plan(
                db,
                query=query,
                top_k=top_k,
                filters=filters,
                options=options,
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
            )
        )
    else:
        retrieve = getattr(provider, "retrieve")
        evidence = tuple(retrieve(db, query=query, top_k=top_k, filters=filters))
    latency_ms = int((time.perf_counter() - started) * 1000)
    return ProviderResult(
        provider=provider_name,
        task_id=None if len(retrieval_tasks) != 1 else retrieval_tasks[0].task_id,
        unit_id=None if len(retrieval_tasks) != 1 else retrieval_tasks[0].unit_id,
        status="executed" if evidence else "empty",
        candidates=(),
        evidence=evidence,
        latency_ms=latency_ms,
        reason=None,
        trace={
            "provider": provider_name,
            "status": "executed" if evidence else "empty",
            "task_count": len(retrieval_tasks),
            "latency_ms": latency_ms,
            "legacy_provider_adapter": True,
        },
    )
def _router_trace(
    *,
    query_plan: QueryPlan,
    provider_results: list[ProviderResult],
    latency_ms: int,
    known_providers: tuple[str, ...],
    executable_providers: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "query_plan_id": query_plan.plan_id,
        "known_providers": list(known_providers),
        "executable_providers": list(executable_providers),
        "status": _router_status(provider_results),
        "latency_ms": latency_ms,
        "provider_results": [serialize_provider_result(result) for result in provider_results],
    }


def serialize_provider_result(result: ProviderResult) -> dict[str, Any]:
    return {
        "provider": result.provider,
        "task_id": result.task_id,
        "unit_id": result.unit_id,
        "status": result.status,
        "candidate_count": len(result.candidates),
        "candidates": [_candidate_trace_payload(candidate) for candidate in result.candidates],
        "evidence_count": len(result.evidence),
        "latency_ms": result.latency_ms,
        "reason": result.reason,
        "trace": dict(result.trace),
    }


def _candidate_trace_payload(candidate: Any) -> dict[str, Any]:
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    source_anchor = metadata.get("source_anchor")
    if not isinstance(source_anchor, dict):
        source_anchor = asdict(source_anchor_from_candidate(candidate))
    lane_attributions = metadata.get("lane_attributions")
    if not isinstance(lane_attributions, list):
        lane_attributions = []
    lanes = metadata.get("lanes")
    if not isinstance(lanes, list):
        lanes = []
    return {
        "candidate_id": getattr(candidate, "candidate_id", None),
        "provider": getattr(candidate, "provider", None) or metadata.get("provider"),
        "chunk_id": getattr(candidate, "chunk_id", None),
        "document_id": getattr(candidate, "document_id", None),
        "parent_id": getattr(candidate, "parent_id", None),
        "source_type": getattr(candidate, "source_type", None),
        "page_start": getattr(candidate, "page_start", None),
        "page_end": getattr(candidate, "page_end", None),
        "rank": (
            getattr(candidate, "final_rank", None)
            or getattr(candidate, "rerank_rank", None)
            or getattr(candidate, "fusion_rank", None)
            or getattr(candidate, "lane_rank", None)
        ),
        "lane": getattr(candidate, "lane", None) or metadata.get("lane"),
        "lanes": lanes,
        "lane_attributions": lane_attributions,
        "retrieval_task_id": getattr(candidate, "retrieval_task_id", None),
        "retrieval_unit_id": getattr(candidate, "retrieval_unit_id", None),
        "dense_score": getattr(candidate, "dense_score", None),
        "lexical_score": getattr(candidate, "lexical_score", None),
        "fusion_score": getattr(candidate, "fusion_score", None),
        "rerank_score": getattr(candidate, "rerank_score", None),
        "weighted_contribution": getattr(candidate, "weighted_contribution", None),
        "source_anchor": source_anchor,
    }


def _router_status(results: list[ProviderResult]) -> str:
    if not results:
        return "empty"
    executed = any(result.status in {"executed", "empty"} for result in results)
    skipped = any(result.status == "skipped_non_executable" for result in results)
    failed = any(result.status == "failed" for result in results)
    if failed:
        return "failed"
    if executed and skipped:
        return "partial"
    if executed:
        return "executed"
    return "skipped_non_executable"


def _ordered_unique(values: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered
