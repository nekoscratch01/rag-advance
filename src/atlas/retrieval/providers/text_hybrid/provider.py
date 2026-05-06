from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from typing import Any

from sqlalchemy.orm import Session

from atlas.db import repositories
from atlas.query_orchestrator.schema import QueryPlan
from atlas.query_runtime.evidence_builder import (
    build_evidence_pack_from_candidates,
    evidence_pack_to_evidence,
)
from atlas.retrieval.contracts import ProviderResult, source_anchor_from_candidate
from atlas.retrieval.models.candidate import Candidate
from atlas.retrieval.models.evidence import Evidence
from atlas.retrieval.models.retrieval_task import RetrievalTask
from atlas.retrieval.providers.text_hybrid.adapters.hybrid import HybridRetriever
from atlas.retrieval.providers.text_hybrid.adapters.mode_switching import ModeSwitchingRetriever
from atlas.retrieval.ranking.fusion import DEFAULT_RRF_K, WeightedRRFInput, weighted_rrf_fuse
from atlas.retrieval.providers.text_hybrid.lanes import (
    SPARSE_BOOST_REPEAT,
    SUPPORTED_LANES,
    TEXT_HYBRID_PROVIDER,
    TextHybridLane,
    lane_filters,
    lane_query_text,
)
from atlas.retrieval.ranking.reranker import Reranker, rerank_with_context


FUSION_BACKEND = "weighted_rrf"
SUPPORTED_PROVIDER_STATUSES = frozenset({"ready", "supported", "ok", "enabled"})


@dataclass(frozen=True)
class TextHybridRun:
    candidates: tuple[Candidate, ...]
    evidence: tuple[Evidence, ...]
    evidence_pack: Any | None
    trace: dict[str, Any]
    latency_ms: int


@dataclass(frozen=True)
class EvidenceBuildResult:
    evidence: tuple[Evidence, ...]
    evidence_pack: Any | None


class TextHybridProvider:
    """Provider-local V1 text retrieval boundary with dense, lexical, and textual table lanes."""

    def __init__(
        self,
        *,
        dense_retriever,
        bm25_retriever,
        hybrid_rrf_retriever: HybridRetriever,
        hybrid_rerank_retriever: HybridRetriever,
        default_mode: str,
        rrf_k: int = DEFAULT_RRF_K,
        rrf_top_k: int = 40,
        reranker: Reranker | None = None,
        reranker_enabled: bool = True,
        reranker_top_k: int = 30,
        reranker_output_k: int | None = 8,
        dense_top_k: int | None = None,
        lexical_top_k: int | None = None,
        max_context_tokens: int | None = None,
    ) -> None:
        self.provider_name = TEXT_HYBRID_PROVIDER
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.rrf_top_k = rrf_top_k
        self.reranker = reranker
        self.reranker_enabled = reranker_enabled
        self.reranker_top_k = reranker_top_k
        self.reranker_output_k = reranker_output_k
        self.dense_top_k = dense_top_k
        self.lexical_top_k = lexical_top_k
        self.max_context_tokens = max_context_tokens
        self.default_mode = default_mode
        self.mode_switcher = ModeSwitchingRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            hybrid_rrf_retriever=hybrid_rrf_retriever,
            hybrid_rerank_retriever=hybrid_rerank_retriever,
            default_mode=default_mode,
        )
        self.lanes = {
            "dense": TextHybridLane("dense", "dense", dense_retriever),
            "bm25": TextHybridLane("bm25", "lexical", bm25_retriever),
            "metric_alias": TextHybridLane("metric_alias", "lexical", bm25_retriever),
            "section": TextHybridLane("section", "lexical", bm25_retriever),
            "table": TextHybridLane("table", "lexical", bm25_retriever),
        }

    def retrieve(
        self,
        db: Session,
        *,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[Evidence]:
        return self.retrieve_with_options(
            db,
            query=query,
            top_k=top_k,
            filters=filters,
            options={},
        )

    def retrieve_with_options(
        self,
        db: Session,
        *,
        query: str,
        top_k: int,
        filters: dict | None = None,
        options: dict | None = None,
    ) -> list[Evidence]:
        evidence = self.mode_switcher.retrieve_with_options(
            db,
            query=query,
            top_k=top_k,
            filters=filters,
            options=options or {},
        )
        return [_with_provider_metadata(item, legacy_mode=True) for item in evidence]

    def retrieve_with_plan(
        self,
        db: Session,
        *,
        query: str,
        top_k: int,
        filters: dict | None = None,
        options: dict | None = None,
        query_plan: QueryPlan,
        retrieval_tasks: list[RetrievalTask],
    ) -> list[Evidence]:
        run = self._run_with_plan(
            db,
            query=query,
            top_k=top_k,
            filters=filters,
            options=options or {},
            query_plan=query_plan,
            retrieval_tasks=retrieval_tasks,
        )
        return list(run.evidence)

    def retrieve_provider_result(
        self,
        db: Session,
        *,
        query: str,
        top_k: int,
        filters: dict | None,
        options: dict,
        query_plan: QueryPlan,
        retrieval_tasks: list[RetrievalTask],
    ) -> ProviderResult:
        run = self._run_with_plan(
            db,
            query=query,
            top_k=top_k,
            filters=filters,
            options=options or {},
            query_plan=query_plan,
            retrieval_tasks=retrieval_tasks,
        )
        return ProviderResult(
            provider="hybrid",
            task_id=None if len(retrieval_tasks) != 1 else retrieval_tasks[0].task_id,
            unit_id=None if len(retrieval_tasks) != 1 else retrieval_tasks[0].unit_id,
            status="executed" if run.candidates or run.evidence else "empty",
            candidates=run.candidates,
            evidence=run.evidence,
            evidence_pack=run.evidence_pack,
            latency_ms=run.latency_ms,
            reason=None,
            trace=run.trace,
        )

    def _run_with_plan(
        self,
        db: Session,
        *,
        query: str,
        top_k: int,
        filters: dict | None,
        options: dict[str, Any],
        query_plan: QueryPlan,
        retrieval_tasks: list[RetrievalTask],
        build_evidence: bool = True,
    ) -> TextHybridRun:
        started = time.perf_counter()
        candidates, trace = self._retrieve_candidates_with_trace(
            db,
            query=query,
            top_k=top_k,
            filters=filters,
            options=options,
            query_plan=query_plan,
            retrieval_tasks=retrieval_tasks,
        )
        evidence_result = (
            self._candidates_to_evidence(
                db,
                candidates,
                top_k=top_k,
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
            )
            if build_evidence
            else EvidenceBuildResult(evidence=(), evidence_pack=None)
        )
        latency_ms = int(trace.get("retrieval_latency_ms") or 0)
        if latency_ms <= 0:
            latency_ms = int((time.perf_counter() - started) * 1000)
        return TextHybridRun(
            candidates=tuple(candidates),
            evidence=evidence_result.evidence,
            evidence_pack=evidence_result.evidence_pack,
            trace=trace,
            latency_ms=latency_ms,
        )

    def retrieve_candidates_with_plan(
        self,
        db: Session,
        *,
        query: str,
        top_k: int,
        filters: dict | None,
        options: dict[str, Any],
        query_plan: QueryPlan,
        retrieval_tasks: list[RetrievalTask],
    ) -> list[Candidate]:
        run = self._run_with_plan(
            db,
            query=query,
            top_k=top_k,
            filters=filters,
            options=options,
            query_plan=query_plan,
            retrieval_tasks=retrieval_tasks,
            build_evidence=False,
        )
        return list(run.candidates)

    def _retrieve_candidates_with_trace(
        self,
        db: Session,
        *,
        query: str,
        top_k: int,
        filters: dict | None,
        options: dict[str, Any],
        query_plan: QueryPlan,
        retrieval_tasks: list[RetrievalTask],
    ) -> tuple[list[Candidate], dict[str, Any]]:
        if top_k <= 0:
            trace = _provider_trace(
                query_plan=query_plan,
                task_traces=[],
                lane_traces=[],
                retrieval_latency_ms=0,
                rrf_k=self.rrf_k,
                fused_count=0,
            )
            return [], trace

        started = time.perf_counter()
        lane_candidate_groups: dict[str, list[Candidate]] = {}
        all_lane_candidates: list[Candidate] = []
        task_traces: list[dict[str, Any]] = []
        lane_traces: list[dict[str, Any]] = []
        mode = _retrieval_mode(options, self.default_mode)
        for task in retrieval_tasks:
            internal_lanes = _task_internal_lanes(task)
            if not _task_is_supported(task):
                task_traces.append(
                    {
                        "provider": task.provider,
                        "text_hybrid_provider": TEXT_HYBRID_PROVIDER,
                        "provider_status": task.provider_status,
                        "unsupported_reason": task.unsupported_reason,
                        "internal_lanes": list(internal_lanes),
                        "task_id": task.task_id,
                        "unit_id": task.unit_id,
                        "status": "skipped",
                        "returned": 0,
                        "latency_ms": 0,
                    }
                )
                continue
            lane_names = _lanes_for_mode(internal_lanes, mode)
            task_traces.append(
                {
                    "provider": task.provider,
                    "text_hybrid_provider": TEXT_HYBRID_PROVIDER,
                    "provider_status": task.provider_status,
                    "unsupported_reason": task.unsupported_reason,
                    "internal_lanes": list(internal_lanes),
                    "lanes": list(lane_names),
                    "task_id": task.task_id,
                    "unit_id": task.unit_id,
                    "status": "executed" if lane_names else "skipped",
                    "skip_reason": None if lane_names else "no_supported_internal_lanes",
                }
            )
            for lane_name in lane_names:
                lane = self.lanes[lane_name]
                lane_top_k = self._lane_top_k(lane_name, task.top_k, top_k)
                lane_query = lane_query_text(task, lane_name)
                lane_started = time.perf_counter()
                current_lane_candidates = lane.retrieve(
                    db,
                    task=task,
                    query_text=lane_query,
                    top_k=lane_top_k,
                    filters=lane_filters(filters, task),
                )
                lane_trace = {
                    "provider": TEXT_HYBRID_PROVIDER,
                    "task_provider": task.provider,
                    "provider_status": task.provider_status,
                    "unsupported_reason": task.unsupported_reason,
                    "lane": lane_name,
                    "lane_family": lane.family,
                    "internal_lanes": list(internal_lanes),
                    "fusion_backend": FUSION_BACKEND,
                    "task_id": task.task_id,
                    "unit_id": task.unit_id,
                    "query_text": lane_query,
                    "metadata_filter": dict(task.metadata_filter),
                    "sparse_boost_terms": list(task.must_have_terms)
                    if lane.family == "lexical"
                    else [],
                    "sparse_boost_repeat": SPARSE_BOOST_REPEAT
                    if lane.family == "lexical"
                    else 0,
                    "requested_top_k": lane_top_k,
                    "returned": len(current_lane_candidates),
                    "latency_ms": int((time.perf_counter() - lane_started) * 1000),
                }
                lane_traces.append(lane_trace)
                all_lane_candidates.extend(current_lane_candidates)
                lane_candidate_groups.setdefault(lane_name, []).extend(current_lane_candidates)

        fused_limit = self.rrf_top_k if self._reranker_enabled(options) else top_k
        fused = weighted_rrf_fuse(
            [
                WeightedRRFInput(lane=lane_name, candidates=candidates)
                for lane_name, candidates in lane_candidate_groups.items()
            ],
            rrf_k=self.rrf_k,
            limit=fused_limit,
        )
        retrieval_latency_ms = int((time.perf_counter() - started) * 1000)
        provider_trace = _provider_trace(
            query_plan=query_plan,
            task_traces=task_traces,
            lane_traces=lane_traces,
            retrieval_latency_ms=retrieval_latency_ms,
            rrf_k=self.rrf_k,
            fused_count=len(fused),
        )
        fused = [
            _with_candidate_provider_metadata(
                candidate,
                query_plan=query_plan,
                provider_trace=provider_trace,
                lane_traces=lane_traces,
                lane_attributions=_lane_attributions_by_chunk(all_lane_candidates),
                retrieval_latency_ms=retrieval_latency_ms,
            )
            for candidate in fused
        ]
        if self._reranker_enabled(options):
            fused = self._rerank(
                query=query,
                candidates=fused,
                top_k=top_k,
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
            )
        return fused[:top_k], provider_trace

    def _rerank(
        self,
        *,
        query: str,
        candidates: list[Candidate],
        top_k: int,
        query_plan: QueryPlan | None = None,
        retrieval_tasks: list[RetrievalTask] | None = None,
    ) -> list[Candidate]:
        if not candidates:
            return []
        if self.reranker is None:
            raise RuntimeError(
                "TextHybridProvider reranker is enabled but no reranker is configured. "
                "Wire a local reranker or set ATLAS_RERANKER_ENABLED=false."
            )
        rerank_top_k = min(self.reranker_top_k, len(candidates))
        output_limit = (
            min(top_k, self.reranker_output_k)
            if self.reranker_output_k is not None
            else top_k
        )
        return rerank_with_context(
            self.reranker,
            query=query,
            candidates=candidates[:rerank_top_k],
            top_k=rerank_top_k,
            query_plan=query_plan,
            retrieval_tasks=retrieval_tasks,
            output_k=output_limit,
        )[:output_limit]

    def _lane_top_k(self, lane_name: str, task_top_k: int, requested_top_k: int) -> int:
        configured = self.dense_top_k if lane_name == "dense" else self.lexical_top_k
        return max(1, configured or task_top_k or requested_top_k)

    def _reranker_enabled(self, options: dict[str, Any]) -> bool:
        if "reranker_enabled" in options:
            return _truthy(options.get("reranker_enabled"))
        if "rerank" in options:
            return _truthy(options.get("rerank"))
        mode = _retrieval_mode(options, self.default_mode)
        if mode in {"hybrid_rrf", "hybrid-rrf", "hybrid_no_rerank"}:
            return False
        return self.reranker_enabled

    def _candidates_to_evidence(
        self,
        db: Session,
        candidates: list[Candidate],
        *,
        top_k: int,
        query_plan: QueryPlan | None = None,
        retrieval_tasks: list[RetrievalTask] | None = None,
    ) -> EvidenceBuildResult:
        if self.max_context_tokens is not None:
            trace_by_parent = _weighted_trace_by_parent(candidates)
            pack = build_evidence_pack_from_candidates(
                candidates,
                parent_resolver=_parent_resolver(db, candidates),
                max_context_tokens=self.max_context_tokens,
                max_blocks=top_k,
                plan_id=query_plan.plan_id if query_plan else None,
                retrieval_unit_coverage=_retrieval_unit_coverage(
                    candidates,
                    query_plan=query_plan,
                    retrieval_tasks=retrieval_tasks or [],
                ),
            )
            evidence = [
                _with_provider_metadata(
                    item,
                    legacy_mode=False,
                    weighted_trace=trace_by_parent.get(item.parent_id or item.chunk_id),
                )
                for item in evidence_pack_to_evidence(pack)
            ]
            return EvidenceBuildResult(
                evidence=tuple(_with_evidence_pack_metadata(item, pack) for item in evidence),
                evidence_pack=pack,
            )
        return EvidenceBuildResult(
            evidence=tuple(
                _candidate_to_evidence(candidate, index)
                for index, candidate in enumerate(candidates[:top_k], start=1)
            ),
            evidence_pack=None,
        )


def _lanes_for_mode(lanes: tuple[str, ...], mode: str) -> tuple[str, ...]:
    if mode in {"dense", "dense_only"}:
        return ("dense",)
    if mode in {"bm25", "lexical", "bm25_only"}:
        return ("bm25",)
    return _normalize_lanes(lanes)


def _task_is_supported(task: RetrievalTask) -> bool:
    return (
        str(task.provider).strip().lower() == "hybrid"
        and str(task.provider_status).strip().lower() in SUPPORTED_PROVIDER_STATUSES
    )


def _task_internal_lanes(task: RetrievalTask) -> tuple[str, ...]:
    lanes = tuple(str(lane) for lane in task.internal_lanes if lane)
    if not lanes:
        lanes = tuple(str(lane) for lane in task.lanes if lane)
    if lanes:
        return lanes
    if str(task.provider).strip().lower() != "hybrid":
        return ()
    return ("dense", "bm25")


def _provider_trace(
    *,
    query_plan: QueryPlan,
    task_traces: list[dict[str, Any]],
    lane_traces: list[dict[str, Any]],
    retrieval_latency_ms: int,
    rrf_k: int,
    fused_count: int,
) -> dict[str, Any]:
    input_lanes = _ordered_unique(
        lane
        for task in task_traces
        for lane in task.get("internal_lanes", [])
    )
    executed_lanes = _ordered_unique(
        str(item.get("lane")) for item in lane_traces if item.get("lane")
    )
    return {
        "provider": TEXT_HYBRID_PROVIDER,
        "provider_status": _provider_trace_status(task_traces),
        "query_plan_id": query_plan.plan_id,
        "planner": query_plan.planner,
        "internal_lanes": input_lanes,
        "executed_lanes": executed_lanes,
        "tasks": task_traces,
        "lanes": lane_traces,
        "fusion_backend": FUSION_BACKEND,
        "fusion": {
            "backend": FUSION_BACKEND,
            "strategy": FUSION_BACKEND,
            "version": "current",
            "rrf_k": rrf_k,
            "input_lanes": executed_lanes,
            "fused_count": fused_count,
        },
        "retrieval_latency_ms": retrieval_latency_ms,
    }


def _provider_trace_status(task_traces: list[dict[str, Any]]) -> str:
    if not task_traces:
        return "empty"
    executed = any(item.get("status") == "executed" for item in task_traces)
    skipped = any(item.get("status") == "skipped" for item in task_traces)
    if executed and skipped:
        return "partial"
    if executed:
        return "ready"
    return "skipped"


def _normalize_lanes(lanes: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(lane for lane in lanes if lane in SUPPORTED_LANES)
    if normalized:
        return normalized
    if lanes:
        return ()
    return ("dense", "bm25")


def _retrieval_mode(options: dict[str, Any], default_mode: str) -> str:
    return str(
        options.get("retrieval_mode")
        or options.get("mode")
        or options.get("benchmark_mode")
        or default_mode
    ).strip().lower()


def _candidate_to_evidence(candidate: Candidate, evidence_index: int) -> Evidence:
    retrieval_score = candidate.rerank_score
    if retrieval_score is None:
        retrieval_score = candidate.fusion_score
    if retrieval_score is None:
        retrieval_score = candidate.lane_score
    if retrieval_score is None:
        retrieval_score = candidate.dense_score or candidate.lexical_score or 0.0
    candidate_metadata = dict(candidate.metadata)
    lane_attributions = candidate_metadata.get("lane_attributions")
    if not isinstance(lane_attributions, list):
        lane_attributions = []
    lanes = candidate_metadata.get("lanes")
    if not isinstance(lanes, list):
        lanes = _ordered_unique(
            str(item.get("lane")) for item in lane_attributions if isinstance(item, dict)
        )
    lane = "multi_lane" if len(lanes) > 1 else (lanes[0] if lanes else candidate.lane)
    metadata = {
        **candidate_metadata,
        "provider": TEXT_HYBRID_PROVIDER,
        "source_anchor": asdict(source_anchor_from_candidate(candidate)),
        "lane": lane,
        "lanes": lanes,
        "lane_attributions": lane_attributions,
        "lane_rank": candidate.lane_rank,
        "lane_score": candidate.lane_score,
        "lane_weight": candidate.lane_weight,
        "retrieval_task_id": candidate.retrieval_task_id,
        "retrieval_unit_id": candidate.retrieval_unit_id,
        "retrieval_unit_weight": candidate.unit_weight,
        "retrieved_by": list(candidate.retrieved_by),
        "dense_rank": candidate.dense_rank,
        "dense_score": candidate.dense_score,
        "lexical_rank": candidate.lexical_rank,
        "lexical_score": candidate.lexical_score,
        "lexical_backend": candidate.lexical_backend,
        "fusion_rank": candidate.fusion_rank,
        "fusion_score": candidate.fusion_score,
        "rerank_rank": candidate.rerank_rank,
        "rerank_score": candidate.rerank_score,
    }
    return Evidence(
        evidence_id=f"c{evidence_index}",
        document_id=candidate.document_id,
        chunk_id=candidate.chunk_id,
        text=candidate.text,
        source_title=candidate.source_title,
        source_uri=candidate.source_uri,
        section_title=candidate.section_title,
        page_start=candidate.page_start,
        page_end=candidate.page_end,
        retrieval_score=float(retrieval_score),
        rank=candidate.final_rank or candidate.rerank_rank or candidate.fusion_rank or evidence_index,
        token_count=candidate.token_count,
        metadata=metadata,
        parent_id=candidate.parent_id,
        retrieved_by=candidate.retrieved_by,
        rerank_score=candidate.rerank_score,
        rerank_rank=candidate.rerank_rank,
    )


def _with_candidate_provider_metadata(
    candidate: Candidate,
    *,
    query_plan: QueryPlan,
    provider_trace: dict[str, Any],
    lane_traces: list[dict[str, Any]],
    lane_attributions: dict[str, list[dict[str, Any]]],
    retrieval_latency_ms: int,
) -> Candidate:
    metadata = dict(candidate.metadata)
    candidate_attributions = lane_attributions.get(candidate.chunk_id, [])
    metadata["provider"] = TEXT_HYBRID_PROVIDER
    metadata["source_anchor"] = asdict(source_anchor_from_candidate(candidate))
    metadata["query_plan_id"] = query_plan.plan_id
    metadata["lane_attributions"] = candidate_attributions
    metadata["lanes"] = _ordered_unique(
        str(item.get("lane")) for item in candidate_attributions if item.get("lane")
    )
    if candidate_attributions:
        metadata["retrieval_tasks"] = candidate_attributions
        metadata["lane_trace"] = _canonical_lane_trace(candidate_attributions)
    metadata["fusion_backend"] = FUSION_BACKEND
    metadata["text_hybrid_provider"] = {
        **provider_trace,
        "lanes": lane_traces,
        "retrieval_latency_ms": retrieval_latency_ms,
    }
    return replace(candidate, provider=TEXT_HYBRID_PROVIDER, metadata=metadata)


def _lane_attributions_by_chunk(candidates: list[Candidate]) -> dict[str, list[dict[str, Any]]]:
    attributions: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        chunk_id = candidate.chunk_id
        if not chunk_id:
            continue
        payloads = _candidate_lane_payloads(candidate)
        for payload in payloads:
            _append_unique_attribution(attributions.setdefault(chunk_id, []), payload)
    return attributions


def _candidate_lane_payloads(candidate: Candidate) -> list[dict[str, Any]]:
    metadata = candidate.metadata or {}
    tasks = metadata.get("retrieval_tasks")
    if isinstance(tasks, list) and tasks:
        return [dict(item) for item in tasks if isinstance(item, dict)]
    lane_trace = metadata.get("lane_trace")
    if isinstance(lane_trace, dict):
        return [dict(lane_trace)]
    return [
        {
            "provider": TEXT_HYBRID_PROVIDER,
            "lane": candidate.lane,
            "fusion_backend": FUSION_BACKEND,
            "lane_rank": candidate.lane_rank,
            "lane_score": candidate.lane_score,
            "lane_weight": candidate.lane_weight,
            "retrieval_task_id": candidate.retrieval_task_id,
            "retrieval_unit_id": candidate.retrieval_unit_id,
            "retrieval_unit_weight": candidate.unit_weight,
        }
    ]


def _canonical_lane_trace(attributions: list[dict[str, Any]]) -> dict[str, Any]:
    lanes = _ordered_unique(
        str(item.get("lane")) for item in attributions if item.get("lane")
    )
    provider_statuses = _ordered_unique(
        str(item.get("provider_status"))
        for item in attributions
        if item.get("provider_status")
    )
    internal_lanes = _ordered_unique(
        str(lane)
        for item in attributions
        for lane in item.get("internal_lanes", [])
    )
    if len(attributions) == 1:
        payload = dict(attributions[0])
        payload["lanes"] = lanes
        payload["fusion_backend"] = FUSION_BACKEND
        return payload
    return {
        "provider": TEXT_HYBRID_PROVIDER,
        "lane": "multi_lane",
        "lanes": lanes,
        "provider_statuses": provider_statuses,
        "internal_lanes": internal_lanes,
        "fusion_backend": FUSION_BACKEND,
        "lane_attributions": attributions,
    }


def _append_unique_attribution(items: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    key = (
        payload.get("lane"),
        payload.get("retrieval_task_id"),
        payload.get("retrieval_unit_id"),
        payload.get("lane_rank"),
    )
    for item in items:
        existing_key = (
            item.get("lane"),
            item.get("retrieval_task_id"),
            item.get("retrieval_unit_id"),
            item.get("lane_rank"),
        )
        if existing_key == key:
            return
    items.append(payload)


def _ordered_unique(values) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _with_provider_metadata(
    evidence: Evidence,
    *,
    legacy_mode: bool,
    weighted_trace: dict[str, Any] | None = None,
) -> Evidence:
    metadata = dict(evidence.metadata)
    metadata.setdefault("provider", TEXT_HYBRID_PROVIDER)
    metadata.setdefault("retrieval_provider", TEXT_HYBRID_PROVIDER)
    metadata.setdefault("provider_contract", "TextHybridProvider")
    if weighted_trace is not None:
        metadata.update(weighted_trace)
    if legacy_mode:
        metadata.setdefault("provider_path", "legacy_mode_switch")
    return replace(evidence, metadata=metadata)


def _with_evidence_pack_metadata(evidence: Evidence, pack) -> Evidence:
    metadata = dict(evidence.metadata)
    metadata["evidence_pack"] = _evidence_pack_summary(pack)
    return replace(evidence, metadata=metadata)


def _evidence_pack_summary(pack) -> dict[str, Any]:
    return {
        "pack_id": pack.pack_id,
        "token_count": pack.token_count,
        "max_context_tokens": pack.max_context_tokens,
        "block_count": len(pack.blocks),
        "dropped_block_count": len(pack.dropped_blocks),
        "blocks": [_evidence_block_summary(block) for block in pack.blocks],
        "dropped_blocks": [_evidence_block_summary(block) for block in pack.dropped_blocks],
        "metadata": dict(pack.metadata),
    }


def _evidence_block_summary(block) -> dict[str, Any]:
    metadata = dict(getattr(block, "metadata", {}) or {})
    return {
        "evidence_id": block.evidence_id,
        "document_id": block.document_id,
        "chunk_ids": list(block.chunk_ids),
        "parent_id": block.parent_id,
        "rank": block.rank,
        "token_count": block.token_count,
        "included_in_prompt": block.included_in_prompt,
        "drop_reason": block.drop_reason,
        "drop_stage": block.drop_stage,
        "coverage": block.coverage,
        "source_anchor": metadata.get("source_anchor"),
    }


def _retrieval_unit_coverage(
    candidates: list[Candidate],
    *,
    query_plan: QueryPlan | None,
    retrieval_tasks: list[RetrievalTask],
) -> dict[str, Any]:
    plan_entities = [entity.value for entity in query_plan.entities] if query_plan else []
    plan_metrics = [metric.canonical_name for metric in query_plan.metrics] if query_plan else []
    plan_periods = [
        period.normalized or period.value
        for period in query_plan.periods
    ] if query_plan else []
    unit_ids = _ordered_unique(
        [
            *(task.unit_id for task in retrieval_tasks),
            *(candidate.retrieval_unit_id for candidate in candidates if candidate.retrieval_unit_id),
        ]
    )
    entities = _ordered_unique(
        [
            *plan_entities,
            *(str(candidate.company) for candidate in candidates if candidate.company),
        ]
    )
    metrics: list[str] = list(plan_metrics)
    periods: list[str] = list(plan_periods)
    for candidate in candidates:
        metadata = candidate.metadata or {}
        metrics.extend(str(item) for item in _dict_items(metadata.get("metrics")))
        periods.extend(str(item) for item in _dict_items(metadata.get("periods")))
    return {
        "retrieval_unit_ids": unit_ids,
        "entities": entities,
        "metrics": _ordered_unique(metrics),
        "periods": _ordered_unique(periods),
    }


def _weighted_trace_by_parent(candidates: list[Candidate]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        key = candidate.parent_id or candidate.metadata.get("parent_id") or candidate.chunk_id
        if key:
            grouped.setdefault(str(key), []).append(candidate)

    return {
        parent_id: _canonical_weighted_parent_trace(parent_candidates)
        for parent_id, parent_candidates in grouped.items()
    }


def _canonical_weighted_parent_trace(candidates: list[Candidate]) -> dict[str, Any]:
    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.final_rank or candidate.rerank_rank or candidate.fusion_rank or 10**9,
            candidate.chunk_id,
        ),
    )
    all_lane_contributions: list[dict[str, Any]] = []
    all_lane_attributions: list[dict[str, Any]] = []
    contributions_by_chunk: dict[str, list[dict[str, Any]]] = {}
    attributions_by_chunk: dict[str, list[dict[str, Any]]] = {}
    for candidate in ordered_candidates:
        metadata = candidate.metadata or {}
        candidate_contributions = _dict_items(metadata.get("lane_contributions"))
        fusion = metadata.get("fusion")
        if isinstance(fusion, dict):
            candidate_contributions.extend(_dict_items(fusion.get("lane_contributions")))
        candidate_attributions = _dict_items(metadata.get("lane_attributions"))
        if candidate.chunk_id:
            contributions_by_chunk.setdefault(candidate.chunk_id, []).extend(
                candidate_contributions
            )
            attributions_by_chunk.setdefault(candidate.chunk_id, []).extend(
                candidate_attributions
            )
        all_lane_contributions.extend(candidate_contributions)
        all_lane_attributions.extend(candidate_attributions)

    best = min(
        ordered_candidates,
        key=lambda candidate: (
            candidate.final_rank or candidate.rerank_rank or candidate.fusion_rank or 10**9,
            candidate.chunk_id,
        ),
    )
    lane_contributions = _dedupe_trace_items(
        contributions_by_chunk.get(best.chunk_id, [])
    )
    lane_attributions = _dedupe_trace_items(attributions_by_chunk.get(best.chunk_id, []))
    parent_child_contributions = _dedupe_trace_items(all_lane_contributions)
    parent_child_attributions = _dedupe_trace_items(all_lane_attributions)
    lanes = _ordered_unique(
        str(item.get("lane")) for item in lane_contributions if item.get("lane")
    )
    parent_lanes = _ordered_unique(
        str(item.get("lane")) for item in parent_child_contributions if item.get("lane")
    )
    provider_statuses = _ordered_unique(
        str(item.get("provider_status"))
        for item in parent_child_attributions
        if item.get("provider_status")
    )
    internal_lanes = _ordered_unique(
        str(lane)
        for item in parent_child_attributions
        for lane in item.get("internal_lanes", [])
    )
    fusion_score = float(best.fusion_score or 0.0)
    weighted_contribution = float(best.weighted_contribution or fusion_score)
    lane = "multi_lane" if len(lanes) > 1 else (lanes[0] if lanes else best.lane)
    fusion = {
        "strategy": "weighted_rrf",
        "backend": FUSION_BACKEND,
        "version": "current",
        "lane": lane,
        "lanes": lanes,
        "parent_lanes": parent_lanes,
        "lane_contributions": lane_contributions,
        "parent_child_contributions": parent_child_contributions,
        "weighted_contribution": weighted_contribution,
        "fusion_score": fusion_score,
        "fusion_rank": best.fusion_rank,
        "final_rank": best.final_rank,
        "winning_chunk_id": best.chunk_id,
        "retrieved_by": list(best.retrieved_by),
    }
    return {
        "provider": TEXT_HYBRID_PROVIDER,
        "retrieval_provider": TEXT_HYBRID_PROVIDER,
        "provider_contract": "TextHybridProvider",
        "provider_status": provider_statuses[0] if len(provider_statuses) == 1 else "mixed",
        "provider_statuses": provider_statuses,
        "internal_lanes": internal_lanes,
        "fusion_backend": FUSION_BACKEND,
        "lane": lane,
        "lanes": lanes,
        "parent_lanes": parent_lanes,
        "lane_attributions": lane_attributions,
        "parent_child_attributions": parent_child_attributions,
        "lane_contributions": lane_contributions,
        "parent_child_contributions": parent_child_contributions,
        "weighted_contribution": weighted_contribution,
        "fusion": fusion,
        "fusion_score": fusion_score,
        "fusion_rank": best.fusion_rank,
        "winning_chunk_id": best.chunk_id,
        "retrieval_task_id": best.retrieval_task_id,
        "retrieval_unit_id": best.retrieval_unit_id,
        "lane_trace": {
            "provider": TEXT_HYBRID_PROVIDER,
            "lane": lane,
            "lanes": lanes,
            "parent_lanes": parent_lanes,
            "provider_statuses": provider_statuses,
            "internal_lanes": internal_lanes,
            "fusion_backend": FUSION_BACKEND,
            "lane_attributions": lane_attributions,
            "parent_child_attributions": parent_child_attributions,
        },
    }


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, list | tuple):
        items: list[dict[str, Any]] = []
        for item in value:
            items.extend(_dict_items(item))
        return items
    return []


def _dedupe_trace_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in items:
        key = (
            item.get("chunk_id"),
            item.get("lane"),
            item.get("rank") or item.get("lane_rank"),
            item.get("retrieval_task_id"),
            item.get("retrieval_unit_id"),
            item.get("weighted_contribution"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _parent_resolver(db: Session, candidates: list[Candidate]):
    parent_ids = []
    for candidate in candidates:
        parent_id = candidate.parent_id or candidate.metadata.get("parent_id")
        if parent_id:
            parent_ids.append(str(parent_id))
    return repositories.get_parent_blocks_by_ids(db, parent_ids)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled", "enable"}
