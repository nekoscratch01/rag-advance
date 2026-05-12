from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import time
from typing import Any, Callable, Mapping

from sqlalchemy.orm import Session

from atlas.core.config import NON_EXECUTABLE_QUERY_PROVIDERS, RESERVED_INTERNAL_PROVIDER_NAMES
from atlas.db import repositories
from atlas.query_orchestrator.schema import QueryPlan
from atlas.query_runtime.evidence_builder import (
    build_evidence_pack_from_candidates,
    evidence_pack_to_evidence,
)
from atlas.retrieval.candidate_fusion import CandidateFusion
from atlas.retrieval.contracts import (
    ProviderResult,
    ProviderRouterResult,
    source_anchor_from_candidate,
)
from atlas.retrieval.models.retrieval_task import RetrievalTask
from atlas.retrieval.providers.base import RetrievalProvider
from atlas.retrieval.ranking.reranker import Reranker, rerank_with_context


RESERVED_INTERNAL_PROVIDER_NAMES = frozenset(
    {"dense", "bm25", "sparse", "table", "section", "metric_alias"}
)
NON_EXECUTABLE_PROVIDER_NAMES = frozenset({"sql"})
TEXT_RERANK_SOURCE_TYPES = frozenset({"text_chunk", "parent_block"})


class ProviderRouter:
    """Runtime boundary between semantic retrieval tasks and executable providers."""

    def __init__(
        self,
        providers: Mapping[str, RetrievalProvider],
        *,
        known_providers: tuple[str, ...] = ("hybrid", "sql", "graph"),
        session_factory: Callable[[], Session] | None = None,
        candidate_fusion: CandidateFusion | None = None,
        reranker: Reranker | None = None,
        reranker_enabled: bool = False,
        reranker_top_k: int = 30,
        reranker_output_k: int | None = 8,
        max_context_tokens: int = 6000,
        non_executable_providers: tuple[str, ...] | None = None,
    ) -> None:
        self.known_providers = tuple(str(provider).strip().lower() for provider in known_providers)
        self.providers = {}
        self.non_executable_provider_names = frozenset(
            str(provider).strip().lower()
            for provider in (
                NON_EXECUTABLE_PROVIDER_NAMES
                if non_executable_providers is None
                else non_executable_providers
            )
            if str(provider).strip()
        )
        self.session_factory = session_factory
        self.candidate_fusion = candidate_fusion or CandidateFusion()
        self.reranker = reranker
        self.reranker_enabled = reranker_enabled
        self.reranker_top_k = reranker_top_k
        self.reranker_output_k = reranker_output_k
        self.max_context_tokens = max_context_tokens
        for name, provider in providers.items():
            if not name or provider is None:
                continue
            provider_name = str(name).strip().lower()
            if provider_name in RESERVED_INTERNAL_PROVIDER_NAMES:
                raise ValueError(
                    f"internal_lane_registered_as_provider:{provider_name}"
                )
            if provider_name in self.non_executable_provider_names:
                raise ValueError(f"non_executable_provider_registered:{provider_name}")
            if provider_name not in self.known_providers:
                raise ValueError(f"unknown_provider_registered:{provider_name}")
            if not isinstance(provider, RetrievalProvider):
                raise TypeError(
                    f"provider_must_inherit_retrieval_provider:{provider_name}"
                )
            actual_provider_name = str(
                getattr(provider, "provider_name", "")
            ).strip().lower()
            if actual_provider_name != provider_name:
                raise ValueError(
                    f"provider_name_mismatch:{provider_name}:{actual_provider_name or '<missing>'}"
                )
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
        if self.session_factory is None:
            return self._retrieve_sync(
                db,
                query=query,
                top_k=top_k,
                filters=filters,
                options=options,
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
            )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.aretrieve(
                    db,
                    query=query,
                    top_k=top_k,
                    filters=filters,
                    options=options,
                    query_plan=query_plan,
                    retrieval_tasks=retrieval_tasks,
                )
            )
        raise RuntimeError(
            "ProviderRouter.retrieve cannot run inside an active event loop; "
            "call aretrieve instead."
        )

    async def aretrieve(
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
        if self.session_factory is None:
            raise RuntimeError(
                "ProviderRouter.aretrieve requires a session_factory for non-blocking "
                "provider execution; use retrieve for legacy synchronous providers."
            )
        started = time.perf_counter()
        effective_top_k = max(0, top_k)
        ordered_results: list[tuple[int, ProviderResult]] = []
        pending: list[
            tuple[int, str, list[RetrievalTask], float, asyncio.Task[ProviderResult]]
        ] = []
        provider_top_k = self._provider_top_k(effective_top_k, options)
        provider_options = self._provider_options(options)
        for order, (provider_name, tasks) in enumerate(_group_tasks(retrieval_tasks).items()):
            executable_tasks: list[RetrievalTask] = []
            for task in tasks:
                if _task_is_non_executable(task, self.non_executable_provider_names):
                    ordered_results.append(
                        (order, _skipped_result(task, known=provider_name in self.known_providers))
                    )
                    continue
                executable_tasks.append(task)
            if not executable_tasks:
                continue
            provider = self.providers.get(provider_name)
            if provider is None:
                ordered_results.extend(
                    (
                        order,
                        _provider_not_registered_result(
                            task,
                            known=provider_name in self.known_providers,
                        ),
                    )
                    for task in executable_tasks
                )
                continue
            provider_call = self._execute_provider(
                provider,
                db,
                query=query,
                top_k=provider_top_k,
                filters=filters,
                options=provider_options,
                query_plan=query_plan,
                retrieval_tasks=executable_tasks,
            )
            pending.append(
                (
                    order,
                    provider_name,
                    executable_tasks,
                    time.perf_counter(),
                    asyncio.create_task(provider_call),
                )
            )

        if pending:
            completed = await asyncio.gather(
                *(task for *_, task in pending),
                return_exceptions=True,
            )
            for pending_item, result in zip(pending, completed, strict=True):
                order, provider_name, executable_tasks, provider_started, _task = pending_item
                if isinstance(result, BaseException):
                    if isinstance(result, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                        raise result
                    ordered_results.append(
                        (
                            order,
                            _provider_failed_result(
                                provider_name,
                                executable_tasks,
                                result,
                                latency_ms=int((time.perf_counter() - provider_started) * 1000),
                            ),
                        )
                    )
                    continue
                ordered_results.append((order, _normalize_provider_result(provider_name, result)))

        return await asyncio.to_thread(
            self._assemble_result_with_owned_session,
            started=started,
            query=query,
            top_k=effective_top_k,
            options=options,
            query_plan=query_plan,
            retrieval_tasks=retrieval_tasks,
            ordered_results=ordered_results,
        )

    def _retrieve_sync(
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
        effective_top_k = max(0, top_k)
        ordered_results: list[tuple[int, ProviderResult]] = []
        provider_top_k = self._provider_top_k(effective_top_k, options)
        provider_options = self._provider_options(options)
        for order, (provider_name, tasks) in enumerate(_group_tasks(retrieval_tasks).items()):
            executable_tasks: list[RetrievalTask] = []
            for task in tasks:
                if _task_is_non_executable(task, self.non_executable_provider_names):
                    ordered_results.append(
                        (order, _skipped_result(task, known=provider_name in self.known_providers))
                    )
                    continue
                executable_tasks.append(task)
            if not executable_tasks:
                continue
            provider = self.providers.get(provider_name)
            if provider is None:
                ordered_results.extend(
                    (
                        order,
                        _provider_not_registered_result(
                            task,
                            known=provider_name in self.known_providers,
                        ),
                    )
                    for task in executable_tasks
                )
                continue
            provider_started = time.perf_counter()
            try:
                result = provider.retrieve_provider_result(
                    db,
                    query=query,
                    top_k=provider_top_k,
                    filters=filters,
                    options=provider_options,
                    query_plan=query_plan,
                    retrieval_tasks=executable_tasks,
                )
                result = _normalize_provider_result(provider_name, result)
            except Exception as exc:
                _rollback_failed_transaction(db)
                result = _provider_failed_result(
                    provider_name,
                    executable_tasks,
                    exc,
                    latency_ms=int((time.perf_counter() - provider_started) * 1000),
                )
            ordered_results.append((order, result))

        return self._assemble_result(
            db,
            started=started,
            query=query,
            top_k=effective_top_k,
            options=options,
            query_plan=query_plan,
            retrieval_tasks=retrieval_tasks,
            ordered_results=ordered_results,
        )

    def _assemble_result_with_owned_session(
        self,
        *,
        started: float,
        query: str,
        top_k: int,
        options: dict,
        query_plan: QueryPlan,
        retrieval_tasks: list[RetrievalTask],
        ordered_results: list[tuple[int, ProviderResult]],
    ) -> ProviderRouterResult:
        if self.session_factory is None:
            raise RuntimeError("session_factory_required")
        provider_db = self.session_factory()
        try:
            return self._assemble_result(
                provider_db,
                started=started,
                query=query,
                top_k=top_k,
                options=options,
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
                ordered_results=ordered_results,
            )
        finally:
            provider_db.close()

    def _assemble_result(
        self,
        db: Session,
        *,
        started: float,
        query: str,
        top_k: int,
        options: dict,
        query_plan: QueryPlan,
        retrieval_tasks: list[RetrievalTask],
        ordered_results: list[tuple[int, ProviderResult]],
    ) -> ProviderRouterResult:
        provider_results = [result for _, result in sorted(ordered_results, key=lambda item: item[0])]
        router_failure = None
        try:
            evidence, evidence_pack = self._build_global_evidence(
                db,
                query=query,
                top_k=top_k,
                options=options,
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
                provider_results=provider_results,
            )
        except Exception as exc:
            _rollback_failed_transaction(db)
            router_failure = _router_failed_result(
                exc,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            provider_results.append(router_failure)
            evidence = []
            evidence_pack = None

        latency_ms = int((time.perf_counter() - started) * 1000)
        trace = _router_trace(
            query_plan=query_plan,
            provider_results=provider_results,
            latency_ms=latency_ms,
            known_providers=self.known_providers,
            executable_providers=self.executable_providers,
            cross_provider_fusion=True,
        )
        if router_failure is not None:
            trace["status"] = "failed"
            trace["router_error"] = dict(router_failure.trace)
        return ProviderRouterResult(
            evidence=tuple(evidence),
            provider_results=tuple(provider_results),
            trace=trace,
            evidence_pack=evidence_pack,
        )

    async def _execute_provider(
        self,
        provider: RetrievalProvider,
        db: Session,
        *,
        query: str,
        top_k: int,
        filters: dict | None,
        options: dict,
        query_plan: QueryPlan,
        retrieval_tasks: list[RetrievalTask],
    ) -> ProviderResult:
        return await asyncio.to_thread(
            self._execute_provider_with_owned_session,
            provider,
            query=query,
            top_k=top_k,
            filters=filters,
            options=options,
            query_plan=query_plan,
            retrieval_tasks=retrieval_tasks,
        )

    def _execute_provider_with_owned_session(
        self,
        provider: RetrievalProvider,
        *,
        query: str,
        top_k: int,
        filters: dict | None,
        options: dict,
        query_plan: QueryPlan,
        retrieval_tasks: list[RetrievalTask],
    ) -> ProviderResult:
        if self.session_factory is None:
            raise RuntimeError("session_factory_required")
        provider_db = self.session_factory()
        try:
            return provider.retrieve_provider_result(
                provider_db,
                query=query,
                top_k=top_k,
                filters=filters,
                options=options,
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
            )
        finally:
            provider_db.close()

    def _build_global_evidence(
        self,
        db: Session,
        *,
        query: str,
        top_k: int,
        options: dict,
        query_plan: QueryPlan,
        retrieval_tasks: list[RetrievalTask],
        provider_results: list[ProviderResult],
    ) -> tuple[list[Any], Any | None]:
        fused_limit = self.reranker_top_k if self._reranker_enabled(options) else top_k
        fusion_window = max(top_k, fused_limit, self.reranker_top_k)
        candidates = self.candidate_fusion.fuse(provider_results, limit=fusion_window)
        max_blocks = top_k
        if self._reranker_enabled(options) and self.reranker is not None:
            ranked_output_k = (
                min(top_k, self.reranker_output_k)
                if self.reranker_output_k is not None
                else top_k
            )
            candidates = _rerank_ranked_text_candidates(
                reranker=self.reranker,
                query=query,
                candidates=candidates,
                reranker_top_k=fused_limit,
                ranked_output_k=ranked_output_k,
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
            )
            max_blocks = len(candidates) if top_k > 0 else 0
        else:
            max_blocks = _max_blocks_with_preserved_candidates(
                candidates,
                ranked_limit=top_k,
            )
        if not candidates:
            return [], None
        pack = build_evidence_pack_from_candidates(
            candidates,
            parent_resolver=_parent_resolver(db, candidates),
            max_context_tokens=self.max_context_tokens,
            max_blocks=max_blocks,
            plan_id=query_plan.plan_id,
        )
        return evidence_pack_to_evidence(pack), pack

    def _reranker_enabled(self, options: dict[str, Any]) -> bool:
        if "cross_provider_reranker_enabled" in options:
            return _truthy(options.get("cross_provider_reranker_enabled"))
        if "global_reranker_enabled" in options:
            return _truthy(options.get("global_reranker_enabled"))
        if "reranker_enabled" in options:
            return _truthy(options.get("reranker_enabled"))
        if "rerank" in options:
            return _truthy(options.get("rerank"))
        mode = str(options.get("retrieval_mode") or "").strip().lower()
        if mode in {"hybrid_rrf", "hybrid-rrf", "hybrid_no_rerank"}:
            return False
        return self.reranker_enabled

    def _provider_options(self, options: dict[str, Any]) -> dict[str, Any]:
        if not (self.reranker is not None and self._reranker_enabled(options)):
            return options
        provider_options = dict(options)
        provider_options["reranker_enabled"] = False
        provider_options["rerank"] = False
        provider_options["provider_local_reranker_disabled_reason"] = "cross_provider_reranker"
        return provider_options

    def _provider_top_k(self, top_k: int, options: dict[str, Any]) -> int:
        if top_k <= 0:
            return 0
        return max(top_k, self.reranker_top_k)


def _parent_resolver(db: Session, candidates: list[Any]):
    parent_ids = []
    for candidate in candidates:
        parent_id = getattr(candidate, "parent_id", None)
        if parent_id is None:
            parent_id = dict(getattr(candidate, "metadata", {}) or {}).get("parent_id")
        if parent_id:
            parent_ids.append(str(parent_id))
    if not hasattr(db, "scalars"):
        return {}
    return repositories.get_parent_blocks_by_ids(db, parent_ids)


def _rollback_session(db: Session) -> None:
    rollback = getattr(db, "rollback", None)
    if not callable(rollback):
        return
    try:
        rollback()
    except Exception:
        return


def _rollback_failed_transaction(db: Session) -> None:
    if not _session_transaction_failed(db):
        return
    _rollback_session(db)


def _session_transaction_failed(db: Session) -> bool:
    if getattr(db, "is_active", True) is False:
        return True
    get_transaction = getattr(db, "get_transaction", None)
    if not callable(get_transaction):
        return False
    try:
        transaction = get_transaction()
    except Exception:
        return False
    return transaction is not None and getattr(transaction, "is_active", True) is False


def _group_tasks(tasks: list[RetrievalTask]) -> dict[str, list[RetrievalTask]]:
    grouped: dict[str, list[RetrievalTask]] = {}
    for task in tasks:
        grouped.setdefault(str(task.provider).strip().lower(), []).append(task)
    return grouped


def _task_is_non_executable(
    task: RetrievalTask,
    non_executable_provider_names: frozenset[str] = NON_EXECUTABLE_PROVIDER_NAMES,
) -> bool:
    if str(task.provider).strip().lower() in non_executable_provider_names:
        return True
    if task.provider_status == "skipped_non_executable":
        return True
    reason = task.unsupported_reason or ""
    return reason.startswith("provider_not_executable:") or reason.startswith(
        "provider_not_executable_in_v1:"
    )


def _skipped_result(task: RetrievalTask, *, known: bool) -> ProviderResult:
    reason = (
        task.unsupported_reason
        or (
            f"provider_not_executable:{task.provider}"
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


def _provider_not_registered_result(task: RetrievalTask, *, known: bool) -> ProviderResult:
    reason = (
        f"provider_not_registered:{task.provider}"
        if known
        else f"unknown_provider:{task.provider}"
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


def _provider_failed_result(
    provider_name: str,
    tasks: list[RetrievalTask],
    exc: BaseException,
    *,
    latency_ms: int,
) -> ProviderResult:
    error_message = " ".join(str(exc).split()) or exc.__class__.__name__
    trace: dict[str, Any] = {
        "provider": provider_name,
        "status": "failed",
        "reason": f"provider_failed:{provider_name}",
        "error_type": exc.__class__.__name__,
        "error_message": error_message,
        "task_ids": [task.task_id for task in tasks],
        "unit_ids": [task.unit_id for task in tasks],
        "planned_text": [task.query_text for task in tasks],
    }
    error_code = getattr(exc, "error_code", None)
    status_code = getattr(exc, "status_code", None)
    if error_code is not None:
        trace["error_code"] = str(getattr(error_code, "value", error_code))
    if status_code is not None:
        trace["status_code"] = status_code
    return ProviderResult(
        provider=provider_name,
        task_id=None if len(tasks) != 1 else tasks[0].task_id,
        unit_id=None if len(tasks) != 1 else tasks[0].unit_id,
        status="failed",
        candidates=(),
        latency_ms=latency_ms,
        reason=f"provider_failed:{provider_name}",
        trace=trace,
    )


def _normalize_provider_result(provider_name: str, result: ProviderResult) -> ProviderResult:
    if result.provider == provider_name:
        return result
    trace = dict(result.trace or {})
    trace.setdefault("provider_reported", result.provider)
    trace["provider"] = provider_name
    return replace(result, provider=provider_name, trace=trace)


def _router_failed_result(exc: Exception, *, latency_ms: int) -> ProviderResult:
    error_message = " ".join(str(exc).split()) or exc.__class__.__name__
    trace = {
        "provider": "router",
        "status": "failed",
        "reason": "router_assembly_failed",
        "error_type": exc.__class__.__name__,
        "error_message": error_message,
    }
    return ProviderResult(
        provider="router",
        task_id=None,
        unit_id=None,
        status="failed",
        candidates=(),
        latency_ms=latency_ms,
        reason="router_assembly_failed",
        trace=trace,
    )


def _router_trace(
    *,
    query_plan: QueryPlan,
    provider_results: list[ProviderResult],
    latency_ms: int,
    known_providers: tuple[str, ...],
    executable_providers: tuple[str, ...],
    cross_provider_fusion: bool = False,
) -> dict[str, Any]:
    return {
        "query_plan_id": query_plan.plan_id,
        "known_providers": list(known_providers),
        "executable_providers": list(executable_providers),
        "status": _router_status(provider_results),
        "latency_ms": latency_ms,
        "cross_provider_fusion": cross_provider_fusion,
        "provider_results": [serialize_provider_result(result) for result in provider_results],
    }


def serialize_provider_result(result: ProviderResult) -> dict[str, Any]:
    return {
        "provider": result.provider,
        "task_id": result.task_id,
        "unit_id": result.unit_id,
        "status": result.status,
        "candidate_count": len(result.candidates),
        "candidates": [
            _candidate_trace_payload(candidate, provider=result.provider)
            for candidate in result.candidates
        ],
        "evidence_count": len(result.evidence),
        "latency_ms": result.latency_ms,
        "reason": result.reason,
        "trace": dict(result.trace),
    }


def _candidate_trace_payload(candidate: Any, *, provider: str) -> dict[str, Any]:
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    raw_provider = getattr(candidate, "provider", None) or metadata.get("provider")
    reported_provider = (
        str(raw_provider)
        if raw_provider is not None and str(raw_provider).strip().lower() != provider
        else None
    )
    source_anchor = metadata.get("source_anchor")
    if not isinstance(source_anchor, dict):
        source_anchor = asdict(source_anchor_from_candidate(candidate))
    source_anchor = _sanitize_trace_source_anchor(source_anchor, provider=provider)
    lane_attributions = metadata.get("lane_attributions")
    if not isinstance(lane_attributions, list):
        lane_attributions = []
    lanes = metadata.get("lanes")
    if not isinstance(lanes, list):
        lanes = []
    return {
        "candidate_id": getattr(candidate, "candidate_id", None),
        "provider": provider,
        "reported_provider": reported_provider,
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
        "rerankable": getattr(candidate, "rerankable", metadata.get("rerankable", True)),
        "fusion_policy": getattr(candidate, "fusion_policy", metadata.get("fusion_policy", "ranked")),
        "structured_payload": getattr(
            candidate,
            "structured_payload",
            metadata.get("structured_payload", {}),
        ),
        "source_anchor": source_anchor,
    }


def _sanitize_trace_source_anchor(source_anchor: dict[str, Any], *, provider: str) -> dict[str, Any]:
    anchor = dict(source_anchor)
    anchor_metadata = dict(anchor.get("metadata") or {})
    reported_provider = None
    raw_provider = anchor_metadata.get("provider")
    if _provider_key(raw_provider) in _unsafe_provider_labels() and _provider_key(raw_provider) != provider:
        reported_provider = str(raw_provider)
    anchor_metadata["provider"] = provider
    if reported_provider is not None:
        anchor_metadata["reported_provider"] = reported_provider
    anchor["metadata"] = anchor_metadata
    return anchor


def _unsafe_provider_labels() -> set[str]:
    return {
        *NON_EXECUTABLE_QUERY_PROVIDERS,
        *RESERVED_INTERNAL_PROVIDER_NAMES,
    }


def _provider_key(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def _router_status(results: list[ProviderResult]) -> str:
    if not results:
        return "empty"
    if all(result.status == "empty" for result in results):
        return "empty"
    executed = any(result.status in {"executed", "success"} for result in results)
    empty = any(result.status == "empty" for result in results)
    skipped = any(result.status == "skipped_non_executable" for result in results)
    diagnostic_empty = any(
        result.status
        in {
            "skipped_not_table_query",
            "cannot_answer_no_table",
            "cannot_answer_low_confidence",
            "unsupported_multi_table",
        }
        for result in results
    )
    failed = any(
        result.status
        in {"failed", "compiler_failed", "validation_failed", "execution_failed", "timeout"}
        for result in results
    )
    if failed and executed:
        return "partial"
    if failed:
        return "failed"
    if (executed or empty) and (skipped or diagnostic_empty):
        return "partial"
    if executed or empty:
        return "executed"
    if diagnostic_empty:
        return "empty"
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


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled", "enable"}


def _rerank_ranked_text_candidates(
    *,
    reranker: Reranker,
    query: str,
    candidates: list[Any],
    reranker_top_k: int,
    ranked_output_k: int,
    query_plan: QueryPlan,
    retrieval_tasks: list[RetrievalTask],
) -> list[Any]:
    rerankable: list[Any] = []
    preserved: list[Any] = []
    for candidate in candidates:
        if _candidate_reranker_eligible(candidate):
            if len(rerankable) < reranker_top_k:
                rerankable.append(candidate)
            continue
        if _candidate_preserved_without_rerank(candidate):
            preserved.append(candidate)

    if ranked_output_k <= 0:
        return _rank_candidates_for_evidence(preserved)

    reranked = (
        rerank_with_context(
            reranker,
            query=query,
            candidates=rerankable,
            top_k=reranker_top_k,
            query_plan=query_plan,
            retrieval_tasks=retrieval_tasks,
            output_k=ranked_output_k,
        )[:ranked_output_k]
        if rerankable
        else []
    )
    return _rank_candidates_for_evidence([*reranked, *preserved])


def _candidate_reranker_eligible(candidate: Any) -> bool:
    return (
        _candidate_fusion_policy(candidate) == "ranked"
        and _candidate_rerankable(candidate)
        and _candidate_source_type(candidate) in TEXT_RERANK_SOURCE_TYPES
        and bool(str(getattr(candidate, "text", "") or "").strip())
    )


def _candidate_preserved_without_rerank(candidate: Any) -> bool:
    return (
        _candidate_fusion_policy(candidate) != "ranked"
        or not _candidate_rerankable(candidate)
        or _candidate_source_type(candidate) not in TEXT_RERANK_SOURCE_TYPES
    )


def _max_blocks_with_preserved_candidates(candidates: list[Any], *, ranked_limit: int) -> int:
    if ranked_limit <= 0:
        return 0
    preserved_count = sum(
        1 for candidate in candidates if _candidate_fusion_policy(candidate) != "ranked"
    )
    return ranked_limit + preserved_count


def _rank_candidates_for_evidence(candidates: list[Any]) -> list[Any]:
    ordered = sorted(enumerate(candidates), key=_policy_evidence_sort_key)
    ranked = []
    for rank, (_order, candidate) in enumerate(ordered, start=1):
        metadata = dict(getattr(candidate, "metadata", {}) or {})
        fusion_policy = _candidate_fusion_policy(candidate)
        rerankable = _candidate_rerankable(candidate)
        structured_payload = _candidate_structured_payload(candidate)
        metadata["final_rank"] = rank
        metadata["fusion_policy"] = fusion_policy
        metadata["rerankable"] = rerankable
        if structured_payload:
            metadata["structured_payload"] = structured_payload
        ranked.append(
            replace(
                candidate,
                final_rank=rank,
                metadata=metadata,
                fusion_policy=fusion_policy,
                rerankable=rerankable,
                structured_payload=structured_payload,
            )
        )
    return ranked


def _policy_evidence_sort_key(item: tuple[int, Any]) -> tuple[int, int, int]:
    order, candidate = item
    policy = _candidate_fusion_policy(candidate)
    if policy == "pinned":
        band = 0
    elif policy == "supporting":
        band = 2
    else:
        band = 1
    rank = (
        getattr(candidate, "rerank_rank", None)
        or getattr(candidate, "final_rank", None)
        or getattr(candidate, "fusion_rank", None)
        or 1_000_000_000
    )
    return (band, int(rank), order)


def _candidate_fusion_policy(candidate: Any) -> str:
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    text = str(
        metadata.get("fusion_policy")
        or getattr(candidate, "fusion_policy", None)
        or "ranked"
    ).strip().lower()
    if text in {"pinned", "supporting"}:
        return text
    return "ranked"


def _candidate_rerankable(candidate: Any) -> bool:
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    value = metadata.get("rerankable", getattr(candidate, "rerankable", True))
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled", "enable"}


def _candidate_source_type(candidate: Any) -> str:
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    return str(
        metadata.get("source_type")
        or getattr(candidate, "source_type", None)
        or "text_chunk"
    ).strip().lower()


def _candidate_structured_payload(candidate: Any) -> dict[str, Any]:
    payload = getattr(candidate, "structured_payload", None)
    if isinstance(payload, dict) and payload:
        return dict(payload)
    metadata_payload = dict(getattr(candidate, "metadata", {}) or {}).get("structured_payload")
    if isinstance(metadata_payload, dict):
        return dict(metadata_payload)
    return {}
