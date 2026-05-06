from dataclasses import dataclass, field, replace
import time
from typing import Any, Protocol

from sqlalchemy.orm import Session

from atlas.core.config import Settings
from atlas.core.errors import AtlasError, ErrorCode
from atlas.core.ids import new_id
from atlas.db import repositories
from atlas.ingestion.chunker import approx_token_count
from atlas.llm.base import AnswerGenerator
from atlas.query_orchestrator.schema import QueryPlan, serialize_query_plan
from atlas.query_orchestrator.service import QueryOrchestrator
from atlas.query_runtime.cache import CACHE_KEY_SCHEMA, QueryCacheStore, make_cache_key
from atlas.query_runtime.citation_builder import build_citations
from atlas.query_runtime.critic_lite import (
    CriticResult,
    post_generation_critic,
    pre_generation_critic,
)
from atlas.query_runtime.trace_logger import (
    get_query_trace_metadata,
    make_generation_event,
    make_query_run,
    make_retrieval_events,
    record_query_trace_metadata,
)
from atlas.retrieval.evidence import Evidence
from atlas.retrieval.retrieval_task import (
    RetrievalTask,
    serialize_retrieval_task,
    tasks_from_plan,
)


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    trace_id: str
    answer: str
    confidence: str
    citations: list[dict]
    details: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    def retrieve(
        self,
        db: Session,
        *,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[Evidence]:
        ...


class QueryRuntime:
    def __init__(
        self,
        *,
        settings: Settings,
        retriever: Retriever,
        generator: AnswerGenerator,
        orchestrator: QueryOrchestrator | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.generator = generator
        self.orchestrator = orchestrator

    def run(
        self,
        db: Session,
        *,
        query: str,
        top_k: int | None,
        filters: dict | None,
        options: dict | None = None,
    ) -> QueryResult:
        query_id = new_id("q")
        trace_id = new_id("tr")
        runtime_options = dict(options or {})
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise AtlasError(
                ErrorCode.INVALID_REQUEST,
                "Query must not be blank.",
                status_code=400,
            )
        requested_top_k = top_k if top_k is not None else _int_option(runtime_options, "top_k")
        effective_top_k = min(
            requested_top_k or self.settings.default_top_k,
            self.settings.max_top_k,
        )
        started = time.perf_counter()
        query_plan, retrieval_tasks, plan_latency_ms = _plan_query(
            self.orchestrator,
            normalized_query,
            runtime_options,
        )
        if query_plan is not None:
            runtime_options["query_plan"] = serialize_query_plan(query_plan)
            runtime_options["retrieval_tasks"] = [
                serialize_retrieval_task(task) for task in retrieval_tasks
            ]
        cache_started = time.perf_counter()
        cache_policy = _cache_policy(self.settings, runtime_options)
        cache_key = _cache_key(
            query=normalized_query,
            filters=filters or {},
            settings=self.settings,
            top_k=effective_top_k,
            options=runtime_options,
            cache_policy=cache_policy,
        )
        cache_latency_ms = int((time.perf_counter() - cache_started) * 1000)
        cache_status = cache_policy
        if cache_key is not None:
            cached = QueryCacheStore.get(db, cache_key)
            cache_latency_ms = int((time.perf_counter() - cache_started) * 1000)
            if cached is not None:
                cache_status = "hit"
                latency_ms = int((time.perf_counter() - started) * 1000)
                answer = str(cached.get("answer") or "")
                confidence = str(cached.get("confidence") or "unknown")
                citations = list(cached.get("citations") or [])
                _record_trace_metadata(
                    query_id=query_id,
                    cache_hit=True,
                    cache_key=cache_key,
                    cache_latency_ms=cache_latency_ms,
                    retrieval_latency_ms=0,
                    generation_latency_ms=0,
                    cache_status=cache_status,
                    retrieval_status="skipped",
                    generation_status="skipped",
                    settings=self.settings,
                    options=runtime_options,
                    effective_top_k=effective_top_k,
                    query_plan=query_plan,
                    retrieval_tasks=retrieval_tasks,
                    plan_latency_ms=plan_latency_ms,
                )
                query_run = make_query_run(
                    query_id=query_id,
                    trace_id=trace_id,
                    user_query=query,
                    normalized_query=normalized_query,
                    answer=answer,
                    confidence=confidence,
                    citations=citations,
                    settings=self.settings,
                    latency_ms=latency_ms,
                )
                _attach_query_details(query_run, {})
                repositories.add_query_trace(db, query_run, [], None)
                db.commit()
                details = _runtime_details(
                    query_plan=query_plan,
                    retrieval_tasks=retrieval_tasks,
                    plan_latency_ms=plan_latency_ms,
                )
                return QueryResult(query_id, trace_id, answer, confidence, citations, details)
            cache_status = "miss"

        retrieval_events = []
        retrieval_latency_ms: int | None = None
        retrieval_started = time.perf_counter()
        try:
            evidence = _retrieve_evidence(
                self.retriever,
                db,
                query=normalized_query,
                top_k=effective_top_k,
                filters=filters,
                options=runtime_options,
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
            )
            retrieval_latency_ms = int((time.perf_counter() - retrieval_started) * 1000)
            retrieval_events = make_retrieval_events(query_id=query_id, evidence=evidence)
        except AtlasError as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            if retrieval_latency_ms is None:
                retrieval_latency_ms = int((time.perf_counter() - retrieval_started) * 1000)
            _record_trace_metadata(
                query_id=query_id,
                cache_hit=False,
                cache_key=cache_key,
                cache_latency_ms=cache_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=0,
                cache_status=cache_status,
                retrieval_status="failed",
                generation_status="skipped",
                settings=self.settings,
                options=runtime_options,
                effective_top_k=effective_top_k,
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
                plan_latency_ms=plan_latency_ms,
            )
            query_run = make_query_run(
                query_id=query_id,
                trace_id=trace_id,
                user_query=query,
                normalized_query=normalized_query,
                answer=None,
                confidence="unknown",
                citations=[],
                settings=self.settings,
                latency_ms=latency_ms,
                error_message=exc.error_message,
            )
            repositories.add_query_trace(db, query_run, retrieval_events, None)
            db.commit()
            exc.trace_id = trace_id
            raise
        except Exception as exc:
            atlas_error = AtlasError(
                ErrorCode.UPSTREAM_VECTOR_STORE_UNAVAILABLE,
                "Dense retrieval failed.",
                status_code=502,
                details={"type": exc.__class__.__name__},
                trace_id=trace_id,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            if retrieval_latency_ms is None:
                retrieval_latency_ms = int((time.perf_counter() - retrieval_started) * 1000)
            _record_trace_metadata(
                query_id=query_id,
                cache_hit=False,
                cache_key=cache_key,
                cache_latency_ms=cache_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=0,
                cache_status=cache_status,
                retrieval_status="failed",
                generation_status="skipped",
                settings=self.settings,
                options=runtime_options,
                effective_top_k=effective_top_k,
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
                plan_latency_ms=plan_latency_ms,
            )
            query_run = make_query_run(
                query_id=query_id,
                trace_id=trace_id,
                user_query=query,
                normalized_query=normalized_query,
                answer=None,
                confidence="unknown",
                citations=[],
                settings=self.settings,
                latency_ms=latency_ms,
                error_message=atlas_error.error_message,
            )
            repositories.add_query_trace(db, query_run, retrieval_events, None)
            db.commit()
            raise atlas_error from exc

        if not evidence:
            empty_pack_details = _retriever_pack_details(self.retriever)
            latency_ms = int((time.perf_counter() - started) * 1000)
            answer = "当前导入的文档中没有检索到足够证据回答这个问题。"
            citations: list[dict] = []
            pre_critic = pre_generation_critic(normalized_query, [])
            details = _runtime_details(
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
                plan_latency_ms=plan_latency_ms,
                extra={
                    **_critic_details(pre_critic=pre_critic, post_critic=None),
                    **_retrieval_trace_details([]),
                    **empty_pack_details,
                },
            )
            _record_trace_metadata(
                query_id=query_id,
                cache_hit=False,
                cache_key=cache_key,
                cache_latency_ms=cache_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=0,
                cache_status=cache_status,
                retrieval_status="completed",
                generation_status="skipped",
                settings=self.settings,
                options=runtime_options,
                effective_top_k=effective_top_k,
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
                plan_latency_ms=plan_latency_ms,
            )
            query_run = make_query_run(
                query_id=query_id,
                trace_id=trace_id,
                user_query=query,
                normalized_query=normalized_query,
                answer=answer,
                confidence="insufficient",
                citations=citations,
                settings=self.settings,
                latency_ms=latency_ms,
            )
            _attach_query_details(query_run, details)
            repositories.add_query_trace(db, query_run, retrieval_events, None)
            db.commit()
            return QueryResult(
                query_id,
                trace_id,
                answer,
                "insufficient",
                citations,
                details,
            )

        generation_started = time.perf_counter()
        generation_evidence = _fit_evidence_to_budget(evidence, self.settings.max_context_tokens)
        pre_critic = pre_generation_critic(normalized_query, generation_evidence)
        if pre_critic.status == "insufficient":
            latency_ms = int((time.perf_counter() - started) * 1000)
            answer = "当前导入的文档中没有检索到足够证据回答这个问题。"
            citations = []
            confidence = _critic_confidence("insufficient", pre_critic.confidence_override, None)
            details = _runtime_details(
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
                plan_latency_ms=plan_latency_ms,
                extra={
                    **_critic_details(pre_critic=pre_critic, post_critic=None),
                    **_retrieval_trace_details(generation_evidence),
                },
            )
            _record_trace_metadata(
                query_id=query_id,
                cache_hit=False,
                cache_key=cache_key,
                cache_latency_ms=cache_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=0,
                cache_status=cache_status,
                retrieval_status="completed",
                generation_status="skipped",
                settings=self.settings,
                options=runtime_options,
                effective_top_k=effective_top_k,
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
                plan_latency_ms=plan_latency_ms,
            )
            query_run = make_query_run(
                query_id=query_id,
                trace_id=trace_id,
                user_query=query,
                normalized_query=normalized_query,
                answer=answer,
                confidence=confidence,
                citations=citations,
                settings=self.settings,
                latency_ms=latency_ms,
            )
            _attach_query_details(query_run, details)
            repositories.add_query_trace(db, query_run, retrieval_events, None)
            db.commit()
            return QueryResult(query_id, trace_id, answer, confidence, citations, details)

        try:
            generated = self.generator.generate(
                query=normalized_query,
                evidence=generation_evidence,
            )
            generation_latency_ms = int((time.perf_counter() - generation_started) * 1000)
            citations = build_citations(
                generated.answer,
                generation_evidence,
                confidence=generated.confidence,
            )
            answer = generated.answer
            post_critic = post_generation_critic(
                normalized_query,
                answer,
                generation_evidence,
                citations,
            )
            details = _runtime_details(
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
                plan_latency_ms=plan_latency_ms,
                extra={
                    **_critic_details(pre_critic=pre_critic, post_critic=post_critic),
                    **_retrieval_trace_details(generation_evidence),
                },
            )
            confidence = _critic_confidence(
                generated.confidence,
                pre_critic.confidence_override,
                post_critic.confidence_override,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            _record_trace_metadata(
                query_id=query_id,
                cache_hit=False,
                cache_key=cache_key,
                cache_latency_ms=cache_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                cache_status=cache_status,
                retrieval_status="completed",
                generation_status="completed",
                settings=self.settings,
                options=runtime_options,
                effective_top_k=effective_top_k,
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
                plan_latency_ms=plan_latency_ms,
            )
            query_run = make_query_run(
                query_id=query_id,
                trace_id=trace_id,
                user_query=query,
                normalized_query=normalized_query,
                answer=answer,
                confidence=confidence,
                citations=citations,
                settings=self.settings,
                latency_ms=latency_ms,
            )
            _attach_query_details(query_run, details)
            generation_event = make_generation_event(
                query_id=query_id,
                settings=self.settings,
                generated=generated,
                latency_ms=generation_latency_ms,
                status="completed",
            )
            repositories.add_query_trace(db, query_run, retrieval_events, generation_event)
            if cache_key is not None and confidence == "supported":
                QueryCacheStore.set(
                    db,
                    cache_key,
                    {
                        "answer": answer,
                        "confidence": confidence,
                        "citations": citations,
                    },
                    metadata={
                        "trace_id": trace_id,
                        "query_id": query_id,
                        "cache_schema": CACHE_KEY_SCHEMA,
                    },
                    ttl_seconds=self.settings.cache_ttl_seconds,
                )
            db.commit()
            return QueryResult(
                query_id=query_id,
                trace_id=trace_id,
                answer=answer,
                confidence=confidence,
                citations=citations,
                details=details,
            )
        except AtlasError as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            generation_latency_ms = int((time.perf_counter() - generation_started) * 1000)
            _record_trace_metadata(
                query_id=query_id,
                cache_hit=False,
                cache_key=cache_key,
                cache_latency_ms=cache_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                cache_status=cache_status,
                retrieval_status="completed",
                generation_status="failed",
                settings=self.settings,
                options=runtime_options,
                effective_top_k=effective_top_k,
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
                plan_latency_ms=plan_latency_ms,
            )
            query_run = make_query_run(
                query_id=query_id,
                trace_id=trace_id,
                user_query=query,
                normalized_query=normalized_query,
                answer=None,
                confidence="unknown",
                citations=[],
                settings=self.settings,
                latency_ms=latency_ms,
                error_message=exc.error_message,
            )
            generation_event = make_generation_event(
                query_id=query_id,
                settings=self.settings,
                generated=None,
                latency_ms=None,
                status="failed",
                error_message=exc.error_message,
            )
            repositories.add_query_trace(db, query_run, retrieval_events, generation_event)
            db.commit()
            exc.trace_id = trace_id
            raise


def _fit_evidence_to_budget(evidence: list, max_tokens: int) -> list:
    if max_tokens <= 0:
        return []

    selected = []
    used_tokens = 0
    for item in evidence:
        token_count = max(1, item.token_count)
        if used_tokens + token_count <= max_tokens:
            selected.append(item)
            used_tokens += token_count
            continue

        remaining = max_tokens - used_tokens
        if remaining > 0:
            text = _truncate_text_to_budget(item.text, remaining)
            if text:
                selected.append(replace(item, text=text, token_count=approx_token_count(text)))
        break

    return selected


def _retrieve_evidence(
    retriever: Retriever,
    db: Session,
    *,
    query: str,
    top_k: int,
    filters: dict | None,
    options: dict,
    query_plan: QueryPlan | None = None,
    retrieval_tasks: list[RetrievalTask] | None = None,
) -> list[Evidence]:
    retrieve_with_plan = getattr(retriever, "retrieve_with_plan", None)
    if callable(retrieve_with_plan) and query_plan is not None:
        return retrieve_with_plan(
            db,
            query=query,
            top_k=top_k,
            filters=filters,
            options=options,
            query_plan=query_plan,
            retrieval_tasks=retrieval_tasks or [],
        )
    retrieve_with_options = getattr(retriever, "retrieve_with_options", None)
    if callable(retrieve_with_options):
        return retrieve_with_options(
            db,
            query=query,
            top_k=top_k,
            filters=filters,
            options=options,
        )
    return retriever.retrieve(db, query=query, top_k=top_k, filters=filters)


def _truncate_text_to_budget(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    if approx_token_count(text) <= token_budget:
        return text

    source_tokens = approx_token_count(text)
    limit = max(1, int(len(text) * token_budget / source_tokens))
    candidate = text[:limit].rstrip()
    while candidate and approx_token_count(candidate) > token_budget:
        limit = max(1, int(limit * 0.8))
        next_candidate = text[:limit].rstrip()
        if next_candidate == candidate:
            break
        candidate = next_candidate
    return candidate


def _critic_confidence(
    generated_confidence: str,
    pre_override: str | None,
    post_override: str | None,
) -> str:
    for value in (post_override, pre_override, generated_confidence):
        if value:
            return value
    return generated_confidence


def _critic_details(
    *,
    pre_critic: CriticResult,
    post_critic: CriticResult | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "pre": pre_critic.to_dict(),
        "post": post_critic.to_dict() if post_critic else None,
        "status": _combined_critic_status(pre_critic, post_critic),
        "warnings": _dedupe([*pre_critic.warnings, *(post_critic.warnings if post_critic else [])]),
        "reasons": _dedupe([*pre_critic.reasons, *(post_critic.reasons if post_critic else [])]),
    }
    override = post_critic.confidence_override if post_critic else None
    if override is None:
        override = pre_critic.confidence_override
    payload["confidence_override"] = override
    return {"critic": payload}


def _runtime_details(
    *,
    query_plan: QueryPlan | None,
    retrieval_tasks: list[RetrievalTask],
    plan_latency_ms: int | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    details = dict(extra or {})
    if query_plan is not None:
        details["query_plan"] = serialize_query_plan(query_plan)
        details["retrieval_tasks"] = [
            serialize_retrieval_task(task) for task in retrieval_tasks
        ]
        details["plan_latency_ms"] = plan_latency_ms
    return details


def _retrieval_trace_details(evidence: list[Evidence]) -> dict[str, Any]:
    return {
        "retrieval_trace": {
            "evidence_count": len(evidence),
            "top_k": [_evidence_trace_item(item) for item in evidence],
        }
    }


def _retriever_pack_details(retriever: object) -> dict[str, Any]:
    pack = getattr(retriever, "last_evidence_pack", None)
    if pack is None:
        return {}
    return {"evidence_pack": _evidence_pack_summary(pack)}


def _evidence_pack_summary(pack: object) -> dict[str, Any]:
    blocks = tuple(getattr(pack, "blocks", ()) or ())
    dropped_blocks = tuple(getattr(pack, "dropped_blocks", ()) or ())
    return {
        "pack_id": getattr(pack, "pack_id", None),
        "token_count": getattr(pack, "token_count", None),
        "max_context_tokens": getattr(pack, "max_context_tokens", None),
        "block_count": len(blocks),
        "dropped_block_count": len(dropped_blocks),
        "dropped_blocks": [
            {
                "evidence_id": getattr(block, "evidence_id", None),
                "chunk_ids": list(getattr(block, "chunk_ids", ()) or ()),
                "parent_id": getattr(block, "parent_id", None),
                "rank": getattr(block, "rank", None),
                "token_count": getattr(block, "token_count", None),
                "drop_reason": getattr(block, "drop_reason", None),
                "drop_stage": getattr(block, "drop_stage", None),
                "coverage": getattr(block, "coverage", None),
            }
            for block in dropped_blocks
        ],
        "metadata": dict(getattr(pack, "metadata", {}) or {}),
    }


def _evidence_trace_item(item: Evidence) -> dict[str, Any]:
    metadata = item.metadata or {}
    return {
        "evidence_id": item.evidence_id,
        "chunk_id": item.chunk_id,
        "rank": item.rank,
        "retrieval_score": item.retrieval_score,
        "retrieved_by": list(item.retrieved_by or metadata.get("retrieved_by") or ()),
        "provider": metadata.get("provider") or metadata.get("retrieval_provider"),
        "lane": metadata.get("lane"),
        "lanes": list(metadata.get("lanes") or ()),
        "lane_attributions": list(metadata.get("lane_attributions") or ()),
        "lane_contributions": list(
            metadata.get("lane_contributions")
            or (
                metadata.get("fusion", {}).get("lane_contributions")
                if isinstance(metadata.get("fusion"), dict)
                else ()
            )
            or ()
        ),
        "weighted_contribution": metadata.get("weighted_contribution"),
        "retrieval_task_id": metadata.get("retrieval_task_id"),
        "retrieval_unit_id": metadata.get("retrieval_unit_id"),
        "fusion_rank": metadata.get("fusion_rank") or metadata.get("best_fusion_rank"),
        "fusion_score": metadata.get("fusion_score") or metadata.get("best_fusion_score"),
        "rerank_rank": metadata.get("rerank_rank") or metadata.get("best_rerank_rank"),
        "rerank_score": metadata.get("rerank_score") or metadata.get("best_rerank_score"),
        "reranker": metadata.get("reranker"),
        "reranker_input": metadata.get("reranker_input"),
        "evidence_pack": metadata.get("evidence_pack"),
        "coverage": metadata.get("coverage"),
        "included_in_prompt": metadata.get("included_in_prompt"),
        "drop_reason": metadata.get("drop_reason"),
    }


def _combined_critic_status(
    pre_critic: CriticResult,
    post_critic: CriticResult | None,
) -> str:
    statuses = [pre_critic.status, post_critic.status if post_critic else None]
    for status in ("unsupported", "insufficient", "warning"):
        if status in statuses:
            return status
    return "ok"


def _attach_query_details(query_run: object, details: dict[str, Any]) -> None:
    combined = dict(details or {})
    query_id = getattr(query_run, "query_id", None)
    if query_id:
        trace_metadata = get_query_trace_metadata(str(query_id))
        if trace_metadata:
            combined["trace"] = trace_metadata
    if not combined:
        return
    existing = getattr(query_run, "details_json", None)
    if isinstance(existing, dict):
        setattr(query_run, "details_json", {**existing, **combined})
        return
    setattr(query_run, "details_json", combined)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _cache_key(
    *,
    query: str,
    filters: dict,
    settings: Settings,
    top_k: int,
    options: dict,
    cache_policy: str,
) -> str | None:
    if cache_policy != "enabled":
        return None
    return make_cache_key(
        query=query,
        filters=filters,
        settings=settings,
        top_k=top_k,
        options=options,
    )


def _cache_policy(settings: Settings, options: dict[str, Any]) -> str:
    explicit_policy = options.get("cache_policy")
    if explicit_policy is not None:
        value = str(explicit_policy).strip().lower()
        if value in {"bypass", "disabled", "disable", "off", "none", "no-cache"}:
            return "bypassed" if value == "bypass" or value == "no-cache" else "disabled"
        if value in {"enabled", "enable", "on", "readwrite", "write-through"}:
            return "enabled"

    if _truthy(options.get("cache_bypass")) or _truthy(options.get("bypass_cache")):
        return "bypassed"
    if "use_cache" in options:
        return "enabled" if _truthy(options.get("use_cache")) else "disabled"
    if "cache_enabled" in options:
        return "enabled" if _truthy(options.get("cache_enabled")) else "disabled"
    return "enabled" if settings.cache_enabled else "disabled"


def _record_trace_metadata(
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
    settings: Settings,
    options: dict[str, Any],
    effective_top_k: int,
    query_plan: QueryPlan | None,
    retrieval_tasks: list[RetrievalTask],
    plan_latency_ms: int | None,
) -> None:
    record_query_trace_metadata(
        query_id=query_id,
        cache_hit=cache_hit,
        cache_key=cache_key,
        cache_latency_ms=cache_latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        cache_status=cache_status,
        retrieval_status=retrieval_status,
        generation_status=generation_status,
        metadata={
            "cache_policy": _cache_policy(settings, options),
            "cache_status": cache_status,
            "retrieval_mode": options.get("retrieval_mode", settings.retrieval_mode),
            "effective_top_k": effective_top_k,
            "prompt_version": options.get("prompt_version", settings.prompt_version),
            "llm_model": options.get("llm_model", settings.llm_model),
            "corpus_version": options.get(
                "corpus_version",
                getattr(settings, "corpus_version", None),
            ),
            "query_plan": serialize_query_plan(query_plan) if query_plan else None,
            "retrieval_tasks": [
                serialize_retrieval_task(task) for task in retrieval_tasks
            ],
            "plan_latency_ms": plan_latency_ms,
        },
    )


def _plan_query(
    orchestrator: QueryOrchestrator | None,
    query: str,
    options: dict[str, Any],
) -> tuple[QueryPlan | None, list[RetrievalTask], int | None]:
    if orchestrator is None or _truthy(options.get("disable_query_plan")):
        return None, [], None
    use_llm = not _truthy(options.get("query_plan_fallback_only"))
    started = time.perf_counter()
    plan = orchestrator.plan(query, use_llm=use_llm)
    latency_ms = int((time.perf_counter() - started) * 1000)
    return plan, tasks_from_plan(plan), latency_ms


def _int_option(options: dict[str, Any], name: str) -> int | None:
    value = options.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "enabled", "enable"}
    return bool(value)
