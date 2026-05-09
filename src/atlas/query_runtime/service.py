import asyncio
from dataclasses import dataclass, field, replace
import inspect
import time
from typing import Any, Protocol

from sqlalchemy.orm import Session

from atlas.core.config import Settings, executable_query_providers, known_query_providers
from atlas.core.errors import AtlasError, ErrorCode
from atlas.core.ids import new_id
from atlas.db import repositories
from atlas.ingestion.chunker import approx_token_count
from atlas.llm.base import AnswerGenerator
from atlas.llm.prompts import ANSWER_INSTRUCTIONS, build_answer_input
from atlas.query_orchestrator.schema import QueryPlan, RetrievalUnit, serialize_query_plan
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
from atlas.retrieval.models.evidence import Evidence
from atlas.retrieval.models.retrieval_task import (
    RetrievalTask,
    serialize_retrieval_task,
    tasks_from_plan,
)
from atlas.retrieval.contracts import ProviderResult, ProviderRouterResult
from atlas.retrieval.router import ProviderRouter, serialize_provider_result
from atlas.retrieval.providers.base import RetrievalContext, RetrievalProvider


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


class _LegacyRetrieverProvider(RetrievalProvider):
    provider_name = "hybrid"

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    async def aretrieve_candidates(self, context: RetrievalContext) -> ProviderResult:
        return self.retrieve_provider_result(
            context.db,
            query=context.query,
            top_k=context.top_k,
            filters=context.filters,
            options=context.options,
            query_plan=context.query_plan,
            retrieval_tasks=context.retrieval_tasks,
        )

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
        started = time.perf_counter()
        evidence = tuple(
            _retrieve_evidence(
                self.retriever,
                db,
                query=query,
                top_k=top_k,
                filters=filters,
                options=options,
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
            )
        )
        return ProviderResult(
            provider="hybrid",
            task_id=None if len(retrieval_tasks) != 1 else retrieval_tasks[0].task_id,
            unit_id=None if len(retrieval_tasks) != 1 else retrieval_tasks[0].unit_id,
            status="executed" if evidence else "empty",
            evidence=evidence,
            latency_ms=int((time.perf_counter() - started) * 1000),
            trace={
                "provider": "hybrid",
                "status": "executed" if evidence else "empty",
                "legacy_adapter": True,
            },
        )


class QueryRuntime:
    def __init__(
        self,
        *,
        settings: Settings,
        generator: AnswerGenerator,
        retriever: Retriever | None = None,
        provider_router: ProviderRouter | None = None,
        orchestrator: QueryOrchestrator | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        providers = {}
        injected_retriever = retriever is not None
        if provider_router is None:
            executable_providers = executable_query_providers(settings)
            if injected_retriever:
                executable_providers = ("hybrid",)
            for provider_name in executable_providers:
                if retriever is not None and provider_name == "hybrid":
                    providers["hybrid"] = (
                        retriever
                        if isinstance(retriever, RetrievalProvider)
                        else _LegacyRetrieverProvider(retriever)
                    )
                    continue
                providers[provider_name] = _auto_wire_provider(settings, provider_name)
        router_reranker = None
        if provider_router is None and settings.reranker_enabled:
            from atlas.backends import BackendBuildContext, build_reranker

            router_reranker = build_reranker(
                settings.reranker_backend,
                BackendBuildContext(settings=settings),
            )
        if provider_router is None:
            from atlas.db.session import SessionLocal
            router_session_factory = SessionLocal
            if injected_retriever:
                router_session_factory = None
        self.provider_router = provider_router or ProviderRouter(
            providers,
            known_providers=_runtime_known_providers(settings, include_hybrid=injected_retriever),
            session_factory=router_session_factory,
            reranker=router_reranker,
            reranker_enabled=settings.reranker_enabled,
            reranker_top_k=settings.reranker_top_k,
            reranker_output_k=settings.reranker_output_k,
            max_context_tokens=settings.max_context_tokens,
        )
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
        query_plan, retrieval_tasks, plan_latency_ms, planner_observability = _plan_query(
            self.orchestrator,
            normalized_query,
            runtime_options,
            self.provider_router.executable_providers,
        )
        if query_plan is not None:
            runtime_options["query_plan"] = serialize_query_plan(query_plan)
            runtime_options["retrieval_tasks"] = [
                serialize_retrieval_task(task) for task in retrieval_tasks
            ]
            runtime_options["executable_providers"] = list(
                self.provider_router.executable_providers
            )
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
                details = _runtime_details(
                    query_plan=query_plan,
                    retrieval_tasks=retrieval_tasks,
                    plan_latency_ms=plan_latency_ms,
                    extra=_llm_io_skipped_details("cache_hit"),
                )
                _attach_query_details(query_run, details)
                repositories.add_query_trace(
                    db,
                    query_run,
                    [],
                    None,
                    observability=planner_observability,
                )
                db.commit()
                return QueryResult(query_id, trace_id, answer, confidence, citations, details)
            cache_status = "miss"

        retrieval_events = []
        retrieval_latency_ms: int | None = None
        router_result: ProviderRouterResult | None = None
        retrieval_started = time.perf_counter()
        try:
            router_result = _retrieve_with_router(
                self.provider_router,
                db,
                query=normalized_query,
                top_k=effective_top_k,
                filters=filters,
                options=runtime_options,
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
            )
            evidence = list(router_result.evidence)
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
            details = _runtime_details(
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
                plan_latency_ms=plan_latency_ms,
                extra=_llm_io_skipped_details("retrieval_failed"),
            )
            _attach_query_details(query_run, details)
            repositories.add_query_trace(
                db,
                query_run,
                retrieval_events,
                None,
                observability=planner_observability,
            )
            db.commit()
            exc.trace_id = trace_id
            raise
        except Exception as exc:
            atlas_error = AtlasError(
                ErrorCode.UPSTREAM_VECTOR_STORE_UNAVAILABLE,
                "Provider retrieval failed.",
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
            details = _runtime_details(
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
                plan_latency_ms=plan_latency_ms,
                extra=_llm_io_skipped_details("retrieval_failed"),
            )
            _attach_query_details(query_run, details)
            repositories.add_query_trace(
                db,
                query_run,
                retrieval_events,
                None,
                observability=planner_observability,
            )
            db.commit()
            raise atlas_error from exc

        if not evidence:
            empty_pack_details = _provider_router_details(router_result)
            if _provider_router_failed(router_result):
                atlas_error = AtlasError(
                    ErrorCode.UPSTREAM_VECTOR_STORE_UNAVAILABLE,
                    "Provider retrieval failed.",
                    status_code=502,
                    details=_provider_failure_details(router_result),
                    trace_id=trace_id,
                )
                latency_ms = int((time.perf_counter() - started) * 1000)
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
                details = _runtime_details(
                    query_plan=query_plan,
                    retrieval_tasks=retrieval_tasks,
                    plan_latency_ms=plan_latency_ms,
                    extra={
                        **_retrieval_trace_details([]),
                        **empty_pack_details,
                        **_llm_io_skipped_details("retrieval_failed"),
                    },
                )
                _attach_query_details(query_run, details)
                repositories.add_query_trace(
                    db,
                    query_run,
                    retrieval_events,
                    None,
                    observability=planner_observability,
                )
                db.commit()
                raise atlas_error
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
                    **_llm_io_skipped_details("no_evidence"),
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
            repositories.add_query_trace(
                db,
                query_run,
                retrieval_events,
                None,
                observability=planner_observability,
            )
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
        prompt_evidence = _dedupe_prompt_evidence_by_chunk_id(evidence)
        generation_evidence = _fit_evidence_to_budget(
            prompt_evidence,
            self.settings.max_context_tokens,
        )
        pre_critic = pre_generation_critic(normalized_query, generation_evidence)
        if pre_critic.status in {"insufficient", "contradicted"}:
            latency_ms = int((time.perf_counter() - started) * 1000)
            answer = (
                "当前导入的文档中存在互相冲突的证据，暂时不能生成可靠答案。"
                if pre_critic.status == "contradicted"
                else "当前导入的文档中没有检索到足够证据回答这个问题。"
            )
            citations = []
            confidence = _critic_confidence(
                pre_critic.status,
                pre_critic.confidence_override,
                None,
            )
            details = _runtime_details(
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
                plan_latency_ms=plan_latency_ms,
                extra={
                    **_critic_details(pre_critic=pre_critic, post_critic=None),
                    **_retrieval_trace_details(generation_evidence),
                    **_provider_router_details(router_result),
                    **_llm_io_skipped_details(
                        f"pre_generation_critic_{pre_critic.status}"
                    ),
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
            repositories.add_query_trace(
                db,
                query_run,
                retrieval_events,
                None,
                observability=planner_observability,
            )
            db.commit()
            return QueryResult(query_id, trace_id, answer, confidence, citations, details)

        answer_llm_call_id = new_id("llmc")
        llm_io_request = _llm_io_request(
            settings=self.settings,
            query=normalized_query,
            evidence=generation_evidence,
        )
        llm_io_request_metadata = _llm_io_request_metadata(
            settings=self.settings,
            evidence=generation_evidence,
        )
        llm_started = time.perf_counter()
        try:
            generated = self.generator.generate(
                query=normalized_query,
                evidence=generation_evidence,
            )
            llm_latency_ms = int((time.perf_counter() - llm_started) * 1000)
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
                    **_provider_router_details(router_result),
                    **_llm_io_completed_details(
                        answer_llm_call_id=answer_llm_call_id,
                    ),
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
            answer_observability = _answer_llm_observability_payload(
                call_id=answer_llm_call_id,
                status="completed",
                request=llm_io_request,
                request_metadata=llm_io_request_metadata,
                generated=generated,
                evidence=generation_evidence,
                latency_ms=llm_latency_ms,
                error_message=None,
            )
            repositories.add_query_trace(
                db,
                query_run,
                retrieval_events,
                generation_event,
                observability=_merge_observability(
                    planner_observability,
                    answer_observability,
                ),
            )
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
            llm_latency_ms = int((time.perf_counter() - llm_started) * 1000)
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
            details = _runtime_details(
                query_plan=query_plan,
                retrieval_tasks=retrieval_tasks,
                plan_latency_ms=plan_latency_ms,
                extra={
                    **_critic_details(pre_critic=pre_critic, post_critic=None),
                    **_retrieval_trace_details(generation_evidence),
                    **_provider_router_details(router_result),
                    **_llm_io_failed_details(
                        answer_llm_call_id=answer_llm_call_id,
                        error_message=exc.error_message,
                    ),
                },
            )
            _attach_query_details(query_run, details)
            answer_observability = _answer_llm_observability_payload(
                call_id=answer_llm_call_id,
                status="failed",
                request=llm_io_request,
                request_metadata=llm_io_request_metadata,
                generated=None,
                evidence=generation_evidence,
                latency_ms=llm_latency_ms,
                error_message=exc.error_message,
            )
            repositories.add_query_trace(
                db,
                query_run,
                retrieval_events,
                generation_event,
                observability=_merge_observability(
                    planner_observability,
                    answer_observability,
                ),
            )
            db.commit()
            exc.trace_id = trace_id
            raise

    async def arun(
        self,
        db: Session | None = None,
        *,
        query: str,
        top_k: int | None,
        filters: dict | None,
        options: dict | None = None,
    ) -> QueryResult:
        if db is not None:
            raise RuntimeError(
                "QueryRuntime.arun cannot use a caller-owned sync Session. "
                "Call run(...) with that session, or call arun(...) without db "
                "so the runtime owns the session inside the worker thread."
            )
        return await asyncio.to_thread(
            self._run_with_owned_session,
            query=query,
            top_k=top_k,
            filters=filters,
            options=options,
        )

    def _run_with_owned_session(
        self,
        *,
        query: str,
        top_k: int | None,
        filters: dict | None,
        options: dict | None = None,
    ) -> QueryResult:
        from atlas.db.session import SessionLocal

        db = SessionLocal()
        try:
            return self.run(
                db,
                query=query,
                top_k=top_k,
                filters=filters,
                options=options,
            )
        finally:
            db.close()


def _auto_wire_provider(settings: Settings, provider_name: str):
    from atlas.backends import (
        BackendBuildContext,
        build_embedder,
        build_graph_store,
        build_reranker,
        build_sparse_encoder,
        build_vector_store,
    )
    from atlas.retrieval.providers.registry import ProviderBuildContext, build_provider

    backend_context = BackendBuildContext(settings=settings)
    context = ProviderBuildContext(
        settings=settings,
        qdrant_factory=lambda: build_vector_store(settings.vector_store_backend, backend_context),
        embedder_factory=lambda: build_embedder(settings.embedding_backend, backend_context),
        sparse_encoder_factory=lambda: build_sparse_encoder(settings.sparse_backend, backend_context),
        reranker_factory=lambda: build_reranker(settings.reranker_backend, backend_context),
        graph_store_factory=lambda: build_graph_store(settings.graph_store_backend, backend_context),
    )
    return build_provider(provider_name, context)


def _dedupe_prompt_evidence_by_chunk_id(evidence: list[Evidence]) -> list[Evidence]:
    selected: list[Evidence] = []
    index_by_key: dict[str, int] = {}
    for item in evidence:
        key = item.chunk_id or item.evidence_id
        if key in index_by_key:
            index = index_by_key[key]
            selected[index] = _merge_prompt_evidence(selected[index], item, key)
            continue
        index_by_key[key] = len(selected)
        selected.append(_with_prompt_provenance(item, key))
    return selected


def _merge_prompt_evidence(base: Evidence, duplicate: Evidence, key: str) -> Evidence:
    metadata = dict(base.metadata or {})
    duplicate_metadata = dict(duplicate.metadata or {})
    provenance = _dedupe_provenance(
        [
            *_prompt_provider_provenance(metadata, base),
            *_prompt_provider_provenance(duplicate_metadata, duplicate),
        ]
    )
    evidence_ids = _dedupe(
        [
            *[str(value) for value in metadata.get("prompt_deduped_evidence_ids") or ()],
            base.evidence_id,
            duplicate.evidence_id,
        ]
    )
    provider_names = _prompt_provider_names_from_provenance(provenance)
    retrieved_by = _dedupe(
        [
            *list(base.retrieved_by or ()),
            *list(duplicate.retrieved_by or ()),
            *_as_text_list(metadata.get("retrieved_by")),
            *_as_text_list(duplicate_metadata.get("retrieved_by")),
            *provider_names,
        ]
    )
    metadata.update(
        {
            "prompt_dedupe_key": key,
            "prompt_deduped": True,
            "prompt_duplicate_count": int(metadata.get("prompt_duplicate_count") or 0) + 1,
            "prompt_deduped_evidence_ids": evidence_ids,
            "prompt_provider_provenance": provenance,
            "prompt_providers": provider_names,
            "retrieved_by": retrieved_by,
        }
    )
    return replace(base, metadata=metadata, retrieved_by=tuple(retrieved_by))


def _with_prompt_provenance(item: Evidence, key: str) -> Evidence:
    metadata = dict(item.metadata or {})
    provenance = _dedupe_provenance(_prompt_provider_provenance(metadata, item))
    provider_names = _prompt_provider_names_from_provenance(provenance)
    retrieved_by = _dedupe(
        [
            *list(item.retrieved_by or ()),
            *_as_text_list(metadata.get("retrieved_by")),
            *provider_names,
        ]
    )
    metadata.update(
        {
            "prompt_dedupe_key": key,
            "prompt_deduped_evidence_ids": list(
                metadata.get("prompt_deduped_evidence_ids") or [item.evidence_id]
            ),
            "prompt_provider_provenance": provenance,
            "prompt_providers": provider_names,
            "retrieved_by": retrieved_by,
        }
    )
    return replace(item, metadata=metadata, retrieved_by=tuple(retrieved_by))


def _prompt_provider_provenance(
    metadata: dict[str, Any],
    item: Evidence,
) -> list[dict[str, Any]]:
    existing = metadata.get("prompt_provider_provenance") or metadata.get("provider_provenance")
    existing_items = (
        [dict(value) for value in existing if isinstance(value, dict)]
        if isinstance(existing, list)
        else []
    )
    providers = _dedupe(
        [
            str(value)
            for value in (
                metadata.get("provider"),
                metadata.get("retrieval_provider"),
                metadata.get("provider_local_provider"),
            )
            if value
        ]
    )
    providers.extend(
        provider for provider in item.retrieved_by if provider and provider not in providers
    )
    providers.extend(
        provider
        for provider in _as_text_list(metadata.get("retrieved_by"))
        if provider and provider not in providers
    )
    if not providers:
        providers = ["unknown"]
    generated_items = [
        {
            "provider": provider,
            "provider_local_provider": metadata.get("provider_local_provider"),
            "evidence_id": item.evidence_id,
            "chunk_id": item.chunk_id,
            "rank": item.rank,
            "retrieval_score": item.retrieval_score,
        }
        for provider in providers
    ]
    return [*existing_items, *generated_items]


def _dedupe_provenance(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = (
            _provenance_key_value(item.get("provider")),
            _provenance_key_value(item.get("provider_local_provider")),
            _provenance_key_value(item.get("evidence_id")),
            _provenance_key_value(item.get("chunk_id")),
            _provenance_key_value(item.get("retrieval_task_id")),
            _provenance_key_value(item.get("retrieval_unit_id")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(item))
    return deduped


def _provenance_key_value(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return tuple(_provenance_key_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted((str(key), _provenance_key_value(item)) for key, item in value.items())
        )
    if isinstance(value, set):
        return tuple(sorted(_provenance_key_value(item) for item in value))
    return value


def _prompt_provider_names_from_provenance(
    provenance: list[dict[str, Any]],
) -> list[str]:
    return _dedupe(
        [
            str(value)
            for item in provenance
            for value in (item.get("provider_local_provider"), item.get("provider"))
            if value
        ]
    )


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


def _runtime_known_providers(settings: Settings, *, include_hybrid: bool) -> tuple[str, ...]:
    providers = list(known_query_providers(settings))
    if include_hybrid and "hybrid" not in providers:
        providers.insert(0, "hybrid")
    return tuple(providers)


def _retrieve_with_router(
    router: ProviderRouter,
    db: Session,
    *,
    query: str,
    top_k: int,
    filters: dict | None,
    options: dict,
    query_plan: QueryPlan | None = None,
    retrieval_tasks: list[RetrievalTask] | None = None,
) -> ProviderRouterResult:
    if query_plan is None:
        query_plan = QueryPlan(
            plan_id=new_id("qp"),
            original_query=query,
            standalone_query=query,
            retrieval_units=(
                RetrievalUnit(
                    unit_id="u0",
                    purpose="runtime_direct_hybrid",
                    text=query,
                    provider="hybrid",
                ),
            ),
            planner="runtime_direct",
            validation_status="validated",
        )
        retrieval_tasks = tasks_from_plan(
            query_plan,
            executable_providers=router.executable_providers,
        )
    return router.retrieve(
        db,
        query=query,
        top_k=top_k,
        filters=filters,
        options=options,
        query_plan=query_plan,
        retrieval_tasks=retrieval_tasks or [],
    )


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
    pre_verification = pre_critic.details.get("verification")
    if isinstance(pre_verification, dict):
        payload["evidence_evaluation"] = pre_verification
    post_verification = post_critic.details.get("verification") if post_critic else None
    if isinstance(post_verification, dict):
        payload["citation_verification"] = post_verification
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


def _provider_router_details(router_result: ProviderRouterResult | None) -> dict[str, Any]:
    if router_result is None:
        return {}
    details: dict[str, Any] = {
        "provider_router_trace": _sanitize_provider_router_trace(router_result.trace),
        "provider_results": [
            _sanitize_serialized_provider_result(serialize_provider_result(result))
            for result in router_result.provider_results
        ],
    }
    if router_result.evidence_pack is not None:
        details["evidence_pack"] = _evidence_pack_summary(router_result.evidence_pack)
        return details
    for result in router_result.provider_results:
        if result.evidence_pack is not None:
            details["evidence_pack"] = _evidence_pack_summary(result.evidence_pack)
            break
    return details


def _sanitize_provider_router_trace(trace: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(trace)
    provider_results = sanitized.get("provider_results")
    if isinstance(provider_results, list):
        sanitized["provider_results"] = [
            _sanitize_serialized_provider_result(item)
            if isinstance(item, dict)
            else item
            for item in provider_results
        ]
    router_error = sanitized.get("router_error")
    if isinstance(router_error, dict):
        sanitized["router_error"] = _sanitize_provider_trace(router_error)
    return sanitized


def _sanitize_serialized_provider_result(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(payload)
    trace = sanitized.get("trace")
    if isinstance(trace, dict):
        sanitized["trace"] = _sanitize_provider_trace(trace)
    return sanitized


def _sanitize_provider_trace(trace: dict[str, Any]) -> dict[str, Any]:
    if trace.get("status") != "failed":
        return dict(trace)
    allowed_keys = {
        "provider",
        "status",
        "reason",
        "error_type",
        "task_ids",
        "unit_ids",
        "error_code",
        "status_code",
    }
    return {key: value for key, value in trace.items() if key in allowed_keys}


def _provider_router_failed(router_result: ProviderRouterResult | None) -> bool:
    if router_result is None:
        return False
    if str(router_result.trace.get("status")) == "failed":
        return True
    return any(result.status == "failed" for result in router_result.provider_results)


def _provider_failure_details(router_result: ProviderRouterResult | None) -> dict[str, Any]:
    if router_result is None:
        return {}
    failures = [
        {
            "provider": result.provider,
            "task_id": result.task_id,
            "unit_id": result.unit_id,
            "reason": result.reason,
            "status": result.status,
            "error_type": dict(result.trace).get("error_type"),
        }
        for result in router_result.provider_results
        if result.status == "failed"
    ]
    return {"provider_failures": failures}


def _llm_io_request(
    *,
    settings: Settings,
    query: str,
    evidence: list[Evidence],
) -> dict[str, Any]:
    return {
        "model": settings.llm_model,
        "instructions": ANSWER_INSTRUCTIONS,
        "input": build_answer_input(query=query, evidence=evidence),
        "max_output_tokens": settings.llm_max_output_tokens,
        "reasoning": {"effort": settings.llm_reasoning_effort},
        "store": False,
    }


def _llm_io_request_metadata(
    *,
    settings: Settings,
    evidence: list[Evidence],
) -> dict[str, Any]:
    return {
        "prompt_version": settings.prompt_version,
        "evidence_ids": [item.evidence_id for item in evidence],
        "evidence_count": len(evidence),
    }


def _answer_llm_observability_payload(
    *,
    call_id: str,
    status: str,
    request: dict[str, Any],
    request_metadata: dict[str, Any],
    generated: object | None,
    evidence: list[Evidence],
    latency_ms: int | None,
    error_message: str | None,
) -> dict[str, Any]:
    usage = getattr(generated, "usage", None)
    usage_payload = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }
    response = None
    if generated is not None:
        response = {
            "raw_output": getattr(generated, "raw_output", None),
            "parsed_answer": getattr(generated, "answer", None),
            "parsed_confidence": getattr(generated, "confidence", None),
            "usage": usage_payload,
        }
    return {
        "answer_llm_call": {
            "call_id": call_id,
            "stage": "answer",
            "attempt_index": 1,
            "sequence_index": 100,
            "status": status,
            "error_message": error_message,
            "latency_ms": latency_ms,
            "model_name": request.get("model"),
            "prompt_version": request_metadata.get("prompt_version"),
            "request": dict(request),
            "request_metadata": dict(request_metadata),
            "response": response,
            "usage": usage_payload,
            "raw_output": response.get("raw_output") if response else None,
            "parsed_answer": response.get("parsed_answer") if response else None,
            "parsed_confidence": response.get("parsed_confidence") if response else None,
        },
        "answer_prompt_evidence": [
            _answer_prompt_evidence_payload(item, rank=rank)
            for rank, item in enumerate(evidence, start=1)
        ],
    }


def _answer_prompt_evidence_payload(item: Evidence, *, rank: int) -> dict[str, Any]:
    metadata = dict(item.metadata or {})
    providers = metadata.get("prompt_providers")
    if not isinstance(providers, list):
        providers = _prompt_provider_names_from_provenance(
            _prompt_provider_provenance(metadata, item)
        )
    return {
        "evidence_id": item.evidence_id,
        "rank": rank,
        "provider": ",".join(str(provider) for provider in providers if provider) or None,
        "chunk_id": item.chunk_id,
        "document_id": item.document_id,
        "page_start": item.page_start,
        "page_end": item.page_end,
        "retrieval_score": item.retrieval_score,
        "token_count": item.token_count,
        "text_snapshot": item.text,
        "metadata": {
            "prompt_dedupe_key": metadata.get("prompt_dedupe_key"),
            "prompt_deduped": metadata.get("prompt_deduped"),
            "prompt_duplicate_count": metadata.get("prompt_duplicate_count"),
            "prompt_deduped_evidence_ids": metadata.get("prompt_deduped_evidence_ids"),
            "prompt_provider_provenance": metadata.get("prompt_provider_provenance"),
            "prompt_providers": metadata.get("prompt_providers"),
        },
    }


def _merge_observability(*items: dict[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if value in (None, {}, []):
                continue
            merged[key] = value
    return merged or None


def _llm_io_completed_details(
    *,
    answer_llm_call_id: str,
) -> dict[str, Any]:
    return {
        "llm_io": {
            "status": "completed",
            "answer_llm_call_id": answer_llm_call_id,
        }
    }


def _llm_io_failed_details(
    *,
    answer_llm_call_id: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "llm_io": {
            "status": "failed",
            "answer_llm_call_id": answer_llm_call_id,
            "error_message": error_message,
        }
    }


def _llm_io_skipped_details(reason: str) -> dict[str, Any]:
    return {
        "llm_io": {
            "status": "skipped",
            "reason": reason,
        }
    }


def _evidence_pack_summary(pack: object) -> dict[str, Any]:
    blocks = tuple(getattr(pack, "blocks", ()) or ())
    dropped_blocks = tuple(getattr(pack, "dropped_blocks", ()) or ())
    return {
        "pack_id": getattr(pack, "pack_id", None),
        "token_count": getattr(pack, "token_count", None),
        "max_context_tokens": getattr(pack, "max_context_tokens", None),
        "block_count": len(blocks),
        "dropped_block_count": len(dropped_blocks),
        "blocks": [_evidence_block_summary(block) for block in blocks],
        "dropped_blocks": [
            _evidence_block_summary(block) for block in dropped_blocks
        ],
        "metadata": dict(getattr(pack, "metadata", {}) or {}),
    }


def _evidence_block_summary(block: object) -> dict[str, Any]:
    metadata = dict(getattr(block, "metadata", {}) or {})
    return {
        "evidence_id": getattr(block, "evidence_id", None),
        "document_id": getattr(block, "document_id", None),
        "chunk_ids": list(getattr(block, "chunk_ids", ()) or ()),
        "parent_id": getattr(block, "parent_id", None),
        "rank": getattr(block, "rank", None),
        "token_count": getattr(block, "token_count", None),
        "included_in_prompt": getattr(block, "included_in_prompt", None),
        "drop_reason": getattr(block, "drop_reason", None),
        "drop_stage": getattr(block, "drop_stage", None),
        "coverage": getattr(block, "coverage", None),
        "source_anchor": metadata.get("source_anchor"),
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
        "provider_status": metadata.get("provider_status"),
        "original_evidence_id": metadata.get("original_evidence_id"),
        "provider_local_evidence_id": metadata.get("provider_local_evidence_id"),
        "provider_local_rank": metadata.get("provider_local_rank"),
        "provider_local_provider": metadata.get("provider_local_provider"),
        "source_anchor": metadata.get("source_anchor"),
        "lane": metadata.get("lane"),
        "lanes": _as_text_list(metadata.get("lanes")),
        "parent_lanes": _as_text_list(metadata.get("parent_lanes")),
        "internal_lanes": _as_text_list(metadata.get("internal_lanes")),
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
        "fusion_backend": metadata.get("fusion_backend")
        or (
            metadata.get("fusion", {}).get("backend")
            if isinstance(metadata.get("fusion"), dict)
            else None
        ),
        "retrieval_task_id": metadata.get("retrieval_task_id"),
        "retrieval_unit_id": metadata.get("retrieval_unit_id"),
        "fusion_rank": metadata.get("fusion_rank") or metadata.get("best_fusion_rank"),
        "fusion_score": metadata.get("fusion_score") or metadata.get("best_fusion_score"),
        "rerank_rank": metadata.get("rerank_rank") or metadata.get("best_rerank_rank"),
        "rerank_score": metadata.get("rerank_score") or metadata.get("best_rerank_score"),
        "reranker": metadata.get("reranker"),
        "reranker_input": metadata.get("reranker_input"),
        "text_hybrid_provider": metadata.get("text_hybrid_provider"),
        "evidence_pack": metadata.get("evidence_pack"),
        "coverage": metadata.get("coverage"),
        "included_in_prompt": metadata.get("included_in_prompt"),
        "drop_reason": metadata.get("drop_reason"),
        "prompt_dedupe_key": metadata.get("prompt_dedupe_key"),
        "prompt_deduped": metadata.get("prompt_deduped"),
        "prompt_duplicate_count": metadata.get("prompt_duplicate_count"),
        "prompt_deduped_evidence_ids": metadata.get("prompt_deduped_evidence_ids"),
        "prompt_provider_provenance": metadata.get("prompt_provider_provenance"),
        "prompt_providers": metadata.get("prompt_providers"),
    }


def _combined_critic_status(
    pre_critic: CriticResult,
    post_critic: CriticResult | None,
) -> str:
    statuses = [pre_critic.status, post_critic.status if post_critic else None]
    for status in (
        "contradicted",
        "unsupported",
        "insufficient",
        "partially_supported",
        "warning",
    ):
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


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple | set):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_as_text_list(item))
        return flattened
    try:
        return [str(item) for item in value if item]
    except TypeError:
        return [str(value)] if value else []


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
    executable_providers: tuple[str, ...],
) -> tuple[QueryPlan | None, list[RetrievalTask], int | None, dict[str, Any] | None]:
    if orchestrator is None or _truthy(options.get("disable_query_plan")):
        return None, [], None, None
    use_llm = not _truthy(options.get("query_plan_fallback_only"))
    started = time.perf_counter()
    plan_with_observability = getattr(orchestrator, "plan_with_observability", None)
    if callable(plan_with_observability):
        if _call_accepts_keyword(plan_with_observability, "executable_providers"):
            plan, planner_observability = plan_with_observability(
                query,
                use_llm=use_llm,
                executable_providers=executable_providers,
            )
        else:
            plan, planner_observability = plan_with_observability(query, use_llm=use_llm)
    else:
        plan_method = getattr(orchestrator, "plan")
        if _call_accepts_keyword(plan_method, "executable_providers"):
            plan = orchestrator.plan(
                query,
                use_llm=use_llm,
                executable_providers=executable_providers,
            )
        else:
            plan = orchestrator.plan(query, use_llm=use_llm)
        planner_observability = getattr(orchestrator, "last_observability", None)
    latency_ms = int((time.perf_counter() - started) * 1000)
    return (
        plan,
        tasks_from_plan(plan, executable_providers=executable_providers),
        latency_ms,
        planner_observability if isinstance(planner_observability, dict) else None,
    )


def _call_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD or name == keyword
        for name, parameter in signature.parameters.items()
    )


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
