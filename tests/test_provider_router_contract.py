import json
import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from atlas.api import dependencies as dependency_module
from atlas.core.config import (
    IMPLEMENTED_RUNTIME_PROVIDERS,
    Settings,
    executable_query_providers,
)
from atlas.core.errors import AtlasError, ErrorCode
from atlas.db.models import GenerationEvent, QueryCache, QueryRun
from atlas.db.repositories import record_v1_trace_family
from atlas.llm.base import GeneratedAnswer, LLMUsage
from atlas.llm.clients import LLMResponse
from atlas.llm.openai_client import OpenAIAnswerGenerator
from atlas.query_orchestrator.schema import QueryPlan, RetrievalUnit
from atlas.query_runtime.citation_builder import build_citations
from atlas.query_runtime.service import (
    QueryRuntime,
    _retrieval_trace_details,
    _runtime_details,
)
from atlas.query_runtime.trace_logger import make_retrieval_events
from atlas.retrieval.contracts import ProviderResult
from atlas.retrieval.models.candidate import Candidate
from atlas.retrieval.models.evidence import Evidence
from atlas.retrieval.models.retrieval_task import RetrievalTask, tasks_from_plan
from atlas.retrieval.providers.base import RetrievalContext, RetrievalProvider
from atlas.retrieval.providers.graph import GraphProvider, PostgresGraphStore
from atlas.retrieval.candidate_adapter import candidates_from_provider_result
from atlas.retrieval.router import ProviderRouter, serialize_provider_result


class _SyncProvider(RetrievalProvider):
    provider_name = "hybrid"

    async def aretrieve_candidates(self, context: RetrievalContext):
        return self.retrieve_provider_result(
            context.db,
            query=context.query,
            top_k=context.top_k,
            filters=context.filters,
            options=context.options,
            query_plan=context.query_plan,
            retrieval_tasks=context.retrieval_tasks,
        )


class _HybridProvider(_SyncProvider):
    def __init__(self) -> None:
        self.calls = []

    def retrieve_provider_result(
        self,
        db,
        *,
        query,
        top_k,
        filters,
        options,
        query_plan,
        retrieval_tasks,
    ):
        self.calls.append(tuple(task.task_id for task in retrieval_tasks))
        evidence = (
            Evidence(
                evidence_id="c1",
                document_id="doc_1",
                chunk_id="chk_1",
                text="Apple management discussed supplier disruption risk.",
                source_title="Apple 10-K",
                source_uri=None,
                section_title="Risk Factors",
                page_start=10,
                page_end=10,
                retrieval_score=1.0,
                rank=1,
                token_count=8,
                metadata={"provider": "text_hybrid", "lane": "dense"},
            ),
        )
        return ProviderResult(
            provider="hybrid",
            task_id=retrieval_tasks[0].task_id,
            unit_id=retrieval_tasks[0].unit_id,
            status="executed",
            evidence=evidence,
            latency_ms=1,
            trace={"provider": "hybrid", "status": "executed"},
        )


class _TopKRecordingHybridProvider(_SyncProvider):
    def __init__(self) -> None:
        self.top_ks = []

    def retrieve_provider_result(
        self,
        db,
        *,
        query,
        top_k,
        filters,
        options,
        query_plan,
        retrieval_tasks,
    ):
        self.top_ks.append(top_k)
        evidence = (
            Evidence(
                evidence_id="c1",
                document_id="doc_1",
                chunk_id="chk_1",
                text="First evidence item.",
                source_title="Source 1",
                source_uri=None,
                section_title="Section",
                page_start=1,
                page_end=1,
                retrieval_score=1.0,
                rank=1,
                token_count=3,
                metadata={"provider": "text_hybrid"},
            ),
            Evidence(
                evidence_id="c2",
                document_id="doc_2",
                chunk_id="chk_2",
                text="Second evidence item.",
                source_title="Source 2",
                source_uri=None,
                section_title="Section",
                page_start=2,
                page_end=2,
                retrieval_score=0.9,
                rank=2,
                token_count=3,
                metadata={"provider": "text_hybrid"},
            ),
        )
        return ProviderResult(
            provider="hybrid",
            task_id=retrieval_tasks[0].task_id,
            unit_id=retrieval_tasks[0].unit_id,
            status="executed",
            evidence=evidence,
            latency_ms=1,
            trace={"provider": "hybrid", "status": "executed", "top_k": top_k},
        )


class _GraphProvider(_SyncProvider):
    provider_name = "graph"

    def __init__(self) -> None:
        self.calls = []

    def retrieve_provider_result(
        self,
        db,
        *,
        query,
        top_k,
        filters,
        options,
        query_plan,
        retrieval_tasks,
    ):
        self.calls.append(tuple(task.task_id for task in retrieval_tasks))
        evidence = (
            Evidence(
                evidence_id="c1",
                document_id="doc_graph",
                chunk_id="chk_graph",
                text="A graph-grounded supplier relationship is supported by this chunk.",
                source_title="Graph Source",
                source_uri=None,
                section_title="Relationships",
                page_start=5,
                page_end=5,
                retrieval_score=0.9,
                rank=1,
                token_count=8,
                metadata={
                    "provider": "graph",
                    "graph_candidate_id": "graph_local:rt_1:entity_1",
                    "retrieved_by": ["graph"],
                    "sources": ["graph"],
                },
                retrieved_by=("graph",),
            ),
        )
        return ProviderResult(
            provider="graph",
            task_id=retrieval_tasks[0].task_id,
            unit_id=retrieval_tasks[0].unit_id,
            status="executed",
            evidence=evidence,
            latency_ms=1,
            trace={"provider": "graph", "status": "executed"},
        )


class _DuplicateChunkGraphProvider(_SyncProvider):
    provider_name = "graph"

    def retrieve_provider_result(
        self,
        db,
        *,
        query,
        top_k,
        filters,
        options,
        query_plan,
        retrieval_tasks,
    ):
        evidence = (
            Evidence(
                evidence_id="g_local",
                document_id="doc_1",
                chunk_id="chk_1",
                text="Graph duplicate for the same Apple supplier chunk.",
                source_title="Apple 10-K",
                source_uri=None,
                section_title="Risk Factors",
                page_start=10,
                page_end=10,
                retrieval_score=0.95,
                rank=1,
                token_count=7,
                metadata={
                    "provider": "graph",
                    "retrieved_by": ["graph"],
                },
                retrieved_by=("graph",),
            ),
        )
        return ProviderResult(
            provider="graph",
            task_id=retrieval_tasks[0].task_id,
            unit_id=retrieval_tasks[0].unit_id,
            status="executed",
            evidence=evidence,
            latency_ms=1,
            trace={"provider": "graph", "status": "executed"},
        )


class _EmptyProvider(_SyncProvider):
    def __init__(self, provider: str) -> None:
        self.provider_name = provider
        self.provider = provider

    def retrieve_provider_result(
        self,
        db,
        *,
        query,
        top_k,
        filters,
        options,
        query_plan,
        retrieval_tasks,
    ):
        return ProviderResult(
            provider=self.provider,
            task_id=None if len(retrieval_tasks) != 1 else retrieval_tasks[0].task_id,
            unit_id=None if len(retrieval_tasks) != 1 else retrieval_tasks[0].unit_id,
            status="empty",
            candidates=(),
            evidence=(),
            latency_ms=1,
            reason="no_results",
            trace={"provider": self.provider, "status": "empty"},
        )


class _SleepingProvider(RetrievalProvider):
    def __init__(
        self,
        provider: str,
        *,
        delay: float,
        active_counter: dict | None = None,
    ) -> None:
        self.provider_name = provider
        self.provider = provider
        self.delay = delay
        self.active_counter = active_counter

    async def aretrieve_candidates(self, context: RetrievalContext):
        if self.active_counter is not None:
            with self.active_counter["lock"]:
                self.active_counter["current"] = self.active_counter.get("current", 0) + 1
                self.active_counter["max"] = max(
                    self.active_counter.get("max", 0),
                    self.active_counter["current"],
                )
        try:
            await asyncio.sleep(self.delay)
        finally:
            if self.active_counter is not None:
                with self.active_counter["lock"]:
                    self.active_counter["current"] = self.active_counter.get("current", 0) - 1
        evidence = (
            Evidence(
                evidence_id="c1",
                document_id=f"doc_{self.provider}",
                chunk_id=f"chk_{self.provider}",
                text=f"{self.provider} evidence.",
                source_title=f"{self.provider} source",
                source_uri=None,
                section_title=None,
                page_start=1,
                page_end=1,
                retrieval_score=1.0,
                rank=1,
                token_count=3,
                metadata={"provider": self.provider},
            ),
        )
        return ProviderResult(
            provider=self.provider,
            task_id=context.retrieval_tasks[0].task_id,
            unit_id=context.retrieval_tasks[0].unit_id,
            status="executed",
            evidence=evidence,
            latency_ms=int(self.delay * 1000),
            trace={"provider": self.provider, "status": "executed"},
        )


class _ThreadRecordingSession:
    def __init__(self, events: list[tuple[str, int]]) -> None:
        self.events = events
        self.events.append(("created", threading.get_ident()))

    def close(self) -> None:
        self.events.append(("closed", threading.get_ident()))


class _ThreadRecordingProvider(_SyncProvider):
    provider_name = "hybrid"

    def __init__(self, events: list[tuple[str, int]]) -> None:
        self.events = events

    def retrieve_provider_result(
        self,
        db,
        *,
        query,
        top_k,
        filters,
        options,
        query_plan,
        retrieval_tasks,
    ):
        db.events.append(("used", threading.get_ident()))
        return ProviderResult(
            provider="hybrid",
            task_id=retrieval_tasks[0].task_id,
            unit_id=retrieval_tasks[0].unit_id,
            status="empty",
            candidates=(),
            latency_ms=1,
            trace={"provider": "hybrid", "status": "empty"},
        )


class _CandidateProvider(_SyncProvider):
    def __init__(
        self,
        provider: str,
        chunk_id: str,
        *,
        local_rank: int,
        parent_id: str | None = None,
        provenance_provider: object | None = None,
    ) -> None:
        self.provider_name = provider
        self.provider = provider
        self.chunk_id = chunk_id
        self.local_rank = local_rank
        self.parent_id = parent_id
        self.provenance_provider = provenance_provider

    def retrieve_provider_result(
        self,
        db,
        *,
        query,
        top_k,
        filters,
        options,
        query_plan,
        retrieval_tasks,
    ):
        candidate = Candidate(
            candidate_id=f"cand_{self.provider}",
            chunk_id=self.chunk_id,
            document_id=f"doc_{self.provider}",
            doc_name=f"{self.provider} doc",
            source_title=f"{self.provider} source",
            company=None,
            text=f"{self.provider} candidate text.",
            page_start=1,
            page_end=1,
            chunk_index=self.local_rank,
            token_count=4,
            retrieved_by=(self.provider,),
            dense_rank=None,
            dense_score=None,
            fusion_rank=self.local_rank,
            fusion_score=1.0 / self.local_rank,
            final_rank=self.local_rank,
            metadata={
                "provider": self.provider,
                "parent_id": self.parent_id,
                "provider_provenance": [
                    {
                        "provider": self.provenance_provider or self.provider,
                        "provider_local_provider": self.provider,
                        "provider_local_evidence_id": f"c{self.local_rank}",
                        "chunk_id": self.chunk_id,
                        "parent_id": self.parent_id,
                    }
                ],
            },
            provider=self.provider,
            parent_id=self.parent_id,
        )
        return ProviderResult(
            provider=self.provider,
            task_id=retrieval_tasks[0].task_id,
            unit_id=retrieval_tasks[0].unit_id,
            status="executed",
            candidates=(candidate,),
            latency_ms=1,
            trace={"provider": self.provider, "status": "executed"},
        )


class _OptionsRecordingCandidateProvider(_CandidateProvider):
    def __init__(self, provider: str, chunk_id: str, *, local_rank: int) -> None:
        super().__init__(provider, chunk_id, local_rank=local_rank)
        self.options_seen: list[dict] = []

    def retrieve_provider_result(
        self,
        db,
        *,
        query,
        top_k,
        filters,
        options,
        query_plan,
        retrieval_tasks,
    ):
        self.options_seen.append(dict(options))
        return super().retrieve_provider_result(
            db,
            query=query,
            top_k=top_k,
            filters=filters,
            options=options,
            query_plan=query_plan,
            retrieval_tasks=retrieval_tasks,
        )


class _WindowCandidateProvider(_SyncProvider):
    provider_name = "hybrid"

    def __init__(self) -> None:
        self.top_ks: list[int] = []

    def retrieve_provider_result(
        self,
        db,
        *,
        query,
        top_k,
        filters,
        options,
        query_plan,
        retrieval_tasks,
    ):
        self.top_ks.append(top_k)
        candidates = tuple(
            _candidate_for_test("hybrid", f"chk_{index}", index)
            for index in range(1, top_k + 1)
        )
        return ProviderResult(
            provider="hybrid",
            task_id=retrieval_tasks[0].task_id,
            unit_id=retrieval_tasks[0].unit_id,
            status="executed",
            candidates=candidates,
            latency_ms=1,
            trace={"provider": "hybrid", "status": "executed", "top_k": top_k},
        )


class _ReverseReranker:
    def rerank(self, *, candidates, **kwargs):
        return [
            replace(
                candidate,
                final_rank=index,
                rerank_rank=index,
                rerank_score=float(index),
            )
            for index, candidate in enumerate(reversed(candidates), start=1)
        ]


class _RecordingReranker:
    def __init__(self) -> None:
        self.seen_chunk_ids: list[list[str]] = []

    def rerank(self, *, candidates, **kwargs):
        self.seen_chunk_ids.append([candidate.chunk_id for candidate in candidates])
        return [
            replace(
                candidate,
                final_rank=index,
                rerank_rank=index,
                rerank_score=float(100 - index),
            )
            for index, candidate in enumerate(candidates, start=1)
        ]


class _FailingReranker:
    def rerank(self, *, candidates, **kwargs):
        raise RuntimeError("reranker exploded with query text")


def _candidate_for_test(provider: str, chunk_id: str, local_rank: int) -> Candidate:
    return Candidate(
        candidate_id=f"cand_{provider}_{local_rank}",
        chunk_id=chunk_id,
        document_id=f"doc_{provider}",
        doc_name=f"{provider} doc",
        source_title=f"{provider} source",
        company=None,
        text=f"{provider} candidate text {local_rank}.",
        page_start=1,
        page_end=1,
        chunk_index=local_rank,
        token_count=4,
        retrieved_by=(provider,),
        dense_rank=None,
        dense_score=None,
        fusion_rank=local_rank,
        fusion_score=1.0 / local_rank,
        final_rank=local_rank,
        metadata={"provider": provider},
        provider=provider,
    )


class _StaticOrchestrator:
    def __init__(self, plan: QueryPlan, observability=None) -> None:
        self.plan_value = plan
        self.last_observability = observability or {}
        self.executable_provider_calls = []

    def plan(self, query, *, use_llm=True):
        return self.plan_value

    def plan_with_observability(
        self,
        query,
        *,
        use_llm=True,
        executable_providers=None,
    ):
        self.executable_provider_calls.append(executable_providers)
        return self.plan_value, self.last_observability


class _Generator:
    model_name = "fake-generator"

    def generate(self, *, query, evidence):
        return GeneratedAnswer(
            answer="Apple discussed supplier disruption risk [c1].",
            confidence="supported",
            usage=LLMUsage(input_tokens=1, output_tokens=1),
            raw_output="{}",
        )


class _RecordingEvidenceGenerator(_Generator):
    def __init__(self) -> None:
        self.evidence = None

    def generate(self, *, query, evidence):
        self.evidence = list(evidence)
        return super().generate(query=query, evidence=evidence)


class _LegacyRecordingRetriever:
    def __init__(self) -> None:
        self.db_seen = None

    def retrieve(self, db, *, query, top_k, filters=None):
        self.db_seen = db
        return [
            Evidence(
                evidence_id="c1",
                document_id="doc_legacy",
                chunk_id="chk_legacy",
                text="Legacy retriever evidence about supplier risk.",
                source_title="Legacy Source",
                source_uri=None,
                section_title=None,
                page_start=1,
                page_end=1,
                retrieval_score=1.0,
                rank=1,
                token_count=6,
                metadata={"provider": "legacy_hybrid"},
            )
        ]


class _DB:
    def __init__(self) -> None:
        self.added = []
        self.commits = 0
        self.rollbacks = 0

    def add(self, value) -> None:
        self.added.append(value)

    def flush(self) -> None:
        return None

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _FailedTransactionDB(_DB):
    is_active = False

    def rollback(self) -> None:
        super().rollback()
        self.is_active = True


class _CacheHitDB(_DB):
    def __init__(self, record: QueryCache) -> None:
        super().__init__()
        self.record = record
        self.cache_gets = []

    def get(self, model, key):
        if model is QueryCache:
            self.cache_gets.append(key)
            return self.record
        return None


class _FailingProvider(_SyncProvider):
    def __init__(self, exc: Exception, *, provider: str = "graph") -> None:
        self.provider_name = provider
        self.exc = exc

    def retrieve_provider_result(
        self,
        db,
        *,
        query,
        top_k,
        filters,
        options,
        query_plan,
        retrieval_tasks,
    ):
        raise self.exc


class _RecordingLLMClient:
    def __init__(self, *, output_text: str, input_tokens: int, output_tokens: int) -> None:
        self.output_text = output_text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.requests = []

    def create_response(self, request):
        self.requests.append(dict(request))
        return LLMResponse(
            output_text=self.output_text,
            raw={"request_index": len(self.requests)},
            usage=SimpleNamespace(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
            ),
        )


def _json_blob(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _nested_keys(value) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key).lower() for key in value}
        for item in value.values():
            keys.update(_nested_keys(item))
        return keys
    if isinstance(value, list | tuple):
        keys: set[str] = set()
        for item in value:
            keys.update(_nested_keys(item))
        return keys
    return set()


def _planner_observability_payload(
    *,
    call_id: str,
    plan_id: str,
    status: str = "completed",
    validation_status: str = "validated",
    error_message: str | None = None,
):
    raw_prompt = f"RAW PLANNER PROMPT SENTINEL {call_id}"
    raw_response = f"RAW PLANNER RESPONSE SENTINEL {call_id}"
    call = {
        "call_id": call_id,
        "stage": "planner",
        "attempt_index": 1,
        "sequence_index": 1,
        "status": status,
        "validation_status": validation_status,
        "error_message": error_message,
        "latency_ms": 9,
        "model_name": "planner-model",
        "planner_version": "query_planner_test",
        "request": {
            "model": "planner-model",
            "instructions": "Planner instructions. " + raw_prompt,
            "input": raw_prompt,
            "max_output_tokens": 2000,
            "reasoning": {"effort": "low"},
            "text": {"format": {"type": "json_schema"}},
            "store": False,
        },
        "response": {
            "raw_output": raw_response,
            "parsed_json": {
                "standalone_query": "Explain Apple supplier disruption risk.",
                "retrieval_units": [{"unit_id": "u_hybrid", "provider": "hybrid"}],
            },
            "parsed_plan_id": plan_id,
            "validation_status": validation_status,
            "usage": {"input_tokens": 31, "output_tokens": 13},
        },
        "usage": {"input_tokens": 31, "output_tokens": 13},
        "raw_output": raw_response,
        "parsed_plan_id": plan_id,
    }
    return {"planner_llm_calls": [call], "planner_llm_call": dict(call)}


def test_provider_router_executes_registered_hybrid_and_skips_sql() -> None:
    plan = QueryPlan(
        plan_id="plan_1",
        original_query="Compare R&D and explain supplier risk.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_sql",
                purpose="structured_calculation",
                text="Apple Microsoft R&D 2023",
                provider="sql",
            ),
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="risk_factor_text",
                text="Apple supplier disruption risk",
                provider="hybrid",
            ),
        ),
    )
    tasks = tasks_from_plan(plan)
    provider = _HybridProvider()
    router = ProviderRouter({"hybrid": provider})

    result = router.retrieve(
        db := _DB(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert len(provider.calls) == 1
    assert len(result.evidence) == 1
    assert [item.status for item in result.provider_results] == [
        "skipped_non_executable",
        "executed",
    ]
    assert result.provider_results[0].provider == "sql"
    assert result.provider_results[0].reason == "provider_not_executable:sql"
    assert result.trace["status"] == "partial"


def test_provider_router_normalizes_top_k_before_provider_call() -> None:
    plan = QueryPlan(
        plan_id="plan_negative_top_k",
        original_query="Find supplier risk evidence.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="risk_factor_text",
                text="supplier disruption risk",
                provider="hybrid",
            ),
        ),
    )
    tasks = tasks_from_plan(plan)
    provider = _TopKRecordingHybridProvider()
    router = ProviderRouter({"hybrid": provider})

    db = _DB()
    result = router.retrieve(
        db,
        query=plan.original_query,
        top_k=-1,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert provider.top_ks == [0]
    assert len(result.provider_results[0].evidence) == 2
    assert result.evidence == ()

    positive_provider = _TopKRecordingHybridProvider()
    positive_router = ProviderRouter({"hybrid": positive_provider}, reranker_top_k=1)
    positive_result = positive_router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=1,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert positive_provider.top_ks == [1]
    assert tuple(item.evidence_id for item in positive_result.evidence) == ("c1",)


def test_provider_router_rejects_sql_registration_and_keeps_sql_tasks_skipped() -> None:
    plan = QueryPlan(
        plan_id="plan_sql",
        original_query="What was Apple's 2023 R&D expense?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_sql",
                purpose="structured_lookup",
                text="Apple R&D expense 2023",
                provider="sql",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid",))
    with pytest.raises(ValueError, match="non_executable_provider_registered:sql"):
        ProviderRouter({"sql": _HybridProvider()})
    hybrid_provider = _HybridProvider()
    router = ProviderRouter({"hybrid": hybrid_provider})

    result = router.retrieve(
        db := _DB(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert hybrid_provider.calls == []
    assert result.evidence == ()
    assert result.provider_results[0].provider == "sql"
    assert result.provider_results[0].status == "skipped_non_executable"


def test_provider_router_skips_manual_ready_sql_task_as_non_executable() -> None:
    task = RetrievalTask(
        task_id="rt_sql",
        plan_id="plan_manual_sql",
        unit_id="u_sql",
        query_text="Apple R&D expense 2023",
        provider="sql",
        provider_status="ready",
    )
    plan = QueryPlan(
        plan_id="plan_manual_sql",
        original_query="What was Apple's 2023 R&D expense?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_sql",
                purpose="structured_lookup",
                text="Apple R&D expense 2023",
                provider="sql",
            ),
        ),
    )
    router = ProviderRouter({"hybrid": _HybridProvider()})

    db = _DB()
    result = router.retrieve(
        db,
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=[task],
    )

    assert result.provider_results[0].provider == "sql"
    assert result.provider_results[0].status == "skipped_non_executable"
    assert result.provider_results[0].reason == "provider_not_executable:sql"


def test_provider_router_trace_status_empty_when_all_provider_results_empty() -> None:
    plan = QueryPlan(
        plan_id="plan_empty",
        original_query="Find graph and text evidence that does not exist.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="empty_text_lookup",
                text="missing text evidence",
                provider="hybrid",
            ),
            RetrievalUnit(
                unit_id="u_graph",
                purpose="empty_graph_lookup",
                text="missing graph evidence",
                provider="graph",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid", "graph"))
    router = ProviderRouter(
        {"hybrid": _EmptyProvider("hybrid"), "graph": _EmptyProvider("graph")}
    )

    db = _DB()
    result = router.retrieve(
        db,
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert result.evidence == ()
    assert [item.status for item in result.provider_results] == ["empty", "empty"]
    assert result.trace["status"] == "empty"


def test_provider_router_preserves_successful_sibling_when_provider_fails() -> None:
    plan = QueryPlan(
        plan_id="plan_partial_failure",
        original_query="Explain supplier risk with graph context.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
            RetrievalUnit(
                unit_id="u_graph",
                purpose="graph",
                text="supplier graph",
                provider="graph",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid", "graph"))
    router = ProviderRouter(
        {
            "hybrid": _CandidateProvider("hybrid", "chk_hybrid", local_rank=1),
            "graph": _FailingProvider(RuntimeError("graph store unavailable")),
        }
    )

    db = _DB()
    result = router.retrieve(
        db,
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert [item.status for item in result.provider_results] == ["executed", "failed"]
    assert result.trace["status"] == "partial"
    assert result.evidence[0].chunk_id == "chk_hybrid"
    assert result.provider_results[1].trace["error_type"] == "RuntimeError"
    assert db.rollbacks == 0


def test_provider_router_rolls_back_failed_caller_transaction_after_provider_error() -> None:
    plan = QueryPlan(
        plan_id="plan_partial_failure_failed_transaction",
        original_query="Explain supplier risk with graph context.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
            RetrievalUnit(
                unit_id="u_graph",
                purpose="graph",
                text="supplier graph",
                provider="graph",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid", "graph"))
    router = ProviderRouter(
        {
            "hybrid": _CandidateProvider("hybrid", "chk_hybrid", local_rank=1),
            "graph": _FailingProvider(RuntimeError("graph store unavailable")),
        }
    )

    db = _FailedTransactionDB()
    result = router.retrieve(
        db,
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert [item.status for item in result.provider_results] == ["executed", "failed"]
    assert result.trace["status"] == "partial"
    assert db.rollbacks == 1


def test_provider_router_closes_owned_sessions_when_provider_fails() -> None:
    plan = QueryPlan(
        plan_id="plan_partial_failure_owned_sessions",
        original_query="Explain supplier risk with graph context.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
            RetrievalUnit(
                unit_id="u_graph",
                purpose="graph",
                text="supplier graph",
                provider="graph",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid", "graph"))
    events: list[tuple[str, int]] = []
    router = ProviderRouter(
        {
            "hybrid": _CandidateProvider("hybrid", "chk_hybrid", local_rank=1),
            "graph": _FailingProvider(RuntimeError("graph store unavailable")),
        },
        session_factory=lambda: _ThreadRecordingSession(events),
    )

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert [item.status for item in result.provider_results] == ["executed", "failed"]
    assert result.trace["status"] == "partial"
    assert sum(1 for event, _thread_id in events if event == "created") == 3
    assert sum(1 for event, _thread_id in events if event == "closed") == 3


def test_provider_router_assigns_global_evidence_ids_before_citation_building() -> None:
    plan = QueryPlan(
        plan_id="plan_multi_provider_evidence",
        original_query="Explain supplier risk with graph context.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="risk_factor_text",
                text="Apple supplier disruption risk",
                provider="hybrid",
            ),
            RetrievalUnit(
                unit_id="u_graph",
                purpose="supplier_relationship_graph",
                text="Apple supplier relationships",
                provider="graph",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid", "graph"))
    router = ProviderRouter({"hybrid": _HybridProvider(), "graph": _GraphProvider()})

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert tuple(item.evidence_id for item in result.evidence) == ("c1", "c2")
    assert {item.chunk_id for item in result.evidence} == {"chk_1", "chk_graph"}
    assert {
        item.metadata["provider_local_provider"] for item in result.evidence
    } == {"hybrid", "graph"}
    assert {
        item.metadata["provider_local_evidence_id"] for item in result.evidence
    } == {"c1"}

    citations = build_citations(
        "Text and graph evidence both support this answer [c1] [c2].",
        list(result.evidence),
        confidence="supported",
    )

    assert [item["citation_id"] for item in citations] == ["c1", "c2"]
    assert {item["chunk_id"] for item in citations} == {"chk_1", "chk_graph"}
    assert citations[0]["supporting_text"] != citations[1]["supporting_text"]


def test_runtime_retrieval_trace_projects_provider_local_evidence_ids() -> None:
    plan = QueryPlan(
        plan_id="plan_trace_provider_local_ids",
        original_query="Explain supplier risk with graph context.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="risk_factor_text",
                text="Apple supplier disruption risk",
                provider="hybrid",
            ),
            RetrievalUnit(
                unit_id="u_graph",
                purpose="supplier_relationship_graph",
                text="Apple supplier relationships",
                provider="graph",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid", "graph"))
    router = ProviderRouter({"hybrid": _HybridProvider(), "graph": _GraphProvider()})

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )
    details = _runtime_details(
        query_plan=plan,
        retrieval_tasks=tasks,
        plan_latency_ms=1,
        extra=_retrieval_trace_details(list(result.evidence)),
    )

    trace_items = details["retrieval_trace"]["top_k"]
    assert [
        (
            item["evidence_id"],
            item["provider_local_provider"],
            item["provider_local_evidence_id"],
            item["provider_local_rank"],
            item["original_evidence_id"],
        )
        for item in trace_items
    ] == [
        ("c1", "hybrid", "c1", 1, "c1"),
        ("c2", "graph", "c1", 1, "c1"),
    ]


def test_provider_router_keeps_global_evidence_ids_contiguous_after_top_k() -> None:
    plan = QueryPlan(
        plan_id="plan_top_k_truncation",
        original_query="Explain supplier risk with graph context.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="risk_factor_text",
                text="Apple supplier disruption risk",
                provider="hybrid",
            ),
            RetrievalUnit(
                unit_id="u_graph",
                purpose="supplier_relationship_graph",
                text="Apple supplier relationships",
                provider="graph",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid", "graph"))
    router = ProviderRouter({"hybrid": _HybridProvider(), "graph": _GraphProvider()})

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=1,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert sum(len(item.evidence) for item in result.provider_results) == 2
    assert tuple(item.evidence_id for item in result.evidence) == ("c1",)

    citations = build_citations(
        "Only retained evidence is cited [c1], while dropped evidence is ignored [c2].",
        list(result.evidence),
        confidence="supported",
    )

    assert [item["citation_id"] for item in citations] == ["c1"]
    assert [item["chunk_id"] for item in citations] == [result.evidence[0].chunk_id]


def test_provider_router_preserves_single_provider_c1_citation_compatibility() -> None:
    plan = QueryPlan(
        plan_id="plan_single_provider_evidence",
        original_query="Explain Apple supplier disruption risk.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="risk_factor_text",
                text="Apple supplier disruption risk",
                provider="hybrid",
            ),
        ),
    )
    tasks = tasks_from_plan(plan)
    router = ProviderRouter({"hybrid": _HybridProvider()})

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert tuple(item.evidence_id for item in result.evidence) == ("c1",)
    assert result.evidence[0].metadata["provider_local_evidence_id"] == "c1"

    citations = build_citations(
        "Supplier disruption risk is supported [c1].",
        list(result.evidence),
        confidence="supported",
    )

    assert [item["citation_id"] for item in citations] == ["c1"]
    assert [item["chunk_id"] for item in citations] == ["chk_1"]


def test_provider_router_rejects_internal_lane_registration() -> None:
    with pytest.raises(ValueError, match="internal_lane_registered_as_provider:bm25"):
        ProviderRouter({"bm25": object()})


def test_provider_router_rejects_provider_without_abc_inheritance() -> None:
    with pytest.raises(TypeError, match="provider_must_inherit_retrieval_provider:hybrid"):
        ProviderRouter({"hybrid": object()})


def test_provider_router_rejects_provider_name_mismatch() -> None:
    class _MismatchedProvider(_SyncProvider):
        provider_name = "graph"

    with pytest.raises(ValueError, match="provider_name_mismatch:hybrid:graph"):
        ProviderRouter({"hybrid": _MismatchedProvider()})


def test_provider_router_aretrieve_runs_provider_groups_concurrently() -> None:
    plan = QueryPlan(
        plan_id="plan_parallel",
        original_query="Explain supplier risk with graph context.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
            RetrievalUnit(
                unit_id="u_graph",
                purpose="graph",
                text="supplier graph",
                provider="graph",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid", "graph"))
    active_counter = {"current": 0, "max": 0, "lock": threading.Lock()}
    router = ProviderRouter(
        {
            "hybrid": _SleepingProvider("hybrid", delay=0.08, active_counter=active_counter),
            "graph": _SleepingProvider("graph", delay=0.08, active_counter=active_counter),
        },
        session_factory=lambda: _ThreadRecordingSession([]),
    )

    result = asyncio.run(
        router.aretrieve(
            _DB(),
            query=plan.original_query,
            top_k=5,
            filters={},
            options={},
            query_plan=plan,
            retrieval_tasks=tasks,
        )
    )

    assert active_counter["max"] == 2
    assert {item.provider for item in result.provider_results} == {"hybrid", "graph"}
    assert result.trace["cross_provider_fusion"] is True


def test_provider_router_session_factory_lifecycle_stays_in_worker_thread() -> None:
    plan = QueryPlan(
        plan_id="plan_thread_session",
        original_query="Explain supplier risk.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid",))
    events: list[tuple[str, int]] = []
    router = ProviderRouter(
        {"hybrid": _ThreadRecordingProvider(events)},
        session_factory=lambda: _ThreadRecordingSession(events),
    )

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=5,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert result.provider_results[0].status == "empty"
    assert [event for event, _thread_id in events] == [
        "created",
        "used",
        "closed",
        "created",
        "closed",
    ]
    provider_thread_ids = {_thread_id for event, _thread_id in events[:3]}
    assembly_thread_ids = {_thread_id for event, _thread_id in events[3:]}
    assert len(provider_thread_ids) == 1
    assert len(assembly_thread_ids) == 1


def test_provider_router_global_reranker_reorders_cross_provider_candidates() -> None:
    plan = QueryPlan(
        plan_id="plan_global_rerank",
        original_query="Explain supplier risk with graph context.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
            RetrievalUnit(
                unit_id="u_graph",
                purpose="graph",
                text="supplier graph",
                provider="graph",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid", "graph"))
    router = ProviderRouter(
        {
            "hybrid": _CandidateProvider("hybrid", "chk_hybrid", local_rank=1),
            "graph": _CandidateProvider("graph", "chk_graph", local_rank=2),
        },
        reranker=_ReverseReranker(),
        reranker_enabled=True,
        reranker_top_k=5,
        reranker_output_k=5,
    )

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=5,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert [item.chunk_id for item in result.evidence] == ["chk_graph", "chk_hybrid"]
    assert result.evidence[0].metadata["rerank_score"] == 1.0


def test_provider_router_preserves_nonrerankable_structured_candidate_with_provenance() -> None:
    plan = QueryPlan(
        plan_id="plan_structured_candidate_policy",
        original_query="What was the calculated metric and supporting text?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text_and_structured_support",
                text="calculated metric supporting text",
                provider="hybrid",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid",))
    structured_payload = {
        "statement": "SELECT metric_value FROM finance_metrics",
        "value": "42.0",
        "unit": "USD millions",
    }
    sql_like_candidate = Candidate(
        candidate_id="cand_sql_metric",
        chunk_id="sql_metric_1",
        document_id="doc_metric",
        doc_name="Metric table",
        source_title="Metric table",
        company="Apple",
        text="Structured metric result: 42.0 USD millions.",
        page_start=None,
        page_end=None,
        chunk_index=1,
        token_count=6,
        retrieved_by=("sql",),
        dense_rank=None,
        dense_score=None,
        fusion_rank=1,
        fusion_score=1.0,
        final_rank=1,
        metadata={
            "provider": "sql",
            "source_type": "structured_result",
            "rerankable": False,
            "fusion_policy": "pinned",
            "structured_payload": structured_payload,
        },
        provider="sql",
    )
    text_candidate = _candidate_for_test("hybrid", "chk_text", 1)
    supporting_candidate = replace(
        _candidate_for_test("hybrid", "chk_supporting", 2),
        fusion_policy="supporting",
        metadata={"provider": "hybrid", "fusion_policy": "supporting"},
    )
    provider = _CandidateProvider("hybrid", "chk_text", local_rank=1)
    provider.retrieve_provider_result = lambda db, **kwargs: ProviderResult(
        provider="hybrid",
        task_id=tasks[0].task_id,
        unit_id=tasks[0].unit_id,
        status="executed",
        candidates=(text_candidate, sql_like_candidate, supporting_candidate),
        latency_ms=1,
        trace={"provider": "hybrid", "status": "executed"},
    )
    reranker = _RecordingReranker()
    router = ProviderRouter(
        {"hybrid": provider},
        reranker=reranker,
        reranker_enabled=True,
        reranker_top_k=5,
        reranker_output_k=1,
    )

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=1,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert reranker.seen_chunk_ids == [["chk_text"]]
    assert [item.chunk_id for item in result.evidence] == [
        "sql_metric_1",
        "chk_text",
        "chk_supporting",
    ]
    structured_evidence = result.evidence[0]
    assert structured_evidence.metadata["source_type"] == "structured_result"
    assert structured_evidence.metadata["rerankable"] is False
    assert structured_evidence.metadata["fusion_policy"] == "pinned"
    assert structured_evidence.metadata["structured_payload"] == structured_payload
    provenance = structured_evidence.metadata["provider_provenance"]
    assert provenance[0]["provider"] == "hybrid"
    assert provenance[0]["provider_local_provider"] == "hybrid"
    assert provenance[0]["candidate_provider"] == "hybrid"
    assert provenance[0]["reported_provider"] == "sql"
    assert structured_evidence.metadata["candidate_provider"] == "hybrid"
    assert structured_evidence.metadata["reported_provider"] == "sql"
    assert result.evidence_pack is not None
    assert result.evidence_pack.blocks[0].source_type == "structured_result"
    assert result.evidence_pack.blocks[0].provider == "hybrid"
    assert result.evidence[2].metadata["fusion_policy"] == "supporting"


def test_candidate_adapter_does_not_mutate_existing_provider_provenance() -> None:
    original_provenance = [{"provider": "legacy", "chunk_id": "chk_legacy"}]
    candidate = replace(
        _candidate_for_test("hybrid", "chk_adapter", 1),
        metadata={"provider": "text_hybrid", "provider_provenance": original_provenance},
        provider="text_hybrid",
    )
    result = ProviderResult(
        provider="hybrid",
        task_id="rt_adapter",
        unit_id="u_adapter",
        status="executed",
        candidates=(candidate,),
    )

    first = candidates_from_provider_result(result)
    second = candidates_from_provider_result(result)

    assert candidate.metadata["provider"] == "text_hybrid"
    assert candidate.metadata["provider_provenance"] == original_provenance
    assert len(first[0].metadata["provider_provenance"]) == 2
    assert len(second[0].metadata["provider_provenance"]) == 2
    assert first[0].metadata["provider"] == "hybrid"
    assert first[0].metadata["candidate_provider"] == "text_hybrid"


def test_provider_router_normalizes_spoofed_provider_result_identity() -> None:
    class _SpoofingProvider(_SyncProvider):
        provider_name = "hybrid"

        def retrieve_provider_result(self, db, **kwargs):
            candidate = replace(
                _candidate_for_test("hybrid", "chk_spoof", 1),
                metadata={
                    "provider": "sql",
                    "source_type": "structured_result",
                    "rerankable": False,
                    "fusion_policy": "pinned",
                },
                provider="sql",
            )
            return ProviderResult(
                provider="sql",
                task_id=kwargs["retrieval_tasks"][0].task_id,
                unit_id=kwargs["retrieval_tasks"][0].unit_id,
                status="executed",
                candidates=(candidate,),
                trace={"provider": "sql", "status": "executed"},
            )

    plan = QueryPlan(
        plan_id="plan_spoof_provider",
        original_query="Return a structured metric.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text",
                text="structured metric",
                provider="hybrid",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid",))
    router = ProviderRouter({"hybrid": _SpoofingProvider()})

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert result.provider_results[0].provider == "hybrid"
    assert result.provider_results[0].trace["provider_reported"] == "sql"
    assert result.evidence_pack is not None
    assert result.evidence_pack.blocks[0].provider == "hybrid"
    assert result.evidence[0].metadata["provider"] == "hybrid"
    assert result.evidence[0].metadata["candidate_provider"] == "hybrid"
    assert result.evidence[0].metadata["reported_provider"] == "sql"
    provenance_anchor = result.evidence[0].metadata["provider_provenance"][0]["source_anchor"]
    assert provenance_anchor["metadata"]["provider"] == "hybrid"
    assert provenance_anchor["metadata"]["reported_provider"] == "sql"
    trace_candidate = result.trace["provider_results"][0]["candidates"][0]
    assert trace_candidate["provider"] == "hybrid"
    assert trace_candidate["reported_provider"] == "sql"
    assert trace_candidate["source_anchor"]["metadata"]["provider"] == "hybrid"
    assert trace_candidate["source_anchor"]["metadata"]["reported_provider"] == "sql"


def test_provider_router_requests_reranker_window_from_provider() -> None:
    plan = QueryPlan(
        plan_id="plan_global_rerank_window",
        original_query="Explain supplier risk with enough alternatives.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid",))
    provider = _WindowCandidateProvider()
    router = ProviderRouter(
        {"hybrid": provider},
        reranker=_ReverseReranker(),
        reranker_enabled=True,
        reranker_top_k=3,
        reranker_output_k=1,
    )

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=1,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert provider.top_ks == [3]
    assert [item.chunk_id for item in result.evidence] == ["chk_3"]


def test_provider_router_requests_fusion_window_without_global_reranker() -> None:
    plan = QueryPlan(
        plan_id="plan_fusion_window_without_rerank",
        original_query="Explain supplier risk with enough alternatives.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid",))
    provider = _WindowCandidateProvider()
    router = ProviderRouter(
        {"hybrid": provider},
        reranker_enabled=False,
        reranker_top_k=3,
    )

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=1,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert provider.top_ks == [3]
    assert [item.chunk_id for item in result.evidence] == ["chk_1"]


def test_provider_router_preserves_provider_results_when_global_rerank_fails() -> None:
    plan = QueryPlan(
        plan_id="plan_rerank_failure",
        original_query="Explain supplier risk.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid",))
    router = ProviderRouter(
        {"hybrid": _CandidateProvider("hybrid", "chk_hybrid", local_rank=1)},
        reranker=_FailingReranker(),
        reranker_enabled=True,
    )

    db = _DB()
    result = router.retrieve(
        db,
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert result.evidence == ()
    assert [item.status for item in result.provider_results] == ["executed", "failed"]
    assert result.provider_results[1].reason == "router_assembly_failed"
    assert result.trace["status"] == "failed"
    assert db.rollbacks == 0


def test_provider_router_rolls_back_failed_caller_transaction_after_assembly_error() -> None:
    plan = QueryPlan(
        plan_id="plan_rerank_failed_transaction",
        original_query="Explain supplier risk.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid",))
    router = ProviderRouter(
        {"hybrid": _CandidateProvider("hybrid", "chk_hybrid", local_rank=1)},
        reranker=_FailingReranker(),
        reranker_enabled=True,
    )

    db = _FailedTransactionDB()
    result = router.retrieve(
        db,
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert result.evidence == ()
    assert result.trace["status"] == "failed"
    assert db.rollbacks == 1


def test_provider_router_disables_provider_local_rerank_when_global_reranker_runs() -> None:
    plan = QueryPlan(
        plan_id="plan_disable_provider_rerank",
        original_query="Explain supplier risk with graph context.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid",))
    provider = _OptionsRecordingCandidateProvider("hybrid", "chk_hybrid", local_rank=1)
    router = ProviderRouter(
        {"hybrid": provider},
        reranker=_ReverseReranker(),
        reranker_enabled=True,
        reranker_top_k=5,
        reranker_output_k=5,
    )

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=5,
        filters={},
        options={"reranker_enabled": True},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert result.evidence[0].chunk_id == "chk_hybrid"
    assert provider.options_seen[0]["reranker_enabled"] is False
    assert provider.options_seen[0]["provider_local_reranker_disabled_reason"] == (
        "cross_provider_reranker"
    )


def test_provider_router_global_reranker_respects_standard_disable_option() -> None:
    plan = QueryPlan(
        plan_id="plan_global_rerank_disabled",
        original_query="Explain supplier risk with graph context.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
            RetrievalUnit(
                unit_id="u_graph",
                purpose="graph",
                text="supplier graph",
                provider="graph",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid", "graph"))
    router = ProviderRouter(
        {
            "hybrid": _CandidateProvider("hybrid", "chk_hybrid", local_rank=1),
            "graph": _CandidateProvider("graph", "chk_graph", local_rank=2),
        },
        reranker=_ReverseReranker(),
        reranker_enabled=True,
        reranker_top_k=5,
        reranker_output_k=5,
    )

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=5,
        filters={},
        options={"reranker_enabled": False},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert [item.chunk_id for item in result.evidence] == ["chk_hybrid", "chk_graph"]
    assert result.evidence[0].metadata.get("rerank_score") is None


def test_provider_router_dedupes_same_chunk_when_only_one_candidate_has_parent_id() -> None:
    plan = QueryPlan(
        plan_id="plan_parent_mismatch_dedupe",
        original_query="Explain supplier risk with graph context.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
            RetrievalUnit(
                unit_id="u_graph",
                purpose="graph",
                text="supplier graph",
                provider="graph",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid", "graph"))
    router = ProviderRouter(
        {
            "hybrid": _CandidateProvider(
                "hybrid",
                "chk_shared",
                local_rank=1,
                parent_id=None,
                provenance_provider=["text_hybrid"],
            ),
            "graph": _CandidateProvider(
                "graph",
                "chk_shared",
                local_rank=2,
                parent_id="parent_shared",
            ),
        },
        reranker_enabled=False,
    )

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=5,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert [item.chunk_id for item in result.evidence] == ["chk_shared"]
    assert result.evidence[0].parent_id == "parent_shared"
    assert result.evidence[0].metadata["source_anchor"]["parent_id"] == "parent_shared"
    provenance = result.evidence[0].metadata["provider_provenance"]
    assert {
        item["provider_local_provider"]
        for item in provenance
    } == {"hybrid", "graph"}
    assert result.evidence[0].metadata["prompt_deduped"] is True


def test_provider_router_preserves_graph_source_anchor_on_cross_provider_dedupe() -> None:
    plan = QueryPlan(
        plan_id="plan_graph_anchor_dedupe",
        original_query="Explain supplier graph support.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
            RetrievalUnit(
                unit_id="u_graph",
                purpose="graph",
                text="supplier graph",
                provider="graph",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid", "graph"))
    graph_anchor = {
        "document_id": "doc_graph",
        "chunk_id": "chk_shared",
        "parent_id": "parent_shared",
        "graph_ids": ["edge:supplier:1"],
        "metadata": {"graph_candidate_id": "graph_candidate_1"},
    }
    hybrid_candidate = replace(
        _candidate_for_test("hybrid", "chk_shared", 1),
        parent_id="parent_shared",
        metadata={"provider": "hybrid", "parent_id": "parent_shared"},
    )
    graph_candidate = replace(
        _candidate_for_test("graph", "chk_shared", 2),
        parent_id="parent_shared",
        metadata={
            "provider": "graph",
            "parent_id": "parent_shared",
            "source_anchor": graph_anchor,
        },
    )
    hybrid_provider = _CandidateProvider("hybrid", "chk_shared", local_rank=1)
    hybrid_provider.retrieve_provider_result = lambda db, **kwargs: ProviderResult(
        provider="hybrid",
        task_id=kwargs["retrieval_tasks"][0].task_id,
        unit_id=kwargs["retrieval_tasks"][0].unit_id,
        status="executed",
        candidates=(hybrid_candidate,),
        latency_ms=1,
        trace={"provider": "hybrid", "status": "executed"},
    )
    graph_provider = _CandidateProvider("graph", "chk_shared", local_rank=2)
    graph_provider.retrieve_provider_result = lambda db, **kwargs: ProviderResult(
        provider="graph",
        task_id=kwargs["retrieval_tasks"][0].task_id,
        unit_id=kwargs["retrieval_tasks"][0].unit_id,
        status="executed",
        candidates=(graph_candidate,),
        latency_ms=1,
        trace={"provider": "graph", "status": "executed"},
    )
    router = ProviderRouter(
        {"hybrid": hybrid_provider, "graph": graph_provider},
        reranker_enabled=False,
    )

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=5,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    metadata = result.evidence[0].metadata
    assert metadata["provider"] == "cross_provider"
    assert any(
        anchor.get("graph_ids") == ["edge:supplier:1"]
        for anchor in metadata["source_anchors"]
    )
    assert any(
        item.get("source_anchor", {}).get("graph_ids") == ["edge:supplier:1"]
        for item in metadata["provider_provenance"]
    )


def test_provider_router_parent_dedupe_uses_wider_candidate_window_than_top_k() -> None:
    plan = QueryPlan(
        plan_id="plan_parent_window",
        original_query="Explain supplier graph support.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
            RetrievalUnit(
                unit_id="u_graph",
                purpose="graph",
                text="supplier graph",
                provider="graph",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid", "graph"))
    graph_anchor = {
        "document_id": "doc_graph",
        "chunk_id": "chk_graph",
        "parent_id": "parent_shared",
        "graph_ids": ["edge:supplier:2"],
    }
    hybrid_candidate = replace(
        _candidate_for_test("hybrid", "chk_hybrid", 1),
        parent_id="parent_shared",
        metadata={"provider": "hybrid", "parent_id": "parent_shared"},
    )
    graph_candidate = replace(
        _candidate_for_test("graph", "chk_graph", 2),
        parent_id="parent_shared",
        metadata={
            "provider": "graph",
            "parent_id": "parent_shared",
            "source_anchor": graph_anchor,
        },
    )
    hybrid_provider = _CandidateProvider("hybrid", "chk_hybrid", local_rank=1)
    hybrid_provider.retrieve_provider_result = lambda db, **kwargs: ProviderResult(
        provider="hybrid",
        task_id=kwargs["retrieval_tasks"][0].task_id,
        unit_id=kwargs["retrieval_tasks"][0].unit_id,
        status="executed",
        candidates=(hybrid_candidate,),
        latency_ms=1,
        trace={"provider": "hybrid", "status": "executed"},
    )
    graph_provider = _CandidateProvider("graph", "chk_graph", local_rank=2)
    graph_provider.retrieve_provider_result = lambda db, **kwargs: ProviderResult(
        provider="graph",
        task_id=kwargs["retrieval_tasks"][0].task_id,
        unit_id=kwargs["retrieval_tasks"][0].unit_id,
        status="executed",
        candidates=(graph_candidate,),
        latency_ms=1,
        trace={"provider": "graph", "status": "executed"},
    )
    router = ProviderRouter({"hybrid": hybrid_provider, "graph": graph_provider})

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=1,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    metadata = result.evidence[0].metadata
    assert len(result.evidence) == 1
    assert set(result.evidence[0].child_ids) == {"chk_hybrid", "chk_graph"}
    assert metadata["provider"] == "cross_provider"
    assert any(
        anchor.get("graph_ids") == ["edge:supplier:2"]
        for anchor in metadata["source_anchors"]
    )
    assert all(isinstance(anchor, dict) for anchor in metadata["source_anchors"])


def test_provider_router_flattens_provenance_for_same_parent_different_chunks() -> None:
    plan = QueryPlan(
        plan_id="plan_parent_level_provenance",
        original_query="Explain supplier risk with multiple chunks.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_text",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid",))
    candidate_a = replace(
        _candidate_for_test("hybrid", "chk_a", 1),
        parent_id="parent_shared",
        retrieval_task_id="rt_a",
        retrieval_unit_id="u_text_a",
        metadata={"provider": "hybrid", "parent_id": "parent_shared"},
    )
    candidate_b = replace(
        _candidate_for_test("hybrid", "chk_b", 2),
        parent_id="parent_shared",
        retrieval_task_id="rt_b",
        retrieval_unit_id="u_text_b",
        metadata={"provider": "hybrid", "parent_id": "parent_shared"},
    )
    provider = _CandidateProvider("hybrid", "chk_a", local_rank=1)
    provider.retrieve_provider_result = lambda db, **kwargs: ProviderResult(
        provider="hybrid",
        task_id=None,
        unit_id=None,
        status="executed",
        candidates=(candidate_a, candidate_b),
        latency_ms=1,
        trace={"provider": "hybrid", "status": "executed"},
    )
    router = ProviderRouter({"hybrid": provider}, reranker_enabled=False)

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=5,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert len(result.evidence) == 1
    assert set(result.evidence[0].child_ids) == {"chk_a", "chk_b"}
    provenance = result.evidence[0].metadata["provider_provenance"]
    prompt_provenance = result.evidence[0].metadata["prompt_provider_provenance"]
    assert all(isinstance(item, dict) for item in provenance)
    assert all(isinstance(item, dict) for item in prompt_provenance)
    assert {item.get("retrieval_task_id") for item in provenance} == {"rt_a", "rt_b"}


def test_provider_router_keeps_provenance_for_same_provider_multi_unit_hits() -> None:
    plan = QueryPlan(
        plan_id="plan_multi_unit_same_chunk",
        original_query="Compare supplier and risk context.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_text",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid",))
    provider = _CandidateProvider("hybrid", "chk_shared", local_rank=1)
    provider.retrieve_provider_result = lambda db, **kwargs: ProviderResult(
        provider="hybrid",
        task_id=None,
        unit_id=None,
        status="executed",
        candidates=(
            replace(
                _candidate_for_test("hybrid", "chk_shared", 1),
                retrieval_task_id="rt_1",
                retrieval_unit_id="u_text",
            ),
            replace(
                _candidate_for_test("hybrid", "chk_shared", 2),
                retrieval_task_id="rt_2",
                retrieval_unit_id="u_risk",
            ),
        ),
        latency_ms=1,
        trace={"provider": "hybrid", "status": "executed"},
    )
    router = ProviderRouter({"hybrid": provider})

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=5,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    provenance = result.evidence[0].metadata["provider_provenance"]
    assert {
        item.get("retrieval_task_id")
        for item in provenance
    } == {"rt_1", "rt_2"}


def test_provider_result_candidate_serialization_omits_candidate_text() -> None:
    candidate = Candidate(
        candidate_id="cand_1",
        chunk_id="chk_1",
        document_id="doc_1",
        doc_name="Apple 10-K",
        source_title="Apple 10-K",
        company="Apple",
        text="This candidate text should stay out of provider_results.",
        page_start=10,
        page_end=10,
        chunk_index=1,
        token_count=9,
        retrieved_by=("dense",),
        dense_rank=1,
        dense_score=0.9,
        provider="text_hybrid",
        lane="dense",
    )
    payload = serialize_provider_result(
        ProviderResult(
            provider="hybrid",
            task_id="rt_1",
            unit_id="u0",
            status="executed",
            candidates=(candidate,),
            trace={},
        )
    )

    assert payload["candidates"][0]["chunk_id"] == "chk_1"
    assert payload["candidates"][0]["source_anchor"]["chunk_id"] == "chk_1"
    assert "text" not in payload["candidates"][0]


def test_executable_query_providers_defaults_hybrid_graph_and_filters_sql() -> None:
    assert "graph" in IMPLEMENTED_RUNTIME_PROVIDERS
    assert Settings(openai_api_key=None).query_runtime_executable_providers == "hybrid,graph"
    assert executable_query_providers(Settings(openai_api_key=None)) == ("hybrid", "graph")
    assert executable_query_providers(
        Settings(openai_api_key=None, query_runtime_executable_providers="hybrid,graph")
    ) == ("hybrid", "graph")

    settings = Settings(
        openai_api_key=None,
        query_runtime_executable_providers="hybrid,sql,graph",
    )

    assert executable_query_providers(settings) == ("hybrid", "graph")

    opt_in_settings = Settings(
        openai_api_key=None,
        sql_provider_enabled=True,
        query_runtime_executable_providers="hybrid,sql,graph",
    )

    assert executable_query_providers(opt_in_settings) == ("hybrid", "sql", "graph")


def test_executable_query_providers_stay_within_known_provider_set() -> None:
    settings = Settings(
        openai_api_key=None,
        query_planner_known_providers="hybrid",
        query_runtime_executable_providers="hybrid",
    )

    assert executable_query_providers(settings) == ("hybrid",)


def test_executable_query_providers_rejects_reserved_lanes() -> None:
    settings = Settings(
        openai_api_key=None,
        query_planner_known_providers="hybrid,dense,sql,graph",
        query_runtime_executable_providers="hybrid,dense,sql,graph",
    )

    with pytest.raises(AtlasError) as exc_info:
        executable_query_providers(settings)

    assert exc_info.value.error_code == ErrorCode.CONFIGURATION_ERROR
    assert "reserved lanes: dense" in exc_info.value.error_message
    assert exc_info.value.details["reserved_providers"] == ["dense"]


def test_executable_query_providers_rejects_unknown_provider_names() -> None:
    settings = Settings(
        openai_api_key=None,
        query_runtime_executable_providers="hybrid,grap",
    )

    with pytest.raises(AtlasError) as exc_info:
        executable_query_providers(settings)

    assert exc_info.value.error_code == ErrorCode.CONFIGURATION_ERROR
    assert "unknown providers: grap" in exc_info.value.error_message
    assert exc_info.value.details["unknown_providers"] == ["grap"]


def test_provider_router_executes_registered_graph_and_trace_events_are_graph() -> None:
    plan = QueryPlan(
        plan_id="plan_graph",
        original_query="Show LocalCo supplier relationships.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_graph",
                purpose="graph_context",
                text="LocalCo supplier relationships",
                provider="graph",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid", "graph"))
    provider = _GraphProvider()
    router = ProviderRouter({"graph": provider})

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )
    retrieval_events = make_retrieval_events(
        query_id="q_graph",
        evidence=list(result.evidence),
    )

    assert len(provider.calls) == 1
    assert tasks[0].provider_status == "ready"
    assert result.provider_results[0].provider == "graph"
    assert result.provider_results[0].status == "executed"
    assert result.evidence[0].metadata["provider"] == "graph"
    assert retrieval_events[0].retriever_type == "graph"
    assert result.trace["status"] == "executed"


def test_provider_router_does_not_execute_graph_without_graph_task() -> None:
    plan = QueryPlan(
        plan_id="plan_hybrid_only_with_graph_registered",
        original_query="Explain Apple supplier disruption risk.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="risk_factor_text",
                text="Apple supplier disruption risk",
                provider="hybrid",
            ),
        ),
    )
    tasks = tasks_from_plan(plan)
    graph_provider = _GraphProvider()
    router = ProviderRouter({"hybrid": _HybridProvider(), "graph": graph_provider})

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert graph_provider.calls == []
    assert [item.provider for item in result.provider_results] == ["hybrid"]
    assert result.trace["executable_providers"] == ["hybrid", "graph"]
    assert result.trace["status"] == "executed"


def test_ready_graph_task_without_registered_provider_has_distinct_reason() -> None:
    plan = QueryPlan(
        plan_id="plan_graph_missing_provider",
        original_query="Show LocalCo supplier relationships.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_graph",
                purpose="graph_context",
                text="LocalCo supplier relationships",
                provider="graph",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid", "graph"))
    router = ProviderRouter({})

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    provider_result = result.provider_results[0]
    assert tasks[0].provider_status == "ready"
    assert provider_result.status == "skipped_non_executable"
    assert provider_result.reason == "provider_not_registered:graph"
    assert provider_result.trace["reason"] == "provider_not_registered:graph"
    assert result.trace["provider_results"][0]["reason"] == "provider_not_registered:graph"


def test_provider_router_respects_hybrid_only_graph_task_even_if_graph_registered() -> None:
    plan = QueryPlan(
        plan_id="plan_graph_hybrid_only",
        original_query="Show LocalCo supplier relationships.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_graph",
                purpose="graph_context",
                text="LocalCo supplier relationships",
                provider="graph",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("hybrid",))
    graph_provider = _GraphProvider()
    router = ProviderRouter({"hybrid": _HybridProvider(), "graph": graph_provider})

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert graph_provider.calls == []
    assert result.evidence == ()
    assert result.provider_results[0].provider == "graph"
    assert result.provider_results[0].status == "skipped_non_executable"
    assert result.provider_results[0].reason == "provider_not_executable:graph"


def test_dependency_provider_router_registers_graph_by_default(monkeypatch) -> None:
    hybrid_provider = _HybridProvider()
    graph_provider = _GraphProvider()

    monkeypatch.setattr(
        dependency_module,
        "get_settings",
        lambda: Settings(openai_api_key=None),
    )
    monkeypatch.setattr(
        dependency_module,
        "get_text_hybrid_provider",
        lambda: hybrid_provider,
    )
    monkeypatch.setattr(
        dependency_module,
        "get_graph_provider",
        lambda: graph_provider,
    )
    dependency_module.get_provider_router.cache_clear()

    default_router = dependency_module.get_provider_router()

    assert default_router.executable_providers == ("hybrid", "graph")
    assert default_router.providers["hybrid"] is hybrid_provider
    assert default_router.providers["graph"] is graph_provider

    monkeypatch.setattr(
        dependency_module,
        "get_settings",
        lambda: Settings(openai_api_key=None, query_runtime_executable_providers="hybrid"),
    )
    dependency_module.get_provider_router.cache_clear()

    hybrid_only_router = dependency_module.get_provider_router()

    assert hybrid_only_router.executable_providers == ("hybrid",)
    assert hybrid_only_router.providers == {"hybrid": hybrid_provider}
    dependency_module.get_provider_router.cache_clear()


def test_dependency_graph_provider_uses_postgres_store_and_context_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        dependency_module,
        "get_settings",
        lambda: Settings(openai_api_key=None, max_context_tokens=1234),
    )
    dependency_module.get_graph_provider.cache_clear()

    provider = dependency_module.get_graph_provider()

    assert isinstance(provider, GraphProvider)
    assert isinstance(provider.store, PostgresGraphStore)
    assert provider.max_context_tokens == 1234
    dependency_module.get_graph_provider.cache_clear()


def test_query_runtime_auto_wires_graph_by_default() -> None:
    runtime = QueryRuntime(
        settings=Settings(
            openai_api_key=None,
            cache_enabled=False,
            max_context_tokens=4321,
        ),
        generator=_Generator(),
    )

    assert runtime.provider_router.executable_providers == ("hybrid", "graph")
    assert "hybrid" in runtime.provider_router.providers
    graph_provider = runtime.provider_router.providers["graph"]
    assert isinstance(graph_provider, GraphProvider)
    assert isinstance(graph_provider.store, PostgresGraphStore)
    assert graph_provider.max_context_tokens == 4321


def test_query_runtime_legacy_retriever_uses_caller_db_and_disables_auto_graph() -> None:
    plan = QueryPlan(
        plan_id="plan_legacy_retriever",
        original_query="Explain supplier risk.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="text",
                text="supplier risk",
                provider="hybrid",
            ),
        ),
    )
    db = _DB()
    retriever = _LegacyRecordingRetriever()
    runtime = QueryRuntime(
        settings=Settings(
            openai_api_key=None,
            cache_enabled=False,
            reranker_enabled=False,
            query_planner_known_providers="graph",
            query_runtime_executable_providers="graph",
        ),
        retriever=retriever,
        generator=_Generator(),
        orchestrator=_StaticOrchestrator(plan),
    )

    runtime.run(db, query=plan.original_query, top_k=3, filters={}, options={})

    assert retriever.db_seen is db
    assert runtime.provider_router.executable_providers == ("hybrid",)
    assert runtime.provider_router.known_providers == ("hybrid", "graph")


def test_query_runtime_auto_wire_respects_narrow_known_provider_set() -> None:
    runtime = QueryRuntime(
        settings=Settings(
            openai_api_key=None,
            cache_enabled=False,
            query_planner_known_providers="hybrid",
        ),
        generator=_Generator(),
    )

    assert runtime.provider_router.known_providers == ("hybrid",)
    assert runtime.provider_router.executable_providers == ("hybrid",)
    assert "graph" not in runtime.provider_router.providers


def test_query_runtime_auto_wired_hybrid_preserves_explicit_rerank() -> None:
    runtime = QueryRuntime(
        settings=Settings(
            openai_api_key=None,
            cache_enabled=False,
            reranker_enabled=False,
        ),
        generator=_Generator(),
    )

    hybrid_provider = runtime.provider_router.providers["hybrid"]

    assert hybrid_provider.reranker is not None
    assert hybrid_provider.reranker_enabled is False
    assert hybrid_provider.mode_switcher.hybrid_rerank_retriever.reranker is not None
    assert hybrid_provider.mode_switcher.hybrid_rrf_retriever.reranker is None


def test_query_runtime_planner_uses_actual_router_executable_providers() -> None:
    plan = QueryPlan(
        plan_id="plan_hybrid_actual_router",
        original_query="Explain Apple supplier disruption risk.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="risk_factor_text",
                text="Apple supplier disruption risk",
                provider="hybrid",
            ),
        ),
    )
    orchestrator = _StaticOrchestrator(plan)
    runtime = QueryRuntime(
        settings=Settings(openai_api_key=None, cache_enabled=False),
        provider_router=ProviderRouter({"hybrid": _HybridProvider()}),
        generator=_Generator(),
        orchestrator=orchestrator,
    )

    runtime.run(_DB(), query=plan.original_query, top_k=3, filters={}, options={})

    assert orchestrator.executable_provider_calls == [("hybrid",)]


def test_runtime_does_not_backfill_pure_sql_plan_as_hybrid() -> None:
    plan = QueryPlan(
        plan_id="plan_sql",
        original_query="What was Apple's 2023 R&D expense?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_sql",
                purpose="structured_lookup",
                text="Apple R&D expense 2023",
                provider="sql",
            ),
        ),
    )
    provider = _HybridProvider()
    runtime = QueryRuntime(
        settings=Settings(openai_api_key=None, cache_enabled=False),
        provider_router=ProviderRouter({"hybrid": provider}),
        generator=_Generator(),
        orchestrator=_StaticOrchestrator(plan),
    )

    result = runtime.run(_DB(), query=plan.original_query, top_k=3, filters={}, options={})

    assert provider.calls == []
    assert result.confidence == "insufficient"
    assert result.details["provider_results"][0]["status"] == "skipped_non_executable"
    assert result.details["retrieval_trace"]["evidence_count"] == 0


def test_runtime_marks_llm_io_skipped_on_cache_hit() -> None:
    plan = QueryPlan(
        plan_id="plan_cache_hit",
        original_query="Explain cached supplier risk.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="risk_factor_text",
                text="cached supplier risk",
                provider="hybrid",
            ),
        ),
    )
    cache_record = QueryCache(
        key="cached_key",
        answer="Cached answer [c1].",
        confidence="supported",
        citations_json=[{"citation_id": "c1", "document_id": "doc_1"}],
        hit_count=0,
    )
    db = _CacheHitDB(cache_record)
    provider = _HybridProvider()
    runtime = QueryRuntime(
        settings=Settings(openai_api_key=None, cache_enabled=True),
        provider_router=ProviderRouter({"hybrid": provider}),
        generator=_Generator(),
        orchestrator=_StaticOrchestrator(plan),
    )

    result = runtime.run(db, query=plan.original_query, top_k=3, filters={}, options={})

    query_run = next(item for item in db.added if item.__tablename__ == "query_runs")
    llm_io = {"status": "skipped", "reason": "cache_hit"}

    assert db.cache_gets
    assert provider.calls == []
    assert result.answer == "Cached answer [c1]."
    assert result.details["llm_io"] == llm_io
    assert query_run.details_json["llm_io"] == llm_io
    assert {"request", "response"}.isdisjoint(result.details["llm_io"])


def test_runtime_persists_successful_planner_llm_call_without_raw_details_payload() -> None:
    plan = QueryPlan(
        plan_id="plan_planner_success",
        original_query="Explain Apple supplier disruption risk.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="risk_factor_text",
                text="Apple supplier disruption risk",
                provider="hybrid",
            ),
        ),
        planner="llm_structured",
    )
    observability = _planner_observability_payload(
        call_id="llmc_planner_success",
        plan_id=plan.plan_id,
    )
    db = _DB()
    runtime = QueryRuntime(
        settings=Settings(openai_api_key=None, cache_enabled=False),
        provider_router=ProviderRouter({"hybrid": _HybridProvider()}),
        generator=_Generator(),
        orchestrator=_StaticOrchestrator(plan, observability),
    )

    result = runtime.run(db, query=plan.original_query, top_k=3, filters={}, options={})

    query_run = next(item for item in db.added if item.__tablename__ == "query_runs")
    query_plan_record = next(item for item in db.added if item.__tablename__ == "query_plans")
    planner_call = next(
        item
        for item in db.added
        if item.__tablename__ == "llm_calls" and item.stage == "planner"
    )
    answer_call = next(
        item
        for item in db.added
        if item.__tablename__ == "llm_calls" and item.stage == "answer"
    )
    raw_prompt = "RAW PLANNER PROMPT SENTINEL llmc_planner_success"
    raw_response = "RAW PLANNER RESPONSE SENTINEL llmc_planner_success"

    assert result.details["query_plan"]["plan_id"] == plan.plan_id
    assert planner_call.call_id == "llmc_planner_success"
    assert planner_call.stage == "planner"
    assert planner_call.status == "completed"
    assert planner_call.validation_status == "validated"
    assert planner_call.parsed_plan_id == plan.plan_id
    assert planner_call.input_tokens == 31
    assert planner_call.output_tokens == 13
    assert planner_call.request_json["input"] == raw_prompt
    assert planner_call.raw_output_text == raw_response
    assert planner_call.raw_payload_hash
    assert planner_call.raw_retention_expires_at is not None
    assert query_plan_record.planner_call_id == planner_call.call_id
    assert query_plan_record.payload_json["metadata"]["planner_llm_call_id"] == (
        planner_call.call_id
    )
    assert query_run.details_json["planner_llm"] == {
        "planner_llm_call_id": planner_call.call_id,
        "planner_llm_status": "completed",
        "planner_validation_status": "validated",
    }
    assert answer_call.stage == "answer"
    assert raw_prompt not in _json_blob(query_run.details_json)
    assert raw_response not in _json_blob(query_run.details_json)
    assert raw_prompt not in _json_blob(query_plan_record.payload_json)
    assert raw_response not in _json_blob(query_plan_record.payload_json)
    assert {"request", "response", "raw_output", "instructions", "input"}.isdisjoint(
        _nested_keys(query_plan_record.payload_json["metadata"])
    )


def test_runtime_persists_invalid_planner_call_on_fallback_not_quality_run() -> None:
    plan = QueryPlan(
        plan_id="plan_planner_fallback",
        original_query="Explain Apple supplier disruption risk.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="risk_factor_text",
                text="Apple supplier disruption risk",
                provider="hybrid",
            ),
        ),
        planner="rule_based_fallback",
        metadata={
            "fallback_reason": "llm_validation_failed",
            "quality_eligible": False,
            "not_quality_reason": "planner_fallback_not_quality_run",
        },
    )
    observability = _planner_observability_payload(
        call_id="llmc_planner_invalid",
        plan_id="plan_bad_llm",
        status="invalid",
        validation_status="invalid",
        error_message="ungrounded_entity:Imaginary Corp",
    )
    db = _DB()
    runtime = QueryRuntime(
        settings=Settings(openai_api_key=None, cache_enabled=False),
        provider_router=ProviderRouter({"hybrid": _EmptyProvider("hybrid")}),
        generator=_Generator(),
        orchestrator=_StaticOrchestrator(plan, observability),
    )

    result = runtime.run(db, query=plan.original_query, top_k=3, filters={}, options={})

    query_run = next(item for item in db.added if item.__tablename__ == "query_runs")
    query_plan_record = next(item for item in db.added if item.__tablename__ == "query_plans")
    planner_call = next(
        item
        for item in db.added
        if item.__tablename__ == "llm_calls" and item.stage == "planner"
    )
    raw_prompt = "RAW PLANNER PROMPT SENTINEL llmc_planner_invalid"
    raw_response = "RAW PLANNER RESPONSE SENTINEL llmc_planner_invalid"

    assert result.confidence == "insufficient"
    assert planner_call.call_id == "llmc_planner_invalid"
    assert planner_call.status == "invalid"
    assert planner_call.validation_status == "invalid"
    assert planner_call.error_message == "ungrounded_entity:Imaginary Corp"
    assert query_plan_record.planner_call_id == planner_call.call_id
    metadata = query_plan_record.payload_json["metadata"]
    assert metadata["fallback_reason"] == "llm_validation_failed"
    assert metadata["quality_eligible"] is False
    assert metadata["not_quality_reason"] == "planner_fallback_not_quality_run"
    assert metadata["planner_llm_call_id"] == planner_call.call_id
    assert metadata["planner_llm_status"] == "invalid"
    assert query_run.details_json["planner_llm"]["planner_llm_status"] == "invalid"
    assert raw_prompt not in _json_blob(query_run.details_json)
    assert raw_response not in _json_blob(query_run.details_json)
    assert raw_prompt not in _json_blob(query_plan_record.payload_json)
    assert raw_response not in _json_blob(query_plan_record.payload_json)
    assert not [
        item
        for item in db.added
        if item.__tablename__ == "llm_calls" and item.stage == "answer"
    ]


def test_runtime_persists_llm_io_to_query_run_and_answer_record() -> None:
    raw_output = '{"answer":"Apple discussed supplier disruption risk [c1]."}'

    class _RawOutputGenerator:
        model_name = "fake-generator"

        def generate(self, *, query, evidence):
            return GeneratedAnswer(
                answer="Apple discussed supplier disruption risk [c1].",
                confidence="supported",
                usage=LLMUsage(input_tokens=12, output_tokens=7),
                raw_output=raw_output,
            )

    plan = QueryPlan(
        plan_id="plan_hybrid",
        original_query="Explain Apple supplier disruption risk.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="risk_factor_text",
                text="Apple supplier disruption risk",
                provider="hybrid",
            ),
        ),
    )
    db = _DB()
    runtime = QueryRuntime(
        settings=Settings(openai_api_key=None, cache_enabled=False),
        provider_router=ProviderRouter({"hybrid": _HybridProvider()}),
        generator=_RawOutputGenerator(),
        orchestrator=_StaticOrchestrator(plan),
    )

    result = runtime.run(db, query=plan.original_query, top_k=3, filters={}, options={})

    query_run = next(item for item in db.added if item.__tablename__ == "query_runs")
    answer_record = next(item for item in db.added if item.__tablename__ == "answers")
    llm_call = next(item for item in db.added if item.__tablename__ == "llm_calls")
    llm_call_evidence = next(
        item for item in db.added if item.__tablename__ == "llm_call_evidence"
    )
    llm_io = result.details["llm_io"]
    request_input_blob = _json_blob(llm_call.request_json["input"])

    assert query_run.details_json["llm_io"] == llm_io
    assert answer_record.answer_call_id == llm_io["answer_llm_call_id"]
    assert answer_record.payload_json["answer_llm_call_id"] == llm_io["answer_llm_call_id"]
    assert "llm_io" not in answer_record.payload_json
    assert llm_io == {
        "status": "completed",
        "answer_llm_call_id": llm_call.call_id,
    }
    assert {"request", "response", "request_metadata"}.isdisjoint(llm_io)
    assert set(llm_call.request_json) == {
        "model",
        "instructions",
        "input",
        "max_output_tokens",
        "reasoning",
        "store",
    }
    assert {"prompt_version", "evidence_ids", "evidence_count"}.isdisjoint(
        llm_call.request_json
    )
    assert llm_call.metadata_json["request_metadata"] == {
        "prompt_version": runtime.settings.prompt_version,
        "evidence_ids": ["c1"],
        "evidence_count": 1,
    }
    assert plan.original_query in request_input_blob
    assert "Apple management discussed supplier disruption risk." in request_input_blob
    assert "c1" in request_input_blob
    assert llm_call.stage == "answer"
    assert llm_call.status == "completed"
    assert llm_call.response_json["raw_output"] == raw_output
    assert llm_call.raw_output_text == raw_output
    assert llm_call.parsed_answer_text == "Apple discussed supplier disruption risk [c1]."
    assert llm_call.parsed_confidence == "supported"
    assert llm_call.input_tokens == 12
    assert llm_call.output_tokens == 7
    assert llm_call.raw_payload_hash
    assert llm_call.raw_redaction_status == "unredacted"
    assert llm_call.raw_encryption_status == "plaintext"
    assert llm_call.raw_retention_expires_at is not None
    assert llm_call_evidence.call_id == llm_call.call_id
    assert llm_call_evidence.evidence_id == "c1"
    assert llm_call_evidence.chunk_id == "chk_1"
    assert llm_call_evidence.text_snapshot == (
        "Apple management discussed supplier disruption risk."
    )
    assert llm_call_evidence.text_hash
    assert llm_call_evidence.snapshot_redaction_status == "unredacted"
    assert llm_call_evidence.snapshot_encryption_status == "plaintext"
    assert llm_call_evidence.snapshot_retention_expires_at is not None
    assert llm_call_evidence.evidence_block_record_id is not None
    assert {"api_key", "authorization", "openai_api_key"}.isdisjoint(
        _nested_keys(llm_call.request_json)
    )


def test_runtime_llm_io_request_matches_openai_answer_generator_client_request() -> None:
    raw_output = json.dumps(
        {
            "confidence": "supported",
            "answer": "Apple discussed supplier disruption risk [c1].",
        }
    )
    fake_client = _RecordingLLMClient(
        output_text=raw_output,
        input_tokens=21,
        output_tokens=9,
    )
    settings = Settings(
        openai_api_key=None,
        cache_enabled=False,
        llm_model="fake-openai-model",
        llm_max_output_tokens=321,
        llm_reasoning_effort="medium",
        prompt_version="prompt-test",
    )
    plan = QueryPlan(
        plan_id="plan_openai_generator",
        original_query="Explain Apple supplier disruption risk.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="risk_factor_text",
                text="Apple supplier disruption risk",
                provider="hybrid",
            ),
        ),
    )
    runtime = QueryRuntime(
        settings=settings,
        provider_router=ProviderRouter({"hybrid": _HybridProvider()}),
        generator=OpenAIAnswerGenerator(settings, client=fake_client),
        orchestrator=_StaticOrchestrator(plan),
    )

    db = _DB()
    result = runtime.run(db, query=plan.original_query, top_k=3, filters={}, options={})

    llm_io = result.details["llm_io"]
    llm_call = next(item for item in db.added if item.__tablename__ == "llm_calls")

    assert len(fake_client.requests) == 1
    assert llm_io == {
        "status": "completed",
        "answer_llm_call_id": llm_call.call_id,
    }
    assert llm_call.request_json == fake_client.requests[0]
    assert set(llm_call.request_json) == {
        "model",
        "instructions",
        "input",
        "max_output_tokens",
        "reasoning",
        "store",
    }
    assert llm_call.metadata_json["request_metadata"] == {
        "prompt_version": "prompt-test",
        "evidence_ids": ["c1"],
        "evidence_count": 1,
    }
    assert llm_call.parsed_answer_text == (
        "Apple discussed supplier disruption risk [c1]."
    )
    assert llm_call.parsed_confidence == "supported"
    assert llm_call.usage_json == {
        "input_tokens": 21,
        "output_tokens": 9,
    }
    assert {"request", "response", "raw_output", "parsed_answer"}.isdisjoint(llm_io)


def test_runtime_dedupes_prompt_evidence_by_chunk_id_and_merges_provider_provenance() -> None:
    plan = QueryPlan(
        plan_id="plan_hybrid_graph_duplicate",
        original_query="Explain Apple supplier disruption risk.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="risk_factor_text",
                text="Apple supplier disruption risk",
                provider="hybrid",
            ),
            RetrievalUnit(
                unit_id="u_graph",
                purpose="supplier_graph_grounding",
                text="Apple supplier disruption risk",
                provider="graph",
            ),
        ),
    )
    db = _DB()
    generator = _RecordingEvidenceGenerator()
    runtime = QueryRuntime(
        settings=Settings(
            openai_api_key=None,
            cache_enabled=False,
            query_runtime_executable_providers="hybrid,graph",
        ),
        provider_router=ProviderRouter(
            {
                "hybrid": _HybridProvider(),
                "graph": _DuplicateChunkGraphProvider(),
            }
        ),
        generator=generator,
        orchestrator=_StaticOrchestrator(plan),
    )

    result = runtime.run(db, query=plan.original_query, top_k=5, filters={}, options={})

    llm_call_evidence = [
        item for item in db.added if item.__tablename__ == "llm_call_evidence"
    ]
    assert generator.evidence is not None
    assert len(generator.evidence) == 1
    assert len(llm_call_evidence) == 1
    assert llm_call_evidence[0].chunk_id == "chk_1"
    prompt_metadata = generator.evidence[0].metadata
    provider_local_provenance = {
        item.get("provider_local_provider")
        for item in prompt_metadata["prompt_provider_provenance"]
    }
    provider_names = set(prompt_metadata["prompt_providers"])
    assert {"hybrid", "graph"} <= provider_local_provenance
    assert {"hybrid", "graph"} <= provider_names
    assert prompt_metadata["prompt_deduped"] is True
    assert set(prompt_metadata["prompt_deduped_evidence_ids"]) == {"c1", "g_local"}
    assert result.details["retrieval_trace"]["evidence_count"] == 1
    assert result.details["retrieval_trace"]["top_k"][0]["prompt_deduped"] is True
    assert result.details["llm_io"]["answer_llm_call_id"]


def test_runtime_sanitizes_failed_provider_trace_on_partial_success() -> None:
    plan = QueryPlan(
        plan_id="plan_partial_failure_trace",
        original_query="Explain Apple supplier disruption risk.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="risk_factor_text",
                text="Apple supplier disruption risk",
                provider="hybrid",
            ),
            RetrievalUnit(
                unit_id="u_graph",
                purpose="supplier_graph_grounding",
                text="Apple supplier disruption risk",
                provider="graph",
            ),
        ),
    )
    runtime = QueryRuntime(
        settings=Settings(openai_api_key=None, cache_enabled=False),
        provider_router=ProviderRouter(
            {
                "hybrid": _HybridProvider(),
                "graph": _FailingProvider(
                    RuntimeError("secret graph connection string leaked")
                ),
            }
        ),
        generator=_Generator(),
        orchestrator=_StaticOrchestrator(plan),
    )

    result = runtime.run(_DB(), query=plan.original_query, top_k=5, filters={}, options={})

    details_blob = _json_blob(result.details)
    assert result.details["provider_results"][1]["status"] == "failed"
    assert result.details["provider_results"][1]["trace"]["error_type"] == "RuntimeError"
    assert "secret graph connection string leaked" not in details_blob
    assert "planned_text" not in result.details["provider_results"][1]["trace"]
    assert "error_message" not in result.details["provider_results"][1]["trace"]


def test_record_v1_trace_family_writes_answer_llm_call_without_raw_answer_payload() -> None:
    call_id = "llmc_answer"
    request = {
        "model": "fake-generator",
        "instructions": "Answer using evidence.",
        "input": (
            "Question: Explain Apple supplier disruption risk.\n"
            "[c1] Apple management discussed supplier disruption risk."
        ),
        "max_output_tokens": 2000,
        "reasoning": {"effort": "low"},
        "store": False,
    }
    observability = {
        "answer_llm_call": {
            "call_id": call_id,
            "stage": "answer",
            "status": "completed",
            "model_name": "fake-generator",
            "prompt_version": "test",
            "request": request,
            "request_metadata": {
                "prompt_version": "test",
                "evidence_ids": ["c1"],
                "evidence_count": 1,
            },
            "response": {
                "raw_output": '{"answer":"Apple discussed supplier disruption risk [c1]."}',
                "parsed_answer": "Apple discussed supplier disruption risk [c1].",
                "parsed_confidence": "supported",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
        "answer_prompt_evidence": [
            {
                "evidence_id": "c1",
                "rank": 1,
                "provider": "text_hybrid",
                "chunk_id": "chk_1",
                "document_id": "doc_1",
                "page_start": 10,
                "page_end": 10,
                "retrieval_score": 1.0,
                "token_count": 8,
                "text_snapshot": "Apple management discussed supplier disruption risk.",
            }
        ],
    }
    legacy_raw_llm_io = {
        "status": "completed",
        "answer_llm_call_id": call_id,
        "request": {
            "input": (
                "Question: Explain Apple supplier disruption risk.\n"
                "[c1] Apple management discussed supplier disruption risk."
            ),
        },
        "response": {
            "raw_output": '{"answer":"Apple discussed supplier disruption risk [c1]."}'
        },
    }
    query_run = QueryRun(
        query_id="q_llm_io",
        trace_id="tr_llm_io",
        user_query="Explain Apple supplier disruption risk.",
        normalized_query="Explain Apple supplier disruption risk.",
        answer="Apple discussed supplier disruption risk [c1].",
        confidence="supported",
        citations_json=[{"citation_id": "c1", "document_id": "doc_1"}],
        model_name="fake-generator",
        prompt_version="test",
        latency_ms=12,
        details_json={
            "llm_io": legacy_raw_llm_io,
            "retrieval_trace": {
                "top_k": [
                    {
                        "evidence_id": "c1",
                        "chunk_id": "chk_1",
                        "rank": 1,
                        "retrieval_score": 1.0,
                    }
                ]
            },
        },
    )
    generation_event = GenerationEvent(
        event_id="gen_llm_io",
        query_id="q_llm_io",
        model_name="fake-generator",
        prompt_version="test",
        status="completed",
    )
    db = _DB()

    record_v1_trace_family(
        db,
        query_run,
        [],
        generation_event,
        observability=observability,
    )

    answer_record = next(item for item in db.added if item.__tablename__ == "answers")
    llm_call = next(item for item in db.added if item.__tablename__ == "llm_calls")
    llm_call_evidence = next(
        item for item in db.added if item.__tablename__ == "llm_call_evidence"
    )
    assert query_run.details_json["llm_io"] == {
        "status": "completed",
        "answer_llm_call_id": call_id,
    }
    assert answer_record.answer_call_id == call_id
    assert answer_record.payload_json == {
        "answer": "Apple discussed supplier disruption risk [c1].",
        "confidence": "supported",
        "model": "fake-generator",
        "prompt_version": "test",
        "generation_event": {
            "event_id": "gen_llm_io",
            "model_name": "fake-generator",
            "prompt_version": "test",
            "input_tokens": None,
            "output_tokens": None,
            "latency_ms": None,
            "status": "completed",
            "error_message": None,
        },
        "answer_llm_call_id": call_id,
    }
    assert llm_call.call_id == call_id
    assert llm_call.request_json == request
    assert llm_call.raw_output_text == (
        '{"answer":"Apple discussed supplier disruption risk [c1]."}'
    )
    assert llm_call.raw_payload_hash
    assert llm_call_evidence.call_id == call_id
    assert llm_call_evidence.chunk_id == "chk_1"
    assert llm_call_evidence.text_snapshot == (
        "Apple management discussed supplier disruption risk."
    )
    assert llm_call_evidence.evidence_block_record_id is not None


def test_runtime_persists_provider_trace_when_generation_fails() -> None:
    class _FailingGenerator:
        model_name = "failing-generator"

        def generate(self, *, query, evidence):
            raise AtlasError(
                ErrorCode.UPSTREAM_LLM_UNAVAILABLE,
                "LLM failed after retrieval.",
                status_code=502,
            )

    plan = QueryPlan(
        plan_id="plan_hybrid",
        original_query="Explain Apple supplier disruption risk.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="risk_factor_text",
                text="Apple supplier disruption risk",
                provider="hybrid",
            ),
        ),
    )
    db = _DB()
    runtime = QueryRuntime(
        settings=Settings(openai_api_key=None, cache_enabled=False),
        provider_router=ProviderRouter({"hybrid": _HybridProvider()}),
        generator=_FailingGenerator(),
        orchestrator=_StaticOrchestrator(plan),
    )

    with pytest.raises(AtlasError):
        runtime.run(db, query=plan.original_query, top_k=3, filters={}, options={})

    query_run = next(item for item in db.added if item.__tablename__ == "query_runs")
    retrieval_result = next(
        item for item in db.added if item.__tablename__ == "retrieval_results"
    )
    answer_record = next(item for item in db.added if item.__tablename__ == "answers")
    llm_call = next(item for item in db.added if item.__tablename__ == "llm_calls")
    llm_call_evidence = next(
        item for item in db.added if item.__tablename__ == "llm_call_evidence"
    )
    assert query_run.details_json["provider_results"][0]["status"] == "executed"
    assert query_run.details_json["retrieval_trace"]["evidence_count"] == 1
    assert query_run.details_json["llm_io"] == {
        "status": "failed",
        "answer_llm_call_id": llm_call.call_id,
        "error_message": "LLM failed after retrieval.",
    }
    assert {"request", "response", "raw_output"}.isdisjoint(
        query_run.details_json["llm_io"]
    )
    assert answer_record.answer_call_id == llm_call.call_id
    assert answer_record.payload_json["answer_llm_call_id"] == llm_call.call_id
    assert "llm_io" not in answer_record.payload_json
    assert llm_call.stage == "answer"
    assert llm_call.status == "failed"
    assert llm_call.error_message == "LLM failed after retrieval."
    assert llm_call.request_json["model"] == runtime.settings.llm_model
    assert llm_call.response_json == {}
    assert llm_call.raw_payload_hash
    assert llm_call.raw_retention_expires_at is not None
    assert llm_call_evidence.call_id == llm_call.call_id
    assert llm_call_evidence.text_snapshot == (
        "Apple management discussed supplier disruption risk."
    )
    assert retrieval_result.payload_json["provider_results"][0]["status"] == "executed"
    assert retrieval_result.payload_json["provider_router_trace"]["status"] == "executed"


@pytest.mark.parametrize("failure_kind", ["atlas_error", "generic_error"])
def test_runtime_marks_llm_io_skipped_when_retrieval_fails(failure_kind: str) -> None:
    plan = QueryPlan(
        plan_id="plan_retrieval_failure",
        original_query="Explain Apple supplier disruption risk.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_hybrid",
                purpose="risk_factor_text",
                text="Apple supplier disruption risk",
                provider="hybrid",
            ),
        ),
    )
    exc = (
        AtlasError(
            ErrorCode.UPSTREAM_VECTOR_STORE_UNAVAILABLE,
            "Vector store failed.",
            status_code=502,
        )
        if failure_kind == "atlas_error"
        else RuntimeError("vector store failed")
    )
    db = _DB()
    runtime = QueryRuntime(
        settings=Settings(openai_api_key=None, cache_enabled=False),
        provider_router=ProviderRouter({"hybrid": _FailingProvider(exc, provider="hybrid")}),
        generator=_Generator(),
        orchestrator=_StaticOrchestrator(plan),
    )

    with pytest.raises(AtlasError) as raised:
        runtime.run(db, query=plan.original_query, top_k=3, filters={}, options={})

    query_run = next(item for item in db.added if item.__tablename__ == "query_runs")
    llm_io = {"status": "skipped", "reason": "retrieval_failed"}

    assert query_run.details_json["llm_io"] == llm_io
    assert query_run.details_json["provider_results"][0]["status"] == "failed"
    assert query_run.details_json["provider_router_trace"]["status"] == "failed"
    assert {"request", "response"}.isdisjoint(query_run.details_json["llm_io"])
    stages = {
        stage["name"]: stage["status"]
        for stage in query_run.details_json["trace"]["stages"]
    }
    assert stages["retrieval"] == "failed"
    assert stages["generation"] == "skipped"
    failure_details = raised.value.details["provider_failures"][0]
    assert set(failure_details) == {
        "provider",
        "task_id",
        "unit_id",
        "reason",
        "status",
        "error_type",
    }
    assert "Vector store failed" not in _json_blob(raised.value.details)
    if failure_kind == "generic_error":
        assert raised.value.error_message == "Provider retrieval failed."
