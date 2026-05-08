import json
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
from atlas.retrieval.models.retrieval_task import tasks_from_plan
from atlas.retrieval.providers.graph import GraphProvider, PostgresGraphStore
from atlas.retrieval.router import ProviderRouter, serialize_provider_result


class _HybridProvider:
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


class _TopKRecordingHybridProvider:
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


class _GraphProvider:
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


class _DuplicateChunkGraphProvider:
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


class _EmptyProvider:
    def __init__(self, provider: str) -> None:
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


class _StaticOrchestrator:
    def __init__(self, plan: QueryPlan, observability=None) -> None:
        self.plan_value = plan
        self.last_observability = observability or {}

    def plan(self, query, *, use_llm=True):
        return self.plan_value

    def plan_with_observability(self, query, *, use_llm=True):
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


class _DB:
    def __init__(self) -> None:
        self.added = []
        self.commits = 0

    def add(self, value) -> None:
        self.added.append(value)

    def flush(self) -> None:
        return None

    def commit(self) -> None:
        self.commits += 1


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


class _FailingProvider:
    def __init__(self, exc: Exception) -> None:
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
        _DB(),
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
    assert result.provider_results[0].reason == "provider_not_executable_in_v1:sql"
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

    result = router.retrieve(
        _DB(),
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
    positive_router = ProviderRouter({"hybrid": positive_provider})
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


def test_provider_router_respects_task_non_executable_status_even_if_registered() -> None:
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
    sql_provider = _HybridProvider()
    router = ProviderRouter({"hybrid": _HybridProvider(), "sql": sql_provider})

    result = router.retrieve(
        _DB(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert sql_provider.calls == []
    assert result.evidence == ()
    assert result.provider_results[0].provider == "sql"
    assert result.provider_results[0].status == "skipped_non_executable"


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

    result = router.retrieve(
        _DB(),
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


def test_executable_query_providers_defaults_hybrid_and_filters_sql() -> None:
    assert "graph" in IMPLEMENTED_RUNTIME_PROVIDERS
    assert Settings(openai_api_key=None).query_runtime_executable_providers == "hybrid"
    assert executable_query_providers(Settings(openai_api_key=None)) == ("hybrid",)
    assert executable_query_providers(
        Settings(openai_api_key=None, query_runtime_executable_providers="hybrid,graph")
    ) == ("hybrid", "graph")

    settings = Settings(
        openai_api_key=None,
        query_runtime_executable_providers="hybrid,sql,graph",
    )

    assert executable_query_providers(settings) == ("hybrid", "graph")


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


def test_dependency_provider_router_registers_graph_only_when_opted_in(monkeypatch) -> None:
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

    assert default_router.executable_providers == ("hybrid",)
    assert default_router.providers == {"hybrid": hybrid_provider}

    monkeypatch.setattr(
        dependency_module,
        "get_settings",
        lambda: Settings(
            openai_api_key=None,
            query_runtime_executable_providers="hybrid,graph",
        ),
    )
    dependency_module.get_provider_router.cache_clear()

    graph_router = dependency_module.get_provider_router()

    assert graph_router.executable_providers == ("hybrid", "graph")
    assert graph_router.providers["hybrid"] is hybrid_provider
    assert graph_router.providers["graph"] is graph_provider
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


def test_query_runtime_auto_wires_graph_when_explicitly_executable() -> None:
    runtime = QueryRuntime(
        settings=Settings(
            openai_api_key=None,
            cache_enabled=False,
            max_context_tokens=4321,
            query_runtime_executable_providers="hybrid,graph",
        ),
        generator=_Generator(),
    )

    provider = runtime.provider_router.providers["graph"]
    assert isinstance(provider, GraphProvider)
    assert isinstance(provider.store, PostgresGraphStore)
    assert provider.max_context_tokens == 4321


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
    assert {"text_hybrid", "graph"} <= provider_names
    assert prompt_metadata["prompt_deduped"] is True
    assert set(prompt_metadata["prompt_deduped_evidence_ids"]) == {"c1", "c2"}
    assert result.details["retrieval_trace"]["evidence_count"] == 1
    assert result.details["retrieval_trace"]["top_k"][0]["prompt_deduped"] is True
    assert result.details["llm_io"]["answer_llm_call_id"]


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
        provider_router=ProviderRouter({"hybrid": _FailingProvider(exc)}),
        generator=_Generator(),
        orchestrator=_StaticOrchestrator(plan),
    )

    with pytest.raises(AtlasError) as raised:
        runtime.run(db, query=plan.original_query, top_k=3, filters={}, options={})

    query_run = next(item for item in db.added if item.__tablename__ == "query_runs")
    llm_io = {"status": "skipped", "reason": "retrieval_failed"}

    assert query_run.details_json["llm_io"] == llm_io
    assert {"request", "response"}.isdisjoint(query_run.details_json["llm_io"])
    stages = {
        stage["name"]: stage["status"]
        for stage in query_run.details_json["trace"]["stages"]
    }
    assert stages["retrieval"] == "failed"
    assert stages["generation"] == "skipped"
    if failure_kind == "generic_error":
        assert raised.value.error_message == "Provider retrieval failed."
