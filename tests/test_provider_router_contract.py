import pytest

from atlas.api import dependencies as dependency_module
from atlas.core.config import (
    IMPLEMENTED_RUNTIME_PROVIDERS,
    Settings,
    executable_query_providers,
)
from atlas.core.errors import AtlasError, ErrorCode
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
from atlas.llm.base import GeneratedAnswer, LLMUsage


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
    def __init__(self, plan: QueryPlan) -> None:
        self.plan_value = plan

    def plan(self, query, *, use_llm=True):
        return self.plan_value


class _Generator:
    model_name = "fake-generator"

    def generate(self, *, query, evidence):
        return GeneratedAnswer(
            answer="Apple discussed supplier disruption risk [c1].",
            confidence="supported",
            usage=LLMUsage(input_tokens=1, output_tokens=1),
            raw_output="{}",
        )


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
    assert query_run.details_json["provider_results"][0]["status"] == "executed"
    assert query_run.details_json["retrieval_trace"]["evidence_count"] == 1
    assert retrieval_result.payload_json["provider_results"][0]["status"] == "executed"
    assert retrieval_result.payload_json["provider_router_trace"]["status"] == "executed"
