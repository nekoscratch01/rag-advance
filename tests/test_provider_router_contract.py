import pytest

from atlas.core.errors import AtlasError, ErrorCode
from atlas.query_orchestrator.schema import QueryPlan, RetrievalUnit
from atlas.query_runtime.service import QueryRuntime
from atlas.retrieval.contracts import ProviderResult
from atlas.retrieval.models.evidence import Evidence
from atlas.retrieval.models.retrieval_task import tasks_from_plan
from atlas.retrieval.router import ProviderRouter
from atlas.core.config import Settings, executable_query_providers
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


def test_provider_router_rejects_internal_lane_registration() -> None:
    with pytest.raises(ValueError, match="internal_lane_registered_as_provider:bm25"):
        ProviderRouter({"bm25": object()})


def test_executable_query_providers_filters_unimplemented_future_providers() -> None:
    settings = Settings(
        openai_api_key=None,
        query_runtime_executable_providers="hybrid,sql,graph",
    )

    assert executable_query_providers(settings) == ("hybrid",)


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
