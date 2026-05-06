from fastapi.testclient import TestClient

from atlas.core.config import Settings
from atlas.main import create_app
from atlas.query_orchestrator.schema import QueryPlan, RetrievalUnit
from atlas.query_runtime.cache import make_cache_key
from atlas.query_runtime.service import _retrieve_evidence
from atlas.retrieval.evidence import Evidence
from atlas.retrieval.retrieval_task import tasks_from_plan


class _PlanAwareRetriever:
    def __init__(self) -> None:
        self.received_plan = None
        self.received_tasks = None

    def retrieve_with_plan(
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
        self.received_plan = query_plan
        self.received_tasks = retrieval_tasks
        return [
            Evidence(
                evidence_id="c1",
                document_id="doc_1",
                chunk_id="chk_1",
                text="evidence",
                source_title="source",
                source_uri=None,
                section_title=None,
                page_start=1,
                page_end=1,
                retrieval_score=1.0,
                rank=1,
                token_count=1,
            )
        ]


def test_retrieve_evidence_prefers_plan_aware_retriever() -> None:
    plan = QueryPlan(
        plan_id="plan_1",
        original_query="What is 3M FY2018 capex?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="3M FY2018 capex",
            ),
        ),
    )
    tasks = tasks_from_plan(plan)
    retriever = _PlanAwareRetriever()

    evidence = _retrieve_evidence(
        retriever,
        db=object(),
        query=plan.original_query,
        top_k=1,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert evidence[0].evidence_id == "c1"
    assert retriever.received_plan == plan
    assert retriever.received_tasks == tasks


def test_query_plan_endpoint_returns_plan_before_query_id_route() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/query/plan",
        json={
            "query": "What is the FY2018 capital expenditure amount for 3M?",
            "options": {"query_plan_fallback_only": True},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_plan"]["planner"] == "rule_based_fallback"
    assert payload["retrieval_tasks"]
    assert payload["query_plan"]["retrieval_units"][0]["unit_id"] == "u0"


def test_cache_key_changes_when_query_plan_changes() -> None:
    settings = Settings(openai_api_key=None)
    base_options = {"retrieval_mode": "hybrid", "cache_policy": "enabled"}
    first_options = {
        **base_options,
        "query_plan": {"plan_id": "plan_1", "planner_version": "v1"},
        "retrieval_tasks": [{"unit_id": "u0", "weight": 1.0}],
    }
    second_options = {
        **base_options,
        "query_plan": {"plan_id": "plan_2", "planner_version": "v1"},
        "retrieval_tasks": [{"unit_id": "u0", "weight": 2.0}],
    }

    first = make_cache_key(
        query="What is 3M FY2018 capex?",
        filters={},
        settings=settings,
        top_k=8,
        options=first_options,
    )
    second = make_cache_key(
        query="What is 3M FY2018 capex?",
        filters={},
        settings=settings,
        top_k=8,
        options=second_options,
    )

    assert first != second


def test_cache_key_ignores_volatile_plan_and_task_ids() -> None:
    settings = Settings(openai_api_key=None)
    base_options = {"retrieval_mode": "hybrid", "cache_policy": "enabled"}
    first_options = {
        **base_options,
        "query_plan": {
            "plan_id": "plan_random_1",
            "original_query": "What is 3M FY2018 capex?",
            "retrieval_units": [{"unit_id": "u0", "text": "3M FY2018 capex"}],
            "planner_version": "v1",
        },
        "retrieval_tasks": [
            {"task_id": "task_random_1", "unit_id": "u0", "query_text": "3M FY2018 capex"}
        ],
    }
    second_options = {
        **base_options,
        "query_plan": {
            "plan_id": "plan_random_2",
            "original_query": "What is 3M FY2018 capex?",
            "retrieval_units": [{"unit_id": "u0", "text": "3M FY2018 capex"}],
            "planner_version": "v1",
        },
        "retrieval_tasks": [
            {"task_id": "task_random_2", "unit_id": "u0", "query_text": "3M FY2018 capex"}
        ],
    }

    first = make_cache_key(
        query="What is 3M FY2018 capex?",
        filters={},
        settings=settings,
        top_k=8,
        options=first_options,
    )
    second = make_cache_key(
        query="What is 3M FY2018 capex?",
        filters={},
        settings=settings,
        top_k=8,
        options=second_options,
    )

    assert first == second


def test_sql_and_graph_units_compile_to_skipped_tasks() -> None:
    plan = QueryPlan(
        plan_id="plan_future",
        original_query="Who supplies Apple Vision Pro displays?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_sql",
                purpose="numerical_aggregation",
                text="Apple Microsoft R&D 2023",
                retrievers=("sql",),
            ),
            RetrievalUnit(
                unit_id="u_graph",
                purpose="supply_chain_discovery",
                text="Apple Vision Pro display suppliers",
                retrievers=("graph",),
            ),
        ),
    )

    tasks = tasks_from_plan(plan)

    assert [task.provider for task in tasks] == ["sql", "graph"]
    assert all(task.provider_status == "skipped" for task in tasks)
    assert all(task.lanes == () for task in tasks)
    assert all(task.unsupported_reason for task in tasks)
