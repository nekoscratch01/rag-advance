from __future__ import annotations

from fastapi.testclient import TestClient

from atlas.api.dependencies import get_query_orchestrator, get_query_runtime
from atlas.core.config import Settings
from atlas.db.models import GenerationEvent, QueryRun, RetrievalEvent
from atlas.db.repositories import record_v1_trace_family
from atlas.db.session import get_db
from atlas.main import create_app
from atlas.query_orchestrator.schema import QueryPlan, RetrievalUnit
from atlas.retrieval.evidence import Evidence
from atlas.retrieval.retrieval_task import tasks_from_plan


class _FakeDB:
    def __init__(self) -> None:
        self.added = []

    def add(self, value) -> None:
        self.added.append(value)

    def flush(self) -> None:
        return None


class _FakeOrchestrator:
    def __init__(self, plan: QueryPlan) -> None:
        self.plan_value = plan

    def plan(self, query, *, use_llm=True):
        return self.plan_value


class _FakeRetriever:
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
                text="3M FY2018 capital expenditures were 1,577 million.",
                source_title="3M 2018 10-K",
                source_uri=None,
                section_title=None,
                page_start=60,
                page_end=60,
                retrieval_score=1.0,
                rank=1,
                token_count=10,
                metadata={"provider": "text_hybrid", "lane": "dense"},
            )
        ]


class _FakeRuntime:
    def __init__(self, retriever: _FakeRetriever) -> None:
        self.settings = Settings(openai_api_key=None)
        self.retriever = retriever


def _plan() -> QueryPlan:
    return QueryPlan(
        plan_id="plan_1",
        original_query="What was 3M FY2018 capex?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="3M FY2018 capex",
                retrievers=("hybrid",),
                metadata={"internal_lanes": ["dense", "bm25"]},
            ),
        ),
        planner="test",
    )


def test_record_v1_trace_family_persists_design_table_family_records() -> None:
    query_run = QueryRun(
        query_id="q_1",
        trace_id="tr_1",
        user_query="What was 3M FY2018 capex?",
        normalized_query="What was 3M FY2018 capex?",
        answer="3M FY2018 capex was 1,577 million [c1].",
        confidence="supported",
        citations_json=[{"citation_id": "c1", "document_id": "doc_1"}],
        model_name="gpt-5-nano",
        prompt_version="test",
        latency_ms=123,
        details_json={
            "query_plan": {"plan_id": "plan_1", "planner": "test"},
            "retrieval_tasks": [{"task_id": "rt_1", "unit_id": "u0"}],
            "retrieval_trace": {
                "top_k": [
                    {
                        "evidence_id": "c1",
                        "chunk_id": "chk_1",
                        "rank": 1,
                        "evidence_pack": {"pack_id": "ep_1"},
                    }
                ]
            },
            "critic": {
                "evidence_evaluation": {"status": "supported"},
                "citation_verification": {"status": "supported"},
            },
            "trace": {
                "stages": [{"name": "retrieval", "status": "completed"}],
                "metadata": {},
            },
        },
    )
    retrieval_event = RetrievalEvent(
        event_id="ret_1",
        query_id="q_1",
        chunk_id="chk_1",
        rank=1,
        retrieval_score=1.0,
        retriever_type="hybrid",
    )
    generation_event = GenerationEvent(
        event_id="gen_1",
        query_id="q_1",
        model_name="gpt-5-nano",
        prompt_version="test",
        status="completed",
    )
    db = _FakeDB()

    record_v1_trace_family(db, query_run, [retrieval_event], generation_event)

    table_names = {item.__tablename__ for item in db.added}
    retrieval_result = next(
        item for item in db.added if item.__tablename__ == "retrieval_results"
    )
    assert {
        "query_plans",
        "retrieval_tasks",
        "retrieval_results",
        "candidates",
        "evidence_blocks",
        "evidence_packs",
        "evidence_evaluations",
        "answers",
        "citations",
        "citation_verifications",
    } <= table_names
    assert retrieval_result.status == "completed"


def test_retrieve_endpoint_returns_plan_tasks_and_evidence() -> None:
    plan = _plan()
    retriever = _FakeRetriever()
    app = create_app()
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_query_orchestrator] = lambda: _FakeOrchestrator(plan)
    app.dependency_overrides[get_query_runtime] = lambda: _FakeRuntime(retriever)
    client = TestClient(app)

    response = client.post(
        "/v1/retrieve",
        json={
            "query": plan.original_query,
            "top_k": 1,
            "options": {"query_plan_fallback_only": True},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_plan"]["plan_id"] == "plan_1"
    assert payload["retrieval_tasks"][0]["unit_id"] == "u0"
    assert payload["evidence"][0]["evidence_id"] == "c1"
    assert retriever.received_plan == plan
    assert [task.unit_id for task in retriever.received_tasks] == [
        task.unit_id for task in tasks_from_plan(plan)
    ]
