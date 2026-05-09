from __future__ import annotations

import json
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import delete, text
from sqlalchemy.exc import SQLAlchemyError

from atlas.api.dependencies import get_provider_router, get_query_orchestrator, get_query_runtime
from atlas.core.config import Settings
from atlas.db.models import (
    AnswerRecord,
    CandidateRecord,
    CitationAuditRecord,
    CitationRecord,
    CitationVerificationRecord,
    EvidenceBlockRecord,
    EvidenceEvaluationRecord,
    EvidencePackRecord,
    GenerationEvent,
    LLMCallEvidenceRecord,
    LLMCallRecord,
    QualityReviewRecord,
    QueryPlanRecord,
    QueryRun,
    RetrievalEvent,
    RetrievalResultRecord,
    RetrievalTaskRecord,
)
from atlas.db.repositories import add_query_trace, get_v1_trace_family, record_v1_trace_family
from atlas.db.session import SessionLocal, get_db, init_db
from atlas.main import create_app
from atlas.query_orchestrator.schema import QueryPlan, RetrievalUnit
from atlas.retrieval.contracts import ProviderResult
from atlas.retrieval.models.evidence import Evidence
from atlas.retrieval.models.retrieval_task import tasks_from_plan
from atlas.retrieval.providers.base import RetrievalContext, RetrievalProvider
from atlas.retrieval.router import ProviderRouter


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
        self.executable_provider_calls = []

    def plan(self, query, *, use_llm=True, executable_providers=None):
        self.executable_provider_calls.append(executable_providers)
        return self.plan_value


class _FakeRetriever(RetrievalProvider):
    provider_name = "hybrid"

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
        evidence = self.retrieve_with_plan(
            db,
            query=query,
            top_k=top_k,
            filters=filters,
            options=options,
            query_plan=query_plan,
            retrieval_tasks=retrieval_tasks,
        )
        return ProviderResult(
            provider="hybrid",
            task_id=retrieval_tasks[0].task_id if retrieval_tasks else None,
            unit_id=retrieval_tasks[0].unit_id if retrieval_tasks else None,
            status="executed" if evidence else "empty",
            evidence=tuple(evidence),
            latency_ms=1,
            trace={"provider": "hybrid", "status": "executed" if evidence else "empty"},
        )

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


class _FakeRuntime:
    def __init__(self, retriever: _FakeRetriever) -> None:
        self.settings = Settings(
            openai_api_key=None,
            query_runtime_executable_providers="hybrid",
        )
        self.retriever = retriever
        self.provider_router = ProviderRouter({"hybrid": retriever})


_TRACE_DELETE_ORDER = (
    CitationAuditRecord,
    QualityReviewRecord,
    CitationVerificationRecord,
    CitationRecord,
    AnswerRecord,
    LLMCallEvidenceRecord,
    EvidenceEvaluationRecord,
    EvidencePackRecord,
    EvidenceBlockRecord,
    CandidateRecord,
    RetrievalResultRecord,
    RetrievalTaskRecord,
    QueryPlanRecord,
    GenerationEvent,
    RetrievalEvent,
    LLMCallRecord,
    QueryRun,
)


def _plan() -> QueryPlan:
    return QueryPlan(
        plan_id="plan_1",
        original_query="What was 3M FY2018 capex?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="3M FY2018 capex",
                provider="hybrid",
                metadata={"internal_lanes": ["dense", "bm25"]},
            ),
        ),
        planner="test",
    )


def _repository_db_or_skip():
    try:
        init_db()
        db = SessionLocal()
        db.execute(text("select 1"))
    except SQLAlchemyError as exc:
        pytest.skip(f"Repository DB unavailable: {exc}")
    return db


def _delete_trace_family_rows(db, query_id: str) -> None:
    for model in _TRACE_DELETE_ORDER:
        db.execute(delete(model).where(model.query_id == query_id))


def _json_blob(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _persist_query_with_raw_llm_trace(db, query_id: str) -> dict[str, str]:
    call_suffix = uuid4().hex[:12]
    planner_call_id = f"llmc_api_planner_{call_suffix}"
    answer_call_id = f"llmc_api_answer_{call_suffix}"
    raw_planner_prompt = f"RAW PLANNER PROMPT SENTINEL {query_id}"
    raw_planner_response = f"RAW PLANNER RESPONSE SENTINEL {query_id}"
    raw_answer_input = f"RAW ANSWER INPUT SENTINEL {query_id}"
    raw_answer_output = f"RAW ANSWER OUTPUT SENTINEL {query_id}"
    raw_evidence_snapshot = f"RAW EVIDENCE SNAPSHOT SENTINEL {query_id}"
    answer_request = {
        "model": "answer-model",
        "instructions": "Answer using evidence.",
        "input": raw_answer_input,
        "max_output_tokens": 2000,
        "reasoning": {"effort": "low"},
        "store": False,
    }
    query_run = QueryRun(
        query_id=query_id,
        trace_id=f"tr_{query_id}",
        user_query="What was 3M FY2018 capex?",
        normalized_query="What was 3M FY2018 capex?",
        answer="3M FY2018 capex was 1,577 million [c1].",
        confidence="supported",
        citations_json=[{"citation_id": "c1", "evidence_id": "c1"}],
        model_name="answer-model",
        prompt_version="test",
        latency_ms=25,
        details_json={
            "query_plan": {
                "plan_id": f"plan_{query_id}",
                "planner": "llm_structured",
                "metadata": {
                    "raw_planner_prompt": raw_planner_prompt,
                    "raw_planner_response": raw_planner_response,
                },
            },
            "retrieval_tasks": [{"task_id": "rt_api", "unit_id": "u0"}],
            "retrieval_trace": {
                "top_k": [
                    {
                        "evidence_id": "c1",
                        "chunk_id": "chk_api_trace",
                        "rank": 1,
                        "retrieval_score": 1.0,
                    }
                ]
            },
            "llm_io": {
                "status": "completed",
                "answer_llm_call_id": answer_call_id,
                "request": {"input": raw_answer_input},
                "response": {"raw_output": raw_answer_output},
            },
            "trace": {
                "stages": [{"name": "retrieval", "status": "completed"}],
                "metadata": {},
            },
        },
    )
    retrieval_event = RetrievalEvent(
        event_id=f"ret_{query_id}",
        query_id=query_id,
        chunk_id="chk_api_trace",
        rank=1,
        retrieval_score=1.0,
        retriever_type="hybrid",
    )
    generation_event = GenerationEvent(
        event_id=f"gen_{query_id}",
        query_id=query_id,
        model_name="answer-model",
        prompt_version="test",
        input_tokens=13,
        output_tokens=8,
        latency_ms=19,
        status="completed",
    )
    add_query_trace(
        db,
        query_run,
        [retrieval_event],
        generation_event,
        observability={
            "planner_llm_call": {
                "call_id": planner_call_id,
                "stage": "planner",
                "attempt_index": 1,
                "sequence_index": 1,
                "status": "completed",
                "validation_status": "validated",
                "model_name": "planner-model",
                "planner_version": "query_planner_test",
                "latency_ms": 7,
                "request": {
                    "model": "planner-model",
                    "instructions": "Planner instructions.",
                    "input": raw_planner_prompt,
                    "max_output_tokens": 2000,
                    "reasoning": {"effort": "low"},
                    "store": False,
                },
                "response": {
                    "raw_output": raw_planner_response,
                    "parsed_plan_id": f"plan_{query_id}",
                    "validation_status": "validated",
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                },
                "usage": {"input_tokens": 11, "output_tokens": 7},
                "raw_output": raw_planner_response,
                "parsed_plan_id": f"plan_{query_id}",
            },
            "answer_llm_call": {
                "call_id": answer_call_id,
                "stage": "answer",
                "attempt_index": 1,
                "sequence_index": 100,
                "status": "completed",
                "model_name": "answer-model",
                "prompt_version": "test",
                "latency_ms": 19,
                "request": answer_request,
                "request_metadata": {
                    "prompt_version": "test",
                    "evidence_ids": ["c1"],
                    "evidence_count": 1,
                },
                "response": {
                    "raw_output": raw_answer_output,
                    "parsed_answer": "3M FY2018 capex was 1,577 million [c1].",
                    "parsed_confidence": "supported",
                    "usage": {"input_tokens": 13, "output_tokens": 8},
                },
                "usage": {"input_tokens": 13, "output_tokens": 8},
            },
            "answer_prompt_evidence": [
                {
                    "evidence_id": "c1",
                    "rank": 1,
                    "provider": "text_hybrid",
                    "chunk_id": "chk_api_trace",
                    "document_id": "doc_api_trace",
                    "page_start": 60,
                    "page_end": 60,
                    "retrieval_score": 1.0,
                    "token_count": 8,
                    "text_snapshot": raw_evidence_snapshot,
                }
            ],
        },
    )
    db.commit()
    return {
        "planner_call_id": planner_call_id,
        "answer_call_id": answer_call_id,
        "raw_planner_prompt": raw_planner_prompt,
        "raw_planner_response": raw_planner_response,
        "raw_answer_input": raw_answer_input,
        "raw_answer_output": raw_answer_output,
        "raw_evidence_snapshot": raw_evidence_snapshot,
    }


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
            "provider_router_trace": {
                "query_plan_id": "plan_1",
                "known_providers": ["hybrid", "sql", "graph"],
                "executable_providers": ["hybrid"],
                "status": "partial",
            },
            "provider_results": [
                {
                    "provider": "sql",
                    "task_id": "rt_sql",
                    "unit_id": "u_sql",
                    "status": "skipped_non_executable",
                    "reason": "provider_not_executable:sql",
                },
                {
                    "provider": "hybrid",
                    "task_id": "rt_1",
                    "unit_id": "u0",
                    "status": "executed",
                    "reason": None,
                    "candidates": [
                        {
                            "candidate_id": "cand_1",
                            "chunk_id": "chk_1",
                            "document_id": "doc_1",
                            "rank": 1,
                            "source_anchor": {
                                "document_id": "doc_1",
                                "chunk_id": "chk_1",
                                "page_start": 60,
                                "page_end": 60,
                            },
                        }
                    ],
                },
            ],
            "retrieval_trace": {
                "top_k": [
                    {
                        "evidence_id": "c1",
                        "chunk_id": "chk_1",
                        "rank": 1,
                        "source_anchor": {
                            "document_id": "doc_1",
                            "chunk_id": "chk_1",
                            "page_start": 60,
                            "page_end": 60,
                        },
                        "evidence_pack": {
                            "pack_id": "ep_1",
                            "blocks": [
                                {
                                    "evidence_id": "c1",
                                    "chunk_ids": ["chk_1"],
                                    "source_anchor": {
                                        "document_id": "doc_1",
                                        "chunk_id": "chk_1",
                                    },
                                }
                            ],
                        },
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
    assert retrieval_result.payload_json["provider_router_trace"]["status"] == "partial"
    assert retrieval_result.payload_json["provider_results"][0]["status"] == (
        "skipped_non_executable"
    )
    candidate_records = [item for item in db.added if item.__tablename__ == "candidates"]
    evidence_block = next(item for item in db.added if item.__tablename__ == "evidence_blocks")
    assert any(
        item.payload_json.get("source_anchor", {}).get("chunk_id") == "chk_1"
        for item in candidate_records
    )
    assert evidence_block.payload_json["source_anchor"]["chunk_id"] == "chk_1"


def test_record_v1_trace_family_persists_json_pointers_in_repository_db() -> None:
    db = _repository_db_or_skip()
    query_id = "repo_trace_json_pointer_test"
    call_suffix = uuid4().hex[:12]
    planner_call_id = f"llmc_repo_planner_{call_suffix}"
    answer_call_id = f"llmc_repo_answer_{call_suffix}"
    raw_planner_prompt = "RAW PLANNER PROMPT repo_trace_json_pointer_test"
    raw_planner_response = "RAW PLANNER RESPONSE repo_trace_json_pointer_test"
    raw_answer_output = '{"answer":"3M FY2018 capex was 1,577 million [c1]."}'
    answer_request = {
        "model": "answer-model",
        "instructions": "Answer using evidence.",
        "input": "Question: What was 3M FY2018 capex?\n[c1] 3M capex was 1,577 million.",
        "max_output_tokens": 2000,
        "reasoning": {"effort": "low"},
        "store": False,
    }

    try:
        _delete_trace_family_rows(db, query_id)
        db.commit()
        db.add(
            QueryRun(
                query_id=query_id,
                trace_id="tr_repo_trace_json_pointer_test",
                user_query="What was 3M FY2018 capex?",
                normalized_query="What was 3M FY2018 capex?",
                answer="3M FY2018 capex was 1,577 million [c1].",
                confidence="supported",
                citations_json=[{"citation_id": "c1", "evidence_id": "c1"}],
                model_name="answer-model",
                prompt_version="test",
                latency_ms=25,
                details_json={
                    "query_plan": {
                        "plan_id": "plan_repo_trace_json_pointer_test",
                        "planner": "llm_structured",
                        "metadata": {
                            "raw_planner_prompt": raw_planner_prompt,
                            "raw_planner_response": raw_planner_response,
                        },
                    },
                    "retrieval_tasks": [{"task_id": "rt_repo", "unit_id": "u0"}],
                    "retrieval_trace": {
                        "top_k": [
                            {
                                "evidence_id": "c1",
                                "chunk_id": "chk_repo",
                                "rank": 1,
                                "retrieval_score": 1.0,
                            }
                        ]
                    },
                    "llm_io": {
                        "status": "completed",
                        "answer_llm_call_id": answer_call_id,
                        "request": {"input": answer_request["input"]},
                        "response": {"raw_output": raw_answer_output},
                    },
                    "trace": {
                        "stages": [{"name": "retrieval", "status": "completed"}],
                        "metadata": {},
                    },
                },
            )
        )
        db.commit()

        query_run = db.get(QueryRun, query_id)
        assert query_run is not None
        retrieval_event = RetrievalEvent(
            event_id="ret_repo_trace_json_pointer_test",
            query_id=query_id,
            chunk_id="chk_repo",
            rank=1,
            retrieval_score=1.0,
            retriever_type="hybrid",
        )
        generation_event = GenerationEvent(
            event_id="gen_repo_trace_json_pointer_test",
            query_id=query_id,
            model_name="answer-model",
            prompt_version="test",
            status="completed",
        )
        record_v1_trace_family(
            db,
            query_run,
            [retrieval_event],
            generation_event,
            observability={
                "planner_llm_call": {
                    "call_id": planner_call_id,
                    "stage": "planner",
                    "attempt_index": 1,
                    "sequence_index": 1,
                    "status": "completed",
                    "validation_status": "validated",
                    "model_name": "planner-model",
                    "planner_version": "query_planner_test",
                    "request": {
                        "model": "planner-model",
                        "instructions": "Planner instructions.",
                        "input": raw_planner_prompt,
                        "max_output_tokens": 2000,
                        "reasoning": {"effort": "low"},
                        "store": False,
                    },
                    "response": {
                        "raw_output": raw_planner_response,
                        "parsed_plan_id": "plan_repo_trace_json_pointer_test",
                        "validation_status": "validated",
                        "usage": {"input_tokens": 11, "output_tokens": 7},
                    },
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                    "raw_output": raw_planner_response,
                    "parsed_plan_id": "plan_repo_trace_json_pointer_test",
                },
                "answer_llm_call": {
                    "call_id": answer_call_id,
                    "stage": "answer",
                    "attempt_index": 1,
                    "sequence_index": 100,
                    "status": "completed",
                    "model_name": "answer-model",
                    "prompt_version": "test",
                    "request": answer_request,
                    "request_metadata": {
                        "prompt_version": "test",
                        "evidence_ids": ["c1"],
                        "evidence_count": 1,
                    },
                    "response": {
                        "raw_output": raw_answer_output,
                        "parsed_answer": "3M FY2018 capex was 1,577 million [c1].",
                        "parsed_confidence": "supported",
                        "usage": {"input_tokens": 13, "output_tokens": 8},
                    },
                    "usage": {"input_tokens": 13, "output_tokens": 8},
                },
                "answer_prompt_evidence": [
                    {
                        "evidence_id": "c1",
                        "rank": 1,
                        "provider": "text_hybrid",
                        "chunk_id": "chk_repo",
                        "document_id": "doc_repo",
                        "page_start": 60,
                        "page_end": 60,
                        "retrieval_score": 1.0,
                        "token_count": 8,
                        "text_snapshot": "3M capex was 1,577 million.",
                    }
                ],
            },
        )
        db.commit()
        db.close()

        db = SessionLocal()
        persisted_run = db.get(QueryRun, query_id)
        assert persisted_run is not None
        details = persisted_run.details_json
        assert details["planner_llm"] == {
            "planner_llm_call_id": planner_call_id,
            "planner_llm_status": "completed",
            "planner_validation_status": "validated",
        }
        assert details["query_plan"]["metadata"]["planner_llm_call_id"] == planner_call_id
        assert details["llm_io"] == {
            "status": "completed",
            "answer_llm_call_id": answer_call_id,
        }
        assert {"request", "response", "request_metadata"}.isdisjoint(details["llm_io"])
        assert raw_planner_prompt not in _json_blob(details)
        assert raw_planner_response not in _json_blob(details)

        v1_trace = get_v1_trace_family(db, query_id)
        assert v1_trace["query_plans"][0]["planner_call_id"] == planner_call_id
        assert (
            v1_trace["query_plans"][0]["payload"]["metadata"]["planner_llm_call_id"]
            == planner_call_id
        )
        assert v1_trace["answers"][0]["answer_call_id"] == answer_call_id
        assert v1_trace["answers"][0]["payload"]["answer_llm_call_id"] == answer_call_id
        llm_calls = {item["stage"]: item for item in v1_trace["llm_calls"]}
        assert set(llm_calls) == {"planner", "answer"}
        assert llm_calls["planner"]["call_id"] == planner_call_id
        assert llm_calls["planner"]["input_text"] == raw_planner_prompt
        assert llm_calls["planner"]["raw_output_text"] == raw_planner_response
        assert llm_calls["answer"]["call_id"] == answer_call_id
        assert llm_calls["answer"]["request"] == answer_request
        assert llm_calls["answer"]["raw_output_text"] == raw_answer_output
        assert len(v1_trace["llm_call_evidence"]) == 1
        assert v1_trace["llm_call_evidence"][0]["call_id"] == answer_call_id
        assert v1_trace["llm_call_evidence"][0]["evidence_id"] == "c1"
        assert v1_trace["llm_call_evidence"][0]["evidence_block_record_id"]
    finally:
        db.rollback()
        _delete_trace_family_rows(db, query_id)
        db.commit()
        db.close()


def test_trace_endpoint_redacts_raw_llm_io_by_default_and_opt_in_returns_raw() -> None:
    db = _repository_db_or_skip()
    query_id = f"api_trace_raw_gate_{uuid4().hex[:12]}"
    try:
        sentinels = _persist_query_with_raw_llm_trace(db, query_id)
        client = TestClient(create_app())

        response = client.get(f"/v1/query/{query_id}/trace")

        assert response.status_code == 200
        payload = response.json()
        blob = _json_blob(payload)
        assert sentinels["raw_planner_prompt"] not in blob
        assert sentinels["raw_planner_response"] not in blob
        assert sentinels["raw_answer_input"] not in blob
        assert sentinels["raw_answer_output"] not in blob
        assert sentinels["raw_evidence_snapshot"] not in blob
        llm_calls = {item["stage"]: item for item in payload["v1_trace"]["llm_calls"]}
        assert set(llm_calls) == {"planner", "answer"}
        assert llm_calls["planner"]["call_id"] == sentinels["planner_call_id"]
        assert llm_calls["planner"]["request"] == "[redacted]"
        assert llm_calls["planner"]["response"] == "[redacted]"
        assert llm_calls["planner"]["input_text"] == "[redacted]"
        assert llm_calls["planner"]["raw_output_text"] == "[redacted]"
        assert llm_calls["planner"]["input_tokens"] == 11
        assert llm_calls["planner"]["latency_ms"] == 7
        assert llm_calls["planner"]["raw_payload_hash"]
        assert llm_calls["planner"]["raw_redaction_status"] == "unredacted"
        assert llm_calls["answer"]["call_id"] == sentinels["answer_call_id"]
        assert llm_calls["answer"]["request"] == "[redacted]"
        assert llm_calls["answer"]["response"] == "[redacted]"
        assert llm_calls["answer"]["parsed_answer_text"] == "[redacted]"
        assert llm_calls["answer"]["output_tokens"] == 8
        evidence = payload["v1_trace"]["llm_call_evidence"][0]
        assert evidence["call_id"] == sentinels["answer_call_id"]
        assert evidence["chunk_id"] == "chk_api_trace"
        assert evidence["provider"] == "text_hybrid"
        assert evidence["text_snapshot"] == "[redacted]"
        assert evidence["text_hash"]

        raw_response = client.get(
            f"/v1/query/{query_id}/trace",
            params={"include_raw_llm_io": "true"},
        )

        assert raw_response.status_code == 200
        raw_payload = raw_response.json()
        raw_calls = {
            item["stage"]: item for item in raw_payload["v1_trace"]["llm_calls"]
        }
        assert raw_calls["planner"]["input_text"] == sentinels["raw_planner_prompt"]
        assert raw_calls["planner"]["raw_output_text"] == sentinels["raw_planner_response"]
        assert raw_calls["answer"]["request"]["input"] == sentinels["raw_answer_input"]
        assert raw_calls["answer"]["raw_output_text"] == sentinels["raw_answer_output"]
        assert (
            raw_payload["v1_trace"]["llm_call_evidence"][0]["text_snapshot"]
            == sentinels["raw_evidence_snapshot"]
        )

        header_response = client.get(
            f"/v1/query/{query_id}/trace",
            headers={"X-Atlas-Include-Raw-Llm-Io": "true"},
        )

        assert header_response.status_code == 200
        header_calls = {
            item["stage"]: item
            for item in header_response.json()["v1_trace"]["llm_calls"]
        }
        assert header_calls["answer"]["request"]["input"] == sentinels["raw_answer_input"]
    finally:
        db.rollback()
        _delete_trace_family_rows(db, query_id)
        db.commit()
        db.close()


def test_query_endpoint_does_not_expose_new_raw_llm_io_payloads() -> None:
    db = _repository_db_or_skip()
    query_id = f"api_query_no_raw_{uuid4().hex[:12]}"
    try:
        sentinels = _persist_query_with_raw_llm_trace(db, query_id)
        client = TestClient(create_app())

        response = client.get(f"/v1/query/{query_id}")

        assert response.status_code == 200
        blob = _json_blob(response.json())
        assert sentinels["raw_planner_prompt"] not in blob
        assert sentinels["raw_planner_response"] not in blob
        assert sentinels["raw_answer_input"] not in blob
        assert sentinels["raw_answer_output"] not in blob
        assert sentinels["raw_evidence_snapshot"] not in blob
    finally:
        db.rollback()
        _delete_trace_family_rows(db, query_id)
        db.commit()
        db.close()


def test_retrieve_endpoint_returns_plan_tasks_and_evidence() -> None:
    plan = _plan()
    retriever = _FakeRetriever()
    orchestrator = _FakeOrchestrator(plan)
    app = create_app()
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_query_orchestrator] = lambda: orchestrator
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
    assert orchestrator.executable_provider_calls == [("hybrid",)]


def test_plan_endpoint_uses_runtime_provider_router_capability() -> None:
    plan = _plan()
    orchestrator = _FakeOrchestrator(plan)
    app = create_app()
    app.dependency_overrides[get_query_orchestrator] = lambda: orchestrator
    app.dependency_overrides[get_provider_router] = lambda: ProviderRouter({})
    client = TestClient(app)

    response = client.post(
        "/v1/query/plan",
        json={
            "query": plan.original_query,
            "top_k": 1,
            "options": {"query_plan_fallback_only": True},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_tasks"][0]["provider"] == "hybrid"
    assert payload["retrieval_tasks"][0]["provider_status"] == "skipped_non_executable"
    assert orchestrator.executable_provider_calls == [()]
