import pytest
from pydantic import ValidationError

from atlas.query_orchestrator.schema import Entity, Metric, Period, QueryPlan, RetrievalUnit
from atlas.query_runtime.verification import VerificationResult
from atlas.retrieval.models.candidate import Candidate
from atlas.retrieval.models.evidence_contract import EvidenceBlock, EvidencePack
from atlas.retrieval.models.retrieval_task import serialize_retrieval_task, tasks_from_plan


def test_query_plan_compiles_to_retrieval_tasks() -> None:
    plan = QueryPlan(
        plan_id="plan_1",
        original_query="What is 3M FY2018 capex?",
        query_type="financial_numeric_fact",
        entities=(Entity(value="3M", aliases=("MMM",)),),
        periods=(Period(value="FY2018", normalized="2018"),),
        metrics=(Metric(canonical_name="capital_expenditure", aliases=("capex",)),),
        metadata_filter={"company": "3M"},
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="3M FY2018 capital expenditure",
                retrievers=("hybrid",),
                metadata_filter={"document_ids": ["doc_1"]},
                must_have_terms=("2018",),
                weight=1.2,
            ),
        ),
        planner="test",
    )

    tasks = tasks_from_plan(plan)

    assert len(tasks) == 1
    assert tasks[0].plan_id == "plan_1"
    assert tasks[0].unit_id == "u0"
    assert tasks[0].provider == "hybrid"
    assert tasks[0].query_text == "3M FY2018 capital expenditure"
    assert tasks[0].lanes == ("dense", "bm25")
    assert tasks[0].provider_status == "ready"
    assert tasks[0].unsupported_reason is None
    assert tasks[0].internal_lanes == ("dense", "bm25")
    assert tasks[0].metadata_filter == {"company": "3M", "document_ids": ["doc_1"]}
    payload = serialize_retrieval_task(tasks[0])
    assert "filters" not in payload
    assert "lanes" not in payload
    assert payload["provider"] == "hybrid"
    assert payload["metadata_filter"] == {"company": "3M", "document_ids": ["doc_1"]}
    assert payload["internal_lanes"] == ["dense", "bm25"]


def test_retrieval_unit_rejects_legacy_retriever_names() -> None:
    with pytest.raises(ValidationError):
        RetrievalUnit(
            unit_id="u0",
            purpose="original",
            text="3M FY2018 capex",
            retrievers=("dense",),
        )


def test_retrieval_unit_rejects_compound_provider_units() -> None:
    with pytest.raises(ValidationError, match="unit_proposals"):
        RetrievalUnit(
            unit_id="u0",
            purpose="compound",
            text="3M FY2018 capex",
            retrievers=("hybrid", "sql"),
        )


def test_query_plan_rejects_legacy_filters_contract() -> None:
    with pytest.raises(ValidationError):
        QueryPlan(
            plan_id="plan_legacy_filters",
            original_query="What is 3M FY2018 capex?",
            filters={"company": "3M"},
            retrieval_units=(
                RetrievalUnit(
                    unit_id="u0",
                    purpose="original",
                    text="3M FY2018 capex",
                ),
            ),
        )


def test_sql_provider_task_is_represented_but_skipped() -> None:
    plan = QueryPlan(
        plan_id="plan_sql",
        original_query="What is 3M FY2018 capex?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="structured_lookup",
                text="3M FY2018 capex",
                retrievers=("sql",),
            ),
        ),
    )

    task = tasks_from_plan(plan)[0]

    assert task.provider == "sql"
    assert task.provider_status == "skipped"
    assert task.unsupported_reason == "provider_not_supported_in_v1_live_retrieval:sql"
    assert task.internal_lanes == ()


def test_candidate_exposes_v1_contract_fields_without_breaking_defaults() -> None:
    candidate = Candidate(
        chunk_id="chk_1",
        document_id="doc_1",
        doc_name="3M 2018 10-K",
        source_title="3M 2018 10-K",
        company="3M",
        text="Purchases of property, plant and equipment 1,577",
        page_start=60,
        page_end=60,
        chunk_index=1,
        token_count=12,
        retrieved_by=("bm25",),
        dense_rank=None,
        dense_score=None,
        lexical_rank=1,
        lexical_score=12.3,
        parent_id="parent_1",
    )

    assert candidate.parent_id == "parent_1"
    assert candidate.provider == "text_hybrid"
    assert candidate.source_type == "text_chunk"
    assert candidate.unit_weight == 1.0
    assert candidate.lane_weight == 1.0


def test_evidence_pack_tracks_prompt_blocks_and_drops() -> None:
    included = EvidenceBlock(
        evidence_id="c1",
        source_type="page_block",
        provider="text_hybrid",
        text="supported text",
        document_id="doc_1",
        doc_name="3M 2018 10-K",
        page_start=60,
        page_end=60,
        chunk_ids=("chk_1",),
        included_in_prompt=True,
    )
    dropped = EvidenceBlock(
        evidence_id="c2",
        source_type="page_block",
        provider="text_hybrid",
        text="extra text",
        document_id="doc_1",
        doc_name="3M 2018 10-K",
        page_start=61,
        page_end=61,
        chunk_ids=("chk_2",),
        drop_reason="token_budget",
    )

    pack = EvidencePack(
        pack_id="pack_1",
        query_id="q_1",
        plan_id="plan_1",
        blocks=(included,),
        dropped_blocks=(dropped,),
        token_count=10,
        max_context_tokens=100,
    )

    assert pack.prompt_blocks == (included,)
    assert pack.dropped_blocks[0].drop_reason == "token_budget"


def test_verification_result_marks_blocking_statuses() -> None:
    result = VerificationResult(
        verification_id="ver_1",
        stage="pre_generation",
        status="insufficient",
        confidence_override="insufficient",
        reasons=("no_evidence",),
    )

    assert result.is_blocking is True
    assert result.to_dict()["reasons"] == ["no_evidence"]


def test_query_plan_requires_at_least_one_retrieval_unit() -> None:
    with pytest.raises(ValidationError):
        QueryPlan(
            plan_id="plan_empty",
            original_query="What is 3M FY2018 capex?",
            retrieval_units=(),
        )
