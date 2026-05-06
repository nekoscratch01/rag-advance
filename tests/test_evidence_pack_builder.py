from __future__ import annotations

from dataclasses import replace

from atlas.query_runtime.evidence_builder import (
    build_evidence_pack_from_candidates,
    evidence_pack_to_evidence,
)
from atlas.retrieval.candidate import Candidate


def _candidate(
    chunk_id: str,
    *,
    parent_id: str | None = None,
    text: str = "3M FY2018 capital expenditures were 1,577 million.",
    final_rank: int = 1,
    token_count: int = 8,
    unit_id: str = "u0",
) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        document_id="doc_1",
        doc_name="3M 2018 10-K",
        source_title="3M 2018 10-K",
        company="3M",
        text=text,
        page_start=60,
        page_end=60,
        chunk_index=final_rank,
        token_count=token_count,
        retrieved_by=("dense", "bm25"),
        dense_rank=final_rank,
        dense_score=0.9,
        lexical_rank=final_rank,
        lexical_score=10.0,
        fusion_rank=final_rank,
        fusion_score=1.0 / final_rank,
        final_rank=final_rank,
        metadata={"parent_id": parent_id, "retrieval_unit_id": unit_id} if parent_id else {"retrieval_unit_id": unit_id},
        parent_id=parent_id,
        retrieval_unit_id=unit_id,
    )


def test_evidence_pack_outputs_prompt_blocks_drop_reasons_and_coverage() -> None:
    candidates = (
        _candidate("chk_1", parent_id="parent_1", final_rank=1, unit_id="u0"),
        _candidate("chk_2", parent_id="parent_2", final_rank=2, unit_id="u1"),
    )

    pack = build_evidence_pack_from_candidates(
        candidates,
        parent_resolver={
            "parent_1": {
                "parent_id": "parent_1",
                "document_id": "doc_1",
                "text": "3M FY2018 capital expenditures were 1,577 million.",
                "page_start": 60,
                "page_end": 60,
                "metadata_json": {"company": "3M"},
            },
            "parent_2": {
                "parent_id": "parent_2",
                "document_id": "doc_1",
                "text": "Extra trailing evidence.",
                "page_start": 61,
                "page_end": 61,
                "metadata_json": {"company": "3M"},
            },
        },
        max_context_tokens=100,
        max_blocks=1,
        query_id="q_1",
        plan_id="plan_1",
        retrieval_unit_coverage={
            "entities": ("3M",),
            "periods": ("2018",),
            "metrics": ("capital_expenditure",),
            "retrieval_unit_ids": ("u0", "u1"),
        },
    )

    assert pack.query_id == "q_1"
    assert pack.plan_id == "plan_1"
    assert len(pack.blocks) == 1
    assert len(pack.dropped_blocks) == 1
    assert pack.prompt_blocks[0].included_in_prompt is True
    assert pack.dropped_blocks[0].drop_reason == "max_blocks"
    assert pack.blocks[0].coverage["entities"]["covered"] == ["3M"]
    assert pack.blocks[0].coverage["periods"]["covered"] == ["2018"]
    assert pack.blocks[0].coverage["metrics"]["covered"] == ["capital_expenditure"]
    assert pack.metadata["included_count"] == 1
    assert pack.metadata["dropped_count"] == 1

    evidence = evidence_pack_to_evidence(pack)
    assert evidence[0].metadata["evidence_pack_id"] == pack.pack_id
    assert evidence[0].metadata["included_in_prompt"] is True
    assert evidence[0].metadata["coverage"]["entities"]["covered"] == ["3M"]


def test_evidence_pack_marks_token_budget_drops() -> None:
    pack = build_evidence_pack_from_candidates(
        (
            _candidate("chk_1", final_rank=1, token_count=20),
            _candidate("chk_2", final_rank=2, token_count=20),
        ),
        max_context_tokens=0,
    )

    assert pack.blocks == ()
    assert len(pack.dropped_blocks) == 2
    assert {block.drop_reason for block in pack.dropped_blocks} == {"token_budget"}


def test_coverage_ignores_retrieval_query_metadata_false_positives() -> None:
    candidate = _candidate(
        "chk_1",
        parent_id="parent_1",
        text="unrelated trailing evidence.",
        final_rank=1,
        unit_id="u0",
    )
    candidate = replace(
        candidate,
        metadata={
            "text_hybrid_provider": {
                "lanes": [
                    {
                        "query_text": "3M FY2018 capital expenditures",
                    }
                ]
            }
        },
    )

    pack = build_evidence_pack_from_candidates(
        (candidate,),
        max_context_tokens=100,
        retrieval_unit_coverage={
            "entities": ("3M",),
            "periods": ("2018",),
            "metrics": ("capital_expenditure",),
        },
    )

    coverage = pack.blocks[0].coverage
    assert coverage["entities"]["covered"] == []
    assert coverage["periods"]["covered"] == []
    assert coverage["metrics"]["covered"] == []
    assert coverage["metrics"]["missing"] == ["capital_expenditure"]
