from __future__ import annotations

from atlas.query_runtime.citation_verifier import verify_citations
from atlas.query_runtime.critic_lite import post_generation_critic, pre_generation_critic
from atlas.query_runtime.evidence_evaluator import evaluate_evidence
from atlas.retrieval.evidence import Evidence


def _evidence(
    evidence_id: str = "c1",
    text: str = "3M FY2018 capital expenditures were 1,577 million.",
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        document_id="doc_1",
        chunk_id="chk_1",
        text=text,
        source_title="3M 2018 10-K",
        source_uri=None,
        section_title="Cash flow statement",
        page_start=60,
        page_end=60,
        retrieval_score=1.0,
        rank=1,
        token_count=12,
        metadata={"retrieved_by": ["dense", "bm25"]},
    )


def test_evidence_evaluator_marks_supported_and_insufficient() -> None:
    supported = evaluate_evidence(
        "What was 3M FY2018 capital expenditure?",
        [_evidence()],
    )
    insufficient = evaluate_evidence(
        "What was 3M FY2018 capital expenditure?",
        [],
    )

    assert supported.status == "supported"
    assert supported.supported_evidence_ids == ("c1",)
    assert insufficient.status == "insufficient"
    assert insufficient.confidence_override == "insufficient"


def test_evidence_evaluator_marks_partial_support_without_blocking() -> None:
    result = evaluate_evidence(
        "What was Microsoft 2018 capital expenditure?",
        [_evidence(text="3M annual report discussion without the year.")],
    )

    assert result.status == "partially_supported"
    assert result.confidence_override is None
    assert result.is_blocking is False


def test_evidence_evaluator_marks_conflicting_supported_evidence_as_contradicted() -> None:
    result = evaluate_evidence(
        "What was 3M 2018 capital expenditure?",
        [
            _evidence("c1", "3M 2018 capital expenditures were 1,577 million."),
            _evidence("c2", "3M 2018 capital expenditures were 1,999 million."),
        ],
    )

    assert result.status == "contradicted"
    assert result.is_blocking is True
    assert "evidence_conflict" in result.reasons


def test_evidence_evaluator_does_not_treat_multi_number_tables_as_contradicted() -> None:
    result = evaluate_evidence(
        "What was 3M 2018 capital expenditure?",
        [
            _evidence(
                "c1",
                "3M 2018 capital expenditures were 1,577 million and 2017 was 1,373 million.",
            ),
            _evidence(
                "c2",
                "3M 2018 capital expenditures table includes 1,577, 1,373, and 1,420.",
            ),
        ],
    )

    assert result.status == "supported"
    assert result.details["conflicts"] == []


def test_citation_verifier_supports_valid_citations_and_numbers() -> None:
    result = verify_citations(
        query="What was 3M FY2018 capex?",
        answer="3M FY2018 capital expenditures were 1,577 million [c1].",
        evidence=[_evidence()],
        citations=[{"citation_id": "c1"}],
    )

    assert result.status == "supported"
    assert result.supported_evidence_ids == ("c1",)
    assert result.details["unsupported_numbers"] == []


def test_citation_verifier_keeps_invalid_citations_unsupported() -> None:
    result = verify_citations(
        query="What was 3M FY2018 capex?",
        answer="3M FY2018 capital expenditures were 1,577 million [c9].",
        evidence=[_evidence()],
        citations=[],
    )

    assert result.status == "unsupported"
    assert result.confidence_override == "unsupported"
    assert result.unsupported_evidence_ids == ("c9",)


def test_citation_verifier_separates_numeric_warning_from_unsupported() -> None:
    result = verify_citations(
        query="What was 3M FY2018 capex?",
        answer="3M FY2018 capital expenditures were 1,999 million [c1].",
        evidence=[_evidence()],
        citations=[{"citation_id": "c1"}],
    )

    assert result.status == "warning"
    assert result.confidence_override is None
    assert result.reasons == ("answer_numbers_missing_from_cited_evidence",)
    assert result.details["auto_citation_policy"] == "never_add_missing_citations"


def test_citation_verifier_checks_doc_and_page_metadata() -> None:
    result = verify_citations(
        query="What was 3M FY2018 capex?",
        answer="3M FY2018 capital expenditures were 1,577 million [c1].",
        evidence=[_evidence()],
        citations=[{"citation_id": "c1", "document_id": "wrong_doc", "page_start": 99}],
    )

    assert result.status == "warning"
    assert "citation_metadata_mismatch" in result.reasons
    assert {
        mismatch["field"] for mismatch in result.details["citation_metadata_mismatches"]
    } == {"document_id", "page_start"}


def test_critic_lite_compat_layer_exposes_new_verifications() -> None:
    pre = pre_generation_critic(
        "What was 3M FY2018 capital expenditure?",
        [_evidence()],
    )
    post = post_generation_critic(
        "What was 3M FY2018 capex?",
        "3M FY2018 capital expenditures were 1,577 million [c1].",
        [_evidence()],
        [{"citation_id": "c1"}],
    )

    assert pre.status == "ok"
    assert pre.details["verification"]["status"] == "supported"
    assert post.status == "ok"
    assert post.details["verification"]["status"] == "supported"
