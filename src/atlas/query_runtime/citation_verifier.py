from __future__ import annotations

from typing import Any

from atlas.core.ids import new_id
from atlas.query_runtime import critic_lite
from atlas.query_runtime.verification import VerificationResult
from atlas.retrieval.models.evidence import Evidence


def verify_citations(
    *,
    query: str,
    answer: str,
    evidence: list[Evidence],
    citations: list[dict[str, Any]],
) -> VerificationResult:
    """Verify that generated citations refer to supplied evidence and support key numbers."""
    evidence_by_id = {item.evidence_id.lower(): item for item in evidence}
    evidence_ids = set(evidence_by_id)
    answer_citation_ids = critic_lite._extract_citation_ids(answer)
    provided_citation_ids = critic_lite._extract_provided_citation_ids(citations)
    referenced_ids = critic_lite._dedupe([*answer_citation_ids, *provided_citation_ids])
    invalid = [citation_id for citation_id in referenced_ids if citation_id not in evidence_ids]
    cited_evidence_ids = [
        citation_id for citation_id in referenced_ids if citation_id in evidence_by_id
    ]
    metadata_mismatches = _citation_metadata_mismatches(citations, evidence_by_id)
    unsupported_numbers = critic_lite._unsupported_answer_numbers(
        answer,
        cited_evidence_ids,
        evidence_by_id,
    )

    warnings: list[str] = []
    reasons: list[str] = []
    if not answer_citation_ids:
        reasons.append("answer_has_no_citations")
    if invalid:
        reasons.append("citation_not_in_evidence_set")
    if unsupported_numbers:
        reasons.append("answer_numbers_missing_from_cited_evidence")
        warnings.append(
            "Answer numbers not found in cited evidence: "
            + ", ".join(number.text for number in unsupported_numbers)
        )
    if metadata_mismatches:
        reasons.append("citation_metadata_mismatch")
        warnings.append("Citation document/page metadata does not match evidence.")

    if "answer_has_no_citations" in reasons or "citation_not_in_evidence_set" in reasons:
        status = "unsupported"
        confidence_override = "unsupported"
    elif (
        "answer_numbers_missing_from_cited_evidence" in reasons
        or "citation_metadata_mismatch" in reasons
    ):
        status = "warning"
        confidence_override = None
    else:
        status = "supported"
        confidence_override = None

    return VerificationResult(
        verification_id=new_id("cv"),
        stage="citation",
        status=status,
        confidence_override=confidence_override,
        warnings=tuple(warnings),
        reasons=tuple(reasons),
        supported_evidence_ids=tuple(cited_evidence_ids),
        unsupported_evidence_ids=tuple(invalid),
        details={
            "query": query,
            "evidence_ids": sorted(evidence_ids),
            "answer_citation_ids": answer_citation_ids,
            "provided_citation_ids": provided_citation_ids,
            "invalid_citation_ids": invalid,
            "checked_numbers": [
                mention.text for mention in critic_lite._extract_numbers(answer)
            ],
            "unsupported_numbers": [mention.text for mention in unsupported_numbers],
            "citation_metadata_mismatches": metadata_mismatches,
            "auto_citation_policy": "never_add_missing_citations",
        },
    )


def _citation_metadata_mismatches(
    citations: list[dict[str, Any]],
    evidence_by_id: dict[str, Evidence],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        raw_id = citation.get("citation_id") or citation.get("evidence_id") or citation.get("id")
        citation_id = str(raw_id).strip().lower() if raw_id is not None else ""
        evidence = evidence_by_id.get(citation_id)
        if evidence is None:
            continue
        checks = {
            "document_id": evidence.document_id,
            "chunk_id": evidence.chunk_id,
            "page_start": evidence.page_start,
            "page_end": evidence.page_end,
        }
        for field, expected in checks.items():
            if field not in citation or citation.get(field) is None:
                continue
            observed = citation.get(field)
            if str(observed) != str(expected):
                mismatches.append(
                    {
                        "citation_id": citation_id,
                        "field": field,
                        "expected": expected,
                        "observed": observed,
                    }
                )
    return mismatches
