from __future__ import annotations

from atlas.core.ids import new_id
from atlas.query_runtime import critic_lite
from atlas.query_runtime.verification import VerificationResult
from atlas.retrieval.evidence import Evidence


def evaluate_evidence(query: str, evidence: list[Evidence]) -> VerificationResult:
    """Evaluate whether retrieved evidence is sufficient before generation."""
    if not evidence:
        return VerificationResult(
            verification_id=new_id("ev"),
            stage="pre_generation",
            status="insufficient",
            confidence_override="insufficient",
            reasons=("no_evidence",),
            details={"evidence_count": 0},
        )

    anchors = critic_lite._extract_query_anchors(query)
    evidence_by_id = {item.evidence_id.lower(): item for item in evidence}
    supported: list[str] = []
    partially_supported: list[str] = []
    unsupported: list[str] = []
    conflicts: list[dict] = []

    for evidence_item in evidence:
        text = critic_lite._normalize_text(critic_lite._evidence_text(evidence_item))
        missing = [
            anchor
            for anchor in anchors
            if not critic_lite._anchor_in_text(anchor, text)
        ]
        if not missing:
            supported.append(evidence_item.evidence_id)
        elif len(missing) < len(anchors):
            partially_supported.append(evidence_item.evidence_id)
        else:
            unsupported.append(evidence_item.evidence_id)

    conflicts = _detect_numeric_conflicts(query, evidence, supported)

    if conflicts:
        status = "contradicted"
        confidence_override = "unsupported"
        reasons: tuple[str, ...] = ("evidence_conflict",)
    elif supported:
        status = "supported"
        confidence_override = None
        reasons: tuple[str, ...] = ()
    elif partially_supported:
        status = "partially_supported"
        confidence_override = None
        reasons = ("query_anchors_partially_supported",)
    else:
        status = "insufficient"
        confidence_override = "insufficient"
        reasons = ("query_anchors_missing_from_evidence",)

    return VerificationResult(
        verification_id=new_id("ev"),
        stage="pre_generation",
        status=status,
        confidence_override=confidence_override,
        warnings=tuple("evidence contradiction detected" for _ in conflicts),
        reasons=reasons,
        supported_evidence_ids=tuple(supported),
        unsupported_evidence_ids=tuple(unsupported),
        details={
            "evidence_count": len(evidence),
            "query_anchors": [critic_lite._anchor_detail(anchor) for anchor in anchors],
            "partially_supported_evidence_ids": partially_supported,
            "conflicts": conflicts,
            "evidence_ids": sorted(evidence_by_id),
        },
    )


def _detect_numeric_conflicts(
    query: str,
    evidence: list[Evidence],
    supported_evidence_ids: list[str],
) -> list[dict]:
    supported_set = set(supported_evidence_ids)
    if len(supported_set) < 2:
        return []
    answer_numbers_by_evidence: dict[str, list[critic_lite._NumberMention]] = {}
    for item in evidence:
        if item.evidence_id not in supported_set:
            continue
        answer_numbers = _answer_like_numbers(query, item.text)
        if len(answer_numbers) != 1:
            return []
        answer_numbers_by_evidence[item.evidence_id] = answer_numbers
    if len(answer_numbers_by_evidence) < 2:
        return []

    first_id, first_mentions = next(iter(answer_numbers_by_evidence.items()))
    first = first_mentions[0]
    conflicts: list[dict] = []
    for evidence_id, mentions in list(answer_numbers_by_evidence.items())[1:]:
        mention = mentions[0]
        if first.unit != mention.unit:
            continue
        if first.value != mention.value:
            conflicts.append(
                {
                    "type": "numeric_conflict",
                    "policy": "blocking",
                    "left_evidence_id": first_id,
                    "left_value": first.text,
                    "right_evidence_id": evidence_id,
                    "right_value": mention.text,
                }
            )
    return conflicts


def _answer_like_numbers(query: str, text: str) -> list[critic_lite._NumberMention]:
    query_aliases = {
        alias
        for anchor in critic_lite._extract_query_anchors(query)
        for alias in anchor.aliases
    }
    mentions: list[critic_lite._NumberMention] = []
    for mention in critic_lite._extract_numbers(text):
        normalized_text = critic_lite._normalize_text(mention.text)
        if normalized_text in query_aliases:
            continue
        if mention.unit is None and 1900 <= int(mention.value) <= 2099:
            continue
        mentions.append(mention)
    return mentions
