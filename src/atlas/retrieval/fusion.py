from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

from atlas.retrieval.candidate import Candidate

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class WeightedRRFInput:
    lane: str
    candidates: Sequence[Candidate]
    weight: float = 1.0


def rrf_fuse(
    dense_candidates: Sequence[Candidate],
    lexical_candidates: Sequence[Candidate],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[Candidate]:
    """Fuse dense and lexical candidate lists with reciprocal rank fusion."""
    return weighted_rrf_fuse(
        (
            WeightedRRFInput(lane="dense", candidates=dense_candidates),
            WeightedRRFInput(lane="bm25", candidates=lexical_candidates),
        ),
        rrf_k=rrf_k,
        limit=limit,
    )


def weighted_rrf_fuse(
    lane_inputs: Sequence[WeightedRRFInput],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[Candidate]:
    """Fuse any number of provider-local lanes with weighted reciprocal rank fusion."""
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative.")
    if limit is not None and limit <= 0:
        return []

    merged: dict[str, Candidate] = {}
    contributions: dict[str, list[dict[str, Any]]] = {}
    for lane_input in lane_inputs:
        lane_weight = _positive_weight(lane_input.weight, label=f"{lane_input.lane}.weight")
        for position, candidate in enumerate(lane_input.candidates, start=1):
            rank = _lane_rank(candidate, lane_input.lane, position)
            raw_score = _lane_raw_score(candidate, lane_input.lane)
            effective_weight = lane_weight * _positive_weight(
                candidate.unit_weight,
                label="candidate.unit_weight",
            ) * _positive_weight(candidate.lane_weight, label="candidate.lane_weight")
            weighted_contribution = effective_weight * _rrf_contribution(rank, rrf_k=rrf_k)
            contribution = {
                "lane": lane_input.lane,
                "chunk_id": candidate.chunk_id,
                "parent_id": candidate.parent_id or candidate.metadata.get("parent_id"),
                "rank": rank,
                "raw_score": raw_score,
                "weight": effective_weight,
                "unit_weight": candidate.unit_weight,
                "lane_weight": candidate.lane_weight,
                "weighted_contribution": weighted_contribution,
                "retrieval_task_id": candidate.retrieval_task_id,
                "retrieval_unit_id": candidate.retrieval_unit_id,
            }
            contributions.setdefault(candidate.chunk_id, []).append(contribution)
            weighted_candidate = replace(
                candidate,
                lane=candidate.lane or lane_input.lane,
                lane_rank=rank,
                lane_score=raw_score,
                weighted_contribution=weighted_contribution,
                fusion_score=weighted_contribution,
            )
            _merge_candidate(
                merged,
                weighted_candidate,
                source=_lane_source_family(lane_input.lane),
                rank=rank,
            )

    scored = []
    for chunk_id, candidate in merged.items():
        lane_contributions = contributions.get(chunk_id, [])
        scored.append(
            _with_weighted_fusion_score(
                candidate,
                lane_contributions=lane_contributions,
                rrf_k=rrf_k,
            )
        )
    scored.sort(key=_weighted_fusion_sort_key)

    fused: list[Candidate] = []
    for rank, candidate in enumerate(scored[:limit], start=1):
        ranked = replace(candidate, fusion_rank=rank, final_rank=rank)
        fused.append(
            replace(
                ranked,
                metadata=_metadata_with_fusion_trace(ranked, rrf_k=rrf_k),
            )
        )
    return fused


def fusion_trace_payload(candidate: Candidate, *, rrf_k: int | None = None) -> dict[str, Any]:
    """Build an explainable trace payload for a fused candidate."""
    if rrf_k is None:
        existing = candidate.metadata.get("fusion") if candidate.metadata else None
        if isinstance(existing, dict):
            existing_rrf_k = existing.get("rrf_k")
            rrf_k = existing_rrf_k if isinstance(existing_rrf_k, int) else None

    payload = {
        "dense_rank": candidate.dense_rank,
        "dense_score": candidate.dense_score,
        "lexical_rank": candidate.lexical_rank,
        "lexical_score": candidate.lexical_score,
        "lexical_backend": candidate.lexical_backend,
        "fusion_rank": candidate.fusion_rank,
        "fusion_score": candidate.fusion_score,
        "final_rank": candidate.final_rank,
        "rrf_k": rrf_k,
        "present_in_dense": candidate.dense_rank is not None,
        "present_in_lexical": candidate.lexical_rank is not None,
        "retrieved_by": list(candidate.retrieved_by),
    }
    fusion = candidate.metadata.get("fusion") if candidate.metadata else None
    if isinstance(fusion, dict):
        for key in (
            "strategy",
            "lane",
            "lanes",
            "lane_contributions",
            "weighted_contribution",
        ):
            if key in fusion:
                payload[key] = fusion[key]
    return payload


def _merge_candidate(
    merged: dict[str, Candidate],
    candidate: Candidate,
    *,
    source: str,
    rank: int,
) -> None:
    existing = merged.get(candidate.chunk_id)
    if existing is None:
        merged[candidate.chunk_id] = _candidate_with_source(candidate, source=source, rank=rank)
        return

    merged[candidate.chunk_id] = _merge_same_chunk(existing, candidate, source=source, rank=rank)


def _candidate_with_source(candidate: Candidate, *, source: str, rank: int) -> Candidate:
    source_label = _source_label(candidate, source)
    dense_rank = candidate.dense_rank
    lexical_rank = candidate.lexical_rank
    dense_score = candidate.dense_score
    lexical_score = candidate.lexical_score

    if source == "dense":
        dense_rank = _rank_or_position(candidate.dense_rank, rank)
    elif source == "lexical":
        lexical_rank = _rank_or_position(candidate.lexical_rank, rank)

    return replace(
        candidate,
        retrieved_by=_retrieved_by(candidate.retrieved_by, source_label),
        dense_rank=dense_rank,
        dense_score=dense_score,
        lexical_rank=lexical_rank,
        lexical_score=lexical_score,
        metadata=dict(candidate.metadata),
    )


def _merge_same_chunk(
    existing: Candidate,
    candidate: Candidate,
    *,
    source: str,
    rank: int,
) -> Candidate:
    incoming = _candidate_with_source(candidate, source=source, rank=rank)
    return replace(
        existing,
        retrieved_by=_retrieved_by(existing.retrieved_by, *incoming.retrieved_by),
        dense_rank=_best_rank(existing.dense_rank, incoming.dense_rank),
        dense_score=_score_for_rank(
            existing_score=existing.dense_score,
            incoming_score=incoming.dense_score,
            existing_rank=existing.dense_rank,
            incoming_rank=incoming.dense_rank,
        ),
        lexical_rank=_best_rank(existing.lexical_rank, incoming.lexical_rank),
        lexical_score=_score_for_rank(
            existing_score=existing.lexical_score,
            incoming_score=incoming.lexical_score,
            existing_rank=existing.lexical_rank,
            incoming_rank=incoming.lexical_rank,
        ),
        lexical_backend=existing.lexical_backend or incoming.lexical_backend,
        metadata=_merge_metadata(existing.metadata, incoming.metadata),
        source_uri=existing.source_uri or incoming.source_uri,
        section_title=existing.section_title or incoming.section_title,
    )


def _with_weighted_fusion_score(
    candidate: Candidate,
    *,
    lane_contributions: list[dict[str, Any]],
    rrf_k: int,
) -> Candidate:
    fusion_score = sum(
        float(item.get("weighted_contribution") or 0.0)
        for item in lane_contributions
    )
    lanes = _ordered_unique(
        str(item.get("lane")) for item in lane_contributions if item.get("lane")
    )
    best = _best_contribution(lane_contributions)
    lane = "multi_lane" if len(lanes) > 1 else (lanes[0] if lanes else candidate.lane)
    return replace(
        candidate,
        lane=lane,
        lane_rank=_int_or_none(best.get("rank")) if best else candidate.lane_rank,
        lane_score=_float_or_none(best.get("raw_score")) if best else candidate.lane_score,
        lane_weight=_float_or_default(best.get("lane_weight"), candidate.lane_weight)
        if best
        else candidate.lane_weight,
        weighted_contribution=fusion_score,
        fusion_score=fusion_score,
        metadata=_metadata_with_weighted_trace(
            candidate,
            lane=lane,
            lanes=lanes,
            lane_contributions=lane_contributions,
            fusion_score=fusion_score,
            rrf_k=rrf_k,
        ),
    )


def _metadata_with_fusion_trace(candidate: Candidate, *, rrf_k: int) -> dict[str, Any]:
    metadata = dict(candidate.metadata)
    existing = metadata.get("fusion")
    payload = fusion_trace_payload(candidate, rrf_k=rrf_k)
    if isinstance(existing, dict):
        payload = {**existing, **payload}
    metadata["fusion"] = payload
    return metadata


def _metadata_with_weighted_trace(
    candidate: Candidate,
    *,
    lane: str | None,
    lanes: list[str],
    lane_contributions: list[dict[str, Any]],
    fusion_score: float,
    rrf_k: int,
) -> dict[str, Any]:
    metadata = dict(candidate.metadata)
    metadata["lane"] = lane
    metadata["lanes"] = lanes
    metadata["weighted_contribution"] = fusion_score
    metadata["lane_contributions"] = list(lane_contributions)
    metadata["fusion"] = {
        "strategy": "weighted_rrf",
        "rrf_k": rrf_k,
        "lane": lane,
        "lanes": lanes,
        "lane_contributions": list(lane_contributions),
        "weighted_contribution": fusion_score,
        "fusion_score": fusion_score,
        "dense_rank": candidate.dense_rank,
        "dense_score": candidate.dense_score,
        "lexical_rank": candidate.lexical_rank,
        "lexical_score": candidate.lexical_score,
        "lexical_backend": candidate.lexical_backend,
        "present_in_dense": candidate.dense_rank is not None,
        "present_in_lexical": candidate.lexical_rank is not None,
        "retrieved_by": list(candidate.retrieved_by),
    }
    return metadata


def _fusion_sort_key(candidate: Candidate) -> tuple[float, int, int, int, str]:
    return (
        -(candidate.fusion_score or 0.0),
        _best_available_rank(candidate),
        candidate.dense_rank or _MISSING_RANK,
        candidate.lexical_rank or _MISSING_RANK,
        candidate.chunk_id,
    )


def _weighted_fusion_sort_key(candidate: Candidate) -> tuple[float, int, int, int, str]:
    return (
        -(candidate.fusion_score or 0.0),
        candidate.lane_rank or _best_available_rank(candidate),
        candidate.dense_rank or _MISSING_RANK,
        candidate.lexical_rank or _MISSING_RANK,
        candidate.chunk_id,
    )


def _rrf_contribution(rank: int | None, *, rrf_k: int) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (rrf_k + rank)


def _lane_rank(candidate: Candidate, lane: str, position: int) -> int:
    if candidate.lane_rank is not None and candidate.lane_rank > 0:
        return candidate.lane_rank
    if lane == "dense":
        return _rank_or_position(candidate.dense_rank, position)
    return _rank_or_position(candidate.lexical_rank, position)


def _lane_raw_score(candidate: Candidate, lane: str) -> float | None:
    if candidate.lane_score is not None:
        return candidate.lane_score
    if lane == "dense":
        return candidate.dense_score
    return candidate.lexical_score


def _lane_source_family(lane: str) -> str:
    return "dense" if lane == "dense" else "lexical"


def _positive_weight(value: float, *, label: str) -> float:
    weight = float(value)
    if weight < 0:
        raise ValueError(f"{label} must be non-negative.")
    return weight


def _best_contribution(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    return max(
        items,
        key=lambda item: (
            float(item.get("weighted_contribution") or 0.0),
            -int(item.get("rank") or _MISSING_RANK),
        ),
    )


def _rank_or_position(rank: int | None, position: int) -> int:
    if rank is not None and rank > 0:
        return rank
    return position


def _best_rank(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _score_for_rank(
    *,
    existing_score: float | None,
    incoming_score: float | None,
    existing_rank: int | None,
    incoming_rank: int | None,
) -> float | None:
    if existing_score is None:
        return incoming_score
    if incoming_score is None:
        return existing_score
    if existing_rank is None:
        return incoming_score
    if incoming_rank is None:
        return existing_score
    return incoming_score if incoming_rank < existing_rank else existing_score


def _best_available_rank(candidate: Candidate) -> int:
    ranks = [
        rank
        for rank in (candidate.dense_rank, candidate.lexical_rank)
        if rank is not None
    ]
    return min(ranks, default=_MISSING_RANK)


def _retrieved_by(existing: Sequence[str], *sources: str) -> tuple[str, ...]:
    values = [*existing, *sources]
    ordered: list[str] = []
    for source in ("dense", "bm25", "metric_alias", "section", "table", "lexical"):
        if source in values:
            ordered.append(source)
    for source in values:
        if source not in ordered:
            ordered.append(source)
    return tuple(ordered)


def _source_label(candidate: Candidate, source: str) -> str:
    if source == "lexical" and "bm25" in candidate.retrieved_by:
        return "bm25"
    return source


def _merge_metadata(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(left)
    metadata.update(right)
    return metadata


def _ordered_unique(values) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_default(value: Any, default: float) -> float:
    parsed = _float_or_none(value)
    return default if parsed is None else parsed


_MISSING_RANK = 1_000_000_000
