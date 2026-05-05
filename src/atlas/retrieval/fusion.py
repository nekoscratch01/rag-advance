from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from atlas.retrieval.candidate import Candidate

DEFAULT_RRF_K = 60


def rrf_fuse(
    dense_candidates: Sequence[Candidate],
    lexical_candidates: Sequence[Candidate],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[Candidate]:
    """Fuse dense and lexical candidate lists with reciprocal rank fusion."""
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative.")
    if limit is not None and limit <= 0:
        return []

    merged: dict[str, Candidate] = {}
    for position, candidate in enumerate(dense_candidates, start=1):
        _merge_candidate(merged, candidate, source="dense", rank=position)
    for position, candidate in enumerate(lexical_candidates, start=1):
        _merge_candidate(merged, candidate, source="lexical", rank=position)

    scored = [
        _with_fusion_score(candidate, rrf_k=rrf_k)
        for candidate in merged.values()
    ]
    scored.sort(key=_fusion_sort_key)

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

    return {
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


def _with_fusion_score(candidate: Candidate, *, rrf_k: int) -> Candidate:
    dense_score = _rrf_contribution(candidate.dense_rank, rrf_k=rrf_k)
    lexical_score = _rrf_contribution(candidate.lexical_rank, rrf_k=rrf_k)
    return replace(candidate, fusion_score=dense_score + lexical_score)


def _metadata_with_fusion_trace(candidate: Candidate, *, rrf_k: int) -> dict[str, Any]:
    metadata = dict(candidate.metadata)
    metadata["fusion"] = fusion_trace_payload(candidate, rrf_k=rrf_k)
    return metadata


def _fusion_sort_key(candidate: Candidate) -> tuple[float, int, int, int, str]:
    return (
        -(candidate.fusion_score or 0.0),
        _best_available_rank(candidate),
        candidate.dense_rank or _MISSING_RANK,
        candidate.lexical_rank or _MISSING_RANK,
        candidate.chunk_id,
    )


def _rrf_contribution(rank: int | None, *, rrf_k: int) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (rrf_k + rank)


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
    for source in ("dense", "bm25", "lexical"):
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


_MISSING_RANK = 1_000_000_000
