from __future__ import annotations

import pytest

from atlas.retrieval.candidate import Candidate
from atlas.retrieval.fusion import WeightedRRFInput, rrf_fuse, weighted_rrf_fuse


def _candidate(
    chunk_id: str,
    *,
    lane: str,
    rank: int,
    score: float,
    unit_weight: float = 1.0,
    lane_weight: float = 1.0,
) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        document_id="doc_1",
        doc_name="doc",
        source_title="doc",
        company=None,
        text=f"{chunk_id} text",
        page_start=1,
        page_end=1,
        chunk_index=rank,
        token_count=5,
        retrieved_by=(lane,),
        dense_rank=rank if lane == "dense" else None,
        dense_score=score if lane == "dense" else None,
        lexical_rank=rank if lane != "dense" else None,
        lexical_score=score if lane != "dense" else None,
        lexical_backend="qdrant_bm25" if lane != "dense" else None,
        lane=lane,
        lane_rank=rank,
        lane_score=score,
        unit_weight=unit_weight,
        lane_weight=lane_weight,
    )


def test_weighted_rrf_promotes_high_weight_lane_and_records_contributions() -> None:
    fused = weighted_rrf_fuse(
        (
            WeightedRRFInput(
                lane="dense",
                candidates=(
                    _candidate("dense_only", lane="dense", rank=1, score=0.9),
                    _candidate("shared", lane="dense", rank=2, score=0.8),
                ),
            ),
            WeightedRRFInput(
                lane="table",
                candidates=(
                    _candidate(
                        "shared",
                        lane="table",
                        rank=1,
                        score=12.0,
                        unit_weight=2.0,
                        lane_weight=1.5,
                    ),
                ),
            ),
        ),
        rrf_k=0,
        limit=2,
    )

    assert [candidate.chunk_id for candidate in fused] == ["shared", "dense_only"]
    assert fused[0].lane == "multi_lane"
    assert fused[0].fusion_score == pytest.approx(3.5)
    assert fused[0].weighted_contribution == pytest.approx(3.5)
    assert fused[0].lane_weight == 1.5
    assert fused[0].metadata["fusion"]["strategy"] == "weighted_rrf"
    assert fused[0].metadata["fusion"]["lanes"] == ["dense", "table"]
    assert {
        item["lane"]: item["weighted_contribution"]
        for item in fused[0].metadata["fusion"]["lane_contributions"]
    } == {"dense": 0.5, "table": 3.0}
    table_contribution = next(
        item
        for item in fused[0].metadata["fusion"]["lane_contributions"]
        if item["lane"] == "table"
    )
    assert table_contribution["unit_weight"] == 2.0
    assert table_contribution["lane_weight"] == 1.5
    assert table_contribution["weight"] == 3.0


def test_rrf_fuse_remains_weight_one_compatibility_wrapper() -> None:
    fused = rrf_fuse(
        [_candidate("dense_only", lane="dense", rank=1, score=0.9)],
        [_candidate("bm25_only", lane="bm25", rank=1, score=9.0)],
        rrf_k=60,
        limit=2,
    )

    assert len(fused) == 2
    assert fused[0].fusion_score == pytest.approx(1 / 61)
    assert fused[0].metadata["fusion"]["strategy"] == "weighted_rrf"
    assert fused[0].metadata["fusion"]["lane_contributions"][0]["weight"] == 1.0


def test_weighted_rrf_rejects_negative_weights() -> None:
    with pytest.raises(ValueError):
        weighted_rrf_fuse(
            (
                WeightedRRFInput(
                    lane="dense",
                    candidates=(
                        _candidate(
                            "bad",
                            lane="dense",
                            rank=1,
                            score=0.9,
                            lane_weight=-1.0,
                        ),
                    ),
                ),
            )
        )
