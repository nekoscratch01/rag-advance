from dataclasses import dataclass, field
from typing import Any, Literal


FusionPolicy = Literal["ranked", "pinned", "supporting"]


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    document_id: str
    doc_name: str
    source_title: str
    company: str | None
    text: str
    page_start: int | None
    page_end: int | None
    chunk_index: int
    token_count: int
    retrieved_by: tuple[str, ...]
    dense_rank: int | None
    dense_score: float | None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    lexical_backend: str | None = None
    fusion_rank: int | None = None
    fusion_score: float | None = None
    rerank_rank: int | None = None
    rerank_score: float | None = None
    final_rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_uri: str | None = None
    section_title: str | None = None
    parent_id: str | None = None
    candidate_id: str | None = None
    provider: str = "text_hybrid"
    source_type: str = "text_chunk"
    lane: str | None = None
    retrieval_task_id: str | None = None
    retrieval_unit_id: str | None = None
    unit_weight: float = 1.0
    lane_weight: float = 1.0
    lane_rank: int | None = None
    lane_score: float | None = None
    weighted_contribution: float | None = None
    rerankable: bool = True
    fusion_policy: FusionPolicy = "ranked"
    structured_payload: dict[str, Any] = field(default_factory=dict)
