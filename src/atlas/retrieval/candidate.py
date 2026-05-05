from dataclasses import dataclass, field
from typing import Any


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
