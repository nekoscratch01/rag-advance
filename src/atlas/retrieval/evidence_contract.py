from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceBlock:
    evidence_id: str
    source_type: str
    provider: str
    text: str
    document_id: str
    doc_name: str
    page_start: int | None
    page_end: int | None
    chunk_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...] = ()
    retrieval_sources: tuple[str, ...] = ()
    best_dense_rank: int | None = None
    best_bm25_rank: int | None = None
    best_rrf_score: float | None = None
    rerank_score: float | None = None
    rank: int | None = None
    retrieval_score: float = 0.0
    source_title: str | None = None
    source_uri: str | None = None
    section_title: str | None = None
    token_count: int = 0
    coverage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    child_ids: tuple[str, ...] = ()
    included_in_prompt: bool = False
    drop_reason: str | None = None
    drop_stage: str | None = None


@dataclass(frozen=True)
class EvidencePack:
    pack_id: str
    query_id: str | None
    plan_id: str | None
    blocks: tuple[EvidenceBlock, ...]
    dropped_blocks: tuple[EvidenceBlock, ...] = ()
    token_count: int = 0
    max_context_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_blocks(self) -> tuple[EvidenceBlock, ...]:
        return tuple(block for block in self.blocks if block.included_in_prompt)

    @property
    def all_blocks(self) -> tuple[EvidenceBlock, ...]:
        return (*self.blocks, *self.dropped_blocks)
