from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from atlas.ingestion.chunker import approx_token_count
from atlas.retrieval.candidate import Candidate
from atlas.retrieval.evidence import Evidence


@dataclass(frozen=True)
class EvidenceBlock:
    document_id: str
    doc_name: str
    source_title: str
    text: str
    page_start: int | None
    page_end: int | None
    chunk_ids: tuple[str, ...]
    retrieved_by: tuple[str, ...]
    sources: tuple[str, ...]
    best_dense_rank: int | None
    best_lexical_rank: int | None
    best_fusion_rank: int | None
    best_fusion_score: float | None
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    source_uri: str | None = None
    section_title: str | None = None
    best_dense_score: float | None = None
    best_lexical_score: float | None = None
    final_rank: int | None = None
    retrieval_score: float = 0.0
    parent_id: str | None = None
    child_ids: tuple[str, ...] = ()
    best_rerank_rank: int | None = None
    best_rerank_score: float | None = None
    rank: int | None = None


@dataclass(frozen=True)
class _CandidateEntry:
    order: int
    chunk_id: str
    parent_id: str | None
    document_id: str
    doc_name: str
    source_title: str
    company: str | None
    text: str
    page_start: int | None
    page_end: int | None
    chunk_index: int | None
    token_count: int
    retrieved_by: tuple[str, ...]
    dense_rank: int | None
    dense_score: float | None
    lexical_rank: int | None
    lexical_score: float | None
    lexical_backend: str | None
    fusion_rank: int | None
    fusion_score: float | None
    final_rank: int | None
    rerank_rank: int | None
    rerank_score: float | None
    metadata: dict[str, Any]
    source_uri: str | None
    section_title: str | None


@dataclass
class _ChunkAccumulator:
    entries: list[_CandidateEntry]

    def add(self, entry: _CandidateEntry) -> None:
        self.entries.append(entry)

    def to_entry(self) -> _CandidateEntry:
        entries = self.entries
        primary = min(entries, key=_entry_sort_key)
        retrieved_by = _unique(
            source
            for entry in sorted(entries, key=_entry_sort_key)
            for source in entry.retrieved_by
        )
        lexical_backends = _unique(
            entry.lexical_backend for entry in entries if entry.lexical_backend
        )
        metadata = _merge_metadata(entry.metadata for entry in entries)
        if lexical_backends:
            metadata.setdefault("lexical_backends", lexical_backends)

        return _CandidateEntry(
            order=min(entry.order for entry in entries),
            chunk_id=primary.chunk_id,
            parent_id=_first_optional_text(entry.parent_id for entry in entries),
            document_id=_first_text(entry.document_id for entry in entries),
            doc_name=_first_text(entry.doc_name for entry in entries),
            source_title=_first_text(entry.source_title or entry.doc_name for entry in entries),
            company=_first_optional_text(entry.company for entry in entries),
            text=primary.text,
            page_start=_min_optional_int(entry.page_start for entry in entries),
            page_end=_max_optional_int(entry.page_end for entry in entries),
            chunk_index=_min_optional_int(entry.chunk_index for entry in entries),
            token_count=max(entry.token_count for entry in entries),
            retrieved_by=retrieved_by,
            dense_rank=_min_optional_int(entry.dense_rank for entry in entries),
            dense_score=_max_optional_float(entry.dense_score for entry in entries),
            lexical_rank=_min_optional_int(entry.lexical_rank for entry in entries),
            lexical_score=_max_optional_float(entry.lexical_score for entry in entries),
            lexical_backend=lexical_backends[0] if lexical_backends else None,
            fusion_rank=_min_optional_int(entry.fusion_rank for entry in entries),
            fusion_score=_max_optional_float(entry.fusion_score for entry in entries),
            final_rank=_min_optional_int(entry.final_rank for entry in entries),
            rerank_rank=_min_optional_int(entry.rerank_rank for entry in entries),
            rerank_score=_max_optional_float(entry.rerank_score for entry in entries),
            metadata=metadata,
            source_uri=_first_optional_text(entry.source_uri for entry in entries),
            section_title=_first_optional_text(entry.section_title for entry in entries),
        )


@dataclass
class _ParentAccumulator:
    parent_key: str
    entries: list[_CandidateEntry]

    @property
    def best_entry(self) -> _CandidateEntry:
        return min(self.entries, key=_entry_sort_key)

    def add(self, entry: _CandidateEntry) -> None:
        self.entries.append(entry)

    def sort_key(self) -> tuple[int, int, int, int]:
        return _entry_sort_key(self.best_entry)

    def to_block(
        self,
        *,
        parent_resolver: Any = None,
        token_budget: int | None = None,
    ) -> EvidenceBlock | None:
        best = self.best_entry
        parent = _resolve_parent(parent_resolver, best)
        parent_metadata = _parent_dict(parent, "metadata", "metadata_json")
        parent_text = _text_value(_parent_value(parent, "text", "content")).strip()
        text = parent_text or best.text
        if token_budget is not None:
            text = _truncate_text_to_budget(text, token_budget)
        text = text.strip()
        if not text:
            return None

        token_count = approx_token_count(text)
        entries = sorted(self.entries, key=_entry_sort_key)
        child_ids = _unique(entry.chunk_id for entry in entries)
        parent_id = _parent_id(parent) or best.parent_id
        metadata = _merge_metadata([*(entry.metadata for entry in entries), parent_metadata])
        rank = _best_rank(best)

        return EvidenceBlock(
            document_id=_first_text(
                (
                    _optional_text(_parent_value(parent, "document_id")),
                    *(entry.document_id for entry in entries),
                )
            ),
            doc_name=_first_text(
                (
                    _optional_text(_parent_value(parent, "doc_name", "document_name")),
                    *(entry.doc_name for entry in entries),
                )
            ),
            source_title=_first_text(
                (
                    _optional_text(_parent_value(parent, "source_title", "title")),
                    *(entry.source_title or entry.doc_name for entry in entries),
                )
            ),
            text=text,
            page_start=_first_int(
                _parent_value(parent, "page_start"),
                _min_optional_int(entry.page_start for entry in entries),
            ),
            page_end=_first_int(
                _parent_value(parent, "page_end"),
                _max_optional_int(entry.page_end for entry in entries),
            ),
            chunk_ids=child_ids,
            retrieved_by=_unique(source for entry in entries for source in entry.retrieved_by),
            sources=_unique(source for entry in entries for source in entry.retrieved_by),
            best_dense_rank=_min_optional_int(entry.dense_rank for entry in entries),
            best_lexical_rank=_min_optional_int(entry.lexical_rank for entry in entries),
            best_fusion_rank=_min_optional_int(entry.fusion_rank for entry in entries),
            best_fusion_score=_max_optional_float(entry.fusion_score for entry in entries),
            token_count=token_count,
            metadata=metadata,
            source_uri=_first_optional_text(
                (
                    _optional_text(_parent_value(parent, "source_uri")),
                    *(entry.source_uri for entry in entries),
                )
            ),
            section_title=_first_optional_text(
                (
                    _optional_text(_parent_value(parent, "section_title")),
                    *(entry.section_title for entry in entries),
                )
            ),
            best_dense_score=_max_optional_float(entry.dense_score for entry in entries),
            best_lexical_score=_max_optional_float(entry.lexical_score for entry in entries),
            final_rank=_min_optional_int(entry.final_rank for entry in entries),
            retrieval_score=_block_retrieval_score(entries),
            parent_id=parent_id,
            child_ids=child_ids,
            best_rerank_rank=_min_optional_int(entry.rerank_rank for entry in entries),
            best_rerank_score=_max_optional_float(entry.rerank_score for entry in entries),
            rank=rank,
        )


def build_evidence_blocks(
    candidates: Sequence[Candidate],
    *,
    parent_resolver: Any = None,
    max_context_tokens: int,
    max_blocks: int | None = None,
) -> list[EvidenceBlock]:
    if max_context_tokens <= 0:
        return []
    if max_blocks is not None and max_blocks <= 0:
        return []

    entries = _dedupe_candidates(candidates)
    parents = _dedupe_parents(entries)

    blocks: list[EvidenceBlock] = []
    used_tokens = 0
    for accumulator in parents:
        if max_blocks is not None and len(blocks) >= max_blocks:
            break

        remaining = max_context_tokens - used_tokens
        if remaining <= 0:
            break

        block = accumulator.to_block(
            parent_resolver=parent_resolver,
            token_budget=remaining,
        )
        if block is None:
            continue

        blocks.append(block)
        used_tokens += max(0, block.token_count)

    return blocks


def evidence_blocks_to_evidence(blocks: Sequence[EvidenceBlock]) -> list[Evidence]:
    ordered_blocks = sorted(
        enumerate(blocks),
        key=lambda item: _evidence_block_sort_key(item[1], item[0]),
    )
    evidence: list[Evidence] = []
    for index, (_, block) in enumerate(ordered_blocks, start=1):
        child_ids = block.child_ids or block.chunk_ids
        retrieved_by = block.retrieved_by or block.sources
        rank = _first_present_int(
            block.rank,
            block.final_rank,
            block.best_rerank_rank,
            block.best_fusion_rank,
            index,
        )
        evidence.append(
            Evidence(
                evidence_id=f"c{index}",
                document_id=block.document_id,
                chunk_id=child_ids[0] if child_ids else "",
                text=block.text,
                source_title=block.source_title or block.doc_name,
                source_uri=block.source_uri,
                section_title=block.section_title,
                page_start=block.page_start,
                page_end=block.page_end,
                retrieval_score=float(block.retrieval_score),
                rank=rank,
                token_count=block.token_count,
                metadata={
                    **block.metadata,
                    "doc_name": block.doc_name,
                    "parent_id": block.parent_id,
                    "child_ids": list(child_ids),
                    "chunk_ids": list(child_ids),
                    "retrieved_by": list(retrieved_by),
                    "sources": list(retrieved_by),
                    "rank": rank,
                    "final_rank": block.final_rank,
                    "best_child_rank": block.rank,
                    "best_dense_rank": block.best_dense_rank,
                    "best_lexical_rank": block.best_lexical_rank,
                    "best_fusion_rank": block.best_fusion_rank,
                    "best_fusion_score": block.best_fusion_score,
                    "best_dense_score": block.best_dense_score,
                    "best_lexical_score": block.best_lexical_score,
                    "rerank_rank": block.best_rerank_rank,
                    "rerank_score": block.best_rerank_score,
                    "best_rerank_rank": block.best_rerank_rank,
                    "best_rerank_score": block.best_rerank_score,
                },
                parent_id=block.parent_id,
                child_ids=child_ids,
                retrieved_by=retrieved_by,
                rerank_score=block.best_rerank_score,
                rerank_rank=block.best_rerank_rank,
            )
        )
    return evidence


def build_evidence_from_candidates(
    candidates: Sequence[Candidate],
    *,
    parent_resolver: Any = None,
    max_context_tokens: int,
    max_blocks: int | None = None,
) -> list[Evidence]:
    blocks = build_evidence_blocks(
        candidates,
        parent_resolver=parent_resolver,
        max_context_tokens=max_context_tokens,
        max_blocks=max_blocks,
    )
    return evidence_blocks_to_evidence(blocks)


def _dedupe_candidates(candidates: Sequence[Candidate]) -> list[_CandidateEntry]:
    chunks: dict[str, _ChunkAccumulator] = {}
    for order, candidate in enumerate(candidates):
        entry = _candidate_to_entry(candidate, order)
        dedupe_key = entry.chunk_id or f"__missing_chunk_id__:{order}"
        if dedupe_key in chunks:
            chunks[dedupe_key].add(entry)
        else:
            chunks[dedupe_key] = _ChunkAccumulator([entry])
    return sorted((chunk.to_entry() for chunk in chunks.values()), key=_entry_sort_key)


def _dedupe_parents(entries: Sequence[_CandidateEntry]) -> list[_ParentAccumulator]:
    parents: dict[str, _ParentAccumulator] = {}
    for entry in sorted(entries, key=_entry_sort_key):
        parent_key = entry.parent_id or entry.chunk_id or f"__missing_parent__:{entry.order}"
        if parent_key in parents:
            parents[parent_key].add(entry)
        else:
            parents[parent_key] = _ParentAccumulator(parent_key, [entry])
    return sorted(parents.values(), key=lambda item: item.sort_key())


def _candidate_to_entry(candidate: Candidate, order: int) -> _CandidateEntry:
    text = _text_value(getattr(candidate, "text", ""))
    token_count = _int_or_none(getattr(candidate, "token_count", None))
    if token_count is None or token_count <= 0:
        token_count = approx_token_count(text) if text else 0

    metadata = _dict_value(getattr(candidate, "metadata", None))
    lexical_backend = _optional_text(getattr(candidate, "lexical_backend", None))
    retrieved_by = _candidate_sources(candidate, lexical_backend)

    return _CandidateEntry(
        order=order,
        chunk_id=_text_value(getattr(candidate, "chunk_id", "")),
        parent_id=_candidate_parent_id(candidate) or _metadata_parent_id(metadata),
        document_id=_text_value(getattr(candidate, "document_id", "")),
        doc_name=_text_value(getattr(candidate, "doc_name", "")),
        source_title=_text_value(getattr(candidate, "source_title", "")),
        company=_optional_text(getattr(candidate, "company", None)),
        text=text,
        page_start=_int_or_none(getattr(candidate, "page_start", None)),
        page_end=_int_or_none(getattr(candidate, "page_end", None)),
        chunk_index=_int_or_none(getattr(candidate, "chunk_index", None)),
        token_count=token_count,
        retrieved_by=retrieved_by,
        dense_rank=_int_or_none(getattr(candidate, "dense_rank", None)),
        dense_score=_float_or_none(getattr(candidate, "dense_score", None)),
        lexical_rank=_int_or_none(getattr(candidate, "lexical_rank", None)),
        lexical_score=_float_or_none(getattr(candidate, "lexical_score", None)),
        lexical_backend=lexical_backend,
        fusion_rank=_int_or_none(getattr(candidate, "fusion_rank", None)),
        fusion_score=_float_or_none(getattr(candidate, "fusion_score", None)),
        final_rank=_int_or_none(getattr(candidate, "final_rank", None)),
        rerank_rank=_int_or_none(
            getattr(candidate, "rerank_rank", None) or metadata.get("rerank_rank")
        ),
        rerank_score=_float_or_none(
            getattr(candidate, "rerank_score", None) or metadata.get("rerank_score")
        ),
        metadata=metadata,
        source_uri=_optional_text(getattr(candidate, "source_uri", None)),
        section_title=_optional_text(getattr(candidate, "section_title", None)),
    )


def _resolve_parent(parent_resolver: Any, entry: _CandidateEntry) -> Any:
    if parent_resolver is None or not entry.parent_id:
        return None

    if isinstance(parent_resolver, Mapping):
        return parent_resolver.get(entry.parent_id)

    get = getattr(parent_resolver, "get", None)
    if callable(get):
        parent = get(entry.parent_id)
        if parent is not None:
            return parent

    if not callable(parent_resolver):
        return None

    try:
        return parent_resolver(entry.parent_id)
    except TypeError:
        return parent_resolver(entry)


def _parent_value(parent: Any, *names: str) -> Any:
    if parent is None:
        return None
    for name in names:
        if isinstance(parent, Mapping) and name in parent:
            return parent[name]
        value = getattr(parent, name, None)
        if value is not None:
            return value
    return None


def _parent_dict(parent: Any, *names: str) -> dict[str, Any]:
    value = _parent_value(parent, *names)
    return _dict_value(value)


def _parent_id(value: Any) -> str | None:
    return _first_optional_text(
        _optional_text(_parent_value(value, key))
        for key in (
            "parent_id",
            "parent_chunk_id",
            "block_id",
            "chunk_id",
            "id",
        )
    )


def _candidate_parent_id(candidate: Any) -> str | None:
    return _first_optional_text(
        _optional_text(_parent_value(candidate, key))
        for key in (
            "parent_id",
            "parent_chunk_id",
            "parent_block_id",
            "block_id",
        )
    )


def _metadata_parent_id(metadata: Mapping[str, Any]) -> str | None:
    return _first_optional_text(
        _optional_text(metadata.get(key))
        for key in (
            "parent_id",
            "parent_chunk_id",
            "parent_block_id",
            "block_id",
        )
    )


def _entry_sort_key(entry: _CandidateEntry) -> tuple[int, int, int, int]:
    return (
        _rank_or_last(entry.final_rank),
        _rank_or_last(entry.rerank_rank),
        _rank_or_last(entry.fusion_rank),
        entry.order,
    )


def _evidence_block_sort_key(block: EvidenceBlock, order: int) -> tuple[int, int, int, int, int]:
    return (
        _rank_or_last(block.rank),
        _rank_or_last(block.final_rank),
        _rank_or_last(block.best_rerank_rank),
        _rank_or_last(block.best_fusion_rank),
        order,
    )


def _best_rank(entry: _CandidateEntry) -> int | None:
    return _first_present_int(entry.final_rank, entry.rerank_rank, entry.fusion_rank)


def _rank_or_last(value: int | None) -> int:
    return value if value is not None else 10**9


def _block_retrieval_score(entries: Sequence[_CandidateEntry]) -> float:
    score = _max_optional_float(entry.rerank_score for entry in entries)
    if score is None:
        score = _max_optional_float(entry.fusion_score for entry in entries)
    if score is None:
        score = _max_optional_float(entry.dense_score for entry in entries)
    if score is None:
        score = _max_optional_float(entry.lexical_score for entry in entries)
    return float(score or 0.0)


def _candidate_sources(candidate: Candidate, lexical_backend: str | None) -> tuple[str, ...]:
    retrieved_by = _tuple_text(getattr(candidate, "retrieved_by", ()))
    if retrieved_by:
        return retrieved_by

    sources: list[str] = []
    if (
        getattr(candidate, "dense_rank", None) is not None
        or getattr(candidate, "dense_score", None) is not None
    ):
        sources.append("dense")
    if (
        getattr(candidate, "lexical_rank", None) is not None
        or getattr(candidate, "lexical_score", None) is not None
        or lexical_backend is not None
    ):
        sources.append("lexical")
    if not sources and getattr(candidate, "fusion_score", None) is not None:
        sources.append("fusion")
    return tuple(sources)


def _truncate_text_to_budget(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    if approx_token_count(text) <= token_budget:
        return text

    source_tokens = approx_token_count(text)
    limit = max(1, int(len(text) * token_budget / source_tokens))
    candidate = text[:limit].rstrip()
    while candidate and approx_token_count(candidate) > token_budget:
        limit = max(1, int(limit * 0.8))
        next_candidate = text[:limit].rstrip()
        if next_candidate == candidate:
            break
        candidate = next_candidate
    return candidate


def _merge_metadata(metadata_items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for metadata in metadata_items:
        for key, value in metadata.items():
            if key not in merged:
                merged[key] = value
                continue
            if merged[key] == value:
                continue
            values = merged[key] if isinstance(merged[key], list) else [merged[key]]
            if not any(existing == value for existing in values):
                values.append(value)
            merged[key] = values
    return merged


def _unique(values: Iterable[str | None]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _first_text(values: Iterable[str | None]) -> str:
    for value in values:
        if value:
            return value
    return ""


def _first_optional_text(values: Iterable[str | None]) -> str | None:
    for value in values:
        if value:
            return value
    return None


def _min_optional_int(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _max_optional_int(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _max_optional_float(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _int_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _first_present_int(*values: int | None) -> int | None:
    for value in values:
        if value is not None:
            return value
    return None


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _tuple_text(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    try:
        return _unique(str(item) for item in value if item)
    except TypeError:
        return (str(value),) if value else ()


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dict_value(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}
