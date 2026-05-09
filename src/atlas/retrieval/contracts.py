from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ProviderStatus = Literal["executed", "empty", "skipped_non_executable", "failed"]


@dataclass(frozen=True)
class SourceAnchor:
    """Provider-neutral provenance pointer.

    V1 fills text/page/chunk anchors. Cell and graph IDs are future-facing fields and
    must not be interpreted as V1 cell provenance or GraphRAG support.
    """

    document_id: str | None
    chunk_id: str | None = None
    parent_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    text_span: str | None = None
    table_id: str | None = None
    cell_ids: tuple[str, ...] = ()
    graph_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    task_id: str | None
    unit_id: str | None
    status: ProviderStatus
    candidates: tuple[Any, ...] = ()
    latency_ms: int = 0
    reason: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[Any, ...] = ()
    evidence_pack: Any | None = None


@dataclass(frozen=True)
class ProviderRouterResult:
    evidence: tuple[Any, ...]
    provider_results: tuple[ProviderResult, ...]
    trace: dict[str, Any]
    evidence_pack: Any | None = None


def source_anchor_from_candidate(candidate: Any) -> SourceAnchor:
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    return SourceAnchor(
        document_id=getattr(candidate, "document_id", None),
        chunk_id=getattr(candidate, "chunk_id", None),
        parent_id=getattr(candidate, "parent_id", None),
        page_start=getattr(candidate, "page_start", None),
        page_end=getattr(candidate, "page_end", None),
        table_id=metadata.get("table_id"),
        metadata={
            key: value
            for key, value in metadata.items()
            if key in {"source_type", "section_name", "section_title", "provider", "lane"}
        },
    )
