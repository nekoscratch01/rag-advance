from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    document_id: str
    chunk_id: str
    text: str
    source_title: str
    source_uri: str | None
    section_title: str | None
    page_start: int | None
    page_end: int | None
    retrieval_score: float
    rank: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    child_ids: tuple[str, ...] = ()
    retrieved_by: tuple[str, ...] = ()
    rerank_score: float | None = None
    rerank_rank: int | None = None

    def __post_init__(self) -> None:
        metadata = self.metadata or {}
        if self.parent_id is None:
            object.__setattr__(
                self,
                "parent_id",
                _optional_text(metadata.get("parent_id") or metadata.get("parent_chunk_id")),
            )
        if not self.child_ids:
            child_ids = _tuple_text(
                metadata.get("child_ids") or metadata.get("chunk_ids") or (self.chunk_id,)
            )
            object.__setattr__(self, "child_ids", child_ids)
        if not self.retrieved_by:
            retrieved_by = _tuple_text(metadata.get("retrieved_by") or metadata.get("sources"))
            object.__setattr__(self, "retrieved_by", retrieved_by)
        if self.rerank_score is None:
            object.__setattr__(self, "rerank_score", _float_or_none(metadata.get("rerank_score")))
        if self.rerank_rank is None:
            object.__setattr__(
                self,
                "rerank_rank",
                _int_or_none(metadata.get("rerank_rank") or metadata.get("best_rerank_rank")),
            )


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
        return tuple(str(item) for item in value if item)
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
