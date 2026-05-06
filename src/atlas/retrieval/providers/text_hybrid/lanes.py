from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.orm import Session

from atlas.retrieval.candidate import Candidate
from atlas.retrieval.retrieval_task import RetrievalTask


TEXT_HYBRID_PROVIDER = "text_hybrid"
SUPPORTED_LANES = frozenset({"dense", "bm25", "metric_alias", "section", "table"})


class CandidateRetriever(Protocol):
    def retrieve_candidates(
        self,
        db: Session,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[Candidate]:
        ...


@dataclass(frozen=True)
class TextHybridLane:
    name: str
    family: str
    retriever: CandidateRetriever

    def retrieve(
        self,
        db: Session,
        *,
        task: RetrievalTask,
        query_text: str,
        top_k: int,
        filters: dict,
    ) -> list[Candidate]:
        candidates = self.retriever.retrieve_candidates(
            db,
            query_text,
            top_k,
            filters,
        )
        return [
            annotate_candidate(
                candidate,
                lane=self.name,
                family=self.family,
                task=task,
                rank=index,
            )
            for index, candidate in enumerate(candidates, start=1)
        ]


def annotate_candidate(
    candidate: Candidate,
    *,
    lane: str,
    family: str,
    task: RetrievalTask,
    rank: int,
) -> Candidate:
    from dataclasses import replace

    lane_score = _lane_score(candidate, family)
    lane_weight = float(task.lane_weights.get(lane, 1.0))
    metadata = dict(candidate.metadata)
    lane_payload = {
        "provider": TEXT_HYBRID_PROVIDER,
        "lane": lane,
        "lane_family": family,
        "lane_rank": rank,
        "lane_score": lane_score,
        "lane_weight": lane_weight,
        "retrieval_task_id": task.task_id,
        "retrieval_unit_id": task.unit_id,
        "retrieval_unit_weight": task.weight,
        "query_text": task.query_text,
        "must_have_terms": list(task.must_have_terms),
        "should_terms": list(task.should_terms),
    }
    metadata["provider"] = TEXT_HYBRID_PROVIDER
    metadata["lane"] = lane
    metadata["lane_family"] = family
    metadata["lane_trace"] = lane_payload
    metadata.setdefault("retrieval_tasks", [])
    if isinstance(metadata["retrieval_tasks"], list):
        metadata["retrieval_tasks"].append(lane_payload)

    retrieved_by = _retrieved_by(candidate.retrieved_by, lane, family)
    dense_rank = candidate.dense_rank
    lexical_rank = candidate.lexical_rank
    if family == "dense" and dense_rank is None:
        dense_rank = rank
    if family == "lexical" and lexical_rank is None:
        lexical_rank = rank

    return replace(
        candidate,
        provider=TEXT_HYBRID_PROVIDER,
        lane=lane,
        lane_rank=rank,
        lane_score=lane_score,
        lane_weight=lane_weight,
        retrieval_task_id=task.task_id,
        retrieval_unit_id=task.unit_id,
        unit_weight=float(task.weight),
        dense_rank=dense_rank,
        lexical_rank=lexical_rank,
        retrieved_by=retrieved_by,
        metadata=metadata,
    )


def lane_query_text(task: RetrievalTask, lane: str) -> str:
    if lane == "metric_alias":
        return _join_unique([task.query_text, *task.should_terms])
    if lane == "section":
        sections = _metadata_terms(task, "section_terms", "sections")
        return _join_unique([task.query_text, *sections, *task.must_have_terms])
    if lane == "table":
        table_terms = _metadata_terms(task, "table_terms", "table_headers")
        return _join_unique([task.query_text, "table row page", *table_terms])
    return task.query_text


def lane_filters(base_filters: dict | None, task: RetrievalTask) -> dict:
    filters = dict(base_filters or {})
    filters.update(task.filters)
    return filters


def _lane_score(candidate: Candidate, family: str) -> float | None:
    if family == "dense":
        return candidate.dense_score
    return candidate.lexical_score


def _retrieved_by(existing: tuple[str, ...], lane: str, family: str) -> tuple[str, ...]:
    values = [*existing]
    if family == "lexical" and "bm25" not in values:
        values.append("bm25")
    if family == "dense" and "dense" not in values:
        values.append("dense")
    if lane not in values:
        values.append(lane)
    ordered: list[str] = []
    for source in ("dense", "bm25", "metric_alias", "section", "table"):
        if source in values:
            ordered.append(source)
    for source in values:
        if source not in ordered:
            ordered.append(source)
    return tuple(ordered)


def _metadata_terms(task: RetrievalTask, *keys: str) -> list[str]:
    terms: list[str] = []
    for key in keys:
        value = task.metadata.get(key)
        if isinstance(value, str):
            terms.append(value)
        elif isinstance(value, list | tuple | set):
            terms.extend(str(item) for item in value if item)
    return terms


def _join_unique(values: list[str]) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for value in values:
        text = " ".join(str(value).split())
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        parts.append(text)
    return " ".join(parts)

