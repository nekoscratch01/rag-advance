from __future__ import annotations

from dataclasses import replace
import inspect
import time
from typing import Any, Protocol, Sequence

from atlas.query_orchestrator.schema import QueryPlan
from atlas.retrieval.models.candidate import Candidate
from atlas.retrieval.models.retrieval_task import RetrievalTask


class Reranker(Protocol):
    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[Candidate],
        top_k: int,
        query_plan: QueryPlan | None = None,
        retrieval_tasks: Sequence[RetrievalTask] | None = None,
        output_k: int | None = None,
    ) -> list[Candidate]:
        ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any | None = None

    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[Candidate],
        top_k: int,
        query_plan: QueryPlan | None = None,
        retrieval_tasks: Sequence[RetrievalTask] | None = None,
        output_k: int | None = None,
    ) -> list[Candidate]:
        if top_k <= 0:
            return []
        selected = list(candidates[:top_k])
        if not selected:
            return []

        started = time.perf_counter()
        model = self._load_model()
        task_index = _task_index(retrieval_tasks or ())
        pairs = [
            (
                _reranker_query(
                    query=query,
                    query_plan=query_plan,
                    candidate=candidate,
                    task_index=task_index,
                ),
                candidate.text,
            )
            for candidate in selected
        ]
        raw_scores = model.predict(pairs, batch_size=self.batch_size)
        latency_ms = int((time.perf_counter() - started) * 1000)
        scores = [float(score) for score in raw_scores]

        scored = list(zip(selected, scores, strict=True))
        scored.sort(key=_rerank_sort_key)

        ranked: list[Candidate] = []
        for rank, (candidate, score) in enumerate(scored, start=1):
            ranked.append(
                replace(
                    candidate,
                    rerank_rank=rank,
                    rerank_score=score,
                    final_rank=rank,
                    metadata=_metadata_with_reranker(
                        candidate,
                        rank=rank,
                        score=score,
                        model_name=self.model_name,
                        latency_ms=latency_ms,
                        candidates_scored=len(selected),
                        output_k=output_k,
                        query_plan=query_plan,
                        retrieval_task=_candidate_task(candidate, task_index),
                        fallback_query=query,
                    ),
                )
            )
        return ranked

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Local reranker requires the 'sentence-transformers' package. "
                "Install project dependencies or set ATLAS_RERANKER_ENABLED=false "
                "to use hybrid RRF without reranking."
            ) from exc

        try:
            self._model = CrossEncoder(self.model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load local reranker model '{self.model_name}'. "
                "Ensure the model is available locally or via the Hugging Face hub, "
                "or set ATLAS_RERANKER_ENABLED=false to use hybrid RRF."
            ) from exc
        return self._model


def rerank_with_context(
    reranker: Reranker,
    *,
    query: str,
    candidates: Sequence[Candidate],
    top_k: int,
    query_plan: QueryPlan | None = None,
    retrieval_tasks: Sequence[RetrievalTask] | None = None,
    output_k: int | None = None,
) -> list[Candidate]:
    kwargs = {
        "query": query,
        "candidates": candidates,
        "top_k": top_k,
        "query_plan": query_plan,
        "retrieval_tasks": retrieval_tasks,
        "output_k": output_k,
    }
    rerank = reranker.rerank
    signature = inspect.signature(rerank)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return rerank(**kwargs)
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return rerank(**{key: value for key, value in kwargs.items() if key in accepted})


def _rerank_sort_key(item: tuple[Candidate, float]) -> tuple[float, int, int, str]:
    candidate, score = item
    return (
        -score,
        candidate.fusion_rank or 1_000_000_000,
        candidate.final_rank or 1_000_000_000,
        candidate.chunk_id,
    )


def _metadata_with_reranker(
    candidate: Candidate,
    *,
    rank: int,
    score: float,
    model_name: str,
    latency_ms: int,
    candidates_scored: int,
    output_k: int | None,
    query_plan: QueryPlan | None,
    retrieval_task: RetrievalTask | None,
    fallback_query: str | None,
) -> dict[str, Any]:
    metadata = dict(candidate.metadata)
    fusion = metadata.get("fusion")
    if isinstance(fusion, dict):
        metadata["fusion"] = {**fusion, "final_rank": rank}
    input_rank = candidate.final_rank or candidate.fusion_rank
    metadata["reranker"] = {
        "enabled": True,
        "model": model_name,
        "rank": rank,
        "output_rank": rank,
        "score": score,
        "input_rank": input_rank,
        "input_fusion_rank": candidate.fusion_rank,
        "latency_ms": latency_ms,
        "candidates_scored": candidates_scored,
        "top_n": candidates_scored,
        "top_m": output_k,
        "query_plan_id": query_plan.plan_id if query_plan else None,
        "retrieval_task_id": retrieval_task.task_id if retrieval_task else candidate.retrieval_task_id,
        "retrieval_unit_id": retrieval_task.unit_id if retrieval_task else candidate.retrieval_unit_id,
    }
    metadata["reranker_input"] = _reranker_input_trace(
        candidate=candidate,
        query_plan=query_plan,
        retrieval_task=retrieval_task,
        fallback_query=fallback_query,
        input_rank=input_rank,
    )
    return metadata


def _reranker_query(
    *,
    query: str,
    query_plan: QueryPlan | None,
    candidate: Candidate,
    task_index: dict[str, RetrievalTask],
) -> str:
    retrieval_task = _candidate_task(candidate, task_index)
    parts = [query]
    if query_plan is not None:
        if query_plan.standalone_query and query_plan.standalone_query != query:
            parts.append(query_plan.standalone_query)
        if query_plan.entities:
            parts.append("Entities: " + ", ".join(entity.value for entity in query_plan.entities))
        if query_plan.periods:
            parts.append(
                "Periods: "
                + ", ".join(period.normalized or period.value for period in query_plan.periods)
            )
        if query_plan.metrics:
            parts.append(
                "Metrics: "
                + ", ".join(metric.canonical_name for metric in query_plan.metrics)
            )
    if retrieval_task is not None:
        parts.append(f"Retrieval unit: {retrieval_task.query_text}")
        if retrieval_task.must_have_terms:
            parts.append("Must include: " + ", ".join(retrieval_task.must_have_terms))
        if retrieval_task.should_terms:
            parts.append("Should include: " + ", ".join(retrieval_task.should_terms))
    return "\n".join(part for part in parts if part)


def _reranker_input_trace(
    *,
    candidate: Candidate,
    query_plan: QueryPlan | None,
    retrieval_task: RetrievalTask | None,
    fallback_query: str | None,
    input_rank: int | None,
) -> dict[str, Any]:
    return {
        "query_plan_id": query_plan.plan_id if query_plan else None,
        "query_type": query_plan.query_type if query_plan else None,
        "retrieval_task_id": retrieval_task.task_id if retrieval_task else candidate.retrieval_task_id,
        "retrieval_unit_id": retrieval_task.unit_id if retrieval_task else candidate.retrieval_unit_id,
        "query_text": retrieval_task.query_text if retrieval_task else fallback_query,
        "must_have_terms": list(retrieval_task.must_have_terms) if retrieval_task else [],
        "should_terms": list(retrieval_task.should_terms) if retrieval_task else [],
        "candidate_id": candidate.candidate_id,
        "chunk_id": candidate.chunk_id,
        "input_rank": input_rank,
        "candidate_text_chars": len(candidate.text),
    }


def _task_index(tasks: Sequence[RetrievalTask]) -> dict[str, RetrievalTask]:
    index: dict[str, RetrievalTask] = {}
    for task in tasks:
        index[task.task_id] = task
        index[task.unit_id] = task
    return index


def _candidate_task(candidate: Candidate, task_index: dict[str, RetrievalTask]) -> RetrievalTask | None:
    for key in (candidate.retrieval_task_id, candidate.retrieval_unit_id):
        if key and key in task_index:
            return task_index[key]
    return None
