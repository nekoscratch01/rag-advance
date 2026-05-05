from __future__ import annotations

from dataclasses import replace
import time
from typing import Any, Protocol, Sequence

from atlas.retrieval.candidate import Candidate


class Reranker(Protocol):
    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[Candidate],
        top_k: int,
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
    ) -> list[Candidate]:
        if top_k <= 0:
            return []
        selected = list(candidates[:top_k])
        if not selected:
            return []

        started = time.perf_counter()
        model = self._load_model()
        pairs = [(query, candidate.text) for candidate in selected]
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
) -> dict[str, Any]:
    metadata = dict(candidate.metadata)
    fusion = metadata.get("fusion")
    if isinstance(fusion, dict):
        metadata["fusion"] = {**fusion, "final_rank": rank}
    metadata["reranker"] = {
        "enabled": True,
        "model": model_name,
        "rank": rank,
        "score": score,
        "input_rank": candidate.final_rank or candidate.fusion_rank,
        "latency_ms": latency_ms,
        "candidates_scored": candidates_scored,
    }
    return metadata
