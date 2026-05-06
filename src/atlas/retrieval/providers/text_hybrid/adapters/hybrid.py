from __future__ import annotations

from typing import Protocol

from sqlalchemy.orm import Session

from atlas.db import repositories
from atlas.retrieval.models.candidate import Candidate
from atlas.retrieval.models.evidence import Evidence
from atlas.retrieval.ranking.fusion import DEFAULT_RRF_K, rrf_fuse
from atlas.retrieval.ranking.reranker import Reranker, rerank_with_context


class CandidateRetriever(Protocol):
    def retrieve_candidates(
        self,
        db: Session,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[Candidate]:
        ...


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: CandidateRetriever,
        lexical_retriever: CandidateRetriever,
        *,
        rrf_k: int = DEFAULT_RRF_K,
        rrf_top_k: int = 40,
        reranker: Reranker | None = None,
        reranker_enabled: bool = True,
        reranker_top_k: int = 30,
        reranker_output_k: int | None = 8,
        dense_top_k: int | None = None,
        lexical_top_k: int | None = None,
        max_context_tokens: int | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.lexical_retriever = lexical_retriever
        self.rrf_k = rrf_k
        self.rrf_top_k = rrf_top_k
        self.reranker = reranker
        self.reranker_enabled = reranker_enabled
        self.reranker_top_k = reranker_top_k
        self.reranker_output_k = reranker_output_k
        self.dense_top_k = dense_top_k
        self.lexical_top_k = lexical_top_k
        self.max_context_tokens = max_context_tokens

    def retrieve(
        self,
        db: Session,
        *,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[Evidence]:
        candidates = self.retrieve_candidates(db, query=query, top_k=top_k, filters=filters)
        if self.max_context_tokens is not None:
            from atlas.query_runtime.evidence_builder import build_evidence_from_candidates

            return build_evidence_from_candidates(
                candidates,
                parent_resolver=_parent_resolver(db, candidates),
                max_context_tokens=self.max_context_tokens,
                max_blocks=top_k,
            )
        return [
            _candidate_to_evidence(candidate, index)
            for index, candidate in enumerate(candidates, start=1)
        ]

    def retrieve_candidates(
        self,
        db: Session,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[Candidate]:
        if top_k <= 0:
            return []

        dense_top_k = self.dense_top_k or top_k
        lexical_top_k = self.lexical_top_k or top_k
        dense_candidates = self.dense_retriever.retrieve_candidates(db, query, dense_top_k, filters)
        lexical_candidates = self.lexical_retriever.retrieve_candidates(
            db,
            query,
            lexical_top_k,
            filters,
        )
        if not self.reranker_enabled:
            return rrf_fuse(
                dense_candidates,
                lexical_candidates,
                rrf_k=self.rrf_k,
                limit=top_k,
            )

        fused = rrf_fuse(
            dense_candidates,
            lexical_candidates,
            rrf_k=self.rrf_k,
            limit=self.rrf_top_k,
        )
        if not fused:
            return []
        if self.reranker is None:
            raise RuntimeError(
                "Hybrid reranker is enabled but no reranker is configured. "
                "Wire a local reranker or set ATLAS_RERANKER_ENABLED=false."
            )

        rerank_top_k = min(self.reranker_top_k, len(fused))
        output_limit = (
            min(top_k, self.reranker_output_k)
            if self.reranker_output_k is not None
            else top_k
        )
        reranked = rerank_with_context(
            self.reranker,
            query=query,
            candidates=fused[:rerank_top_k],
            top_k=rerank_top_k,
            output_k=output_limit,
        )
        return reranked[:output_limit]


def _candidate_to_evidence(candidate: Candidate, evidence_index: int) -> Evidence:
    return Evidence(
        evidence_id=f"c{evidence_index}",
        document_id=candidate.document_id,
        chunk_id=candidate.chunk_id,
        text=candidate.text,
        source_title=candidate.source_title,
        source_uri=candidate.source_uri,
        section_title=candidate.section_title,
        page_start=candidate.page_start,
        page_end=candidate.page_end,
        retrieval_score=float(
            candidate.rerank_score
            if candidate.rerank_score is not None
            else candidate.fusion_score or 0.0
        ),
        rank=candidate.final_rank or candidate.fusion_rank or evidence_index,
        token_count=candidate.token_count,
        metadata={
            **candidate.metadata,
            "retrieved_by": list(candidate.retrieved_by),
            "dense_rank": candidate.dense_rank,
            "dense_score": candidate.dense_score,
            "lexical_rank": candidate.lexical_rank,
            "lexical_score": candidate.lexical_score,
            "lexical_backend": candidate.lexical_backend,
            "fusion_rank": candidate.fusion_rank,
            "fusion_score": candidate.fusion_score,
            "rerank_rank": candidate.rerank_rank,
            "rerank_score": candidate.rerank_score,
        },
    )


def _parent_resolver(db: Session, candidates: list[Candidate]):
    parent_ids = []
    for candidate in candidates:
        parent_id = candidate.metadata.get("parent_id") if candidate.metadata else None
        if parent_id:
            parent_ids.append(str(parent_id))
    return repositories.get_parent_blocks_by_ids(db, parent_ids)
