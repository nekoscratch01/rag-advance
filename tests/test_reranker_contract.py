from __future__ import annotations

from dataclasses import replace

from atlas.query_orchestrator.schema import Entity, Metric, Period, QueryPlan, RetrievalUnit
from atlas.retrieval.candidate import Candidate
from atlas.retrieval.providers.text_hybrid import TextHybridProvider
from atlas.retrieval.reranker import CrossEncoderReranker, rerank_with_context
from atlas.retrieval.retrieval_task import tasks_from_plan


class _FakeCrossEncoder:
    def __init__(self) -> None:
        self.pairs = []

    def predict(self, pairs, batch_size):
        self.pairs = list(pairs)
        return [0.1, 0.9]


class _CandidateRetriever:
    def __init__(self, source: str) -> None:
        self.source = source

    def retrieve_candidates(self, db, query, top_k, filters=None):
        return [
            Candidate(
                chunk_id=f"{self.source}_{rank}",
                document_id="doc_1",
                doc_name="3M 2018 10-K",
                source_title="3M 2018 10-K",
                company="3M",
                text=f"{self.source} candidate {rank} capital expenditures 1,577",
                page_start=10,
                page_end=10,
                chunk_index=rank,
                token_count=8,
                retrieved_by=(self.source,),
                dense_rank=rank if self.source == "dense" else None,
                dense_score=0.9 if self.source == "dense" else None,
                lexical_rank=rank if self.source != "dense" else None,
                lexical_score=0.8 if self.source != "dense" else None,
                lexical_backend="qdrant_bm25" if self.source != "dense" else None,
                final_rank=rank,
            )
            for rank in range(1, min(top_k, 2) + 1)
        ]

    def retrieve(self, db, *, query, top_k, filters=None):
        return []


class _RecordingReranker:
    model_name = "recording-reranker"

    def __init__(self) -> None:
        self.query_plan = None
        self.retrieval_tasks = None
        self.output_k = None

    def rerank(
        self,
        *,
        query,
        candidates,
        top_k,
        query_plan=None,
        retrieval_tasks=None,
        output_k=None,
    ):
        self.query_plan = query_plan
        self.retrieval_tasks = list(retrieval_tasks or [])
        self.output_k = output_k
        ranked = []
        for rank, candidate in enumerate(candidates[:top_k], start=1):
            metadata = dict(candidate.metadata)
            metadata["reranker"] = {
                "model": self.model_name,
                "input_rank": candidate.final_rank,
                "output_rank": rank,
                "score": float(10 - rank),
                "latency_ms": 0,
                "top_n": top_k,
                "top_m": output_k,
                "query_plan_id": query_plan.plan_id if query_plan else None,
                "retrieval_unit_id": candidate.retrieval_unit_id,
            }
            ranked.append(
                replace(
                    candidate,
                    rerank_rank=rank,
                    rerank_score=float(10 - rank),
                    final_rank=rank,
                    metadata=metadata,
                )
            )
        return ranked


class _LegacySignatureReranker:
    def rerank(self, *, query, candidates, top_k):
        return list(candidates[:top_k])


def _plan() -> QueryPlan:
    return QueryPlan(
        plan_id="plan_1",
        original_query="What was 3M FY2018 capex?",
        query_type="financial_numeric_fact",
        entities=(Entity(value="3M"),),
        periods=(Period(value="FY2018", normalized="2018"),),
        metrics=(Metric(canonical_name="capital_expenditure", aliases=("capex",)),),
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="metric",
                text="3M FY2018 capital expenditures",
                retrievers=("dense", "bm25"),
                must_have_terms=("3M", "2018"),
                should_terms=("capital expenditures",),
            ),
        ),
    )


def test_cross_encoder_reranker_builds_plan_aware_inputs_and_trace() -> None:
    plan = _plan()
    task = tasks_from_plan(plan)[0]
    candidates = (
        _candidate("chk_1", retrieval_unit_id=task.unit_id, fusion_rank=1),
        _candidate("chk_2", retrieval_unit_id=task.unit_id, fusion_rank=2),
    )
    reranker = CrossEncoderReranker("fake-cross-encoder")
    model = _FakeCrossEncoder()
    reranker._model = model

    reranked = reranker.rerank(
        query=plan.original_query,
        candidates=candidates,
        top_k=2,
        query_plan=plan,
        retrieval_tasks=(task,),
        output_k=1,
    )

    assert [candidate.chunk_id for candidate in reranked] == ["chk_2", "chk_1"]
    assert "Entities: 3M" in model.pairs[0][0]
    assert "Metrics: capital_expenditure" in model.pairs[0][0]
    assert "Retrieval unit: 3M FY2018 capital expenditures" in model.pairs[0][0]
    assert "Must include: 3M, 2018" in model.pairs[0][0]
    assert reranked[0].metadata["reranker"]["input_rank"] == 2
    assert reranked[0].metadata["reranker"]["output_rank"] == 1
    assert reranked[0].metadata["reranker"]["score"] == 0.9
    assert reranked[0].metadata["reranker"]["model"] == "fake-cross-encoder"
    assert reranked[0].metadata["reranker"]["top_n"] == 2
    assert reranked[0].metadata["reranker"]["top_m"] == 1
    assert reranked[0].metadata["reranker_input"]["query_plan_id"] == "plan_1"
    assert reranked[0].metadata["reranker_input"]["retrieval_unit_id"] == "u0"


def test_text_hybrid_provider_passes_plan_and_tasks_to_reranker() -> None:
    plan = _plan()
    tasks = tasks_from_plan(plan)
    reranker = _RecordingReranker()
    provider = TextHybridProvider(
        dense_retriever=_CandidateRetriever("dense"),
        bm25_retriever=_CandidateRetriever("bm25"),
        hybrid_rrf_retriever=_CandidateRetriever("hybrid_rrf"),
        hybrid_rerank_retriever=_CandidateRetriever("hybrid_rerank"),
        default_mode="hybrid",
        reranker=reranker,
        reranker_enabled=True,
        reranker_top_k=4,
        reranker_output_k=1,
        max_context_tokens=None,
    )

    evidence = provider.retrieve_with_plan(
        object(),
        query=plan.original_query,
        top_k=2,
        filters={},
        options={"retrieval_mode": "hybrid"},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert reranker.query_plan == plan
    assert reranker.retrieval_tasks == tasks
    assert reranker.output_k == 1
    assert len(evidence) == 1
    assert evidence[0].metadata["reranker"]["query_plan_id"] == "plan_1"
    assert evidence[0].metadata["reranker"]["top_n"] == 4
    assert evidence[0].metadata["reranker"]["top_m"] == 1


def test_rerank_with_context_preserves_legacy_reranker_signature() -> None:
    candidates = (
        _candidate("chk_1", retrieval_unit_id="u0", fusion_rank=1),
        _candidate("chk_2", retrieval_unit_id="u0", fusion_rank=2),
    )

    reranked = rerank_with_context(
        _LegacySignatureReranker(),
        query="query",
        candidates=candidates,
        top_k=1,
        query_plan=_plan(),
        retrieval_tasks=tasks_from_plan(_plan()),
        output_k=1,
    )

    assert [candidate.chunk_id for candidate in reranked] == ["chk_1"]


def _candidate(
    chunk_id: str,
    *,
    retrieval_unit_id: str,
    fusion_rank: int,
) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        document_id="doc_1",
        doc_name="doc",
        source_title="doc",
        company=None,
        text=f"{chunk_id} candidate text",
        page_start=1,
        page_end=1,
        chunk_index=fusion_rank,
        token_count=4,
        retrieved_by=("dense",),
        dense_rank=fusion_rank,
        dense_score=1.0 / fusion_rank,
        fusion_rank=fusion_rank,
        fusion_score=1.0 / fusion_rank,
        final_rank=fusion_rank,
        retrieval_unit_id=retrieval_unit_id,
    )
