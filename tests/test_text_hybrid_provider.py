from __future__ import annotations

import pytest

from atlas.core.config import Settings
from atlas.llm.base import GeneratedAnswer, LLMUsage
from atlas.query_orchestrator.schema import QueryPlan, RetrievalUnit
from atlas.query_runtime.service import QueryRuntime
from atlas.retrieval.candidate import Candidate
from atlas.retrieval.evidence import Evidence
from atlas.retrieval.providers.text_hybrid import TextHybridProvider
from atlas.retrieval.retrieval_task import tasks_from_plan


class _CandidateRetriever:
    def __init__(
        self,
        source: str,
        *,
        chunk_id: str | None = None,
        parent_id: str | None = None,
    ) -> None:
        self.source = source
        self.chunk_id = chunk_id
        self.parent_id = parent_id
        self.calls: list[dict] = []
        self.evidence_calls: list[dict] = []

    def retrieve_candidates(self, db, query, top_k, filters=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": dict(filters or {})})
        rank = len(self.calls)
        return [
            Candidate(
                chunk_id=self.chunk_id or f"{self.source}_{rank}",
                document_id="doc_1",
                doc_name="3M 2018 10-K",
                source_title="3M 2018 10-K",
                company="3M",
                text=f"3M FY2018 capital expenditures were 1,577 million. {self.source} candidate {rank}",
                page_start=10,
                page_end=10,
                chunk_index=rank,
                token_count=6,
                retrieved_by=(self.source,),
                dense_rank=rank if self.source == "dense" else None,
                dense_score=0.9 if self.source == "dense" else None,
                lexical_rank=rank if self.source == "bm25" else None,
                lexical_score=0.8 if self.source == "bm25" else None,
                lexical_backend="qdrant_bm25" if self.source == "bm25" else None,
                final_rank=rank,
                parent_id=self.parent_id,
                metadata={"parent_id": self.parent_id} if self.parent_id else {},
            )
        ]

    def retrieve(self, db, *, query, top_k, filters=None):
        self.evidence_calls.append({"query": query, "top_k": top_k, "filters": dict(filters or {})})
        return [
            Evidence(
                evidence_id="c1",
                document_id="doc_legacy",
                chunk_id="chk_legacy",
                text="legacy evidence",
                source_title="legacy",
                source_uri=None,
                section_title=None,
                page_start=1,
                page_end=1,
                retrieval_score=1.0,
                rank=1,
                token_count=2,
                metadata={"retrieved_by": [self.source]},
            )
        ]


class _LegacyHybridRetriever(_CandidateRetriever):
    pass


class _StaticOrchestrator:
    def __init__(self, plan: QueryPlan) -> None:
        self._plan = plan

    def plan(self, query, *, use_llm=True):
        return self._plan


class _Generator:
    model_name = "test-generator"

    def generate(self, *, query, evidence):
        return GeneratedAnswer(
            answer="3M FY2018 capital expenditures were 1,577 million [c1].",
            confidence="supported",
            usage=LLMUsage(input_tokens=10, output_tokens=8),
            raw_output="{}",
        )


class _FakeDB:
    def __init__(self) -> None:
        self.added = []
        self.commits = 0

    def add(self, value) -> None:
        self.added.append(value)

    def flush(self) -> None:
        return None

    def commit(self) -> None:
        self.commits += 1


class _EmptyRetriever(TextHybridProvider):
    def __init__(self, pack) -> None:
        self.last_evidence_pack = pack

    def retrieve_with_plan(
        self,
        db,
        *,
        query,
        top_k,
        filters,
        options,
        query_plan,
        retrieval_tasks,
    ):
        return []


def _provider(
    *,
    default_mode: str = "hybrid_rrf",
    reranker_enabled: bool = False,
    shared_chunk_id: str | None = None,
    shared_parent_id: str | None = None,
    max_context_tokens: int | None = None,
) -> tuple[TextHybridProvider, _CandidateRetriever, _CandidateRetriever]:
    dense = _CandidateRetriever(
        "dense",
        chunk_id=shared_chunk_id,
        parent_id=shared_parent_id,
    )
    bm25 = _CandidateRetriever(
        "bm25",
        chunk_id=shared_chunk_id,
        parent_id=shared_parent_id,
    )
    return (
        TextHybridProvider(
            dense_retriever=dense,
            bm25_retriever=bm25,
            hybrid_rrf_retriever=_LegacyHybridRetriever("hybrid_rrf"),
            hybrid_rerank_retriever=_LegacyHybridRetriever("hybrid_rerank"),
            default_mode=default_mode,
            rrf_top_k=10,
            reranker_enabled=reranker_enabled,
            max_context_tokens=max_context_tokens,
        ),
        dense,
        bm25,
    )


def test_text_hybrid_provider_executes_v1_lanes_from_retrieval_tasks() -> None:
    provider, dense, bm25 = _provider()
    plan = QueryPlan(
        plan_id="plan_1",
        original_query="What was 3M FY2018 capex?",
        metadata_filter={"tenant": "test"},
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="3M FY2018 capex",
                retrievers=("hybrid",),
                metadata_filter={"document_ids": ["doc_1"]},
                should_terms=("capital expenditures", "purchases of property"),
                top_k=3,
                metadata={"internal_lanes": ["dense", "bm25", "metric_alias", "section", "table"]},
            ),
        ),
        planner="test",
    )

    evidence = provider.retrieve_with_plan(
        object(),
        query=plan.original_query,
        top_k=5,
        filters={"runtime": "direct"},
        options={"retrieval_mode": "hybrid_rrf"},
        query_plan=plan,
        retrieval_tasks=tasks_from_plan(plan),
    )

    assert evidence
    assert len(dense.calls) == 1
    assert len(bm25.calls) == 4
    assert bm25.calls[0]["query"].endswith("capital expenditures purchases of property")
    assert bm25.calls[1]["query"].endswith("capital expenditures purchases of property")
    assert "table row page" in bm25.calls[-1]["query"]
    assert bm25.calls[0]["filters"] == {
        "runtime": "direct",
        "tenant": "test",
        "document_ids": ["doc_1"],
    }
    assert evidence[0].metadata["provider"] == "text_hybrid"
    assert evidence[0].metadata["metadata_filter"] == {
        "tenant": "test",
        "document_ids": ["doc_1"],
    }
    assert evidence[0].metadata["internal_lanes"] == [
        "dense",
        "bm25",
        "metric_alias",
        "section",
        "table",
    ]
    assert evidence[0].metadata["text_hybrid_provider"]["query_plan_id"] == "plan_1"
    assert evidence[0].metadata["text_hybrid_provider"]["fusion"]["backend"] == "weighted_rrf"
    assert evidence[0].metadata["retrieval_unit_id"] == "u0"
    assert evidence[0].metadata["fusion_score"] is not None


def test_text_hybrid_provider_skips_unsupported_provider_tasks_without_fake_evidence() -> None:
    provider, dense, bm25 = _provider()
    plan = QueryPlan(
        plan_id="plan_1",
        original_query="What was 3M FY2018 capex?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="sql_0",
                purpose="structured_lookup",
                text="3M FY2018 capex",
                retrievers=("sql",),
            ),
        ),
        planner="test",
    )

    evidence = provider.retrieve_with_plan(
        object(),
        query=plan.original_query,
        top_k=5,
        filters={},
        options={"retrieval_mode": "hybrid_rrf"},
        query_plan=plan,
        retrieval_tasks=tasks_from_plan(plan),
    )

    assert evidence == []
    assert dense.calls == []
    assert bm25.calls == []
    assert provider.last_retrieval_trace is not None
    assert provider.last_retrieval_trace["provider_status"] == "skipped"
    assert provider.last_retrieval_trace["tasks"][0]["provider"] == "sql"
    assert provider.last_retrieval_trace["tasks"][0]["unsupported_reason"]


def test_text_hybrid_provider_keeps_legacy_modes_available() -> None:
    provider, dense, bm25 = _provider()

    evidence = provider.retrieve_with_options(
        object(),
        query="plain dense query",
        top_k=1,
        filters={},
        options={"retrieval_mode": "dense_only"},
    )

    assert evidence[0].metadata["provider"] == "text_hybrid"
    assert evidence[0].metadata["provider_path"] == "legacy_mode_switch"
    assert len(dense.evidence_calls) == 1
    assert len(bm25.calls) == 0
    assert len(bm25.evidence_calls) == 0

    provider.retrieve_with_options(
        object(),
        query="plain bm25 query",
        top_k=1,
        filters={},
        options={"retrieval_mode": "bm25_only"},
    )
    provider.retrieve_with_options(
        object(),
        query="plain hybrid query",
        top_k=1,
        filters={},
        options={"retrieval_mode": "hybrid_rrf"},
    )

    assert len(bm25.evidence_calls) == 1
    assert len(provider.mode_switcher.hybrid_rrf_retriever.evidence_calls) == 1


def test_plan_aware_provider_respects_dense_only_mode() -> None:
    provider, dense, bm25 = _provider(default_mode="dense")
    plan = QueryPlan(
        plan_id="plan_1",
        original_query="What was 3M FY2018 capex?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="3M FY2018 capex",
                retrievers=("hybrid",),
                metadata={"internal_lanes": ["dense", "bm25", "table"]},
            ),
        ),
    )

    evidence = provider.retrieve_with_plan(
        object(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=tasks_from_plan(plan),
    )

    assert evidence
    assert len(dense.calls) == 1
    assert len(bm25.calls) == 0
    assert evidence[0].metadata["retrieved_by"] == ["dense"]


def test_plan_aware_provider_preserves_multi_lane_attribution_after_fusion() -> None:
    provider, dense, bm25 = _provider(shared_chunk_id="shared_chunk")
    plan = QueryPlan(
        plan_id="plan_1",
        original_query="What was 3M FY2018 capex?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="3M FY2018 capex",
                retrievers=("hybrid",),
                metadata={"internal_lanes": ["dense", "bm25"]},
            ),
        ),
    )

    evidence = provider.retrieve_with_plan(
        object(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={"retrieval_mode": "hybrid_rrf"},
        query_plan=plan,
        retrieval_tasks=tasks_from_plan(plan),
    )

    assert len(dense.calls) == 1
    assert len(bm25.calls) == 1
    assert evidence[0].chunk_id == "shared_chunk"
    assert evidence[0].metadata["lane"] == "multi_lane"
    assert evidence[0].metadata["lanes"] == ["dense", "bm25"]
    assert evidence[0].metadata["lane_trace"]["lane"] == "multi_lane"
    assert {
        item["lane"] for item in evidence[0].metadata["lane_attributions"]
    } == {"dense", "bm25"}
    assert evidence[0].metadata["fusion"]["strategy"] == "weighted_rrf"
    assert {
        item["lane"] for item in evidence[0].metadata["fusion"]["lane_contributions"]
    } == {"dense", "bm25"}


def test_plan_aware_provider_uses_lane_weights_in_weighted_rrf() -> None:
    provider, dense, bm25 = _provider()
    plan = QueryPlan(
        plan_id="plan_1",
        original_query="What was 3M FY2018 capex?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="3M FY2018 capex",
                retrievers=("hybrid",),
                lane_weights={"table": 4.0},
                metadata={"internal_lanes": ["dense", "table"]},
            ),
        ),
    )

    evidence = provider.retrieve_with_plan(
        object(),
        query=plan.original_query,
        top_k=2,
        filters={},
        options={"retrieval_mode": "hybrid_rrf"},
        query_plan=plan,
        retrieval_tasks=tasks_from_plan(plan),
    )

    assert len(dense.calls) == 1
    assert len(bm25.calls) == 1
    assert evidence[0].metadata["lanes"] == ["table"]
    assert evidence[0].metadata["weighted_contribution"] > evidence[1].metadata["weighted_contribution"]


def test_parent_evidence_preserves_canonical_weighted_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "atlas.retrieval.providers.text_hybrid.provider.repositories.get_parent_blocks_by_ids",
        lambda db, parent_ids: {},
    )
    provider, dense, bm25 = _provider(
        shared_parent_id="parent_1",
        max_context_tokens=1000,
    )
    plan = QueryPlan(
        plan_id="plan_1",
        original_query="What was 3M FY2018 capex?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="3M FY2018 capex",
                retrievers=("hybrid",),
                lane_weights={"table": 3.0},
                metadata={"internal_lanes": ["dense", "bm25", "table"]},
            ),
        ),
    )

    evidence = provider.retrieve_with_plan(
        object(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={"retrieval_mode": "hybrid_rrf"},
        query_plan=plan,
        retrieval_tasks=tasks_from_plan(plan),
    )

    assert len(evidence) == 1
    metadata = evidence[0].metadata
    assert len(dense.calls) == 1
    assert len(bm25.calls) == 2
    assert isinstance(metadata["weighted_contribution"], float)
    assert isinstance(metadata["fusion"], dict)
    assert metadata["evidence_pack"]["block_count"] == 1
    assert metadata["evidence_pack"]["dropped_block_count"] == 0
    assert isinstance(metadata["lane_contributions"], list)
    assert all(isinstance(item, dict) for item in metadata["lane_contributions"])
    assert {
        item["lane"] for item in metadata["lane_contributions"]
    } == {"table"}
    assert sum(
        item["weighted_contribution"] for item in metadata["lane_contributions"]
    ) == pytest.approx(metadata["weighted_contribution"])
    assert {
        item["lane"] for item in metadata["parent_child_contributions"]
    } == {"dense", "bm25", "table"}
    assert metadata["parent_lanes"] == ["table", "dense", "bm25"]
    assert metadata["lane_trace"]["lane"] == "table"


def test_query_runtime_uses_text_hybrid_provider_and_persists_plan_trace() -> None:
    provider, dense, bm25 = _provider()
    plan = QueryPlan(
        plan_id="plan_1",
        original_query="What was 3M FY2018 capex?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="3M FY2018 capex",
                retrievers=("hybrid",),
                metadata={"internal_lanes": ["dense", "bm25", "table"]},
            ),
        ),
    )
    runtime = QueryRuntime(
        settings=Settings(
            openai_api_key=None,
            cache_enabled=False,
            retrieval_mode="hybrid_rrf",
            reranker_enabled=False,
            max_context_tokens=6000,
        ),
        retriever=provider,
        generator=_Generator(),
        orchestrator=_StaticOrchestrator(plan),
    )
    db = _FakeDB()

    result = runtime.run(
        db,
        query=plan.original_query,
        top_k=3,
        filters={},
        options={"retrieval_mode": "hybrid_rrf"},
    )

    query_runs = [item for item in db.added if item.__class__.__name__ == "QueryRun"]
    assert result.details["query_plan"]["plan_id"] == "plan_1"
    assert len(dense.calls) == 1
    assert len(bm25.calls) == 2
    assert db.commits == 1
    assert query_runs
    assert query_runs[0].details_json["trace"]["metadata"]["query_plan"]["plan_id"] == "plan_1"
    assert result.details["retrieval_trace"]["evidence_count"] == 3
    assert {
        lane
        for item in result.details["retrieval_trace"]["top_k"]
        for lane in item["lanes"]
    } == {"dense", "bm25", "table"}
    assert all(
        item["lane_contributions"]
        for item in result.details["retrieval_trace"]["top_k"]
    )


def test_query_runtime_exposes_empty_evidence_pack_drop_reasons() -> None:
    from atlas.query_runtime.evidence_builder import build_evidence_pack_from_candidates

    plan = QueryPlan(
        plan_id="plan_1",
        original_query="What was 3M FY2018 capex?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="3M FY2018 capex",
            ),
        ),
    )
    pack = build_evidence_pack_from_candidates(
        (
            Candidate(
                chunk_id="chk_1",
                document_id="doc_1",
                doc_name="doc",
                source_title="doc",
                company="3M",
                text="3M FY2018 capex was 1,577.",
                page_start=1,
                page_end=1,
                chunk_index=1,
                token_count=10,
                retrieved_by=("dense",),
                dense_rank=1,
                dense_score=0.9,
            ),
        ),
        max_context_tokens=0,
        plan_id="plan_1",
    )
    runtime = QueryRuntime(
        settings=Settings(openai_api_key=None, cache_enabled=False),
        retriever=_EmptyRetriever(pack),
        generator=_Generator(),
        orchestrator=_StaticOrchestrator(plan),
    )

    result = runtime.run(
        _FakeDB(),
        query=plan.original_query,
        top_k=1,
        filters={},
        options={"retrieval_mode": "hybrid_rrf"},
    )

    assert result.confidence == "insufficient"
    assert result.details["evidence_pack"]["dropped_block_count"] == 1
    assert result.details["evidence_pack"]["dropped_blocks"][0]["drop_reason"] == "token_budget"


def test_text_hybrid_provider_skips_unsupported_future_provider_tasks() -> None:
    provider, dense, bm25 = _provider()
    plan = QueryPlan(
        plan_id="plan_future",
        original_query="Who supplies Apple Vision Pro displays?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_graph",
                purpose="supply_chain_discovery",
                text="Apple Vision Pro display suppliers",
                retrievers=("graph",),
            ),
        ),
    )

    evidence = provider.retrieve_with_plan(
        object(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={"retrieval_mode": "hybrid_rrf"},
        query_plan=plan,
        retrieval_tasks=tasks_from_plan(plan),
    )

    assert evidence == []
    assert dense.calls == []
    assert bm25.calls == []


def test_legacy_retrieve_options_clears_stale_evidence_pack() -> None:
    from atlas.query_runtime.evidence_builder import build_evidence_pack_from_candidates

    provider, _, _ = _provider()
    provider.last_evidence_pack = build_evidence_pack_from_candidates((), max_context_tokens=0)

    provider.retrieve_with_options(
        object(),
        query="plain dense query",
        top_k=1,
        filters={},
        options={"retrieval_mode": "dense_only"},
    )

    assert provider.last_evidence_pack is None
