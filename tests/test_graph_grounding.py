from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from atlas.llm.prompts import build_answer_input
from atlas.query_orchestrator.schema import Entity, QueryPlan, RetrievalUnit
from atlas.retrieval.contracts import SourceAnchor
from atlas.retrieval.models.retrieval_task import RetrievalTask, tasks_from_plan
from atlas.retrieval.providers.graph.models import (
    DEFAULT_DEGREE_CAP,
    GraphEntity,
    GraphEntityMatch,
    GraphFilters,
    GraphNeighborhood,
    GraphPath,
    GraphRelationship,
)
from atlas.retrieval.providers.graph.provider import GraphProvider
from atlas.retrieval.router import serialize_provider_result


GRAPH_VERSION = "test_graph_grounding"
CHUNK_TEXT = "Hydrated child chunk text from source storage."
PARENT_TEXT = "Parent block text must not be expanded into graph evidence."
TOXIC = "DO_NOT_USE_AS_EVIDENCE"


class _FakeGraphStore:
    def __init__(
        self,
        *,
        entities: tuple[GraphEntity, ...],
        matches_by_query: dict[str, tuple[str, ...]],
        paths: tuple[GraphPath, ...],
    ) -> None:
        self.entities = {entity.entity_id: entity for entity in entities}
        self.matches_by_query = matches_by_query
        self.paths = paths

    def get_entity(self, db, entity_id: str, *, graph_version: str) -> GraphEntity | None:
        return self.entities.get(entity_id)

    def find_entities(
        self,
        db,
        *,
        query_text: str,
        filters: GraphFilters,
        aliases: tuple[str, ...] = (),
        limit: int = 10,
    ) -> tuple[GraphEntityMatch, ...]:
        entity_ids = self.matches_by_query.get(query_text, ())
        return tuple(
            GraphEntityMatch(
                entity=self.entities[entity_id],
                score=1.0,
                matched_name=query_text,
                match_type="fake",
                rank=index,
            )
            for index, entity_id in enumerate(entity_ids[:limit], start=1)
        )

    def get_neighbors(
        self,
        db,
        *,
        entity_id: str,
        degree_cap: int = DEFAULT_DEGREE_CAP,
        relation_types: tuple[str, ...] | None = None,
        filters: GraphFilters | None = None,
    ) -> GraphNeighborhood:
        raise AssertionError("path grounding tests should not call get_neighbors")

    def find_paths(
        self,
        db,
        *,
        source_entity_id: str,
        target_entity_id: str,
        max_hops: int = 2,
        degree_cap: int = DEFAULT_DEGREE_CAP,
        relation_types: tuple[str, ...] | None = None,
        filters: GraphFilters | None = None,
        max_paths: int = 20,
    ) -> tuple[GraphPath, ...]:
        return self.paths[:max_paths]

    def get_relationships(self, db, ids, *, graph_version: str):
        return ()

    def get_chunks_for_entities(
        self,
        db,
        entity_ids,
        *,
        graph_version: str,
        max_source_chunks_per_result: int = 3,
    ):
        return {}

    def get_chunks_for_relationships(
        self,
        db,
        relationship_ids,
        *,
        graph_version: str,
        max_source_chunks_per_result: int = 3,
    ):
        return {}


def test_graph_provider_evidence_uses_hydrated_chunk_and_sanitizes_graph_text() -> None:
    provider = GraphProvider(store=_path_store(), default_graph_version=GRAPH_VERSION)
    plan = _plan()

    result = provider.retrieve_provider_result(
        _db_with_chunks(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=_tasks(plan),
    )

    assert result.status == "executed"
    assert result.candidates
    assert result.evidence
    assert result.evidence_pack is not None
    assert result.trace["evidence_count"] == 1
    assert result.trace["evidence_pack_id"] == result.evidence_pack.pack_id
    assert result.trace["dropped_evidence_count"] == 0

    candidate = result.candidates[0]
    evidence = result.evidence[0]
    block = result.evidence_pack.prompt_blocks[0]
    assert candidate.text == CHUNK_TEXT
    assert candidate.source_type == "text_chunk"
    assert candidate.source_title == "Graph Source"
    assert evidence.text == CHUNK_TEXT
    assert evidence.source_title == "Graph Source"
    assert block.text == CHUNK_TEXT
    assert block.source_type == "text_chunk"
    assert block.source_title == "Graph Source"
    assert block.parent_id == "parent_graph"
    assert evidence.parent_id == "parent_graph"
    assert evidence.metadata["source_type"] == "text_chunk"
    assert PARENT_TEXT not in evidence.text

    prompt = build_answer_input(query=plan.original_query, evidence=list(result.evidence))
    serialized = serialize_provider_result(result)
    trace_anchor = result.trace["graph_candidates"][0]["source_anchors"][0]
    assert serialized["candidates"][0]["source_type"] == "text_chunk"
    assert serialized["candidates"][0]["source_anchor"]["chunk_id"] == "chunk_graph"
    assert TOXIC not in evidence.text
    assert TOXIC not in prompt
    assert TOXIC not in repr(result.trace)
    assert TOXIC not in repr(serialized["trace"])
    assert TOXIC not in repr(trace_anchor)
    assert TOXIC not in repr(candidate.metadata)
    assert TOXIC not in repr(evidence.metadata)
    assert "text_span" not in trace_anchor
    assert "source_title" not in trace_anchor["metadata"]

    for metadata in (candidate.metadata, evidence.metadata):
        assert metadata["provider"] == "graph"
        assert metadata["graph_candidate_id"].startswith("graph_path:")
        assert metadata["entity_ids"] == ["ent_source", "ent_target"]
        assert metadata["relationship_ids"] == ["rel_path"]
        assert metadata["graph_path"] == {
            "path_id": "path_1",
            "graph_version": GRAPH_VERSION,
            "source_entity_id": "ent_source",
            "target_entity_id": "ent_target",
            "entity_ids": ["ent_source", "ent_target"],
            "relationship_ids": ["rel_path"],
            "hops": 1,
        }
        assert metadata["graph_score"] == 0.9
        assert metadata["grounding_strength"] == 1.0
        assert metadata["graph"]["source_type"] == "graph_path"
        assert metadata["source_anchor"]["chunk_id"] == "chunk_graph"
        assert "text_span" not in metadata["source_anchor"]
        assert "source_title" not in metadata["source_anchor"]["metadata"]
        assert metadata["grounded_source_chunk_ids"] == ["chunk_graph"]
        assert metadata["retrieval_task_id"] == candidate.retrieval_task_id
        assert metadata["retrieval_unit_id"] == candidate.retrieval_unit_id


def test_anchor_metadata_source_title_is_not_prompt_visible_fallback() -> None:
    provider = GraphProvider(store=_path_store(), default_graph_version=GRAPH_VERSION)
    plan = _plan()

    result = provider.retrieve_provider_result(
        _db_with_chunks(document_title=None, document_source_title="Document Safe Title"),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=_tasks(plan),
    )

    candidate = result.candidates[0]
    evidence = result.evidence[0]
    prompt = build_answer_input(query=plan.original_query, evidence=list(result.evidence))

    assert candidate.source_title == "Document Safe Title"
    assert evidence.source_title == "Document Safe Title"
    assert TOXIC not in prompt
    assert TOXIC not in repr(result.trace)


def test_graph_provider_zero_context_budget_drops_grounded_evidence() -> None:
    provider = GraphProvider(
        store=_path_store(),
        default_graph_version=GRAPH_VERSION,
        max_context_tokens=0,
    )
    plan = _plan()

    result = provider.retrieve_provider_result(
        _db_with_chunks(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=_tasks(plan),
    )

    assert result.status == "executed"
    assert result.candidates
    assert result.evidence == ()
    assert result.evidence_pack is not None
    assert result.evidence_pack.blocks == ()
    assert len(result.evidence_pack.dropped_blocks) == 1
    assert result.evidence_pack.dropped_blocks[0].drop_reason == "token_budget"
    assert result.evidence_pack.dropped_blocks[0].source_type == "text_chunk"
    assert result.evidence_pack.dropped_blocks[0].text == CHUNK_TEXT
    assert result.trace["evidence_count"] == 0
    assert result.trace["evidence_pack_id"] == result.evidence_pack.pack_id
    assert result.trace["dropped_evidence_count"] == 1


def _path_store() -> _FakeGraphStore:
    source = _entity("ent_source", "SourceCo")
    target = _entity("ent_target", "TargetCo")
    relationship = GraphRelationship(
        relationship_id="rel_path",
        graph_version=GRAPH_VERSION,
        source_entity_id=source.entity_id,
        target_entity_id=target.entity_id,
        relation_type="affects",
        confidence=0.9,
        metadata={"description": TOXIC},
    )
    path = GraphPath(
        graph_version=GRAPH_VERSION,
        source_entity_id=source.entity_id,
        target_entity_id=target.entity_id,
        entity_ids=(source.entity_id, target.entity_id),
        relationships=(relationship,),
        score=0.9,
        source_anchors=(_anchor("chunk_graph"),),
        path_id="path_1",
        metadata={
            "path_text": TOXIC,
            "summary": TOXIC,
            "description": TOXIC,
        },
    )
    return _FakeGraphStore(
        entities=(source, target),
        matches_by_query={
            "SourceCo": ("ent_source",),
            "TargetCo": ("ent_target",),
        },
        paths=(path,),
    )


def _plan() -> QueryPlan:
    return QueryPlan(
        plan_id="plan_graph_grounding",
        original_query="SourceCo to TargetCo",
        entities=(Entity(value="SourceCo"), Entity(value="TargetCo")),
        metadata={"graph_version": GRAPH_VERSION},
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_graph",
                purpose="graph_path",
                text="SourceCo to TargetCo",
                provider="graph",
            ),
        ),
    )


def _tasks(plan: QueryPlan) -> list[RetrievalTask]:
    return tasks_from_plan(plan, executable_providers=("hybrid", "graph"))


def _entity(entity_id: str, canonical_name: str) -> GraphEntity:
    return GraphEntity(
        entity_id=entity_id,
        graph_version=GRAPH_VERSION,
        canonical_name=canonical_name,
        entity_type="company",
    )


def _anchor(chunk_id: str) -> SourceAnchor:
    return SourceAnchor(
        document_id="doc_graph",
        chunk_id=chunk_id,
        parent_id="parent_graph",
        page_start=7,
        page_end=8,
        text_span=TOXIC,
        graph_ids=("relationship:rel_path",),
        metadata={
            "source_title": TOXIC,
            "chunk_index": 3,
            "summary": TOXIC,
        },
    )


def _db_with_chunks(
    *,
    document_title: str | None = "Graph Source",
    document_source_title: str | None = None,
):
    document = SimpleNamespace(
        document_id="doc_graph",
        title=document_title,
        source_title=document_source_title,
        source_uri="s3://graph-source",
        metadata_json={
            "company": "SourceCo",
            "description": TOXIC,
        },
    )
    chunk = SimpleNamespace(
        chunk_id="chunk_graph",
        document_id=document.document_id,
        document=document,
        parent_id="parent_graph",
        chunk_index=3,
        text=CHUNK_TEXT,
        section_title="Graph Section",
        page_start=7,
        page_end=8,
        token_count=7,
        metadata_json={
            "section_name": "Graph Section",
            "summary": TOXIC,
            "path_text": TOXIC,
        },
    )
    return SimpleNamespace(
        chunks_by_id={chunk.chunk_id: chunk},
        parent_blocks_by_id={
            "parent_graph": SimpleNamespace(parent_id="parent_graph", text=PARENT_TEXT)
        },
    )
