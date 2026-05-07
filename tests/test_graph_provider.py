from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from atlas.query_orchestrator.schema import Entity, QueryPlan, RetrievalUnit
from atlas.retrieval.contracts import SourceAnchor
from atlas.retrieval.models.candidate import Candidate
from atlas.retrieval.models.retrieval_task import RetrievalTask, tasks_from_plan
from atlas.retrieval.providers.graph.models import (
    DEFAULT_DEGREE_CAP,
    GraphCandidate,
    GraphEntity,
    GraphEntityMatch,
    GraphFilters,
    GraphNeighborhood,
    GraphPath,
    GraphRelationship,
)
from atlas.retrieval.providers.graph.provider import GraphProvider


GRAPH_VERSION = "test_graph_provider"


class _FakeGraphStore:
    def __init__(
        self,
        *,
        entities: tuple[GraphEntity, ...] = (),
        matches_by_query: dict[str, tuple[str, ...]] | None = None,
        neighborhood: GraphNeighborhood | None = None,
        paths: tuple[GraphPath, ...] = (),
    ) -> None:
        self.entities = {entity.entity_id: entity for entity in entities}
        self.matches_by_query = matches_by_query or {}
        self.neighborhood = neighborhood
        self.paths = paths
        self.find_entities_calls: list[dict[str, Any]] = []
        self.get_neighbors_calls: list[dict[str, Any]] = []
        self.find_paths_calls: list[dict[str, Any]] = []

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
        self.find_entities_calls.append(
            {
                "query_text": query_text,
                "filters": filters,
                "aliases": aliases,
                "limit": limit,
            }
        )
        entity_ids: list[str] = []
        for term in (query_text, *aliases):
            for entity_id in self.matches_by_query.get(term, ()):
                if entity_id not in entity_ids:
                    entity_ids.append(entity_id)
        matches = [
            GraphEntityMatch(
                entity=self.entities[entity_id],
                score=1.0,
                matched_name=query_text,
                match_type="fake",
                rank=index,
            )
            for index, entity_id in enumerate(entity_ids[:limit], start=1)
            if entity_id in self.entities
        ]
        return tuple(matches)

    def get_neighbors(
        self,
        db,
        *,
        entity_id: str,
        degree_cap: int = DEFAULT_DEGREE_CAP,
        relation_types: tuple[str, ...] | None = None,
        filters: GraphFilters | None = None,
    ) -> GraphNeighborhood:
        self.get_neighbors_calls.append(
            {
                "entity_id": entity_id,
                "filters": filters,
                "degree_cap": degree_cap,
                "relation_types": relation_types,
            }
        )
        if self.neighborhood is not None:
            return self.neighborhood
        return GraphNeighborhood(
            center_entity=self.entities[entity_id],
            metadata={"degree_seen": 0, "neighbors_returned": 0},
        )

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
        self.find_paths_calls.append(
            {
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
                "filters": filters,
                "max_hops": max_hops,
                "max_paths": max_paths,
                "degree_cap": degree_cap,
                "relation_types": relation_types,
            }
        )
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


def test_query_plan_entities_take_priority_over_fallback() -> None:
    plan_entity = _entity("ent_plan", "PlanCo")
    fallback_entity = _entity("ent_fallback", "FallbackCo")
    store = _FakeGraphStore(
        entities=(plan_entity, fallback_entity),
        matches_by_query={
            "PlanCo": ("ent_plan",),
            "PC": ("ent_plan",),
            "FallbackCo supplier map": ("ent_fallback",),
        },
        neighborhood=_neighborhood(plan_entity),
    )
    provider = GraphProvider(store=store, default_graph_version=GRAPH_VERSION)
    plan = _plan(
        original_query="OriginalCo supplier map",
        entities=(Entity(value="PlanCo", aliases=("PC",)),),
        unit_text="FallbackCo supplier map",
    )

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
    assert store.get_neighbors_calls[0]["entity_id"] == "ent_plan"
    assert [call["query_text"] for call in store.find_entities_calls] == ["PlanCo"]
    assert all(call["query_text"] != "FallbackCo supplier map" for call in store.find_entities_calls)


def test_fallback_entity_resolution_uses_task_query_text_not_original_query() -> None:
    fallback_entity = _entity("ent_task", "TaskCo")
    store = _FakeGraphStore(
        entities=(fallback_entity,),
        matches_by_query={
            "TaskCo supplier map": ("ent_task",),
            "OriginalCo supplier map": (),
        },
        neighborhood=_neighborhood(fallback_entity),
    )
    provider = GraphProvider(store=store, default_graph_version=GRAPH_VERSION)
    plan = _plan(
        original_query="OriginalCo supplier map",
        unit_text="TaskCo supplier map",
    )

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
    assert [call["query_text"] for call in store.find_entities_calls] == [
        "TaskCo supplier map"
    ]
    assert store.find_entities_calls[0]["query_text"] != plan.original_query


def test_task_metadata_entities_are_used_before_fallback() -> None:
    metadata_entity = _entity("ent_meta", "MetaCo")
    fallback_entity = _entity("ent_fallback", "FallbackCo")
    store = _FakeGraphStore(
        entities=(metadata_entity, fallback_entity),
        matches_by_query={
            "MetaCo": ("ent_meta",),
            "MCO": ("ent_meta",),
            "FallbackCo supplier map": ("ent_fallback",),
        },
        neighborhood=_neighborhood(metadata_entity),
    )
    provider = GraphProvider(store=store, default_graph_version=GRAPH_VERSION)
    plan = _plan(
        original_query="OriginalCo supplier map",
        unit_text="FallbackCo supplier map",
        unit_metadata={"entities": ["MetaCo"], "aliases": ["MCO"]},
    )

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
    assert store.get_neighbors_calls[0]["entity_id"] == "ent_meta"
    assert [call["query_text"] for call in store.find_entities_calls] == ["MetaCo"]


def test_local_mode_calls_get_neighbors() -> None:
    entity = _entity("ent_local", "LocalCo")
    store = _FakeGraphStore(
        entities=(entity,),
        matches_by_query={"LocalCo relation map": ("ent_local",)},
        neighborhood=_neighborhood(entity),
    )
    provider = GraphProvider(store=store, default_graph_version=GRAPH_VERSION)
    plan = _plan(unit_text="LocalCo relation map")

    result = provider.retrieve_provider_result(
        _db_with_chunks(),
        query=plan.original_query,
        top_k=3,
        filters={"relation_types": ["affects"]},
        options={},
        query_plan=plan,
        retrieval_tasks=_tasks(plan),
    )

    assert result.status == "executed"
    assert len(store.get_neighbors_calls) == 1
    assert store.get_neighbors_calls[0]["entity_id"] == "ent_local"
    assert store.get_neighbors_calls[0]["relation_types"] == ("affects",)
    assert store.get_neighbors_calls[0]["filters"].relation_types == ("affects",)
    assert store.find_paths_calls == []


def test_path_mode_calls_find_paths() -> None:
    source = _entity("ent_source", "SourceCo")
    target = _entity("ent_target", "TargetCo")
    relationship = _relationship("rel_path", source.entity_id, target.entity_id)
    path = GraphPath(
        graph_version=GRAPH_VERSION,
        source_entity_id=source.entity_id,
        target_entity_id=target.entity_id,
        entity_ids=(source.entity_id, target.entity_id),
        relationships=(relationship,),
        score=0.9,
        source_anchors=(_anchor("chunk_graph"),),
        path_id="path_1",
    )
    store = _FakeGraphStore(
        entities=(source, target),
        matches_by_query={
            "SourceCo": ("ent_source",),
            "TargetCo": ("ent_target",),
        },
        paths=(path,),
    )
    provider = GraphProvider(store=store, default_graph_version=GRAPH_VERSION)
    plan = _plan(
        entities=(Entity(value="SourceCo"), Entity(value="TargetCo")),
        unit_text="SourceCo to TargetCo",
    )

    result = provider.retrieve_provider_result(
        _db_with_chunks(),
        query=plan.original_query,
        top_k=3,
        filters={"relation_types": ["affects"]},
        options={},
        query_plan=plan,
        retrieval_tasks=_tasks(plan),
    )

    assert result.status == "executed"
    assert len(store.find_paths_calls) == 1
    assert store.find_paths_calls[0]["source_entity_id"] == "ent_source"
    assert store.find_paths_calls[0]["target_entity_id"] == "ent_target"
    assert store.find_paths_calls[0]["max_hops"] == 2
    assert store.find_paths_calls[0]["max_paths"] == 20
    assert store.find_paths_calls[0]["relation_types"] == ("affects",)
    assert store.find_paths_calls[0]["filters"].relation_types == ("affects",)
    assert store.get_neighbors_calls == []
    candidate = result.candidates[0]
    assert candidate.metadata["graph_candidate_id"].startswith("graph_path:")
    assert candidate.metadata["grounded_source_chunk_ids"] == ["chunk_graph"]
    assert candidate.metadata["graph_path"] == {
        "path_id": "path_1",
        "graph_version": GRAPH_VERSION,
        "source_entity_id": "ent_source",
        "target_entity_id": "ent_target",
        "entity_ids": ["ent_source", "ent_target"],
        "relationship_ids": ["rel_path"],
        "hops": 1,
    }
    assert candidate.metadata["graph"]["graph_path"] == candidate.metadata["graph_path"]


def test_generic_metadata_mode_does_not_override_inferred_path_mode() -> None:
    source = _entity("ent_source", "SourceCo")
    target = _entity("ent_target", "TargetCo")
    relationship = _relationship("rel_path", source.entity_id, target.entity_id)
    path = GraphPath(
        graph_version=GRAPH_VERSION,
        source_entity_id=source.entity_id,
        target_entity_id=target.entity_id,
        entity_ids=(source.entity_id, target.entity_id),
        relationships=(relationship,),
        score=0.9,
        source_anchors=(_anchor("chunk_graph"),),
        path_id="path_1",
    )
    store = _FakeGraphStore(
        entities=(source, target),
        matches_by_query={
            "SourceCo": ("ent_source",),
            "TargetCo": ("ent_target",),
        },
        paths=(path,),
    )
    provider = GraphProvider(store=store, default_graph_version=GRAPH_VERSION)
    plan = _plan(
        entities=(Entity(value="SourceCo"), Entity(value="TargetCo")),
        unit_text="SourceCo to TargetCo",
        unit_metadata={"mode": "local"},
    )

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
    assert result.trace["tasks"][0]["mode"] == "path"
    assert len(store.find_paths_calls) == 1
    assert store.get_neighbors_calls == []


def test_generic_metadata_mode_dense_only_is_ignored_by_graph_mode_state_machine() -> None:
    entity = _entity("ent_local", "LocalCo")
    store = _FakeGraphStore(
        entities=(entity,),
        matches_by_query={"LocalCo relation map": ("ent_local",)},
        neighborhood=_neighborhood(entity),
    )
    provider = GraphProvider(store=store, default_graph_version=GRAPH_VERSION)
    plan = _plan(
        unit_text="LocalCo relation map",
        unit_metadata={"mode": "dense_only"},
    )

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
    assert result.reason is None
    assert result.trace["tasks"][0]["mode"] == "local"
    assert result.trace["tasks"][0]["reason"] is None
    assert len(store.get_neighbors_calls) == 1
    assert store.find_paths_calls == []


@pytest.mark.parametrize("mode", ["global", "community", "drift"])
def test_unsupported_graph_modes_return_clear_empty_status(mode: str) -> None:
    store = _FakeGraphStore()
    provider = GraphProvider(store=store, default_graph_version=GRAPH_VERSION)
    plan = _plan(unit_metadata={"graph_mode": mode})

    result = provider.retrieve_provider_result(
        _db_with_chunks(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=_tasks(plan),
    )

    assert result.status == "empty"
    assert result.reason == f"unsupported_graph_mode:{mode}"
    assert result.trace["tasks"][0]["reason"] == f"unsupported_graph_mode:{mode}"
    assert store.get_neighbors_calls == []
    assert store.find_paths_calls == []


def test_no_resolved_entities_returns_empty_reason() -> None:
    store = _FakeGraphStore()
    provider = GraphProvider(store=store, default_graph_version=GRAPH_VERSION)
    plan = _plan(original_query="OriginalCo supplier map", unit_text="TaskCo supplier map")

    result = provider.retrieve_provider_result(
        _db_with_chunks(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=_tasks(plan),
    )

    assert result.status == "empty"
    assert result.reason == "no_resolved_entity"
    assert result.trace["tasks"][0]["reason"] == "no_resolved_entity"
    assert store.find_entities_calls[0]["query_text"] == "TaskCo supplier map"


def test_degree_cap_default_and_truncation_trace_include_hub_metadata() -> None:
    center = _entity("ent_hub", "HubCo")
    neighbors = tuple(_entity(f"ent_neighbor_{index}", f"Neighbor {index}") for index in range(25))
    relationships = tuple(
        _relationship(
            f"rel_{index}",
            center.entity_id,
            neighbors[index].entity_id,
            relation_type="related_to",
        )
        for index in range(25)
    )
    neighborhood = GraphNeighborhood(
        center_entity=center,
        neighbors=neighbors,
        relationships=relationships,
        source_anchors=(_anchor("chunk_graph"),),
        degree_cap=DEFAULT_DEGREE_CAP,
        truncated=True,
        metadata={
            "degree_seen": 30,
            "neighbors_returned": 25,
            "relationships_returned": 25,
        },
    )
    store = _FakeGraphStore(
        entities=(center, *neighbors),
        matches_by_query={"HubCo graph": ("ent_hub",)},
        neighborhood=neighborhood,
    )
    provider = GraphProvider(store=store, default_graph_version=GRAPH_VERSION)
    plan = _plan(unit_text="HubCo graph")

    result = provider.retrieve_provider_result(
        _db_with_chunks(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={},
        query_plan=plan,
        retrieval_tasks=_tasks(plan),
    )

    task_trace = result.trace["tasks"][0]
    assert store.get_neighbors_calls[0]["degree_cap"] == DEFAULT_DEGREE_CAP
    assert result.trace["truncated"] is True
    assert result.trace["hub_cap_applied"] is True
    assert result.trace["degree_seen"] == 30
    assert task_trace["truncated"] is True
    assert task_trace["hub_cap_applied"] is True
    assert task_trace["degree_seen"] == 30
    assert task_trace["neighbors_examined"] == 25
    assert task_trace["neighbors_returned"] == 25
    assert task_trace["truncated_reason"] == "degree_cap"
    assert task_trace["cap_config"]["degree_cap"] == DEFAULT_DEGREE_CAP


def test_graph_provider_emits_only_grounded_candidates_and_no_graph_evidence() -> None:
    entity = _entity("ent_local", "LocalCo")
    store = _FakeGraphStore(
        entities=(entity,),
        matches_by_query={"LocalCo relation map": ("ent_local",)},
        neighborhood=_neighborhood(entity),
    )
    provider = GraphProvider(store=store, default_graph_version=GRAPH_VERSION)
    plan = _plan(unit_text="LocalCo relation map")

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
    assert result.evidence == ()
    assert result.candidates
    assert all(isinstance(candidate, Candidate) for candidate in result.candidates)
    assert all(not isinstance(candidate, GraphCandidate) for candidate in result.candidates)
    candidate = result.candidates[0]
    assert candidate.provider == "graph"
    assert candidate.text == "Grounded chunk text from source storage."
    assert candidate.metadata["graph_candidate_id"].startswith("graph_local:")
    assert candidate.metadata["grounded_source_chunk_ids"] == ["chunk_graph"]
    assert candidate.metadata["graph"]["graph_candidate_id"] == candidate.metadata["graph_candidate_id"]
    assert candidate.metadata["graph"]["grounded_source_chunk_ids"] == ["chunk_graph"]
    json.dumps(candidate.metadata["source_anchor"])
    json.dumps(candidate.metadata)
    assert "graph_text" not in repr(result.evidence)


def test_grounded_source_chunk_ids_only_include_hydrated_candidate_chunk() -> None:
    entity = _entity("ent_local", "LocalCo")
    store = _FakeGraphStore(
        entities=(entity,),
        matches_by_query={"LocalCo relation map": ("ent_local",)},
        neighborhood=_neighborhood(
            entity,
            source_anchors=(_anchor("chunk_graph"), _anchor("missing_chunk")),
        ),
    )
    provider = GraphProvider(store=store, default_graph_version=GRAPH_VERSION)
    plan = _plan(unit_text="LocalCo relation map")

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
    assert result.evidence == ()
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.chunk_id == "chunk_graph"
    assert candidate.metadata["grounded_source_chunk_ids"] == ["chunk_graph"]
    assert candidate.metadata["graph"]["grounded_source_chunk_ids"] == ["chunk_graph"]
    assert candidate.metadata["source_anchor"]["chunk_id"] == "chunk_graph"
    assert [
        anchor["chunk_id"]
        for anchor in result.trace["graph_candidates"][0]["source_anchors"]
    ] == ["chunk_graph", "missing_chunk"]


def _plan(
    *,
    original_query: str = "LocalCo relation map",
    entities: tuple[Entity, ...] = (),
    unit_text: str = "LocalCo relation map",
    unit_metadata: dict[str, Any] | None = None,
) -> QueryPlan:
    return QueryPlan(
        plan_id="plan_graph",
        original_query=original_query,
        entities=entities,
        metadata={"graph_version": GRAPH_VERSION},
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_graph",
                purpose="graph_context",
                text=unit_text,
                provider="graph",
                metadata=unit_metadata or {},
            ),
        ),
    )


def _tasks(plan: QueryPlan) -> list[RetrievalTask]:
    return tasks_from_plan(plan, executable_providers=("hybrid", "graph"))


def _entity(entity_id: str, canonical_name: str, entity_type: str = "company") -> GraphEntity:
    return GraphEntity(
        entity_id=entity_id,
        graph_version=GRAPH_VERSION,
        canonical_name=canonical_name,
        entity_type=entity_type,
        aliases=(),
    )


def _relationship(
    relationship_id: str,
    source_entity_id: str,
    target_entity_id: str,
    *,
    relation_type: str = "affects",
) -> GraphRelationship:
    return GraphRelationship(
        relationship_id=relationship_id,
        graph_version=GRAPH_VERSION,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relation_type=relation_type,
        confidence=0.8,
    )


def _anchor(chunk_id: str) -> SourceAnchor:
    return SourceAnchor(
        document_id="doc_graph",
        chunk_id=chunk_id,
        parent_id="parent_graph",
        page_start=7,
        page_end=8,
        text_span="grounded graph span",
        graph_ids=("relationship:rel_local",),
        metadata={
            "source_title": "Graph Source",
            "chunk_index": 3,
            "graph_ids": ["relationship:rel_local"],
        },
    )


def _neighborhood(
    entity: GraphEntity,
    *,
    source_anchors: tuple[SourceAnchor, ...] | None = None,
) -> GraphNeighborhood:
    neighbor = _entity("ent_neighbor", "NeighborCo")
    relationship = _relationship("rel_local", entity.entity_id, neighbor.entity_id)
    return GraphNeighborhood(
        center_entity=entity,
        neighbors=(neighbor,),
        relationships=(relationship,),
        source_anchors=source_anchors or (_anchor("chunk_graph"),),
        degree_cap=DEFAULT_DEGREE_CAP,
        truncated=False,
        metadata={
            "degree_seen": 1,
            "neighbors_returned": 1,
            "relationships_returned": 1,
        },
    )


def _db_with_chunks():
    document = SimpleNamespace(
        document_id="doc_graph",
        title="Graph Source",
        source_uri="s3://graph-source",
        metadata_json={"company": "LocalCo"},
    )
    chunk = SimpleNamespace(
        chunk_id="chunk_graph",
        document_id=document.document_id,
        document=document,
        parent_id="parent_graph",
        chunk_index=3,
        text="Grounded chunk text from source storage.",
        section_title="Graph Section",
        page_start=7,
        page_end=8,
        token_count=6,
        metadata_json={"section_name": "Graph Section"},
    )
    return SimpleNamespace(chunks_by_id={chunk.chunk_id: chunk})
