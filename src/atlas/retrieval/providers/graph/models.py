from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from atlas.retrieval.contracts import SourceAnchor


GRAPH_PROVIDER = "graph"
GRAPH_PROVIDER_VERSION = "3.0.0"

SUPPORTED_GRAPH_MODES: tuple[str, ...] = ("local", "path")
UNSUPPORTED_GRAPH_MODES: tuple[str, ...] = ("global", "community", "drift")

GraphMode = Literal["local", "path", "global", "community", "drift"]
SupportedGraphMode = Literal["local", "path"]
UnsupportedGraphMode = Literal["global", "community", "drift"]
GraphCandidateSourceType = Literal[
    "graph_node",
    "graph_edge",
    "graph_path",
    "graph_neighborhood",
    "community_report",
]
GraphObjectType = Literal["entity", "relationship", "community", "path"]

DEFAULT_DEGREE_CAP = 25
DEFAULT_MAX_HOPS = 2
DEFAULT_MAX_PATHS = 20
DEFAULT_MAX_SOURCE_CHUNKS_PER_RESULT = 3


def canonical_graph_relation_types(
    relation_types: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    return tuple(
        sorted({relation_type for relation_type in relation_types or () if relation_type})
    )


@dataclass(frozen=True)
class GraphFilters:
    graph_version: str
    corpus_version: str | None = None
    entity_types: tuple[str, ...] = ()
    relation_types: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relation_types",
            canonical_graph_relation_types(self.relation_types),
        )


@dataclass(frozen=True)
class GraphObjectRef:
    graph_version: str
    object_id: str
    object_type: GraphObjectType


@dataclass(frozen=True)
class GraphEntityRef:
    graph_version: str
    entity_id: str
    object_type: Literal["entity"] = "entity"

    @property
    def object_id(self) -> str:
        return self.entity_id

    def as_object_ref(self) -> GraphObjectRef:
        return GraphObjectRef(
            graph_version=self.graph_version,
            object_id=self.entity_id,
            object_type=self.object_type,
        )


@dataclass(frozen=True)
class GraphRelationshipRef:
    graph_version: str
    relationship_id: str
    object_type: Literal["relationship"] = "relationship"

    @property
    def object_id(self) -> str:
        return self.relationship_id

    def as_object_ref(self) -> GraphObjectRef:
        return GraphObjectRef(
            graph_version=self.graph_version,
            object_id=self.relationship_id,
            object_type=self.object_type,
        )


@dataclass(frozen=True)
class GraphEntity:
    entity_id: str
    graph_version: str
    canonical_name: str
    entity_type: str
    canonical_name_norm: str | None = None
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True)
class GraphEntityMatch:
    entity: GraphEntity
    score: float
    matched_name: str | None = None
    match_type: str | None = None
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphRelationship:
    relationship_id: str
    graph_version: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True)
class GraphPath:
    graph_version: str
    source_entity_id: str
    target_entity_id: str
    entity_ids: tuple[str, ...]
    relationships: tuple[GraphRelationship, ...]
    score: float = 0.0
    source_anchors: tuple[SourceAnchor, ...] = ()
    path_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for relationship in self.relationships:
            if relationship.graph_version != self.graph_version:
                raise ValueError("GraphPath relationships must match graph_version")

    @property
    def relationship_ids(self) -> tuple[str, ...]:
        return tuple(relationship.relationship_id for relationship in self.relationships)

    @property
    def hops(self) -> int:
        return len(self.relationships)


@dataclass(frozen=True)
class GraphNeighborhood:
    center_entity: GraphEntity
    neighbors: tuple[GraphEntity, ...] = ()
    relationships: tuple[GraphRelationship, ...] = ()
    source_anchors: tuple[SourceAnchor, ...] = ()
    degree_cap: int = DEFAULT_DEGREE_CAP
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphCandidate:
    """Graph-only context object; graph_text is not evidence text."""

    candidate_id: str
    source_type: GraphCandidateSourceType
    graph_version: str
    graph_text: str
    entity_ids: tuple[str, ...] = ()
    relationship_ids: tuple[str, ...] = ()
    object_refs: tuple[GraphObjectRef, ...] = ()
    community_id: str | None = None
    source_anchors: tuple[SourceAnchor, ...] = ()
    graph_score: float = 0.0
    rank: int | None = None
    grounding_strength: float = 0.0
    provider: str = GRAPH_PROVIDER
    mode: GraphMode | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for ref in self.object_refs:
            if ref.graph_version != self.graph_version:
                raise ValueError("GraphCandidate object_refs must match graph_version")

    @property
    def primary_source_anchor(self) -> SourceAnchor | None:
        return self.source_anchors[0] if self.source_anchors else None

    @property
    def source_chunk_ids(self) -> tuple[str, ...]:
        return tuple(anchor.chunk_id for anchor in self.source_anchors if anchor.chunk_id)

    @property
    def source_document_ids(self) -> tuple[str, ...]:
        return tuple(anchor.document_id for anchor in self.source_anchors if anchor.document_id)

    @property
    def entity_refs(self) -> tuple[GraphEntityRef, ...]:
        return tuple(
            GraphEntityRef(graph_version=self.graph_version, entity_id=entity_id)
            for entity_id in self.entity_ids
        )

    @property
    def relationship_refs(self) -> tuple[GraphRelationshipRef, ...]:
        return tuple(
            GraphRelationshipRef(
                graph_version=self.graph_version,
                relationship_id=relationship_id,
            )
            for relationship_id in self.relationship_ids
        )

    @property
    def provenance_refs(self) -> tuple[GraphObjectRef, ...]:
        if self.object_refs:
            return self.object_refs

        refs: list[GraphObjectRef] = [
            ref.as_object_ref() for ref in self.entity_refs + self.relationship_refs
        ]
        if self.community_id:
            refs.append(
                GraphObjectRef(
                    graph_version=self.graph_version,
                    object_id=self.community_id,
                    object_type="community",
                )
            )
        return tuple(refs)
