from __future__ import annotations

from typing import Any, Protocol

from atlas.retrieval.contracts import SourceAnchor
from atlas.retrieval.providers.graph.models import (
    DEFAULT_DEGREE_CAP,
    DEFAULT_MAX_HOPS,
    DEFAULT_MAX_PATHS,
    DEFAULT_MAX_SOURCE_CHUNKS_PER_RESULT,
    GraphEntity,
    GraphEntityRef,
    GraphEntityMatch,
    GraphFilters,
    GraphNeighborhood,
    GraphPath,
    GraphRelationship,
    GraphRelationshipRef,
)


class GraphStore(Protocol):
    """Storage contract for version-local graph object identities."""

    def get_entity(
        self,
        db: Any,
        entity_id: str,
        *,
        graph_version: str,
    ) -> GraphEntity | None:
        ...

    def find_entities(
        self,
        db: Any,
        *,
        query_text: str,
        filters: GraphFilters,
        aliases: tuple[str, ...] = (),
        limit: int = 10,
    ) -> tuple[GraphEntityMatch, ...]:
        """Find entities within filters.graph_version."""
        ...

    def get_neighbors(
        self,
        db: Any,
        *,
        entity_id: str,
        filters: GraphFilters,
        degree_cap: int = DEFAULT_DEGREE_CAP,
    ) -> GraphNeighborhood:
        """Fetch a neighborhood within filters.graph_version."""
        ...

    def find_paths(
        self,
        db: Any,
        *,
        source_entity_id: str,
        target_entity_id: str,
        filters: GraphFilters,
        max_hops: int = DEFAULT_MAX_HOPS,
        max_paths: int = DEFAULT_MAX_PATHS,
        degree_cap: int = DEFAULT_DEGREE_CAP,
    ) -> tuple[GraphPath, ...]:
        """Find paths within filters.graph_version."""
        ...

    def get_relationships(
        self,
        db: Any,
        ids: tuple[str, ...],
        *,
        graph_version: str,
    ) -> tuple[GraphRelationship, ...]:
        ...

    def get_chunks_for_entities(
        self,
        db: Any,
        entity_ids: tuple[str, ...],
        *,
        graph_version: str,
        max_source_chunks_per_result: int = DEFAULT_MAX_SOURCE_CHUNKS_PER_RESULT,
    ) -> dict[str, tuple[SourceAnchor, ...]]:
        ...

    def get_chunks_for_relationships(
        self,
        db: Any,
        relationship_ids: tuple[str, ...],
        *,
        graph_version: str,
        max_source_chunks_per_result: int = DEFAULT_MAX_SOURCE_CHUNKS_PER_RESULT,
    ) -> dict[str, tuple[SourceAnchor, ...]]:
        ...


__all__ = [
    "DEFAULT_DEGREE_CAP",
    "DEFAULT_MAX_HOPS",
    "DEFAULT_MAX_PATHS",
    "DEFAULT_MAX_SOURCE_CHUNKS_PER_RESULT",
    "GraphEntity",
    "GraphEntityRef",
    "GraphEntityMatch",
    "GraphFilters",
    "GraphNeighborhood",
    "GraphPath",
    "GraphRelationship",
    "GraphRelationshipRef",
    "GraphStore",
    "SourceAnchor",
]
