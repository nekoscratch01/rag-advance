from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from atlas.db.models import (
    Chunk,
    Document,
    GraphEntityAnchor,
    GraphEntityRecord,
    GraphRelationshipAnchor,
    GraphRelationshipRecord,
    ParentBlock,
)
from atlas.retrieval.contracts import SourceAnchor
from atlas.retrieval.providers.graph.models import (
    DEFAULT_MAX_SOURCE_CHUNKS_PER_RESULT,
    GraphEntity,
    GraphEntityMatch,
    GraphFilters,
    GraphNeighborhood,
    GraphPath,
    GraphRelationship,
)


_AnchorT = TypeVar("_AnchorT", GraphEntityAnchor, GraphRelationshipAnchor)


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def _dedupe_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return tuple(deduped)


def _entity_from_record(record: GraphEntityRecord) -> GraphEntity:
    return GraphEntity(
        entity_id=record.entity_id,
        graph_version=record.graph_version,
        canonical_name=record.canonical_name,
        canonical_name_norm=record.canonical_name_norm,
        entity_type=record.entity_type,
        aliases=tuple(record.aliases_json or ()),
        metadata=dict(record.metadata_json or {}),
        created_at=record.created_at,
    )


def _relationship_from_record(record: GraphRelationshipRecord) -> GraphRelationship:
    return GraphRelationship(
        relationship_id=record.relationship_id,
        graph_version=record.graph_version,
        source_entity_id=record.source_entity_id,
        target_entity_id=record.target_entity_id,
        relation_type=record.relation_type,
        confidence=record.confidence,
        metadata=dict(record.metadata_json or {}),
        created_at=record.created_at,
    )


def _relationship_anchor_scope(filters: GraphFilters):
    anchor_scope = (
        select(GraphRelationshipAnchor.anchor_id)
        .join(Chunk, GraphRelationshipAnchor.chunk_id == Chunk.chunk_id)
        .where(
            GraphRelationshipAnchor.graph_version == GraphRelationshipRecord.graph_version,
            GraphRelationshipAnchor.relationship_id
            == GraphRelationshipRecord.relationship_id,
        )
    )
    if filters.document_ids:
        anchor_scope = anchor_scope.where(Chunk.document_id.in_(filters.document_ids))
    if filters.chunk_ids:
        anchor_scope = anchor_scope.where(Chunk.chunk_id.in_(filters.chunk_ids))
    return anchor_scope.exists()


def _relationship_conditions(
    *,
    filters: GraphFilters,
    entity_id: str | None = None,
) -> list[Any]:
    conditions: list[Any] = [GraphRelationshipRecord.graph_version == filters.graph_version]
    if entity_id is not None:
        conditions.append(
            or_(
                GraphRelationshipRecord.source_entity_id == entity_id,
                GraphRelationshipRecord.target_entity_id == entity_id,
            )
        )
    if filters.relation_types:
        conditions.append(GraphRelationshipRecord.relation_type.in_(filters.relation_types))
    if filters.document_ids or filters.chunk_ids:
        conditions.append(_relationship_anchor_scope(filters))
    return conditions


def _source_anchor(
    anchor: GraphEntityAnchor | GraphRelationshipAnchor,
    chunk: Chunk,
    document: Document,
    parent: ParentBlock | None,
    *,
    object_type: str,
    object_id: str,
) -> SourceAnchor:
    graph_ids = (f"{object_type}:{object_id}",)
    metadata: dict[str, Any] = {
        "graph_version": anchor.graph_version,
        "graph_object_type": object_type,
        "anchor_id": anchor.anchor_id,
        "text_span_hash": anchor.text_span_hash,
        "source_title": document.title,
        "chunk_index": chunk.chunk_index,
        "graph_ids": list(graph_ids),
    }
    anchor_metadata = dict(anchor.metadata_json or {})
    if anchor_metadata:
        metadata["anchor_metadata"] = anchor_metadata

    return SourceAnchor(
        document_id=document.document_id,
        chunk_id=chunk.chunk_id,
        parent_id=chunk.parent_id,
        page_start=(
            chunk.page_start
            if chunk.page_start is not None
            else getattr(parent, "page_start", None)
        ),
        page_end=(
            chunk.page_end
            if chunk.page_end is not None
            else getattr(parent, "page_end", None)
        ),
        text_span=anchor.text_span,
        graph_ids=graph_ids,
        metadata=metadata,
    )


def _dedupe_anchors(anchors: Iterable[SourceAnchor]) -> tuple[SourceAnchor, ...]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[SourceAnchor] = []
    for anchor in anchors:
        key = (
            anchor.document_id,
            anchor.chunk_id,
            anchor.text_span,
            anchor.graph_ids,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(anchor)
    return tuple(deduped)


class PostgresGraphStore:
    """Postgres/SQLAlchemy implementation of the graph storage protocol."""

    def get_entity(
        self,
        db: Session,
        entity_id: str,
        *,
        graph_version: str,
    ) -> GraphEntity | None:
        record = db.execute(
            select(GraphEntityRecord).where(
                GraphEntityRecord.graph_version == graph_version,
                GraphEntityRecord.entity_id == entity_id,
            )
        ).scalar_one_or_none()
        return _entity_from_record(record) if record is not None else None

    def find_entities(
        self,
        db: Session,
        *,
        query_text: str,
        filters: GraphFilters,
        aliases: tuple[str, ...] = (),
        limit: int = 10,
    ) -> tuple[GraphEntityMatch, ...]:
        if limit <= 0:
            return ()

        query_terms = tuple(
            term
            for term in (
                _normalize_text(query_text),
                *(_normalize_text(alias) for alias in aliases),
            )
            if term
        )
        if not query_terms:
            return ()

        stmt = select(GraphEntityRecord).where(
            GraphEntityRecord.graph_version == filters.graph_version
        )
        if filters.entity_types:
            stmt = stmt.where(GraphEntityRecord.entity_type.in_(filters.entity_types))
        if filters.document_ids or filters.chunk_ids:
            anchor_scope = (
                select(GraphEntityAnchor.anchor_id)
                .join(Chunk, GraphEntityAnchor.chunk_id == Chunk.chunk_id)
                .where(
                    GraphEntityAnchor.graph_version == GraphEntityRecord.graph_version,
                    GraphEntityAnchor.entity_id == GraphEntityRecord.entity_id,
                )
            )
            if filters.document_ids:
                anchor_scope = anchor_scope.where(Chunk.document_id.in_(filters.document_ids))
            if filters.chunk_ids:
                anchor_scope = anchor_scope.where(Chunk.chunk_id.in_(filters.chunk_ids))
            stmt = stmt.where(anchor_scope.exists())

        records = db.execute(stmt.order_by(GraphEntityRecord.entity_id)).scalars().all()
        matches: list[GraphEntityMatch] = []
        for record in records:
            score, matched_name, match_type = self._score_entity_match(record, query_terms)
            if score <= 0.0:
                continue
            matches.append(
                GraphEntityMatch(
                    entity=_entity_from_record(record),
                    score=score,
                    matched_name=matched_name,
                    match_type=match_type,
                    metadata={"query_terms": query_terms},
                )
            )

        matches.sort(
            key=lambda match: (
                -match.score,
                match.entity.canonical_name.casefold(),
                match.entity.entity_id,
            )
        )
        ranked: list[GraphEntityMatch] = []
        for rank, match in enumerate(matches[:limit], start=1):
            ranked.append(
                GraphEntityMatch(
                    entity=match.entity,
                    score=match.score,
                    matched_name=match.matched_name,
                    match_type=match.match_type,
                    rank=rank,
                    metadata=match.metadata,
                )
            )
        return tuple(ranked)

    def get_neighbors(
        self,
        db: Session,
        *,
        entity_id: str,
        filters: GraphFilters,
        degree_cap: int = 25,
    ) -> GraphNeighborhood:
        center = self.get_entity(db, entity_id, graph_version=filters.graph_version)
        if center is None:
            raise ValueError(f"graph_entity_not_found:{filters.graph_version}:{entity_id}")

        cap = max(0, degree_cap)
        conditions = _relationship_conditions(filters=filters, entity_id=entity_id)
        if filters.entity_types:
            source_type_match = (
                select(GraphEntityRecord.entity_id)
                .where(
                    GraphEntityRecord.graph_version == GraphRelationshipRecord.graph_version,
                    GraphEntityRecord.entity_id == GraphRelationshipRecord.source_entity_id,
                    GraphEntityRecord.entity_type.in_(filters.entity_types),
                )
                .exists()
            )
            target_type_match = (
                select(GraphEntityRecord.entity_id)
                .where(
                    GraphEntityRecord.graph_version == GraphRelationshipRecord.graph_version,
                    GraphEntityRecord.entity_id == GraphRelationshipRecord.target_entity_id,
                    GraphEntityRecord.entity_type.in_(filters.entity_types),
                )
                .exists()
            )
            conditions.append(
                or_(
                    and_(
                        GraphRelationshipRecord.source_entity_id == entity_id,
                        target_type_match,
                    ),
                    and_(
                        GraphRelationshipRecord.target_entity_id == entity_id,
                        source_type_match,
                    ),
                )
            )

        degree_seen = db.execute(
            select(func.count())
            .select_from(GraphRelationshipRecord)
            .where(*conditions)
        ).scalar_one()

        relationship_records: list[GraphRelationshipRecord] = []
        if cap > 0:
            relationship_records = (
                db.execute(
                    select(GraphRelationshipRecord)
                    .where(*conditions)
                    .order_by(
                        GraphRelationshipRecord.confidence.desc(),
                        GraphRelationshipRecord.relationship_id,
                    )
                    .limit(cap)
                )
                .scalars()
                .all()
            )

        relationships = tuple(_relationship_from_record(record) for record in relationship_records)
        neighbor_ids = _dedupe_preserve_order(
            (
                record.target_entity_id
                if record.source_entity_id == entity_id
                else record.source_entity_id
            )
            for record in relationship_records
        )
        neighbors = self._get_entities_by_ids(
            db,
            graph_version=filters.graph_version,
            entity_ids=neighbor_ids,
        )
        relationship_anchor_map = self._get_chunks_for_relationships(
            db,
            tuple(relationship.relationship_id for relationship in relationships),
            graph_version=filters.graph_version,
            max_source_chunks_per_result=DEFAULT_MAX_SOURCE_CHUNKS_PER_RESULT,
            filters=filters,
        )
        source_anchors = _dedupe_anchors(
            anchor
            for relationship in relationships
            for anchor in relationship_anchor_map.get(relationship.relationship_id, ())
        )
        return GraphNeighborhood(
            center_entity=center,
            neighbors=neighbors,
            relationships=relationships,
            source_anchors=source_anchors,
            degree_cap=cap,
            truncated=degree_seen > cap,
            metadata={
                "degree_seen": degree_seen,
                "neighbors_returned": len(neighbors),
                "relationships_returned": len(relationships),
            },
        )

    def find_paths(
        self,
        db: Session,
        *,
        source_entity_id: str,
        target_entity_id: str,
        filters: GraphFilters,
        max_hops: int = 2,
        max_paths: int = 20,
        degree_cap: int = 25,
    ) -> tuple[GraphPath, ...]:
        if max_hops < 1 or max_paths <= 0 or degree_cap <= 0:
            return ()

        source = self.get_entity(db, source_entity_id, graph_version=filters.graph_version)
        target = self.get_entity(db, target_entity_id, graph_version=filters.graph_version)
        if source is None or target is None:
            return ()

        paths: list[tuple[str, ...]] = []
        path_relationships: list[tuple[GraphRelationshipRecord, ...]] = []

        for first_hop in self._adjacent_relationship_records(
            db,
            entity_id=source_entity_id,
            filters=filters,
            degree_cap=degree_cap,
        ):
            mid_entity_id = self._other_entity_id(first_hop, source_entity_id)
            if mid_entity_id is None:
                continue
            if mid_entity_id == target_entity_id:
                paths.append((source_entity_id, target_entity_id))
                path_relationships.append((first_hop,))
                continue

            if max_hops < 2:
                continue
            if not self._entity_allowed_as_intermediate(
                db,
                graph_version=filters.graph_version,
                entity_id=mid_entity_id,
                filters=filters,
            ):
                continue

            for second_hop in self._adjacent_relationship_records(
                db,
                entity_id=mid_entity_id,
                filters=filters,
                degree_cap=degree_cap,
            ):
                if second_hop.relationship_id == first_hop.relationship_id:
                    continue
                end_entity_id = self._other_entity_id(second_hop, mid_entity_id)
                if end_entity_id != target_entity_id:
                    continue
                paths.append((source_entity_id, mid_entity_id, target_entity_id))
                path_relationships.append((first_hop, second_hop))

        graph_paths: list[GraphPath] = []
        for entity_ids, records in zip(paths, path_relationships, strict=True):
            relationships = tuple(_relationship_from_record(record) for record in records)
            anchors_by_relationship = self._get_chunks_for_relationships(
                db,
                tuple(relationship.relationship_id for relationship in relationships),
                graph_version=filters.graph_version,
                max_source_chunks_per_result=DEFAULT_MAX_SOURCE_CHUNKS_PER_RESULT,
                filters=filters,
            )
            source_anchors = _dedupe_anchors(
                anchor
                for relationship in relationships
                for anchor in anchors_by_relationship.get(relationship.relationship_id, ())
            )
            score = sum(
                relationship.confidence for relationship in relationships
            ) / len(relationships)
            relationship_ids = tuple(relationship.relationship_id for relationship in relationships)
            graph_paths.append(
                GraphPath(
                    graph_version=filters.graph_version,
                    source_entity_id=source_entity_id,
                    target_entity_id=target_entity_id,
                    entity_ids=entity_ids,
                    relationships=relationships,
                    score=score,
                    source_anchors=source_anchors,
                    path_id=":".join((filters.graph_version, *relationship_ids)),
                    metadata={
                        "hops": len(relationships),
                        "degree_cap": degree_cap,
                        "relationship_ids": relationship_ids,
                    },
                )
            )

        graph_paths.sort(
            key=lambda path: (
                path.hops,
                -path.score,
                path.relationship_ids,
            )
        )
        return tuple(graph_paths[:max_paths])

    def get_relationships(
        self,
        db: Session,
        ids: tuple[str, ...],
        *,
        graph_version: str,
    ) -> tuple[GraphRelationship, ...]:
        ordered_ids = _dedupe_preserve_order(ids)
        if not ordered_ids:
            return ()

        records = (
            db.execute(
                select(GraphRelationshipRecord).where(
                    GraphRelationshipRecord.graph_version == graph_version,
                    GraphRelationshipRecord.relationship_id.in_(ordered_ids),
                )
            )
            .scalars()
            .all()
        )
        by_id = {record.relationship_id: record for record in records}
        return tuple(
            _relationship_from_record(by_id[relationship_id])
            for relationship_id in ordered_ids
            if relationship_id in by_id
        )

    def get_chunks_for_entities(
        self,
        db: Session,
        entity_ids: tuple[str, ...],
        *,
        graph_version: str,
        max_source_chunks_per_result: int = DEFAULT_MAX_SOURCE_CHUNKS_PER_RESULT,
    ) -> dict[str, tuple[SourceAnchor, ...]]:
        ordered_ids = _dedupe_preserve_order(entity_ids)
        return self._get_chunks_for_objects(
            db,
            object_ids=ordered_ids,
            graph_version=graph_version,
            anchor_model=GraphEntityAnchor,
            id_attr="entity_id",
            object_type="entity",
            max_source_chunks_per_result=max_source_chunks_per_result,
        )

    def get_chunks_for_relationships(
        self,
        db: Session,
        relationship_ids: tuple[str, ...],
        *,
        graph_version: str,
        max_source_chunks_per_result: int = DEFAULT_MAX_SOURCE_CHUNKS_PER_RESULT,
    ) -> dict[str, tuple[SourceAnchor, ...]]:
        return self._get_chunks_for_relationships(
            db,
            relationship_ids,
            graph_version=graph_version,
            max_source_chunks_per_result=max_source_chunks_per_result,
            filters=None,
        )

    def _get_chunks_for_relationships(
        self,
        db: Session,
        relationship_ids: tuple[str, ...],
        *,
        graph_version: str,
        max_source_chunks_per_result: int = DEFAULT_MAX_SOURCE_CHUNKS_PER_RESULT,
        filters: GraphFilters | None = None,
    ) -> dict[str, tuple[SourceAnchor, ...]]:
        ordered_ids = _dedupe_preserve_order(relationship_ids)
        return self._get_chunks_for_objects(
            db,
            object_ids=ordered_ids,
            graph_version=graph_version,
            anchor_model=GraphRelationshipAnchor,
            id_attr="relationship_id",
            object_type="relationship",
            max_source_chunks_per_result=max_source_chunks_per_result,
            filters=filters,
        )

    def _score_entity_match(
        self,
        record: GraphEntityRecord,
        query_terms: tuple[str, ...],
    ) -> tuple[float, str | None, str | None]:
        canonical_norm = _normalize_text(record.canonical_name_norm or record.canonical_name)
        aliases = tuple(_normalize_text(alias) for alias in record.aliases_json or () if alias)
        best_score = 0.0
        best_name: str | None = None
        best_type: str | None = None

        for term in query_terms:
            if term == canonical_norm:
                return 1.0, record.canonical_name, "canonical_exact"

            for raw_alias, alias_norm in zip(record.aliases_json or (), aliases, strict=False):
                if term == alias_norm:
                    return 0.95, raw_alias, "alias_exact"

            if term and (term in canonical_norm or canonical_norm in term):
                if best_score < 0.75:
                    best_score = 0.75
                    best_name = record.canonical_name
                    best_type = "canonical_partial"

            for raw_alias, alias_norm in zip(record.aliases_json or (), aliases, strict=False):
                if term and alias_norm and (term in alias_norm or alias_norm in term):
                    if best_score < 0.70:
                        best_score = 0.70
                        best_name = raw_alias
                        best_type = "alias_partial"

        return best_score, best_name, best_type

    def _get_entities_by_ids(
        self,
        db: Session,
        *,
        graph_version: str,
        entity_ids: tuple[str, ...],
    ) -> tuple[GraphEntity, ...]:
        ordered_ids = _dedupe_preserve_order(entity_ids)
        if not ordered_ids:
            return ()
        records = (
            db.execute(
                select(GraphEntityRecord).where(
                    GraphEntityRecord.graph_version == graph_version,
                    GraphEntityRecord.entity_id.in_(ordered_ids),
                )
            )
            .scalars()
            .all()
        )
        by_id = {record.entity_id: record for record in records}
        return tuple(
            _entity_from_record(by_id[entity_id])
            for entity_id in ordered_ids
            if entity_id in by_id
        )

    def _adjacent_relationship_records(
        self,
        db: Session,
        *,
        entity_id: str,
        filters: GraphFilters,
        degree_cap: int,
    ) -> tuple[GraphRelationshipRecord, ...]:
        conditions = _relationship_conditions(filters=filters, entity_id=entity_id)
        return tuple(
            db.execute(
                select(GraphRelationshipRecord)
                .where(*conditions)
                .order_by(
                    GraphRelationshipRecord.confidence.desc(),
                    GraphRelationshipRecord.relationship_id,
                )
                .limit(max(0, degree_cap))
            )
            .scalars()
            .all()
        )

    def _entity_allowed_as_intermediate(
        self,
        db: Session,
        *,
        graph_version: str,
        entity_id: str,
        filters: GraphFilters,
    ) -> bool:
        if not filters.entity_types:
            return True
        return (
            db.execute(
                select(GraphEntityRecord.entity_id).where(
                    GraphEntityRecord.graph_version == graph_version,
                    GraphEntityRecord.entity_id == entity_id,
                    GraphEntityRecord.entity_type.in_(filters.entity_types),
                )
            ).scalar_one_or_none()
            is not None
        )

    def _get_chunks_for_objects(
        self,
        db: Session,
        *,
        object_ids: tuple[str, ...],
        graph_version: str,
        anchor_model: type[_AnchorT],
        id_attr: str,
        object_type: str,
        max_source_chunks_per_result: int,
        filters: GraphFilters | None = None,
    ) -> dict[str, tuple[SourceAnchor, ...]]:
        grouped: dict[str, list[SourceAnchor]] = {object_id: [] for object_id in object_ids}
        if not object_ids or max_source_chunks_per_result <= 0:
            return {object_id: tuple(anchors) for object_id, anchors in grouped.items()}

        object_id_column = getattr(anchor_model, id_attr)
        stmt = (
            select(anchor_model, Chunk, Document, ParentBlock)
            .join(Chunk, anchor_model.chunk_id == Chunk.chunk_id)
            .join(Document, Chunk.document_id == Document.document_id)
            .outerjoin(ParentBlock, Chunk.parent_id == ParentBlock.parent_id)
            .where(
                anchor_model.graph_version == graph_version,
                object_id_column.in_(object_ids),
            )
            .order_by(object_id_column, Chunk.chunk_index, anchor_model.anchor_id)
        )
        if filters is not None:
            if filters.document_ids:
                stmt = stmt.where(Chunk.document_id.in_(filters.document_ids))
            if filters.chunk_ids:
                stmt = stmt.where(Chunk.chunk_id.in_(filters.chunk_ids))

        for anchor, chunk, document, parent in db.execute(stmt).all():
            object_id = getattr(anchor, id_attr)
            if object_id not in grouped:
                continue
            if len(grouped[object_id]) >= max_source_chunks_per_result:
                continue
            grouped[object_id].append(
                _source_anchor(
                    anchor,
                    chunk,
                    document,
                    parent,
                    object_type=object_type,
                    object_id=object_id,
                )
            )

        return {object_id: tuple(anchors) for object_id, anchors in grouped.items()}

    @staticmethod
    def _other_entity_id(
        relationship: GraphRelationshipRecord,
        entity_id: str,
    ) -> str | None:
        if relationship.source_entity_id == entity_id:
            return relationship.target_entity_id
        if relationship.target_entity_id == entity_id:
            return relationship.source_entity_id
        return None


__all__ = ["PostgresGraphStore"]
