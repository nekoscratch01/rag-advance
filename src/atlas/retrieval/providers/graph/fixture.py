from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import delete

from atlas.db.models import (
    Chunk,
    GraphCommunity,
    GraphEntityAnchor,
    GraphEntityRecord,
    GraphIndex,
    GraphRelationshipAnchor,
    GraphRelationshipRecord,
    WHOLE_CHUNK_TEXT_SPAN_HASH,
)
from atlas.retrieval.providers.graph.models import DEFAULT_DEGREE_CAP


GRAPH_FIXTURE_SCHEMA_VERSION = "graph_fixture_v1"
GRAPH_FIXTURE_LOADER_VERSION = "graph_fixture_loader_v1"
GRAPH_INDEX_STATUS_LOADED = "loaded"


class GraphFixtureError(ValueError):
    """Base error for graph fixture loading."""


class GraphFixtureValidationError(GraphFixtureError):
    """Raised when fixture shape or source anchors are invalid."""


class GraphFixtureHashConflictError(GraphFixtureError):
    """Raised when a graph_version already exists with different fixture content."""


@dataclass(frozen=True)
class GraphFixtureLoadResult:
    graph_version: str
    fixture_schema_version: str
    fixture_hash: str
    loader_version: str
    row_counts: dict[str, int]
    status: str
    loaded: bool
    noop: bool = False
    replaced: bool = False
    hub_like_entity_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _AnchorSpec:
    anchor_id: str
    chunk_id: str
    text_span: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _EntitySpec:
    entity_id: str
    canonical_name: str
    canonical_name_norm: str
    entity_type: str
    aliases: list[str]
    metadata: dict[str, Any]
    anchors: list[_AnchorSpec]


@dataclass(frozen=True)
class _RelationshipSpec:
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    confidence: float
    metadata: dict[str, Any]
    anchors: list[_AnchorSpec]


@dataclass(frozen=True)
class _CommunitySpec:
    community_id: str
    level: int
    summary: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _ValidatedFixture:
    graph_version: str
    corpus_version: str | None
    fixture_schema_version: str
    fixture_hash: str
    metadata: dict[str, Any]
    entities: list[_EntitySpec]
    relationships: list[_RelationshipSpec]
    communities: list[_CommunitySpec]
    hub_like_entity_ids: tuple[str, ...]
    row_counts: dict[str, int]

    @property
    def anchor_chunk_ids(self) -> tuple[str, ...]:
        chunk_ids = [
            anchor.chunk_id
            for entity in self.entities
            for anchor in entity.anchors
        ]
        chunk_ids.extend(
            anchor.chunk_id
            for relationship in self.relationships
            for anchor in relationship.anchors
        )
        return tuple(dict.fromkeys(chunk_ids))


def load_graph_fixture_file(
    db: Any,
    path: str | Path,
    *,
    replace: bool = False,
) -> GraphFixtureLoadResult:
    with Path(path).open("r", encoding="utf-8") as handle:
        fixture = json.load(handle)
    return load_graph_fixture(db, fixture, replace=replace)


def load_graph_fixture(
    db: Any,
    fixture: Mapping[str, Any],
    *,
    replace: bool = False,
) -> GraphFixtureLoadResult:
    validated = validate_graph_fixture(fixture)
    _validate_anchor_chunks_exist(db, validated.anchor_chunk_ids)

    existing = db.get(GraphIndex, validated.graph_version)
    if existing is not None and existing.fixture_hash == validated.fixture_hash and not replace:
        return GraphFixtureLoadResult(
            graph_version=validated.graph_version,
            fixture_schema_version=validated.fixture_schema_version,
            fixture_hash=validated.fixture_hash,
            loader_version=GRAPH_FIXTURE_LOADER_VERSION,
            row_counts=dict(existing.row_counts_json or validated.row_counts),
            status=existing.status,
            loaded=False,
            noop=True,
            hub_like_entity_ids=validated.hub_like_entity_ids,
        )

    if existing is not None and existing.fixture_hash != validated.fixture_hash and not replace:
        raise GraphFixtureHashConflictError(
            "graph_version already exists with different fixture_hash: "
            f"graph_version={validated.graph_version!r} "
            f"existing_hash={existing.fixture_hash!r} incoming_hash={validated.fixture_hash!r}; "
            "pass replace=True to replace this graph fixture"
        )

    replaced = existing is not None
    if replaced:
        _delete_graph_rows(db, validated.graph_version)
        _update_graph_index(existing, validated)
        graph_index = existing
    else:
        graph_index = _build_graph_index(validated)

    db.add(graph_index)
    db.flush()

    db.add_all(_build_entity_records(validated))
    db.flush()

    db.add_all(_build_relationship_records(validated))
    db.flush()

    db.add_all(_build_entity_anchor_records(validated))
    db.add_all(_build_relationship_anchor_records(validated))
    db.add_all(_build_community_records(validated))
    db.flush()

    return GraphFixtureLoadResult(
        graph_version=validated.graph_version,
        fixture_schema_version=validated.fixture_schema_version,
        fixture_hash=validated.fixture_hash,
        loader_version=GRAPH_FIXTURE_LOADER_VERSION,
        row_counts=dict(validated.row_counts),
        status=GRAPH_INDEX_STATUS_LOADED,
        loaded=True,
        replaced=replaced,
        hub_like_entity_ids=validated.hub_like_entity_ids,
    )


def validate_graph_fixture(fixture: Mapping[str, Any]) -> _ValidatedFixture:
    root = _require_mapping(fixture, "fixture")
    fixture_hash = canonical_fixture_hash(root)
    fixture_schema_version = _require_str(root, "fixture_schema_version")
    if fixture_schema_version != GRAPH_FIXTURE_SCHEMA_VERSION:
        raise GraphFixtureValidationError(
            "fixture_schema_version must be "
            f"{GRAPH_FIXTURE_SCHEMA_VERSION!r}; got {fixture_schema_version!r}"
        )

    graph_version = _require_str(root, "graph_version")
    corpus_version = _optional_str(root.get("corpus_version"), "corpus_version")
    metadata = _optional_mapping(root.get("metadata"), "metadata")
    entities = _parse_entities(_require_list(root, "entities"))
    relationships = _parse_relationships(_optional_list(root.get("relationships"), "relationships"))
    communities = _parse_communities(_optional_list(root.get("communities"), "communities"))

    if not entities:
        raise GraphFixtureValidationError("entities must contain at least one entity")

    entity_ids = [entity.entity_id for entity in entities]
    _reject_duplicates(entity_ids, "entity_id")

    relationship_ids = [relationship.relationship_id for relationship in relationships]
    _reject_duplicates(relationship_ids, "relationship_id")

    known_entity_ids = set(entity_ids)
    for relationship in relationships:
        if relationship.source_entity_id not in known_entity_ids:
            raise GraphFixtureValidationError(
                "relationship source_entity_id references unknown entity: "
                f"relationship_id={relationship.relationship_id!r} "
                f"source_entity_id={relationship.source_entity_id!r}"
            )
        if relationship.target_entity_id not in known_entity_ids:
            raise GraphFixtureValidationError(
                "relationship target_entity_id references unknown entity: "
                f"relationship_id={relationship.relationship_id!r} "
                f"target_entity_id={relationship.target_entity_id!r}"
            )

    _reject_duplicates([community.community_id for community in communities], "community_id")
    _reject_duplicates(
        [anchor.anchor_id for entity in entities for anchor in entity.anchors],
        "entity_anchor_id",
    )
    _reject_duplicates(
        [
            anchor.anchor_id
            for relationship in relationships
            for anchor in relationship.anchors
        ],
        "relationship_anchor_id",
    )

    hub_like_entity_ids = _hub_like_entity_ids(entities, relationships)
    if not hub_like_entity_ids:
        raise GraphFixtureValidationError(
            "fixture must include at least one hub-like entity signal: "
            "set entity.metadata.hub_like=true or include more than "
            f"{DEFAULT_DEGREE_CAP} incident relationships for an entity"
        )

    row_counts = {
        "entities": len(entities),
        "relationships": len(relationships),
        "entity_anchors": sum(len(entity.anchors) for entity in entities),
        "relationship_anchors": sum(
            len(relationship.anchors) for relationship in relationships
        ),
        "communities": len(communities),
        "hub_like_entities": len(hub_like_entity_ids),
    }

    return _ValidatedFixture(
        graph_version=graph_version,
        corpus_version=corpus_version,
        fixture_schema_version=fixture_schema_version,
        fixture_hash=fixture_hash,
        metadata=metadata,
        entities=entities,
        relationships=relationships,
        communities=communities,
        hub_like_entity_ids=hub_like_entity_ids,
        row_counts=row_counts,
    )


def canonical_fixture_hash(fixture: Mapping[str, Any]) -> str:
    try:
        normalized = json.dumps(
            fixture,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise GraphFixtureValidationError(f"fixture must be JSON serializable: {exc}") from exc
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _parse_entities(raw_entities: list[Any]) -> list[_EntitySpec]:
    entities: list[_EntitySpec] = []
    for index, raw_entity in enumerate(raw_entities):
        path = f"entities[{index}]"
        entity = _require_mapping(raw_entity, path)
        _reject_unknown_keys(
            entity,
            path,
            {
                "entity_id",
                "canonical_name",
                "canonical_name_norm",
                "entity_type",
                "aliases",
                "description",
                "metadata",
                "anchors",
            },
        )
        metadata = _metadata_with_optional_description(entity, path)
        entity_id = _require_str(entity, f"{path}.entity_id")
        anchors = _parse_anchors(
            _require_non_empty_list(entity.get("anchors"), f"{path}.anchors"),
            owner_id=entity_id,
            owner_path=path,
        )
        canonical_name = _require_str(entity, f"{path}.canonical_name")
        entities.append(
            _EntitySpec(
                entity_id=entity_id,
                canonical_name=canonical_name,
                canonical_name_norm=_optional_str(
                    entity.get("canonical_name_norm"),
                    f"{path}.canonical_name_norm",
                )
                or _normalize_name(canonical_name),
                entity_type=_require_str(entity, f"{path}.entity_type"),
                aliases=_optional_str_list(entity.get("aliases"), f"{path}.aliases"),
                metadata=metadata,
                anchors=anchors,
            )
        )
    return entities


def _parse_relationships(raw_relationships: list[Any]) -> list[_RelationshipSpec]:
    relationships: list[_RelationshipSpec] = []
    for index, raw_relationship in enumerate(raw_relationships):
        path = f"relationships[{index}]"
        relationship = _require_mapping(raw_relationship, path)
        _reject_unknown_keys(
            relationship,
            path,
            {
                "relationship_id",
                "source_entity_id",
                "target_entity_id",
                "relation_type",
                "confidence",
                "description",
                "metadata",
                "anchors",
            },
        )
        metadata = _metadata_with_optional_description(relationship, path)
        relationship_id = _require_str(relationship, f"{path}.relationship_id")
        anchors = _parse_anchors(
            _require_non_empty_list(relationship.get("anchors"), f"{path}.anchors"),
            owner_id=relationship_id,
            owner_path=path,
        )
        relationships.append(
            _RelationshipSpec(
                relationship_id=relationship_id,
                source_entity_id=_require_str(relationship, f"{path}.source_entity_id"),
                target_entity_id=_require_str(relationship, f"{path}.target_entity_id"),
                relation_type=_require_str(relationship, f"{path}.relation_type"),
                confidence=_optional_confidence(
                    relationship.get("confidence"),
                    f"{path}.confidence",
                ),
                metadata=metadata,
                anchors=anchors,
            )
        )
    return relationships


def _parse_communities(raw_communities: list[Any]) -> list[_CommunitySpec]:
    communities: list[_CommunitySpec] = []
    for index, raw_community in enumerate(raw_communities):
        path = f"communities[{index}]"
        community = _require_mapping(raw_community, path)
        _reject_unknown_keys(
            community,
            path,
            {"community_id", "level", "summary", "metadata"},
        )
        communities.append(
            _CommunitySpec(
                community_id=_require_str(community, f"{path}.community_id"),
                level=_require_int(community, f"{path}.level"),
                summary=_optional_str(community.get("summary"), f"{path}.summary"),
                metadata=_optional_mapping(community.get("metadata"), f"{path}.metadata"),
            )
        )
    return communities


def _parse_anchors(
    raw_anchors: list[Any],
    *,
    owner_id: str,
    owner_path: str,
) -> list[_AnchorSpec]:
    anchors: list[_AnchorSpec] = []
    for index, raw_anchor in enumerate(raw_anchors):
        path = f"{owner_path}.anchors[{index}]"
        anchor = _require_mapping(raw_anchor, path)
        _reject_unknown_keys(anchor, path, {"anchor_id", "chunk_id", "text_span", "metadata"})
        text_span = _optional_str(anchor.get("text_span"), f"{path}.text_span", allow_empty=True)
        anchors.append(
            _AnchorSpec(
                anchor_id=_optional_str(anchor.get("anchor_id"), f"{path}.anchor_id")
                or f"{owner_id}_anchor_{index}",
                chunk_id=_require_str(anchor, f"{path}.chunk_id"),
                text_span=text_span,
                metadata=_optional_mapping(anchor.get("metadata"), f"{path}.metadata"),
            )
        )
    return anchors


def _build_graph_index(validated: _ValidatedFixture) -> GraphIndex:
    return GraphIndex(
        graph_version=validated.graph_version,
        corpus_version=validated.corpus_version,
        fixture_schema_version=validated.fixture_schema_version,
        fixture_hash=validated.fixture_hash,
        loader_version=GRAPH_FIXTURE_LOADER_VERSION,
        row_counts_json=dict(validated.row_counts),
        status=GRAPH_INDEX_STATUS_LOADED,
        loaded_at=datetime.now(UTC),
        metadata_json=_graph_index_metadata(validated),
    )


def _update_graph_index(graph_index: GraphIndex, validated: _ValidatedFixture) -> None:
    graph_index.corpus_version = validated.corpus_version
    graph_index.fixture_schema_version = validated.fixture_schema_version
    graph_index.fixture_hash = validated.fixture_hash
    graph_index.loader_version = GRAPH_FIXTURE_LOADER_VERSION
    graph_index.row_counts_json = dict(validated.row_counts)
    graph_index.status = GRAPH_INDEX_STATUS_LOADED
    graph_index.loaded_at = datetime.now(UTC)
    graph_index.metadata_json = _graph_index_metadata(validated)


def _graph_index_metadata(validated: _ValidatedFixture) -> dict[str, Any]:
    metadata = dict(validated.metadata)
    metadata["fixture_loader"] = {
        "loader_version": GRAPH_FIXTURE_LOADER_VERSION,
        "hub_like_entity_ids": list(validated.hub_like_entity_ids),
        "hub_like_entity_count": len(validated.hub_like_entity_ids),
        "degree_cap": DEFAULT_DEGREE_CAP,
    }
    return metadata


def _build_entity_records(validated: _ValidatedFixture) -> list[GraphEntityRecord]:
    return [
        GraphEntityRecord(
            graph_version=validated.graph_version,
            entity_id=entity.entity_id,
            canonical_name=entity.canonical_name,
            canonical_name_norm=entity.canonical_name_norm,
            entity_type=entity.entity_type,
            aliases_json=list(entity.aliases),
            metadata_json=dict(entity.metadata),
        )
        for entity in validated.entities
    ]


def _build_relationship_records(validated: _ValidatedFixture) -> list[GraphRelationshipRecord]:
    return [
        GraphRelationshipRecord(
            graph_version=validated.graph_version,
            relationship_id=relationship.relationship_id,
            source_entity_id=relationship.source_entity_id,
            target_entity_id=relationship.target_entity_id,
            relation_type=relationship.relation_type,
            confidence=relationship.confidence,
            metadata_json=dict(relationship.metadata),
        )
        for relationship in validated.relationships
    ]


def _build_entity_anchor_records(validated: _ValidatedFixture) -> list[GraphEntityAnchor]:
    records: list[GraphEntityAnchor] = []
    for entity in validated.entities:
        for anchor in entity.anchors:
            records.append(
                GraphEntityAnchor(
                    graph_version=validated.graph_version,
                    anchor_id=anchor.anchor_id,
                    entity_id=entity.entity_id,
                    chunk_id=anchor.chunk_id,
                    text_span=anchor.text_span,
                    text_span_hash=_text_span_hash(anchor.text_span),
                    metadata_json=dict(anchor.metadata),
                )
            )
    return records


def _build_relationship_anchor_records(
    validated: _ValidatedFixture,
) -> list[GraphRelationshipAnchor]:
    records: list[GraphRelationshipAnchor] = []
    for relationship in validated.relationships:
        for anchor in relationship.anchors:
            records.append(
                GraphRelationshipAnchor(
                    graph_version=validated.graph_version,
                    anchor_id=anchor.anchor_id,
                    relationship_id=relationship.relationship_id,
                    chunk_id=anchor.chunk_id,
                    text_span=anchor.text_span,
                    text_span_hash=_text_span_hash(anchor.text_span),
                    metadata_json=dict(anchor.metadata),
                )
            )
    return records


def _build_community_records(validated: _ValidatedFixture) -> list[GraphCommunity]:
    return [
        GraphCommunity(
            graph_version=validated.graph_version,
            community_id=community.community_id,
            level=community.level,
            summary=community.summary,
            metadata_json=dict(community.metadata),
        )
        for community in validated.communities
    ]


def _delete_graph_rows(db: Any, graph_version: str) -> None:
    for model in (
        GraphRelationshipAnchor,
        GraphEntityAnchor,
        GraphCommunity,
        GraphRelationshipRecord,
        GraphEntityRecord,
    ):
        db.execute(delete(model).where(model.graph_version == graph_version))
    db.flush()


def _validate_anchor_chunks_exist(db: Any, chunk_ids: tuple[str, ...]) -> None:
    missing = [chunk_id for chunk_id in chunk_ids if db.get(Chunk, chunk_id) is None]
    if missing:
        raise GraphFixtureValidationError(
            "graph fixture anchors reference missing chunks: "
            + ", ".join(sorted(missing))
        )


def _hub_like_entity_ids(
    entities: list[_EntitySpec],
    relationships: list[_RelationshipSpec],
) -> tuple[str, ...]:
    degrees: Counter[str] = Counter()
    for relationship in relationships:
        degrees[relationship.source_entity_id] += 1
        degrees[relationship.target_entity_id] += 1

    hub_like_ids: list[str] = []
    for entity in entities:
        metadata = entity.metadata
        explicit_signal = any(
            metadata.get(key) is True
            for key in ("hub_like", "is_hub_like", "hub_entity")
        )
        if explicit_signal or degrees[entity.entity_id] > DEFAULT_DEGREE_CAP:
            hub_like_ids.append(entity.entity_id)
    return tuple(hub_like_ids)


def _text_span_hash(text_span: str | None) -> str:
    if text_span:
        return hashlib.sha256(text_span.encode("utf-8")).hexdigest()
    return WHOLE_CHUNK_TEXT_SPAN_HASH


def _metadata_with_optional_description(
    value: Mapping[str, Any],
    path: str,
) -> dict[str, Any]:
    metadata = _optional_mapping(value.get("metadata"), f"{path}.metadata")
    description = _optional_str(value.get("description"), f"{path}.description", allow_empty=True)
    if description is not None:
        metadata = dict(metadata)
        metadata.setdefault("description", description)
    return metadata


def _normalize_name(value: str) -> str:
    return " ".join(value.casefold().split())


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GraphFixtureValidationError(f"{path} must be an object")
    return dict(value)


def _require_list(value: Mapping[str, Any], key: str) -> list[Any]:
    if key not in value:
        raise GraphFixtureValidationError(f"{key} is required")
    raw_value = value[key]
    if not isinstance(raw_value, list):
        raise GraphFixtureValidationError(f"{key} must be a list")
    return raw_value


def _optional_list(value: Any, path: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise GraphFixtureValidationError(f"{path} must be a list")
    return value


def _require_non_empty_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise GraphFixtureValidationError(f"{path} must be a non-empty list")
    if not value:
        raise GraphFixtureValidationError(f"{path} must have at least one anchor")
    return value


def _require_str(value: Mapping[str, Any], path: str) -> str:
    key = path.rsplit(".", 1)[-1]
    if key not in value:
        raise GraphFixtureValidationError(f"{path} is required")
    return _validate_str(value[key], path)


def _optional_str(value: Any, path: str, *, allow_empty: bool = False) -> str | None:
    if value is None:
        return None
    return _validate_str(value, path, allow_empty=allow_empty)


def _validate_str(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise GraphFixtureValidationError(f"{path} must be a string")
    if not allow_empty and not value.strip():
        raise GraphFixtureValidationError(f"{path} must be a non-empty string")
    return value


def _require_int(value: Mapping[str, Any], path: str) -> int:
    key = path.rsplit(".", 1)[-1]
    if key not in value:
        raise GraphFixtureValidationError(f"{path} is required")
    raw_value = value[key]
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise GraphFixtureValidationError(f"{path} must be an integer")
    return raw_value


def _optional_confidence(value: Any, path: str) -> float:
    if value is None:
        return 1.0
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise GraphFixtureValidationError(f"{path} must be a number")
    confidence = float(value)
    if not math.isfinite(confidence):
        raise GraphFixtureValidationError(f"{path} must be finite")
    return confidence


def _optional_mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise GraphFixtureValidationError(f"{path} must be an object")
    return dict(value)


def _optional_str_list(value: Any, path: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise GraphFixtureValidationError(f"{path} must be a list")
    strings: list[str] = []
    for index, item in enumerate(value):
        strings.append(_validate_str(item, f"{path}[{index}]"))
    return strings


def _reject_unknown_keys(
    value: Mapping[str, Any],
    path: str,
    allowed_keys: set[str],
) -> None:
    unknown_keys = sorted(set(value) - allowed_keys)
    if unknown_keys:
        raise GraphFixtureValidationError(
            f"{path} contains unknown keys: {', '.join(unknown_keys)}"
        )


def _reject_duplicates(values: list[str], label: str) -> None:
    duplicates = sorted(
        value for value, count in Counter(values).items() if count > 1
    )
    if duplicates:
        raise GraphFixtureValidationError(
            f"duplicate {label} values are not allowed: {', '.join(duplicates)}"
        )
