from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from atlas.db.models import (
    Chunk,
    GraphCommunity,
    GraphEntityAnchor,
    GraphEntityRecord,
    GraphIndex,
    GraphRelationshipAnchor,
    GraphRelationshipRecord,
)
from atlas.retrieval.providers.graph.fixture import (
    GRAPH_FIXTURE_LOADER_VERSION,
    GraphFixtureHashConflictError,
    GraphFixtureValidationError,
    canonical_fixture_hash,
    load_graph_fixture,
    load_graph_fixture_file,
    validate_graph_fixture,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "graph" / "hub_fixture.json"
CHUNK_IDS = ("chunk_supply", "chunk_product")


class _FakeSession:
    def __init__(self, chunk_ids: tuple[str, ...] = CHUNK_IDS) -> None:
        self.chunks = {
            chunk_id: SimpleNamespace(chunk_id=chunk_id)
            for chunk_id in chunk_ids
        }
        self.graph_indexes: dict[str, GraphIndex] = {}
        self.graph_entities: dict[tuple[str, str], GraphEntityRecord] = {}
        self.graph_relationships: dict[tuple[str, str], GraphRelationshipRecord] = {}
        self.graph_entity_anchors: dict[tuple[str, str], GraphEntityAnchor] = {}
        self.graph_relationship_anchors: dict[tuple[str, str], GraphRelationshipAnchor] = {}
        self.graph_communities: dict[tuple[str, str], GraphCommunity] = {}
        self.flushes = 0
        self._pending_operations: list[str] = []
        self.flush_batches: list[tuple[str, ...]] = []

    def get(self, model: Any, key: str) -> Any:
        if model is Chunk:
            return self.chunks.get(key)
        if model is GraphIndex:
            return self.graph_indexes.get(key)
        raise AssertionError(f"unexpected get({model}, {key})")

    def add(self, value: Any) -> None:
        if isinstance(value, GraphIndex):
            self.graph_indexes[value.graph_version] = value
            self._pending_operations.append("add:GraphIndex")
        elif isinstance(value, GraphEntityRecord):
            self.graph_entities[(value.graph_version, value.entity_id)] = value
            self._pending_operations.append("add:GraphEntityRecord")
        elif isinstance(value, GraphRelationshipRecord):
            self.graph_relationships[(value.graph_version, value.relationship_id)] = value
            self._pending_operations.append("add:GraphRelationshipRecord")
        elif isinstance(value, GraphEntityAnchor):
            self.graph_entity_anchors[(value.graph_version, value.anchor_id)] = value
            self._pending_operations.append("add:GraphEntityAnchor")
        elif isinstance(value, GraphRelationshipAnchor):
            self.graph_relationship_anchors[(value.graph_version, value.anchor_id)] = value
            self._pending_operations.append("add:GraphRelationshipAnchor")
        elif isinstance(value, GraphCommunity):
            self.graph_communities[(value.graph_version, value.community_id)] = value
            self._pending_operations.append("add:GraphCommunity")
        else:
            raise AssertionError(f"unexpected add({type(value)!r})")

    def add_all(self, values: list[Any]) -> None:
        for value in values:
            self.add(value)

    def execute(self, statement: Any) -> None:
        table_name = statement.table.name
        graph_version = _delete_graph_version(statement)
        stores = {
            "graph_relationship_anchors": self.graph_relationship_anchors,
            "graph_entity_anchors": self.graph_entity_anchors,
            "graph_communities": self.graph_communities,
            "graph_relationships": self.graph_relationships,
            "graph_entities": self.graph_entities,
        }
        store = stores.get(table_name)
        if store is None:
            raise AssertionError(f"unexpected delete from {table_name}")
        self._pending_operations.append(f"delete:{table_name}")
        for key in [key for key in store if key[0] == graph_version]:
            del store[key]

    def flush(self) -> None:
        self.flushes += 1
        self.flush_batches.append(tuple(self._pending_operations))
        self._pending_operations = []


def _delete_graph_version(statement: Any) -> str:
    for criterion in getattr(statement, "_where_criteria", ()):
        value = getattr(getattr(criterion, "right", None), "value", None)
        if isinstance(value, str):
            return value
    raise AssertionError("delete statement did not include a graph_version value")


def _sample_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_fixture_schema_validation_and_canonical_hash() -> None:
    fixture = _sample_fixture()
    reordered = {
        "metadata": fixture["metadata"],
        "corpus_version": fixture["corpus_version"],
        "graph_version": fixture["graph_version"],
        "fixture_schema_version": fixture["fixture_schema_version"],
        "communities": fixture["communities"],
        "relationships": fixture["relationships"],
        "entities": fixture["entities"],
    }

    assert canonical_fixture_hash(fixture) == canonical_fixture_hash(reordered)
    validated = validate_graph_fixture(fixture)
    assert validated.graph_version == "test_graph_v3_fixture"
    assert validated.row_counts == {
        "entities": 3,
        "relationships": 2,
        "entity_anchors": 3,
        "relationship_anchors": 2,
        "communities": 1,
        "hub_like_entities": 1,
    }

    invalid = copy.deepcopy(fixture)
    del invalid["graph_version"]
    with pytest.raises(GraphFixtureValidationError, match="graph_version is required"):
        validate_graph_fixture(invalid)


def test_missing_chunk_rejected() -> None:
    fixture = _sample_fixture()
    fixture["entities"][0]["anchors"][0]["chunk_id"] = "missing_chunk"

    with pytest.raises(GraphFixtureValidationError, match="missing_chunk"):
        load_graph_fixture(_FakeSession(), fixture)


def test_empty_anchor_rejected() -> None:
    fixture = _sample_fixture()
    fixture["relationships"][0]["anchors"] = []

    with pytest.raises(GraphFixtureValidationError, match="must have at least one anchor"):
        load_graph_fixture(_FakeSession(), fixture)


def test_repeated_load_is_idempotent_noop() -> None:
    db = _FakeSession()
    fixture = _sample_fixture()

    first = load_graph_fixture(db, fixture)
    second = load_graph_fixture(db, copy.deepcopy(fixture))

    assert first.loaded is True
    assert first.noop is False
    assert second.loaded is False
    assert second.noop is True
    assert second.fixture_hash == first.fixture_hash
    assert second.loader_version == GRAPH_FIXTURE_LOADER_VERSION
    assert len(db.graph_entities) == 3
    assert len(db.graph_relationships) == 2
    assert db.flushes == 4
    assert db.flush_batches == [
        ("add:GraphIndex",),
        ("add:GraphEntityRecord",) * 3,
        ("add:GraphRelationshipRecord",) * 2,
        (
            ("add:GraphEntityAnchor",) * 3
            + ("add:GraphRelationshipAnchor",) * 2
            + ("add:GraphCommunity",)
        ),
    ]


def test_hash_conflict_fails_without_replace() -> None:
    db = _FakeSession()
    fixture = _sample_fixture()
    load_graph_fixture(db, fixture)

    changed = copy.deepcopy(fixture)
    changed["metadata"]["changed"] = True

    with pytest.raises(GraphFixtureHashConflictError, match="different fixture_hash"):
        load_graph_fixture(db, changed, replace=False)


def test_replace_succeeds_and_clears_old_rows() -> None:
    db = _FakeSession()
    fixture = _sample_fixture()
    load_graph_fixture(db, fixture)

    changed = copy.deepcopy(fixture)
    changed["relationships"] = changed["relationships"][:1]
    changed["communities"] = []
    changed["entities"][1]["canonical_name"] = "Supplier Constraint Network"

    result = load_graph_fixture(db, changed, replace=True)

    assert result.loaded is True
    assert result.replaced is True
    assert result.row_counts["relationships"] == 1
    assert len(db.graph_relationships) == 1
    assert len(db.graph_relationship_anchors) == 1
    assert len(db.graph_communities) == 0
    assert ("test_graph_v3_fixture", "rel_apple_vision_pro") not in db.graph_relationships
    assert (
        db.graph_entities[("test_graph_v3_fixture", "ent_supplier_network")].canonical_name
        == "Supplier Constraint Network"
    )
    assert db.graph_indexes["test_graph_v3_fixture"].fixture_hash == result.fixture_hash
    assert db.flush_batches[4:] == [
        (
            "delete:graph_relationship_anchors",
            "delete:graph_entity_anchors",
            "delete:graph_communities",
            "delete:graph_relationships",
            "delete:graph_entities",
        ),
        ("add:GraphIndex",),
        ("add:GraphEntityRecord",) * 3,
        ("add:GraphRelationshipRecord",),
        (
            ("add:GraphEntityAnchor",) * 3
            + ("add:GraphRelationshipAnchor",)
        ),
    ]


def test_sample_fixture_includes_hub_like_entity_and_preserves_graph_only_text() -> None:
    db = _FakeSession()
    result = load_graph_fixture_file(db, FIXTURE_PATH)
    graph_index = db.graph_indexes["test_graph_v3_fixture"]
    apple = db.graph_entities[("test_graph_v3_fixture", "ent_apple")]
    community = db.graph_communities[("test_graph_v3_fixture", "comm_apple_supply")]

    assert result.hub_like_entity_ids == ("ent_apple",)
    assert graph_index.row_counts_json["hub_like_entities"] == 1
    assert graph_index.metadata_json["fixture_loader"]["hub_like_entity_ids"] == ["ent_apple"]
    assert "DO_NOT_USE_AS_EVIDENCE" in graph_index.metadata_json["notes"]
    assert "DO_NOT_USE_AS_EVIDENCE" in apple.metadata_json["description"]
    assert "DO_NOT_USE_AS_EVIDENCE" in community.summary
    assert set(db.chunks) == set(CHUNK_IDS)
