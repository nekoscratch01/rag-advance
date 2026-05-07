from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from atlas.core.config import get_settings
from atlas.db.models import (
    Base,
    Chunk,
    Document,
    GraphEntityAnchor,
    GraphEntityRecord,
    GraphIndex,
    GraphRelationshipAnchor,
    GraphRelationshipRecord,
    ParentBlock,
)
from atlas.retrieval.providers.graph.models import GraphFilters
from atlas.retrieval.providers.graph.postgres_store import PostgresGraphStore


@pytest.fixture(scope="module")
def postgres_engine() -> Iterator[Engine]:
    url = get_settings().database_url
    if not url.startswith("postgresql"):
        pytest.skip("Postgres graph store tests require a PostgreSQL database URL")

    engine = create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 1},
    )
    try:
        with engine.connect() as connection:
            connection.execute(text("select 1")).scalar_one()
        Base.metadata.create_all(bind=engine)
    except SQLAlchemyError as exc:
        engine.dispose()
        pytest.skip(f"Postgres graph store tests require a live PostgreSQL database: {exc}")

    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def graph_fixture(postgres_engine: Engine) -> Iterator[tuple[Session, dict]]:
    connection = postgres_engine.connect()
    transaction = connection.begin()
    db = Session(bind=connection, autoflush=False, expire_on_commit=False)
    try:
        data = _seed_graph_fixture(db)
        yield db, data
    finally:
        db.close()
        transaction.rollback()
        connection.close()


def test_get_entity_scopes_by_graph_version(graph_fixture: tuple[Session, dict]) -> None:
    db, data = graph_fixture
    store = PostgresGraphStore()

    entity = store.get_entity(
        db,
        data["entities"]["company"],
        graph_version=data["graph_version"],
    )

    assert entity is not None
    assert entity.entity_id == data["entities"]["company"]
    assert entity.canonical_name == "3M Company"
    assert entity.aliases == ("3M", "MMM")
    assert store.get_entity(db, data["entities"]["company"], graph_version="missing") is None


def test_find_entities_matches_aliases(graph_fixture: tuple[Session, dict]) -> None:
    db, data = graph_fixture
    store = PostgresGraphStore()

    matches = store.find_entities(
        db,
        query_text="MMM supplier relationships",
        filters=GraphFilters(graph_version=data["graph_version"]),
        aliases=("MMM",),
        limit=5,
    )

    assert matches
    assert matches[0].entity.entity_id == data["entities"]["company"]
    assert matches[0].match_type == "alias_exact"
    assert matches[0].rank == 1


def test_relation_types_filter_neighbors_and_relationship_fetch(
    graph_fixture: tuple[Session, dict],
) -> None:
    db, data = graph_fixture
    store = PostgresGraphStore()

    neighborhood = store.get_neighbors(
        db,
        entity_id=data["entities"]["company"],
        degree_cap=25,
        relation_types=("affects",),
        filters=GraphFilters(
            graph_version=data["graph_version"],
            relation_types=("mentions",),
        ),
    )
    relationships = store.get_relationships(
        db,
        (data["relationships"]["mentions"], data["relationships"]["affects"]),
        graph_version=data["graph_version"],
    )

    assert tuple(rel.relationship_id for rel in neighborhood.relationships) == (
        data["relationships"]["affects"],
    )
    assert tuple(rel.relation_type for rel in neighborhood.relationships) == ("affects",)
    assert tuple(rel.relationship_id for rel in relationships) == (
        data["relationships"]["mentions"],
        data["relationships"]["affects"],
    )


def test_document_and_chunk_filters_apply_to_entities_and_neighbors(
    graph_fixture: tuple[Session, dict],
) -> None:
    db, data = graph_fixture
    store = PostgresGraphStore()

    doc_matches = store.find_entities(
        db,
        query_text="3M",
        filters=GraphFilters(
            graph_version=data["graph_version"],
            document_ids=(data["documents"]["doc_a"],),
        ),
    )
    wrong_chunk_matches = store.find_entities(
        db,
        query_text="3M",
        filters=GraphFilters(
            graph_version=data["graph_version"],
            chunk_ids=(data["chunks"]["chunk_b1"],),
        ),
    )
    doc_neighborhood = store.get_neighbors(
        db,
        entity_id=data["entities"]["company"],
        filters=GraphFilters(
            graph_version=data["graph_version"],
            document_ids=(data["documents"]["doc_b"],),
        ),
    )
    chunk_neighborhood = store.get_neighbors(
        db,
        entity_id=data["entities"]["company"],
        filters=GraphFilters(
            graph_version=data["graph_version"],
            chunk_ids=(data["chunks"]["chunk_a1"],),
        ),
    )

    assert tuple(match.entity.entity_id for match in doc_matches) == (
        data["entities"]["company"],
    )
    assert wrong_chunk_matches == ()
    assert tuple(rel.relationship_id for rel in doc_neighborhood.relationships) == (
        data["relationships"]["mentions"],
    )
    assert tuple(anchor.document_id for anchor in doc_neighborhood.source_anchors) == (
        data["documents"]["doc_b"],
    )
    assert tuple(anchor.chunk_id for anchor in doc_neighborhood.source_anchors) == (
        data["chunks"]["chunk_b1"],
    )
    assert tuple(rel.relationship_id for rel in chunk_neighborhood.relationships) == (
        data["relationships"]["affects"],
    )
    assert tuple(anchor.chunk_id for anchor in chunk_neighborhood.source_anchors) == (
        data["chunks"]["chunk_a1"],
    )


def test_degree_cap_truncates_neighborhood(graph_fixture: tuple[Session, dict]) -> None:
    db, data = graph_fixture
    store = PostgresGraphStore()

    neighborhood = store.get_neighbors(
        db,
        entity_id=data["entities"]["company"],
        filters=GraphFilters(
            graph_version=data["graph_version"],
            relation_types=("related_to",),
        ),
        degree_cap=25,
    )

    assert len(neighborhood.relationships) == 25
    assert len(neighborhood.neighbors) == 25
    assert neighborhood.truncated is True
    assert neighborhood.metadata["degree_seen"] == 30
    assert neighborhood.metadata["neighbors_returned"] == 25
    assert neighborhood.metadata["relationships_returned"] == 25


def test_source_anchor_hydration_enforces_max_per_graph_object(
    graph_fixture: tuple[Session, dict],
) -> None:
    db, data = graph_fixture
    store = PostgresGraphStore()

    entity_chunks = store.get_chunks_for_entities(
        db,
        (data["entities"]["company"], data["entities"]["supplier"]),
        graph_version=data["graph_version"],
        max_source_chunks_per_result=2,
    )
    relationship_chunks = store.get_chunks_for_relationships(
        db,
        (data["relationships"]["affects"],),
        graph_version=data["graph_version"],
        max_source_chunks_per_result=1,
    )

    company_anchors = entity_chunks[data["entities"]["company"]]
    relationship_anchors = relationship_chunks[data["relationships"]["affects"]]

    assert len(company_anchors) == 2
    assert tuple(anchor.chunk_id for anchor in company_anchors) == (
        data["chunks"]["chunk_a1"],
        data["chunks"]["chunk_a2"],
    )
    assert company_anchors[0].document_id == data["documents"]["doc_a"]
    assert company_anchors[0].page_start == 10
    assert company_anchors[0].text_span == "3M Company"
    assert company_anchors[0].graph_ids == (f"entity:{data['entities']['company']}",)
    assert company_anchors[0].metadata["graph_ids"] == [
        f"entity:{data['entities']['company']}"
    ]

    assert len(relationship_anchors) == 1
    assert relationship_anchors[0].chunk_id == data["chunks"]["chunk_a1"]
    assert relationship_anchors[0].text_span == "3M affected margins"
    assert relationship_anchors[0].graph_ids == (
        f"relationship:{data['relationships']['affects']}",
    )


def test_find_paths_returns_local_one_and_two_hop_paths(
    graph_fixture: tuple[Session, dict],
) -> None:
    db, data = graph_fixture
    store = PostgresGraphStore()

    paths = store.find_paths(
        db,
        source_entity_id=data["entities"]["company"],
        target_entity_id=data["entities"]["margin"],
        relation_types=("affects", "part_of", "reports_metric"),
        filters=GraphFilters(
            graph_version=data["graph_version"],
            relation_types=("mentions",),
        ),
        max_hops=2,
        max_paths=10,
        degree_cap=10,
    )

    assert tuple(path.hops for path in paths) == (1, 2)
    assert paths[0].relationship_ids == (data["relationships"]["affects"],)
    assert paths[1].entity_ids == (
        data["entities"]["company"],
        data["entities"]["segment"],
        data["entities"]["margin"],
    )
    assert paths[1].relationship_ids == (
        data["relationships"]["part_of"],
        data["relationships"]["reports_metric"],
    )
    assert all(path.source_anchors for path in paths)


def test_find_paths_applies_max_paths_after_final_ranking(
    graph_fixture: tuple[Session, dict],
) -> None:
    db, data = graph_fixture
    store = PostgresGraphStore()
    bridge_relationship_id = f"{data['relationships']['mentions']}_via_segment"
    bridge_anchor_id = f"{bridge_relationship_id}_anchor"
    db.add(
        _relationship(
            data["graph_version"],
            bridge_relationship_id,
            data["entities"]["segment"],
            data["entities"]["supplier"],
            "supplies",
            confidence=0.995,
        )
    )
    db.flush()
    db.add(
        _relationship_anchor(
            data["graph_version"],
            bridge_anchor_id,
            bridge_relationship_id,
            data["chunks"]["chunk_a2"],
            "segment supplier bridge",
        )
    )
    db.flush()

    paths = store.find_paths(
        db,
        source_entity_id=data["entities"]["company"],
        target_entity_id=data["entities"]["supplier"],
        filters=GraphFilters(
            graph_version=data["graph_version"],
            relation_types=("mentions", "part_of", "supplies"),
        ),
        max_hops=2,
        max_paths=1,
        degree_cap=10,
    )

    assert len(paths) == 1
    assert paths[0].hops == 1
    assert paths[0].relationship_ids == (data["relationships"]["mentions"],)


def test_find_paths_source_anchors_respect_chunk_filter(
    graph_fixture: tuple[Session, dict],
) -> None:
    db, data = graph_fixture
    store = PostgresGraphStore()

    paths = store.find_paths(
        db,
        source_entity_id=data["entities"]["company"],
        target_entity_id=data["entities"]["margin"],
        filters=GraphFilters(
            graph_version=data["graph_version"],
            relation_types=("affects", "part_of", "reports_metric"),
            chunk_ids=(data["chunks"]["chunk_a1"],),
        ),
        max_hops=2,
        max_paths=10,
        degree_cap=10,
    )

    assert tuple(path.relationship_ids for path in paths) == (
        (data["relationships"]["affects"],),
    )
    assert tuple(anchor.chunk_id for anchor in paths[0].source_anchors) == (
        data["chunks"]["chunk_a1"],
    )


def _seed_graph_fixture(db: Session) -> dict:
    suffix = uuid4().hex[:10]
    graph_version = f"test_graph_{suffix}"
    doc_a = f"doc_a_{suffix}"
    doc_b = f"doc_b_{suffix}"
    parent_a = f"parent_a_{suffix}"
    chunk_a1 = f"chunk_a1_{suffix}"
    chunk_a2 = f"chunk_a2_{suffix}"
    chunk_a3 = f"chunk_a3_{suffix}"
    chunk_b1 = f"chunk_b1_{suffix}"

    documents = [
        Document(
            document_id=doc_a,
            title="3M Annual Report",
            source_uri=None,
            file_type="pdf",
            content_hash=f"hash_doc_a_{suffix}",
            language="en",
            metadata_json={},
        ),
        Document(
            document_id=doc_b,
            title="Supplier Note",
            source_uri=None,
            file_type="txt",
            content_hash=f"hash_doc_b_{suffix}",
            language="en",
            metadata_json={},
        ),
    ]
    parent = ParentBlock(
        parent_id=parent_a,
        document_id=doc_a,
        parent_type="page",
        page_start=10,
        page_end=12,
        text="3M operating margin context.",
        child_ids_json=[chunk_a1, chunk_a2, chunk_a3],
        metadata_json={},
    )
    chunks = [
        _chunk(chunk_a1, doc_a, parent_a, 0, "3M affected margins.", page=10),
        _chunk(chunk_a2, doc_a, parent_a, 1, "Segment operating income.", page=11),
        _chunk(chunk_a3, doc_a, parent_a, 2, "Additional 3M discussion.", page=12),
        _chunk(chunk_b1, doc_b, None, 0, "Supplier relationship note.", page=3),
    ]
    db.add_all([*documents, parent, *chunks])
    db.flush()

    index = GraphIndex(
        graph_version=graph_version,
        corpus_version=f"corpus_{suffix}",
        fixture_schema_version="test",
        fixture_hash=f"fixture_{suffix}",
        loader_version="test",
        row_counts_json={},
        status="loaded",
        loaded_at=datetime.now(UTC),
        metadata_json={},
    )
    db.add(index)
    db.flush()

    company = f"company_{suffix}"
    margin = f"margin_{suffix}"
    supplier = f"supplier_{suffix}"
    segment = f"segment_{suffix}"
    entities = [
        _entity(graph_version, company, "3M Company", "company", aliases=("3M", "MMM")),
        _entity(graph_version, margin, "Operating Margin", "metric"),
        _entity(graph_version, supplier, "Key Supplier", "company"),
        _entity(graph_version, segment, "Safety and Industrial Segment", "segment"),
    ]
    neighbor_ids: list[str] = []
    for index_number in range(30):
        neighbor_id = f"neighbor_{index_number}_{suffix}"
        neighbor_ids.append(neighbor_id)
        entities.append(
            _entity(graph_version, neighbor_id, f"Neighbor {index_number}", "topic")
        )
    db.add_all(entities)
    db.flush()

    relationship_ids = {
        "affects": f"rel_affects_{suffix}",
        "mentions": f"rel_mentions_{suffix}",
        "part_of": f"rel_part_of_{suffix}",
        "reports_metric": f"rel_reports_{suffix}",
    }
    relationships = [
        _relationship(
            graph_version,
            relationship_ids["affects"],
            company,
            margin,
            "affects",
            confidence=0.99,
        ),
        _relationship(
            graph_version,
            relationship_ids["mentions"],
            company,
            supplier,
            "mentions",
            confidence=0.80,
        ),
        _relationship(
            graph_version,
            relationship_ids["part_of"],
            company,
            segment,
            "part_of",
            confidence=0.97,
        ),
        _relationship(
            graph_version,
            relationship_ids["reports_metric"],
            segment,
            margin,
            "reports_metric",
            confidence=0.96,
        ),
    ]
    related_relationship_ids: list[str] = []
    for index_number, neighbor_id in enumerate(neighbor_ids):
        relationship_id = f"rel_related_{index_number}_{suffix}"
        related_relationship_ids.append(relationship_id)
        relationships.append(
            _relationship(
                graph_version,
                relationship_id,
                company,
                neighbor_id,
                "related_to",
                confidence=0.70 - (index_number / 1000),
            )
        )
    db.add_all(relationships)
    db.flush()

    entity_anchors = [
        _entity_anchor(graph_version, f"ea_company_1_{suffix}", company, chunk_a1, "3M Company"),
        _entity_anchor(graph_version, f"ea_company_2_{suffix}", company, chunk_a2, "3M"),
        _entity_anchor(graph_version, f"ea_company_3_{suffix}", company, chunk_a3, "MMM"),
        _entity_anchor(graph_version, f"ea_margin_{suffix}", margin, chunk_a1, "margins"),
        _entity_anchor(graph_version, f"ea_supplier_{suffix}", supplier, chunk_b1, "Supplier"),
        _entity_anchor(graph_version, f"ea_segment_{suffix}", segment, chunk_a2, "Segment"),
    ]
    relationship_anchors = [
        _relationship_anchor(
            graph_version,
            f"ra_affects_1_{suffix}",
            relationship_ids["affects"],
            chunk_a1,
            "3M affected margins",
        ),
        _relationship_anchor(
            graph_version,
            f"ra_affects_2_{suffix}",
            relationship_ids["affects"],
            chunk_a2,
            "margin impact",
        ),
        _relationship_anchor(
            graph_version,
            f"ra_mentions_{suffix}",
            relationship_ids["mentions"],
            chunk_b1,
            "supplier relationship",
        ),
        _relationship_anchor(
            graph_version,
            f"ra_mentions_doc_a_{suffix}",
            relationship_ids["mentions"],
            chunk_a3,
            "supplier relationship in annual report",
        ),
        _relationship_anchor(
            graph_version,
            f"ra_part_of_{suffix}",
            relationship_ids["part_of"],
            chunk_a2,
            "part of segment",
        ),
        _relationship_anchor(
            graph_version,
            f"ra_reports_{suffix}",
            relationship_ids["reports_metric"],
            chunk_a2,
            "segment reports operating margin",
        ),
    ]
    for index_number, relationship_id in enumerate(related_relationship_ids):
        relationship_anchors.append(
            _relationship_anchor(
                graph_version,
                f"ra_related_{index_number}_{suffix}",
                relationship_id,
                chunk_a2,
                f"related neighbor {index_number}",
            )
        )
    db.add_all([*entity_anchors, *relationship_anchors])
    db.flush()

    assert db.execute(
        select(GraphEntityRecord).where(GraphEntityRecord.graph_version == graph_version)
    ).scalars().first()

    return {
        "graph_version": graph_version,
        "documents": {"doc_a": doc_a, "doc_b": doc_b},
        "chunks": {
            "chunk_a1": chunk_a1,
            "chunk_a2": chunk_a2,
            "chunk_a3": chunk_a3,
            "chunk_b1": chunk_b1,
        },
        "entities": {
            "company": company,
            "margin": margin,
            "supplier": supplier,
            "segment": segment,
            "neighbors": tuple(neighbor_ids),
        },
        "relationships": {
            **relationship_ids,
            "related": tuple(related_relationship_ids),
        },
    }


def _chunk(
    chunk_id: str,
    document_id: str,
    parent_id: str | None,
    chunk_index: int,
    body: str,
    *,
    page: int,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        parent_id=parent_id,
        chunk_index=chunk_index,
        text=body,
        text_hash=f"hash_{chunk_id}",
        section_title="Test Section",
        page_start=page,
        page_end=page,
        token_count=len(body.split()),
        embedding_model="test",
        embedding_dim=3,
        metadata_json={},
    )


def _entity(
    graph_version: str,
    entity_id: str,
    canonical_name: str,
    entity_type: str,
    *,
    aliases: tuple[str, ...] = (),
) -> GraphEntityRecord:
    return GraphEntityRecord(
        graph_version=graph_version,
        entity_id=entity_id,
        canonical_name=canonical_name,
        canonical_name_norm=" ".join(canonical_name.casefold().split()),
        entity_type=entity_type,
        aliases_json=list(aliases),
        metadata_json={},
    )


def _relationship(
    graph_version: str,
    relationship_id: str,
    source_entity_id: str,
    target_entity_id: str,
    relation_type: str,
    *,
    confidence: float,
) -> GraphRelationshipRecord:
    return GraphRelationshipRecord(
        graph_version=graph_version,
        relationship_id=relationship_id,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relation_type=relation_type,
        confidence=confidence,
        metadata_json={},
    )


def _entity_anchor(
    graph_version: str,
    anchor_id: str,
    entity_id: str,
    chunk_id: str,
    text_span: str,
) -> GraphEntityAnchor:
    return GraphEntityAnchor(
        graph_version=graph_version,
        anchor_id=anchor_id,
        entity_id=entity_id,
        chunk_id=chunk_id,
        text_span=text_span,
        text_span_hash=f"{anchor_id}_hash",
        metadata_json={"fixture": "graph_store"},
    )


def _relationship_anchor(
    graph_version: str,
    anchor_id: str,
    relationship_id: str,
    chunk_id: str,
    text_span: str,
) -> GraphRelationshipAnchor:
    return GraphRelationshipAnchor(
        graph_version=graph_version,
        anchor_id=anchor_id,
        relationship_id=relationship_id,
        chunk_id=chunk_id,
        text_span=text_span,
        text_span_hash=f"{anchor_id}_hash",
        metadata_json={"fixture": "graph_store"},
    )
