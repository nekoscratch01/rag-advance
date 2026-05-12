from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas.ingestion.contracts import (
    DEFAULT_STRUCTURED_ARTIFACT_ENVELOPE_VERSION,
    LoadedDocument,
    LoadedPage,
    StructuredArtifact,
)
from atlas.ingestion.structured.contracts import (
    ChildChunk,
    ParentChunk,
    ParsedDocumentIR,
    ProvenancePolicy,
    SourceLocator,
    TableCard,
    TableCellIR,
    TableRowIR,
    content_hash,
    parsed_document_from_loaded_document,
    stable_id,
)


def test_source_locator_defaults_are_serializable_and_canonical() -> None:
    locator = SourceLocator(
        source_uri="local:facts.csv",
        document_id="doc_1",
        storage_ref="postgres:structured_artifacts/art_1",
        storage_format="postgres_jsonb",
        table_id="tbl_1",
        table_range="A1:C3",
        row_locator={"row_index": 2, "row_hash": "abc"},
        column_locator={"column_index": 1, "column_name": "revenue"},
        locator_precision="cell",
        locator_confidence=1.0,
        is_exact=True,
        locator_method="csv_header_cell_ref",
    )

    payload = locator.to_payload()

    assert payload["locator_precision"] == "cell"
    assert payload["locator_confidence"] == 1.0
    assert payload["is_exact"] is True
    assert payload["locator_method"] == "csv_header_cell_ref"
    assert payload["storage_ref"] == "postgres:structured_artifacts/art_1"
    assert payload["storage_format"] == "postgres_jsonb"
    assert payload["row_locator"] == {"row_hash": "abc", "row_index": 2}
    assert payload["column_locator"] == {"column_index": 1, "column_name": "revenue"}
    json.dumps(payload)


def test_provenance_policy_defaults_are_layered_not_cell_required() -> None:
    policy = ProvenancePolicy()
    payload = policy.to_payload()

    assert payload["require_storage_locator"] is True
    assert payload["require_table_locator"] is True
    assert payload["require_column_locator"] is True
    assert payload["require_row_locator_for_tables"] is False
    assert payload["require_cell_locator_for_tables"] is False
    assert payload["row_locator_policy"] == "required_when_materialized"
    assert payload["cell_locator_policy"] == "required_when_materialized"


def test_stable_id_and_content_hash_are_deterministic() -> None:
    left = {"b": 2, "a": [{"z": "same"}]}
    right = {"a": [{"z": "same"}], "b": 2}

    assert stable_id("v4", left) == stable_id("v4", right)
    assert content_hash(left) == content_hash(right)
    assert stable_id("v4", left) != stable_id("v4", {"a": "different"})


def test_loaded_document_round_trips_through_parsed_document_ir_projection() -> None:
    loaded = LoadedDocument(
        path=Path("sample.md"),
        title="Sample",
        text="Page one\n\nPage two",
        file_type="md",
        language="en",
        pages=[
            LoadedPage(page_number=1, text="Page one"),
            LoadedPage(page_number=2, text="Page two"),
        ],
    )

    parsed = parsed_document_from_loaded_document(
        loaded,
        document_id="doc_1",
        source_uri="local:sample.md",
        parser_name="unit",
    )
    projected = parsed.to_loaded_document()

    assert isinstance(parsed, ParsedDocumentIR)
    assert parsed.document_id == "doc_1"
    assert parsed.source_locator.document_id == "doc_1"
    assert [page.page_number for page in projected.pages] == [1, 2]
    assert [page.text for page in projected.pages] == ["Page one", "Page two"]
    assert projected.title == loaded.title


def test_parent_and_child_chunks_accept_canonical_payload_fields() -> None:
    locator = SourceLocator(document_id="doc_1", page_number=3, locator_precision="page")
    parent = ParentChunk(
        parent_chunk_id="par_1",
        document_id="doc_1",
        text="Parent text",
        section_title="Results",
        page_start=3,
        page_end=3,
        token_count=2,
        child_ids=["chk_1"],
        source_locator=locator,
        source_element_ids=["el_1"],
        metadata={"chunk_role": "parent"},
    )
    child = ChildChunk(
        child_chunk_id="chk_1",
        parent_chunk_id="par_1",
        document_id="doc_1",
        text="Child text",
        chunk_index=0,
        child_index=0,
        source_locator=locator,
        source_element_ids=["el_1"],
        metadata={"chunk_role": "child"},
    )

    parent_payload = parent.to_payload()
    child_payload = child.to_payload()

    assert parent_payload["id"] == "par_1"
    assert parent_payload["child_ids"] == ["chk_1"]
    assert parent_payload["source_locator"]["locator_precision"] == "page"
    assert parent_payload["source_element_ids"] == ["el_1"]
    assert parent_payload["content_hash"] == content_hash("Parent text")
    assert child_payload["id"] == "chk_1"
    assert child_payload["parent_chunk_id"] == "par_1"
    assert child_payload["index_policy"] == "ranked_child"
    assert child_payload["content_hash"] == content_hash("Child text")


def test_table_card_flags_are_routing_only_and_csv_payload_compatible() -> None:
    direct = TableCard(
        card_id="card_1",
        table_id="tbl_1",
        document_id="doc_1",
        routing_text="CSV table schema for tbl_1.",
    )
    direct_payload = direct.to_payload()

    assert direct_payload["value_answer_evidence_allowed"] is False
    assert direct_payload["schema_answer_evidence_allowed"] is True
    assert direct_payload["answer_evidence_allowed"] is False
    assert direct_payload["index_object_type"] == "schema_routing"
    assert direct_payload["evidence_role"] == "routing_only"
    assert direct_payload["source_derived_text"] == "CSV table schema for tbl_1."
    assert direct_payload["computed_from_source_text"] == "CSV table schema for tbl_1."
    assert direct_payload["inferred_text"] == "CSV table schema for tbl_1."

    from atlas.ingestion.structured.tables import csv_to_table_ir_and_cards

    result = csv_to_table_ir_and_cards(
        "metric,value\nrevenue,10\n",
        document_id="doc_1",
        table_id="tbl_1",
    )
    csv_payload = result.table_card.to_payload()

    assert csv_payload["card_type"] == "table_schema"
    assert csv_payload["index_object_type"] == "schema_routing"
    assert csv_payload["evidence_role"] == "routing_only"
    assert csv_payload["value_answer_evidence_allowed"] is False
    assert csv_payload["schema_answer_evidence_allowed"] is True
    assert csv_payload["answer_evidence_allowed"] is False


def test_row_and_cell_ir_are_optional_materialization_contracts() -> None:
    row = TableRowIR(
        table_id="tbl_1",
        document_id="doc_1",
        row_index=0,
        values={"metric": "revenue", "value": "10"},
    )
    cell = TableCellIR(
        table_id="tbl_1",
        document_id="doc_1",
        row_id=row.row_id,
        row_index=0,
        row_hash=row.row_hash,
        column_name="value",
        cell_ref="B2",
        value="10",
        source_locator=SourceLocator(
            table_id="tbl_1",
            row_index=0,
            column_locator={"column_name": "value"},
            cell_ref="B2",
            locator_precision="cell",
            is_exact=True,
        ),
    )

    assert row.materialized is False
    assert row.row_hash == content_hash({"metric": "revenue", "value": "10"})
    assert cell.materialized is False
    assert cell.to_payload()["source_locator"]["cell_ref"] == "B2"


def test_structured_artifact_envelope_rejects_non_json_payload_values() -> None:
    artifact = StructuredArtifact(
        "schema_routing_card",
        {"columns": ("metric", "value"), "row_count": 1},
        metadata={"source": "unit"},
    )
    envelope = artifact.to_envelope_payload()

    assert artifact.envelope_version == DEFAULT_STRUCTURED_ARTIFACT_ENVELOPE_VERSION
    assert envelope["envelope_version"] == DEFAULT_STRUCTURED_ARTIFACT_ENVELOPE_VERSION
    assert artifact.payload == {"columns": ["metric", "value"], "row_count": 1}
    assert envelope["payload"] == artifact.payload
    json.dumps(envelope)

    with pytest.raises(TypeError):
        StructuredArtifact("bad", ["not", "an", "object"])  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        StructuredArtifact("bad", {"not_json": object()})
