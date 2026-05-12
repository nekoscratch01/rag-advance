from __future__ import annotations

import pytest

from atlas.ingestion.structured.chunking import chunk_document_ir
from atlas.ingestion.structured.contracts import (
    ChildChunk,
    DocumentElementIR,
    ParentChunk,
    ParsedDocumentIR,
    SchemaRoutingCard,
    SourceLocator,
    TableColumnIR,
    TableIR,
    content_hash,
)
from atlas.ingestion.structured.tables import build_schema_routing_cards, csv_to_table_ir_and_cards
from atlas.ingestion.structured.tables import should_skip_text_chunking_for_tabular_source


def test_csv_table_and_column_locators_are_canonical_and_complete() -> None:
    result = csv_to_table_ir_and_cards(
        "metric,value\nrevenue,10\ncapex,3\n",
        document_id="doc_csv",
        table_id="tbl_csv",
        source_uri="local:facts.csv",
        table_name="facts",
    )

    table = result.table_ir
    assert isinstance(table, TableIR)
    assert table.source_locator.locator_precision == "table"
    assert table.source_locator.locator_method == "parser"
    assert table.source_locator.locator_confidence == 1.0
    assert table.source_locator.is_exact is True
    assert table.source_locator.document_id == "doc_csv"
    assert table.source_locator.source_uri == "local:facts.csv"
    assert table.source_locator.table_id == "tbl_csv"
    assert table.source_locator.table_range

    column = table.columns[0]
    assert isinstance(column, TableColumnIR)
    assert column.source_locator.locator_precision == "column"
    assert column.source_locator.locator_method == "parser"
    assert column.source_locator.locator_confidence == 1.0
    assert column.source_locator.is_exact is True
    assert column.source_locator.column_locator["column_name"] == "metric"
    assert column.source_locator.column_locator["column_index"] == 0


def test_csv_row_locators_are_metadata_not_text_chunks() -> None:
    result = csv_to_table_ir_and_cards(
        "metric,value\nrevenue,10\ncapex,3\n",
        document_id="doc_csv",
        table_id="tbl_csv",
    )

    metadata = result.table_ir.metadata
    row_locators = metadata["row_locators"]

    assert metadata["raw_rows_text_chunked"] is False
    assert metadata["ordinary_text_chunks_emitted"] is False
    assert metadata["rows_materialized_as_text_chunks"] is False
    assert metadata["cells_materialized"] is False
    assert len(row_locators) == 2
    assert {"row_index", "row_hash"} <= set(row_locators[0])
    assert row_locators[0]["row_locator"] == {
        "row_index": row_locators[0]["row_index"],
        "row_hash": row_locators[0]["row_hash"],
    }
    assert row_locators[0]["source_locator"]["locator_precision"] == "row"
    assert row_locators[0]["ordinary_text_chunk"] is False
    assert "cells" not in row_locators[0]


def test_schema_cards_have_three_text_lanes_and_routing_flags() -> None:
    result = csv_to_table_ir_and_cards(
        "metric,value\nrevenue,10\n",
        document_id="doc_csv",
        table_id="tbl_csv",
        source_uri="local:facts.csv",
    )

    assert result.cards
    for card in result.cards:
        assert isinstance(card, SchemaRoutingCard)
        payload = card.to_payload()
        assert payload["source_derived_text"]
        assert payload["computed_from_source_text"]
        assert payload["inferred_text"]
        assert payload["value_answer_evidence_allowed"] is False
        assert payload["schema_answer_evidence_allowed"] is True
        assert payload["answer_evidence_allowed"] is False
        assert payload["index_object_type"] == "schema_routing"
        assert payload["evidence_role"] == "routing_only"
        assert payload["metadata"]["value_answer_evidence_allowed"] is False
        assert payload["metadata"]["schema_answer_evidence_allowed"] is True


def test_build_schema_routing_cards_preserves_source_identity() -> None:
    original = csv_to_table_ir_and_cards(
        "metric,value\nrevenue,10\n",
        document_id="doc_csv",
        table_id="tbl_csv",
        source_uri="local:facts.csv",
    ).table_ir

    rebuilt = build_schema_routing_cards(original)

    assert rebuilt
    for card in rebuilt:
        assert card.document_id == "doc_csv"
        assert card.table_id == "tbl_csv"
        assert card.source_locator.document_id == "doc_csv"
        assert card.source_locator.table_id == "tbl_csv"
        assert card.source_locator.source_uri == "local:facts.csv"


def test_chunking_emits_canonical_parent_child_locator_and_hash_fields() -> None:
    document = ParsedDocumentIR(
        document_id="doc_text",
        path="facts.md",
        text="Revenue increased.\n\nCapex declined.",
        file_type="md",
        elements=[
            DocumentElementIR(
                element_id="el_1",
                element_type="text",
                text="Revenue increased.\n\nCapex declined.",
                source_locator=SourceLocator(
                    source_uri="local:facts.md",
                    document_id="doc_text",
                    page_number=2,
                    page_start=2,
                    page_end=2,
                    element_id="el_1",
                ),
            )
        ],
    )

    result = chunk_document_ir(document, child_target_tokens=8, child_overlap_tokens=0)
    parent = result.parent_chunks[0]
    child = result.child_chunks[0]

    assert isinstance(parent, ParentChunk)
    assert isinstance(child, ChildChunk)
    assert parent.include_in_main_index is False
    assert parent.source_locator.document_id == "doc_text"
    assert parent.source_element_ids == ("el_1",)
    assert parent.content_hash == content_hash(parent.text)
    assert child.source_locator.document_id == "doc_text"
    assert child.source_element_ids == ("el_1",)
    assert child.content_hash == content_hash(child.text)


def test_tabular_document_ir_is_rejected_by_text_chunking() -> None:
    document = ParsedDocumentIR(
        document_id="doc_csv",
        text="metric,value\nrevenue,10\n",
        file_type="csv",
    )

    with pytest.raises(ValueError, match="tabular_document_ir_must_use_structured_table_worker"):
        chunk_document_ir(document)


def test_html_document_ir_is_rejected_by_text_chunking() -> None:
    document = ParsedDocumentIR(
        document_id="doc_html",
        text="<table><tr><td>revenue</td><td>10</td></tr></table>",
        file_type="html",
    )

    with pytest.raises(ValueError, match="tabular_document_ir_must_use_structured_table_worker"):
        chunk_document_ir(document)


def test_table_like_elements_do_not_fallback_to_raw_text_chunks() -> None:
    document = ParsedDocumentIR(
        document_id="doc_table_like",
        path="facts.md",
        text="metric,value\nrevenue,10\n",
        file_type="md",
        elements=[
            DocumentElementIR(
                element_id="tbl_1",
                element_type="html_table",
                text="metric value\nrevenue 10",
            ),
            DocumentElementIR(
                element_id="row_1",
                element_type="csv_row",
                text="revenue,10",
            ),
        ],
    )

    result = chunk_document_ir(document)

    assert result.parent_chunks == ()
    assert result.child_chunks == ()


@pytest.mark.parametrize("file_type", ["html", "htm", ".html", ".htm"])
def test_tabular_helper_covers_html_file_types(file_type: str) -> None:
    assert should_skip_text_chunking_for_tabular_source(file_type) is True
