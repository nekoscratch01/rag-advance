from qdrant_client import models

from atlas.retrieval.retrievers.bm25 import _build_filter as build_bm25_filter
from atlas.retrieval.retrievers.dense import _build_filter as build_dense_filter


def _must_conditions(qdrant_filter):
    assert qdrant_filter is not None
    return list(qdrant_filter.must or ())


def test_metadata_filter_ignores_unsupported_payload_keys() -> None:
    qdrant_filter = build_dense_filter(
        {
            "company": "3M",
            "metric": "capital_expenditure",
            "document_ids": ["doc_1"],
        }
    )

    conditions = _must_conditions(qdrant_filter)

    assert [condition.key for condition in conditions] == ["document_id"]


def test_metadata_filter_preserves_scalar_document_id_and_numeric_page_values() -> None:
    qdrant_filter = build_bm25_filter(
        {
            "document_ids": ["doc_1"],
            "section_name": "Item 1A. Risk Factors",
            "page_start": [60, 61],
        }
    )

    conditions = _must_conditions(qdrant_filter)
    by_key = {condition.key: condition for condition in conditions}

    assert set(by_key) == {"document_id", "section_title", "page_start"}
    assert by_key["document_id"].match == models.MatchAny(any=["doc_1"])
    assert by_key["section_title"].match == models.MatchValue(value="Item 1A. Risk Factors")
    assert by_key["page_start"].match == models.MatchAny(any=[60, 61])
