from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas.db.models import TableCell, TableRow
from atlas.ingestion.contracts import (
    DEFAULT_STRUCTURED_ARTIFACT_ENVELOPE_VERSION,
    StructuredArtifact,
)
from atlas.ingestion.structured.writer import (
    StructuredArtifactValidationError,
    StructuredArtifactWriter,
    UnsupportedStructuredArtifactSchemaVersion,
)


class _FakeDB:
    def __init__(self) -> None:
        self.merged = []
        self.flush_count = 0

    def merge(self, value):
        self.merged.append(value)
        return value

    def flush(self) -> None:
        self.flush_count += 1


def test_default_structured_artifact_envelope_version_passes_schema_validation(
    tmp_path: Path,
) -> None:
    writer = StructuredArtifactWriter(output_dir=tmp_path, write_csv=False)

    result = writer.write(
        None,
        artifacts=[
            StructuredArtifact(
                artifact_type="table",
                payload={
                    "table_id": "tbl_default_envelope",
                    "columns": ["metric", "value"],
                },
            )
        ],
        materialization_policy="none",
        allow_partial=False,
    )

    raw_record = _read_jsonl(result.artifact_dir / "raw_artifacts.jsonl")[0]
    assert result.status == "completed"
    assert result.artifact_counts["table_assets"] == 1
    assert raw_record["schema_version"] == DEFAULT_STRUCTURED_ARTIFACT_ENVELOPE_VERSION
    assert raw_record["envelope_version"] == DEFAULT_STRUCTURED_ARTIFACT_ENVELOPE_VERSION


def test_structured_artifact_envelope_version_passes_schema_validation(
    tmp_path: Path,
) -> None:
    writer = StructuredArtifactWriter(output_dir=tmp_path, write_csv=False)

    result = writer.write(
        None,
        artifacts=[_table_artifact()],
        materialization_policy="none",
        allow_partial=False,
    )

    assert result.status == "completed"
    assert result.artifact_counts["table_assets"] == 1


def test_payload_schema_version_cannot_be_hidden_by_good_envelope_version(
    tmp_path: Path,
) -> None:
    writer = StructuredArtifactWriter(output_dir=tmp_path, write_csv=False)

    with pytest.raises(UnsupportedStructuredArtifactSchemaVersion):
        writer.write(
            None,
            artifacts=[
                {
                    "artifact_type": "table",
                    "envelope_version": "v4.phase1",
                    "payload": {
                        "schema_version": "v4.unsupported",
                        "table_id": "tbl_bad_schema",
                        "columns": ["metric", "value"],
                    },
                }
            ],
            materialization_policy="none",
            allow_partial=False,
        )


def test_conflicting_schema_version_declarations_fail_fast(tmp_path: Path) -> None:
    writer = StructuredArtifactWriter(output_dir=tmp_path, write_csv=False)

    with pytest.raises(StructuredArtifactValidationError):
        writer.write(
            None,
            artifacts=[
                {
                    "artifact_type": "table",
                    "schema_version": "v4.structured.v1",
                    "envelope_version": "v4.phase1",
                    "payload": {
                        "table_id": "tbl_conflicting_schema",
                        "columns": ["metric", "value"],
                    },
                }
            ],
            materialization_policy="none",
            allow_partial=False,
        )


def test_default_envelope_conflicting_payload_schema_version_fails_fast(
    tmp_path: Path,
) -> None:
    writer = StructuredArtifactWriter(output_dir=tmp_path, write_csv=False)

    with pytest.raises(StructuredArtifactValidationError):
        writer.write(
            None,
            artifacts=[
                StructuredArtifact(
                    artifact_type="table",
                    payload={
                        "schema_version": "v4.phase1",
                        "table_id": "tbl_default_conflict",
                        "columns": ["metric", "value"],
                    },
                )
            ],
            materialization_policy="none",
            allow_partial=False,
        )


def test_unsupported_non_routing_artifact_raises_when_partial_disallowed(
    tmp_path: Path,
) -> None:
    writer = StructuredArtifactWriter(output_dir=tmp_path, write_csv=False)

    with pytest.raises(StructuredArtifactValidationError):
        writer.write(
            None,
            artifacts=[
                StructuredArtifact(
                    artifact_type="unrecognized_payload",
                    payload={"title": "not a table or fact"},
                    envelope_version="v4.phase1",
                )
            ],
            materialization_policy="none",
            artifact_id="batch_invalid",
            allow_partial=False,
        )

    manifest = json.loads(
        (tmp_path / "batch_invalid" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert any(error["code"] == "unsupported_artifact_type" for error in manifest["errors"])


def test_unsupported_non_routing_artifact_can_return_failed_partial_result(
    tmp_path: Path,
) -> None:
    writer = StructuredArtifactWriter(output_dir=tmp_path, write_csv=False)

    result = writer.write(
        None,
        artifacts=[
            StructuredArtifact(
                artifact_type="unrecognized_payload",
                payload={"title": "not a table or fact"},
                envelope_version="v4.phase1",
            )
        ],
        materialization_policy="none",
        artifact_id="batch_partial_invalid",
        allow_partial=True,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.status == "failed"
    assert result.errors
    assert manifest["status"] == "failed"
    assert any(error["code"] == "unsupported_artifact_type" for error in result.errors)


def test_raw_artifacts_jsonl_contains_complete_envelope_fields(tmp_path: Path) -> None:
    writer = StructuredArtifactWriter(output_dir=tmp_path, write_csv=False)
    artifact = _table_artifact(
        artifact_id="orig_table_1",
        content_hash="hash_table_1",
        source_locator={"source_path": "/tmp/source.csv", "page_number": 3},
        provenance_policy={"materialization_policy": "facts"},
        schema_routing_card={"card_id": "route_1", "table_id": "tbl_writer"},
        artifact_manifest={"manifest_id": "manifest_1", "artifact_ids": ["orig_table_1"]},
        metadata={"parser": "unit"},
    )

    result = writer.write(None, artifacts=[artifact], materialization_policy="none")

    raw_record = _read_jsonl(result.artifact_dir / "raw_artifacts.jsonl")[0]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert raw_record["artifact_id"] == "orig_table_1"
    assert raw_record["content_hash"] == "hash_table_1"
    assert raw_record["envelope_version"] == "v4.phase1"
    assert raw_record["source_locator"]["page_number"] == 3
    assert raw_record["provenance_policy"]["materialization_policy"] == "facts"
    assert raw_record["schema_routing_card"]["card_id"] == "route_1"
    assert raw_record["artifact_manifest"]["manifest_id"] == "manifest_1"
    assert raw_record["metadata"] == {"parser": "unit"}
    assert raw_record["payload"]["table_id"] == "tbl_writer"
    assert manifest["raw_artifact_envelope_count"] == 1
    assert manifest["raw_artifact_envelopes"][0]["artifact_id"] == "orig_table_1"
    assert manifest["raw_artifact_envelopes"][0]["content_hash"] == "hash_table_1"
    assert manifest["raw_artifact_envelopes"][0]["envelope_version"] == "v4.phase1"


def test_schema_routing_card_is_not_materialized_as_table_asset(tmp_path: Path) -> None:
    writer = StructuredArtifactWriter(output_dir=tmp_path, write_csv=False)
    routing_card = StructuredArtifact(
        artifact_type="schema_routing_card",
        payload={
            "card_id": "route_1",
            "card_type": "table_schema",
            "source_type": "schema_routing_card",
            "table_id": "tbl_route",
            "columns": ["metric", "value"],
        },
        artifact_id="route_1",
        envelope_version="v4.phase1",
    )

    result = writer.write(None, artifacts=[routing_card], materialization_policy="none")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.artifact_counts["raw_artifacts"] == 1
    assert result.artifact_counts["table_assets"] == 0
    assert not (result.artifact_dir / "table_assets.jsonl").exists()
    assert manifest["schema_routing_cards"][0]["card_id"] == "route_1"


def test_conflicting_dedupe_records_manifest_warning(tmp_path: Path) -> None:
    writer = StructuredArtifactWriter(output_dir=tmp_path, write_csv=False)
    first = _table_artifact(title="First title")
    second = _table_artifact(title="Second title")

    result = writer.write(None, artifacts=[first, second], materialization_policy="none")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.artifact_counts["raw_artifacts"] == 2
    assert result.artifact_counts["table_assets"] == 1
    assert any(warning["code"] == "dedupe_conflict" for warning in result.warnings)
    assert any(warning["code"] == "dedupe_conflict" for warning in manifest["warnings"])


@pytest.mark.parametrize(
    ("policy", "expected_rows", "expected_cells", "expect_row_objects"),
    [
        ("none", 0, 0, False),
        ("facts", 0, 0, False),
        ("full", 1, 1, True),
    ],
)
def test_materialization_policy_controls_rows_and_cells_counts(
    tmp_path: Path,
    policy: str,
    expected_rows: int,
    expected_cells: int,
    expect_row_objects: bool,
) -> None:
    writer = StructuredArtifactWriter(output_dir=tmp_path, write_csv=False)
    db = _FakeDB()

    result = writer.write(
        db,
        artifacts=[_table_artifact()],
        materialization_policy=policy,
        artifact_id=f"batch_{policy}",
    )

    assert result.materialized_counts["structured_artifacts"] == 1
    assert result.materialized_counts["table_rows"] == expected_rows
    assert result.materialized_counts["table_cells"] == expected_cells
    assert any(isinstance(record, TableRow) for record in db.merged) is expect_row_objects
    assert any(isinstance(record, TableCell) for record in db.merged) is expect_row_objects


def _table_artifact(
    *,
    artifact_id: str = "orig_table",
    content_hash: str = "hash_table",
    title: str = "Writer Table",
    source_locator: dict | None = None,
    provenance_policy: dict | None = None,
    schema_routing_card: dict | None = None,
    artifact_manifest: dict | None = None,
    metadata: dict | None = None,
) -> StructuredArtifact:
    return StructuredArtifact(
        artifact_type="table",
        payload={
            "table_id": "tbl_writer",
            "title": title,
            "columns": [
                {"column_id": "col_metric", "column_index": 0, "name": "metric"},
                {"column_id": "col_value", "column_index": 1, "name": "value"},
            ],
            "rows": [
                {"row_id": "row_revenue", "row_index": 0, "row_label": "Revenue"},
            ],
            "cells": [
                {
                    "cell_id": "cell_revenue_value",
                    "row_id": "row_revenue",
                    "row_index": 0,
                    "column_id": "col_value",
                    "column_index": 1,
                    "value": "10",
                    "numeric_value": 10,
                },
            ],
        },
        artifact_id=artifact_id,
        content_hash=content_hash,
        source_locator=source_locator,
        provenance_policy=provenance_policy,
        schema_routing_card=schema_routing_card,
        artifact_manifest=artifact_manifest,
        envelope_version="v4.phase1",
        metadata=metadata or {},
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
