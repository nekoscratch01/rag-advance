from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sqlalchemy.orm import Session

from atlas.core.ids import new_id
from atlas.db.models import (
    FinancialFact,
    FinancialFactCell,
    StructuredArtifactRecord,
    TableAsset,
    TableCell,
    TableColumn,
    TableProfile,
    TableRow,
    utcnow,
)
from atlas.ingestion.contracts import (
    DEFAULT_STRUCTURED_ARTIFACT_ENVELOPE_VERSION,
    StructuredArtifact as ExtractedStructuredArtifact,
)


STRUCTURED_ARTIFACT_SCHEMA_VERSION = "atlas.v4.structured.v1"
# Envelope versions are validated as version declarations, including the legacy default.
SUPPORTED_SCHEMA_VERSIONS = frozenset(
    {
        STRUCTURED_ARTIFACT_SCHEMA_VERSION,
        "v4.structured.v1",
        "v4.phase1",
        DEFAULT_STRUCTURED_ARTIFACT_ENVELOPE_VERSION,
    }
)
MATERIALIZATION_POLICIES = frozenset({"none", "facts", "full"})
MaterializationPolicy = Literal["none", "facts", "full"]


class StructuredArtifactWriterError(RuntimeError):
    pass


class UnsupportedStructuredArtifactSchemaVersion(StructuredArtifactWriterError):
    pass


class StructuredArtifactValidationError(StructuredArtifactWriterError):
    pass


class StructuredArtifactMaterializationError(StructuredArtifactWriterError):
    pass


@dataclass(frozen=True)
class ManifestIssue:
    severity: str
    code: str
    message: str
    path: str | None = None
    artifact_index: int | None = None
    artifact_type: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.path is not None:
            payload["path"] = self.path
        if self.artifact_index is not None:
            payload["artifact_index"] = self.artifact_index
        if self.artifact_type is not None:
            payload["artifact_type"] = self.artifact_type
        return payload


@dataclass(frozen=True)
class StructuredArtifactWriteResult:
    artifact_id: str
    status: str
    artifact_dir: Path
    manifest_path: Path
    artifact_counts: dict[str, int]
    materialized_counts: dict[str, int]
    warnings: list[dict[str, Any]]
    errors: list[dict[str, Any]]


@dataclass
class _NormalizedArtifacts:
    raw_artifacts: list[dict[str, Any]]
    table_assets: list[dict[str, Any]]
    table_columns: list[dict[str, Any]]
    table_profiles: list[dict[str, Any]]
    table_rows: list[dict[str, Any]]
    table_cells: list[dict[str, Any]]
    financial_facts: list[dict[str, Any]]
    financial_fact_cells: list[dict[str, Any]]
    warnings: list[ManifestIssue]
    errors: list[ManifestIssue]


@dataclass(frozen=True)
class _ArtifactParts:
    artifact_type: str
    payload: dict[str, Any]
    schema_version: str | None
    schema_version_declarations: tuple["_SchemaVersionDeclaration", ...]
    envelope_version: str | None
    envelope: dict[str, Any]


@dataclass(frozen=True)
class _SchemaVersionDeclaration:
    path: str
    value: str


class StructuredArtifactManifestWriter:
    def write(self, manifest: Mapping[str, Any], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(path, dict(manifest))
        return path


class StructuredArtifactWriter:
    def __init__(
        self,
        *,
        output_dir: Path | str,
        schema_version: str = STRUCTURED_ARTIFACT_SCHEMA_VERSION,
        supported_schema_versions: set[str] | frozenset[str] = SUPPORTED_SCHEMA_VERSIONS,
        manifest_writer: StructuredArtifactManifestWriter | None = None,
        write_csv: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.schema_version = schema_version
        self.supported_schema_versions = frozenset(supported_schema_versions)
        self.manifest_writer = manifest_writer or StructuredArtifactManifestWriter()
        self.write_csv = write_csv

    def write(
        self,
        db: Session | None,
        *,
        artifacts: Sequence[ExtractedStructuredArtifact | Mapping[str, Any]],
        document_id: str | None = None,
        ingestion_run_id: str | None = None,
        materialization_policy: MaterializationPolicy = "facts",
        metadata: Mapping[str, Any] | None = None,
        artifact_id: str | None = None,
        allow_partial: bool = True,
    ) -> StructuredArtifactWriteResult:
        policy = _validate_materialization_policy(materialization_policy)
        self._validate_schema_versions(artifacts)

        resolved_artifact_id = artifact_id or new_id("sar")
        artifact_dir = self.output_dir / resolved_artifact_id
        normalized = self._normalize_artifacts(
            artifacts,
            artifact_id=resolved_artifact_id,
            document_id=document_id,
        )
        if db is None and policy != "none":
            normalized.warnings.append(
                ManifestIssue(
                    severity="warning",
                    code="db_session_missing",
                    message="Materialization skipped because no SQLAlchemy session was provided.",
                )
            )

        artifact_counts = _artifact_counts(normalized)
        status = _status_for(normalized)
        file_entries = self._write_artifact_files(artifact_dir, normalized)
        raw_artifact_path = _path_from_file_entries(file_entries, "raw_artifacts", "jsonl")
        payload_hash = _sha256_file(Path(raw_artifact_path)) if raw_artifact_path else None
        materialized_counts = _empty_materialized_counts()
        manifest_path = artifact_dir / "manifest.json"

        if normalized.errors and not allow_partial:
            manifest = self._manifest(
                artifact_id=resolved_artifact_id,
                artifact_dir=artifact_dir,
                manifest_path=manifest_path,
                document_id=document_id,
                ingestion_run_id=ingestion_run_id,
                materialization_policy=policy,
                status="failed",
                files=file_entries,
                artifact_counts=artifact_counts,
                materialized_counts=materialized_counts,
                warnings=normalized.warnings,
                errors=normalized.errors,
                metadata=metadata,
            )
            self.manifest_writer.write(manifest, manifest_path)
            raise StructuredArtifactValidationError("structured_artifact_validation_failed")

        try:
            if db is not None:
                materialized_counts = _expected_materialized_counts(policy, normalized)
                self._materialize(
                    db,
                    normalized=normalized,
                    artifact_id=resolved_artifact_id,
                    document_id=document_id,
                    ingestion_run_id=ingestion_run_id,
                    materialization_policy=policy,
                    status=status,
                    artifact_dir=artifact_dir,
                    manifest_path=manifest_path,
                    raw_artifacts_path=Path(raw_artifact_path) if raw_artifact_path else None,
                    payload_hash=payload_hash,
                    artifact_counts=artifact_counts,
                    materialized_counts=materialized_counts,
                    metadata=metadata,
                )
        except Exception as exc:
            materialization_error = ManifestIssue(
                severity="error",
                code="materialization_failed",
                message=str(exc),
            )
            failed_errors = [*normalized.errors, materialization_error]
            manifest = self._manifest(
                artifact_id=resolved_artifact_id,
                artifact_dir=artifact_dir,
                manifest_path=manifest_path,
                document_id=document_id,
                ingestion_run_id=ingestion_run_id,
                materialization_policy=policy,
                status="failed",
                files=file_entries,
                artifact_counts=artifact_counts,
                materialized_counts=_empty_materialized_counts(),
                warnings=normalized.warnings,
                errors=failed_errors,
                metadata=metadata,
            )
            self.manifest_writer.write(manifest, manifest_path)
            raise StructuredArtifactMaterializationError(
                "structured_artifact_materialization_failed"
            ) from exc

        manifest = self._manifest(
            artifact_id=resolved_artifact_id,
            artifact_dir=artifact_dir,
            manifest_path=manifest_path,
            document_id=document_id,
            ingestion_run_id=ingestion_run_id,
            materialization_policy=policy,
            status=status,
            files=file_entries,
            artifact_counts=artifact_counts,
            materialized_counts=materialized_counts,
            warnings=normalized.warnings,
            errors=normalized.errors,
            metadata=metadata,
        )
        self.manifest_writer.write(manifest, manifest_path)
        return StructuredArtifactWriteResult(
            artifact_id=resolved_artifact_id,
            status=status,
            artifact_dir=artifact_dir,
            manifest_path=manifest_path,
            artifact_counts=artifact_counts,
            materialized_counts=materialized_counts,
            warnings=[issue.as_dict() for issue in normalized.warnings],
            errors=[issue.as_dict() for issue in normalized.errors],
        )

    def mark_batch_orphaned(
        self,
        db: Session | None,
        *,
        write_result: StructuredArtifactWriteResult,
        message: str,
        code: str = "ingestion_failed_after_structured_artifact_write",
    ) -> None:
        issue = ManifestIssue(
            severity="error",
            code=code,
            message=message,
        )
        manifest = _read_json_object(write_result.manifest_path)
        if not manifest:
            manifest = {
                "schema_version": self.schema_version,
                "artifact_id": write_result.artifact_id,
                "artifact_type": "structured_artifact_batch",
                "manifest_path": str(write_result.manifest_path),
                "artifact_counts": write_result.artifact_counts,
                "materialized_counts": write_result.materialized_counts,
                "warnings": write_result.warnings,
                "errors": write_result.errors,
            }
        manifest["status"] = "orphaned"
        manifest["updated_at"] = utcnow().isoformat()
        manifest["errors"] = [
            *_issue_payloads(manifest.get("errors")),
            issue.as_dict(),
        ]
        metadata = _as_dict(manifest.get("metadata"))
        metadata["orphaned"] = True
        metadata["orphan_reason"] = message
        manifest["metadata"] = metadata
        self.manifest_writer.write(manifest, write_result.manifest_path)
        _mark_structured_artifact_record_orphaned(
            db,
            artifact_id=write_result.artifact_id,
            issue=issue,
        )

    def _validate_schema_versions(
        self,
        artifacts: Sequence[ExtractedStructuredArtifact | Mapping[str, Any]],
    ) -> None:
        for artifact_index, artifact in enumerate(artifacts):
            parts = _artifact_parts(artifact)
            declarations = parts.schema_version_declarations
            if not declarations:
                raise UnsupportedStructuredArtifactSchemaVersion(
                    "Missing structured artifact schema version declaration "
                    f"at artifact index {artifact_index} ({parts.artifact_type}). "
                    f"Supported versions: {sorted(self.supported_schema_versions)}"
                )
            unsupported = [
                declaration
                for declaration in declarations
                if declaration.value not in self.supported_schema_versions
            ]
            if unsupported:
                raise UnsupportedStructuredArtifactSchemaVersion(
                    "Unsupported structured artifact schema version declaration(s) "
                    f"{_schema_version_declaration_payloads(unsupported)!r} "
                    f"at artifact index {artifact_index} ({parts.artifact_type}). "
                    f"Supported versions: {sorted(self.supported_schema_versions)}"
                )
            declared_values = {declaration.value for declaration in declarations}
            if len(declared_values) > 1:
                raise StructuredArtifactValidationError(
                    "Conflicting structured artifact schema version declarations "
                    f"{_schema_version_declaration_payloads(declarations)!r} "
                    f"at artifact index {artifact_index} ({parts.artifact_type})."
                )

    def _normalize_artifacts(
        self,
        artifacts: Sequence[ExtractedStructuredArtifact | Mapping[str, Any]],
        *,
        artifact_id: str,
        document_id: str | None,
    ) -> _NormalizedArtifacts:
        normalized = _NormalizedArtifacts(
            raw_artifacts=[],
            table_assets=[],
            table_columns=[],
            table_profiles=[],
            table_rows=[],
            table_cells=[],
            financial_facts=[],
            financial_fact_cells=[],
            warnings=[],
            errors=[],
        )
        for artifact_index, artifact in enumerate(artifacts):
            parts = _artifact_parts(artifact)
            normalized.raw_artifacts.append(
                _raw_artifact_record(
                    parts,
                    artifact_index=artifact_index,
                    batch_artifact_id=artifact_id,
                )
            )
            self._normalize_payload(
                normalized,
                artifact_index=artifact_index,
                artifact_type=parts.artifact_type,
                payload=parts.payload,
                artifact_id=artifact_id,
                document_id=document_id,
                envelope_metadata=_envelope_metadata(parts),
            )

        normalized.table_assets = _dedupe_records(
            normalized.table_assets,
            ("table_id",),
            record_type="table_assets",
            warnings=normalized.warnings,
        )
        normalized.table_columns = _dedupe_records(
            normalized.table_columns,
            ("column_id",),
            record_type="table_columns",
            warnings=normalized.warnings,
        )
        normalized.table_profiles = _dedupe_records(
            normalized.table_profiles,
            ("table_id",),
            record_type="table_profiles",
            warnings=normalized.warnings,
        )
        normalized.table_rows = _dedupe_records(
            normalized.table_rows,
            ("row_id",),
            record_type="table_rows",
            warnings=normalized.warnings,
        )
        normalized.table_cells = _dedupe_records(
            normalized.table_cells,
            ("cell_id",),
            record_type="table_cells",
            warnings=normalized.warnings,
        )
        normalized.financial_facts = _dedupe_records(
            normalized.financial_facts,
            ("fact_id",),
            record_type="financial_facts",
            warnings=normalized.warnings,
        )
        normalized.financial_fact_cells = _dedupe_records(
            normalized.financial_fact_cells,
            ("fact_id", "cell_id"),
            record_type="financial_fact_cells",
            warnings=normalized.warnings,
        )
        return normalized

    def _normalize_payload(
        self,
        normalized: _NormalizedArtifacts,
        *,
        artifact_index: int,
        artifact_type: str,
        payload: dict[str, Any],
        artifact_id: str,
        document_id: str | None,
        envelope_metadata: dict[str, Any],
    ) -> None:
        artifact_kind = artifact_type.lower()
        if _is_schema_routing_artifact(artifact_kind, payload):
            return
        if _is_noop_artifact(artifact_kind, payload):
            return

        table_payloads = _records_from(payload.get("tables") or payload.get("table_assets"))
        fact_payloads = _records_from(payload.get("financial_facts") or payload.get("facts"))

        if artifact_kind in {"table", "table_asset", "table_assets"}:
            table_payloads = [payload]
        elif artifact_kind in {"financial_fact", "financial_facts", "fact", "facts"}:
            fact_payloads = [payload]
        elif not table_payloads and not fact_payloads:
            if _looks_like_table(payload):
                table_payloads = [payload]
            elif _looks_like_fact(payload):
                fact_payloads = [payload]

        if not table_payloads and not fact_payloads:
            normalized.errors.append(
                ManifestIssue(
                    severity="error",
                    code="unsupported_artifact_type",
                    message="Structured artifact did not contain table or financial fact records.",
                    artifact_index=artifact_index,
                    artifact_type=artifact_type,
                )
            )
            return

        for table_index, table_payload in enumerate(table_payloads):
            self._normalize_table(
                normalized,
                table_payload=table_payload,
                artifact_id=artifact_id,
                artifact_index=artifact_index,
                table_index=table_index,
                document_id=document_id,
                envelope_metadata=envelope_metadata,
            )
        for fact_index, fact_payload in enumerate(fact_payloads):
            self._normalize_financial_fact(
                normalized,
                fact_payload=fact_payload,
                artifact_id=artifact_id,
                artifact_index=artifact_index,
                fact_index=fact_index,
                document_id=document_id,
                source_table_id=_optional_str(fact_payload.get("source_table_id")),
                envelope_metadata=envelope_metadata,
            )

    def _normalize_table(
        self,
        normalized: _NormalizedArtifacts,
        *,
        table_payload: dict[str, Any],
        artifact_id: str,
        artifact_index: int,
        table_index: int,
        document_id: str | None,
        envelope_metadata: dict[str, Any],
    ) -> None:
        table_id = _optional_str(table_payload.get("table_id") or table_payload.get("id"))
        if table_id is None:
            table_id = _stable_id("tbl", artifact_id, artifact_index, table_index, table_payload)
        table_document_id = _optional_str(table_payload.get("document_id")) or document_id
        columns_payload = _column_records(table_payload)
        rows_payload = _records_from(table_payload.get("rows") or table_payload.get("table_rows"))
        cells_payload = _records_from(
            table_payload.get("cells") or table_payload.get("table_cells")
        )
        page_start = _optional_int(table_payload.get("page_start") or table_payload.get("page"))
        page_end = _optional_int(table_payload.get("page_end")) or page_start
        normalized.table_assets.append(
            {
                "table_id": table_id,
                "artifact_id": artifact_id,
                "document_id": table_document_id,
                "page_start": page_start,
                "page_end": page_end,
                "table_title": _optional_str(
                    table_payload.get("table_title") or table_payload.get("title")
                ),
                "source_type": _optional_str(table_payload.get("source_type")) or "unknown",
                "extraction_method": _optional_str(table_payload.get("extraction_method"))
                or _optional_str(table_payload.get("extractor"))
                or "unknown",
                "extraction_confidence": _optional_float(
                    table_payload.get("extraction_confidence") or table_payload.get("confidence")
                ),
                "row_count": _optional_int(table_payload.get("row_count"))
                or len(rows_payload)
                or None,
                "column_count": _optional_int(table_payload.get("column_count"))
                or len(columns_payload)
                or None,
                "metadata_json": _metadata_with_envelope(
                    table_payload,
                    envelope_metadata,
                ),
            }
        )

        column_ids_by_index: dict[int, str] = {}
        for column_index, column_payload in enumerate(columns_payload):
            column_id = _optional_str(column_payload.get("column_id") or column_payload.get("id"))
            if column_id is None:
                column_id = _stable_id(
                    "col",
                    table_id,
                    column_payload.get("name") or column_payload.get("header"),
                    column_index,
                )
            resolved_column_index = _optional_int(column_payload.get("column_index"))
            if resolved_column_index is None:
                resolved_column_index = column_index
            column_ids_by_index[resolved_column_index] = column_id
            normalized.table_columns.append(
                {
                    "column_id": column_id,
                    "table_id": table_id,
                    "column_index": resolved_column_index,
                    "name": _optional_str(
                        column_payload.get("name") or column_payload.get("header")
                    )
                    or f"column_{resolved_column_index}",
                    "canonical_name": _optional_str(column_payload.get("canonical_name")),
                    "data_type": _optional_str(column_payload.get("data_type")) or "unknown",
                    "unit": _optional_str(column_payload.get("unit")),
                    "period": _optional_str(column_payload.get("period")),
                    "metadata_json": _metadata(column_payload),
                }
            )

        for row_index, row_payload in enumerate(rows_payload):
            row_id = _optional_str(row_payload.get("row_id") or row_payload.get("id"))
            resolved_row_index = _optional_int(row_payload.get("row_index"))
            if resolved_row_index is None:
                resolved_row_index = row_index
            if row_id is None:
                row_id = _stable_id(
                    "row",
                    table_id,
                    resolved_row_index,
                    row_payload.get("row_label") or row_payload.get("label"),
                )
            normalized.table_rows.append(
                {
                    "row_id": row_id,
                    "table_id": table_id,
                    "row_index": resolved_row_index,
                    "row_label": _optional_str(
                        row_payload.get("row_label") or row_payload.get("label")
                    ),
                    "canonical_metric": _optional_str(row_payload.get("canonical_metric")),
                    "metadata_json": _metadata(row_payload),
                }
            )
            for column_index, cell_payload in enumerate(_cell_records_from_row(row_payload)):
                self._normalize_cell(
                    normalized,
                    cell_payload=cell_payload,
                    table_id=table_id,
                    row_id=row_id,
                    row_index=resolved_row_index,
                    column_id=column_ids_by_index.get(column_index),
                    column_index=column_index,
                    page_number=page_start,
                )

        for cell_payload in cells_payload:
            column_index = _optional_int(cell_payload.get("column_index"))
            self._normalize_cell(
                normalized,
                cell_payload=cell_payload,
                table_id=table_id,
                row_id=_optional_str(cell_payload.get("row_id")),
                row_index=_optional_int(cell_payload.get("row_index")),
                column_id=_optional_str(cell_payload.get("column_id"))
                or (column_ids_by_index.get(column_index) if column_index is not None else None),
                column_index=column_index,
                page_number=page_start,
            )

        self._normalize_table_profile(
            normalized,
            table_payload=table_payload,
            table_id=table_id,
            artifact_id=artifact_id,
            document_id=table_document_id,
            columns_payload=columns_payload,
            rows_payload=rows_payload,
            envelope_metadata=envelope_metadata,
        )
        for fact_index, fact_payload in enumerate(
            _records_from(table_payload.get("financial_facts") or table_payload.get("facts"))
        ):
            self._normalize_financial_fact(
                normalized,
                fact_payload=fact_payload,
                artifact_id=artifact_id,
                artifact_index=artifact_index,
                fact_index=fact_index,
                document_id=table_document_id,
                source_table_id=table_id,
                envelope_metadata=envelope_metadata,
            )

    def _normalize_table_profile(
        self,
        normalized: _NormalizedArtifacts,
        *,
        table_payload: dict[str, Any],
        table_id: str,
        artifact_id: str,
        document_id: str | None,
        columns_payload: list[dict[str, Any]],
        rows_payload: list[dict[str, Any]],
        envelope_metadata: dict[str, Any],
    ) -> None:
        profile = _as_dict(table_payload.get("profile") or table_payload.get("table_profile"))
        profile_warnings = _issue_payloads(profile.get("warnings"))
        profile_errors = _issue_payloads(profile.get("errors"))
        normalized.table_profiles.append(
            {
                "table_id": table_id,
                "artifact_id": artifact_id,
                "document_id": document_id,
                "row_count": _optional_int(
                    profile.get("row_count") or table_payload.get("row_count")
                )
                or len(rows_payload)
                or None,
                "column_count": _optional_int(
                    profile.get("column_count") or table_payload.get("column_count")
                )
                or len(columns_payload)
                or None,
                "numeric_column_count": _optional_int(profile.get("numeric_column_count")),
                "empty_cell_count": _optional_int(profile.get("empty_cell_count")),
                "profile_json": profile,
                "warnings_json": profile_warnings,
                "errors_json": profile_errors,
                "metadata_json": _metadata_with_envelope(profile, envelope_metadata),
            }
        )

    def _normalize_cell(
        self,
        normalized: _NormalizedArtifacts,
        *,
        cell_payload: dict[str, Any],
        table_id: str,
        row_id: str | None,
        row_index: int | None,
        column_id: str | None,
        column_index: int | None,
        page_number: int | None,
    ) -> None:
        raw_value = cell_payload.get("raw_value")
        if raw_value is None:
            raw_value = cell_payload.get("value", cell_payload.get("text"))
        normalized_value = cell_payload.get("normalized_value", cell_payload.get("normalized"))
        numeric_value = _optional_float(cell_payload.get("numeric_value"))
        if numeric_value is None:
            numeric_value = _optional_float(normalized_value)
        resolved_cell_id = _optional_str(cell_payload.get("cell_id") or cell_payload.get("id"))
        if resolved_cell_id is None:
            resolved_cell_id = _stable_id(
                "cell",
                table_id,
                row_id,
                row_index,
                column_id,
                column_index,
                raw_value,
            )
        normalized.table_cells.append(
            {
                "cell_id": resolved_cell_id,
                "table_id": table_id,
                "row_id": row_id,
                "column_id": column_id,
                "row_index": row_index,
                "column_index": column_index,
                "raw_value": _optional_str(raw_value),
                "normalized_value_json": normalized_value,
                "numeric_value": numeric_value,
                "unit": _optional_str(cell_payload.get("unit")),
                "bbox_json": _dict_or_none(
                    cell_payload.get("bbox") or cell_payload.get("bbox_json")
                ),
                "page_number": _optional_int(cell_payload.get("page_number")) or page_number,
                "provenance_json": _as_dict(cell_payload.get("provenance")),
                "metadata_json": _metadata(cell_payload),
            }
        )

    def _normalize_financial_fact(
        self,
        normalized: _NormalizedArtifacts,
        *,
        fact_payload: dict[str, Any],
        artifact_id: str,
        artifact_index: int,
        fact_index: int,
        document_id: str | None,
        source_table_id: str | None,
        envelope_metadata: dict[str, Any],
    ) -> None:
        company = _optional_str(fact_payload.get("company") or fact_payload.get("company_name"))
        metric = _optional_str(fact_payload.get("metric") or fact_payload.get("canonical_metric"))
        value = _optional_float(
            fact_payload.get("value")
            if fact_payload.get("value") is not None
            else fact_payload.get("numeric_value")
        )
        if company is None or metric is None or value is None:
            normalized.errors.append(
                ManifestIssue(
                    severity="error",
                    code="invalid_financial_fact",
                    message="Financial fact requires company, metric, and numeric value.",
                    path=f"artifacts[{artifact_index}].financial_facts[{fact_index}]",
                    artifact_index=artifact_index,
                    artifact_type="financial_fact",
                )
            )
            return

        source_cell_ids = _source_cell_ids(fact_payload)
        resolved_source_table_id = (
            _optional_str(fact_payload.get("source_table_id")) or source_table_id
        )
        fact_id = _optional_str(fact_payload.get("fact_id") or fact_payload.get("id"))
        if fact_id is None:
            fact_id = _stable_id(
                "fact",
                company,
                fact_payload.get("fiscal_year"),
                fact_payload.get("fiscal_period"),
                metric,
                value,
                resolved_source_table_id,
                source_cell_ids,
            )
        source_document_id = _optional_str(fact_payload.get("source_document_id")) or document_id
        normalized.financial_facts.append(
            {
                "fact_id": fact_id,
                "artifact_id": artifact_id,
                "company": company,
                "company_norm": _optional_str(fact_payload.get("company_norm"))
                or _normalize_token(company),
                "fiscal_year": _optional_int(fact_payload.get("fiscal_year")),
                "fiscal_period": _optional_str(fact_payload.get("fiscal_period")),
                "metric": metric,
                "metric_norm": _optional_str(fact_payload.get("metric_norm"))
                or _normalize_token(metric),
                "value": value,
                "raw_value": _optional_str(fact_payload.get("raw_value")),
                "unit": _optional_str(fact_payload.get("unit")),
                "scale": _optional_str(fact_payload.get("scale")),
                "currency": _optional_str(fact_payload.get("currency")),
                "source_document_id": source_document_id,
                "source_page": _optional_int(fact_payload.get("source_page")),
                "source_table_id": resolved_source_table_id,
                "source_cell_ids_json": source_cell_ids,
                "confidence": _optional_float(fact_payload.get("confidence")),
                "metadata_json": _metadata_with_envelope(fact_payload, envelope_metadata),
            }
        )
        for cell in _source_cell_records(fact_payload):
            cell_id = _optional_str(cell.get("cell_id") or cell.get("id"))
            if cell_id is None:
                continue
            normalized.financial_fact_cells.append(
                {
                    "fact_id": fact_id,
                    "cell_id": cell_id,
                    "table_id": _optional_str(cell.get("table_id")) or resolved_source_table_id,
                    "row_id": _optional_str(cell.get("row_id")),
                    "column_id": _optional_str(cell.get("column_id")),
                    "page_number": _optional_int(cell.get("page_number")),
                    "bbox_json": _dict_or_none(cell.get("bbox") or cell.get("bbox_json")),
                    "provenance_json": _as_dict(cell.get("provenance")),
                    "metadata_json": _metadata(cell),
                }
            )

    def _write_artifact_files(
        self,
        artifact_dir: Path,
        normalized: _NormalizedArtifacts,
    ) -> list[dict[str, Any]]:
        specs = [
            ("raw_artifacts", normalized.raw_artifacts),
            ("table_assets", normalized.table_assets),
            ("table_columns", normalized.table_columns),
            ("table_profiles", normalized.table_profiles),
            ("table_rows", normalized.table_rows),
            ("table_cells", normalized.table_cells),
            ("financial_facts", normalized.financial_facts),
            ("financial_fact_cells", normalized.financial_fact_cells),
        ]
        file_entries: list[dict[str, Any]] = []
        for name, records in specs:
            if name != "raw_artifacts" and not records:
                continue
            jsonl_path = artifact_dir / f"{name}.jsonl"
            _write_jsonl_atomic(jsonl_path, records)
            file_entries.append(_file_entry(name, "jsonl", jsonl_path, len(records)))
            if self.write_csv and records:
                csv_path = artifact_dir / f"{name}.csv"
                _write_csv_atomic(csv_path, records)
                file_entries.append(_file_entry(name, "csv", csv_path, len(records)))
        return file_entries

    def _materialize(
        self,
        db: Session,
        *,
        normalized: _NormalizedArtifacts,
        artifact_id: str,
        document_id: str | None,
        ingestion_run_id: str | None,
        materialization_policy: MaterializationPolicy,
        status: str,
        artifact_dir: Path,
        manifest_path: Path,
        raw_artifacts_path: Path | None,
        payload_hash: str | None,
        artifact_counts: dict[str, int],
        materialized_counts: dict[str, int],
        metadata: Mapping[str, Any] | None,
    ) -> None:
        db.merge(
            StructuredArtifactRecord(
                artifact_id=artifact_id,
                schema_version=self.schema_version,
                artifact_type="structured_artifact_batch",
                document_id=document_id,
                ingestion_run_id=ingestion_run_id,
                materialization_policy=materialization_policy,
                status=status,
                artifact_root_path=str(artifact_dir),
                manifest_path=str(manifest_path),
                raw_artifacts_path=str(raw_artifacts_path) if raw_artifacts_path else None,
                payload_hash=payload_hash,
                artifact_counts_json=artifact_counts,
                materialized_counts_json=materialized_counts,
                warnings_json=[issue.as_dict() for issue in normalized.warnings],
                errors_json=[issue.as_dict() for issue in normalized.errors],
                metadata_json=dict(metadata or {}),
            )
        )
        if materialization_policy == "none":
            db.flush()
            return

        for record in normalized.table_assets:
            db.merge(TableAsset(**record))
        for record in normalized.table_columns:
            db.merge(TableColumn(**record))
        for record in normalized.table_profiles:
            db.merge(TableProfile(**record))
        for record in normalized.financial_facts:
            db.merge(FinancialFact(**record))
        for record in normalized.financial_fact_cells:
            db.merge(FinancialFactCell(**record))

        if materialization_policy == "full":
            for record in normalized.table_rows:
                db.merge(TableRow(**record))
            for record in normalized.table_cells:
                db.merge(TableCell(**record))
        db.flush()

    def _manifest(
        self,
        *,
        artifact_id: str,
        artifact_dir: Path,
        manifest_path: Path,
        document_id: str | None,
        ingestion_run_id: str | None,
        materialization_policy: MaterializationPolicy,
        status: str,
        files: list[dict[str, Any]],
        artifact_counts: dict[str, int],
        materialized_counts: dict[str, int],
        warnings: list[ManifestIssue],
        errors: list[ManifestIssue],
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        raw_artifact_envelopes = _raw_artifact_envelope_summaries_from_files(files)
        schema_routing_cards = _schema_routing_card_summaries_from_records(
            raw_artifact_envelopes
        )
        return {
            "schema_version": self.schema_version,
            "artifact_id": artifact_id,
            "artifact_type": "structured_artifact_batch",
            "status": status,
            "document_id": document_id,
            "ingestion_run_id": ingestion_run_id,
            "materialization_policy": materialization_policy,
            "artifact_root_path": str(artifact_dir),
            "manifest_path": str(manifest_path),
            "created_at": utcnow().isoformat(),
            "files": files,
            "raw_artifact_envelope_count": len(raw_artifact_envelopes),
            "raw_artifact_envelopes": raw_artifact_envelopes,
            "schema_routing_cards": schema_routing_cards,
            "artifact_counts": artifact_counts,
            "materialized_counts": materialized_counts,
            "warnings": [issue.as_dict() for issue in warnings],
            "errors": [issue.as_dict() for issue in errors],
            "metadata": dict(metadata or {}),
        }


def _artifact_parts(
    artifact: ExtractedStructuredArtifact | Mapping[str, Any],
) -> _ArtifactParts:
    if isinstance(artifact, ExtractedStructuredArtifact):
        envelope = dict(artifact.to_envelope_payload())
        artifact_type = (
            _optional_str(envelope.get("artifact_type"))
            or artifact.artifact_type
            or "structured_artifact"
        )
        payload = _payload_from_envelope(envelope)
        envelope_version = _optional_str(envelope.get("envelope_version"))
        schema_version_declarations = _schema_version_declarations(
            payload,
            envelope_version=envelope.get("envelope_version"),
            envelope_version_alias=envelope.get("envelopeVersion"),
        )
        schema_version = _resolved_schema_version(schema_version_declarations)
        return _ArtifactParts(
            artifact_type=artifact_type,
            payload=payload,
            schema_version=schema_version,
            schema_version_declarations=schema_version_declarations,
            envelope_version=envelope_version,
            envelope=envelope,
        )

    if not isinstance(artifact, Mapping):
        return _ArtifactParts(
            artifact_type="unknown",
            payload={},
            schema_version=None,
            schema_version_declarations=(),
            envelope_version=None,
            envelope={"artifact_type": "unknown", "payload": {}},
        )

    artifact_type = _optional_str(artifact.get("artifact_type") or artifact.get("type"))
    payload_value = artifact.get("payload")
    if isinstance(payload_value, Mapping):
        payload = dict(payload_value)
        envelope = dict(artifact)
    else:
        payload = dict(artifact)
        payload.pop("artifact_type", None)
        payload.pop("type", None)
        envelope = dict(artifact)
        envelope["payload"] = payload
    envelope["artifact_type"] = artifact_type or "structured_artifact"
    envelope["payload"] = payload
    envelope_version = _optional_str(artifact.get("envelope_version"))
    if envelope_version is None:
        envelope_version = _optional_str(artifact.get("envelopeVersion"))
    if envelope_version is not None:
        envelope["envelope_version"] = envelope_version
    schema_version_declarations = _schema_version_declarations(
        payload,
        wrapper_schema_version=artifact.get("schema_version"),
        wrapper_schema_version_alias=artifact.get("schemaVersion"),
        envelope_version=artifact.get("envelope_version"),
        envelope_version_alias=artifact.get("envelopeVersion"),
    )
    schema_version = _resolved_schema_version(schema_version_declarations)
    return _ArtifactParts(
        artifact_type=artifact_type or "structured_artifact",
        payload=payload,
        schema_version=schema_version,
        schema_version_declarations=schema_version_declarations,
        envelope_version=envelope_version,
        envelope=envelope,
    )


def _payload_from_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    payload_value = envelope.get("payload")
    if isinstance(payload_value, Mapping):
        return dict(payload_value)
    return {}


def _schema_version_declarations(
    payload: Mapping[str, Any],
    *,
    wrapper_schema_version: Any = None,
    wrapper_schema_version_alias: Any = None,
    envelope_version: Any = None,
    envelope_version_alias: Any = None,
) -> tuple[_SchemaVersionDeclaration, ...]:
    declarations: list[_SchemaVersionDeclaration] = []
    for path, value in (
        ("payload.schema_version", payload.get("schema_version")),
        ("payload.schemaVersion", payload.get("schemaVersion")),
        ("wrapper.schema_version", wrapper_schema_version),
        ("wrapper.schemaVersion", wrapper_schema_version_alias),
        ("envelope_version", envelope_version),
        ("envelopeVersion", envelope_version_alias),
    ):
        text = _optional_str(value)
        if text is not None:
            declarations.append(_SchemaVersionDeclaration(path=path, value=text))
    return tuple(declarations)


def _resolved_schema_version(
    declarations: Sequence[_SchemaVersionDeclaration],
) -> str | None:
    if not declarations:
        return None
    return declarations[0].value


def _schema_version_declaration_payloads(
    declarations: Sequence[_SchemaVersionDeclaration],
) -> list[dict[str, str]]:
    return [
        {"path": declaration.path, "value": declaration.value}
        for declaration in declarations
    ]


def _raw_artifact_record(
    parts: _ArtifactParts,
    *,
    artifact_index: int,
    batch_artifact_id: str,
) -> dict[str, Any]:
    record = dict(parts.envelope)
    record["batch_artifact_id"] = batch_artifact_id
    record["artifact_index"] = artifact_index
    record["artifact_type"] = parts.artifact_type
    record["schema_version"] = parts.schema_version
    record["payload"] = parts.payload
    if parts.envelope_version is not None:
        record["envelope_version"] = parts.envelope_version
    return record


def _envelope_metadata(parts: _ArtifactParts) -> dict[str, Any]:
    envelope = parts.envelope
    metadata: dict[str, Any] = {
        "artifact_type": parts.artifact_type,
        "schema_version": parts.schema_version,
        "envelope_version": parts.envelope_version,
        "artifact_id": envelope.get("artifact_id"),
        "content_hash": envelope.get("content_hash"),
    }
    for field_name in (
        "source_locator",
        "provenance_policy",
        "schema_routing_card",
        "artifact_manifest",
        "metadata",
    ):
        value = envelope.get(field_name)
        if value is not None:
            metadata[field_name] = value
    return metadata


def _validate_materialization_policy(value: str) -> MaterializationPolicy:
    if value not in MATERIALIZATION_POLICIES:
        raise ValueError(
            f"Unsupported materialization_policy {value!r}. "
            f"Expected one of {sorted(MATERIALIZATION_POLICIES)}."
        )
    return value  # type: ignore[return-value]


def _records_from(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        records: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                records.append(dict(item))
        return records
    return []


def _column_records(table_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = table_payload.get("columns") or table_payload.get("table_columns")
    if value is None:
        value = table_payload.get("headers")
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        records: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if isinstance(item, Mapping):
                records.append(dict(item))
            else:
                records.append({"name": str(item), "column_index": index})
        return records
    return _records_from(value)


def _cell_records_from_row(row_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = row_payload.get("cells")
    if value is None:
        value = row_payload.get("values")
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        records: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                records.append(dict(item))
            else:
                records.append({"raw_value": item})
        return records
    return []


def _source_cell_records(fact_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_cells = fact_payload.get("source_cells")
    if source_cells is None:
        source_cells = fact_payload.get("cells")
    records = _records_from(source_cells)
    for cell_id in _source_cell_ids_from_value(fact_payload.get("source_cell_ids")):
        records.append({"cell_id": cell_id})
    return records


def _source_cell_ids(fact_payload: Mapping[str, Any]) -> list[str]:
    ids = _source_cell_ids_from_value(fact_payload.get("source_cell_ids"))
    ids.extend(_source_cell_ids_from_value(fact_payload.get("cell_ids")))
    for cell in _records_from(fact_payload.get("source_cells")):
        cell_id = _optional_str(cell.get("cell_id") or cell.get("id"))
        if cell_id is not None:
            ids.append(cell_id)
    return sorted(set(ids))


def _source_cell_ids_from_value(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [str(item) for item in value if item is not None]
    return []


def _looks_like_table(payload: Mapping[str, Any]) -> bool:
    return any(key in payload for key in ("table_id", "columns", "headers", "rows", "cells"))


def _looks_like_fact(payload: Mapping[str, Any]) -> bool:
    return "metric" in payload and ("value" in payload or "numeric_value" in payload)


def _is_schema_routing_artifact(artifact_kind: str, payload: Mapping[str, Any]) -> bool:
    if artifact_kind in {"schema_routing", "schema_routing_card"}:
        return True
    if _optional_str(payload.get("source_type")) == "schema_routing_card":
        return True
    if _optional_str(payload.get("index_object_type")) == "schema_routing":
        return True
    return _optional_str(payload.get("card_type")) in {
        "schema_routing",
        "table_schema",
        "column_schema",
        "table_profile",
    }


def _is_noop_artifact(artifact_kind: str, payload: Mapping[str, Any]) -> bool:
    if artifact_kind in {"noop", "no_op"}:
        return True
    return _optional_str(payload.get("source_type")) in {"noop", "no_op"}


def _metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata_json")
    if metadata is None:
        metadata = payload.get("metadata")
    return _as_dict(metadata)


def _metadata_with_envelope(
    payload: Mapping[str, Any],
    envelope_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = _metadata(payload)
    if envelope_metadata:
        metadata["structured_artifact_envelope"] = dict(envelope_metadata)
    return metadata


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    payload = _as_dict(value)
    return payload or None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text_value = str(value).strip()
    if not text_value or text_value in {"-", "--", "—", "N/A", "n/a"}:
        return None
    negative = text_value.startswith("(") and text_value.endswith(")")
    cleaned = (
        text_value.strip("()")
        .replace("$", "")
        .replace(",", "")
        .replace("%", "")
        .strip()
    )
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return -number if negative else number


def _normalize_token(value: str) -> str:
    return "_".join(value.strip().lower().split())


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = _json_dumps(parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _issue_payloads(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        payloads: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                payloads.append(dict(item))
            else:
                payloads.append({"message": str(item)})
        return payloads
    return []


def _dedupe_records(
    records: list[dict[str, Any]],
    key_fields: tuple[str, ...],
    *,
    record_type: str,
    warnings: list[ManifestIssue],
) -> list[dict[str, Any]]:
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = tuple(record.get(field) for field in key_fields)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = record
            continue
        if existing == record:
            continue
        warnings.append(
            ManifestIssue(
                severity="warning",
                code="dedupe_conflict",
                message=(
                    f"Conflicting {record_type} records shared key "
                    f"{_dedupe_key_payload(key_fields, key)!r}; keeping the first record."
                ),
                path=record_type,
                artifact_type=record_type,
            )
        )
    return list(deduped.values())


def _dedupe_key_payload(
    key_fields: tuple[str, ...],
    key: tuple[Any, ...],
) -> dict[str, Any]:
    return {field: value for field, value in zip(key_fields, key, strict=False)}


def _raw_artifact_envelope_summaries_from_files(
    files: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw_artifacts_path = _path_from_file_entries(files, "raw_artifacts", "jsonl")
    if raw_artifacts_path is None:
        return []
    path = Path(raw_artifacts_path)
    if not path.exists():
        return []
    summaries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, Mapping):
                summaries.append(_raw_artifact_envelope_summary(record))
    return summaries


def _raw_artifact_envelope_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _as_dict(record.get("metadata"))
    artifact_manifest = _as_dict(record.get("artifact_manifest"))
    schema_routing_card = _schema_routing_card_payload(record)
    summary: dict[str, Any] = {
        "artifact_index": record.get("artifact_index"),
        "artifact_type": record.get("artifact_type"),
        "schema_version": record.get("schema_version"),
        "envelope_version": record.get("envelope_version"),
        "artifact_id": record.get("artifact_id"),
        "content_hash": record.get("content_hash"),
        "has_source_locator": isinstance(record.get("source_locator"), Mapping),
        "has_provenance_policy": isinstance(record.get("provenance_policy"), Mapping),
        "has_schema_routing_card": bool(schema_routing_card),
        "has_artifact_manifest": bool(artifact_manifest),
        "metadata_keys": sorted(str(key) for key in metadata),
    }
    manifest_id = artifact_manifest.get("manifest_id")
    if manifest_id is not None:
        summary["artifact_manifest_id"] = manifest_id
    if schema_routing_card:
        summary["schema_routing_card"] = schema_routing_card
    return summary


def _schema_routing_card_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = _as_dict(record.get("payload"))
    envelope_card = _as_dict(record.get("schema_routing_card"))
    card = envelope_card
    artifact_kind = str(record.get("artifact_type") or "").lower()
    if not card and _is_schema_routing_artifact(artifact_kind, payload):
        card = payload
    if not card:
        return {}
    summary: dict[str, Any] = {}
    for key in (
        "id",
        "card_id",
        "schema_card_id",
        "card_type",
        "artifact_id",
        "artifact_type",
        "table_id",
        "document_id",
        "source_type",
        "semantic_domain",
        "proposed_schema_name",
        "proposed_table_name",
        "content_hash",
        "routing_version",
    ):
        value = card.get(key)
        if value is not None:
            summary[key] = value
    return summary


def _schema_routing_card_summaries_from_records(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for summary in summaries:
        card = _as_dict(summary.get("schema_routing_card"))
        if not card:
            continue
        key = _json_dumps(card)
        if key in seen:
            continue
        seen.add(key)
        cards.append(card)
    return cards


def _artifact_counts(normalized: _NormalizedArtifacts) -> dict[str, int]:
    return {
        "raw_artifacts": len(normalized.raw_artifacts),
        "table_assets": len(normalized.table_assets),
        "table_columns": len(normalized.table_columns),
        "table_profiles": len(normalized.table_profiles),
        "table_rows": len(normalized.table_rows),
        "table_cells": len(normalized.table_cells),
        "financial_facts": len(normalized.financial_facts),
        "financial_fact_cells": len(normalized.financial_fact_cells),
    }


def _empty_materialized_counts() -> dict[str, int]:
    return {
        "structured_artifacts": 0,
        "table_assets": 0,
        "table_columns": 0,
        "table_profiles": 0,
        "table_rows": 0,
        "table_cells": 0,
        "financial_facts": 0,
        "financial_fact_cells": 0,
    }


def _expected_materialized_counts(
    policy: MaterializationPolicy,
    normalized: _NormalizedArtifacts,
) -> dict[str, int]:
    counts = _empty_materialized_counts()
    counts["structured_artifacts"] = 1
    if policy == "none":
        return counts
    counts["table_assets"] = len(normalized.table_assets)
    counts["table_columns"] = len(normalized.table_columns)
    counts["table_profiles"] = len(normalized.table_profiles)
    counts["financial_facts"] = len(normalized.financial_facts)
    counts["financial_fact_cells"] = len(normalized.financial_fact_cells)
    if policy == "full":
        counts["table_rows"] = len(normalized.table_rows)
        counts["table_cells"] = len(normalized.table_cells)
    return counts


def _mark_structured_artifact_record_orphaned(
    db: Session | None,
    *,
    artifact_id: str,
    issue: ManifestIssue,
) -> None:
    if db is None:
        return
    try:
        record = db.get(StructuredArtifactRecord, artifact_id)
    except Exception:
        return
    if record is None:
        return
    record.status = "orphaned"
    record.errors_json = [
        *_issue_payloads(record.errors_json),
        issue.as_dict(),
    ]
    metadata = _as_dict(record.metadata_json)
    metadata["orphaned"] = True
    metadata["orphan_reason"] = issue.message
    record.metadata_json = metadata


def _status_for(normalized: _NormalizedArtifacts) -> str:
    valid_record_count = (
        len(normalized.table_assets)
        + len(normalized.table_columns)
        + len(normalized.table_profiles)
        + len(normalized.table_rows)
        + len(normalized.table_cells)
        + len(normalized.financial_facts)
        + len(normalized.financial_fact_cells)
    )
    if normalized.errors and valid_record_count:
        return "partial_success"
    if normalized.errors:
        return "failed"
    return "completed"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(_json_dumps(payload) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_json_dumps(record))
            handle.write("\n")
    tmp_path.replace(path)


def _write_csv_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = sorted({key for record in records for key in record})
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_value(record.get(key)) for key in fieldnames})
    tmp_path.replace(path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return _json_dumps(value)
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _file_entry(name: str, file_format: str, path: Path, row_count: int) -> dict[str, Any]:
    return {
        "name": name,
        "format": file_format,
        "path": str(path),
        "row_count": row_count,
        "sha256": _sha256_file(path),
    }


def _path_from_file_entries(
    entries: Sequence[Mapping[str, Any]],
    name: str,
    file_format: str,
) -> str | None:
    for entry in entries:
        if entry.get("name") == name and entry.get("format") == file_format:
            path = entry.get("path")
            return str(path) if path else None
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
