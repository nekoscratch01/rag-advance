from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from atlas.ingestion.chunker import approx_token_count
from atlas.retrieval.contracts import SourceAnchor
from atlas.retrieval.models.candidate import Candidate
from atlas.retrieval.providers.sql.models import (
    SQLExecutionResult,
    SQLResultEvidence,
    SQLSchemaContext,
    SQLValidationResult,
)


def build_sql_result_evidence(
    *,
    schema: SQLSchemaContext,
    sql: str,
    validation: SQLValidationResult,
    execution: SQLExecutionResult,
    task_id: str | None,
    unit_id: str | None,
) -> tuple[SQLResultEvidence, Candidate]:
    text = format_sql_result_text(schema, execution)
    evidence_id = _stable_id("sql_ev", schema.table_id, sql, execution.rows)
    candidate_id = _stable_id("sql_cand", schema.table_id, sql, execution.rows)
    chunk_id = _stable_id("sql_chunk", schema.table_id, sql, execution.rows)
    source_anchor = SourceAnchor(
        document_id=schema.document_id,
        table_id=schema.table_id,
        metadata={
            "provider": "sql",
            "source_type": "sql_result",
            "source_uri": schema.source_uri,
            "source_locator": schema.source_locator,
        },
    )
    manifest_id = _manifest_id(schema)
    storage_ref = _storage_ref(schema)
    structured_payload = {
        "provider": "sql",
        "provider_version": "sql_provider_v1",
        "source_type": "sql_result",
        "dialect": "duckdb",
        "candidate_table_id": schema.table_id,
        "table_id": schema.table_id,
        "raw_table_name": schema.raw_table_name,
        "display_table_name": schema.display_table_name,
        "safe_table_name": schema.safe_table_name,
        "sql": sql,
        "validated_sql": sql,
        "validation_status": validation.status,
        "execution_status": execution.status,
        "columns": list(execution.columns),
        "rows": [dict(row) for row in execution.rows],
        "row_count": execution.row_count,
        "truncated": execution.truncated,
        "result_bytes": execution.result_bytes,
        "used_column_ids": list(validation.used_column_ids),
        "used_safe_columns": list(validation.used_safe_columns),
        "safe_to_raw": dict(schema.safe_to_raw),
        "safe_to_raw_identifier_map": dict(schema.safe_to_raw),
        "manifest_id": manifest_id,
        "storage_ref": storage_ref,
        "source_anchor": asdict(source_anchor),
        "source_locator": dict(schema.source_locator),
        "execution_warnings": list(execution.warnings),
        "answer_synthesis_verified": False,
    }
    evidence = SQLResultEvidence(
        evidence_id=evidence_id,
        table_id=schema.table_id,
        text=text,
        sql=sql,
        columns=execution.columns,
        rows=execution.rows,
        structured_payload=structured_payload,
    )
    metadata = {
        "provider": "sql",
        "source_type": "sql_result",
        "rerankable": False,
        "fusion_policy": "pinned",
        "structured_payload": structured_payload,
        "source_anchor": asdict(source_anchor),
        "provider_local_evidence_id": evidence_id,
        "retrieval_task_id": task_id,
        "retrieval_unit_id": unit_id,
    }
    candidate = Candidate(
        candidate_id=candidate_id,
        chunk_id=chunk_id,
        document_id=schema.document_id or f"sql:{schema.table_id}",
        doc_name=schema.display_table_name,
        source_title=f"SQL result for table {schema.display_table_name}",
        company=None,
        text=text,
        page_start=None,
        page_end=None,
        chunk_index=0,
        token_count=approx_token_count(text),
        retrieved_by=("sql",),
        dense_rank=None,
        dense_score=None,
        fusion_rank=1,
        fusion_score=1.0,
        final_rank=1,
        metadata=metadata,
        source_uri=schema.source_uri,
        provider="sql",
        source_type="sql_result",
        retrieval_task_id=task_id,
        retrieval_unit_id=unit_id,
        rerankable=False,
        fusion_policy="pinned",
        structured_payload=structured_payload,
    )
    return evidence, candidate


def format_sql_result_text(schema: SQLSchemaContext, execution: SQLExecutionResult) -> str:
    header = f"SQL result for table {schema.display_table_name} ({schema.table_id}):"
    if not execution.rows:
        return f"{header}\n(no rows)"
    lines = [header]
    for row_index, row in enumerate(execution.rows, start=1):
        if len(execution.rows) > 1:
            lines.append(f"row {row_index}:")
        for column in execution.columns:
            lines.append(f"{column} = {_format_value(row.get(column))}")
    if execution.truncated:
        lines.append(f"(truncated to {execution.row_count} rows)")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if value is None:
        return "NULL"
    return str(value)


def _stable_id(prefix: str, *parts: Any) -> str:
    data = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(data).hexdigest()[:24]}"


def _manifest_id(schema: SQLSchemaContext) -> str | None:
    for container in (schema.metadata, schema.source_locator):
        for key in (
            "manifest_id",
            "structured_manifest_id",
            "artifact_manifest_id",
        ):
            value = container.get(key)
            if value:
                return str(value)
    return None


def _storage_ref(schema: SQLSchemaContext) -> dict[str, Any]:
    for container in (schema.metadata, schema.source_locator):
        value = container.get("storage_ref") or container.get("storage")
        if isinstance(value, dict):
            return dict(value)
        if value:
            return {"ref": str(value)}
    return {}
