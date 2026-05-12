from __future__ import annotations

import csv
import hashlib
import inspect
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

from atlas.ingestion.structured.contracts import (
    ColumnCard,
    ProfileCard,
    SchemaRoutingCard,
    SourceLocator,
    TableCard,
    TableColumnIR,
    TableIR,
    content_hash as structured_content_hash,
)

__all__ = [
    "ColumnCard",
    "ProfileCard",
    "SchemaRoutingCard",
    "TableCard",
    "TableIR",
    "TableExtractionResult",
    "build_schema_routing_cards",
    "csv_to_table_ir",
    "csv_to_table_ir_and_cards",
    "csv_to_table_ir_with_schema_cards",
    "excel_to_table_ir_and_cards",
    "parse_csv_to_table_ir",
    "should_skip_text_chunking_for_tabular_source",
]

CSV_SUFFIXES = {".csv", ".tsv"}
EXCEL_SUFFIXES = {".xls", ".xlsx", ".xlsm"}
HTML_SUFFIXES = {".html", ".htm"}
TABULAR_SUFFIXES = CSV_SUFFIXES | EXCEL_SUFFIXES | HTML_SUFFIXES
TABULAR_FILE_TYPES = {
    "csv",
    "tsv",
    "excel",
    "xls",
    "xlsx",
    "xlsm",
    "html",
    "htm",
} | TABULAR_SUFFIXES


@dataclass(frozen=True)
class TableExtractionResult:
    table_ir: Any
    cards: tuple[Any, ...]
    table_card: Any
    column_cards: tuple[Any, ...]
    profile_card: Any


def csv_to_table_ir_and_cards(
    source: str | bytes | Path,
    *,
    document_id: str | None = None,
    table_id: str | None = None,
    source_uri: str | None = None,
    table_name: str | None = None,
    has_header: bool = True,
    delimiter: str | None = None,
    encoding: str = "utf-8-sig",
) -> TableExtractionResult:
    csv_text, inferred_source_uri = _read_text_source(source, encoding=encoding)
    source_uri = source_uri or inferred_source_uri
    parsed = _parse_csv_text(csv_text, has_header=has_header, delimiter=delimiter)
    document_id = document_id or _stable_id("v4doc", source_uri or "", csv_text)
    table_id = table_id or _stable_id("v4tbl", document_id, table_name or "", csv_text)

    columns = _build_columns(
        table_id=table_id,
        headers=parsed["headers"],
        rows=parsed["rows"],
        original_headers=parsed["original_headers"],
    )
    metadata = {
        "source_uri": source_uri,
        "source_type": "csv",
        "has_header": has_header,
        "delimiter": parsed["delimiter"],
        "raw_rows_stored_as": "table_ir_rows",
        "raw_rows_text_chunked": False,
        "ordinary_text_chunks_emitted": False,
        "schema_routing_cards_only": True,
    }
    table_ir = _make_table_ir(
        table_id=table_id,
        document_id=document_id,
        source_uri=source_uri,
        table_name=table_name,
        columns=columns,
        rows=parsed["rows"],
        metadata=metadata,
    )
    table_card, column_cards, profile_card = _schema_cards_from_parts(
        table_ir=table_ir,
        document_id=document_id,
        table_id=table_id,
        table_name=table_name,
        source_uri=source_uri,
        columns=columns,
        row_count=len(parsed["rows"]),
        metadata=metadata,
    )
    cards = (table_card, *column_cards, profile_card)
    return TableExtractionResult(
        table_ir=table_ir,
        cards=cards,
        table_card=table_card,
        column_cards=column_cards,
        profile_card=profile_card,
    )


def parse_csv_to_table_ir(source: str | bytes | Path, **kwargs: Any) -> Any:
    return csv_to_table_ir_and_cards(source, **kwargs).table_ir


def csv_to_table_ir(source: str | bytes | Path, **kwargs: Any) -> Any:
    return parse_csv_to_table_ir(source, **kwargs)


def csv_to_table_ir_with_schema_cards(
    source: str | bytes | Path,
    **kwargs: Any,
) -> TableExtractionResult:
    return csv_to_table_ir_and_cards(source, **kwargs)


def build_schema_routing_cards(table_ir: Any) -> tuple[Any, ...]:
    table_id = str(_get(table_ir, "table_id", "id", default=""))
    document_id = str(_get(table_ir, "document_id", "doc_id", default=""))
    columns = [_column_dict(column) for column in (_get(table_ir, "columns", default=()) or ())]
    row_count = _optional_int(_get(table_ir, "row_count", default=None))
    if row_count is None:
        rows = _get(table_ir, "rows", default=()) or ()
        row_count = len(rows) if isinstance(rows, Sequence) else 0
    metadata = _metadata_dict(table_ir)
    table_name = _optional_str(_get(table_ir, "title", "table_title", "name", default=None))
    source_locator = _get(table_ir, "source_locator", default=None)
    source_uri = _optional_str(
        _get(
            table_ir,
            "source_uri",
            default=_get(source_locator, "source_uri", default=metadata.get("source_uri")),
        )
    )

    table_card, column_cards, profile_card = _schema_cards_from_parts(
        table_ir=table_ir,
        document_id=document_id,
        table_id=table_id,
        table_name=table_name,
        source_uri=source_uri,
        columns=columns,
        row_count=row_count,
        metadata=metadata,
    )
    return (table_card, *column_cards, profile_card)


def excel_to_table_ir_and_cards(*args: Any, **kwargs: Any) -> TableExtractionResult:
    raise NotImplementedError(
        "excel_structured_ingestion_contract_pending; raw Excel rows must not be emitted as "
        "ordinary TextChunk records"
    )


def should_skip_text_chunking_for_tabular_source(file_type_or_suffix: str) -> bool:
    return _normalized_file_type(file_type_or_suffix) in TABULAR_FILE_TYPES


def _read_text_source(source: str | bytes | Path, *, encoding: str) -> tuple[str, str | None]:
    if isinstance(source, Path):
        if source.suffix.lower() in EXCEL_SUFFIXES:
            raise NotImplementedError(
                "excel_structured_ingestion_contract_pending; raw Excel rows must not be "
                "emitted as ordinary TextChunk records"
            )
        return source.read_text(encoding=encoding), f"local:{source}"
    if isinstance(source, bytes):
        return source.decode(encoding), None
    if "\n" not in source and "\r" not in source:
        path = Path(source)
        if path.exists() and path.is_file():
            suffix = path.suffix.lower()
            if suffix in EXCEL_SUFFIXES:
                raise NotImplementedError(
                    "excel_structured_ingestion_contract_pending; raw Excel rows must not be "
                    "emitted as ordinary TextChunk records"
                )
            return path.read_text(encoding=encoding), f"local:{path}"
    return source, None


def _parse_csv_text(
    csv_text: str,
    *,
    has_header: bool,
    delimiter: str | None,
) -> dict[str, Any]:
    dialect = _csv_dialect(csv_text, delimiter=delimiter)
    reader = csv.reader(StringIO(csv_text), dialect)
    raw_rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not raw_rows:
        return {
            "headers": (),
            "original_headers": (),
            "rows": (),
            "delimiter": getattr(dialect, "delimiter", delimiter or ","),
        }

    width = max(len(row) for row in raw_rows)
    normalized_rows = [_pad_row(row, width) for row in raw_rows]
    if has_header:
        original_headers = tuple(normalized_rows[0])
        headers = _dedupe_headers(original_headers)
        data_rows = normalized_rows[1:]
    else:
        original_headers = tuple(f"column_{index + 1}" for index in range(width))
        headers = original_headers
        data_rows = normalized_rows

    rows = tuple(
        {headers[index]: row[index] for index in range(width)}
        for row in data_rows
    )
    return {
        "headers": headers,
        "original_headers": original_headers,
        "rows": rows,
        "delimiter": getattr(dialect, "delimiter", delimiter or ","),
    }


def _csv_dialect(csv_text: str, *, delimiter: str | None) -> type[csv.Dialect] | csv.Dialect:
    if delimiter:
        class ExplicitDialect(csv.excel):
            pass

        ExplicitDialect.delimiter = delimiter
        return ExplicitDialect
    try:
        return csv.Sniffer().sniff(csv_text[:8192])
    except csv.Error:
        return csv.excel


def _pad_row(row: list[str], width: int) -> tuple[str, ...]:
    return tuple([*row, *([""] * (width - len(row)))])


def _dedupe_headers(headers: Sequence[str]) -> tuple[str, ...]:
    seen: Counter[str] = Counter()
    deduped: list[str] = []
    for index, header in enumerate(headers):
        canonical = _canonical_name(header) or f"column_{index + 1}"
        seen[canonical] += 1
        if seen[canonical] > 1:
            canonical = f"{canonical}_{seen[canonical]}"
        deduped.append(canonical)
    return tuple(deduped)


def _build_columns(
    *,
    table_id: str,
    headers: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    original_headers: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    columns: list[dict[str, Any]] = []
    for index, header in enumerate(headers):
        original_name = original_headers[index] if index < len(original_headers) else header
        values = [str(row.get(header, "")) for row in rows]
        profile = _profile_values(values)
        canonical_name = _canonical_name(original_name) or header
        column_id = _stable_id("v4col", table_id, str(index), canonical_name)
        columns.append(
            {
                "column_id": column_id,
                "table_id": table_id,
                "name": header,
                "original_name": original_name,
                "canonical_name": canonical_name,
                "column_index": index,
                "data_type": profile["data_type"],
                "semantic_role": _semantic_role(canonical_name),
                "unit": _unit_hint(canonical_name),
                "period": _period_hint(canonical_name),
                "nullable": profile["null_count"] > 0,
                "profile": profile,
                "metadata": {
                    "value_samples_included": False,
                    "raw_values_in_schema_card": False,
                },
            }
        )
    return tuple(columns)


def _profile_values(values: Sequence[str]) -> dict[str, Any]:
    stripped = [value.strip() for value in values]
    non_empty = [value for value in stripped if value]
    numeric_count = sum(1 for value in non_empty if _parse_float(value) is not None)
    integer_count = sum(1 for value in non_empty if _parse_int(value) is not None)
    boolean_count = sum(1 for value in non_empty if _parse_bool(value) is not None)
    date_count = sum(1 for value in non_empty if _parse_date(value) is not None)
    data_type = _inferred_type(
        non_empty_count=len(non_empty),
        numeric_count=numeric_count,
        integer_count=integer_count,
        boolean_count=boolean_count,
        date_count=date_count,
    )
    return {
        "observed_row_count": len(values),
        "null_count": len(values) - len(non_empty),
        "non_null_count": len(non_empty),
        "numeric_parse_count": numeric_count,
        "integer_parse_count": integer_count,
        "boolean_parse_count": boolean_count,
        "date_parse_count": date_count,
        "data_type": data_type,
        "value_samples_included": False,
        "raw_values_included": False,
    }


def _inferred_type(
    *,
    non_empty_count: int,
    numeric_count: int,
    integer_count: int,
    boolean_count: int,
    date_count: int,
) -> str:
    if non_empty_count == 0:
        return "empty"
    if boolean_count == non_empty_count:
        return "boolean"
    if integer_count == non_empty_count:
        return "integer"
    if numeric_count == non_empty_count:
        return "number"
    if date_count == non_empty_count:
        return "date"
    return "string"


def _make_table_ir(
    *,
    table_id: str,
    document_id: str,
    source_uri: str | None,
    table_name: str | None,
    columns: tuple[dict[str, Any], ...],
    rows: tuple[dict[str, str], ...],
    metadata: dict[str, Any],
) -> Any:
    contract_columns = tuple(
        _make_table_column_ir(
            column=column,
            document_id=document_id,
            table_id=table_id,
            source_uri=source_uri,
        )
        for column in columns
    )
    row_payload = tuple(dict(row) for row in rows)
    row_locators = _row_locator_payloads(
        table_id=table_id,
        document_id=document_id,
        source_uri=source_uri,
        rows=row_payload,
        has_header=bool(metadata.get("has_header")),
    )
    table_metadata = {
        **metadata,
        "document_id": document_id,
        "row_count": len(row_payload),
        "column_count": len(contract_columns),
        "row_locators": row_locators,
        "row_locator_policy": "physical_row_index_and_row_hash",
        "rows_materialized_as_text_chunks": False,
        "cells_materialized": False,
    }
    table_locator = _table_source_locator(
        document_id=document_id,
        table_id=table_id,
        source_uri=source_uri,
        row_count=len(row_payload),
        column_count=len(contract_columns),
        has_header=bool(metadata.get("has_header")),
    )
    payload = {
        "id": table_id,
        "table_id": table_id,
        "document_id": document_id,
        "source_type": "csv",
        "source_uri": source_uri,
        "title": table_name,
        "table_title": table_name,
        "name": table_name,
        "columns": contract_columns,
        "rows": row_payload,
        "row_count": len(rows),
        "column_count": len(columns),
        "text": "",
        "source_locator": table_locator,
        "source_element_ids": (table_id,),
        "extraction_method": "csv_stdlib",
        "extraction_confidence": 1.0,
        "content_hash": structured_content_hash(
            {
                "table_id": table_id,
                "columns": columns,
                "rows": row_payload,
            }
        ),
        "metadata": table_metadata,
        "metadata_json": table_metadata,
    }
    return _make_contract(TableIR, payload)


def _make_table_column_ir(
    *,
    column: Mapping[str, Any],
    document_id: str,
    table_id: str,
    source_uri: str | None,
) -> Any:
    metadata = {
        **dict(column.get("metadata") or {}),
        "semantic_role": column.get("semantic_role"),
        "profile": column.get("profile"),
        "raw_values_included": False,
    }
    source_locator = _column_source_locator(
        document_id=document_id,
        table_id=table_id,
        source_uri=source_uri,
        column=column,
    )
    payload = {
        "column_id": column.get("column_id") or "",
        "table_id": table_id,
        "document_id": document_id,
        "name": _display_column_name(column),
        "original_name": column.get("original_name"),
        "column_index": column.get("column_index"),
        "canonical_name": column.get("canonical_name"),
        "data_type": column.get("data_type") or "unknown",
        "semantic_role": column.get("semantic_role"),
        "unit": column.get("unit"),
        "period": column.get("period"),
        "source_locator": source_locator,
        "content_hash": structured_content_hash(
            {
                "table_id": table_id,
                "column_id": column.get("column_id"),
                "name": _display_column_name(column),
                "data_type": column.get("data_type"),
            }
        ),
        "metadata": metadata,
    }
    return _make_contract(TableColumnIR, payload)


def _schema_cards_from_parts(
    *,
    table_ir: Any,
    document_id: str,
    table_id: str,
    table_name: str | None,
    source_uri: str | None,
    columns: Sequence[Mapping[str, Any]],
    row_count: int,
    metadata: Mapping[str, Any],
) -> tuple[Any, tuple[Any, ...], Any]:
    table_card = _table_card(
        table_ir=table_ir,
        document_id=document_id,
        table_id=table_id,
        table_name=table_name,
        source_uri=source_uri,
        columns=columns,
        row_count=row_count,
    )
    column_cards = tuple(
        _column_card(
            document_id=document_id,
            table_id=table_id,
            source_uri=source_uri,
            column=_column_dict(column),
            row_count=row_count,
            table_ir=table_ir,
        )
        for column in columns
    )
    profile_card = _profile_card(
        table_ir=table_ir,
        document_id=document_id,
        table_id=table_id,
        table_name=table_name,
        source_uri=source_uri,
        columns=columns,
        row_count=row_count,
        metadata=metadata,
    )
    _ = table_ir
    return table_card, column_cards, profile_card


def _table_card(
    *,
    table_ir: Any,
    document_id: str,
    table_id: str,
    table_name: str | None,
    source_uri: str | None,
    columns: Sequence[Mapping[str, Any]],
    row_count: int,
) -> Any:
    column_summary = ", ".join(
        f"{_display_column_name(column)} ({column.get('data_type', 'unknown')})"
        for column in columns
    )
    title = table_name or table_id
    routing_text = (
        f"CSV table schema for {title}: {row_count} rows, {len(columns)} columns. "
        f"Columns: {column_summary or 'none'}. Raw row values are omitted from this routing card."
    )
    payload = _schema_card_payload(
        card_cls=TableCard,
        card_id=_stable_id("v4card", table_id, "table"),
        document_id=document_id,
        table_id=table_id,
        card_type="table_schema",
        title=f"Table schema: {title}",
        routing_text=routing_text,
        source_uri=source_uri,
        source_locator=_table_card_source_locator(
            table_ir=table_ir,
            document_id=document_id,
            table_id=table_id,
            source_uri=source_uri,
        ),
        source_element_ids=_source_element_ids_from_table(table_ir, table_id),
        structured_payload={
            "table_id": table_id,
            "row_count": row_count,
            "column_count": len(columns),
            "columns": [
                {
                    "column_id": column.get("column_id"),
                    "name": _display_column_name(column),
                    "data_type": column.get("data_type"),
                    "semantic_role": column.get("semantic_role"),
                }
                for column in columns
            ],
            "raw_rows_included": False,
        },
    )
    return _make_contract(TableCard, payload)


def _column_card(
    *,
    document_id: str,
    table_id: str,
    source_uri: str | None,
    column: dict[str, Any],
    row_count: int,
    table_ir: Any,
) -> Any:
    name = _display_column_name(column)
    profile = dict(column.get("profile") or {})
    routing_text = (
        f"CSV column schema for {name}: inferred type {column.get('data_type', 'unknown')}, "
        f"role {column.get('semantic_role', 'unknown')}, nulls "
        f"{profile.get('null_count', 0)} of {row_count}. Raw column values are omitted."
    )
    payload = _schema_card_payload(
        card_cls=ColumnCard,
        card_id=_stable_id("v4card", table_id, "column", str(column.get("column_id") or name)),
        document_id=document_id,
        table_id=table_id,
        card_type="column_schema",
        title=f"Column schema: {name}",
        routing_text=routing_text,
        source_uri=source_uri,
        source_locator=_column_card_source_locator(
            column=column,
            table_ir=table_ir,
            document_id=document_id,
            table_id=table_id,
            source_uri=source_uri,
        ),
        source_element_ids=_source_element_ids_from_table(table_ir, table_id),
        structured_payload={
            "table_id": table_id,
            "column_id": column.get("column_id"),
            "column_name": name,
            "canonical_name": column.get("canonical_name"),
            "data_type": column.get("data_type"),
            "semantic_role": column.get("semantic_role"),
            "unit": column.get("unit"),
            "period": column.get("period"),
            "profile": {
                key: value
                for key, value in profile.items()
                if key
                in {
                    "observed_row_count",
                    "null_count",
                    "non_null_count",
                    "numeric_parse_count",
                    "integer_parse_count",
                    "boolean_parse_count",
                    "date_parse_count",
                    "data_type",
                    "value_samples_included",
                    "raw_values_included",
                }
            },
            "raw_values_included": False,
        },
    )
    payload.update(
        {
            "column_id": column.get("column_id"),
            "column_name": name,
        }
    )
    return _make_contract(ColumnCard, payload)


def _profile_card(
    *,
    table_ir: Any,
    document_id: str,
    table_id: str,
    table_name: str | None,
    source_uri: str | None,
    columns: Sequence[Mapping[str, Any]],
    row_count: int,
    metadata: Mapping[str, Any],
) -> Any:
    type_counts = Counter(str(column.get("data_type", "unknown")) for column in columns)
    type_summary = ", ".join(f"{key}={value}" for key, value in sorted(type_counts.items()))
    title = table_name or table_id
    routing_text = (
        f"CSV table profile for {title}: {row_count} rows, {len(columns)} columns, "
        f"inferred column types {type_summary or 'none'}. Profile excludes raw row values."
    )
    payload = _schema_card_payload(
        card_cls=ProfileCard,
        card_id=_stable_id("v4card", table_id, "profile"),
        document_id=document_id,
        table_id=table_id,
        card_type="table_profile",
        title=f"Table profile: {title}",
        routing_text=routing_text,
        source_uri=source_uri,
        source_locator=_table_card_source_locator(
            table_ir=table_ir,
            document_id=document_id,
            table_id=table_id,
            source_uri=source_uri,
        ),
        source_element_ids=_source_element_ids_from_table(table_ir, table_id),
        structured_payload={
            "table_id": table_id,
            "row_count": row_count,
            "column_count": len(columns),
            "type_counts": dict(type_counts),
            "delimiter": metadata.get("delimiter"),
            "has_header": metadata.get("has_header"),
            "raw_rows_included": False,
            "raw_values_included": False,
        },
    )
    return _make_contract(ProfileCard, payload)


def _schema_card_payload(
    *,
    card_cls: type[Any],
    card_id: str,
    document_id: str,
    table_id: str,
    card_type: str,
    title: str,
    routing_text: str,
    source_uri: str | None,
    source_locator: Any,
    source_element_ids: tuple[str, ...],
    structured_payload: dict[str, Any],
) -> dict[str, Any]:
    columns = structured_payload.get("columns")
    if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes, bytearray)):
        columns = ()
    measure_columns = [
        str(column.get("name") or column.get("column_name"))
        for column in columns
        if isinstance(column, Mapping) and column.get("semantic_role") == "measure"
    ]
    dimension_columns = [
        str(column.get("name") or column.get("column_name"))
        for column in columns
        if isinstance(column, Mapping) and column.get("semantic_role") == "dimension"
    ]
    period_columns = [
        str(column.get("name") or column.get("column_name"))
        for column in columns
        if isinstance(column, Mapping) and column.get("semantic_role") == "period"
    ]
    evidence_policy = {
        "routing_only": True,
        "value_answer_evidence_allowed": False,
        "schema_answer_evidence_allowed": True,
        "raw_rows_text_chunked": False,
    }
    metadata = {
        "source_uri": source_uri,
        "source_type": "csv",
        "card_contract": getattr(card_cls, "__name__", "SchemaRoutingCard"),
        "routing_only": True,
        "value_answer_evidence_allowed": False,
        "schema_answer_evidence_allowed": True,
        "answer_evidence_allowed": False,
        "index_object_type": "schema_routing",
        "evidence_role": "routing_only",
        "raw_rows_included": False,
        "raw_values_included": False,
        "ordinary_text_chunk": False,
        "evidence_policy": evidence_policy,
    }
    locator_payload = _locator_payload(source_locator)
    if locator_payload:
        metadata["source_locator"] = locator_payload
    return {
        "id": card_id,
        "card_id": card_id,
        "schema_card_id": card_id,
        "artifact_id": table_id,
        "artifact_type": "table",
        "table_id": table_id,
        "document_id": document_id,
        "card_type": card_type,
        "source_type": "schema_routing_card",
        "semantic_domain": "structured_table",
        "proposed_table_name": table_id,
        "primary_key_candidates": [],
        "measure_columns": measure_columns,
        "dimension_columns": dimension_columns,
        "period_columns": period_columns,
        "confidence": 1.0,
        "title": title,
        "text": routing_text,
        "routing_text": routing_text,
        "routing_only": True,
        "source_derived_text": routing_text,
        "computed_from_source_text": routing_text,
        "inferred_text": routing_text,
        "value_answer_evidence_allowed": False,
        "schema_answer_evidence_allowed": True,
        "answer_evidence_allowed": False,
        "index_object_type": "schema_routing",
        "evidence_role": "routing_only",
        "structured_payload": structured_payload,
        "source_locator": source_locator,
        "source_element_ids": source_element_ids,
        "metadata": metadata,
        "metadata_json": metadata,
    }


def _table_source_locator(
    *,
    document_id: str,
    table_id: str,
    source_uri: str | None,
    row_count: int,
    column_count: int,
    has_header: bool,
) -> Any:
    first_data_row_index = 1 if has_header else 0
    last_data_row_index = first_data_row_index + row_count - 1 if row_count else None
    table_range = {
        "format": "csv_physical_row_column_index",
        "row_start": first_data_row_index if row_count else None,
        "row_end": last_data_row_index,
        "column_start": 0 if column_count else None,
        "column_end": column_count - 1 if column_count else None,
        "has_header": has_header,
    }
    return _make_contract(
        SourceLocator,
        {
            "source_uri": source_uri,
            "document_id": document_id,
            "table_id": table_id,
            "table_range": table_range,
            "locator_precision": "table",
            "locator_confidence": 1.0,
            "is_exact": True,
            "locator_method": "parser",
        },
    )


def _column_source_locator(
    *,
    document_id: str,
    table_id: str,
    source_uri: str | None,
    column: Mapping[str, Any],
) -> Any:
    column_index = _optional_int(column.get("column_index"))
    name = _display_column_name(column)
    column_locator = {
        "column_index": column_index,
        "column_id": column.get("column_id"),
        "column_name": name,
        "canonical_name": column.get("canonical_name"),
    }
    return _make_contract(
        SourceLocator,
        {
            "source_uri": source_uri,
            "document_id": document_id,
            "table_id": table_id,
            "column_index": column_index,
            "column_locator": column_locator,
            "locator_precision": "column",
            "locator_confidence": 1.0,
            "is_exact": True,
            "locator_method": "parser",
        },
    )


def _row_locator_payloads(
    *,
    table_id: str,
    document_id: str,
    source_uri: str | None,
    rows: Sequence[Mapping[str, Any]],
    has_header: bool,
) -> list[dict[str, Any]]:
    first_data_row_index = 1 if has_header else 0
    row_locators: list[dict[str, Any]] = []
    for data_row_index, row in enumerate(rows):
        physical_row_index = first_data_row_index + data_row_index
        row_hash = structured_content_hash(dict(row))
        row_locator = {
            "row_index": physical_row_index,
            "row_hash": row_hash,
        }
        source_locator = _make_contract(
            SourceLocator,
            {
                "source_uri": source_uri,
                "document_id": document_id,
                "table_id": table_id,
                "row_index": physical_row_index,
                "row_locator": row_locator,
                "locator_precision": "row",
                "locator_confidence": 1.0,
                "is_exact": True,
                "locator_method": "parser",
            },
        )
        row_locators.append(
            {
                "row_id": _stable_id("v4row", table_id, str(physical_row_index), row_hash),
                "row_index": physical_row_index,
                "data_row_index": data_row_index,
                "row_hash": row_hash,
                "row_locator": row_locator,
                "source_locator": _locator_payload(source_locator),
                "materialized": False,
                "cells_materialized": False,
                "ordinary_text_chunk": False,
            }
        )
    return row_locators


def _table_card_source_locator(
    *,
    table_ir: Any,
    document_id: str,
    table_id: str,
    source_uri: str | None,
) -> Any:
    existing = _locator_payload(_get(table_ir, "source_locator", default=None))
    existing.update(
        {
            "source_uri": existing.get("source_uri") or source_uri,
            "document_id": existing.get("document_id") or document_id,
            "table_id": existing.get("table_id") or table_id,
            "locator_precision": existing.get("locator_precision")
            if existing.get("locator_precision") not in {None, "", "unknown"}
            else "table",
            "locator_confidence": existing.get("locator_confidence")
            if existing.get("locator_confidence") not in {None, 0.0}
            else 1.0,
            "is_exact": True if existing.get("is_exact") is None else bool(existing.get("is_exact")),
            "locator_method": existing.get("locator_method")
            if existing.get("locator_method") not in {None, "", "unspecified"}
            else "parser",
        }
    )
    return _make_contract(SourceLocator, existing)


def _column_card_source_locator(
    *,
    column: Mapping[str, Any],
    table_ir: Any,
    document_id: str,
    table_id: str,
    source_uri: str | None,
) -> Any:
    existing = _locator_payload(column.get("source_locator"))
    if not existing:
        return _column_source_locator(
            document_id=document_id,
            table_id=table_id,
            source_uri=source_uri,
            column=column,
        )
    existing.update(
        {
            "source_uri": existing.get("source_uri") or source_uri,
            "document_id": existing.get("document_id") or document_id,
            "table_id": existing.get("table_id") or table_id,
            "locator_precision": existing.get("locator_precision")
            if existing.get("locator_precision") not in {None, "", "unknown"}
            else "column",
            "locator_confidence": existing.get("locator_confidence")
            if existing.get("locator_confidence") not in {None, 0.0}
            else 1.0,
            "is_exact": True if existing.get("is_exact") is None else bool(existing.get("is_exact")),
            "locator_method": existing.get("locator_method")
            if existing.get("locator_method") not in {None, "", "unspecified"}
            else "parser",
        }
    )
    if not existing.get("column_locator"):
        existing["column_locator"] = {
            "column_index": _optional_int(column.get("column_index")),
            "column_id": column.get("column_id"),
            "column_name": _display_column_name(column),
            "canonical_name": column.get("canonical_name"),
        }
    _ = table_ir
    return _make_contract(SourceLocator, existing)


def _source_element_ids_from_table(table_ir: Any, table_id: str) -> tuple[str, ...]:
    raw = _get(table_ir, "source_element_ids", default=None)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        values = tuple(str(item) for item in raw if str(item))
        if values:
            return values
    return (table_id,) if table_id else ()


def _locator_payload(locator: Any) -> dict[str, Any]:
    if locator is None:
        return {}
    if isinstance(locator, Mapping):
        return {str(key): value for key, value in locator.items()}
    to_payload = getattr(locator, "to_payload", None)
    if callable(to_payload):
        payload = to_payload()
        return dict(payload) if isinstance(payload, Mapping) else {}
    result: dict[str, Any] = {}
    for name in _source_locator_field_names():
        if hasattr(locator, name):
            result[name] = getattr(locator, name)
    return result


def _source_locator_field_names() -> tuple[str, ...]:
    names = _contract_field_names(SourceLocator)
    if names:
        return tuple(sorted(names))
    return (
        "source_uri",
        "source_path",
        "document_id",
        "storage_ref",
        "storage_format",
        "storage_offset",
        "storage_length",
        "page_number",
        "page_start",
        "page_end",
        "sheet_name",
        "element_id",
        "table_id",
        "table_range",
        "row_index",
        "row_locator",
        "column_index",
        "column_locator",
        "cell_ref",
        "char_start",
        "char_end",
        "bbox",
        "locator_precision",
        "locator_confidence",
        "is_exact",
        "locator_method",
        "locator_version",
    )


def _display_column_name(column: Mapping[str, Any]) -> str:
    return str(
        column.get("original_name")
        or column.get("name")
        or column.get("canonical_name")
        or ""
    )


def _column_dict(column: Any) -> dict[str, Any]:
    if isinstance(column, Mapping):
        result = dict(column)
        metadata = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}
        if "semantic_role" not in result and isinstance(metadata, Mapping):
            result["semantic_role"] = metadata.get("semantic_role")
        if "profile" not in result and isinstance(metadata, Mapping):
            result["profile"] = metadata.get("profile")
        return result
    result: dict[str, Any] = {}
    for key in (
        "column_id",
        "table_id",
        "name",
        "original_name",
        "canonical_name",
        "column_index",
        "data_type",
        "semantic_role",
        "unit",
        "period",
        "nullable",
        "profile",
        "source_locator",
        "metadata",
    ):
        if hasattr(column, key):
            result[key] = getattr(column, key)
    metadata = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}
    if "semantic_role" not in result and isinstance(metadata, Mapping):
        result["semantic_role"] = metadata.get("semantic_role")
    if "profile" not in result and isinstance(metadata, Mapping):
        result["profile"] = metadata.get("profile")
    return result


def _semantic_role(name: str) -> str:
    lowered = name.lower()
    if re.search(r"\b(year|fy|fiscal|period|quarter|date|month)\b", lowered):
        return "period"
    if re.search(r"\b(id|identifier|code|ticker|symbol|cusip|isin)\b", lowered):
        return "identifier"
    if re.search(r"\b(company|segment|region|country|category|name)\b", lowered):
        return "dimension"
    if re.search(r"\b(unit|currency|scale)\b", lowered):
        return "unit"
    if re.search(
        r"\b(amount|value|revenue|sales|income|expense|cash|asset|liabil|margin)\b",
        lowered,
    ):
        return "measure"
    return "unknown"


def _unit_hint(name: str) -> str | None:
    lowered = name.lower()
    if "usd" in lowered or "dollar" in lowered or "$" in lowered:
        return "currency_usd"
    if "percent" in lowered or "pct" in lowered or "%" in lowered or "margin" in lowered:
        return "percent"
    if "share" in lowered:
        return "shares"
    return None


def _period_hint(name: str) -> str | None:
    match = re.search(r"\b(19|20)\d{2}\b", name)
    return match.group(0) if match else None


def _canonical_name(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_")


def _parse_float(value: str) -> float | None:
    normalized = value.strip().replace(",", "")
    if not normalized:
        return None
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = f"-{normalized[1:-1]}"
    normalized = normalized.replace("$", "").replace("%", "").strip()
    try:
        return float(normalized)
    except ValueError:
        return None


def _parse_int(value: str) -> int | None:
    parsed = _parse_float(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def _parse_bool(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    return None


def _parse_date(value: str) -> datetime | None:
    stripped = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(stripped, fmt)
        except ValueError:
            continue
    return None


def _make_contract(contract_cls: type[Any], payload: dict[str, Any]) -> Any:
    field_names = _contract_field_names(contract_cls)
    if field_names is None:
        return contract_cls(**payload)
    filtered = {key: value for key, value in payload.items() if key in field_names}
    try:
        return contract_cls(**filtered)
    except TypeError:
        return contract_cls(**payload)


def _contract_field_names(contract_cls: type[Any]) -> set[str] | None:
    model_fields = getattr(contract_cls, "model_fields", None)
    if isinstance(model_fields, Mapping):
        return set(model_fields)
    if is_dataclass(contract_cls):
        return {item.name for item in fields(contract_cls)}
    annotations = getattr(contract_cls, "__annotations__", None)
    if isinstance(annotations, Mapping) and annotations:
        return set(annotations)
    try:
        signature = inspect.signature(contract_cls)
    except (TypeError, ValueError):
        return None
    names = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name != "self"
    }
    return names or None


def _metadata_dict(value: Any) -> dict[str, Any]:
    metadata = _get(value, "metadata", "metadata_json", default=None)
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalized_file_type(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("."):
        return raw
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    return raw


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _stable_id(prefix: str, *parts: str) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"
