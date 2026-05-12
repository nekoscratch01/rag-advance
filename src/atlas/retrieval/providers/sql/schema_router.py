from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import select

from atlas.db.models import (
    StructuredArtifactRecord,
    TableAsset,
    TableCell,
    TableColumn,
    TableRow,
)
from atlas.retrieval.providers.sql.models import SchemaRouteResult, SQLTableContext


class AtlasSchemaRouter:
    """Route a table/numeric question to exactly one structured table."""

    def __init__(
        self,
        *,
        min_table_score: float = 0.15,
        min_score_margin: float = 0.10,
        max_candidate_tables: int = 1,
    ) -> None:
        self.min_table_score = min_table_score
        self.min_score_margin = min_score_margin
        self.max_candidate_tables = max_candidate_tables

    def route(
        self,
        question: str,
        *,
        db: Any = None,
        filters: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> SchemaRouteResult:
        tables = _dedupe_tables(
            [
                *_tables_from_options(options or {}),
                *_tables_from_db(db, filters=filters or {}),
            ]
        )
        trace: dict[str, Any] = {
            "available_table_count": len(tables),
            "min_table_score": self.min_table_score,
            "min_score_margin": self.min_score_margin,
            "max_candidate_tables": self.max_candidate_tables,
            "table_score_threshold": self.min_table_score,
            "score_margin_threshold": self.min_score_margin,
            "selected_table_id": None,
            "top1_score": None,
            "top1_top2_margin": None,
        }
        if not tables:
            return SchemaRouteResult(
                status="cannot_answer_no_table",
                reason="no_structured_table_available",
                trace=trace,
            )

        scored = tuple(
            sorted(
                (_score_table(table, question) for table in tables),
                key=lambda table: (-table.score, table.table_id),
            )
        )
        trace["candidates"] = [
            {
                "table_id": table.table_id,
                "display_name": table.display_name,
                "score": table.score,
                "score_details": dict(table.score_details),
            }
            for table in scored[:5]
        ]
        top = scored[0]
        trace["selected_table_id"] = top.table_id
        trace["top1_score"] = top.score
        if top.score < self.min_table_score:
            return SchemaRouteResult(
                status="cannot_answer_low_confidence",
                reason="top_table_below_score_threshold",
                candidates=scored,
                trace=trace,
            )

        if len(scored) > 1:
            margin = top.score - scored[1].score
            trace["top1_top2_margin"] = margin
            if margin < self.min_score_margin:
                return SchemaRouteResult(
                    status="unsupported_multi_table",
                    reason="ambiguous_table_routing_margin",
                    candidates=scored,
                    trace=trace,
                )

        if self.max_candidate_tables != 1:
            trace["requested_max_candidate_tables"] = self.max_candidate_tables
        return SchemaRouteResult(
            status="success",
            table=top,
            reason="single_table_selected",
            candidates=scored[:1],
            trace=trace,
        )


def _tables_from_options(options: Mapping[str, Any]) -> list[SQLTableContext]:
    tables: list[SQLTableContext] = []
    for key in (
        "structured_sql_tables",
        "sql_tables",
        "structured_tables",
        "tables",
    ):
        tables.extend(_tables_from_payload(options.get(key)))
    for key in (
        "structured_artifacts",
        "structured_artifact_records",
        "schema_routing_cards",
        "manifest",
        "structured_manifest",
    ):
        tables.extend(_tables_from_payload(options.get(key)))
    return tables


def _tables_from_db(db: Any, *, filters: Mapping[str, Any]) -> list[SQLTableContext]:
    if db is None or not hasattr(db, "scalars"):
        return []
    tables: list[SQLTableContext] = []
    tables.extend(_tables_from_structured_artifact_records(db, filters=filters))
    tables.extend(_tables_from_materialized_db(db, filters=filters))
    return tables


def _tables_from_structured_artifact_records(
    db: Any,
    *,
    filters: Mapping[str, Any],
) -> list[SQLTableContext]:
    try:
        stmt = select(StructuredArtifactRecord).limit(50)
        document_ids = _string_list(filters.get("document_ids") or filters.get("document_id"))
        if document_ids:
            stmt = stmt.where(StructuredArtifactRecord.document_id.in_(document_ids))
        records = list(db.scalars(stmt).all())
    except Exception:
        return []

    tables: list[SQLTableContext] = []
    for record in records:
        for path_value in (
            getattr(record, "raw_artifacts_path", None),
            getattr(record, "manifest_path", None),
        ):
            if not path_value:
                continue
            tables.extend(_tables_from_file(Path(path_value)))
    return tables


def _tables_from_materialized_db(db: Any, *, filters: Mapping[str, Any]) -> list[SQLTableContext]:
    try:
        stmt = select(TableAsset).limit(100)
        document_ids = _string_list(filters.get("document_ids") or filters.get("document_id"))
        if document_ids:
            stmt = stmt.where(TableAsset.document_id.in_(document_ids))
        assets = list(db.scalars(stmt).all())
    except Exception:
        return []
    if not assets:
        return []

    table_ids = [asset.table_id for asset in assets if getattr(asset, "table_id", None)]
    columns_by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_table = _materialized_rows_by_table(db, table_ids)
    try:
        for column in db.scalars(select(TableColumn).where(TableColumn.table_id.in_(table_ids))).all():
            columns_by_table[column.table_id].append(
                {
                    "column_id": column.column_id,
                    "name": column.name,
                    "canonical_name": column.canonical_name,
                    "column_index": column.column_index,
                    "data_type": column.data_type,
                    "unit": column.unit,
                    "period": column.period,
                    "metadata": dict(column.metadata_json or {}),
                }
            )
    except Exception:
        pass

    tables = []
    for asset in assets:
        table_id = str(asset.table_id)
        metadata = dict(asset.metadata_json or {})
        title = str(asset.table_title or table_id)
        tables.append(
            SQLTableContext(
                table_id=table_id,
                raw_source_name=title,
                display_name=title,
                document_id=asset.document_id,
                source_uri=metadata.get("source_uri"),
                source_locator=_dict_value(metadata.get("source_locator")),
                rows=tuple(rows_by_table.get(table_id, ())),
                columns=tuple(columns_by_table.get(table_id, ())),
                row_count=asset.row_count,
                column_count=asset.column_count,
                routing_text=_routing_text(title, columns_by_table.get(table_id, ()), asset.row_count),
                metadata=metadata,
            )
        )
    return tables


def _materialized_rows_by_table(db: Any, table_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    rows_by_table: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    if not table_ids:
        return {}
    try:
        columns = {
            column.column_id: column.name
            for column in db.scalars(select(TableColumn).where(TableColumn.table_id.in_(table_ids))).all()
        }
        for row in db.scalars(select(TableRow).where(TableRow.table_id.in_(table_ids))).all():
            if row.row_index is not None:
                rows_by_table[row.table_id][int(row.row_index)] = {}
        for cell in db.scalars(select(TableCell).where(TableCell.table_id.in_(table_ids))).all():
            if cell.row_index is None:
                continue
            column_name = columns.get(cell.column_id or "", f"column_{cell.column_index or 0}")
            value = cell.numeric_value if cell.numeric_value is not None else cell.raw_value
            rows_by_table[cell.table_id].setdefault(int(cell.row_index), {})[column_name] = value
    except Exception:
        return {}
    return {
        table_id: [row for _, row in sorted(rows.items())]
        for table_id, rows in rows_by_table.items()
    }


def _tables_from_file(path: Path) -> list[SQLTableContext]:
    if not path.exists() or not path.is_file():
        return []
    if path.suffix.lower() == ".jsonl":
        return _tables_from_payload(_read_jsonl(path))
    if path.suffix.lower() == ".json":
        try:
            return _tables_from_payload(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return []
    return []


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, Mapping):
                    records.append(dict(value))
    except Exception:
        return []
    return records


def _tables_from_payload(value: Any) -> list[SQLTableContext]:
    table_payloads: dict[str, dict[str, Any]] = {}
    cards_by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    _collect_payload(value, table_payloads=table_payloads, cards_by_table=cards_by_table)
    tables: list[SQLTableContext] = []
    for table_id, payload in sorted(table_payloads.items()):
        cards = tuple(cards_by_table.get(table_id, ()))
        merged_columns = _merge_columns(payload, cards)
        rows = tuple(dict(row) for row in _records_from(payload.get("rows") or payload.get("table_rows")))
        name = _optional_str(
            payload.get("table_title")
            or payload.get("title")
            or payload.get("name")
            or payload.get("proposed_table_name")
        ) or table_id
        routing_text = " ".join(
            item
            for item in [
                _optional_str(payload.get("routing_text") or payload.get("text")),
                *(_optional_str(card.get("routing_text") or card.get("text")) or "" for card in cards),
                _routing_text(name, merged_columns, len(rows) or _optional_int(payload.get("row_count"))),
            ]
            if item
        )
        tables.append(
            SQLTableContext(
                table_id=table_id,
                raw_source_name=name,
                display_name=name,
                document_id=_optional_str(payload.get("document_id")),
                source_uri=_optional_str(payload.get("source_uri")),
                source_locator=_dict_value(payload.get("source_locator")),
                rows=rows,
                columns=tuple(merged_columns),
                row_count=_optional_int(payload.get("row_count")) or len(rows) or None,
                column_count=_optional_int(payload.get("column_count")) or len(merged_columns) or None,
                routing_text=routing_text,
                schema_cards=cards,
                metadata=_dict_value(payload.get("metadata") or payload.get("metadata_json")),
            )
        )
    for table_id, cards in sorted(cards_by_table.items()):
        if table_id in table_payloads:
            continue
        table_card = cards[0]
        name = _optional_str(
            table_card.get("title")
            or table_card.get("proposed_table_name")
            or table_card.get("table_name")
        ) or table_id
        columns = _merge_columns({}, tuple(cards))
        routing_text = " ".join(
            _optional_str(card.get("routing_text") or card.get("text")) or "" for card in cards
        )
        tables.append(
            SQLTableContext(
                table_id=table_id,
                raw_source_name=name,
                display_name=name,
                document_id=_optional_str(table_card.get("document_id")),
                source_uri=_optional_str(table_card.get("source_uri")),
                source_locator=_dict_value(table_card.get("source_locator")),
                columns=tuple(columns),
                row_count=_optional_int(table_card.get("row_count")),
                column_count=len(columns) or None,
                routing_text=routing_text,
                schema_cards=tuple(cards),
                metadata=_dict_value(table_card.get("metadata") or table_card.get("metadata_json")),
            )
        )
    return tables


def _collect_payload(
    value: Any,
    *,
    table_payloads: dict[str, dict[str, Any]],
    cards_by_table: dict[str, list[dict[str, Any]]],
) -> None:
    if value is None:
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _collect_payload(item, table_payloads=table_payloads, cards_by_table=cards_by_table)
        return
    if not isinstance(value, Mapping):
        return
    payload = dict(value.get("payload")) if isinstance(value.get("payload"), Mapping) else dict(value)
    artifact_type = str(value.get("artifact_type") or payload.get("artifact_type") or "").lower()
    if _is_schema_card(artifact_type, payload):
        table_id = _optional_str(payload.get("table_id") or payload.get("artifact_id"))
        if table_id:
            cards_by_table[table_id].append(payload)
        return
    if _looks_like_table(payload) or artifact_type in {"table", "table_asset", "table_assets"}:
        table_id = _optional_str(payload.get("table_id") or payload.get("id") or value.get("artifact_id"))
        if table_id:
            table_payloads[table_id] = {**table_payloads.get(table_id, {}), **payload}
        return
    for key in ("tables", "table_assets", "raw_artifacts", "schema_routing_cards"):
        if key in value:
            _collect_payload(value.get(key), table_payloads=table_payloads, cards_by_table=cards_by_table)


def _score_table(table: SQLTableContext, question: str) -> SQLTableContext:
    q_tokens = _tokens(question)
    haystack = " ".join(
        [
            table.table_id,
            table.raw_source_name,
            table.display_name,
            table.routing_text,
            " ".join(str(column.get("name") or column.get("original_name") or "") for column in table.columns),
        ]
    )
    table_tokens = _tokens(haystack)
    overlap = q_tokens & table_tokens
    denominator = max(3, len(q_tokens))
    overlap_score = len(overlap) / denominator
    exact_bonus = 0.25 if table.table_id.lower() in question.lower() else 0.0
    numeric_bonus = 0.05 if _has_numeric_column(table) and (_tokens(question) & {"sum", "total", "avg", "average", "count", "top", "highest", "lowest"}) else 0.0
    score = min(1.0, overlap_score + exact_bonus + numeric_bonus)
    return SQLTableContext(
        **{
            **table.__dict__,
            "score": round(score, 6),
            "score_details": {
                "overlap_tokens": sorted(overlap),
                "overlap_score": round(overlap_score, 6),
                "exact_bonus": exact_bonus,
                "numeric_bonus": numeric_bonus,
            },
        }
    )


def _dedupe_tables(tables: Iterable[SQLTableContext]) -> list[SQLTableContext]:
    by_id: dict[str, SQLTableContext] = {}
    for table in tables:
        existing = by_id.get(table.table_id)
        if existing is None:
            by_id[table.table_id] = table
            continue
        rows = existing.rows or table.rows
        columns = existing.columns or table.columns
        by_id[table.table_id] = SQLTableContext(
            **{
                **existing.__dict__,
                "rows": rows,
                "columns": columns,
                "routing_text": " ".join(item for item in (existing.routing_text, table.routing_text) if item),
                "schema_cards": (*existing.schema_cards, *table.schema_cards),
            }
        )
    return list(by_id.values())


def _merge_columns(payload: Mapping[str, Any], cards: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = []
    columns.extend(_column_records(payload))
    for card in cards:
        structured_payload = _dict_value(card.get("structured_payload"))
        columns.extend(_column_records(structured_payload))
        if card.get("column_id") or card.get("column_name"):
            columns.append(
                {
                    "column_id": card.get("column_id"),
                    "name": card.get("column_name") or card.get("name"),
                    "canonical_name": card.get("canonical_name"),
                    "data_type": card.get("data_type"),
                    "semantic_role": card.get("semantic_role"),
                    "unit": card.get("unit"),
                    "period": card.get("period"),
                    "profile": card.get("profile"),
                }
            )
    if not columns:
        for row in _records_from(payload.get("rows") or payload.get("table_rows")):
            columns.extend({"name": key} for key in row)
            break
    deduped: dict[str, dict[str, Any]] = {}
    for index, column in enumerate(columns):
        name = _optional_str(
            column.get("original_name")
            or column.get("name")
            or column.get("column_name")
            or column.get("canonical_name")
        ) or f"column_{index + 1}"
        deduped.setdefault(name, {**column, "name": name, "column_index": column.get("column_index", index)})
    return list(deduped.values())


def _column_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    columns = payload.get("columns") or payload.get("table_columns") or ()
    if isinstance(columns, Mapping):
        return [{"name": key, **_dict_value(value)} for key, value in columns.items()]
    records = []
    if isinstance(columns, Sequence) and not isinstance(columns, (str, bytes, bytearray)):
        for index, column in enumerate(columns):
            if isinstance(column, Mapping):
                records.append(dict(column))
            else:
                records.append({"name": str(column), "column_index": index})
    return records


def _records_from(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _is_schema_card(artifact_type: str, payload: Mapping[str, Any]) -> bool:
    if artifact_type in {"schema_routing", "schema_routing_card"}:
        return True
    return str(payload.get("source_type") or "").lower() == "schema_routing_card"


def _looks_like_table(payload: Mapping[str, Any]) -> bool:
    return bool(payload.get("table_id") or payload.get("id")) and (
        "columns" in payload or "rows" in payload or "table_rows" in payload
    )


def _routing_text(name: str, columns: Sequence[Mapping[str, Any]], row_count: int | None) -> str:
    column_text = ", ".join(
        str(column.get("name") or column.get("column_name") or column.get("canonical_name") or "")
        for column in columns
    )
    return f"Table {name}: {row_count or 0} rows. Columns: {column_text}."


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", str(value).lower()) if len(token) > 1}


def _has_numeric_column(table: SQLTableContext) -> bool:
    return any(
        str(column.get("data_type") or "").lower() in {"number", "numeric", "integer", "float", "double"}
        for column in table.columns
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _dict_value(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
