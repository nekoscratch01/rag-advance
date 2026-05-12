from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from atlas.retrieval.providers.sql.models import (
    SQLColumnContext,
    SQLSchemaContext,
    SQLTableContext,
)


RESERVED_WORDS = frozenset(
    {
        "all",
        "alter",
        "and",
        "attach",
        "by",
        "call",
        "cast",
        "copy",
        "create",
        "delete",
        "detach",
        "distinct",
        "drop",
        "except",
        "from",
        "group",
        "having",
        "insert",
        "install",
        "intersect",
        "join",
        "limit",
        "load",
        "or",
        "order",
        "pragma",
        "select",
        "table",
        "union",
        "update",
        "where",
        "with",
    }
)
SAFE_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


class IdentifierNormalizer:
    def normalize(self, table: SQLTableContext) -> SQLSchemaContext:
        table_base = safe_identifier_base(table.raw_source_name or table.table_id, fallback="table")
        safe_table = _dedupe_identifier(table_base, Counter())

        raw_columns = _columns_from_table(table)
        seen: Counter[str] = Counter()
        columns: list[SQLColumnContext] = []
        safe_to_raw: dict[str, dict[str, Any]] = {
            safe_table: {
                "object_type": "table",
                "table_id": table.table_id,
                "raw_source_name": table.raw_source_name,
                "display_name": table.display_name,
            }
        }
        raw_to_safe: dict[str, str] = {table.raw_source_name: safe_table, table.table_id: safe_table}
        ambiguous_raw_to_safe: dict[str, list[str]] = {}

        for index, column in enumerate(raw_columns):
            raw_name = _column_raw_name(column, index)
            display_name = _column_display_name(column, raw_name)
            safe_name = _dedupe_identifier(
                safe_identifier_base(raw_name, fallback=f"column_{index + 1}"),
                seen,
            )
            column_id = str(
                column.get("column_id")
                or column.get("id")
                or f"{table.table_id}:{index}:{safe_name}"
            )
            column_context = SQLColumnContext(
                column_id=column_id,
                raw_source_name=raw_name,
                display_name=display_name,
                safe_identifier=safe_name,
                column_index=_optional_int(column.get("column_index"), default=index),
                data_type=str(column.get("data_type") or "unknown"),
                semantic_role=_optional_str(column.get("semantic_role")),
                unit=_optional_str(column.get("unit")),
                period=_optional_str(column.get("period")),
                source_locator=_dict_value(column.get("source_locator")),
                metadata=_dict_value(column.get("metadata") or column.get("metadata_json")),
            )
            columns.append(column_context)
            safe_to_raw[safe_name] = {
                "object_type": "column",
                "table_id": table.table_id,
                "column_id": column_id,
                "raw_source_name": raw_name,
                "display_name": display_name,
                "column_index": column_context.column_index,
                "data_type": column_context.data_type,
            }
            _record_raw_to_safe(
                raw_to_safe,
                ambiguous_raw_to_safe,
                raw_name,
                safe_name,
            )
            canonical_name = _optional_str(column.get("canonical_name"))
            if canonical_name:
                _record_raw_to_safe(
                    raw_to_safe,
                    ambiguous_raw_to_safe,
                    canonical_name,
                    safe_name,
                )

        metadata = dict(table.metadata)
        if ambiguous_raw_to_safe:
            metadata["ambiguous_raw_to_safe"] = {
                raw_name: list(safe_names)
                for raw_name, safe_names in sorted(ambiguous_raw_to_safe.items())
            }
            ambiguous_safe_names = {
                safe_name
                for safe_names in ambiguous_raw_to_safe.values()
                for safe_name in safe_names
            }
            for safe_name in ambiguous_safe_names:
                if safe_name in safe_to_raw:
                    safe_to_raw[safe_name]["raw_name_ambiguous"] = True

        return SQLSchemaContext(
            table_id=table.table_id,
            raw_table_name=table.raw_source_name,
            display_table_name=table.display_name,
            safe_table_name=safe_table,
            document_id=table.document_id,
            source_uri=table.source_uri,
            source_locator=dict(table.source_locator),
            columns=tuple(columns),
            rows=tuple(dict(row) for row in table.rows),
            row_count=table.row_count,
            score=table.score,
            safe_to_raw=safe_to_raw,
            raw_to_safe=raw_to_safe,
            metadata=metadata,
        )


def safe_identifier_base(value: str, *, fallback: str = "identifier") -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    parts: list[str] = []
    last_was_separator = False
    for char in text:
        if "a" <= char <= "z" or "0" <= char <= "9":
            parts.append(char)
            last_was_separator = False
        elif char == "_":
            if not last_was_separator:
                parts.append("_")
            last_was_separator = True
        elif ord(char) > 127 and char.isalnum():
            if parts and not last_was_separator:
                parts.append("_")
            parts.append(f"u{ord(char):x}")
            parts.append("_")
            last_was_separator = True
        else:
            if parts and not last_was_separator:
                parts.append("_")
            last_was_separator = True
    base = re.sub(r"_+", "_", "".join(parts)).strip("_") or fallback
    if base[0].isdigit():
        base = f"c_{base}"
    if base in RESERVED_WORDS:
        base = f"{base}_col"
    if not SAFE_IDENTIFIER_RE.match(base):
        base = fallback
    return base


def is_safe_identifier(value: str) -> bool:
    return bool(SAFE_IDENTIFIER_RE.match(value)) and value not in RESERVED_WORDS


def _dedupe_identifier(base: str, seen: Counter[str]) -> str:
    safe = base if is_safe_identifier(base) else safe_identifier_base(base)
    seen[safe] += 1
    if seen[safe] == 1:
        return safe
    return f"{safe}_{seen[safe]}"


def _record_raw_to_safe(
    raw_to_safe: dict[str, str],
    ambiguous_raw_to_safe: dict[str, list[str]],
    raw_name: str,
    safe_name: str,
) -> None:
    existing = raw_to_safe.setdefault(raw_name, safe_name)
    if existing == safe_name:
        return
    values = ambiguous_raw_to_safe.setdefault(raw_name, [existing])
    if safe_name not in values:
        values.append(safe_name)


def _columns_from_table(table: SQLTableContext) -> tuple[dict[str, Any], ...]:
    if table.columns:
        return tuple(dict(column) for column in table.columns)
    keys: list[str] = []
    seen = set()
    for row in table.rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(str(key))
    return tuple({"name": key, "column_index": index} for index, key in enumerate(keys))


def _column_raw_name(column: Mapping[str, Any], index: int) -> str:
    for key in ("original_name", "name", "column_name", "canonical_name", "header"):
        value = _optional_str(column.get(key))
        if value:
            return value
    return f"column_{index + 1}"


def _column_display_name(column: Mapping[str, Any], raw_name: str) -> str:
    return (
        _optional_str(column.get("display_name"))
        or _optional_str(column.get("name"))
        or raw_name
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any, *, default: int | None = None) -> int | None:
    if isinstance(value, bool) or value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dict_value(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}
