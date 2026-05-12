from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atlas.retrieval.providers.sql.models import SQLColumnContext, SQLSchemaContext


@dataclass(frozen=True)
class DuckDBTableIndex:
    path: Path | None
    table_name: str
    warnings: tuple[str, ...] = ()


class DuckDBIndexBuilder:
    def __init__(self, *, index_dir: str | Path | None = None) -> None:
        self.index_dir = Path(index_dir) if index_dir else None

    def build(self, schema: SQLSchemaContext) -> DuckDBTableIndex:
        duckdb = _duckdb_module()
        if self.index_dir is None:
            return self._build_in_memory(duckdb, schema)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.index_dir / f"{schema.safe_table_name}_{_schema_hash(schema)}.duckdb"
        if db_path.exists():
            return DuckDBTableIndex(path=db_path, table_name=schema.safe_table_name)
        connection = duckdb.connect(str(db_path))
        try:
            _create_table(connection, schema)
        finally:
            connection.close()
        return DuckDBTableIndex(path=db_path, table_name=schema.safe_table_name)

    def _build_in_memory(self, duckdb: Any, schema: SQLSchemaContext) -> DuckDBTableIndex:
        connection = duckdb.connect(database=":memory:")
        try:
            _create_table(connection, schema)
            connection.close()
        finally:
            try:
                connection.close()
            except Exception:
                pass
        return DuckDBTableIndex(
            path=None,
            table_name=schema.safe_table_name,
            warnings=("duckdb_index_in_memory_not_reopenable_read_only",),
        )


def create_connection_for_schema(
    schema: SQLSchemaContext,
    *,
    index_dir: str | Path | None,
) -> tuple[Any, DuckDBTableIndex]:
    duckdb = _duckdb_module()
    if index_dir is None:
        connection = duckdb.connect(database=":memory:")
        _create_table(connection, schema)
        return connection, DuckDBTableIndex(
            path=None,
            table_name=schema.safe_table_name,
            warnings=("duckdb_runtime_in_memory_read_only_unavailable",),
        )
    index = DuckDBIndexBuilder(index_dir=index_dir).build(schema)
    connection = duckdb.connect(str(index.path), read_only=True)
    return connection, index


def _duckdb_module() -> Any:
    try:
        import duckdb  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "optional_dependency_missing: duckdb is required for SQLProvider execution; "
            "install atlas-rag-kernel[structured-sql]"
        ) from exc
    return duckdb


def _create_table(connection: Any, schema: SQLSchemaContext) -> None:
    columns_sql = ", ".join(
        f"{column.safe_identifier} {_duckdb_type(column)}" for column in schema.columns
    )
    connection.execute(f"DROP TABLE IF EXISTS {schema.safe_table_name}")
    connection.execute(f"CREATE TABLE {schema.safe_table_name} ({columns_sql})")
    if not schema.rows:
        return
    placeholders = ", ".join("?" for _ in schema.columns)
    insert_sql = f"INSERT INTO {schema.safe_table_name} VALUES ({placeholders})"
    rows = [
        tuple(_coerce_value(_row_value(row, column), column) for column in schema.columns)
        for row in schema.rows
    ]
    if rows:
        connection.executemany(insert_sql, rows)


def _duckdb_type(column: SQLColumnContext) -> str:
    data_type = str(column.data_type or "").lower()
    if data_type in {"integer", "int", "bigint"}:
        return "BIGINT"
    if data_type in {"number", "numeric", "float", "double", "decimal"}:
        return "DOUBLE"
    if data_type == "boolean":
        return "BOOLEAN"
    return "VARCHAR"


def _coerce_value(value: Any, column: SQLColumnContext) -> Any:
    data_type = str(column.data_type or "").lower()
    if value is None or value == "":
        return None
    if data_type in {"integer", "int", "bigint"}:
        parsed = _parse_number(value)
        return int(parsed) if parsed is not None else None
    if data_type in {"number", "numeric", "float", "double", "decimal"}:
        return _parse_number(value)
    if data_type == "boolean":
        lowered = str(value).strip().lower()
        if lowered in {"true", "t", "yes", "y", "1"}:
            return True
        if lowered in {"false", "f", "no", "n", "0"}:
            return False
        return None
    return str(value)


def _parse_number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"
    text = text.replace("$", "").replace("%", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _row_value(row: dict[str, Any], column: SQLColumnContext) -> Any:
    for key in (
        column.raw_source_name,
        column.display_name,
        column.safe_identifier,
        column.column_id,
    ):
        if key in row:
            return row[key]
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in (column.raw_source_name, column.display_name, column.safe_identifier):
        if str(key).lower() in lowered:
            return lowered[str(key).lower()]
    return None


def _schema_hash(schema: SQLSchemaContext) -> str:
    payload = {
        "table_id": schema.table_id,
        "columns": [
            {
                "id": column.column_id,
                "raw": column.raw_source_name,
                "safe": column.safe_identifier,
                "type": column.data_type,
            }
            for column in schema.columns
        ],
        "rows": schema.rows,
    }
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]
