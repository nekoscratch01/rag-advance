from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


SQL_PROVIDER = "sql"
SQL_PROVIDER_VERSION = "sql_provider_v1"

SQLProviderStatus = Literal[
    "success",
    "skipped_not_table_query",
    "cannot_answer_no_table",
    "cannot_answer_low_confidence",
    "unsupported_multi_table",
    "compiler_failed",
    "validation_failed",
    "execution_failed",
    "timeout",
]

SQL_PROVIDER_STATUSES: tuple[SQLProviderStatus, ...] = (
    "success",
    "skipped_not_table_query",
    "cannot_answer_no_table",
    "cannot_answer_low_confidence",
    "unsupported_multi_table",
    "compiler_failed",
    "validation_failed",
    "execution_failed",
    "timeout",
)


@dataclass(frozen=True)
class SQLIntentDecision:
    allowed: bool
    status: SQLProviderStatus
    reason: str
    intent_type: str | None = None
    signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class SQLColumnContext:
    column_id: str
    raw_source_name: str
    display_name: str
    safe_identifier: str
    column_index: int | None = None
    data_type: str = "unknown"
    semantic_role: str | None = None
    unit: str | None = None
    period: str | None = None
    source_locator: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SQLTableContext:
    table_id: str
    raw_source_name: str
    display_name: str
    document_id: str | None = None
    source_uri: str | None = None
    source_locator: dict[str, Any] = field(default_factory=dict)
    rows: tuple[dict[str, Any], ...] = ()
    columns: tuple[dict[str, Any], ...] = ()
    row_count: int | None = None
    column_count: int | None = None
    routing_text: str = ""
    schema_cards: tuple[dict[str, Any], ...] = ()
    score: float = 0.0
    score_details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SQLSchemaContext:
    table_id: str
    raw_table_name: str
    display_table_name: str
    safe_table_name: str
    document_id: str | None
    source_uri: str | None
    source_locator: dict[str, Any]
    columns: tuple[SQLColumnContext, ...]
    rows: tuple[dict[str, Any], ...]
    row_count: int | None
    score: float
    safe_to_raw: dict[str, dict[str, Any]]
    raw_to_safe: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def column_ids_by_safe_name(self) -> dict[str, str]:
        return {column.safe_identifier: column.column_id for column in self.columns}

    @property
    def column_by_safe_name(self) -> dict[str, SQLColumnContext]:
        return {column.safe_identifier: column for column in self.columns}

    @property
    def safe_column_names(self) -> tuple[str, ...]:
        return tuple(column.safe_identifier for column in self.columns)

    @property
    def safe_to_raw_identifier_map(self) -> dict[str, dict[str, Any]]:
        return self.safe_to_raw


@dataclass(frozen=True)
class SchemaRouteResult:
    status: SQLProviderStatus
    table: SQLTableContext | None = None
    reason: str | None = None
    candidates: tuple[SQLTableContext, ...] = ()
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SQLDraft:
    status: SQLProviderStatus
    sql: str | None = None
    raw_output: str | None = None
    reason: str | None = None
    compiler_version: str = "atlas_sql_compiler_v1"
    prompt: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SQLValidationResult:
    valid: bool
    status: SQLProviderStatus
    sql: str | None = None
    reason: str | None = None
    used_column_ids: tuple[str, ...] = ()
    used_safe_columns: tuple[str, ...] = ()
    limit: int | None = None
    validator_backend: str = "fallback"
    warnings: tuple[str, ...] = ()
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SQLExecutionResult:
    status: SQLProviderStatus
    columns: tuple[str, ...] = ()
    rows: tuple[dict[str, Any], ...] = ()
    row_count: int = 0
    truncated: bool = False
    result_bytes: int = 0
    latency_ms: int = 0
    warnings: tuple[str, ...] = ()
    reason: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SQLResultEvidence:
    evidence_id: str
    table_id: str
    text: str
    sql: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    structured_payload: dict[str, Any]
