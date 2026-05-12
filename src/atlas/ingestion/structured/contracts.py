from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypeAlias

from atlas.ingestion.contracts import LoadedDocument, LoadedPage, StructuredArtifact


IR_VERSION = "v4.phase1"

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]


def stable_id(prefix: str, *parts: JSONValue, length: int = 16) -> str:
    if not prefix:
        raise ValueError("stable_id prefix is required")
    if length <= 0:
        raise ValueError("stable_id length must be positive")
    joined = "\x1f".join(_canonical_text(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"


def content_hash(value: str | bytes | JSONValue) -> str:
    if isinstance(value, bytes):
        data = value
    elif isinstance(value, str):
        data = value.encode("utf-8")
    else:
        data = _canonical_json_bytes(value)
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class SourceLocator:
    source_uri: str | None = None
    source_path: str | None = None
    document_id: str | None = None
    storage_ref: str | None = None
    storage_format: str | None = None
    storage_offset: int | None = None
    storage_length: int | None = None
    page_number: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    sheet_name: str | None = None
    element_id: str | None = None
    table_id: str | None = None
    table_range: JSONValue = None
    row_index: int | None = None
    row_locator: dict[str, JSONValue] | None = None
    column_index: int | None = None
    column_locator: dict[str, JSONValue] | None = None
    cell_ref: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    bbox: dict[str, float] | None = None
    locator_precision: str = "unknown"
    locator_confidence: float = 0.0
    is_exact: bool = False
    locator_method: str = "unspecified"
    locator_version: str = IR_VERSION

    def to_payload(self) -> dict[str, JSONValue]:
        return _payload(self)


@dataclass(frozen=True)
class ProvenancePolicy:
    policy_id: str = "v4_phase1_layered_provenance_c"
    require_source_locator: bool = True
    require_storage_locator: bool = True
    require_table_locator: bool = True
    require_column_locator: bool = True
    require_page_locator: bool = False
    require_row_locator_for_tables: bool = False
    require_cell_locator_for_tables: bool = False
    require_row_locator_for_materialized_rows: bool = True
    require_cell_locator_for_materialized_cells: bool = True
    materialization_policy: str = "facts"
    row_locator_policy: str = "required_when_materialized"
    cell_locator_policy: str = "required_when_materialized"
    allow_document_level_fallback: bool = True
    parser_name: str | None = None
    parser_version: str | None = None
    notes: list[str] = field(default_factory=list)
    policy_version: str = IR_VERSION

    def to_payload(self) -> dict[str, JSONValue]:
        return _payload(self)


@dataclass(frozen=True)
class TableColumnIR:
    column_id: str = ""
    table_id: str = ""
    document_id: str = ""
    name: str = ""
    original_name: str | None = None
    column_index: int | None = None
    canonical_name: str | None = None
    data_type: str = "unknown"
    semantic_role: str | None = None
    unit: str | None = None
    period: str | None = None
    nullable: bool | None = None
    profile: dict[str, JSONValue] = field(default_factory=dict)
    source_locator: SourceLocator = field(default_factory=SourceLocator)
    content_hash: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def to_payload(self) -> dict[str, JSONValue]:
        return _payload(self)


@dataclass(frozen=True)
class TableIR:
    id: str = ""
    table_id: str = ""
    document_id: str = ""
    source_type: str | None = None
    source_uri: str | None = None
    title: str | None = None
    table_title: str | None = None
    name: str | None = None
    columns: list[TableColumnIR] = field(default_factory=list)
    rows: list[dict[str, JSONValue]] = field(default_factory=list)
    row_count: int | None = None
    column_count: int | None = None
    text: str = ""
    source_locator: SourceLocator = field(default_factory=SourceLocator)
    source_element_ids: tuple[str, ...] = field(default_factory=tuple)
    extraction_method: str | None = None
    extraction_confidence: float | None = None
    content_hash: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    metadata_json: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        resolved_table_id = self.table_id or self.id
        resolved_id = self.id or resolved_table_id
        if resolved_table_id:
            object.__setattr__(self, "table_id", resolved_table_id)
        if resolved_id:
            object.__setattr__(self, "id", resolved_id)
        if self.table_title is None and self.title is not None:
            object.__setattr__(self, "table_title", self.title)
        if self.title is None and self.table_title is not None:
            object.__setattr__(self, "title", self.table_title)
        if self.name is None and self.title is not None:
            object.__setattr__(self, "name", self.title)
        if self.row_count is None:
            object.__setattr__(self, "row_count", len(self.rows))
        if self.column_count is None:
            object.__setattr__(self, "column_count", len(self.columns))
        if not self.metadata_json and self.metadata:
            object.__setattr__(self, "metadata_json", dict(self.metadata))

    @classmethod
    def from_rows(
        cls,
        *,
        rows: list[dict[str, JSONValue]],
        columns: list[TableColumnIR] | None = None,
        title: str | None = None,
        table_id: str | None = None,
        text: str = "",
        source_locator: SourceLocator | None = None,
        extraction_method: str | None = None,
        extraction_confidence: float | None = None,
        metadata: dict[str, JSONValue] | None = None,
    ) -> TableIR:
        table_hash = content_hash(
            {
                "title": title,
                "columns": [column.to_payload() for column in columns or []],
                "rows": rows,
                "text": text,
            }
        )
        return cls(
            table_id=table_id or stable_id("tbl", table_hash),
            title=title,
            columns=columns or [],
            rows=rows,
            text=text,
            source_locator=source_locator or SourceLocator(),
            extraction_method=extraction_method,
            extraction_confidence=extraction_confidence,
            content_hash=table_hash,
            metadata=metadata or {},
        )

    def to_payload(self) -> dict[str, JSONValue]:
        return _payload(self)

    def to_structured_artifact(
        self,
        *,
        provenance_policy: ProvenancePolicy | None = None,
        schema_routing_card: SchemaRoutingCard | None = None,
        artifact_manifest: ArtifactManifest | None = None,
        metadata: dict[str, JSONValue] | None = None,
    ) -> StructuredArtifact:
        return StructuredArtifact(
            artifact_type="table",
            payload=self.to_payload(),
            artifact_id=self.table_id or None,
            content_hash=self.content_hash,
            source_locator=self.source_locator.to_payload(),
            provenance_policy=(
                provenance_policy.to_payload() if provenance_policy is not None else None
            ),
            schema_routing_card=(
                schema_routing_card.to_payload() if schema_routing_card is not None else None
            ),
            artifact_manifest=(
                artifact_manifest.to_payload() if artifact_manifest is not None else None
            ),
            envelope_version=IR_VERSION,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class TableRowIR:
    row_id: str = ""
    table_id: str = ""
    document_id: str = ""
    row_index: int | None = None
    row_hash: str | None = None
    source_locator: SourceLocator = field(default_factory=SourceLocator)
    cell_ids: tuple[str, ...] = field(default_factory=tuple)
    materialized: bool = False
    values: dict[str, JSONValue] = field(default_factory=dict)
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.row_hash is None:
            object.__setattr__(self, "row_hash", content_hash(self.values))
        if not self.row_id:
            object.__setattr__(
                self,
                "row_id",
                stable_id("row", self.table_id, self.row_index, self.row_hash),
            )
        object.__setattr__(self, "cell_ids", _string_tuple(self.cell_ids))

    def to_payload(self) -> dict[str, JSONValue]:
        return _payload(self)


@dataclass(frozen=True)
class TableCellIR:
    cell_id: str = ""
    table_id: str = ""
    document_id: str = ""
    row_id: str | None = None
    row_index: int | None = None
    row_hash: str | None = None
    column_id: str | None = None
    column_name: str | None = None
    column_index: int | None = None
    cell_ref: str | None = None
    value: JSONValue = None
    value_hash: str | None = None
    source_locator: SourceLocator = field(default_factory=SourceLocator)
    materialized: bool = False
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.value_hash is None:
            object.__setattr__(self, "value_hash", content_hash(self.value))
        if not self.cell_id:
            object.__setattr__(
                self,
                "cell_id",
                stable_id(
                    "cell",
                    self.table_id,
                    self.row_index,
                    self.row_hash,
                    self.column_id or self.column_name or self.column_index,
                    self.cell_ref,
                    self.value_hash,
                ),
            )

    def to_payload(self) -> dict[str, JSONValue]:
        return _payload(self)


@dataclass(frozen=True)
class DocumentElementIR:
    element_id: str = ""
    element_type: str = "text"
    text: str = ""
    order_index: int | None = None
    section_title: str | None = None
    heading_level: int | None = None
    table: TableIR | None = None
    source_locator: SourceLocator = field(default_factory=SourceLocator)
    content_hash: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def to_payload(self) -> dict[str, JSONValue]:
        return _payload(self)


@dataclass(frozen=True)
class ParentChunk:
    id: str = ""
    parent_chunk_id: str = ""
    parent_id: str = ""
    chunk_id: str = ""
    document_id: str = ""
    text: str = ""
    chunk_type: str = "parent"
    kind: str = "parent"
    section_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    token_count: int = 0
    child_ids: tuple[str, ...] = field(default_factory=tuple)
    child_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    indexable: bool = False
    main_index: bool = False
    include_in_main_index: bool = False
    index_policy: str = "children_only"
    source_locator: SourceLocator = field(default_factory=SourceLocator)
    source_element_ids: tuple[str, ...] = field(default_factory=tuple)
    content_hash: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    metadata_json: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        resolved_id = self.parent_chunk_id or self.parent_id or self.chunk_id or self.id
        if resolved_id:
            for field_name in ("id", "parent_chunk_id", "parent_id", "chunk_id"):
                if not getattr(self, field_name):
                    object.__setattr__(self, field_name, resolved_id)
        child_ids = _string_tuple(self.child_ids or self.child_chunk_ids)
        child_chunk_ids = _string_tuple(self.child_chunk_ids or child_ids)
        object.__setattr__(self, "child_ids", child_ids)
        object.__setattr__(self, "child_chunk_ids", child_chunk_ids)
        object.__setattr__(self, "source_element_ids", _string_tuple(self.source_element_ids))
        if self.content_hash is None:
            object.__setattr__(self, "content_hash", content_hash(self.text))
        if not self.metadata_json and self.metadata:
            object.__setattr__(self, "metadata_json", dict(self.metadata))

    def to_payload(self) -> dict[str, JSONValue]:
        return _payload(self)


@dataclass(frozen=True)
class ChildChunk:
    id: str = ""
    child_chunk_id: str = ""
    chunk_id: str = ""
    parent_chunk_id: str = ""
    parent_id: str = ""
    document_id: str = ""
    text: str = ""
    chunk_index: int = 0
    child_index: int = 0
    section_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    token_count: int = 0
    indexable: bool = True
    main_index: bool = True
    include_in_main_index: bool = True
    index_policy: str = "ranked_child"
    source_locator: SourceLocator = field(default_factory=SourceLocator)
    source_element_ids: tuple[str, ...] = field(default_factory=tuple)
    content_hash: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    metadata_json: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        resolved_child_id = self.child_chunk_id or self.chunk_id or self.id
        if resolved_child_id:
            for field_name in ("id", "child_chunk_id", "chunk_id"):
                if not getattr(self, field_name):
                    object.__setattr__(self, field_name, resolved_child_id)
        resolved_parent_id = self.parent_chunk_id or self.parent_id
        if resolved_parent_id:
            if not self.parent_chunk_id:
                object.__setattr__(self, "parent_chunk_id", resolved_parent_id)
            if not self.parent_id:
                object.__setattr__(self, "parent_id", resolved_parent_id)
        object.__setattr__(self, "source_element_ids", _string_tuple(self.source_element_ids))
        if self.content_hash is None:
            object.__setattr__(self, "content_hash", content_hash(self.text))
        if not self.metadata_json and self.metadata:
            object.__setattr__(self, "metadata_json", dict(self.metadata))

    def to_payload(self) -> dict[str, JSONValue]:
        return _payload(self)


@dataclass(frozen=True)
class SchemaRoutingCard:
    id: str = ""
    card_id: str = ""
    schema_card_id: str = ""
    artifact_id: str | None = None
    artifact_type: str = "table"
    table_id: str = ""
    document_id: str = ""
    card_type: str = "schema_routing"
    source_type: str = "schema_routing_card"
    semantic_domain: str | None = None
    proposed_schema_name: str | None = None
    proposed_table_name: str | None = None
    primary_key_candidates: list[str] = field(default_factory=list)
    measure_columns: list[str] = field(default_factory=list)
    dimension_columns: list[str] = field(default_factory=list)
    period_columns: list[str] = field(default_factory=list)
    confidence: float | None = None
    title: str = ""
    text: str = ""
    routing_text: str = ""
    routing_only: bool = True
    source_derived_text: str = ""
    computed_from_source_text: str = ""
    inferred_text: str = ""
    value_answer_evidence_allowed: bool = False
    schema_answer_evidence_allowed: bool = True
    answer_evidence_allowed: bool = False
    index_object_type: str = "schema_routing"
    evidence_role: str = "routing_only"
    structured_payload: dict[str, JSONValue] = field(default_factory=dict)
    source_locator: SourceLocator = field(default_factory=SourceLocator)
    source_element_ids: tuple[str, ...] = field(default_factory=tuple)
    content_hash: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    metadata_json: dict[str, JSONValue] = field(default_factory=dict)
    routing_version: str = IR_VERSION

    def __post_init__(self) -> None:
        resolved_card_id = self.card_id or self.schema_card_id or self.id
        if resolved_card_id:
            for field_name in ("id", "card_id", "schema_card_id"):
                if not getattr(self, field_name):
                    object.__setattr__(self, field_name, resolved_card_id)
        if self.artifact_id is None and self.table_id:
            object.__setattr__(self, "artifact_id", self.table_id)
        if not self.proposed_table_name and self.table_id:
            object.__setattr__(self, "proposed_table_name", self.table_id)
        resolved_text = self.routing_text or self.text
        if resolved_text:
            if not self.routing_text:
                object.__setattr__(self, "routing_text", resolved_text)
            if not self.text:
                object.__setattr__(self, "text", resolved_text)
            if not self.source_derived_text:
                object.__setattr__(self, "source_derived_text", resolved_text)
            if not self.computed_from_source_text:
                object.__setattr__(self, "computed_from_source_text", resolved_text)
            if not self.inferred_text:
                object.__setattr__(self, "inferred_text", resolved_text)
        object.__setattr__(self, "routing_only", True)
        object.__setattr__(self, "source_type", "schema_routing_card")
        object.__setattr__(self, "value_answer_evidence_allowed", False)
        object.__setattr__(self, "schema_answer_evidence_allowed", True)
        object.__setattr__(self, "answer_evidence_allowed", False)
        object.__setattr__(self, "index_object_type", "schema_routing")
        object.__setattr__(self, "evidence_role", "routing_only")
        object.__setattr__(self, "source_element_ids", _string_tuple(self.source_element_ids))
        if self.content_hash is None:
            object.__setattr__(
                self,
                "content_hash",
                content_hash(
                    {
                        "card_id": self.card_id,
                        "card_type": self.card_type,
                        "table_id": self.table_id,
                        "text": self.text,
                        "structured_payload": self.structured_payload,
                    }
                ),
            )
        if not self.metadata_json and self.metadata:
            object.__setattr__(self, "metadata_json", dict(self.metadata))

    def to_payload(self) -> dict[str, JSONValue]:
        return _payload(self)


@dataclass(frozen=True)
class TableCard(SchemaRoutingCard):
    card_type: str = "table_schema"


@dataclass(frozen=True)
class ColumnCard(SchemaRoutingCard):
    card_type: str = "column_schema"
    column_id: str | None = None
    column_name: str | None = None


@dataclass(frozen=True)
class ProfileCard(SchemaRoutingCard):
    card_type: str = "table_profile"


@dataclass(frozen=True)
class ArtifactManifest:
    manifest_id: str = ""
    document_id: str | None = None
    artifact_ids: list[str] = field(default_factory=list)
    table_ids: list[str] = field(default_factory=list)
    schema_routing_cards: list[SchemaRoutingCard] = field(default_factory=list)
    provenance_policy: ProvenancePolicy = field(default_factory=ProvenancePolicy)
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    manifest_version: str = IR_VERSION

    def to_payload(self) -> dict[str, JSONValue]:
        return _payload(self)


@dataclass(frozen=True)
class ParsedDocumentIR:
    document_id: str = ""
    path: str = ""
    title: str = ""
    text: str = ""
    file_type: str = ""
    language: str = ""
    elements: list[DocumentElementIR] = field(default_factory=list)
    source_locator: SourceLocator = field(default_factory=SourceLocator)
    provenance_policy: ProvenancePolicy = field(default_factory=ProvenancePolicy)
    schema_routing_cards: list[SchemaRoutingCard] = field(default_factory=list)
    artifact_manifest: ArtifactManifest | None = None
    content_hash: str | None = None
    parser_name: str | None = None
    parser_version: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    ir_version: str = IR_VERSION

    @classmethod
    def from_loaded_document(
        cls,
        loaded: LoadedDocument,
        *,
        document_id: str | None = None,
        source_uri: str | None = None,
        parser_name: str | None = None,
        parser_version: str | None = None,
        provenance_policy: ProvenancePolicy | None = None,
        metadata: dict[str, JSONValue] | None = None,
    ) -> ParsedDocumentIR:
        page_payload = [
            {"page_number": page.page_number, "text": page.text} for page in loaded.pages
        ]
        document_hash = content_hash({"text": loaded.text, "pages": page_payload})
        resolved_document_id = document_id or stable_id(
            "docir",
            source_uri or f"local:{loaded.path}",
            document_hash,
        )
        policy = provenance_policy or ProvenancePolicy(
            parser_name=parser_name,
            parser_version=parser_version,
        )
        source_locator = SourceLocator(
            source_uri=source_uri,
            source_path=str(loaded.path),
            document_id=resolved_document_id,
        )
        elements = _legacy_page_elements(
            loaded=loaded,
            document_id=resolved_document_id,
            source_uri=source_uri,
        )
        artifact_manifest = ArtifactManifest(
            manifest_id=stable_id("manifest", resolved_document_id, document_hash),
            document_id=resolved_document_id,
            artifact_ids=[element.element_id for element in elements if element.element_id],
            provenance_policy=policy,
        )
        return cls(
            document_id=resolved_document_id,
            path=str(loaded.path),
            title=loaded.title,
            text=loaded.text,
            file_type=loaded.file_type,
            language=loaded.language,
            elements=elements,
            source_locator=source_locator,
            provenance_policy=policy,
            artifact_manifest=artifact_manifest,
            content_hash=document_hash,
            parser_name=parser_name,
            parser_version=parser_version,
            metadata=metadata or {},
        )

    def to_loaded_document(self) -> LoadedDocument:
        page_elements = [
            element
            for element in self.elements
            if element.element_type == "page_text"
            or element.metadata.get("legacy_projection") == "page"
        ]
        page_elements.sort(
            key=lambda element: (
                element.order_index if element.order_index is not None else 10**9,
                element.source_locator.page_number
                if element.source_locator.page_number is not None
                else 10**9,
            )
        )
        pages = [
            LoadedPage(
                page_number=element.source_locator.page_number,
                text=element.text,
            )
            for element in page_elements
        ]
        return LoadedDocument(
            path=Path(self.path or self.source_locator.source_path or ""),
            title=self.title,
            text=self.text,
            file_type=self.file_type,
            language=self.language,
            pages=pages,
        )

    def to_payload(self) -> dict[str, JSONValue]:
        return _payload(self)


def parsed_document_from_loaded_document(
    loaded: LoadedDocument,
    *,
    document_id: str | None = None,
    source_uri: str | None = None,
    parser_name: str | None = None,
    parser_version: str | None = None,
    provenance_policy: ProvenancePolicy | None = None,
    metadata: dict[str, JSONValue] | None = None,
) -> ParsedDocumentIR:
    return ParsedDocumentIR.from_loaded_document(
        loaded,
        document_id=document_id,
        source_uri=source_uri,
        parser_name=parser_name,
        parser_version=parser_version,
        provenance_policy=provenance_policy,
        metadata=metadata,
    )


def legacy_loaded_document_projection(parsed: ParsedDocumentIR) -> LoadedDocument:
    return parsed.to_loaded_document()


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return (str(value),)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


def _legacy_page_elements(
    *,
    loaded: LoadedDocument,
    document_id: str,
    source_uri: str | None,
) -> list[DocumentElementIR]:
    elements: list[DocumentElementIR] = []
    for index, page in enumerate(loaded.pages):
        element_hash = content_hash(page.text)
        element_id = stable_id(
            "el",
            document_id,
            "page_text",
            index,
            page.page_number,
            element_hash,
        )
        locator = SourceLocator(
            source_uri=source_uri,
            source_path=str(loaded.path),
            document_id=document_id,
            page_number=page.page_number,
            page_start=page.page_number,
            page_end=page.page_number,
            element_id=element_id,
        )
        elements.append(
            DocumentElementIR(
                element_id=element_id,
                element_type="page_text",
                text=page.text,
                order_index=index,
                source_locator=locator,
                content_hash=element_hash,
                metadata={"legacy_projection": "page"},
            )
        )
    return elements


def _payload(value: object) -> dict[str, JSONValue]:
    data = asdict(value)
    payload = _canonical_json_value(data)
    if not isinstance(payload, dict):
        raise TypeError("V4 contract payload serialization must produce a mapping")
    return payload


def _canonical_text(value: JSONValue) -> str:
    return _canonical_json_bytes(value).decode("utf-8")


def _canonical_json_bytes(value: JSONValue) -> bytes:
    return json.dumps(
        _canonical_json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_json_value(item) for item in value]
    raise TypeError(
        "V4 contract payloads only accept JSON-compatible primitives, lists, and mappings"
    )
