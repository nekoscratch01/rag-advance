from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FinanceBenchDocument:
    document_id: str
    doc_name: str
    doc_link: str | None
    source_url: str | None
    fallback_url: str
    pdf_sha256: str | None
    status: str
    row_ids: list[str]
    local_pdf_path: str | None = None
    parser_name: str | None = None
    parser_version: str | None = None
    corpus_version: str | None = None
    byte_count: int | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    duplicate_of: str | None = None
    aliases: list[str] | None = None
    download_attempts: list[dict[str, Any]] | None = None
    failure_kind: str | None = None
    failure_reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FinanceBenchPage:
    page_id: str
    document_id: str
    doc_name: str
    page_number: int
    text: str
    text_sha256: str
    pdf_sha256: str
    source_uri: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FinanceBenchParentBlock:
    parent_id: str
    document_id: str
    parent_type: str
    page_start: int
    page_end: int
    text: str
    child_ids_json: list[str]
    metadata_json: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FinanceBenchChildChunk:
    chunk_id: str
    parent_id: str
    document_id: str
    doc_name: str
    chunk_index: int
    text: str
    text_hash: str
    section_title: str | None
    page_start: int
    page_end: int
    token_count: int
    pdf_sha256: str
    source_uri: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FinanceBenchChunk:
    chunk_id: str
    document_id: str
    doc_name: str
    chunk_index: int
    text: str
    text_hash: str
    section_title: str | None
    page_start: int
    page_end: int
    token_count: int
    pdf_sha256: str
    source_uri: str | None
    parent_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FinanceBenchPrepareResult:
    dataset_id: str
    revision: str | None
    row_count: int
    manifest_count: int
    page_count: int
    chunk_count: int
    failure_count: int
    out_dir: str
    evals_path: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
