from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from qdrant_client import QdrantClient, models

from atlas.core.config import Settings, bm25_sparse_enabled
from atlas.core.errors import AtlasError, ErrorCode
from atlas.db.models import Chunk, Document, ParentBlock
from atlas.embeddings.bm25_sparse import BM25SparseEncoder
from atlas.ingestion.chunker import chunk_text
from atlas.ingestion.contracts import (
    SUPPORTED_SUFFIXES,
    ChunkInput,
    DocumentSource,
    LoadedDocument,
    LoadedPage,
    StructuredArtifact,
)
from atlas.ingestion.path_policy import resolve_allowed_document_path
from atlas.ingestion.registry import (
    chunker_registry,
    document_loader_registry,
    document_parser_for_suffix,
    document_parser_registry,
    parent_block_builder_registry,
    structured_extractor_registry,
    vector_indexer_registry,
)
from atlas.ingestion.structured.adapters import csv_structured_artifacts
from atlas.vector.collections import ensure_chunk_collection


V4_PROFILE_SUFFIXES = {".csv", ".xlsx", ".html", ".htm"}
V4_PROFILE_SUPPORTED_SUFFIXES = SUPPORTED_SUFFIXES | V4_PROFILE_SUFFIXES
INGESTION_PROFILE_METADATA_KEY = "atlas_ingestion_profile"
INDEX_NAMESPACE_METADATA_KEY = "index_namespace"


@dataclass(frozen=True)
class LocalDocumentLoader:
    name: str = "local"

    def load(self, path_value: str, *, allowed_roots: list[Path]) -> DocumentSource:
        path = resolve_allowed_document_path(path_value, allowed_roots)
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise AtlasError(
                ErrorCode.UNSUPPORTED_FILE_TYPE,
                f"Unsupported file type: {suffix}. Atlas supports PDF, Markdown, and TXT.",
                status_code=400,
                details={"path": str(path), "supported": sorted(SUPPORTED_SUFFIXES)},
            )
        if not path.exists() or not path.is_file():
            raise AtlasError(
                ErrorCode.INVALID_REQUEST,
                f"File does not exist: {path}",
                status_code=400,
                details={"path": str(path)},
            )
        return DocumentSource(path=path, suffix=suffix, content=path.read_bytes())


@dataclass(frozen=True)
class V4ProfileLocalDocumentLoader:
    name: str = "local_v4_profile"

    def load(self, path_value: str, *, allowed_roots: list[Path]) -> DocumentSource:
        path = resolve_allowed_document_path(path_value, allowed_roots)
        suffix = path.suffix.lower()
        if suffix not in V4_PROFILE_SUPPORTED_SUFFIXES:
            raise AtlasError(
                ErrorCode.UNSUPPORTED_FILE_TYPE,
                (
                    f"Unsupported file type for V4 profile: {suffix}. "
                    "Atlas V4 profile supports PDF, Markdown, TXT, CSV, XLSX, and HTML."
                ),
                status_code=400,
                details={
                    "path": str(path),
                    "supported": sorted(V4_PROFILE_SUPPORTED_SUFFIXES),
                },
            )
        if not path.exists() or not path.is_file():
            raise AtlasError(
                ErrorCode.INVALID_REQUEST,
                f"File does not exist: {path}",
                status_code=400,
                details={"path": str(path)},
            )
        return DocumentSource(path=path, suffix=suffix, content=path.read_bytes())


@dataclass(frozen=True)
class MarkdownDocumentParser:
    name: str = "markdown"
    supported_suffixes: tuple[str, ...] = (".md", ".markdown")

    def parse(self, source: DocumentSource) -> LoadedDocument:
        text = source.content.decode("utf-8")
        return _loaded_text_document(source.path, text=text, file_type="markdown")


@dataclass(frozen=True)
class TxtDocumentParser:
    name: str = "txt"
    supported_suffixes: tuple[str, ...] = (".txt",)

    def parse(self, source: DocumentSource) -> LoadedDocument:
        text = source.content.decode("utf-8")
        return _loaded_text_document(source.path, text=text, file_type="txt")


@dataclass(frozen=True)
class PdfDocumentParser:
    name: str = "pdf"
    supported_suffixes: tuple[str, ...] = (".pdf",)

    def parse(self, source: DocumentSource) -> LoadedDocument:
        pages = _load_pdf_pages(source.path)
        text = "\n\n".join(page.text for page in pages if page.text.strip())
        return LoadedDocument(
            path=source.path,
            title=_extract_title(source.path, text),
            text=text,
            file_type="pdf",
            language=_detect_language(text),
            pages=pages,
        )


@dataclass(frozen=True)
class CsvDocumentParser:
    name: str = "csv"
    supported_suffixes: tuple[str, ...] = (".csv",)

    def parse(self, source: DocumentSource) -> LoadedDocument:
        text = _decode_text(source.content)
        try:
            rows = list(csv.reader(io.StringIO(text)))
        except csv.Error as exc:
            raise AtlasError(
                ErrorCode.INVALID_REQUEST,
                f"CSV parse failed: {source.path}",
                status_code=400,
                details={"path": str(source.path), "reason": str(exc)},
            ) from exc
        return _loaded_text_document(
            source.path,
            text=_rows_to_text(f"CSV table: {source.path.name}", rows),
            file_type="csv",
        )


@dataclass(frozen=True)
class HtmlDocumentParser:
    name: str = "html"
    supported_suffixes: tuple[str, ...] = (".html", ".htm")

    def parse(self, source: DocumentSource) -> LoadedDocument:
        parser = _HTMLTableTextParser()
        parser.feed(_decode_text(source.content))
        return _loaded_text_document(
            source.path,
            text=parser.to_text(title=source.path.name),
            file_type="html",
        )


@dataclass(frozen=True)
class XlsxDocumentParser:
    name: str = "xlsx"
    supported_suffixes: tuple[str, ...] = (".xlsx",)

    def parse(self, source: DocumentSource) -> LoadedDocument:
        try:
            with ZipFile(io.BytesIO(source.content)) as archive:
                shared_strings = _xlsx_shared_strings(archive)
                sheet_names = _xlsx_sheet_names(archive)
                sheet_texts = []
                for index, sheet_path in enumerate(_xlsx_sheet_paths(archive), start=1):
                    rows = _xlsx_rows(archive, sheet_path, shared_strings)
                    if not rows:
                        continue
                    title = (
                        sheet_names[index - 1]
                        if index <= len(sheet_names)
                        else Path(sheet_path).stem
                    )
                    sheet_texts.append(_rows_to_text(f"XLSX sheet: {title}", rows))
        except BadZipFile as exc:
            raise AtlasError(
                ErrorCode.INVALID_REQUEST,
                f"Invalid XLSX file: {source.path}",
                status_code=400,
                details={"path": str(source.path), "reason": "bad_zip_file"},
            ) from exc
        return _loaded_text_document(
            source.path,
            text="\n\n".join(sheet_texts),
            file_type="xlsx",
        )


@dataclass(frozen=True)
class DefaultChunker:
    name: str = "default"

    def chunk(
        self,
        loaded: LoadedDocument,
        *,
        target_tokens: int,
        overlap_tokens: int,
    ) -> list[ChunkInput]:
        if loaded.file_type == "pdf":
            items: list[ChunkInput] = []
            for page in loaded.pages:
                drafts = chunk_text(
                    page.text,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                )
                for draft in drafts:
                    items.append(
                        ChunkInput(
                            text=draft.text,
                            section_title=draft.section_title,
                            page_start=page.page_number,
                            page_end=page.page_number,
                            token_count=draft.token_count,
                        )
                    )
            return items

        drafts = chunk_text(
            loaded.text,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        return [
            ChunkInput(
                text=draft.text,
                section_title=draft.section_title,
                page_start=None,
                page_end=None,
                token_count=draft.token_count,
            )
            for draft in drafts
        ]


@dataclass(frozen=True)
class DefaultParentBlockBuilder:
    name: str = "default"

    def build(self, *, document_id: str, loaded: LoadedDocument) -> list[ParentBlock]:
        if loaded.file_type == "pdf":
            parents = []
            for page in loaded.pages:
                if page.page_number is None or not page.text.strip():
                    continue
                parents.append(
                    ParentBlock(
                        parent_id=parent_id(document_id, page.page_number),
                        document_id=document_id,
                        parent_type="page",
                        page_start=page.page_number,
                        page_end=page.page_number,
                        text=page.text,
                        child_ids_json=[],
                        metadata_json={
                            "source_path": str(loaded.path),
                            "title": loaded.title,
                            "parent_type": "page",
                        },
                    )
                )
            return parents

        return [
            ParentBlock(
                parent_id=parent_id(document_id, 0),
                document_id=document_id,
                parent_type="document",
                page_start=0,
                page_end=0,
                text=loaded.text,
                child_ids_json=[],
                metadata_json={
                    "source_path": str(loaded.path),
                    "title": loaded.title,
                    "parent_type": "document",
                },
            )
        ]


class QdrantVectorIndexer:
    name = "qdrant"

    def __init__(
        self,
        *,
        settings: Settings,
        qdrant: QdrantClient,
        sparse_encoder: BM25SparseEncoder | None = None,
    ) -> None:
        self.settings = settings
        self.qdrant = qdrant
        self.sparse_encoder = sparse_encoder

    def prepare(self, *, document: Document | None = None) -> None:
        ensure_chunk_collection(
            self.qdrant,
            self.settings,
            collection_name=(
                self._collection_for_document(document)
                if document is not None
                else self.settings.qdrant_collection
            ),
        )

    def index(
        self,
        *,
        document: Document,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        collection_name = self._collection_for_document(document)
        sparse_vectors = None
        if bm25_sparse_enabled(self.settings):
            sparse_vectors = self._sparse_encoder().embed_texts([chunk.text for chunk in chunks])

        points = []
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            payload = {
                "document_id": document.document_id,
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id,
                "title": document.title,
                "source_uri": document.source_uri,
                "file_type": document.file_type,
                "section_title": chunk.section_title,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "language": document.language,
                "embedding_model": chunk.embedding_model,
            }
            for metadata_key in (
                INGESTION_PROFILE_METADATA_KEY,
                INDEX_NAMESPACE_METADATA_KEY,
            ):
                metadata_value = _metadata_value(
                    chunk.metadata_json,
                    document.metadata_json,
                    key=metadata_key,
                )
                if metadata_value is not None:
                    payload[metadata_key] = metadata_value
            points.append(
                models.PointStruct(
                    id=qdrant_point_id(chunk.chunk_id),
                    vector=point_vector(
                        settings=self.settings,
                        dense_embedding=embedding,
                        sparse_vector=sparse_vectors[index] if sparse_vectors is not None else None,
                    ),
                    payload=payload,
                )
            )
        if points:
            ensure_chunk_collection(self.qdrant, self.settings, collection_name=collection_name)
            self.qdrant.upsert(collection_name=collection_name, points=points)

    def cleanup(self, chunks: list[Chunk]) -> None:
        points_by_collection: dict[str, list[str]] = {}
        for chunk in chunks:
            collection_name = self._collection_for_chunk(chunk)
            points_by_collection.setdefault(collection_name, []).append(
                qdrant_point_id(chunk.chunk_id)
            )
        if not points_by_collection:
            return
        for collection_name, point_ids in points_by_collection.items():
            try:
                self.qdrant.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=point_ids),
                )
            except Exception:
                pass

    def _sparse_encoder(self) -> BM25SparseEncoder:
        if self.sparse_encoder is None:
            raise RuntimeError("sparse_encoder_required_for_qdrant_vector_indexer")
        return self.sparse_encoder

    def _collection_for_document(self, document: Document) -> str:
        profile = _metadata_value(
            document.metadata_json,
            key=INGESTION_PROFILE_METADATA_KEY,
        )
        if str(profile or "").strip().lower() == "v4":
            return self.settings.v4_qdrant_collection
        return self.settings.qdrant_collection

    def _collection_for_chunk(self, chunk: Chunk) -> str:
        profile = _metadata_value(
            getattr(chunk, "metadata_json", {}),
            key=INGESTION_PROFILE_METADATA_KEY,
        )
        if str(profile or "").strip().lower() == "v4":
            return self.settings.v4_qdrant_collection
        return self.settings.qdrant_collection


@dataclass(frozen=True)
class NoopStructuredExtractor:
    name: str = "noop"

    def extract(self, loaded: LoadedDocument) -> list[StructuredArtifact]:
        return []


@dataclass(frozen=True)
class V4ProfileStructuredExtractor:
    name: str = "v4_profile"

    def extract(self, loaded: LoadedDocument) -> list[StructuredArtifact]:
        if loaded.file_type == "csv":
            return csv_structured_artifacts(loaded)
        return []


def register_builtin_ingestion_components() -> None:
    document_loader_registry.register_if_missing("local", LocalDocumentLoader())
    document_loader_registry.register_if_missing(
        "local_v4_profile",
        V4ProfileLocalDocumentLoader(),
    )
    document_parser_registry.register_if_missing("markdown", MarkdownDocumentParser())
    document_parser_registry.register_if_missing("txt", TxtDocumentParser())
    document_parser_registry.register_if_missing("pdf", PdfDocumentParser())
    document_parser_registry.register_if_missing("csv", CsvDocumentParser())
    document_parser_registry.register_if_missing("html", HtmlDocumentParser())
    document_parser_registry.register_if_missing("xlsx", XlsxDocumentParser())
    chunker_registry.register_if_missing("default", DefaultChunker())
    parent_block_builder_registry.register_if_missing("default", DefaultParentBlockBuilder())
    vector_indexer_registry.register_if_missing("qdrant", QdrantVectorIndexer)
    structured_extractor_registry.register_if_missing(
        "v4_profile",
        V4ProfileStructuredExtractor(),
    )
    structured_extractor_registry.register_if_missing("noop", NoopStructuredExtractor())


def load_local_document_with_v4_profile(
    path_value: str,
    *,
    allowed_roots: list[Path],
) -> LoadedDocument:
    source = document_loader_registry.get("local_v4_profile").load(
        path_value,
        allowed_roots=allowed_roots,
    )
    parser = document_parser_for_suffix(source.suffix)
    if parser is None:
        raise AtlasError(
            ErrorCode.UNSUPPORTED_FILE_TYPE,
            (
                f"Unsupported file type for V4 profile: {source.suffix}. "
                "Atlas V4 profile supports PDF, Markdown, TXT, CSV, XLSX, and HTML."
            ),
            status_code=400,
            details={
                "path": str(source.path),
                "supported": sorted(V4_PROFILE_SUPPORTED_SUFFIXES),
            },
        )

    loaded = parser.parse(source)
    if not loaded.text.strip():
        raise AtlasError(
            ErrorCode.INVALID_REQUEST,
            f"Document has no extractable text: {loaded.path}",
            status_code=400,
            details={"path": str(loaded.path)},
        )
    return loaded


def _loaded_text_document(path: Path, *, text: str, file_type: str) -> LoadedDocument:
    return LoadedDocument(
        path=path,
        title=_extract_title(path, text),
        text=text,
        file_type=file_type,
        language=_detect_language(text),
        pages=[LoadedPage(page_number=None, text=text)],
    )


def _load_pdf_pages(path: Path) -> list[LoadedPage]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise AtlasError(
            ErrorCode.CONFIGURATION_ERROR,
            "PDF support requires pypdf. Install dependencies with `python -m pip install -e .`.",
            status_code=500,
        ) from exc

    reader = PdfReader(str(path))
    pages: list[LoadedPage] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(LoadedPage(page_number=index, text=text))
    return pages


def _extract_title(path: Path, text: str) -> str:
    match = re.search(r"^\s*#\s+(.+?)\s*$", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return path.name


def _detect_language(text: str) -> str:
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-z]+", text))
    if cjk_chars >= latin_words:
        return "zh"
    return "en"


def _decode_text(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def _rows_to_text(title: str, rows: list[list[str]]) -> str:
    non_empty_rows = [
        [_normalize_cell(cell) for cell in row]
        for row in rows
        if any(str(cell).strip() for cell in row)
    ]
    if not non_empty_rows:
        return ""
    return "\n".join([title, *[" | ".join(row) for row in non_empty_rows]])


def _normalize_cell(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


class _HTMLTableTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._table_depth = 0
        self._current_rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None
        self._tables: list[list[list[str]]] = []
        self._fallback_text: list[str] = []

    def handle_starttag(self, tag: str, _attrs) -> None:
        tag = tag.lower()
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._current_rows = []
            return
        if self._table_depth <= 0:
            return
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"}:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        text = _normalize_cell(data)
        if not text:
            return
        if self._table_depth > 0 and self._current_cell is not None:
            self._current_cell.append(text)
        elif self._table_depth == 0:
            self._fallback_text.append(text)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._table_depth <= 0:
            return
        if tag in {"td", "th"} and self._current_cell is not None:
            if self._current_row is not None:
                self._current_row.append(" ".join(self._current_cell).strip())
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if any(cell.strip() for cell in self._current_row):
                self._current_rows.append(self._current_row)
            self._current_row = None
        elif tag == "table":
            if self._table_depth == 1 and self._current_rows:
                self._tables.append(self._current_rows)
            self._table_depth -= 1

    def to_text(self, *, title: str) -> str:
        if self._tables:
            return "\n\n".join(
                _rows_to_text(f"HTML table {index}: {title}", rows)
                for index, rows in enumerate(self._tables, start=1)
            )
        return "\n".join(self._fallback_text)


def _xlsx_sheet_paths(archive: ZipFile) -> list[str]:
    return sorted(
        name
        for name in archive.namelist()
        if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
    )


def _xlsx_sheet_names(archive: ZipFile) -> list[str]:
    if "xl/workbook.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    return [
        str(element.attrib.get("name", "")).strip()
        for element in root.iter()
        if _local_name(element.tag) == "sheet" and str(element.attrib.get("name", "")).strip()
    ]


def _xlsx_shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        _normalize_cell(" ".join(text for text in item.itertext()))
        for item in root.iter()
        if _local_name(item.tag) == "si"
    ]


def _xlsx_rows(
    archive: ZipFile,
    sheet_path: str,
    shared_strings: list[str],
) -> list[list[str]]:
    root = ElementTree.fromstring(archive.read(sheet_path))
    rows = []
    for row in root.iter():
        if _local_name(row.tag) != "row":
            continue
        values = [
            _xlsx_cell_value(cell, shared_strings)
            for cell in row
            if _local_name(cell.tag) == "c"
        ]
        if any(value.strip() for value in values):
            rows.append(values)
    return rows


def _xlsx_cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return _normalize_cell(" ".join(cell.itertext()))

    raw_value = ""
    for child in cell:
        if _local_name(child.tag) == "v":
            raw_value = child.text or ""
            break
    raw_value = raw_value.strip()
    if cell_type == "s" and raw_value.isdigit():
        index = int(raw_value)
        if index < len(shared_strings):
            return shared_strings[index]
    return _normalize_cell(raw_value)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _metadata_value(
    *metadata_objects: Any,
    key: str,
) -> Any | None:
    for metadata in metadata_objects:
        if isinstance(metadata, dict) and key in metadata:
            return metadata[key]
    return None


def parent_id_for_chunk(
    document_id: str,
    item: ChunkInput,
    parent_by_page: dict[int, ParentBlock],
) -> str | None:
    if isinstance(item.page_start, int) and item.page_start in parent_by_page:
        return parent_by_page[item.page_start].parent_id
    return parent_id(document_id, 0) if parent_by_page else None


def parent_id(document_id: str, page_number: int) -> str:
    raw = f"{document_id}:page:{page_number}"
    return f"par_{uuid5(NAMESPACE_URL, raw).hex[:32]}"


def point_vector(
    *,
    settings: Settings,
    dense_embedding: list[float],
    sparse_vector: models.SparseVector | None,
):
    if sparse_vector is None:
        return dense_embedding
    return {
        settings.qdrant_dense_vector_name: dense_embedding,
        settings.qdrant_sparse_vector_name: sparse_vector,
    }


def qdrant_point_id(chunk_id: str) -> str:
    raw_uuid = chunk_id.removeprefix("chk_")
    try:
        return str(UUID(raw_uuid))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, chunk_id))
