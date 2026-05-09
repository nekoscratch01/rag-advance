from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from uuid import NAMESPACE_URL, UUID, uuid5

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
    document_parser_registry,
    parent_block_builder_registry,
    structured_extractor_registry,
    vector_indexer_registry,
)
from atlas.vector.collections import ensure_chunk_collection


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

    def prepare(self) -> None:
        ensure_chunk_collection(self.qdrant, self.settings)

    def index(
        self,
        *,
        document: Document,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        sparse_vectors = None
        if bm25_sparse_enabled(self.settings):
            sparse_vectors = self._sparse_encoder().embed_texts([chunk.text for chunk in chunks])

        points = []
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            points.append(
                models.PointStruct(
                    id=qdrant_point_id(chunk.chunk_id),
                    vector=point_vector(
                        settings=self.settings,
                        dense_embedding=embedding,
                        sparse_vector=sparse_vectors[index] if sparse_vectors is not None else None,
                    ),
                    payload={
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
                    },
                )
            )
        if points:
            self.qdrant.upsert(collection_name=self.settings.qdrant_collection, points=points)

    def cleanup(self, chunks: list[Chunk]) -> None:
        point_ids = [qdrant_point_id(chunk.chunk_id) for chunk in chunks]
        if not point_ids:
            return
        try:
            self.qdrant.delete(
                collection_name=self.settings.qdrant_collection,
                points_selector=models.PointIdsList(points=point_ids),
            )
        except Exception:
            pass

    def _sparse_encoder(self) -> BM25SparseEncoder:
        if self.sparse_encoder is None:
            raise RuntimeError("sparse_encoder_required_for_qdrant_vector_indexer")
        return self.sparse_encoder


@dataclass(frozen=True)
class NoopStructuredExtractor:
    name: str = "noop"

    def extract(self, loaded: LoadedDocument) -> list[StructuredArtifact]:
        return []


def register_builtin_ingestion_components() -> None:
    document_loader_registry.register_if_missing("local", LocalDocumentLoader())
    document_parser_registry.register_if_missing("markdown", MarkdownDocumentParser())
    document_parser_registry.register_if_missing("txt", TxtDocumentParser())
    document_parser_registry.register_if_missing("pdf", PdfDocumentParser())
    chunker_registry.register_if_missing("default", DefaultChunker())
    parent_block_builder_registry.register_if_missing("default", DefaultParentBlockBuilder())
    vector_indexer_registry.register_if_missing("qdrant", QdrantVectorIndexer)
    structured_extractor_registry.register_if_missing("noop", NoopStructuredExtractor())


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
