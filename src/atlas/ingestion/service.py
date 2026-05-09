from dataclasses import dataclass
import hashlib
from typing import Any

from qdrant_client import QdrantClient
from sqlalchemy import delete
from sqlalchemy.orm import Session

from atlas.core.config import Settings
from atlas.core.ids import new_id
from atlas.db import repositories
from atlas.db.models import Chunk, Document, IngestionRun, ParentBlock, utcnow
from atlas.embeddings.base import Embedder
from atlas.embeddings.bm25_sparse import BM25SparseEncoder
from atlas.ingestion.builtins import (
    parent_id as _parent_id,
    parent_id_for_chunk as _parent_id_for_chunk_input,
    point_vector as _builtin_point_vector,
    qdrant_point_id as _builtin_qdrant_point_id,
)
from atlas.ingestion.contracts import (
    ChunkInput,
    Chunker,
    LoadedDocument,
    ParentBlockBuilder,
    StructuredExtractor,
    VectorIndexer,
)
from atlas.ingestion.loaders import load_local_document
from atlas.ingestion.path_policy import allowed_document_roots
from atlas.ingestion.registry import (
    chunker_registry,
    parent_block_builder_registry,
    structured_extractor_registry,
    vector_indexer_registry,
)


@dataclass(frozen=True)
class IngestedDocumentSummary:
    document_id: str | None
    title: str
    status: str
    chunk_count: int
    error_message: str | None = None


@dataclass(frozen=True)
class IngestionResult:
    ingestion_run_id: str
    documents: list[IngestedDocumentSummary]


class IngestionService:
    def __init__(
        self,
        *,
        settings: Settings,
        embedder: Embedder,
        qdrant: QdrantClient,
        sparse_encoder: BM25SparseEncoder | None = None,
        chunker: Chunker | None = None,
        parent_block_builder: ParentBlockBuilder | None = None,
        vector_indexer: VectorIndexer | None = None,
        structured_extractor: StructuredExtractor | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.qdrant = qdrant
        self.sparse_encoder = sparse_encoder
        self.chunker = chunker or chunker_registry.get("default")
        self.parent_block_builder = (
            parent_block_builder or parent_block_builder_registry.get("default")
        )
        self.vector_indexer = vector_indexer or vector_indexer_registry.build(
            "qdrant",
            settings=settings,
            qdrant=qdrant,
            sparse_encoder=sparse_encoder,
        )
        self.structured_extractor = structured_extractor or structured_extractor_registry.get(
            "noop"
        )

    def ingest_paths(
        self,
        db: Session,
        *,
        paths: list[str],
        source_uri: str | None,
        metadata: dict,
    ) -> IngestionResult:
        self.vector_indexer.prepare()

        ingestion_run_id = new_id("ing")
        ingestion_run = IngestionRun(
            ingestion_run_id=ingestion_run_id,
            status="running",
            input_paths_json=paths,
            document_ids_json=[],
            summary_json={},
        )
        db.add(ingestion_run)
        db.commit()

        summaries: list[IngestedDocumentSummary] = []
        document_ids: list[str] = []

        for path in paths:
            try:
                summary = self._ingest_one_path(
                    db,
                    path=path,
                    source_uri=source_uri,
                    metadata=metadata,
                )
                summaries.append(summary)
                if summary.document_id is not None:
                    document_ids.append(summary.document_id)
            except Exception as exc:
                db.rollback()
                summaries.append(
                    IngestedDocumentSummary(
                        document_id=None,
                        title=path,
                        status="failed",
                        chunk_count=0,
                        error_message=str(exc),
                    )
                )

        failed_count = sum(1 for item in summaries if item.status == "failed")
        ingestion_run = db.get(IngestionRun, ingestion_run_id)
        if ingestion_run is not None:
            ingestion_run.status = (
                "completed" if failed_count == 0 else "partial_failed" if document_ids else "failed"
            )
            ingestion_run.document_ids_json = document_ids
            ingestion_run.summary_json = _summary_payload(paths, summaries)
            ingestion_run.error_message = (
                f"{failed_count} document(s) failed" if failed_count else None
            )
            ingestion_run.finished_at = utcnow()
            db.commit()

        return IngestionResult(ingestion_run_id=ingestion_run_id, documents=summaries)

    def _ingest_one_path(
        self,
        db: Session,
        *,
        path: str,
        source_uri: str | None,
        metadata: dict,
    ) -> IngestedDocumentSummary:
        loaded = load_local_document(path, allowed_roots=allowed_document_roots(self.settings))
        content_hash = _sha256(loaded.text)
        existing = repositories.get_document_by_hash(db, content_hash)
        if existing is not None:
            chunks = repositories.get_chunks_for_document(db, existing.document_id)
            return IngestedDocumentSummary(
                document_id=existing.document_id,
                title=existing.title,
                status="skipped_duplicate",
                chunk_count=len(chunks),
            )

        chunk_inputs = self.chunker.chunk(
            loaded,
            target_tokens=self.settings.chunk_target_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
        )
        embeddings = self.embedder.embed_texts([item.text for item in chunk_inputs])

        document = Document(
            document_id=new_id("doc"),
            title=loaded.title,
            source_uri=source_uri or f"local:{loaded.path}",
            file_type=loaded.file_type,
            content_hash=content_hash,
            language=loaded.language,
            metadata_json={**metadata, "path": str(loaded.path)},
        )
        db.add(document)

        parent_blocks = self.parent_block_builder.build(
            document_id=document.document_id,
            loaded=loaded,
        )
        parent_by_page = {
            parent.page_start: parent
            for parent in parent_blocks
            if parent.page_start == parent.page_end
        }
        for parent in parent_blocks:
            db.add(parent)

        chunks: list[Chunk] = []
        for chunk_index, item in enumerate(chunk_inputs):
            parent_id = _parent_id_for_chunk_input(document.document_id, item, parent_by_page)
            chunk = Chunk(
                chunk_id=new_id("chk"),
                document_id=document.document_id,
                parent_id=parent_id,
                chunk_index=chunk_index,
                text=item.text,
                text_hash=_sha256(item.text),
                section_title=item.section_title,
                page_start=item.page_start,
                page_end=item.page_end,
                token_count=item.token_count,
                embedding_model=self.embedder.model_name,
                embedding_dim=self.embedder.dimension,
                metadata_json={
                    "source_path": str(loaded.path),
                    "parent_id": parent_id,
                    "page_start": item.page_start,
                    "page_end": item.page_end,
                },
            )
            parent = parent_by_page.get(item.page_start) if item.page_start is not None else None
            if parent is None and parent_blocks:
                parent = parent_blocks[0]
            if parent is not None:
                parent.child_ids_json = [*parent.child_ids_json, chunk.chunk_id]
            chunks.append(chunk)
            db.add(chunk)

        try:
            db.flush()
            self._upsert_vectors(document=document, chunks=chunks, embeddings=embeddings)
            db.commit()
        except Exception:
            self._cleanup_failed_ingest(db, document_id=document.document_id, chunks=chunks)
            raise

        return IngestedDocumentSummary(
            document_id=document.document_id,
            title=document.title,
            status="ingested",
            chunk_count=len(chunks),
        )

    def _upsert_vectors(
        self,
        *,
        document: Document,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        self.vector_indexer.index(document=document, chunks=chunks, embeddings=embeddings)

    def _delete_qdrant_points(self, chunks: list[Chunk]) -> None:
        self.vector_indexer.cleanup(chunks)

    def _cleanup_failed_ingest(
        self,
        db: Session,
        *,
        document_id: str,
        chunks: list[Chunk],
    ) -> None:
        try:
            self._delete_qdrant_points(chunks)
        except Exception:
            pass
        try:
            self._delete_document_rows(db, document_id)
        except Exception:
            pass

    @staticmethod
    def _delete_document_rows(db: Session, document_id: str) -> None:
        db.rollback()
        db.execute(delete(Chunk).where(Chunk.document_id == document_id))
        db.execute(delete(ParentBlock).where(ParentBlock.document_id == document_id))
        db.execute(delete(Document).where(Document.document_id == document_id))
        db.commit()


def _chunk_loaded_document(
    loaded: LoadedDocument,
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, Any]]:
    return [
        _chunk_input_payload(item)
        for item in chunker_registry.get("default").chunk(
            loaded,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
    ]


def _chunk_input_payload(item: ChunkInput) -> dict[str, Any]:
    return {
        "text": item.text,
        "section_title": item.section_title,
        "page_start": item.page_start,
        "page_end": item.page_end,
        "token_count": item.token_count,
    }


def _summary_payload(paths: list[str], summaries: list[IngestedDocumentSummary]) -> dict[str, Any]:
    return {
        "total": len(paths),
        "ingested": sum(1 for item in summaries if item.status == "ingested"),
        "skipped_duplicate": sum(1 for item in summaries if item.status == "skipped_duplicate"),
        "failed": sum(1 for item in summaries if item.status == "failed"),
        "documents": [
            {
                "document_id": item.document_id,
                "title": item.title,
                "status": item.status,
                "chunk_count": item.chunk_count,
                "error_message": item.error_message,
            }
            for item in summaries
        ],
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parent_blocks_for_loaded_document(
    document_id: str,
    loaded: LoadedDocument,
) -> list[ParentBlock]:
    return parent_block_builder_registry.get("default").build(
        document_id=document_id,
        loaded=loaded,
    )


def _parent_id_for_chunk(
    document_id: str,
    item: dict[str, Any] | ChunkInput,
    parent_by_page: dict[int, ParentBlock],
) -> str | None:
    if isinstance(item, ChunkInput):
        return _parent_id_for_chunk_input(document_id, item, parent_by_page)
    page_start = item.get("page_start")
    if isinstance(page_start, int) and page_start in parent_by_page:
        return parent_by_page[page_start].parent_id
    return _parent_id(document_id, 0) if parent_by_page else None


def _point_vector(**kwargs):
    return _builtin_point_vector(**kwargs)


def _qdrant_point_id(chunk_id: str) -> str:
    return _builtin_qdrant_point_id(chunk_id)
