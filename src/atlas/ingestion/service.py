from dataclasses import dataclass
import hashlib
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient, models
from sqlalchemy import delete
from sqlalchemy.orm import Session

from atlas.core.config import Settings, bm25_sparse_enabled
from atlas.core.ids import new_id
from atlas.db import repositories
from atlas.db.models import Chunk, Document, IngestionRun, ParentBlock, utcnow
from atlas.embeddings.base import Embedder
from atlas.embeddings.bm25_sparse import BM25SparseEncoder
from atlas.ingestion.chunker import chunk_text
from atlas.ingestion.loaders import LoadedDocument, load_local_document
from atlas.ingestion.path_policy import allowed_document_roots
from atlas.vector.collections import ensure_chunk_collection


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
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.qdrant = qdrant
        self.sparse_encoder = sparse_encoder

    def ingest_paths(
        self,
        db: Session,
        *,
        paths: list[str],
        source_uri: str | None,
        metadata: dict,
    ) -> IngestionResult:
        ensure_chunk_collection(self.qdrant, self.settings)

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
            ingestion_run.error_message = f"{failed_count} document(s) failed" if failed_count else None
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

        chunk_inputs = _chunk_loaded_document(
            loaded,
            target_tokens=self.settings.chunk_target_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
        )
        embeddings = self.embedder.embed_texts([item["text"] for item in chunk_inputs])

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

        parent_blocks = _parent_blocks_for_loaded_document(document.document_id, loaded)
        parent_by_page = {
            parent.page_start: parent
            for parent in parent_blocks
            if parent.page_start == parent.page_end
        }
        for parent in parent_blocks:
            db.add(parent)

        chunks: list[Chunk] = []
        for chunk_index, item in enumerate(chunk_inputs):
            parent_id = _parent_id_for_chunk(document.document_id, item, parent_by_page)
            chunk = Chunk(
                chunk_id=new_id("chk"),
                document_id=document.document_id,
                parent_id=parent_id,
                chunk_index=chunk_index,
                text=item["text"],
                text_hash=_sha256(item["text"]),
                section_title=item["section_title"],
                page_start=item["page_start"],
                page_end=item["page_end"],
                token_count=item["token_count"],
                embedding_model=self.embedder.model_name,
                embedding_dim=self.embedder.dimension,
                metadata_json={
                    "source_path": str(loaded.path),
                    "parent_id": parent_id,
                    "page_start": item["page_start"],
                    "page_end": item["page_end"],
                },
            )
            parent = parent_by_page.get(item["page_start"]) if item["page_start"] is not None else None
            if parent is None and parent_blocks:
                parent = parent_blocks[0]
            if parent is not None:
                parent.child_ids_json = [*parent.child_ids_json, chunk.chunk_id]
            chunks.append(chunk)
            db.add(chunk)

        db.flush()
        db.commit()

        try:
            self._upsert_vectors(document=document, chunks=chunks, embeddings=embeddings)
        except Exception:
            self._delete_qdrant_points(chunks)
            self._delete_document_rows(db, document.document_id)
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
        sparse_vectors = None
        if bm25_sparse_enabled(self.settings):
            sparse_vectors = self._sparse_encoder().embed_texts([chunk.text for chunk in chunks])

        points = []
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            points.append(
                models.PointStruct(
                    id=_qdrant_point_id(chunk.chunk_id),
                    vector=_point_vector(
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

    def _sparse_encoder(self) -> BM25SparseEncoder:
        if self.sparse_encoder is None:
            self.sparse_encoder = BM25SparseEncoder(self.settings)
        return self.sparse_encoder

    def _delete_qdrant_points(self, chunks: list[Chunk]) -> None:
        point_ids = [_qdrant_point_id(chunk.chunk_id) for chunk in chunks]
        if not point_ids:
            return
        try:
            self.qdrant.delete(
                collection_name=self.settings.qdrant_collection,
                points_selector=models.PointIdsList(points=point_ids),
            )
        except Exception:
            pass

    @staticmethod
    def _delete_document_rows(db: Session, document_id: str) -> None:
        db.rollback()
        db.execute(delete(Chunk).where(Chunk.document_id == document_id))
        db.execute(delete(Document).where(Document.document_id == document_id))
        db.commit()


def _chunk_loaded_document(
    loaded: LoadedDocument,
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, Any]]:
    if loaded.file_type == "pdf":
        items: list[dict[str, Any]] = []
        for page in loaded.pages:
            drafts = chunk_text(
                page.text,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            )
            for draft in drafts:
                items.append(
                    {
                        "text": draft.text,
                        "section_title": draft.section_title,
                        "page_start": page.page_number,
                        "page_end": page.page_number,
                        "token_count": draft.token_count,
                    }
                )
        return items

    drafts = chunk_text(
        loaded.text,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )
    return [
        {
            "text": draft.text,
            "section_title": draft.section_title,
            "page_start": None,
            "page_end": None,
            "token_count": draft.token_count,
        }
        for draft in drafts
    ]


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


def _parent_blocks_for_loaded_document(document_id: str, loaded: LoadedDocument) -> list[ParentBlock]:
    if loaded.file_type == "pdf":
        parents = []
        for page in loaded.pages:
            if page.page_number is None or not page.text.strip():
                continue
            parents.append(
                ParentBlock(
                    parent_id=_parent_id(document_id, page.page_number),
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
            parent_id=_parent_id(document_id, 0),
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


def _parent_id_for_chunk(
    document_id: str,
    item: dict[str, Any],
    parent_by_page: dict[int, ParentBlock],
) -> str | None:
    page_start = item.get("page_start")
    if isinstance(page_start, int) and page_start in parent_by_page:
        return parent_by_page[page_start].parent_id
    return _parent_id(document_id, 0) if parent_by_page else None


def _parent_id(document_id: str, page_number: int) -> str:
    raw = f"{document_id}:page:{page_number}"
    return f"par_{uuid5(NAMESPACE_URL, raw).hex[:32]}"


def _point_vector(
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


def _qdrant_point_id(chunk_id: str) -> str:
    raw_uuid = chunk_id.removeprefix("chk_")
    try:
        return str(UUID(raw_uuid))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, chunk_id))
