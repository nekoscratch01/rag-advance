from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models
from sqlalchemy import delete
from sqlalchemy.orm import Session

from atlas.core.config import Settings, bm25_sparse_enabled
from atlas.db.models import Chunk, Document, ParentBlock
from atlas.embeddings.base import Embedder
from atlas.embeddings.bm25_sparse import BM25SparseEncoder
from atlas.ingestion.service import _point_vector, _qdrant_point_id
from atlas.vector.collections import ensure_chunk_collection


@dataclass(frozen=True)
class FinanceBenchImportResult:
    document_count: int
    parent_count: int
    child_count: int
    vector_count: int
    collection: str


def import_prepared_financebench(
    db: Session,
    *,
    corpus_dir: str | Path,
    settings: Settings,
    embedder: Embedder,
    qdrant: QdrantClient,
    sparse_encoder: BM25SparseEncoder | None = None,
    batch_size: int = 64,
    reset_existing: bool = True,
    require_hybrid: bool = True,
) -> FinanceBenchImportResult:
    """Import frozen FinanceBench parent/child artifacts into Postgres and Qdrant."""
    if require_hybrid and not bm25_sparse_enabled(settings):
        raise RuntimeError(
            "FinanceBench V1 import requires hybrid sparse vectors. Run with "
            "ATLAS_RETRIEVAL_MODE=hybrid and ATLAS_BM25_ENABLED=true, or pass "
            "require_hybrid=False for a dense-only diagnostic import."
        )

    root = Path(corpus_dir)
    manifest = [
        item for item in _read_jsonl(root / "manifest.jsonl") if item.get("status") == "parsed"
    ]
    parents = _read_jsonl(root / "parsed" / "parent_blocks.jsonl")
    children = _read_jsonl(root / "parsed" / "child_chunks.jsonl")
    if not manifest:
        raise RuntimeError(f"No parsed FinanceBench documents found in {root / 'manifest.jsonl'}")
    if not parents or not children:
        raise RuntimeError(f"FinanceBench parent/child artifacts are missing under {root / 'parsed'}")

    ensure_chunk_collection(qdrant, settings)

    document_ids = [str(item["document_id"]) for item in manifest]
    child_ids = [str(item["chunk_id"]) for item in children]
    if reset_existing:
        _delete_existing_rows(db, document_ids)
        _delete_existing_vectors(qdrant, settings, child_ids)

    documents_by_id = _document_records(manifest)
    parents_by_id = _parent_records(parents)
    chunks = _chunk_records(children, documents_by_id=documents_by_id, embedder=embedder)

    db.add_all(documents_by_id.values())
    db.flush()
    db.add_all(parents_by_id.values())
    db.flush()
    db.add_all(chunks)
    db.commit()

    vector_count = _upsert_child_vectors(
        qdrant,
        settings=settings,
        embedder=embedder,
        sparse_encoder=sparse_encoder,
        children=children,
        documents_by_id=documents_by_id,
        batch_size=batch_size,
    )

    return FinanceBenchImportResult(
        document_count=len(documents_by_id),
        parent_count=len(parents_by_id),
        child_count=len(chunks),
        vector_count=vector_count,
        collection=settings.qdrant_collection,
    )


def _document_records(manifest: list[dict[str, Any]]) -> dict[str, Document]:
    documents: dict[str, Document] = {}
    for item in manifest:
        document_id = str(item["document_id"])
        documents[document_id] = Document(
            document_id=document_id,
            title=str(item["doc_name"]),
            source_uri=item.get("source_url") or item.get("local_pdf_path"),
            file_type="pdf",
            content_hash=str(item.get("pdf_sha256") or document_id),
            language="en",
            metadata_json={
                "dataset": "financebench",
                "doc_name": item.get("doc_name"),
                "doc_link": item.get("doc_link"),
                "pdf_sha256": item.get("pdf_sha256"),
                "local_pdf_path": item.get("local_pdf_path"),
                "parser_name": item.get("parser_name"),
                "parser_version": item.get("parser_version"),
                "corpus_version": item.get("corpus_version"),
                "row_ids": item.get("row_ids") or [],
            },
        )
    return documents


def _parent_records(parents: list[dict[str, Any]]) -> dict[str, ParentBlock]:
    records: dict[str, ParentBlock] = {}
    for item in parents:
        metadata = _metadata(item)
        records[str(item["parent_id"])] = ParentBlock(
            parent_id=str(item["parent_id"]),
            document_id=str(item["document_id"]),
            parent_type=str(item.get("parent_type") or "page"),
            page_start=int(item["page_start"]),
            page_end=int(item["page_end"]),
            text=_clean_text(item.get("text")),
            child_ids_json=[str(child_id) for child_id in item.get("child_ids_json") or []],
            metadata_json=metadata,
        )
    return records


def _chunk_records(
    children: list[dict[str, Any]],
    *,
    documents_by_id: dict[str, Document],
    embedder: Embedder,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for item in children:
        document = documents_by_id[str(item["document_id"])]
        metadata = {
            "dataset": "financebench",
            "doc_name": item.get("doc_name"),
            "parent_id": item.get("parent_id"),
            "pdf_sha256": item.get("pdf_sha256"),
            "source_uri": item.get("source_uri"),
            "corpus_version": document.metadata_json.get("corpus_version"),
        }
        chunks.append(
            Chunk(
                chunk_id=str(item["chunk_id"]),
                document_id=str(item["document_id"]),
                parent_id=str(item["parent_id"]),
                chunk_index=int(item["chunk_index"]),
                text=_clean_text(item.get("text")),
                text_hash=str(item.get("text_hash") or item.get("chunk_id")),
                section_title=item.get("section_title"),
                page_start=_optional_int(item.get("page_start")),
                page_end=_optional_int(item.get("page_end")),
                token_count=int(item.get("token_count") or 0),
                embedding_model=embedder.model_name,
                embedding_dim=embedder.dimension,
                metadata_json=metadata,
            )
        )
    return chunks


def _upsert_child_vectors(
    qdrant: QdrantClient,
    *,
    settings: Settings,
    embedder: Embedder,
    sparse_encoder: BM25SparseEncoder | None,
    children: list[dict[str, Any]],
    documents_by_id: dict[str, Document],
    batch_size: int,
) -> int:
    vector_count = 0
    sparse_enabled = bm25_sparse_enabled(settings)
    if sparse_enabled and sparse_encoder is None:
        sparse_encoder = BM25SparseEncoder(settings)

    for batch in _batched(children, max(1, batch_size)):
        texts = [_clean_text(item.get("text")) for item in batch]
        dense_embeddings = embedder.embed_texts(texts)
        sparse_vectors = sparse_encoder.embed_texts(texts) if sparse_enabled else [None] * len(batch)
        points = []
        for item, dense_embedding, sparse_vector in zip(
            batch,
            dense_embeddings,
            sparse_vectors,
            strict=True,
        ):
            document = documents_by_id[str(item["document_id"])]
            points.append(
                models.PointStruct(
                    id=_qdrant_point_id(str(item["chunk_id"])),
                    vector=_point_vector(
                        settings=settings,
                        dense_embedding=dense_embedding,
                        sparse_vector=sparse_vector,
                    ),
                    payload={
                        "document_id": item["document_id"],
                        "chunk_id": item["chunk_id"],
                        "parent_id": item["parent_id"],
                        "title": document.title,
                        "doc_name": item.get("doc_name"),
                        "source_uri": item.get("source_uri") or document.source_uri,
                        "file_type": "pdf",
                        "section_title": item.get("section_title"),
                        "page_start": item.get("page_start"),
                        "page_end": item.get("page_end"),
                        "language": "en",
                        "embedding_model": embedder.model_name,
                        "corpus_version": document.metadata_json.get("corpus_version"),
                    },
                )
            )
        if points:
            qdrant.upsert(collection_name=settings.qdrant_collection, points=points)
            vector_count += len(points)
    return vector_count


def _delete_existing_rows(db: Session, document_ids: list[str]) -> None:
    if not document_ids:
        return
    db.execute(delete(Chunk).where(Chunk.document_id.in_(document_ids)))
    db.execute(delete(ParentBlock).where(ParentBlock.document_id.in_(document_ids)))
    db.execute(delete(Document).where(Document.document_id.in_(document_ids)))
    db.commit()


def _delete_existing_vectors(
    qdrant: QdrantClient,
    settings: Settings,
    child_ids: list[str],
) -> None:
    for batch in _batched(child_ids, 256):
        point_ids = [_qdrant_point_id(chunk_id) for chunk_id in batch]
        if not point_ids:
            continue
        try:
            qdrant.delete(
                collection_name=settings.qdrant_collection,
                points_selector=models.PointIdsList(points=point_ids),
            )
        except Exception:
            pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata_json")
    if isinstance(metadata, dict):
        return dict(metadata)
    metadata = item.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _clean_text(value: Any) -> str:
    return str(value or "").replace("\x00", "")


def _batched(items: Iterable[Any], batch_size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
