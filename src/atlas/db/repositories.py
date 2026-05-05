from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from atlas.db.models import Chunk, Document, GenerationEvent, ParentBlock, QueryRun, RetrievalEvent


def get_document_by_hash(db: Session, content_hash: str) -> Document | None:
    return db.scalar(select(Document).where(Document.content_hash == content_hash))


def get_chunks_for_document(db: Session, document_id: str) -> list[Chunk]:
    return list(
        db.scalars(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index.asc())
        )
    )


def get_chunks_by_ids(db: Session, chunk_ids: Iterable[str]) -> dict[str, Chunk]:
    ids = list(chunk_ids)
    if not ids:
        return {}
    chunks = db.scalars(select(Chunk).where(Chunk.chunk_id.in_(ids))).all()
    return {chunk.chunk_id: chunk for chunk in chunks}


def get_parent_blocks_by_ids(db: Session, parent_ids: Iterable[str]) -> dict[str, ParentBlock]:
    ids = [parent_id for parent_id in parent_ids if parent_id]
    if not ids:
        return {}
    parent_blocks = db.scalars(select(ParentBlock).where(ParentBlock.parent_id.in_(ids))).all()
    return {parent.parent_id: parent for parent in parent_blocks}


def get_parent_blocks_for_chunks(db: Session, chunks: Iterable[Chunk]) -> dict[str, ParentBlock]:
    chunk_list = list(chunks)
    parent_blocks = get_parent_blocks_by_ids(
        db,
        [chunk.parent_id for chunk in chunk_list if chunk.parent_id],
    )
    return {
        chunk.chunk_id: parent_blocks[chunk.parent_id]
        for chunk in chunk_list
        if chunk.parent_id and chunk.parent_id in parent_blocks
    }


def get_query_run(db: Session, query_id: str) -> QueryRun | None:
    return db.get(QueryRun, query_id)


def add_query_trace(
    db: Session,
    query_run: QueryRun,
    retrieval_events: list[RetrievalEvent],
    generation_event: GenerationEvent | None,
) -> None:
    db.add(query_run)
    db.flush()
    for event in retrieval_events:
        db.add(event)
    if generation_event is not None:
        db.add(generation_event)
