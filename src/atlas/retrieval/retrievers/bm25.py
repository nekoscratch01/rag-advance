from qdrant_client import QdrantClient, models
from sqlalchemy.orm import Session

from atlas.core.config import Settings
from atlas.core.errors import AtlasError, ErrorCode
from atlas.db import repositories
from atlas.retrieval.models.candidate import Candidate
from atlas.retrieval.models.evidence import Evidence


class BM25Retriever:
    def __init__(
        self,
        *,
        settings: Settings,
        sparse_encoder,
        qdrant: QdrantClient,
    ) -> None:
        self.settings = settings
        self.sparse_encoder = sparse_encoder
        self.qdrant = qdrant

    def retrieve_candidates(
        self,
        db: Session,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[Candidate]:
        query_vector = self.sparse_encoder.embed_query(query)
        qdrant_filter = _build_filter(filters or {})
        points = self._query_points(
            query_vector=query_vector,
            top_k=top_k,
            qdrant_filter=qdrant_filter,
        )
        chunk_ids = [str(point.payload.get("chunk_id")) for point in points if point.payload]
        chunk_map = repositories.get_chunks_by_ids(db, chunk_ids)

        candidates: list[Candidate] = []
        for rank, point in enumerate(points, start=1):
            chunk_id = str(point.payload.get("chunk_id")) if point.payload else ""
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                continue
            metadata = _candidate_metadata(chunk)
            candidates.append(
                Candidate(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document.document_id,
                    doc_name=chunk.document.title,
                    source_title=chunk.document.title,
                    company=_metadata_value(metadata, "company"),
                    text=chunk.text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    chunk_index=chunk.chunk_index,
                    token_count=chunk.token_count,
                    retrieved_by=("bm25",),
                    dense_rank=None,
                    dense_score=None,
                    lexical_rank=rank,
                    lexical_score=float(point.score),
                    lexical_backend="qdrant_bm25",
                    final_rank=rank,
                    metadata=metadata,
                    source_uri=chunk.document.source_uri,
                    section_title=chunk.section_title,
                )
            )
        return candidates

    def retrieve(
        self,
        db: Session,
        *,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[Evidence]:
        candidates = self.retrieve_candidates(db, query=query, top_k=top_k, filters=filters)
        return [
            _candidate_to_evidence(candidate, index)
            for index, candidate in enumerate(candidates, start=1)
        ]

    def _query_points(
        self,
        *,
        query_vector: models.SparseVector,
        top_k: int,
        qdrant_filter: models.Filter | None,
    ):
        try:
            if hasattr(self.qdrant, "query_points"):
                result = self.qdrant.query_points(
                    collection_name=self.settings.qdrant_collection,
                    query=query_vector,
                    using=self.settings.qdrant_sparse_vector_name,
                    query_filter=qdrant_filter,
                    limit=top_k,
                    with_payload=True,
                )
                return result.points
            return self.qdrant.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=models.NamedSparseVector(
                    name=self.settings.qdrant_sparse_vector_name,
                    vector=query_vector,
                ),
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise AtlasError(
                ErrorCode.UPSTREAM_VECTOR_STORE_UNAVAILABLE,
                "Qdrant BM25 retrieval failed.",
                status_code=502,
                details={
                    "collection": self.settings.qdrant_collection,
                    "sparse_vector_name": self.settings.qdrant_sparse_vector_name,
                    "type": exc.__class__.__name__,
                },
            ) from exc


def _build_filter(filters: dict) -> models.Filter | None:
    conditions = _filter_conditions(filters)
    return models.Filter(must=conditions) if conditions else None


def _filter_conditions(filters: dict) -> list[models.FieldCondition]:
    conditions: list[models.FieldCondition] = []
    document_ids = filters.get("document_ids")
    if document_ids:
        conditions.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchAny(any=list(document_ids)),
            )
        )
    for key, value in filters.items():
        if key == "document_ids" or value is None:
            continue
        payload_key = _payload_key(key)
        if payload_key is None:
            continue
        if isinstance(value, list | tuple | set):
            values = [item for item in value if item is not None]
            if values:
                conditions.append(
                    models.FieldCondition(
                        key=payload_key,
                        match=models.MatchAny(any=values),
                    )
                )
            continue
        conditions.append(
            models.FieldCondition(
                key=payload_key,
                match=models.MatchValue(value=value),
            )
        )
    return conditions


def _payload_key(key: str) -> str | None:
    payload_key = {
        "section_name": "section_title",
        "document_type": "file_type",
        "filing_type": "file_type",
    }.get(key, key)
    supported = {
        "document_id",
        "parent_id",
        "title",
        "source_uri",
        "file_type",
        "section_title",
        "page_start",
        "page_end",
        "language",
        "embedding_model",
    }
    return payload_key if payload_key in supported else None


def _candidate_metadata(chunk) -> dict:
    metadata = {}
    document_metadata = chunk.document.metadata_json or {}
    chunk_metadata = chunk.metadata_json or {}
    metadata.update(document_metadata)
    metadata.update(chunk_metadata)
    parent_id = getattr(chunk, "parent_id", None)
    if parent_id:
        metadata["parent_id"] = parent_id
    return metadata


def _metadata_value(metadata: dict, key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    return str(value)


def _candidate_to_evidence(candidate: Candidate, evidence_index: int) -> Evidence:
    retrieval_score = candidate.lexical_score if candidate.lexical_score is not None else 0.0
    return Evidence(
        evidence_id=f"c{evidence_index}",
        document_id=candidate.document_id,
        chunk_id=candidate.chunk_id,
        text=candidate.text,
        source_title=candidate.source_title,
        source_uri=candidate.source_uri,
        section_title=candidate.section_title,
        page_start=candidate.page_start,
        page_end=candidate.page_end,
        retrieval_score=float(retrieval_score),
        rank=candidate.lexical_rank or candidate.final_rank or evidence_index,
        token_count=candidate.token_count,
        metadata={
            **candidate.metadata,
            "retrieved_by": list(candidate.retrieved_by),
            "lexical_rank": candidate.lexical_rank,
            "lexical_score": candidate.lexical_score,
            "lexical_backend": candidate.lexical_backend,
        },
    )
