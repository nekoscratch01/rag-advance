from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import inspect
from typing import Any

from qdrant_client import QdrantClient
from sqlalchemy import delete
from sqlalchemy.orm import Session

from atlas.core.config import Settings
from atlas.core.errors import AtlasError, ErrorCode
from atlas.core.ids import new_id
from atlas.db import repositories
from atlas.db.models import Chunk, Document, IngestionRun, ParentBlock, utcnow
from atlas.embeddings.base import Embedder
from atlas.embeddings.bm25_sparse import BM25SparseEncoder
from atlas.ingestion.builtins import (
    load_local_document_with_v4_profile,
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
    StructuredArtifact,
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
from atlas.ingestion.structured.writer import (
    StructuredArtifactValidationError,
    StructuredArtifactWriteResult,
    StructuredArtifactWriter,
)
from atlas.ingestion.structured.tables import should_skip_text_chunking_for_tabular_source


DEFAULT_INGESTION_PROFILE = "default"
V4_INGESTION_PROFILE = "v4"
INGESTION_PROFILE_METADATA_KEY = "atlas_ingestion_profile"
INDEX_NAMESPACE_METADATA_KEY = "index_namespace"
V4_INDEX_NAMESPACE = "v4"
_V4_STRUCTURED_EXTRACTOR_CANDIDATES = (
    "v4",
    "v4_profile",
    "structured",
    "table",
    "table_profile",
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
        structured_artifact_writer: StructuredArtifactWriter | None = None,
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
        self._structured_extractor_explicit = structured_extractor is not None
        self.structured_extractor = structured_extractor or structured_extractor_registry.get(
            "noop"
        )
        self.structured_artifact_writer = (
            structured_artifact_writer
            or StructuredArtifactWriter(
                output_dir=self.settings.v4_structured_artifact_output_dir
            )
        )

    def ingest_paths(
        self,
        db: Session,
        *,
        paths: list[str],
        source_uri: str | None,
        metadata: dict,
        ingestion_profile: str | None = None,
    ) -> IngestionResult:
        profile = _resolve_ingestion_profile(
            ingestion_profile=ingestion_profile,
            metadata=metadata,
        )
        vector_index_prepared: set[str] = set()

        def prepare_vector_index(document: Document) -> None:
            prepare_key = _vector_prepare_key(document)
            if prepare_key in vector_index_prepared:
                return
            _prepare_vector_indexer(self.vector_indexer, document=document)
            vector_index_prepared.add(prepare_key)

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
                    ingestion_profile=profile,
                    ingestion_run_id=ingestion_run_id,
                    prepare_vector_index=prepare_vector_index,
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
        ingestion_profile: str | None = None,
        ingestion_run_id: str | None = None,
        prepare_vector_index: Callable[[Document], None] | None = None,
    ) -> IngestedDocumentSummary:
        profile = _resolve_ingestion_profile(
            ingestion_profile=ingestion_profile,
            metadata=metadata,
        )
        loaded = _load_document_for_profile(
            path,
            settings=self.settings,
            ingestion_profile=profile,
        )
        raw_content_hash = _sha256(loaded.text)
        content_hash = _content_hash_for_profile(
            raw_content_hash,
            ingestion_profile=profile,
        )
        existing = _find_duplicate_document_for_profile(
            db,
            content_hash=content_hash,
            raw_content_hash=raw_content_hash,
            ingestion_profile=profile,
        )
        if existing is not None:
            chunks = repositories.get_chunks_for_document(db, existing.document_id)
            return IngestedDocumentSummary(
                document_id=existing.document_id,
                title=existing.title,
                status="skipped_duplicate",
                chunk_count=len(chunks),
            )

        structured_artifacts = self._extract_structured_artifacts(
            loaded,
            ingestion_profile=profile,
        )
        text_chunking_skipped = _should_skip_text_chunking(
            loaded,
            ingestion_profile=profile,
        )
        if text_chunking_skipped:
            chunk_inputs = []
        else:
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
            metadata_json={
                **metadata,
                "path": str(loaded.path),
                **_profile_metadata(
                    ingestion_profile=profile,
                    structured_artifacts=structured_artifacts,
                    text_chunking_skipped=text_chunking_skipped,
                    indexable_chunk_count=len(chunk_inputs),
                ),
            },
        )
        db.add(document)

        if _should_build_parent_blocks(
            loaded,
            ingestion_profile=profile,
            text_chunking_skipped=text_chunking_skipped,
        ):
            parent_blocks = self.parent_block_builder.build(
                document_id=document.document_id,
                loaded=loaded,
            )
        else:
            parent_blocks = []
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
                    **_chunk_profile_metadata(
                        ingestion_profile=profile,
                    ),
                },
            )
            parent = parent_by_page.get(item.page_start) if item.page_start is not None else None
            if parent is None and parent_blocks:
                parent = parent_blocks[0]
            if parent is not None:
                parent.child_ids_json = [*parent.child_ids_json, chunk.chunk_id]
            chunks.append(chunk)
            db.add(chunk)

        structured_write_result: StructuredArtifactWriteResult | None = None
        try:
            structured_write_result = self._write_structured_artifacts(
                db,
                artifacts=structured_artifacts,
                document=document,
                loaded=loaded,
                ingestion_profile=profile,
                ingestion_run_id=ingestion_run_id,
            )
            if (
                structured_write_result is not None
                and structured_write_result.status != "completed"
            ):
                raise StructuredArtifactValidationError(
                    "structured_artifact_write_not_completed"
                )
            document.metadata_json = {
                **document.metadata_json,
                **_structured_artifact_write_metadata(
                    ingestion_profile=profile,
                    loaded=loaded,
                    write_result=structured_write_result,
                ),
            }
            db.flush()
            self._upsert_vectors(
                document=document,
                chunks=chunks,
                embeddings=embeddings,
                prepare_vector_index=prepare_vector_index,
            )
            db.commit()
        except Exception as exc:
            if structured_write_result is not None:
                self._mark_structured_artifact_write_orphaned(
                    db,
                    write_result=structured_write_result,
                    exc=exc,
                )
            self._cleanup_failed_ingest(db, document_id=document.document_id, chunks=chunks)
            raise

        return IngestedDocumentSummary(
            document_id=document.document_id,
            title=document.title,
            status="ingested",
            chunk_count=len(chunks),
        )

    def _extract_structured_artifacts(
        self,
        loaded: LoadedDocument,
        *,
        ingestion_profile: str,
    ) -> list[StructuredArtifact]:
        if ingestion_profile != V4_INGESTION_PROFILE:
            return []
        extractor = self._structured_extractor_for_profile(ingestion_profile)
        return list(extractor.extract(loaded))

    def _structured_extractor_for_profile(
        self,
        ingestion_profile: str,
    ) -> StructuredExtractor:
        if self._structured_extractor_explicit or ingestion_profile != V4_INGESTION_PROFILE:
            return self.structured_extractor
        return _registered_v4_structured_extractor() or self.structured_extractor

    def _upsert_vectors(
        self,
        *,
        document: Document,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        prepare_vector_index: Callable[[Document], None] | None = None,
    ) -> None:
        if not chunks:
            return
        if prepare_vector_index is None:
            _prepare_vector_indexer(self.vector_indexer, document=document)
        else:
            prepare_vector_index(document)
        self.vector_indexer.index(document=document, chunks=chunks, embeddings=embeddings)

    def _delete_qdrant_points(self, chunks: list[Chunk]) -> None:
        self.vector_indexer.cleanup(chunks)

    def _write_structured_artifacts(
        self,
        db: Session,
        *,
        artifacts: list[StructuredArtifact],
        document: Document,
        loaded: LoadedDocument,
        ingestion_profile: str,
        ingestion_run_id: str | None,
    ) -> StructuredArtifactWriteResult | None:
        if ingestion_profile != V4_INGESTION_PROFILE or not artifacts:
            return None
        return self.structured_artifact_writer.write(
            db,
            artifacts=artifacts,
            document_id=document.document_id,
            ingestion_run_id=ingestion_run_id,
            materialization_policy="facts",
            allow_partial=False,
            metadata={
                INGESTION_PROFILE_METADATA_KEY: V4_INGESTION_PROFILE,
                "source_path": str(loaded.path),
                "source_uri": document.source_uri,
                "file_type": loaded.file_type,
                "structured_artifact_types": [
                    artifact.artifact_type for artifact in artifacts
                ],
            },
        )

    def _mark_structured_artifact_write_orphaned(
        self,
        db: Session,
        *,
        write_result: StructuredArtifactWriteResult,
        exc: Exception,
    ) -> None:
        try:
            self.structured_artifact_writer.mark_batch_orphaned(
                db,
                write_result=write_result,
                message=str(exc),
            )
        except Exception:
            pass

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


def _resolve_ingestion_profile(
    *,
    ingestion_profile: str | None,
    metadata: dict,
) -> str:
    metadata_profile = metadata.get(INGESTION_PROFILE_METADATA_KEY)
    if ingestion_profile is not None and metadata_profile is not None:
        explicit_profile = _normalize_ingestion_profile(ingestion_profile)
        profile_from_metadata = _normalize_ingestion_profile(metadata_profile)
        if explicit_profile != profile_from_metadata:
            raise AtlasError(
                ErrorCode.INVALID_REQUEST,
                "Conflicting ingestion profiles.",
                status_code=400,
                details={
                    "ingestion_profile": ingestion_profile,
                    INGESTION_PROFILE_METADATA_KEY: metadata_profile,
                },
            )
        return explicit_profile
    return _normalize_ingestion_profile(
        ingestion_profile if ingestion_profile is not None else metadata_profile
    )


def _normalize_ingestion_profile(value: Any) -> str:
    if value is None:
        return DEFAULT_INGESTION_PROFILE
    profile = str(value).strip().lower()
    if not profile or profile == DEFAULT_INGESTION_PROFILE:
        return DEFAULT_INGESTION_PROFILE
    if profile == V4_INGESTION_PROFILE:
        return V4_INGESTION_PROFILE
    raise AtlasError(
        ErrorCode.INVALID_REQUEST,
        f"Unsupported ingestion profile: {value}.",
        status_code=400,
        details={
            "supported_profiles": [DEFAULT_INGESTION_PROFILE, V4_INGESTION_PROFILE],
            "v4_opt_in_metadata_key": INGESTION_PROFILE_METADATA_KEY,
        },
    )


def _load_document_for_profile(
    path: str,
    *,
    settings: Settings,
    ingestion_profile: str,
) -> LoadedDocument:
    allowed_roots = allowed_document_roots(settings)
    if ingestion_profile == V4_INGESTION_PROFILE:
        return load_local_document_with_v4_profile(path, allowed_roots=allowed_roots)
    return load_local_document(path, allowed_roots=allowed_roots)


def _registered_v4_structured_extractor() -> StructuredExtractor | None:
    names = set(structured_extractor_registry.names)
    for name in _V4_STRUCTURED_EXTRACTOR_CANDIDATES:
        if name in names:
            return structured_extractor_registry.get(name)

    non_noop_names = [name for name in structured_extractor_registry.names if name != "noop"]
    if len(non_noop_names) == 1:
        return structured_extractor_registry.get(non_noop_names[0])
    return None


def _prepare_vector_indexer(
    vector_indexer: VectorIndexer,
    *,
    document: Document,
) -> None:
    if _call_accepts_keyword(vector_indexer.prepare, "document"):
        vector_indexer.prepare(document=document)
        return
    vector_indexer.prepare()


def _call_accepts_keyword(function: Callable[..., Any], keyword: str) -> bool:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return False
    return any(
        name == keyword or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for name, parameter in signature.parameters.items()
    )


def _vector_prepare_key(document: Document) -> str:
    metadata = document.metadata_json if isinstance(document.metadata_json, dict) else {}
    return ":".join(
        (
            _document_ingestion_profile(document),
            str(metadata.get(INDEX_NAMESPACE_METADATA_KEY) or ""),
        )
    )


def _should_skip_text_chunking(
    loaded: LoadedDocument,
    *,
    ingestion_profile: str,
) -> bool:
    if ingestion_profile != V4_INGESTION_PROFILE:
        return False
    return _is_v4_tabular_file_type(loaded.file_type)


def _should_build_parent_blocks(
    loaded: LoadedDocument,
    *,
    ingestion_profile: str,
    text_chunking_skipped: bool,
) -> bool:
    return not (
        ingestion_profile == V4_INGESTION_PROFILE
        and text_chunking_skipped
        and _is_v4_tabular_file_type(loaded.file_type)
    )


def _is_v4_tabular_file_type(file_type_or_suffix: str | None) -> bool:
    normalized = str(file_type_or_suffix or "").strip().lower()
    return should_skip_text_chunking_for_tabular_source(normalized)


def _profile_metadata(
    *,
    ingestion_profile: str,
    structured_artifacts: list[StructuredArtifact],
    text_chunking_skipped: bool,
    indexable_chunk_count: int,
) -> dict[str, Any]:
    if ingestion_profile != V4_INGESTION_PROFILE:
        return {}
    metadata = {
        INGESTION_PROFILE_METADATA_KEY: V4_INGESTION_PROFILE,
        "structured_artifact_count": len(structured_artifacts),
        "structured_artifact_types": [
            artifact.artifact_type
            for artifact in structured_artifacts
        ],
        "text_chunking_skipped": text_chunking_skipped,
    }
    if indexable_chunk_count:
        metadata[INDEX_NAMESPACE_METADATA_KEY] = V4_INDEX_NAMESPACE
    return metadata


def _chunk_profile_metadata(*, ingestion_profile: str) -> dict[str, Any]:
    if ingestion_profile != V4_INGESTION_PROFILE:
        return {}
    return {
        INGESTION_PROFILE_METADATA_KEY: V4_INGESTION_PROFILE,
        INDEX_NAMESPACE_METADATA_KEY: V4_INDEX_NAMESPACE,
    }


def _structured_artifact_write_metadata(
    *,
    ingestion_profile: str,
    loaded: LoadedDocument,
    write_result: StructuredArtifactWriteResult | None,
) -> dict[str, Any]:
    if ingestion_profile != V4_INGESTION_PROFILE:
        return {}
    if write_result is not None:
        return {
            "structured_artifact_batch_id": write_result.artifact_id,
            "structured_artifact_manifest_path": str(write_result.manifest_path),
            "structured_artifact_status": write_result.status,
            "structured_artifact_warnings": write_result.warnings,
            "structured_artifact_errors": write_result.errors,
            "structured_artifact_materialized_counts": write_result.materialized_counts,
            "structured_artifact_counts": write_result.artifact_counts,
        }
    warnings = []
    if loaded.file_type in {"csv", "xlsx", "html", "htm"}:
        warnings.append(
            {
                "severity": "warning",
                "code": "no_structured_artifacts",
                "message": (
                    "No structured artifacts were extracted for this V4 table intake "
                    f"document ({loaded.file_type})."
                ),
            }
        )
    return {
        "structured_artifact_status": "no_artifacts",
        "structured_artifact_warnings": warnings,
        "structured_artifact_errors": [],
        "structured_artifact_materialized_counts": {},
    }


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


def _content_hash_for_profile(raw_content_hash: str, *, ingestion_profile: str) -> str:
    if ingestion_profile == V4_INGESTION_PROFILE:
        return _sha256(f"{V4_INGESTION_PROFILE}:{raw_content_hash}")
    return raw_content_hash


def _find_duplicate_document_for_profile(
    db: Session,
    *,
    content_hash: str,
    raw_content_hash: str,
    ingestion_profile: str,
) -> Document | None:
    checked: set[str] = set()
    for candidate_hash in (content_hash, raw_content_hash):
        if candidate_hash in checked:
            continue
        checked.add(candidate_hash)
        existing = repositories.get_document_by_hash(db, candidate_hash)
        if existing is None:
            continue
        if _document_ingestion_profile(existing) == ingestion_profile:
            return existing
    return None


def _document_ingestion_profile(document: Document) -> str:
    metadata = document.metadata_json if isinstance(document.metadata_json, dict) else {}
    value = metadata.get(INGESTION_PROFILE_METADATA_KEY)
    if value is None:
        return DEFAULT_INGESTION_PROFILE
    profile = str(value).strip().lower()
    if not profile or profile == DEFAULT_INGESTION_PROFILE:
        return DEFAULT_INGESTION_PROFILE
    return profile


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
