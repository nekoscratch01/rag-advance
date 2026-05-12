from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from atlas.core.config import Settings
from atlas.core.errors import AtlasError, ErrorCode
from atlas.db.models import Chunk, Document, IngestionRun, ParentBlock
from atlas.embeddings.base import Embedder
from atlas.ingestion.builtins import QdrantVectorIndexer
from atlas.ingestion.loaders import load_local_document
from atlas.ingestion.registry import (
    chunker_registry,
    document_loader_registry,
    document_parser_registry,
    parent_block_builder_registry,
    structured_extractor_registry,
    vector_indexer_registry,
)
from atlas.ingestion.service import IngestionService


class _FakeEmbedder(Embedder):
    model_name = "fake-embedder"
    dimension = 3

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class _FakeVectorIndexer:
    name = "fake"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.prepared = False
        self.indexed = []
        self.cleaned = []

    def prepare(self) -> None:
        self.prepared = True

    def index(
        self,
        *,
        document: Document,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        self.indexed.append((document, chunks, embeddings))
        if self.fail:
            raise RuntimeError("vector index failed")

    def cleanup(self, chunks: list[Chunk]) -> None:
        self.cleaned.append(list(chunks))


class _FakeDB:
    def __init__(self) -> None:
        self.added = []
        self.commits = 0
        self.rollbacks = 0
        self.executed = []

    def add(self, value) -> None:
        self.added.append(value)

    def flush(self) -> None:
        return None

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def execute(self, statement) -> None:
        self.executed.append(statement)

    def get(self, model, key):
        for item in self.added:
            if model is IngestionRun and getattr(item, "ingestion_run_id", None) == key:
                return item
        return None


def test_builtin_ingestion_components_are_registered() -> None:
    assert "local" in document_loader_registry.names
    assert {"markdown", "txt", "pdf"}.issubset(set(document_parser_registry.names))
    assert "default" in chunker_registry.names
    assert "default" in parent_block_builder_registry.names
    assert "qdrant" in vector_indexer_registry.names
    assert "noop" in structured_extractor_registry.names


def test_local_markdown_loader_uses_registry(tmp_path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\n\nBody text.", encoding="utf-8")

    loaded = load_local_document(str(path), allowed_roots=[tmp_path])

    assert loaded.title == "Title"
    assert loaded.file_type == "markdown"
    assert loaded.text.startswith("# Title")


def test_local_loader_keeps_unsupported_type_error(tmp_path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text("name,value\nalpha,1\n", encoding="utf-8")

    with pytest.raises(AtlasError) as exc:
        load_local_document(str(path), allowed_roots=[tmp_path])

    assert exc.value.error_code == ErrorCode.UNSUPPORTED_FILE_TYPE
    assert exc.value.status_code == 400
    assert exc.value.error_message == (
        "Unsupported file type: .csv. Atlas supports PDF, Markdown, and TXT."
    )


def test_ingestion_service_accepts_fake_vector_indexer(monkeypatch, tmp_path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\n\nBody text.", encoding="utf-8")
    monkeypatch.setattr("atlas.db.repositories.get_document_by_hash", lambda db, content_hash: None)
    fake_indexer = _FakeVectorIndexer()
    service = IngestionService(
        settings=Settings(openai_api_key=None, document_roots=str(tmp_path)),
        embedder=_FakeEmbedder(),
        qdrant=object(),
        vector_indexer=fake_indexer,
    )

    summary = service._ingest_one_path(
        _FakeDB(),
        path=str(path),
        source_uri=None,
        metadata={},
    )

    assert summary.status == "ingested"
    assert fake_indexer.indexed


def test_ingestion_service_public_flow_prepares_indexer_and_records_summary(
    monkeypatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\n\nBody text.", encoding="utf-8")
    monkeypatch.setattr("atlas.db.repositories.get_document_by_hash", lambda db, content_hash: None)
    fake_indexer = _FakeVectorIndexer()
    fake_db = _FakeDB()
    service = IngestionService(
        settings=Settings(openai_api_key=None, document_roots=str(tmp_path)),
        embedder=_FakeEmbedder(),
        qdrant=object(),
        vector_indexer=fake_indexer,
    )

    result = service.ingest_paths(
        fake_db,
        paths=[str(path)],
        source_uri=None,
        metadata={},
    )

    assert fake_indexer.prepared is True
    assert fake_indexer.indexed
    assert [item.status for item in result.documents] == ["ingested"]
    assert fake_db.commits >= 2
    run = next(item for item in fake_db.added if isinstance(item, IngestionRun))
    assert run.status == "completed"
    assert run.document_ids_json == [result.documents[0].document_id]


def test_ingestion_service_public_flow_reports_failed_document(
    monkeypatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\n\nBody text.", encoding="utf-8")
    monkeypatch.setattr("atlas.db.repositories.get_document_by_hash", lambda db, content_hash: None)
    fake_indexer = _FakeVectorIndexer(fail=True)
    fake_db = _FakeDB()
    service = IngestionService(
        settings=Settings(openai_api_key=None, document_roots=str(tmp_path)),
        embedder=_FakeEmbedder(),
        qdrant=object(),
        vector_indexer=fake_indexer,
    )

    result = service.ingest_paths(
        fake_db,
        paths=[str(path)],
        source_uri=None,
        metadata={},
    )

    assert fake_indexer.prepared is True
    assert [item.status for item in result.documents] == ["failed"]
    assert "vector index failed" in (result.documents[0].error_message or "")
    run = next(item for item in fake_db.added if isinstance(item, IngestionRun))
    assert run.status == "failed"
    assert run.document_ids_json == []


def test_duplicate_document_skips_vector_indexing(monkeypatch, tmp_path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\n\nBody text.", encoding="utf-8")
    existing = Document(
        document_id="doc_existing",
        title="Title",
        source_uri=f"local:{path}",
        file_type="markdown",
        content_hash="already-present",
        language="en",
        metadata_json={"path": str(path)},
    )
    existing_chunks = [
        Chunk(
            chunk_id="chk_existing",
            document_id="doc_existing",
            parent_id=None,
            chunk_index=0,
            text="Body text.",
            text_hash="hash",
            section_title=None,
            page_start=None,
            page_end=None,
            token_count=2,
            embedding_model="fake-embedder",
            embedding_dim=3,
            metadata_json={},
        )
    ]
    monkeypatch.setattr(
        "atlas.db.repositories.get_document_by_hash",
        lambda db, content_hash: existing,
    )
    monkeypatch.setattr(
        "atlas.db.repositories.get_chunks_for_document",
        lambda db, document_id: existing_chunks,
    )
    fake_indexer = _FakeVectorIndexer()
    service = IngestionService(
        settings=Settings(openai_api_key=None, document_roots=str(tmp_path)),
        embedder=_FakeEmbedder(),
        qdrant=object(),
        vector_indexer=fake_indexer,
    )

    summary = service._ingest_one_path(
        _FakeDB(),
        path=str(path),
        source_uri=None,
        metadata={},
    )

    assert summary.status == "skipped_duplicate"
    assert summary.document_id == "doc_existing"
    assert summary.chunk_count == 1
    assert fake_indexer.indexed == []


def test_duplicate_check_allows_same_content_across_profiles(monkeypatch, tmp_path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\n\nBody text.", encoding="utf-8")
    raw_hash = hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    existing_default = Document(
        document_id="doc_existing_default",
        title="Title",
        source_uri=f"local:{path}",
        file_type="markdown",
        content_hash=raw_hash,
        language="en",
        metadata_json={"path": str(path)},
    )

    monkeypatch.setattr(
        "atlas.db.repositories.get_document_by_hash",
        lambda db, content_hash: existing_default if content_hash == raw_hash else None,
    )
    fake_indexer = _FakeVectorIndexer()
    fake_db = _FakeDB()
    service = IngestionService(
        settings=Settings(openai_api_key=None, document_roots=str(tmp_path)),
        embedder=_FakeEmbedder(),
        qdrant=object(),
        vector_indexer=fake_indexer,
    )

    summary = service._ingest_one_path(
        fake_db,
        path=str(path),
        source_uri=None,
        metadata={},
        ingestion_profile="v4",
    )

    documents = [item for item in fake_db.added if isinstance(item, Document)]
    assert summary.status == "ingested"
    assert fake_indexer.indexed
    assert len(documents) == 1
    assert documents[0].document_id != existing_default.document_id
    assert documents[0].content_hash != raw_hash
    assert documents[0].metadata_json["atlas_ingestion_profile"] == "v4"


def test_duplicate_check_skips_same_v4_profile(monkeypatch, tmp_path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\n\nBody text.", encoding="utf-8")
    existing_v4 = Document(
        document_id="doc_existing_v4",
        title="Title",
        source_uri=f"local:{path}",
        file_type="markdown",
        content_hash="profile-scoped-hash",
        language="en",
        metadata_json={"atlas_ingestion_profile": "v4"},
    )
    existing_chunks = [
        Chunk(
            chunk_id="chk_existing_v4",
            document_id="doc_existing_v4",
            parent_id=None,
            chunk_index=0,
            text="Body text.",
            text_hash="hash",
            section_title=None,
            page_start=None,
            page_end=None,
            token_count=2,
            embedding_model="fake-embedder",
            embedding_dim=3,
            metadata_json={"atlas_ingestion_profile": "v4"},
        )
    ]
    monkeypatch.setattr(
        "atlas.db.repositories.get_document_by_hash",
        lambda db, content_hash: existing_v4,
    )
    monkeypatch.setattr(
        "atlas.db.repositories.get_chunks_for_document",
        lambda db, document_id: existing_chunks,
    )
    fake_indexer = _FakeVectorIndexer()
    service = IngestionService(
        settings=Settings(openai_api_key=None, document_roots=str(tmp_path)),
        embedder=_FakeEmbedder(),
        qdrant=object(),
        vector_indexer=fake_indexer,
    )

    summary = service._ingest_one_path(
        _FakeDB(),
        path=str(path),
        source_uri=None,
        metadata={},
        ingestion_profile="v4",
    )

    assert summary.status == "skipped_duplicate"
    assert summary.document_id == "doc_existing_v4"
    assert summary.chunk_count == 1
    assert fake_indexer.indexed == []


def test_ingestion_cleans_up_when_vector_indexer_fails(monkeypatch, tmp_path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\n\nBody text.", encoding="utf-8")
    monkeypatch.setattr("atlas.db.repositories.get_document_by_hash", lambda db, content_hash: None)
    fake_indexer = _FakeVectorIndexer(fail=True)
    fake_db = _FakeDB()
    service = IngestionService(
        settings=Settings(openai_api_key=None, document_roots=str(tmp_path)),
        embedder=_FakeEmbedder(),
        qdrant=object(),
        vector_indexer=fake_indexer,
    )

    with pytest.raises(RuntimeError, match="vector index failed"):
        service._ingest_one_path(
            fake_db,
            path=str(path),
            source_uri=None,
            metadata={},
        )

    assert fake_indexer.cleaned
    assert fake_db.rollbacks >= 1
    assert fake_db.executed
    executed_text = "\n".join(str(statement) for statement in fake_db.executed)
    assert ParentBlock.__tablename__ in executed_text
    assert Chunk.__tablename__ in executed_text
    assert Document.__tablename__ in executed_text


def test_qdrant_vector_indexer_cleanup_swallows_delete_errors() -> None:
    class _BrokenQdrant:
        def delete(self, **_kwargs) -> None:
            raise RuntimeError("delete failed")

    indexer = QdrantVectorIndexer(
        settings=Settings(openai_api_key=None, bm25_enabled=False),
        qdrant=_BrokenQdrant(),
    )

    indexer.cleanup([SimpleNamespace(chunk_id="chk_bad")])


def test_qdrant_vector_indexer_requires_injected_sparse_encoder_when_bm25_enabled() -> None:
    indexer = QdrantVectorIndexer(
        settings=Settings(openai_api_key=None, bm25_enabled=True),
        qdrant=object(),
    )

    with pytest.raises(RuntimeError, match="sparse_encoder_required"):
        indexer.index(
            document=Document(
                document_id="doc_sparse_required",
                title="Sparse Required",
                source_uri="local:test",
                file_type="markdown",
                content_hash="hash_sparse_required",
                language="en",
                metadata_json={},
            ),
            chunks=[
                Chunk(
                    chunk_id="chk_sparse_required",
                    document_id="doc_sparse_required",
                    parent_id=None,
                    chunk_index=0,
                    text="Body text.",
                    text_hash="hash",
                    section_title=None,
                    page_start=None,
                    page_end=None,
                    token_count=2,
                    embedding_model="fake-embedder",
                    embedding_dim=3,
                    metadata_json={},
                )
            ],
            embeddings=[[1.0, 0.0, 0.0]],
        )


def test_qdrant_vector_indexer_routes_v4_chunks_to_v4_collection() -> None:
    class _SpyQdrant:
        def __init__(self) -> None:
            self.collections = set()
            self.created = []
            self.upserts = []
            self.deletes = []

        def collection_exists(self, collection_name: str) -> bool:
            return collection_name in self.collections

        def create_collection(self, **kwargs) -> None:
            self.created.append(kwargs["collection_name"])
            self.collections.add(kwargs["collection_name"])

        def upsert(self, **kwargs) -> None:
            self.upserts.append(kwargs)

        def delete(self, **kwargs) -> None:
            self.deletes.append(kwargs)

    qdrant = _SpyQdrant()
    settings = Settings(
        openai_api_key=None,
        bm25_enabled=False,
        retrieval_mode="dense",
        embedding_dim=3,
        qdrant_collection="atlas_default_test",
        v4_qdrant_collection="atlas_v4_test",
    )
    indexer = QdrantVectorIndexer(settings=settings, qdrant=qdrant)
    document = Document(
        document_id="doc_v4",
        title="V4",
        source_uri="local:v4.md",
        file_type="markdown",
        content_hash="hash_v4",
        language="en",
        metadata_json={"atlas_ingestion_profile": "v4"},
    )
    chunk = Chunk(
        chunk_id="chk_v4",
        document_id="doc_v4",
        parent_id=None,
        chunk_index=0,
        text="V4 body text.",
        text_hash="hash",
        section_title=None,
        page_start=None,
        page_end=None,
        token_count=3,
        embedding_model="fake-embedder",
        embedding_dim=3,
        metadata_json={"atlas_ingestion_profile": "v4"},
    )

    indexer.index(document=document, chunks=[chunk], embeddings=[[1.0, 0.0, 0.0]])
    indexer.cleanup([chunk])

    assert qdrant.created == ["atlas_v4_test"]
    assert [item["collection_name"] for item in qdrant.upserts] == ["atlas_v4_test"]
    assert [item["collection_name"] for item in qdrant.deletes] == ["atlas_v4_test"]
    assert "atlas_default_test" not in [item["collection_name"] for item in qdrant.upserts]
