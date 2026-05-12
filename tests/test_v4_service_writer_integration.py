from __future__ import annotations

import json
from pathlib import Path

from qdrant_client import QdrantClient

from atlas.core.config import Settings
from atlas.db.models import Chunk, Document, IngestionRun, ParentBlock, StructuredArtifactRecord
from atlas.embeddings.base import Embedder
from atlas.ingestion.contracts import StructuredArtifact
from atlas.ingestion.service import INGESTION_PROFILE_METADATA_KEY, IngestionService


class _FakeEmbedder(Embedder):
    model_name = "fake-embedder"
    dimension = 3

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class _FakeVectorIndexer:
    name = "fake"

    def __init__(self) -> None:
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
        self.indexed.append((document, list(chunks), embeddings))

    def cleanup(self, chunks: list[Chunk]) -> None:
        self.cleaned.append(list(chunks))


class _FailingVectorIndexer(_FakeVectorIndexer):
    def index(
        self,
        *,
        document: Document,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        super().index(document=document, chunks=chunks, embeddings=embeddings)
        raise RuntimeError("vector index failed")


class _InvalidStructuredExtractor:
    name = "invalid_structured"

    def extract(self, loaded) -> list:
        return [
            StructuredArtifact(
                artifact_type="unrecognized_payload",
                payload={"title": loaded.title},
                envelope_version="v4.phase1",
            )
        ]


class _ValidStructuredExtractor:
    name = "valid_structured"

    def extract(self, loaded) -> list:
        return [
            StructuredArtifact(
                artifact_type="table",
                payload={
                    "table_id": f"tbl_{loaded.file_type}",
                    "title": loaded.title,
                    "columns": [{"name": "metric"}, {"name": "value"}],
                    "rows": [],
                },
                envelope_version="v4.phase1",
            )
        ]


class _FakeDB:
    def __init__(self) -> None:
        self.added = []
        self.merged = []
        self.commits = 0
        self.rollbacks = 0
        self.executed = []

    def add(self, value) -> None:
        self.added.append(value)

    def merge(self, value):
        self.merged.append(value)
        return value

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


def test_csv_v4_service_writes_manifest_and_structured_artifact_record(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "facts.csv"
    path.write_text("metric,value\nrevenue,10\ncapex,3\n", encoding="utf-8")
    monkeypatch.setattr("atlas.db.repositories.get_document_by_hash", lambda db, hash_: None)
    fake_db = _FakeDB()
    fake_indexer = _FakeVectorIndexer()
    service = _service(tmp_path, fake_indexer)

    result = service.ingest_paths(
        fake_db,
        paths=[str(path)],
        source_uri=None,
        metadata={INGESTION_PROFILE_METADATA_KEY: "v4"},
    )

    document = _only_added(fake_db, Document)
    artifact_record = _only_merged(fake_db, StructuredArtifactRecord)
    manifest_path = Path(document.metadata_json["structured_artifact_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result.documents[0].status == "ingested"
    assert result.documents[0].chunk_count == 0
    assert manifest_path.exists()
    assert document.metadata_json["structured_artifact_status"] == "completed"
    assert document.metadata_json["structured_artifact_batch_id"] == artifact_record.artifact_id
    assert artifact_record.document_id == document.document_id
    assert artifact_record.ingestion_run_id == result.ingestion_run_id
    assert artifact_record.materialization_policy == "facts"
    assert artifact_record.artifact_type == "structured_artifact_batch"
    assert manifest["document_id"] == document.document_id
    assert manifest["ingestion_run_id"] == result.ingestion_run_id
    assert fake_indexer.prepared is False
    assert fake_indexer.indexed == []
    assert _added(fake_db, ParentBlock) == []


def test_html_v4_table_intake_does_not_index_raw_table_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "facts.html"
    path.write_text(
        "<table><tr><th>metric</th><th>value</th></tr>"
        "<tr><td>revenue</td><td>10</td></tr></table>",
        encoding="utf-8",
    )
    monkeypatch.setattr("atlas.db.repositories.get_document_by_hash", lambda db, hash_: None)
    fake_db = _FakeDB()
    fake_indexer = _FakeVectorIndexer()
    service = _service(tmp_path, fake_indexer)

    result = service.ingest_paths(
        fake_db,
        paths=[str(path)],
        source_uri=None,
        metadata={INGESTION_PROFILE_METADATA_KEY: "v4"},
    )

    document = _only_added(fake_db, Document)
    assert result.documents[0].chunk_count == 0
    assert fake_indexer.prepared is False
    assert fake_indexer.indexed == []
    assert document.metadata_json["structured_artifact_status"] == "no_artifacts"
    assert document.metadata_json["structured_artifact_warnings"]
    assert _merged(fake_db, StructuredArtifactRecord) == []
    assert _added(fake_db, ParentBlock) == []


def test_markdown_v4_indexable_chunks_are_namespaced(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\n\nBody text for V4 indexing.", encoding="utf-8")
    monkeypatch.setattr("atlas.db.repositories.get_document_by_hash", lambda db, hash_: None)
    fake_db = _FakeDB()
    fake_indexer = _FakeVectorIndexer()
    service = _service(tmp_path, fake_indexer)

    result = service.ingest_paths(
        fake_db,
        paths=[str(path)],
        source_uri=None,
        metadata={INGESTION_PROFILE_METADATA_KEY: "v4"},
    )

    document, chunks, _embeddings = fake_indexer.indexed[0]
    assert result.documents[0].chunk_count == len(chunks)
    assert fake_indexer.prepared is True
    assert chunks
    assert document.metadata_json[INGESTION_PROFILE_METADATA_KEY] == "v4"
    assert document.metadata_json["index_namespace"] == "v4"
    assert document.metadata_json["structured_artifact_status"] == "no_artifacts"
    assert document.metadata_json["structured_artifact_warnings"] == []
    assert chunks[0].metadata_json[INGESTION_PROFILE_METADATA_KEY] == "v4"
    assert chunks[0].metadata_json["index_namespace"] == "v4"


def test_markdown_v4_qdrant_prepare_uses_v4_collection_without_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\n\nBody text for V4 Qdrant indexing.", encoding="utf-8")
    monkeypatch.setattr("atlas.db.repositories.get_document_by_hash", lambda db, hash_: None)
    qdrant = QdrantClient(":memory:")
    settings = Settings(
        openai_api_key=None,
        document_roots=str(tmp_path),
        bm25_enabled=False,
        retrieval_mode="dense",
        embedding_dim=3,
        qdrant_collection="atlas_default_test",
        v4_qdrant_collection="atlas_v4_test",
        v4_structured_artifact_output_dir=str(tmp_path / "artifacts"),
    )
    service = IngestionService(
        settings=settings,
        embedder=_FakeEmbedder(),
        qdrant=qdrant,
    )

    result = service.ingest_paths(
        _FakeDB(),
        paths=[str(path)],
        source_uri=None,
        metadata={INGESTION_PROFILE_METADATA_KEY: "v4"},
    )

    points, _next_page = qdrant.scroll(
        collection_name="atlas_v4_test",
        limit=10,
        with_payload=True,
    )
    assert result.documents[0].status == "ingested"
    assert result.documents[0].chunk_count == len(points)
    assert qdrant.collection_exists("atlas_v4_test")
    assert not qdrant.collection_exists("atlas_default_test")
    assert points[0].payload[INGESTION_PROFILE_METADATA_KEY] == "v4"
    assert points[0].payload["index_namespace"] == "v4"


def test_v4_service_invalid_structured_artifact_fails_document(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("metric,value\nrevenue,10\n", encoding="utf-8")
    monkeypatch.setattr("atlas.db.repositories.get_document_by_hash", lambda db, hash_: None)
    fake_db = _FakeDB()
    fake_indexer = _FakeVectorIndexer()
    service = _service(
        tmp_path,
        fake_indexer,
        structured_extractor=_InvalidStructuredExtractor(),
    )

    result = service.ingest_paths(
        fake_db,
        paths=[str(path)],
        source_uri=None,
        metadata={INGESTION_PROFILE_METADATA_KEY: "v4"},
    )

    manifests = list((tmp_path / "artifacts").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert result.documents[0].status == "failed"
    assert result.documents[0].document_id is None
    assert "structured_artifact_validation_failed" in result.documents[0].error_message
    assert manifest["status"] == "failed"
    assert any(error["code"] == "unsupported_artifact_type" for error in manifest["errors"])


def test_v4_service_marks_manifest_orphaned_when_vector_index_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "facts.md"
    path.write_text("# Facts\n\nRevenue was 10.", encoding="utf-8")
    monkeypatch.setattr("atlas.db.repositories.get_document_by_hash", lambda db, hash_: None)
    fake_db = _FakeDB()
    fake_indexer = _FailingVectorIndexer()
    service = _service(
        tmp_path,
        fake_indexer,
        structured_extractor=_ValidStructuredExtractor(),
    )

    result = service.ingest_paths(
        fake_db,
        paths=[str(path)],
        source_uri=None,
        metadata={INGESTION_PROFILE_METADATA_KEY: "v4"},
    )

    manifests = list((tmp_path / "artifacts").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert result.documents[0].status == "failed"
    assert manifest["status"] == "orphaned"
    assert any(
        error["code"] == "ingestion_failed_after_structured_artifact_write"
        for error in manifest["errors"]
    )
    assert fake_indexer.prepared is True
    assert fake_indexer.indexed[0][1]
    assert fake_indexer.cleaned[0]


def _service(
    tmp_path: Path,
    fake_indexer: _FakeVectorIndexer,
    *,
    structured_extractor=None,
) -> IngestionService:
    return IngestionService(
        settings=Settings(
            openai_api_key=None,
            document_roots=str(tmp_path),
            v4_structured_artifact_output_dir=str(tmp_path / "artifacts"),
        ),
        embedder=_FakeEmbedder(),
        qdrant=object(),
        vector_indexer=fake_indexer,
        structured_extractor=structured_extractor,
    )


def _only_added(fake_db: _FakeDB, model):
    matches = [item for item in fake_db.added if isinstance(item, model)]
    assert len(matches) == 1
    return matches[0]


def _added(fake_db: _FakeDB, model):
    return [item for item in fake_db.added if isinstance(item, model)]


def _only_merged(fake_db: _FakeDB, model):
    matches = _merged(fake_db, model)
    assert len(matches) == 1
    return matches[0]


def _merged(fake_db: _FakeDB, model):
    return [item for item in fake_db.merged if isinstance(item, model)]
