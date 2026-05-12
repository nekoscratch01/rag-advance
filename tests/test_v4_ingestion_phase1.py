from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from atlas.core.config import Settings
from atlas.core.errors import AtlasError, ErrorCode
from atlas.db.models import Chunk, Document, IngestionRun, StructuredArtifactRecord
from atlas.embeddings.base import Embedder
from atlas.ingestion.contracts import StructuredArtifact
from atlas.ingestion.loaders import load_local_document
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
        self.indexed.append((document, chunks, embeddings))

    def cleanup(self, chunks: list[Chunk]) -> None:
        self.cleaned.append(list(chunks))


class _FakeStructuredExtractor:
    name = "fake_structured"

    def __init__(self) -> None:
        self.seen = []

    def extract(self, loaded) -> list[StructuredArtifact]:
        self.seen.append(loaded)
        return [
            StructuredArtifact(
                artifact_type=f"{loaded.file_type}_table_hint",
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
        self.commits = 0
        self.rollbacks = 0
        self.executed = []
        self.merged = []

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


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("sample.csv", b"metric,value\nrevenue,10\n"),
        ("sample.xlsx", b"not really an xlsx"),
        ("sample.html", b"<table><tr><td>metric</td></tr></table>"),
    ],
)
def test_default_profile_rejects_structured_file_types(
    tmp_path,
    filename: str,
    content: bytes,
) -> None:
    path = tmp_path / filename
    path.write_bytes(content)

    with pytest.raises(AtlasError) as exc:
        load_local_document(str(path), allowed_roots=[tmp_path])

    assert exc.value.error_code == ErrorCode.UNSUPPORTED_FILE_TYPE
    assert exc.value.error_message == (
        f"Unsupported file type: {path.suffix}. Atlas supports PDF, Markdown, and TXT."
    )


def test_v4_profile_ingests_csv_and_calls_structured_extractor(monkeypatch, tmp_path) -> None:
    path = tmp_path / "facts.csv"
    path.write_text("metric,value\nrevenue,10\ncapex,3\n", encoding="utf-8")
    monkeypatch.setattr("atlas.db.repositories.get_document_by_hash", lambda db, hash_: None)
    fake_indexer = _FakeVectorIndexer()
    fake_extractor = _FakeStructuredExtractor()
    service = _service(tmp_path, fake_indexer, fake_extractor)
    fake_db = _FakeDB()

    summary = service._ingest_one_path(
        fake_db,
        path=str(path),
        source_uri=None,
        metadata={},
        ingestion_profile="v4",
    )

    document = _only_added(fake_db, Document)
    assert summary.status == "ingested"
    assert summary.chunk_count == 0
    assert document.file_type == "csv"
    assert document.metadata_json[INGESTION_PROFILE_METADATA_KEY] == "v4"
    assert document.metadata_json["structured_artifact_count"] == 1
    assert document.metadata_json["structured_artifact_types"] == ["csv_table_hint"]
    assert document.metadata_json["structured_artifact_status"] == "completed"
    assert document.metadata_json["structured_artifact_manifest_path"]
    assert document.metadata_json["text_chunking_skipped"] is True
    assert fake_extractor.seen[0].file_type == "csv"
    assert fake_indexer.prepared is False
    assert fake_indexer.indexed == []
    assert _only_merged(fake_db, StructuredArtifactRecord).document_id == document.document_id


def test_v4_profile_uses_registered_csv_contract_when_available(monkeypatch, tmp_path) -> None:
    pytest.importorskip("atlas.ingestion.structured.tables")
    path = tmp_path / "facts.csv"
    path.write_text("metric,value\nrevenue,10\ncapex,3\n", encoding="utf-8")
    monkeypatch.setattr("atlas.db.repositories.get_document_by_hash", lambda db, hash_: None)
    fake_indexer = _FakeVectorIndexer()
    service = IngestionService(
        settings=Settings(
            openai_api_key=None,
            document_roots=str(tmp_path),
            v4_structured_artifact_output_dir=str(tmp_path / "artifacts"),
        ),
        embedder=_FakeEmbedder(),
        qdrant=object(),
        vector_indexer=fake_indexer,
    )
    fake_db = _FakeDB()

    summary = service._ingest_one_path(
        fake_db,
        path=str(path),
        source_uri=None,
        metadata={},
        ingestion_profile="v4",
    )

    document = _only_added(fake_db, Document)
    assert summary.status == "ingested"
    assert document.metadata_json["structured_artifact_count"] >= 2
    assert "table" in document.metadata_json["structured_artifact_types"]
    assert "schema_routing_card" in document.metadata_json["structured_artifact_types"]
    assert document.metadata_json["structured_artifact_status"] == "completed"
    assert document.metadata_json["text_chunking_skipped"] is True
    assert fake_indexer.prepared is False
    assert fake_indexer.indexed == []


def test_v4_profile_can_be_enabled_from_metadata_for_current_api_shape(
    monkeypatch,
    tmp_path,
) -> None:
    path = tmp_path / "facts.html"
    path.write_text(
        "<table><tr><th>metric</th><th>value</th></tr>"
        "<tr><td>revenue</td><td>10</td></tr></table>",
        encoding="utf-8",
    )
    monkeypatch.setattr("atlas.db.repositories.get_document_by_hash", lambda db, hash_: None)
    fake_indexer = _FakeVectorIndexer()
    fake_db = _FakeDB()
    service = _service(tmp_path, fake_indexer)

    result = service.ingest_paths(
        fake_db,
        paths=[str(path)],
        source_uri=None,
        metadata={INGESTION_PROFILE_METADATA_KEY: "v4"},
    )

    document = _only_added(fake_db, Document)
    run = _only_added(fake_db, IngestionRun)
    assert [item.status for item in result.documents] == ["ingested"]
    assert run.status == "completed"
    assert document.file_type == "html"
    assert document.metadata_json[INGESTION_PROFILE_METADATA_KEY] == "v4"
    assert document.metadata_json["structured_artifact_status"] == "no_artifacts"
    assert document.metadata_json["structured_artifact_warnings"]
    assert fake_indexer.prepared is False
    assert fake_indexer.indexed == []


def test_v4_profile_ingests_xlsx_with_stdlib_parser(monkeypatch, tmp_path) -> None:
    path = tmp_path / "facts.xlsx"
    _write_minimal_xlsx(path)
    monkeypatch.setattr("atlas.db.repositories.get_document_by_hash", lambda db, hash_: None)
    fake_indexer = _FakeVectorIndexer()
    service = _service(tmp_path, fake_indexer)
    fake_db = _FakeDB()

    summary = service._ingest_one_path(
        fake_db,
        path=str(path),
        source_uri=None,
        metadata={},
        ingestion_profile="v4",
    )

    document = _only_added(fake_db, Document)
    assert summary.status == "ingested"
    assert summary.chunk_count == 0
    assert document.file_type == "xlsx"
    assert document.metadata_json["text_chunking_skipped"] is True
    assert document.metadata_json["structured_artifact_status"] == "no_artifacts"
    assert document.metadata_json["structured_artifact_warnings"]
    assert fake_indexer.prepared is False
    assert fake_indexer.indexed == []


def test_invalid_v4_profile_is_rejected_before_index_prepare(tmp_path) -> None:
    path = tmp_path / "facts.csv"
    path.write_text("metric,value\nrevenue,10\n", encoding="utf-8")
    fake_indexer = _FakeVectorIndexer()
    service = _service(tmp_path, fake_indexer, _FakeStructuredExtractor())

    with pytest.raises(AtlasError) as exc:
        service.ingest_paths(
            _FakeDB(),
            paths=[str(path)],
            source_uri=None,
            metadata={INGESTION_PROFILE_METADATA_KEY: "v5"},
        )

    assert exc.value.error_code == ErrorCode.INVALID_REQUEST
    assert fake_indexer.prepared is False


def test_default_profile_does_not_call_structured_extractor(monkeypatch, tmp_path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\n\nBody text.", encoding="utf-8")
    monkeypatch.setattr("atlas.db.repositories.get_document_by_hash", lambda db, hash_: None)
    fake_extractor = _FakeStructuredExtractor()
    service = _service(tmp_path, _FakeVectorIndexer(), fake_extractor)

    summary = service._ingest_one_path(
        _FakeDB(),
        path=str(path),
        source_uri=None,
        metadata={},
    )

    assert summary.status == "ingested"
    assert fake_extractor.seen == []


def _service(
    tmp_path,
    fake_indexer: _FakeVectorIndexer,
    fake_extractor: _FakeStructuredExtractor | None = None,
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
        structured_extractor=fake_extractor,
    )


def _only_added(fake_db: _FakeDB, model):
    matches = [item for item in fake_db.added if isinstance(item, model)]
    assert len(matches) == 1
    return matches[0]


def _only_merged(fake_db: _FakeDB, model):
    matches = [item for item in fake_db.merged if isinstance(item, model)]
    assert len(matches) == 1
    return matches[0]


def _write_minimal_xlsx(path: Path) -> None:
    workbook_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Facts" sheetId="1" r:id="rId1"/></sheets>
</workbook>
"""
    sheet_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1">
      <c r="A1" t="inlineStr"><is><t>metric</t></is></c>
      <c r="B1" t="inlineStr"><is><t>value</t></is></c>
    </row>
    <row r="2">
      <c r="A2" t="inlineStr"><is><t>revenue</t></is></c>
      <c r="B2"><v>10</v></c>
    </row>
  </sheetData>
</worksheet>
"""
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
