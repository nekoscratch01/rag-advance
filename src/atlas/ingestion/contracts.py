from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from atlas.db.models import Chunk, Document, ParentBlock


SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


@dataclass(frozen=True)
class DocumentSource:
    path: Path
    suffix: str
    content: bytes


@dataclass(frozen=True)
class LoadedPage:
    page_number: int | None
    text: str


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    title: str
    text: str
    file_type: str
    language: str
    pages: list[LoadedPage]


@dataclass(frozen=True)
class ChunkInput:
    text: str
    section_title: str | None
    page_start: int | None
    page_end: int | None
    token_count: int


@dataclass(frozen=True)
class StructuredArtifact:
    artifact_type: str
    payload: dict[str, Any]


class DocumentLoader(Protocol):
    name: str

    def load(self, path_value: str, *, allowed_roots: list[Path]) -> DocumentSource:
        ...


class DocumentParser(Protocol):
    name: str
    supported_suffixes: tuple[str, ...]

    def parse(self, source: DocumentSource) -> LoadedDocument:
        ...


class Chunker(Protocol):
    name: str

    def chunk(
        self,
        loaded: LoadedDocument,
        *,
        target_tokens: int,
        overlap_tokens: int,
    ) -> list[ChunkInput]:
        ...


class ParentBlockBuilder(Protocol):
    name: str

    def build(self, *, document_id: str, loaded: LoadedDocument) -> list[ParentBlock]:
        ...


class VectorIndexer(Protocol):
    name: str

    def prepare(self) -> None:
        ...

    def index(
        self,
        *,
        document: Document,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        ...

    def cleanup(self, chunks: list[Chunk]) -> None:
        ...


class StructuredExtractor(Protocol):
    name: str

    def extract(self, loaded: LoadedDocument) -> list[StructuredArtifact]:
        ...
