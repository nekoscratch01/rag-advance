from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias

from atlas.db.models import Chunk, Document, ParentBlock


SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
DEFAULT_STRUCTURED_ARTIFACT_ENVELOPE_VERSION = "structured_artifact.v1"

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]


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
    artifact_id: str | None = None
    content_hash: str | None = None
    source_locator: dict[str, Any] | None = None
    provenance_policy: dict[str, Any] | None = None
    schema_routing_card: dict[str, Any] | None = None
    artifact_manifest: dict[str, Any] | None = None
    envelope_version: str = DEFAULT_STRUCTURED_ARTIFACT_ENVELOPE_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "payload",
            coerce_json_object(self.payload, field_name="payload"),
        )
        for field_name in (
            "source_locator",
            "provenance_policy",
            "schema_routing_card",
            "artifact_manifest",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    coerce_json_object(value, field_name=field_name),
                )
        object.__setattr__(
            self,
            "metadata",
            coerce_json_object(self.metadata, field_name="metadata"),
        )

    def to_envelope_payload(self) -> dict[str, JSONValue]:
        return coerce_json_object(
            {
                "artifact_type": self.artifact_type,
                "payload": self.payload,
                "artifact_id": self.artifact_id,
                "content_hash": self.content_hash,
                "source_locator": self.source_locator,
                "provenance_policy": self.provenance_policy,
                "schema_routing_card": self.schema_routing_card,
                "artifact_manifest": self.artifact_manifest,
                "envelope_version": self.envelope_version,
                "metadata": self.metadata,
            },
            field_name="structured_artifact",
        )


def coerce_json_object(value: Any, *, field_name: str = "payload") -> dict[str, JSONValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a JSON object")
    coerced = _coerce_json_value(value, field_path=field_name)
    if not isinstance(coerced, dict):
        raise TypeError(f"{field_name} must be a JSON object")
    return coerced


def _coerce_json_value(value: Any, *, field_path: str) -> JSONValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{field_path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _coerce_json_value(item, field_path=f"{field_path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _coerce_json_value(item, field_path=f"{field_path}[]")
            for item in value
        ]
    raise TypeError(f"{field_path} contains a non JSON-compatible value")


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

    def prepare(self, *, document: Document | None = None) -> None:
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
