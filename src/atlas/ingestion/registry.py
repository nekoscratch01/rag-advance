from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from atlas.ingestion.contracts import (
    Chunker,
    DocumentLoader,
    DocumentParser,
    ParentBlockBuilder,
    StructuredExtractor,
    VectorIndexer,
)


T = TypeVar("T")


class IngestionComponentRegistry(Generic[T]):
    def __init__(self, *, namespace: str) -> None:
        self.namespace = namespace
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> None:
        key = _key(name)
        if not key:
            raise ValueError(f"{self.namespace}_component_name_required")
        if key in self._items:
            raise ValueError(f"{self.namespace}_component_already_registered:{key}")
        self._items[key] = item

    def register_if_missing(self, name: str, item: T) -> None:
        key = _key(name)
        if key not in self._items:
            self.register(key, item)

    def get(self, name: str) -> T:
        key = _key(name)
        try:
            return self._items[key]
        except KeyError as exc:
            raise ValueError(f"{self.namespace}_component_not_registered:{key}") from exc

    def build(self, name: str, *args, **kwargs):
        item = self.get(name)
        if not isinstance(item, Callable):
            raise TypeError(f"{self.namespace}_component_not_callable:{_key(name)}")
        return item(*args, **kwargs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._items)

    @property
    def items(self) -> tuple[tuple[str, T], ...]:
        return tuple(self._items.items())


document_loader_registry: IngestionComponentRegistry[DocumentLoader] = IngestionComponentRegistry(
    namespace="ingestion_document_loader"
)
document_parser_registry: IngestionComponentRegistry[DocumentParser] = IngestionComponentRegistry(
    namespace="ingestion_document_parser"
)
chunker_registry: IngestionComponentRegistry[Chunker] = IngestionComponentRegistry(
    namespace="ingestion_chunker"
)
parent_block_builder_registry: IngestionComponentRegistry[ParentBlockBuilder] = (
    IngestionComponentRegistry(namespace="ingestion_parent_block_builder")
)
vector_indexer_registry: IngestionComponentRegistry[VectorIndexer] = IngestionComponentRegistry(
    namespace="ingestion_vector_indexer"
)
structured_extractor_registry: IngestionComponentRegistry[StructuredExtractor] = (
    IngestionComponentRegistry(namespace="ingestion_structured_extractor")
)


def document_parser_for_suffix(suffix: str) -> DocumentParser | None:
    normalized = _key(suffix)
    for _, parser in document_parser_registry.items:
        if normalized in parser.supported_suffixes:
            return parser
    return None


def _key(name: str) -> str:
    return str(name).strip().lower()
