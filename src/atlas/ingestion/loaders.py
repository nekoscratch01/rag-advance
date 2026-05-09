from pathlib import Path

from atlas.core.errors import AtlasError, ErrorCode
from atlas.ingestion.contracts import (
    SUPPORTED_SUFFIXES,
    DocumentSource,
    LoadedDocument,
    LoadedPage,
)
from atlas.ingestion.registry import (
    document_loader_registry,
    document_parser_for_suffix,
)


__all__ = [
    "SUPPORTED_SUFFIXES",
    "LoadedDocument",
    "LoadedPage",
    "load_local_document",
]


def load_local_document(path_value: str, *, allowed_roots: list[Path]) -> LoadedDocument:
    source = document_loader_registry.get("local").load(path_value, allowed_roots=allowed_roots)
    parser = document_parser_for_suffix(source.suffix)
    if parser is None:
        raise _unsupported_file_type(source)

    loaded = parser.parse(source)
    if not loaded.text.strip():
        raise AtlasError(
            ErrorCode.INVALID_REQUEST,
            f"Document has no extractable text: {loaded.path}",
            status_code=400,
            details={"path": str(loaded.path)},
        )
    return loaded


def _unsupported_file_type(source: DocumentSource) -> AtlasError:
    return AtlasError(
        ErrorCode.UNSUPPORTED_FILE_TYPE,
        f"Unsupported file type: {source.suffix}. Atlas supports PDF, Markdown, and TXT.",
        status_code=400,
        details={"path": str(source.path), "supported": sorted(SUPPORTED_SUFFIXES)},
    )
