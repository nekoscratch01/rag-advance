from __future__ import annotations

import hashlib
import inspect
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from atlas.ingestion.chunker import approx_token_count
from atlas.ingestion.structured.contracts import (
    ChildChunk,
    ParentChunk,
    ParsedDocumentIR as DocumentIR,
    SourceLocator,
    content_hash as structured_content_hash,
)

__all__ = [
    "ChildChunk",
    "DocumentIR",
    "ParentChunk",
    "StructuredChunkingConfig",
    "StructuredChunkingResult",
    "build_parent_child_chunks",
    "chunk_document_ir",
]

TABULAR_FILE_TYPES = {
    "csv",
    "tsv",
    "excel",
    "xls",
    "xlsx",
    "xlsm",
    "html",
    "htm",
    ".csv",
    ".tsv",
    ".xls",
    ".xlsx",
    ".xlsm",
    ".html",
    ".htm",
}

TABLE_LIKE_ELEMENT_TYPES = {
    "table",
    "table_ref",
    "table_reference",
    "table_row",
    "table_cell",
    "html_table",
    "html_row",
    "html_cell",
    "csv_row",
    "csv_cell",
    "excel_row",
    "excel_cell",
    "xlsx_row",
    "xlsx_cell",
}


@dataclass(frozen=True)
class StructuredChunkingConfig:
    child_target_tokens: int = 600
    child_overlap_tokens: int = 80
    parent_target_tokens: int = 2_400
    parent_in_main_index: bool = False


@dataclass(frozen=True)
class StructuredChunkingResult:
    parent_chunks: tuple[Any, ...]
    child_chunks: tuple[Any, ...]


@dataclass(frozen=True)
class _BoundaryBlock:
    boundary_id: str
    text: str
    section_title: str | None
    page_start: int | None
    page_end: int | None
    ordinal: int
    source_locator: Any
    source_element_ids: tuple[str, ...]
    metadata: dict[str, Any]


def chunk_document_ir(
    document_ir: Any,
    *,
    child_target_tokens: int = 600,
    child_overlap_tokens: int = 80,
    parent_target_tokens: int = 2_400,
    parent_in_main_index: bool = False,
) -> StructuredChunkingResult:
    """Chunk Markdown/TXT DocumentIR without crossing source boundaries.

    Parent chunks represent source boundaries and are children-only by default.
    Child chunks are the main-index units.
    """

    file_type = _normalized_file_type(_get(document_ir, "file_type", "mime_type", "suffix"))
    if file_type in TABULAR_FILE_TYPES:
        raise ValueError(
            "tabular_document_ir_must_use_structured_table_worker; raw CSV/Excel rows must not become "
            "ordinary text chunks"
        )

    config = StructuredChunkingConfig(
        child_target_tokens=max(1, child_target_tokens),
        child_overlap_tokens=max(0, child_overlap_tokens),
        parent_target_tokens=max(1, parent_target_tokens),
        parent_in_main_index=parent_in_main_index,
    )
    document_id = _document_id(document_ir)
    boundaries = _boundary_blocks_from_document_ir(
        document_ir,
        document_id=document_id,
        parent_target_tokens=config.parent_target_tokens,
    )

    parent_chunks: list[Any] = []
    child_chunks: list[Any] = []

    for boundary in boundaries:
        parent_id = _stable_id("v4par", document_id, boundary.boundary_id, boundary.text)
        child_drafts = _child_texts_for_boundary(
            boundary.text,
            target_tokens=config.child_target_tokens,
            overlap_tokens=config.child_overlap_tokens,
        )
        child_ids = tuple(
            _stable_id("v4chk", document_id, parent_id, str(index), text)
            for index, text in enumerate(child_drafts)
        )

        parent_chunks.append(
            _parent_chunk(
                document_id=document_id,
                parent_id=parent_id,
                boundary=boundary,
                child_ids=child_ids,
                include_in_main_index=config.parent_in_main_index,
            )
        )
        for local_index, (child_id, child_text) in enumerate(
            zip(child_ids, child_drafts, strict=True)
        ):
            child_chunks.append(
                _child_chunk(
                    document_id=document_id,
                    child_id=child_id,
                    parent_id=parent_id,
                    boundary=boundary,
                    child_text=child_text,
                    chunk_index=len(child_chunks),
                    child_index=local_index,
                )
            )

    return StructuredChunkingResult(
        parent_chunks=tuple(parent_chunks),
        child_chunks=tuple(child_chunks),
    )


def build_parent_child_chunks(document_ir: Any, **kwargs: Any) -> StructuredChunkingResult:
    return chunk_document_ir(document_ir, **kwargs)


def _boundary_blocks_from_document_ir(
    document_ir: Any,
    *,
    document_id: str,
    parent_target_tokens: int,
) -> tuple[_BoundaryBlock, ...]:
    blocks = _explicit_ir_blocks(document_ir)
    if blocks is None:
        text = str(_get(document_ir, "text", "content", default="") or "")
        blocks = _markdown_or_text_blocks(text)
    if not blocks:
        return ()
    document_locator = _get(document_ir, "source_locator", default=None)
    source_uri = _optional_str(
        _get(
            document_ir,
            "source_uri",
            "path",
            default=_get(document_locator, "source_uri", "source_path", default=None),
        )
    )

    normalized = [
        block
        for block in (
            _coerce_boundary_block(
                block,
                document_id=document_id,
                ordinal=index,
                document_locator=document_locator,
                source_uri=source_uri,
            )
            for index, block in enumerate(blocks)
        )
        if block.text.strip()
    ]
    if not normalized:
        return ()

    split: list[_BoundaryBlock] = []
    for block in normalized:
        split.extend(_split_large_parent_boundary(block, parent_target_tokens=parent_target_tokens))
    return tuple(split)


def _explicit_ir_blocks(document_ir: Any) -> list[Any] | None:
    for name in ("blocks", "boundaries", "sections", "pages", "elements"):
        value = _get(document_ir, name)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if not value:
                continue
            return [
                item
                for item in value
                if _get(item, "text", "content", default="")
                and not _is_table_like_element(item)
            ]
    return None


def _is_table_like_element(block: Any) -> bool:
    raw = _get(block, "element_type", "type", "kind", "block_type", default="")
    normalized = str(raw or "").strip().lower().replace("-", "_")
    return normalized in TABLE_LIKE_ELEMENT_TYPES


def _coerce_boundary_block(
    block: Any,
    *,
    document_id: str,
    ordinal: int,
    document_locator: Any,
    source_uri: str | None,
) -> _BoundaryBlock:
    text = str(_get(block, "text", "content", default="") or "").strip()
    section_title = _optional_str(
        _get(block, "section_title", "heading", "title", "name", default=None)
    )
    locator = _get(block, "source_locator", default=None)
    page_start = _optional_int(
        _get(
            block,
            "page_start",
            "page_number",
            "page",
            default=_get(locator, "page_start", "page_number", "page", default=None),
        )
    )
    page_end = _optional_int(
        _get(
            block,
            "page_end",
            "page_number",
            "page",
            default=_get(locator, "page_end", "page_number", "page", default=page_start),
        )
    )
    raw_boundary_id = _get(block, "boundary_id", "block_id", "section_id", "page_id", default=None)
    raw_boundary_id = raw_boundary_id or _get(block, "element_id", default=None)
    boundary_id = str(raw_boundary_id or _stable_id("v4bnd", document_id, str(ordinal), text))
    source_element_ids = _source_element_ids_for_boundary(
        block=block,
        locator=locator,
        boundary_id=boundary_id,
    )
    source_locator = _source_locator_for_boundary(
        locator=locator,
        document_locator=document_locator,
        document_id=document_id,
        source_uri=source_uri,
        boundary_id=boundary_id,
        page_start=page_start,
        page_end=page_end,
    )
    metadata = _metadata_dict(block)
    metadata.update(
        {
            "source_boundary_id": boundary_id,
            "source_boundary_ordinal": ordinal,
            "source_element_ids": source_element_ids,
            "boundary_first": True,
        }
    )
    return _BoundaryBlock(
        boundary_id=boundary_id,
        text=text,
        section_title=section_title,
        page_start=page_start,
        page_end=page_end,
        ordinal=ordinal,
        source_locator=source_locator,
        source_element_ids=source_element_ids,
        metadata=metadata,
    )


def _markdown_or_text_blocks(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if not text:
        return []

    if re.search(r"^\s{0,3}#{1,6}\s+\S+", text, flags=re.MULTILINE):
        return _markdown_heading_blocks(text)
    return _plain_text_blocks(text)


def _markdown_heading_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current_title: str | None = None
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        value = "\n".join(current).strip()
        if value:
            blocks.append({"text": value, "section_title": current_title})
        current = []

    for line in text.splitlines():
        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            flush()
            current_title = heading.group(2).strip()
            current = [line.rstrip()]
            continue
        current.append(line.rstrip())

    flush()
    return blocks


def _plain_text_blocks(text: str) -> list[dict[str, Any]]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n+", text) if item.strip()]
    if not paragraphs:
        return [{"text": text.strip(), "section_title": None}]
    return [{"text": paragraph, "section_title": None} for paragraph in paragraphs]


def _split_large_parent_boundary(
    boundary: _BoundaryBlock,
    *,
    parent_target_tokens: int,
) -> list[_BoundaryBlock]:
    if approx_token_count(boundary.text) <= parent_target_tokens:
        return [boundary]

    paragraphs = _paragraph_units(boundary.text)
    grouped: list[_BoundaryBlock] = []
    current: list[str] = []
    part_index = 0

    def emit() -> None:
        nonlocal current, part_index
        text = "\n\n".join(item for item in current if item.strip()).strip()
        if not text:
            current = []
            return
        metadata = {
            **boundary.metadata,
            "source_boundary_id": boundary.boundary_id,
            "source_boundary_part": part_index,
            "parent_boundary_split": True,
        }
        grouped.append(
            _BoundaryBlock(
                boundary_id=f"{boundary.boundary_id}:part:{part_index}",
                text=text,
                section_title=boundary.section_title,
                page_start=boundary.page_start,
                page_end=boundary.page_end,
                ordinal=boundary.ordinal,
                source_locator=boundary.source_locator,
                source_element_ids=boundary.source_element_ids,
                metadata=metadata,
            )
        )
        current = []
        part_index += 1

    for paragraph in paragraphs:
        pieces = _split_oversized_unit(paragraph, parent_target_tokens)
        for piece in pieces:
            candidate = "\n\n".join([*current, piece]).strip()
            if current and approx_token_count(candidate) > parent_target_tokens:
                emit()
            current.append(piece)
    emit()
    return grouped


def _child_texts_for_boundary(
    text: str,
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> tuple[str, ...]:
    units: list[str] = []
    for paragraph in _paragraph_units(text):
        units.extend(_split_oversized_unit(paragraph, target_tokens))

    chunks: list[str] = []
    current: list[str] = []

    def emit(*, keep_overlap: bool) -> None:
        nonlocal current
        merged = "\n\n".join(part.strip() for part in current if part.strip()).strip()
        if not merged:
            current = []
            return
        chunks.append(merged)
        overlap = _tail_by_approx_tokens(merged, overlap_tokens) if keep_overlap else ""
        current = [overlap] if overlap else []

    for unit in units:
        candidate = "\n\n".join([*current, unit]).strip()
        if current and approx_token_count(candidate) > target_tokens:
            emit(keep_overlap=True)
        current.append(unit)

    emit(keep_overlap=False)
    return tuple(chunks)


def _paragraph_units(text: str) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n+", text) if item.strip()]
    return paragraphs or [text.strip()]


def _split_oversized_unit(text: str, target_tokens: int) -> list[str]:
    if approx_token_count(text) <= target_tokens:
        return [text.strip()]

    sentences = [
        item.strip()
        for item in re.split(r"(?<=[。！？.!?])\s+", text)
        if item.strip()
    ]
    if len(sentences) <= 1:
        return _split_by_token_budget(text, target_tokens)

    pieces: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*current, sentence]).strip()
        if current and approx_token_count(candidate) > target_tokens:
            pieces.append(" ".join(current).strip())
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        pieces.append(" ".join(current).strip())
    return pieces


def _split_by_token_budget(text: str, token_budget: int) -> list[str]:
    tokens = _token_pieces(text)
    pieces: list[str] = []
    current: list[str] = []
    used_tokens = 0
    for token in tokens:
        token_cost = 0 if token.isspace() else approx_token_count(token)
        if current and token_cost and used_tokens + token_cost > token_budget:
            pieces.append("".join(current).strip())
            current = []
            used_tokens = 0
        current.append(token)
        used_tokens += token_cost
    if current:
        pieces.append("".join(current).strip())
    return [piece for piece in pieces if piece]


def _tail_by_approx_tokens(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    tail: list[str] = []
    used_tokens = 0
    for piece in reversed(_token_pieces(text)):
        piece_tokens = 0 if piece.isspace() else approx_token_count(piece)
        if piece_tokens and used_tokens + piece_tokens > token_budget:
            break
        tail.append(piece)
        used_tokens += piece_tokens
    return "".join(reversed(tail)).strip()


def _token_pieces(text: str) -> list[str]:
    return re.findall(
        r"\s+|[\u4e00-\u9fff]|[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?|[^\w\s]",
        text,
    )


def _parent_chunk(
    *,
    document_id: str,
    parent_id: str,
    boundary: _BoundaryBlock,
    child_ids: tuple[str, ...],
    include_in_main_index: bool,
) -> Any:
    metadata = {
        **boundary.metadata,
        "chunk_generation": "v4_boundary_first",
        "chunk_role": "parent",
        "boundary_first": True,
        "crosses_source_boundary": False,
        "main_index": include_in_main_index,
        "include_in_main_index": include_in_main_index,
        "index_policy": "ranked_parent" if include_in_main_index else "children_only",
        "source_element_ids": boundary.source_element_ids,
        "content_hash": structured_content_hash(boundary.text),
    }
    payload = {
        "id": parent_id,
        "parent_chunk_id": parent_id,
        "parent_id": parent_id,
        "chunk_id": parent_id,
        "document_id": document_id,
        "text": boundary.text,
        "chunk_type": "parent",
        "kind": "parent",
        "section_title": boundary.section_title,
        "page_start": boundary.page_start,
        "page_end": boundary.page_end,
        "token_count": approx_token_count(boundary.text),
        "child_ids": child_ids,
        "child_chunk_ids": child_ids,
        "indexable": include_in_main_index,
        "main_index": include_in_main_index,
        "include_in_main_index": include_in_main_index,
        "index_policy": metadata["index_policy"],
        "source_locator": boundary.source_locator,
        "source_element_ids": boundary.source_element_ids,
        "content_hash": metadata["content_hash"],
        "metadata": metadata,
        "metadata_json": metadata,
    }
    return _make_contract(ParentChunk, payload)


def _child_chunk(
    *,
    document_id: str,
    child_id: str,
    parent_id: str,
    boundary: _BoundaryBlock,
    child_text: str,
    chunk_index: int,
    child_index: int,
) -> Any:
    metadata = {
        **boundary.metadata,
        "chunk_generation": "v4_boundary_first",
        "chunk_role": "child",
        "boundary_first": True,
        "crosses_source_boundary": False,
        "source_boundary_id": boundary.boundary_id,
        "parent_chunk_id": parent_id,
        "main_index": True,
        "include_in_main_index": True,
        "index_policy": "ranked_child",
        "source_element_ids": boundary.source_element_ids,
        "content_hash": structured_content_hash(child_text),
    }
    payload = {
        "id": child_id,
        "child_chunk_id": child_id,
        "chunk_id": child_id,
        "parent_chunk_id": parent_id,
        "parent_id": parent_id,
        "document_id": document_id,
        "text": child_text,
        "chunk_index": chunk_index,
        "child_index": child_index,
        "section_title": boundary.section_title,
        "page_start": boundary.page_start,
        "page_end": boundary.page_end,
        "token_count": approx_token_count(child_text),
        "indexable": True,
        "main_index": True,
        "include_in_main_index": True,
        "index_policy": "ranked_child",
        "source_locator": boundary.source_locator,
        "source_element_ids": boundary.source_element_ids,
        "content_hash": metadata["content_hash"],
        "metadata": metadata,
        "metadata_json": metadata,
    }
    return _make_contract(ChildChunk, payload)


def _source_locator_for_boundary(
    *,
    locator: Any,
    document_locator: Any,
    document_id: str,
    source_uri: str | None,
    boundary_id: str,
    page_start: int | None,
    page_end: int | None,
) -> Any:
    payload = _locator_payload(document_locator)
    payload.update(
        {
            key: value
            for key, value in _locator_payload(locator).items()
            if value not in (None, "", (), [], {})
        }
    )
    payload.update(
        {
            "document_id": payload.get("document_id") or document_id,
            "source_uri": payload.get("source_uri") or source_uri,
            "element_id": payload.get("element_id") or boundary_id,
            "page_start": payload.get("page_start") if payload.get("page_start") is not None else page_start,
            "page_end": payload.get("page_end") if payload.get("page_end") is not None else page_end,
            "locator_precision": payload.get("locator_precision")
            if payload.get("locator_precision") not in {None, "", "unknown"}
            else ("page" if page_start is not None else "element"),
            "locator_confidence": payload.get("locator_confidence")
            if payload.get("locator_confidence") not in {None, 0.0}
            else 1.0,
            "is_exact": True,
            "locator_method": payload.get("locator_method")
            if payload.get("locator_method") not in {None, "", "unspecified"}
            else "parser",
        }
    )
    if page_start is not None and page_start == page_end and payload.get("page_number") is None:
        payload["page_number"] = page_start
    return _make_contract(SourceLocator, payload)


def _source_element_ids_for_boundary(
    *,
    block: Any,
    locator: Any,
    boundary_id: str,
) -> tuple[str, ...]:
    raw = _get(block, "source_element_ids", "element_ids", default=None)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        values = tuple(str(item) for item in raw if str(item))
        if values:
            return values
    for candidate in (
        _get(block, "element_id", default=None),
        _get(locator, "element_id", default=None),
        boundary_id,
    ):
        if candidate:
            return (str(candidate),)
    return ()


def _locator_payload(locator: Any) -> dict[str, Any]:
    if locator is None:
        return {}
    if isinstance(locator, Mapping):
        return {str(key): value for key, value in locator.items()}
    to_payload = getattr(locator, "to_payload", None)
    if callable(to_payload):
        payload = to_payload()
        return dict(payload) if isinstance(payload, Mapping) else {}
    result: dict[str, Any] = {}
    for name in _source_locator_field_names():
        if hasattr(locator, name):
            result[name] = getattr(locator, name)
    return result


def _source_locator_field_names() -> tuple[str, ...]:
    names = _contract_field_names(SourceLocator)
    if names:
        return tuple(sorted(names))
    return (
        "source_uri",
        "source_path",
        "document_id",
        "storage_ref",
        "storage_format",
        "storage_offset",
        "storage_length",
        "page_number",
        "page_start",
        "page_end",
        "sheet_name",
        "element_id",
        "table_id",
        "table_range",
        "row_index",
        "row_locator",
        "column_index",
        "column_locator",
        "cell_ref",
        "char_start",
        "char_end",
        "bbox",
        "locator_precision",
        "locator_confidence",
        "is_exact",
        "locator_method",
        "locator_version",
    )


def _make_contract(contract_cls: type[Any], payload: dict[str, Any]) -> Any:
    field_names = _contract_field_names(contract_cls)
    if field_names is None:
        return contract_cls(**payload)
    filtered = {key: value for key, value in payload.items() if key in field_names}
    try:
        return contract_cls(**filtered)
    except TypeError:
        return contract_cls(**payload)


def _contract_field_names(contract_cls: type[Any]) -> set[str] | None:
    model_fields = getattr(contract_cls, "model_fields", None)
    if isinstance(model_fields, Mapping):
        return set(model_fields)
    if is_dataclass(contract_cls):
        return {item.name for item in fields(contract_cls)}
    annotations = getattr(contract_cls, "__annotations__", None)
    if isinstance(annotations, Mapping) and annotations:
        return set(annotations)
    try:
        signature = inspect.signature(contract_cls)
    except (TypeError, ValueError):
        return None
    names = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name != "self"
    }
    return names or None


def _metadata_dict(value: Any) -> dict[str, Any]:
    metadata = _get(value, "metadata", "metadata_json", default=None)
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _document_id(document_ir: Any) -> str:
    explicit = _get(document_ir, "document_id", "doc_id", "id", default=None)
    if explicit:
        return str(explicit)
    text = str(_get(document_ir, "text", "content", default="") or "")
    return _stable_id("v4doc", text)


def _normalized_file_type(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("."):
        return raw
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    return raw


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _stable_id(prefix: str, *parts: str) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"
