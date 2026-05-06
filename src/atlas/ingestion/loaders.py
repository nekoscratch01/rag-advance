from dataclasses import dataclass
from pathlib import Path
import re

from atlas.core.errors import AtlasError, ErrorCode


SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


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


def load_local_document(path_value: str, *, allowed_roots: list[Path]) -> LoadedDocument:
    from atlas.ingestion.path_policy import resolve_allowed_document_path

    path = resolve_allowed_document_path(path_value, allowed_roots)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise AtlasError(
            ErrorCode.UNSUPPORTED_FILE_TYPE,
            f"Unsupported file type: {suffix}. Atlas supports PDF, Markdown, and TXT.",
            status_code=400,
            details={"path": str(path), "supported": sorted(SUPPORTED_SUFFIXES)},
        )
    if not path.exists() or not path.is_file():
        raise AtlasError(
            ErrorCode.INVALID_REQUEST,
            f"File does not exist: {path}",
            status_code=400,
            details={"path": str(path)},
        )

    if suffix == ".pdf":
        pages = _load_pdf_pages(path)
        text = "\n\n".join(page.text for page in pages if page.text.strip())
        file_type = "pdf"
    else:
        text = path.read_text(encoding="utf-8")
        pages = [LoadedPage(page_number=None, text=text)]
        file_type = "markdown" if suffix in {".md", ".markdown"} else "txt"

    if not text.strip():
        raise AtlasError(
            ErrorCode.INVALID_REQUEST,
            f"Document has no extractable text: {path}",
            status_code=400,
            details={"path": str(path)},
        )

    return LoadedDocument(
        path=path,
        title=_extract_title(path, text),
        text=text,
        file_type=file_type,
        language=_detect_language(text),
        pages=pages,
    )


def _load_pdf_pages(path: Path) -> list[LoadedPage]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise AtlasError(
            ErrorCode.CONFIGURATION_ERROR,
            "PDF support requires pypdf. Install dependencies with `python -m pip install -e .`.",
            status_code=500,
        ) from exc

    reader = PdfReader(str(path))
    pages: list[LoadedPage] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(LoadedPage(page_number=index, text=text))
    return pages


def _extract_title(path: Path, text: str) -> str:
    match = re.search(r"^\s*#\s+(.+?)\s*$", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return path.name


def _detect_language(text: str) -> str:
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-z]+", text))
    if cjk_chars >= latin_words:
        return "zh"
    return "en"
