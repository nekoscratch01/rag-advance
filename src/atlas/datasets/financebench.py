import hashlib
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

from atlas.datasets.jsonl import write_json_atomic, write_jsonl_atomic
from atlas.datasets.schemas import (
    FinanceBenchChildChunk,
    FinanceBenchDocument,
    FinanceBenchPage,
    FinanceBenchParentBlock,
    FinanceBenchPrepareResult,
)
from atlas.ingestion.chunker import chunk_text


DATASET_ID = "PatronusAI/financebench"
DATASET_SPLIT = "train"
FALLBACK_PDF_BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"
DEFAULT_TARGET_TOKENS = 600
DEFAULT_OVERLAP_TOKENS = 80
PARSER_NAME = "pypdf"


class FinanceBenchPreparationError(RuntimeError):
    pass


@dataclass
class _DocumentRequest:
    doc_name: str
    doc_link: str | None
    declared_pdf_sha256: str | None
    row_ids: list[str]


@dataclass(frozen=True)
class _DownloadedPdf:
    content: bytes
    source_url: str
    attempts: list[dict[str, Any]]


@dataclass(frozen=True)
class _ExtractedPage:
    page_number: int
    text: str


def prepare_financebench(
    *,
    out_dir: str | Path,
    evals_path: str | Path,
    limit: int | None = None,
    strict: bool = False,
    revision: str | None = None,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> FinanceBenchPrepareResult:
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to zero")

    out_path = Path(out_dir)
    evals_output_path = Path(evals_path)

    rows = _load_rows(revision=revision, limit=limit)
    requests, manifest_records = _build_document_requests(rows, strict=strict)

    pages: list[FinanceBenchPage] = []
    parent_blocks: list[FinanceBenchParentBlock] = []
    child_chunks: list[FinanceBenchChildChunk] = []
    pages_by_doc_name: dict[str, list[FinanceBenchPage]] = {}
    doc_name_to_canonical: dict[str, str] = {}
    sha_to_doc_id: dict[str, str] = {}
    sha_to_doc_name: dict[str, str] = {}
    sha_to_local_pdf_path: dict[str, str] = {}
    canonical_aliases: dict[str, list[str]] = {}

    httpx = _import_httpx()
    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(120.0, connect=30.0),
        headers={"User-Agent": "atlas-financebench-adapter/0.1"},
    ) as client:
        for request in requests:
            try:
                downloaded = _download_pdf(client, request)
                pdf_sha256 = _sha256_bytes(downloaded.content)
                _validate_declared_sha(request, pdf_sha256)

                duplicate_of = sha_to_doc_id.get(pdf_sha256)
                if duplicate_of is not None:
                    canonical_name = sha_to_doc_name[pdf_sha256]
                    doc_name_to_canonical[request.doc_name] = canonical_name
                    canonical_aliases.setdefault(duplicate_of, []).append(request.doc_name)
                    manifest_records.append(
                        FinanceBenchDocument(
                            document_id=_stable_id("fbdoc_alias", request.doc_name, pdf_sha256),
                            doc_name=request.doc_name,
                            doc_link=request.doc_link,
                            source_url=downloaded.source_url,
                            fallback_url=_fallback_pdf_url(request.doc_name),
                            pdf_sha256=pdf_sha256,
                            status="skipped_duplicate_sha",
                            row_ids=request.row_ids,
                            local_pdf_path=sha_to_local_pdf_path.get(pdf_sha256),
                            corpus_version=_corpus_version_id(pdf_sha256, target_tokens, overlap_tokens),
                            byte_count=len(downloaded.content),
                            duplicate_of=duplicate_of,
                            download_attempts=downloaded.attempts,
                        )
                    )
                    continue

                document_id = _document_id_for_sha(pdf_sha256)
                local_pdf_path = _write_pdf_atomic(out_path, request.doc_name, downloaded.content)
                extracted_pages = _extract_pdf_pages(downloaded.content)
                doc_pages, doc_parent_blocks, doc_child_chunks = _make_page_and_chunk_records(
                    document_id=document_id,
                    doc_name=request.doc_name,
                    pdf_sha256=pdf_sha256,
                    source_uri=downloaded.source_url,
                    extracted_pages=extracted_pages,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                )

                sha_to_doc_id[pdf_sha256] = document_id
                sha_to_doc_name[pdf_sha256] = request.doc_name
                sha_to_local_pdf_path[pdf_sha256] = str(local_pdf_path)
                doc_name_to_canonical[request.doc_name] = request.doc_name
                pages_by_doc_name[request.doc_name] = doc_pages
                pages.extend(doc_pages)
                parent_blocks.extend(doc_parent_blocks)
                child_chunks.extend(doc_child_chunks)

                manifest_records.append(
                    FinanceBenchDocument(
                        document_id=document_id,
                        doc_name=request.doc_name,
                        doc_link=request.doc_link,
                        source_url=downloaded.source_url,
                        fallback_url=_fallback_pdf_url(request.doc_name),
                        pdf_sha256=pdf_sha256,
                        status="parsed",
                        row_ids=request.row_ids,
                        local_pdf_path=str(local_pdf_path),
                        parser_name=PARSER_NAME,
                        parser_version=_pypdf_version(),
                        corpus_version=_corpus_version_id(pdf_sha256, target_tokens, overlap_tokens),
                        byte_count=len(downloaded.content),
                        page_count=len(doc_pages),
                        chunk_count=len(doc_child_chunks),
                        download_attempts=downloaded.attempts,
                    )
                )
            except Exception as exc:
                if strict:
                    raise
                manifest_records.append(
                    _failure_record(
                        doc_name=request.doc_name,
                        doc_link=request.doc_link,
                        row_ids=request.row_ids,
                        kind="document_prepare_failed",
                        reason=str(exc),
                    )
                )

    eval_cases = _make_eval_cases(
        rows,
        doc_name_to_canonical=doc_name_to_canonical,
        pages_by_doc_name=pages_by_doc_name,
    )

    manifest_payloads = _manifest_payloads(manifest_records, canonical_aliases)
    failure_count = sum(1 for item in manifest_payloads if item.get("status") == "failed")

    manifest_path = out_path / "manifest.jsonl"
    pages_path = out_path / "parsed" / "pages.jsonl"
    parent_blocks_path = out_path / "parsed" / "parent_blocks.jsonl"
    child_chunks_path = out_path / "parsed" / "child_chunks.jsonl"
    chunks_path = out_path / "parsed" / "chunks.jsonl"
    version_path = out_path / "corpus_version.json"

    write_jsonl_atomic(manifest_path, manifest_payloads)
    write_jsonl_atomic(pages_path, [page.to_json() for page in pages])
    write_jsonl_atomic(parent_blocks_path, [parent.to_json() for parent in parent_blocks])
    write_jsonl_atomic(child_chunks_path, [chunk.to_json() for chunk in child_chunks])
    write_jsonl_atomic(chunks_path, [chunk.to_json() for chunk in child_chunks])
    _write_yaml_atomic(evals_output_path, {"cases": eval_cases})
    write_json_atomic(
        version_path,
        {
            "dataset_id": DATASET_ID,
            "split": DATASET_SPLIT,
            "revision": revision,
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "row_count": len(rows),
            "manifest_count": len(manifest_payloads),
            "document_count": sum(
                1 for item in manifest_payloads if item.get("status") == "parsed"
            ),
            "duplicate_sha_count": sum(
                1 for item in manifest_payloads if item.get("status") == "skipped_duplicate_sha"
            ),
            "failure_count": failure_count,
            "page_count": len(pages),
            "parent_block_count": len(parent_blocks),
            "chunk_count": len(child_chunks),
            "chunking": {
                "target_tokens": target_tokens,
                "overlap_tokens": overlap_tokens,
                "page_bounded": True,
                "parent_child": True,
            },
            "parser": {
                "name": PARSER_NAME,
                "version": _pypdf_version(),
            },
            "outputs": {
                "manifest": str(manifest_path),
                "pdfs": str(out_path / "pdfs"),
                "pages": str(pages_path),
                "parent_blocks": str(parent_blocks_path),
                "child_chunks": str(child_chunks_path),
                "chunks": str(chunks_path),
                "evals": str(evals_output_path),
            },
        },
    )

    return FinanceBenchPrepareResult(
        dataset_id=DATASET_ID,
        revision=revision,
        row_count=len(rows),
        manifest_count=len(manifest_payloads),
        page_count=len(pages),
        chunk_count=len(child_chunks),
        failure_count=failure_count,
        out_dir=str(out_path),
        evals_path=str(evals_output_path),
    )


def _load_rows(*, revision: str | None, limit: int | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise FinanceBenchPreparationError(
            "FinanceBench preparation requires the `datasets` package. "
            "Install it before running scripts/prepare_financebench.py."
        ) from exc

    kwargs: dict[str, Any] = {"split": DATASET_SPLIT}
    if revision:
        kwargs["revision"] = revision
    dataset = load_dataset(DATASET_ID, **kwargs)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [_json_safe(dict(row)) for row in dataset]


def _build_document_requests(
    rows: list[Mapping[str, Any]],
    *,
    strict: bool,
) -> tuple[list[_DocumentRequest], list[FinanceBenchDocument]]:
    requests_by_name: dict[str, _DocumentRequest] = {}
    doc_link_to_name: dict[str, str] = {}
    failures: list[FinanceBenchDocument] = []

    for index, row in enumerate(rows):
        row_id = _case_id(row, index)
        doc_name = _clean_optional(row.get("doc_name"))
        doc_link = _clean_optional(row.get("doc_link"))
        declared_sha = _clean_optional(row.get("pdf_sha256"))
        declared_sha = declared_sha.lower() if declared_sha else None

        if doc_name is None:
            _record_or_raise(
                failures,
                strict=strict,
                doc_name="",
                doc_link=doc_link,
                row_ids=[row_id],
                kind="missing_doc_name",
                reason=f"Row {row_id} does not include doc_name.",
            )
            continue

        if doc_link:
            linked_doc_name = doc_link_to_name.get(doc_link)
            if linked_doc_name and linked_doc_name != doc_name:
                _record_or_raise(
                    failures,
                    strict=strict,
                    doc_name=doc_name,
                    doc_link=doc_link,
                    row_ids=[row_id],
                    kind="doc_link_conflict",
                    reason=(
                        f"doc_link {doc_link!r} maps to multiple doc_name values: "
                        f"{linked_doc_name!r} and {doc_name!r}."
                    ),
                )
                continue

        existing = requests_by_name.get(doc_name)
        if existing is None:
            existing = _DocumentRequest(
                doc_name=doc_name,
                doc_link=doc_link,
                declared_pdf_sha256=declared_sha,
                row_ids=[],
            )
            requests_by_name[doc_name] = existing
        else:
            if doc_link and existing.doc_link and doc_link != existing.doc_link:
                _record_or_raise(
                    failures,
                    strict=strict,
                    doc_name=doc_name,
                    doc_link=doc_link,
                    row_ids=[row_id],
                    kind="doc_name_conflict",
                    reason=(
                        f"doc_name {doc_name!r} maps to multiple doc_link values: "
                        f"{existing.doc_link!r} and {doc_link!r}."
                    ),
                )
                continue
            if doc_link and existing.doc_link is None:
                existing.doc_link = doc_link

            if (
                declared_sha
                and existing.declared_pdf_sha256
                and declared_sha != existing.declared_pdf_sha256
            ):
                _record_or_raise(
                    failures,
                    strict=strict,
                    doc_name=doc_name,
                    doc_link=doc_link,
                    row_ids=[row_id],
                    kind="declared_sha_conflict",
                    reason=(
                        f"doc_name {doc_name!r} maps to multiple pdf_sha256 values: "
                        f"{existing.declared_pdf_sha256!r} and {declared_sha!r}."
                    ),
                )
                continue
            if declared_sha and existing.declared_pdf_sha256 is None:
                existing.declared_pdf_sha256 = declared_sha

        existing.row_ids.append(row_id)

        if doc_link:
            doc_link_to_name[doc_link] = doc_name

    return list(requests_by_name.values()), failures


def _record_or_raise(
    failures: list[FinanceBenchDocument],
    *,
    strict: bool,
    doc_name: str,
    doc_link: str | None,
    row_ids: list[str],
    kind: str,
    reason: str,
) -> None:
    if strict:
        raise FinanceBenchPreparationError(reason)
    failures.append(
        _failure_record(
            doc_name=doc_name,
            doc_link=doc_link,
            row_ids=row_ids,
            kind=kind,
            reason=reason,
        )
    )


def _download_pdf(client: Any, request: _DocumentRequest) -> _DownloadedPdf:
    attempts: list[dict[str, Any]] = []
    for url in _candidate_pdf_urls(request):
        try:
            response = client.get(url)
            attempt = {
                "url": url,
                "status_code": response.status_code,
                "ok": response.is_success,
            }
            if response.is_success and _looks_like_pdf(response.content):
                attempts.append(attempt)
                return _DownloadedPdf(
                    content=response.content,
                    source_url=url,
                    attempts=attempts,
                )
            if response.is_success:
                attempt["error"] = "response did not look like a PDF"
            else:
                attempt["error"] = response.reason_phrase
            attempts.append(attempt)
        except Exception as exc:
            attempts.append(
                {
                    "url": url,
                    "ok": False,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )

    reason = "; ".join(
        f"{item['url']} -> {item.get('status_code', 'error')} {item.get('error', '')}".strip()
        for item in attempts
    )
    raise FinanceBenchPreparationError(
        f"Failed to download PDF for {request.doc_name!r}. Attempts: {reason}"
    )


def _candidate_pdf_urls(request: _DocumentRequest) -> list[str]:
    urls = []
    if request.doc_link:
        urls.append(request.doc_link)
    urls.append(_fallback_pdf_url(request.doc_name))
    return list(dict.fromkeys(urls))


def _fallback_pdf_url(doc_name: str) -> str:
    return f"{FALLBACK_PDF_BASE}/{quote(doc_name, safe='')}.pdf"


def _looks_like_pdf(content: bytes) -> bool:
    return b"%PDF" in content[:1024]


def _validate_declared_sha(request: _DocumentRequest, actual_sha256: str) -> None:
    if request.declared_pdf_sha256 and request.declared_pdf_sha256 != actual_sha256:
        raise FinanceBenchPreparationError(
            f"Declared pdf_sha256 for {request.doc_name!r} does not match downloaded PDF: "
            f"{request.declared_pdf_sha256!r} != {actual_sha256!r}."
        )


def _extract_pdf_pages(content: bytes) -> list[_ExtractedPage]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise FinanceBenchPreparationError(
            "FinanceBench PDF parsing requires the `pypdf` package."
        ) from exc

    reader = PdfReader(BytesIO(content))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            pass

    pages = [
        _ExtractedPage(page_number=index, text=page.extract_text() or "")
        for index, page in enumerate(reader.pages, start=1)
    ]
    if not pages:
        raise FinanceBenchPreparationError("Downloaded PDF contained no pages.")
    return pages


def _make_page_and_chunk_records(
    *,
    document_id: str,
    doc_name: str,
    pdf_sha256: str,
    source_uri: str | None,
    extracted_pages: list[_ExtractedPage],
    target_tokens: int,
    overlap_tokens: int,
) -> tuple[list[FinanceBenchPage], list[FinanceBenchParentBlock], list[FinanceBenchChildChunk]]:
    pages: list[FinanceBenchPage] = []
    parent_blocks: list[FinanceBenchParentBlock] = []
    chunks: list[FinanceBenchChildChunk] = []
    chunk_index = 0

    for page in extracted_pages:
        page_id = f"{document_id}_p{page.page_number:04d}"
        text_sha256 = _sha256_text(page.text)
        parent_id = _parent_id_for_page(document_id, page.page_number)
        child_ids: list[str] = []

        pages.append(
            FinanceBenchPage(
                page_id=page_id,
                document_id=document_id,
                doc_name=doc_name,
                page_number=page.page_number,
                text=page.text,
                text_sha256=text_sha256,
                pdf_sha256=pdf_sha256,
                source_uri=source_uri,
            )
        )

        if page.text.strip():
            for draft in chunk_text(
                page.text,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            ):
                chunk_id = f"fbchk_{document_id.removeprefix('fbdoc_')}_{chunk_index:05d}"
                chunks.append(
                    FinanceBenchChildChunk(
                        chunk_id=chunk_id,
                        parent_id=parent_id,
                        document_id=document_id,
                        doc_name=doc_name,
                        chunk_index=chunk_index,
                        text=draft.text,
                        text_hash=_sha256_text(draft.text),
                        section_title=draft.section_title,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        token_count=draft.token_count,
                        pdf_sha256=pdf_sha256,
                        source_uri=source_uri,
                    )
                )
                child_ids.append(chunk_id)
                chunk_index += 1

        parent_blocks.append(
            FinanceBenchParentBlock(
                parent_id=parent_id,
                document_id=document_id,
                parent_type="page",
                page_start=page.page_number,
                page_end=page.page_number,
                text=page.text,
                child_ids_json=child_ids,
                metadata_json={
                    "doc_name": doc_name,
                    "page_id": page_id,
                    "pdf_sha256": pdf_sha256,
                    "source_uri": source_uri,
                    "text_sha256": text_sha256,
                },
            )
        )

    return pages, parent_blocks, chunks


def _make_eval_cases(
    rows: list[Mapping[str, Any]],
    *,
    doc_name_to_canonical: Mapping[str, str],
    pages_by_doc_name: Mapping[str, list[FinanceBenchPage]],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        doc_name = _clean_optional(row.get("doc_name"))
        canonical_doc_name = doc_name_to_canonical.get(doc_name or "", doc_name or "")
        case: dict[str, Any] = {
            "id": _case_id(row, index),
            "question": str(row.get("question") or ""),
            "expected_confidence": "supported",
            "expected_sources": [canonical_doc_name] if canonical_doc_name else [],
            "expected_keywords": _expected_keywords(row.get("answer")),
            "answer": row.get("answer"),
            "metadata": {
                "dataset": "financebench",
                "company": row.get("company"),
                "doc_name": doc_name,
                "canonical_doc_name": canonical_doc_name or None,
                "doc_link": row.get("doc_link"),
                "doc_type": row.get("doc_type"),
                "doc_period": row.get("doc_period"),
                "question_type": row.get("question_type"),
                "question_reasoning": row.get("question_reasoning"),
                "dataset_subset_label": row.get("dataset_subset_label"),
                "gics_sector": row.get("gics_sector"),
            },
            "justification": row.get("justification"),
            "evidence": [],
        }

        evidence_items = row.get("evidence") or []
        if isinstance(evidence_items, list):
            case["evidence"] = [
                _eval_evidence_payload(
                    evidence,
                    fallback_doc_name=doc_name,
                    fallback_canonical_doc_name=canonical_doc_name,
                    doc_name_to_canonical=doc_name_to_canonical,
                    pages_by_doc_name=pages_by_doc_name,
                )
                for evidence in evidence_items
                if isinstance(evidence, Mapping)
            ]
        cases.append(case)
    return cases


def _eval_evidence_payload(
    evidence: Mapping[str, Any],
    *,
    fallback_doc_name: str | None,
    fallback_canonical_doc_name: str | None,
    doc_name_to_canonical: Mapping[str, str],
    pages_by_doc_name: Mapping[str, list[FinanceBenchPage]],
) -> dict[str, Any]:
    evidence_doc_name = _clean_optional(evidence.get("doc_name")) or fallback_doc_name
    canonical_doc_name = doc_name_to_canonical.get(
        evidence_doc_name or "",
        fallback_canonical_doc_name or evidence_doc_name or "",
    )
    raw_page_num = _int_or_none(evidence.get("evidence_page_num"))
    page_num_normalized = _infer_page_num(
        evidence,
        pages_by_doc_name.get(canonical_doc_name, []),
    )
    return {
        "doc_name": evidence_doc_name,
        "canonical_doc_name": canonical_doc_name or None,
        "evidence_text": evidence.get("evidence_text"),
        "evidence_text_full_page": evidence.get("evidence_text_full_page"),
        "evidence_page_num_raw": raw_page_num,
        "evidence_page_num_normalized": page_num_normalized,
        "page_num_normalized": page_num_normalized,
    }


def _infer_page_num(
    evidence: Mapping[str, Any],
    pages: list[FinanceBenchPage],
) -> int | None:
    full_page = _match_normalize(evidence.get("evidence_text_full_page"))
    snippet = _match_normalize(evidence.get("evidence_text"))
    raw_page = _int_or_none(evidence.get("evidence_page_num"))
    candidates: dict[int, int] = {}

    for page in pages:
        page_text = _match_normalize(page.text)
        score = 0
        if full_page:
            score = max(score, _page_match_score(full_page, page_text, base_score=90))
        if snippet:
            score = max(score, _page_match_score(snippet, page_text, base_score=70))
            score = max(score, _window_match_score(snippet, page_text))
        if score:
            candidates[page.page_number] = score

    if not candidates:
        return None

    def sort_key(item: tuple[int, int]) -> tuple[int, int, int]:
        page_number, score = item
        if raw_page is None:
            raw_distance = 0
        else:
            raw_distance = min(
                abs(page_number - raw_page),
                abs(page_number - (raw_page + 1)),
                abs(page_number - (raw_page - 1)),
            )
        return (-score, raw_distance, page_number)

    page_number, _score = sorted(candidates.items(), key=sort_key)[0]
    return page_number


def _page_match_score(needle: str, page_text: str, *, base_score: int) -> int:
    if not needle or not page_text:
        return 0
    if needle == page_text:
        return base_score + 10
    if needle in page_text:
        return base_score
    if len(page_text) >= 200 and page_text in needle:
        return base_score - 5
    return 0


def _window_match_score(needle: str, page_text: str) -> int:
    if len(needle) < 240 or not page_text:
        return 0
    windows = [needle[:180], needle[len(needle) // 2 : len(needle) // 2 + 180], needle[-180:]]
    hits = sum(1 for window in windows if window and window in page_text)
    return 40 + hits * 10 if hits else 0


def _manifest_payloads(
    records: list[FinanceBenchDocument],
    canonical_aliases: Mapping[str, list[str]],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for record in records:
        payload = record.to_json()
        aliases = canonical_aliases.get(record.document_id)
        if aliases:
            payload["aliases"] = sorted(set(aliases))
        payloads.append(payload)
    return payloads


def _failure_record(
    *,
    doc_name: str,
    doc_link: str | None,
    row_ids: list[str],
    kind: str,
    reason: str,
) -> FinanceBenchDocument:
    return FinanceBenchDocument(
        document_id=_stable_id("fbdoc_failed", kind, doc_name, doc_link or "", reason),
        doc_name=doc_name,
        doc_link=doc_link,
        source_url=None,
        fallback_url=_fallback_pdf_url(doc_name) if doc_name else "",
        pdf_sha256=None,
        status="failed",
        row_ids=row_ids,
        failure_kind=kind,
        failure_reason=reason,
    )


def _expected_keywords(answer: Any) -> list[str]:
    text = str(answer).strip() if answer is not None else ""
    return [text] if text else []


def _case_id(row: Mapping[str, Any], index: int) -> str:
    value = _clean_optional(row.get("financebench_id"))
    return value or f"financebench_row_{index:05d}"


def _document_id_for_sha(pdf_sha256: str) -> str:
    return f"fbdoc_{pdf_sha256[:16]}"


def _parent_id_for_page(document_id: str, page_number: int) -> str:
    return f"fbpar_{document_id.removeprefix('fbdoc_')}_p{page_number:04d}"


def _stable_id(prefix: str, *parts: str) -> str:
    joined = "\x1f".join(parts)
    return f"{prefix}_{hashlib.sha256(joined.encode('utf-8')).hexdigest()[:16]}"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _corpus_version_id(pdf_sha256: str, target_tokens: int, overlap_tokens: int) -> str:
    return _stable_id(
        "fbcorpus",
        DATASET_ID,
        DATASET_SPLIT,
        pdf_sha256,
        PARSER_NAME,
        _pypdf_version() or "",
        str(target_tokens),
        str(overlap_tokens),
    )


def _write_pdf_atomic(out_path: Path, doc_name: str, content: bytes) -> Path:
    pdf_path = out_path / "pdfs" / f"{_safe_filename(doc_name)}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = pdf_path.with_name(f".{pdf_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, pdf_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return pdf_path


def _safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return safe.strip("._") or "document"


def _pypdf_version() -> str | None:
    try:
        import pypdf
    except ImportError:
        return None
    return getattr(pypdf, "__version__", None)


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _match_normalize(value: Any) -> str:
    if value is None:
        return ""
    return "".join(char for char in str(value).casefold() if char.isalnum())


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _write_yaml_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False, width=120)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _import_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise FinanceBenchPreparationError(
            "FinanceBench PDF download requires the `httpx` package."
        ) from exc
    return httpx
