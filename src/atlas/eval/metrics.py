import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from typing import Any


def expected_confidence_hit(actual: str | None, expected: str | None) -> bool:
    if expected is None:
        return True
    return actual == expected


def source_hit(citations: list[dict[str, Any]], expected_sources: list[str]) -> bool:
    if not expected_sources:
        return True
    titles = [str(item.get("source_title", "")) for item in citations]
    return any(expected in title for expected in expected_sources for title in titles)


def keyword_hit(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    hits = sum(1 for keyword in expected_keywords if keyword in answer)
    return hits / len(expected_keywords)


def answer_gold_contains(answer: str | None, expected_answer: str | None) -> bool | None:
    if not expected_answer:
        return None
    if answer is None:
        return False
    return _normalize_text(expected_answer) in _normalize_text(answer)


def answer_numeric_match(answer: str | None, expected_answer: str | None) -> bool | None:
    if not expected_answer:
        return None

    expected_numbers = _extract_numbers(expected_answer)
    if not expected_numbers:
        return None
    actual_numbers = _extract_numbers(answer or "")
    if not actual_numbers:
        return False

    return all(
        any(_numbers_match(expected, actual) for actual in actual_numbers)
        for expected in expected_numbers
    )


def dense_retrieval_metrics(
    retrieved_top_k: list[dict[str, Any]],
    expected_evidence: list[dict[str, Any]] | dict[str, Any] | None,
    expected_sources: list[str] | None = None,
) -> dict[str, Any]:
    normalized_evidence = normalize_expected_evidence(expected_evidence, expected_sources)
    has_doc_expectation = any(
        item["document_ids"] or item["doc_hints"] for item in normalized_evidence
    )
    has_page_expectation = any(item["page_candidates"] for item in normalized_evidence)

    first_doc_rank = _first_match_rank(
        retrieved_top_k,
        normalized_evidence,
        match_page=False,
    )
    first_page_rank = _first_match_rank(
        retrieved_top_k,
        normalized_evidence,
        match_page=True,
    )

    return {
        "retrieval_doc_hit": _hit_value(first_doc_rank, has_doc_expectation),
        "retrieval_page_hit": _hit_value(first_page_rank, has_page_expectation),
        "retrieval_doc_mrr": _mrr_value(first_doc_rank, has_doc_expectation),
        "retrieval_page_mrr": _mrr_value(first_page_rank, has_page_expectation),
        "first_doc_match_rank": first_doc_rank,
        "first_page_match_rank": first_page_rank,
        "normalized_expected_evidence": normalized_evidence,
    }


def normalize_expected_evidence(
    expected_evidence: list[dict[str, Any]] | dict[str, Any] | None,
    expected_sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    source_hints = [str(source) for source in expected_sources or [] if source]
    raw_items = _as_evidence_items(expected_evidence)
    if not raw_items and source_hints:
        raw_items = [{"source_title": source} for source in source_hints]

    normalized: list[dict[str, Any]] = []
    for raw_item in raw_items:
        item = _as_mapping(raw_item)
        document_ids = _collect_string_values(item, ["document_id", "doc_id"])
        doc_hints = _collect_doc_hints(item)
        if not document_ids and not doc_hints:
            doc_hints.extend(source_hints)

        normalized.append(
            {
                "original": raw_item,
                "document_ids": _dedupe(document_ids),
                "doc_hints": _dedupe(doc_hints),
                "page_candidates": _collect_page_candidates(item),
            }
        )

    return normalized


def answer_metric_details(answer: str | None, expected_answer: str | None) -> dict[str, Any]:
    expected_numbers = _extract_numbers(expected_answer or "")
    actual_numbers = _extract_numbers(answer or "")
    matched_expected_numbers = [
        number["text"]
        for number in expected_numbers
        if any(_numbers_match(number, actual) for actual in actual_numbers)
    ]
    return {
        "answer_gold_contains": answer_gold_contains(answer, expected_answer),
        "answer_numeric_match": answer_numeric_match(answer, expected_answer),
        "expected_numbers": [number["text"] for number in expected_numbers],
        "actual_numbers": [number["text"] for number in actual_numbers],
        "matched_expected_numbers": matched_expected_numbers,
    }


def critic_metric_details(
    *,
    actual_confidence: str | None,
    expected_confidence: str | None,
    expected_answer: str | None,
    expected_evidence: list[dict[str, Any]],
    expected_sources: list[str],
    expected_keywords: list[str],
    details: dict[str, Any] | None,
) -> dict[str, Any]:
    critic = _critic_payload(details)
    pre = _dict_value(critic.get("pre"))
    post = _dict_value(critic.get("post"))
    expected_answerable = _expected_answerable(
        expected_confidence=expected_confidence,
        expected_answer=expected_answer,
        expected_evidence=expected_evidence,
        expected_sources=expected_sources,
        expected_keywords=expected_keywords,
    )

    if actual_confidence is None:
        unsupported_answer = None
        false_insufficient = None
    else:
        unsupported_answer = (
            actual_confidence == "unsupported"
            or post.get("status") == "unsupported"
            or critic.get("status") == "unsupported"
        )
        false_insufficient = actual_confidence == "insufficient" and expected_answerable

    return {
        "critic": critic,
        "critic_status": critic.get("status"),
        "critic_pre_status": pre.get("status"),
        "critic_post_status": post.get("status"),
        "critic_reasons": list(critic.get("reasons") or []),
        "critic_warnings": list(critic.get("warnings") or []),
        "expected_answerable": expected_answerable,
        "unsupported_answer": unsupported_answer,
        "false_insufficient": false_insufficient,
    }


def _hit_value(rank: int | None, has_expectation: bool) -> bool | None:
    if not has_expectation:
        return None
    return rank is not None


def _mrr_value(rank: int | None, has_expectation: bool) -> float | None:
    if not has_expectation:
        return None
    return 1.0 / rank if rank else 0.0


def _first_match_rank(
    retrieved_top_k: list[dict[str, Any]],
    normalized_evidence: list[dict[str, Any]],
    *,
    match_page: bool,
) -> int | None:
    for fallback_rank, retrieved in enumerate(retrieved_top_k, start=1):
        rank = _coerce_int(retrieved.get("rank")) or fallback_rank
        if any(
            _evidence_matches(retrieved, evidence, match_page=match_page)
            for evidence in normalized_evidence
        ):
            return rank
    return None


def _evidence_matches(
    retrieved: dict[str, Any],
    evidence: dict[str, Any],
    *,
    match_page: bool,
) -> bool:
    if match_page and not evidence["page_candidates"]:
        return False

    has_doc_expectation = bool(evidence["document_ids"] or evidence["doc_hints"])
    if has_doc_expectation and not _document_matches(retrieved, evidence):
        return False

    if match_page:
        return _page_matches(retrieved, evidence)
    return has_doc_expectation


def _document_matches(retrieved: dict[str, Any], evidence: dict[str, Any]) -> bool:
    retrieved_document_id = str(retrieved.get("document_id") or "")
    if retrieved_document_id and retrieved_document_id in evidence["document_ids"]:
        return True

    retrieved_hints = _retrieved_doc_hints(retrieved)
    for expected_hint in evidence["doc_hints"]:
        expected_key = _normalize_doc_key(expected_hint)
        if not expected_key:
            continue
        for retrieved_hint in retrieved_hints:
            retrieved_key = _normalize_doc_key(retrieved_hint)
            if not retrieved_key:
                continue
            if (
                expected_key == retrieved_key
                or expected_key in retrieved_key
                or retrieved_key in expected_key
            ):
                return True
    return False


def _page_matches(retrieved: dict[str, Any], evidence: dict[str, Any]) -> bool:
    retrieved_start = _coerce_int(retrieved.get("page_start"))
    retrieved_end = _coerce_int(retrieved.get("page_end")) or retrieved_start
    if retrieved_start is None:
        return False
    if retrieved_end is None:
        retrieved_end = retrieved_start
    if retrieved_start > retrieved_end:
        retrieved_start, retrieved_end = retrieved_end, retrieved_start

    for candidate in evidence["page_candidates"]:
        expected_start = candidate["page_start"]
        expected_end = candidate["page_end"]
        if expected_start <= retrieved_end and expected_end >= retrieved_start:
            return True
    return False


def _retrieved_doc_hints(retrieved: dict[str, Any]) -> list[str]:
    hints = _collect_doc_hints(retrieved)
    metadata = retrieved.get("metadata")
    if isinstance(metadata, dict):
        hints.extend(_collect_doc_hints(metadata))
    document_metadata = retrieved.get("document_metadata")
    if isinstance(document_metadata, dict):
        hints.extend(_collect_doc_hints(document_metadata))
    return _dedupe(hints)


def _collect_doc_hints(item: dict[str, Any]) -> list[str]:
    values = _collect_string_values(
        item,
        [
            "doc_name",
            "canonical_doc_name",
            "document_name",
            "file_name",
            "filename",
            "title",
            "source_title",
            "source",
            "source_uri",
            "doc_link",
            "document",
        ],
    )
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        values.extend(_collect_doc_hints(metadata))
    return values


def _collect_page_candidates(item: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    _add_page_range(
        candidates,
        item.get("page_start_raw"),
        item.get("page_end_raw"),
        basis="raw",
    )
    _add_page_range(
        candidates,
        item.get("raw_page_start"),
        item.get("raw_page_end"),
        basis="raw",
    )
    _add_page_range(
        candidates,
        item.get("page_start"),
        item.get("page_end"),
        basis="normalized",
    )

    for key in [
        "page_num_normalized",
        "page_number_normalized",
        "normalized_page",
        "normalized_page_num",
    ]:
        for value in _page_values(item.get(key)):
            _add_page(candidates, value, basis="normalized")

    for key in [
        "evidence_page_num_raw",
        "evidence_page_raw",
        "page_num_raw",
        "raw_page",
        "page_raw",
        "raw_pages",
    ]:
        for value in _page_values(item.get(key)):
            _add_page(candidates, value, basis="raw")

    for key in [
        "page",
        "pages",
        "page_num",
        "page_number",
        "page_numbers",
        "evidence_page",
        "evidence_page_num",
    ]:
        for value in _page_values(item.get(key)):
            _add_page(candidates, value, basis="ambiguous")

    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend(_collect_page_candidates(metadata))

    return _dedupe_page_candidates(candidates)


def _add_page_range(
    candidates: list[dict[str, Any]],
    raw_start: Any,
    raw_end: Any,
    *,
    basis: str,
) -> None:
    start = _coerce_int(raw_start)
    end = _coerce_int(raw_end) if raw_end is not None else start
    if start is None:
        return
    if end is None:
        end = start
    if start > end:
        start, end = end, start

    if basis == "normalized":
        candidates.append(
            {
                "page_start": start,
                "page_end": end,
                "basis": "normalized",
                "raw_page_start": start - 1,
                "raw_page_end": end - 1,
            }
        )
        return

    candidates.append(
        {
            "page_start": start,
            "page_end": end,
            "basis": basis,
            "raw_page_start": start,
            "raw_page_end": end,
        }
    )
    candidates.append(
        {
            "page_start": start + 1,
            "page_end": end + 1,
            "basis": f"normalized_from_{basis}",
            "raw_page_start": start,
            "raw_page_end": end,
        }
    )


def _add_page(candidates: list[dict[str, Any]], value: Any, *, basis: str) -> None:
    page = _coerce_int(value)
    if page is None:
        return
    if basis == "ambiguous":
        _add_page_range(candidates, page, page, basis="raw")
        candidates.append(
            {
                "page_start": page,
                "page_end": page,
                "basis": "normalized",
                "raw_page_start": page - 1,
                "raw_page_end": page - 1,
            }
        )
        return
    _add_page_range(candidates, page, page, basis=basis)


def _page_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return list(value)
    if isinstance(value, str):
        return re.findall(r"-?\d+", value)
    return [value]


def _as_evidence_items(
    expected_evidence: list[dict[str, Any]] | dict[str, Any] | None,
) -> list[Any]:
    if expected_evidence is None:
        return []
    if isinstance(expected_evidence, list | tuple):
        return list(expected_evidence)
    return [expected_evidence]


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return {"source_title": value}
    return {"value": value}


def _collect_string_values(item: dict[str, Any], keys: list[str]) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, list | tuple | set):
            values.extend(str(item_value) for item_value in value if item_value is not None)
        else:
            values.append(str(value))
    return values


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _dedupe_page_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, str]] = set()
    deduped: list[dict[str, Any]] = []
    for candidate in candidates:
        key = (candidate["page_start"], candidate["page_end"], candidate["basis"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _normalize_doc_key(value: str) -> str:
    text = str(value).strip().lower()
    if not text:
        return ""
    text = text.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    if "/" in text:
        text = PurePosixPath(text).name
    return re.sub(r"[^a-z0-9]+", "", text)


def _normalize_text(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        match = re.search(r"-?\d+", value.replace(",", ""))
        return int(match.group(0)) if match else None
    return None


def _critic_payload(details: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(details, dict):
        return {}
    critic = details.get("critic")
    return critic if isinstance(critic, dict) else {}


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _expected_answerable(
    *,
    expected_confidence: str | None,
    expected_answer: str | None,
    expected_evidence: list[dict[str, Any]],
    expected_sources: list[str],
    expected_keywords: list[str],
) -> bool:
    if expected_confidence == "insufficient":
        return False
    if expected_confidence:
        return True
    return bool(expected_answer or expected_evidence or expected_sources or expected_keywords)


_NUMBER_RE = re.compile(
    r"(?P<prefix>[$€£])?\s*"
    r"(?P<paren>\()?"
    r"(?P<number>-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)"
    r"\)?\s*"
    r"(?P<percent>%)?"
    r"(?:\s*(?P<scale>thousand|million|billion|trillion))?",
    re.IGNORECASE,
)

_SCALE_FACTORS = {
    "thousand": Decimal("1000"),
    "million": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "trillion": Decimal("1000000000000"),
}


def _extract_numbers(text: str) -> list[dict[str, Any]]:
    cleaned = re.sub(r"\[c\d+\]", " ", text or "", flags=re.IGNORECASE)
    numbers: list[dict[str, Any]] = []
    for match in _NUMBER_RE.finditer(cleaned):
        raw = match.group("number")
        try:
            value = Decimal(raw.replace(",", ""))
        except InvalidOperation:
            continue
        if match.group("paren"):
            value = -abs(value)
        scale = (match.group("scale") or "").lower()
        scaled_value = value * _SCALE_FACTORS.get(scale, Decimal("1"))
        numbers.append(
            {
                "text": match.group(0).strip(),
                "value": value,
                "scaled_value": scaled_value,
                "is_percent": bool(match.group("percent")),
                "scale": scale or None,
            }
        )
    return numbers


def _numbers_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    expected_values = [expected["scaled_value"], expected["value"]]
    actual_values = [actual["scaled_value"], actual["value"]]

    if expected["is_percent"]:
        expected_values.append(expected["value"] / Decimal("100"))
    if actual["is_percent"]:
        actual_values.append(actual["value"] / Decimal("100"))

    return any(
        _decimal_close(expected_value, actual_value)
        for expected_value in expected_values
        for actual_value in actual_values
    )


def _decimal_close(left: Decimal, right: Decimal) -> bool:
    left_float = float(left)
    right_float = float(right)
    return math.isclose(left_float, right_float, rel_tol=1e-4, abs_tol=1e-6)
