from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from atlas.retrieval.evidence import Evidence


@dataclass(frozen=True)
class CriticResult:
    status: str
    confidence_override: str | None
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "confidence_override": self.confidence_override,
            "warnings": list(self.warnings),
            "reasons": list(self.reasons),
            "details": _json_safe(self.details),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class _Anchor:
    kind: str
    value: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class _NumberMention:
    text: str
    value: Decimal
    unit: str | None


_CITATION_RE = re.compile(r"\[([cC]\d+)\]")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_QUARTER_RE = re.compile(
    r"\b(?:q[1-4]|[1-4]q|first quarter|second quarter|third quarter|fourth quarter|"
    r"1st quarter|2nd quarter|3rd quarter|4th quarter)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<prefix>[$€£])?"
    r"(?P<number>\(?-?\d+(?:,\d{3})*(?:\.\d+)?\)?)"
    r"\s*"
    r"(?P<unit>%|percent|percentage points?|bps|basis points?|thousand|million|"
    r"billion|trillion|mm|bn|m)?"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CORPORATE_SUFFIX_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,5}\s+"
    r"(?:Inc\.?|Corp\.?|Corporation|Company|Co\.?|Ltd\.?|Limited|LLC|PLC|"
    r"Holdings|Group|Technologies|Technology)\b"
)
_TICKER_RE = re.compile(r"\b(?:NASDAQ|NYSE|AMEX|LON|TSX)\s*:\s*([A-Z]{1,6})\b")

_DOCUMENT_ANCHORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("10-k", ("10-k", "10k", "form 10-k", "annual report")),
    ("10-q", ("10-q", "10q", "form 10-q", "quarterly report")),
    ("earnings release", ("earnings release", "press release")),
    ("proxy statement", ("proxy statement", "def 14a")),
)
_KNOWN_COMPANIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("3m", ("3m", "mmm")),
    ("adobe", ("adobe", "adobe inc", "adbe")),
    ("alphabet", ("alphabet", "google", "goog", "googl")),
    ("amazon", ("amazon", "amazon.com", "amzn")),
    ("american express", ("american express", "amex", "axp")),
    ("apple", ("apple", "apple inc", "aapl")),
    ("berkshire hathaway", ("berkshire hathaway", "brk")),
    ("best buy", ("best buy", "bby")),
    ("boeing", ("boeing", "ba")),
    ("chevron", ("chevron", "cvx")),
    ("coca-cola", ("coca-cola", "coca cola", "ko")),
    ("costco", ("costco", "cost")),
    ("cvs", ("cvs", "cvs health")),
    ("deere", ("deere", "john deere", "de")),
    ("delta", ("delta", "delta air lines", "dal")),
    ("exxon", ("exxon", "exxon mobil", "exxonmobil", "xom")),
    ("ford", ("ford", "ford motor", "f")),
    ("general motors", ("general motors", "gm")),
    ("home depot", ("home depot", "hd")),
    ("ibm", ("ibm", "international business machines")),
    ("johnson & johnson", ("johnson & johnson", "johnson and johnson", "jnj")),
    ("jpmorgan", ("jpmorgan", "jp morgan", "jpmorgan chase", "jpm")),
    ("kroger", ("kroger", "kr")),
    ("lockheed martin", ("lockheed martin", "lmt")),
    ("mcdonald's", ("mcdonald's", "mcdonalds", "mcd")),
    ("meta", ("meta", "meta platforms", "facebook", "fb")),
    ("microsoft", ("microsoft", "microsoft corporation", "msft")),
    ("netflix", ("netflix", "nflx")),
    ("nike", ("nike", "nke")),
    ("nvidia", ("nvidia", "nvda")),
    ("oracle", ("oracle", "orcl")),
    ("pfizer", ("pfizer", "pfe")),
    ("salesforce", ("salesforce", "crm")),
    ("target", ("target", "target corporation", "tgt")),
    ("tesla", ("tesla", "tesla inc", "tsla")),
    ("verizon", ("verizon", "vz")),
    ("walmart", ("walmart", "wal-mart", "wmt")),
)
_SCALE_UNITS = {
    "thousand": "thousand",
    "m": "million",
    "mm": "million",
    "million": "million",
    "bn": "billion",
    "billion": "billion",
    "trillion": "trillion",
}
_PERCENT_UNITS = {"%", "percent", "percentage point", "percentage points"}
_BPS_UNITS = {"bp", "bps", "basis point", "basis points"}


def pre_generation_critic(query: str, evidence: list[Evidence]) -> CriticResult:
    if not evidence:
        return CriticResult(
            status="insufficient",
            confidence_override="insufficient",
            reasons=["no_evidence"],
            details={"evidence_count": 0},
        )

    anchors = _extract_query_anchors(query)
    evidence_text = _combined_evidence_text(evidence)
    missing = [anchor for anchor in anchors if not _anchor_in_text(anchor, evidence_text)]

    details: dict[str, Any] = {
        "evidence_count": len(evidence),
        "query_anchors": [_anchor_detail(anchor) for anchor in anchors],
        "missing_query_anchors": [_anchor_detail(anchor) for anchor in missing],
    }
    if missing:
        return CriticResult(
            status="warning",
            confidence_override=None,
            warnings=[_format_missing_anchor_warning(missing)],
            reasons=["query_anchors_missing_from_evidence"],
            details=details,
        )

    return CriticResult(
        status="ok",
        confidence_override=None,
        details=details,
    )


def post_generation_critic(
    query: str,
    answer: str,
    evidence: list[Evidence],
    citations: list[dict[str, Any]],
) -> CriticResult:
    evidence_by_id = {item.evidence_id.lower(): item for item in evidence}
    evidence_ids = set(evidence_by_id)
    answer_citation_ids = _extract_citation_ids(answer)
    provided_citation_ids = _extract_provided_citation_ids(citations)
    referenced_ids = _dedupe([*answer_citation_ids, *provided_citation_ids])
    invalid_citation_ids = [
        citation_id for citation_id in referenced_ids if citation_id not in evidence_ids
    ]

    warnings: list[str] = []
    reasons: list[str] = []
    details: dict[str, Any] = {
        "query_anchors": [_anchor_detail(anchor) for anchor in _extract_query_anchors(query)],
        "evidence_ids": sorted(evidence_ids),
        "answer_citation_ids": answer_citation_ids,
        "provided_citation_ids": provided_citation_ids,
        "invalid_citation_ids": invalid_citation_ids,
        "numeric_mismatch_policy": "unsupported",
        "checked_numbers": [],
        "unsupported_numbers": [],
    }

    if not answer_citation_ids:
        reasons.append("answer_has_no_citations")

    if invalid_citation_ids:
        reasons.append("citation_not_in_evidence_set")

    cited_evidence_ids = [
        citation_id for citation_id in referenced_ids if citation_id in evidence_by_id
    ]
    unsupported_numbers = _unsupported_answer_numbers(answer, cited_evidence_ids, evidence_by_id)
    details["checked_numbers"] = [mention.text for mention in _extract_numbers(answer)]
    details["unsupported_numbers"] = [mention.text for mention in unsupported_numbers]

    if unsupported_numbers:
        warnings.append(_format_unsupported_number_warning(unsupported_numbers))
        reasons.append("answer_numbers_missing_from_cited_evidence")

    if (
        "answer_has_no_citations" in reasons
        or "citation_not_in_evidence_set" in reasons
        or "answer_numbers_missing_from_cited_evidence" in reasons
    ):
        return CriticResult(
            status="unsupported",
            confidence_override="unsupported",
            warnings=warnings,
            reasons=reasons,
            details=details,
        )

    if warnings:
        return CriticResult(
            status="warning",
            confidence_override=None,
            warnings=warnings,
            reasons=reasons,
            details=details,
        )

    return CriticResult(
        status="ok",
        confidence_override=None,
        reasons=reasons,
        details=details,
    )


def _extract_query_anchors(query: str) -> list[_Anchor]:
    anchors: list[_Anchor] = []
    normalized_query = _normalize_text(query)

    for year in _dedupe(_YEAR_RE.findall(query)):
        anchors.append(_Anchor("year", year, (year,)))

    for match in _dedupe(match.group(0) for match in _QUARTER_RE.finditer(query)):
        anchors.append(_quarter_anchor(match))

    for value, aliases in _DOCUMENT_ANCHORS:
        if any(alias in normalized_query for alias in aliases):
            anchors.append(_Anchor("document", value, aliases))

    for match in _CORPORATE_SUFFIX_RE.findall(query):
        company = match.strip(" .,?;:")
        anchors.append(_Anchor("company", company, (_normalize_text(company),)))

    for ticker in _TICKER_RE.findall(query):
        anchors.append(_Anchor("company", ticker, (ticker.lower(),)))

    for company, aliases in _KNOWN_COMPANIES:
        if any(_alias_in_text(alias, normalized_query) for alias in aliases):
            anchors.append(_Anchor("company", company, aliases))

    return _dedupe_anchors(anchors)


def _quarter_anchor(text: str) -> _Anchor:
    normalized = _normalize_text(text)
    if normalized in {"q1", "1q", "first quarter", "1st quarter"}:
        return _Anchor("quarter", "q1", ("q1", "1q", "first quarter", "1st quarter"))
    if normalized in {"q2", "2q", "second quarter", "2nd quarter"}:
        return _Anchor("quarter", "q2", ("q2", "2q", "second quarter", "2nd quarter"))
    if normalized in {"q3", "3q", "third quarter", "3rd quarter"}:
        return _Anchor("quarter", "q3", ("q3", "3q", "third quarter", "3rd quarter"))
    return _Anchor("quarter", "q4", ("q4", "4q", "fourth quarter", "4th quarter"))


def _extract_citation_ids(answer: str) -> list[str]:
    return _dedupe(match.lower() for match in _CITATION_RE.findall(answer or ""))


def _extract_provided_citation_ids(citations: list[dict[str, Any]]) -> list[str]:
    citation_ids: list[str] = []
    for item in citations:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("citation_id") or item.get("evidence_id") or item.get("id")
        if raw_id is not None:
            citation_ids.append(str(raw_id).strip().lower())
    return _dedupe(citation_ids)


def _unsupported_answer_numbers(
    answer: str,
    cited_evidence_ids: list[str],
    evidence_by_id: dict[str, Evidence],
) -> list[_NumberMention]:
    answer_numbers = _extract_numbers(answer)
    if not answer_numbers or not cited_evidence_ids:
        return []

    cited_text = "\n".join(
        _evidence_text(evidence_by_id[citation_id]) for citation_id in cited_evidence_ids
    )
    evidence_numbers = _extract_numbers(cited_text)
    unsupported: list[_NumberMention] = []
    for answer_number in answer_numbers:
        if not any(
            _numbers_match(answer_number, evidence_number) for evidence_number in evidence_numbers
        ):
            unsupported.append(answer_number)
    return unsupported


def _extract_numbers(text: str) -> list[_NumberMention]:
    scrubbed = _CITATION_RE.sub(" ", text or "")
    mentions: list[_NumberMention] = []
    for match in _NUMBER_RE.finditer(scrubbed):
        raw_number = match.group("number")
        value = _decimal_value(raw_number)
        if value is None:
            continue
        raw_text = match.group(0).strip()
        unit = _normalized_number_unit(match.group("prefix"), match.group("unit"))
        mentions.append(_NumberMention(raw_text, value, unit))
    return mentions


def _numbers_match(left: _NumberMention, right: _NumberMention) -> bool:
    if left.value != right.value:
        return False
    if left.unit is None:
        return True
    return left.unit == right.unit


def _decimal_value(raw_number: str) -> Decimal | None:
    cleaned = raw_number.strip().replace(",", "")
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = f"-{cleaned[1:-1]}"
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _normalized_number_unit(prefix: str | None, raw_unit: str | None) -> str | None:
    unit = _normalize_text(raw_unit or "")
    if unit in _SCALE_UNITS:
        return _SCALE_UNITS[unit]
    if unit in _PERCENT_UNITS:
        return "percent"
    if unit in _BPS_UNITS:
        return "basis_points"
    if prefix:
        return "currency"
    return None


def _combined_evidence_text(evidence: list[Evidence]) -> str:
    return _normalize_text("\n".join(_evidence_text(item) for item in evidence))


def _evidence_text(item: Evidence) -> str:
    metadata_values = " ".join(_flatten_metadata_values(item.metadata))
    parts = [
        item.evidence_id,
        item.document_id,
        item.chunk_id,
        item.source_title,
        item.source_uri or "",
        item.section_title or "",
        item.text,
        metadata_values,
    ]
    return " ".join(part for part in parts if part)


def _flatten_metadata_values(metadata: dict[str, Any]) -> Iterable[str]:
    for value in metadata.values():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            for child in value:
                if child is not None:
                    yield str(child)
            continue
        yield str(value)


def _anchor_in_text(anchor: _Anchor, text: str) -> bool:
    return any(_alias_in_text(alias, text) for alias in anchor.aliases)


def _alias_in_text(alias: str, text: str) -> bool:
    alias = _normalize_text(alias)
    if not alias:
        return False
    if re.search(r"[a-z0-9]", alias):
        return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text) is not None
    return alias in text


def _normalize_text(text: str) -> str:
    return " ".join(str(text).casefold().replace("\u2019", "'").split())


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _dedupe_anchors(anchors: Iterable[_Anchor]) -> list[_Anchor]:
    seen: set[tuple[str, str]] = set()
    result: list[_Anchor] = []
    for anchor in anchors:
        key = (anchor.kind, anchor.value.casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append(anchor)
    return result


def _anchor_detail(anchor: _Anchor) -> dict[str, Any]:
    return {
        "kind": anchor.kind,
        "value": anchor.value,
        "aliases": list(anchor.aliases),
    }


def _format_missing_anchor_warning(anchors: list[_Anchor]) -> str:
    values = ", ".join(f"{anchor.kind}={anchor.value}" for anchor in anchors)
    return f"Query anchors missing from evidence: {values}"


def _format_unsupported_number_warning(numbers: list[_NumberMention]) -> str:
    values = ", ".join(number.text for number in numbers)
    return f"Answer numbers not found in cited evidence: {values}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(child) for child in value]
    return value
