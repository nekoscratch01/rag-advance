import re
from typing import Any

from atlas.retrieval.evidence import Evidence


def build_citations(answer: str, evidence: list[Evidence], *, confidence: str) -> list[dict[str, Any]]:
    cited_ids = _extract_citation_ids(answer)
    evidence_by_id = {item.evidence_id: item for item in evidence}

    selected: list[Evidence] = []
    for citation_id in cited_ids:
        item = evidence_by_id.get(citation_id)
        if item is not None:
            selected.append(item)

    return [
        {
            "citation_id": item.evidence_id,
            "document_id": item.document_id,
            "chunk_id": item.chunk_id,
            "parent_id": item.parent_id,
            "child_ids": list(item.child_ids),
            "source_title": item.source_title,
            "source_uri": item.source_uri,
            "section_title": item.section_title,
            "page_start": item.page_start,
            "page_end": item.page_end,
            "retrieved_by": list(item.retrieved_by),
            "supporting_text": _supporting_text(item.text),
            "retrieval_score": item.retrieval_score,
        }
        for item in selected
    ]


def _extract_citation_ids(answer: str) -> list[str]:
    seen: set[str] = set()
    citation_ids: list[str] = []
    for match in re.findall(r"\[c(\d+)\]", answer):
        citation_id = f"c{match}"
        if citation_id not in seen:
            seen.add(citation_id)
            citation_ids.append(citation_id)
    return citation_ids


def _supporting_text(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= 320:
        return text
    return f"{text[:320]}..."
