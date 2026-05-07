from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import re
from typing import Any

from atlas.core.ids import new_id
from atlas.ingestion.chunker import approx_token_count
from atlas.query_runtime.evidence_builder import evidence_pack_to_evidence
from atlas.retrieval.contracts import SourceAnchor
from atlas.retrieval.models.candidate import Candidate
from atlas.retrieval.models.evidence import Evidence
from atlas.retrieval.models.evidence_contract import EvidenceBlock, EvidencePack
from atlas.retrieval.providers.graph.models import GRAPH_PROVIDER, GRAPH_PROVIDER_VERSION


_UNSAFE_GRAPH_TEXT_PARTS = frozenset(
    {
        "answer",
        "description",
        "descriptions",
        "narrative",
        "narratives",
        "prompt",
        "prompts",
        "summary",
        "summaries",
        "text",
        "texts",
    }
)
_UNSAFE_ANCHOR_METADATA_KEYS = frozenset({"source_title"})


@dataclass(frozen=True)
class GraphEvidenceBuildResult:
    evidence: tuple[Evidence, ...]
    evidence_pack: EvidencePack


def build_graph_evidence_from_candidates(
    candidates: Sequence[Candidate],
    *,
    max_context_tokens: int | None,
    max_blocks: int | None = None,
    query_id: str | None = None,
    plan_id: str | None = None,
) -> GraphEvidenceBuildResult:
    pack = build_graph_evidence_pack_from_candidates(
        candidates,
        max_context_tokens=max_context_tokens,
        max_blocks=max_blocks,
        query_id=query_id,
        plan_id=plan_id,
    )
    return GraphEvidenceBuildResult(
        evidence=tuple(evidence_pack_to_evidence(pack)),
        evidence_pack=pack,
    )


def build_graph_evidence_pack_from_candidates(
    candidates: Sequence[Candidate],
    *,
    max_context_tokens: int | None,
    max_blocks: int | None = None,
    query_id: str | None = None,
    plan_id: str | None = None,
) -> EvidencePack:
    ordered = tuple(candidates)
    if max_context_tokens is not None and max_context_tokens <= 0:
        dropped = tuple(
            _candidate_to_block(
                candidate,
                evidence_index=index,
                included=False,
                drop_reason="token_budget",
            )
            for index, candidate in enumerate(ordered, start=1)
        )
        return _pack(
            blocks=(),
            dropped_blocks=dropped,
            token_count=0,
            max_context_tokens=max_context_tokens,
            query_id=query_id,
            plan_id=plan_id,
            candidate_count=len(ordered),
        )

    included: list[EvidenceBlock] = []
    dropped: list[EvidenceBlock] = []
    used_tokens = 0
    for order, candidate in enumerate(ordered, start=1):
        block = _candidate_to_block(
            candidate,
            evidence_index=len(included) + 1,
            included=True,
            drop_reason=None,
        )
        if max_blocks is not None and len(included) >= max_blocks:
            dropped.append(
                _candidate_to_block(
                    candidate,
                    evidence_index=order,
                    included=False,
                    drop_reason="max_blocks",
                )
            )
            continue

        if (
            max_context_tokens is not None
            and used_tokens + max(0, block.token_count) > max_context_tokens
        ):
            dropped.append(
                _candidate_to_block(
                    candidate,
                    evidence_index=order,
                    included=False,
                    drop_reason="token_budget",
                )
            )
            continue

        included.append(block)
        used_tokens += max(0, block.token_count)

    return _pack(
        blocks=tuple(included),
        dropped_blocks=tuple(dropped),
        token_count=used_tokens,
        max_context_tokens=max_context_tokens,
        query_id=query_id,
        plan_id=plan_id,
        candidate_count=len(ordered),
    )


def sanitize_graph_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if _is_unsafe_graph_text_key(key):
                continue
            sanitized[str(key)] = sanitize_graph_metadata(item)
        return sanitized
    if isinstance(value, list | tuple | set):
        return [sanitize_graph_metadata(item) for item in value]
    return value


def graph_source_anchor_payload(anchor: SourceAnchor | Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(anchor, SourceAnchor):
        payload = asdict(anchor)
    elif isinstance(anchor, Mapping):
        payload = dict(anchor)
    else:
        payload = {
            "document_id": getattr(anchor, "document_id", None),
            "chunk_id": getattr(anchor, "chunk_id", None),
            "parent_id": getattr(anchor, "parent_id", None),
            "page_start": getattr(anchor, "page_start", None),
            "page_end": getattr(anchor, "page_end", None),
            "table_id": getattr(anchor, "table_id", None),
            "cell_ids": getattr(anchor, "cell_ids", ()),
            "graph_ids": getattr(anchor, "graph_ids", ()),
            "metadata": getattr(anchor, "metadata", {}),
        }
    sanitized = sanitize_graph_metadata(
        {
            key: value
            for key, value in payload.items()
            if key
            in {
                "document_id",
                "chunk_id",
                "parent_id",
                "page_start",
                "page_end",
                "table_id",
                "cell_ids",
                "graph_ids",
                "metadata",
            }
        }
    )
    metadata = sanitized.get("metadata")
    sanitized["metadata"] = _source_anchor_metadata_payload(metadata)
    return sanitized


def _pack(
    *,
    blocks: tuple[EvidenceBlock, ...],
    dropped_blocks: tuple[EvidenceBlock, ...],
    token_count: int,
    max_context_tokens: int | None,
    query_id: str | None,
    plan_id: str | None,
    candidate_count: int,
) -> EvidencePack:
    return EvidencePack(
        pack_id=new_id("ep"),
        query_id=query_id,
        plan_id=plan_id,
        blocks=blocks,
        dropped_blocks=dropped_blocks,
        token_count=token_count,
        max_context_tokens=max_context_tokens,
        metadata={
            "evidence_builder": "graph_grounded_chunk_pack_v1",
            "provider": GRAPH_PROVIDER,
            "provider_version": GRAPH_PROVIDER_VERSION,
            "candidate_count": candidate_count,
            "included_count": len(blocks),
            "dropped_count": len(dropped_blocks),
            "parent_expansion": False,
            "source_type": "text_chunk",
        },
    )


def _candidate_to_block(
    candidate: Candidate,
    *,
    evidence_index: int,
    included: bool,
    drop_reason: str | None,
) -> EvidenceBlock:
    text = str(getattr(candidate, "text", "") or "")
    token_count = _candidate_token_count(candidate, text)
    metadata = _graph_evidence_metadata(candidate)
    metadata["prompt_inclusion"] = {
        "included": included,
        "drop_reason": drop_reason,
        "token_count": token_count,
        "rank": _candidate_rank(candidate),
    }
    coverage = _candidate_coverage(metadata)
    metadata["coverage"] = coverage
    chunk_id = str(getattr(candidate, "chunk_id", "") or "")
    return EvidenceBlock(
        evidence_id=f"{'c' if included else 'd'}{evidence_index}",
        source_type="text_chunk",
        provider=GRAPH_PROVIDER,
        text=text,
        document_id=str(getattr(candidate, "document_id", "") or ""),
        doc_name=str(getattr(candidate, "doc_name", "") or ""),
        page_start=_int_or_none(getattr(candidate, "page_start", None)),
        page_end=_int_or_none(getattr(candidate, "page_end", None)),
        chunk_ids=(chunk_id,) if chunk_id else (),
        candidate_ids=_tuple_text((getattr(candidate, "candidate_id", None),)),
        retrieval_sources=_tuple_text(getattr(candidate, "retrieved_by", ())) or (GRAPH_PROVIDER,),
        best_dense_rank=_int_or_none(getattr(candidate, "dense_rank", None)),
        best_bm25_rank=_int_or_none(getattr(candidate, "lexical_rank", None)),
        best_rrf_score=_float_or_none(getattr(candidate, "fusion_score", None)),
        rerank_score=_float_or_none(getattr(candidate, "rerank_score", None)),
        rank=_candidate_rank(candidate),
        retrieval_score=_candidate_score(candidate, metadata),
        source_title=getattr(candidate, "source_title", None),
        source_uri=getattr(candidate, "source_uri", None),
        section_title=getattr(candidate, "section_title", None),
        token_count=token_count,
        coverage=coverage,
        metadata=metadata,
        parent_id=getattr(candidate, "parent_id", None),
        child_ids=(chunk_id,) if chunk_id else (),
        included_in_prompt=included,
        drop_reason=drop_reason,
        drop_stage="prompt_pack" if drop_reason else None,
    )


def _graph_evidence_metadata(candidate: Candidate) -> dict[str, Any]:
    source = sanitize_graph_metadata(dict(getattr(candidate, "metadata", {}) or {}))
    graph = source.get("graph") if isinstance(source.get("graph"), Mapping) else {}
    metadata = dict(source)

    graph_candidate_id = _first_present(
        source.get("graph_candidate_id"),
        graph.get("graph_candidate_id") if isinstance(graph, Mapping) else None,
        getattr(candidate, "candidate_id", None),
    )
    entity_ids = _list_text(
        source.get("entity_ids")
        or (graph.get("entity_ids") if isinstance(graph, Mapping) else ())
    )
    relationship_ids = _list_text(
        source.get("relationship_ids")
        or (graph.get("relationship_ids") if isinstance(graph, Mapping) else ())
    )
    grounded_source_chunk_ids = _list_text(
        source.get("grounded_source_chunk_ids")
        or (graph.get("grounded_source_chunk_ids") if isinstance(graph, Mapping) else ())
        or (getattr(candidate, "chunk_id", None),)
    )
    source_anchor = source.get("source_anchor")
    if not isinstance(source_anchor, Mapping):
        source_anchor = {
            "document_id": getattr(candidate, "document_id", None),
            "chunk_id": getattr(candidate, "chunk_id", None),
            "parent_id": getattr(candidate, "parent_id", None),
            "page_start": getattr(candidate, "page_start", None),
            "page_end": getattr(candidate, "page_end", None),
            "graph_ids": (),
            "metadata": {},
        }
    graph_path = source.get("graph_path")
    if graph_path is None and isinstance(graph, Mapping):
        graph_path = graph.get("graph_path")

    metadata.update(
        {
            "provider": GRAPH_PROVIDER,
            "graph_candidate_id": graph_candidate_id,
            "entity_ids": entity_ids,
            "relationship_ids": relationship_ids,
            "graph_score": _first_present(
                source.get("graph_score"),
                graph.get("graph_score") if isinstance(graph, Mapping) else None,
                getattr(candidate, "fusion_score", None),
                getattr(candidate, "lane_score", None),
            ),
            "grounding_strength": _first_present(
                source.get("grounding_strength"),
                graph.get("grounding_strength") if isinstance(graph, Mapping) else None,
            ),
            "source_anchor": graph_source_anchor_payload(source_anchor),
            "grounded_source_chunk_ids": grounded_source_chunk_ids,
            "retrieval_task_id": _first_present(
                source.get("retrieval_task_id"),
                getattr(candidate, "retrieval_task_id", None),
            ),
            "retrieval_unit_id": _first_present(
                source.get("retrieval_unit_id"),
                getattr(candidate, "retrieval_unit_id", None),
            ),
            "graph_provider": {
                "provider": GRAPH_PROVIDER,
                "provider_version": GRAPH_PROVIDER_VERSION,
            },
        }
    )
    if graph_path is not None:
        metadata["graph_path"] = sanitize_graph_metadata(graph_path)
    return sanitize_graph_metadata(metadata)


def _candidate_coverage(metadata: Mapping[str, Any]) -> dict[str, Any]:
    unit_id = metadata.get("retrieval_unit_id")
    unit_ids = [str(unit_id)] if unit_id else []
    return {
        "retrieval_unit_ids": unit_ids,
        "covered_retrieval_unit_ids": unit_ids,
    }


def _candidate_token_count(candidate: Candidate, text: str) -> int:
    value = _int_or_none(getattr(candidate, "token_count", None))
    if value is not None and value > 0:
        return value
    return approx_token_count(text) if text else 0


def _candidate_rank(candidate: Candidate) -> int | None:
    return _first_int(
        getattr(candidate, "final_rank", None),
        getattr(candidate, "rerank_rank", None),
        getattr(candidate, "fusion_rank", None),
        getattr(candidate, "lane_rank", None),
    )


def _candidate_score(candidate: Candidate, metadata: Mapping[str, Any]) -> float:
    return float(
        _first_float(
            getattr(candidate, "rerank_score", None),
            getattr(candidate, "fusion_score", None),
            getattr(candidate, "lane_score", None),
            metadata.get("graph_score"),
            0.0,
        )
        or 0.0
    )


def _is_unsafe_graph_text_key(key: Any) -> bool:
    normalized = _normalize_key(key)
    parts = set(normalized.split("_"))
    return bool(parts & _UNSAFE_GRAPH_TEXT_PARTS)


def _source_anchor_metadata_payload(value: Any) -> dict[str, Any]:
    sanitized = sanitize_graph_metadata(value)
    if not isinstance(sanitized, Mapping):
        return {}
    return _drop_source_anchor_metadata_keys(sanitized)


def _drop_source_anchor_metadata_keys(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _drop_source_anchor_metadata_keys(item)
            for key, item in value.items()
            if _normalize_key(key) not in _UNSAFE_ANCHOR_METADATA_KEYS
        }
    if isinstance(value, list | tuple | set):
        return [_drop_source_anchor_metadata_keys(item) for item in value]
    return value


def _normalize_key(key: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key))
    return text.replace("-", "_").replace(" ", "_").lower()


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _int_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _float_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tuple_text(value: Any) -> tuple[str, ...]:
    return tuple(_list_text(value))


def _list_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    try:
        values = list(value)
    except TypeError:
        values = [value]
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        if item is None:
            continue
        text = str(item)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


__all__ = [
    "GraphEvidenceBuildResult",
    "build_graph_evidence_from_candidates",
    "build_graph_evidence_pack_from_candidates",
    "graph_source_anchor_payload",
    "sanitize_graph_metadata",
]
