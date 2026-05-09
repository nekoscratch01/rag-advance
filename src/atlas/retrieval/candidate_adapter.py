from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

from atlas.core.config import NON_EXECUTABLE_QUERY_PROVIDERS, RESERVED_INTERNAL_PROVIDER_NAMES
from atlas.retrieval.contracts import ProviderResult, source_anchor_from_candidate
from atlas.retrieval.models.candidate import Candidate, FusionPolicy
from atlas.retrieval.models.evidence import Evidence


def candidates_from_provider_result(result: ProviderResult) -> tuple[Candidate, ...]:
    if result.candidates:
        return tuple(
            _with_provider_provenance(
                candidate,
                provider=result.provider,
                local_rank=index,
                local_evidence_id=None,
            )
            for index, candidate in enumerate(result.candidates, start=1)
        )
    return tuple(
        _candidate_from_evidence(item, provider=result.provider, local_rank=index)
        for index, item in enumerate(result.evidence, start=1)
    )


def _with_provider_provenance(
    candidate: Candidate,
    *,
    provider: str,
    local_rank: int,
    local_evidence_id: str | None,
) -> Candidate:
    metadata = deepcopy(candidate.metadata or {})
    provider_local_rank = (
        candidate.final_rank
        or candidate.rerank_rank
        or candidate.fusion_rank
        or candidate.lane_rank
        or local_rank
    )
    raw_candidate_provider = (
        metadata.get("candidate_provider")
        or metadata.get("source_provider")
        or metadata.get("provider")
        or candidate.provider
        or provider
    )
    candidate_provider, reported_provider = _safe_candidate_provider_label(
        raw_candidate_provider,
        fallback=provider,
    )
    source_anchor = metadata.get("source_anchor")
    if not isinstance(source_anchor, dict):
        source_anchor = asdict(source_anchor_from_candidate(candidate))
    source_anchor = _sanitize_source_anchor(
        source_anchor,
        provider=provider,
        reported_provider=reported_provider,
    )
    metadata["source_anchor"] = source_anchor
    source_type = str(metadata.get("source_type") or candidate.source_type)
    rerankable = _bool_or_default(metadata.get("rerankable"), candidate.rerankable)
    fusion_policy = _fusion_policy(metadata.get("fusion_policy") or candidate.fusion_policy)
    metadata["source_type"] = source_type
    metadata["rerankable"] = rerankable
    metadata["fusion_policy"] = fusion_policy
    metadata["provider"] = provider
    metadata["candidate_provider"] = candidate_provider
    metadata["source_provider"] = candidate_provider
    metadata["implementation_provider"] = candidate_provider
    if reported_provider is not None:
        metadata["reported_provider"] = reported_provider
    if candidate.structured_payload:
        metadata.setdefault("structured_payload", dict(candidate.structured_payload))
    provenance = {
        "provider": provider,
        "provider_local_provider": provider,
        "candidate_provider": candidate_provider,
        "source_provider": candidate_provider,
        "implementation_provider": candidate_provider,
        **({"reported_provider": reported_provider} if reported_provider is not None else {}),
        "provider_local_rank": provider_local_rank,
        "provider_local_evidence_id": local_evidence_id,
        "chunk_id": candidate.chunk_id,
        "parent_id": candidate.parent_id or metadata.get("parent_id"),
        "source_type": source_type,
        "rerankable": rerankable,
        "fusion_policy": fusion_policy,
        "retrieval_task_id": candidate.retrieval_task_id,
        "retrieval_unit_id": candidate.retrieval_unit_id,
        "source_anchor": source_anchor,
    }
    metadata["provider_local_provider"] = provider
    metadata["provider_local_rank"] = provider_local_rank
    if local_evidence_id is not None:
        metadata["provider_local_evidence_id"] = local_evidence_id
        metadata["original_evidence_id"] = local_evidence_id
    provider_provenance = metadata.get("provider_provenance")
    if isinstance(provider_provenance, list):
        metadata["provider_provenance"] = [
            *_sanitize_provider_provenance(
                provider_provenance,
                provider=provider,
                fallback_candidate_provider=candidate_provider,
            ),
            provenance,
        ]
    else:
        metadata["provider_provenance"] = [provenance]
    return replace(
        candidate,
        provider=provider,
        source_type=source_type,
        rerankable=rerankable,
        fusion_policy=fusion_policy,
        final_rank=provider_local_rank,
        metadata=metadata,
    )


def _sanitize_provider_provenance(
    values: list[Any],
    *,
    provider: str,
    fallback_candidate_provider: str,
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        item = deepcopy(value)
        raw_provider = item.get("provider")
        raw_candidate_provider = (
            item.get("candidate_provider")
            or item.get("source_provider")
            or item.get("implementation_provider")
            or raw_provider
            or item.get("provider_local_provider")
            or fallback_candidate_provider
        )
        candidate_provider, reported_provider = _safe_candidate_provider_label(
            raw_candidate_provider,
            fallback=provider,
        )
        if reported_provider is None and _provider_key(raw_provider) in _unsafe_provider_labels():
            reported_provider = str(raw_provider)
        item["provider"] = provider
        item["provider_local_provider"] = provider
        item["candidate_provider"] = candidate_provider
        item["source_provider"] = candidate_provider
        item["implementation_provider"] = candidate_provider
        if reported_provider is not None:
            item["reported_provider"] = reported_provider
        source_anchor = item.get("source_anchor")
        if isinstance(source_anchor, dict):
            item["source_anchor"] = _sanitize_source_anchor(
                source_anchor,
                provider=provider,
                reported_provider=reported_provider,
            )
        sanitized.append(item)
    return sanitized


def _sanitize_source_anchor(
    source_anchor: dict[str, Any],
    *,
    provider: str,
    reported_provider: str | None,
) -> dict[str, Any]:
    anchor = deepcopy(source_anchor)
    anchor_metadata = anchor.get("metadata")
    if isinstance(anchor_metadata, dict):
        anchor_metadata = deepcopy(anchor_metadata)
    else:
        anchor_metadata = {}
    raw_anchor_provider = anchor_metadata.get("provider")
    if reported_provider is None and _provider_key(raw_anchor_provider) in _unsafe_provider_labels():
        reported_provider = str(raw_anchor_provider)
    anchor_metadata["provider"] = provider
    if reported_provider is not None:
        anchor_metadata["reported_provider"] = reported_provider
    anchor["metadata"] = anchor_metadata
    return anchor


def _safe_candidate_provider_label(value: Any, *, fallback: str) -> tuple[str, str | None]:
    text = str(value).strip() if value is not None else ""
    if not text:
        return fallback, None
    if _provider_key(text) in _unsafe_provider_labels():
        return fallback, text
    return text, None


def _unsafe_provider_labels() -> set[str]:
    return {
        *NON_EXECUTABLE_QUERY_PROVIDERS,
        *RESERVED_INTERNAL_PROVIDER_NAMES,
    }


def _provider_key(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def _candidate_from_evidence(item: Evidence, *, provider: str, local_rank: int) -> Candidate:
    metadata = dict(item.metadata or {})
    doc_name = str(metadata.get("doc_name") or metadata.get("document_name") or item.source_title)
    retrieved_by = item.retrieved_by or _tuple_text(metadata.get("retrieved_by") or metadata.get("sources"))
    provider_label = str(
        metadata.get("provider")
        or metadata.get("retrieval_provider")
        or metadata.get("provider_local_provider")
        or provider
    )
    candidate = Candidate(
        candidate_id=metadata.get("candidate_id"),
        chunk_id=item.chunk_id,
        document_id=item.document_id,
        doc_name=doc_name,
        source_title=item.source_title,
        company=_optional_text(metadata.get("company")),
        text=item.text,
        page_start=item.page_start,
        page_end=item.page_end,
        chunk_index=_int_or_default(metadata.get("chunk_index"), 0),
        token_count=item.token_count,
        retrieved_by=retrieved_by or (provider,),
        dense_rank=_int_or_none(metadata.get("dense_rank") or metadata.get("best_dense_rank")),
        dense_score=_float_or_none(metadata.get("dense_score") or metadata.get("best_dense_score")),
        lexical_rank=_int_or_none(
            metadata.get("lexical_rank")
            or metadata.get("bm25_rank")
            or metadata.get("best_lexical_rank")
        ),
        lexical_score=_float_or_none(
            metadata.get("lexical_score") or metadata.get("best_lexical_score")
        ),
        lexical_backend=_optional_text(metadata.get("lexical_backend")),
        fusion_rank=_int_or_none(metadata.get("fusion_rank") or metadata.get("best_fusion_rank")),
        fusion_score=_float_or_none(
            metadata.get("fusion_score")
            or metadata.get("best_fusion_score")
            or item.retrieval_score
        ),
        rerank_rank=item.rerank_rank,
        rerank_score=item.rerank_score,
        final_rank=item.rank or local_rank,
        metadata=metadata,
        source_uri=item.source_uri,
        section_title=item.section_title,
        parent_id=item.parent_id or metadata.get("parent_id"),
        provider=provider_label,
        source_type=str(metadata.get("source_type") or "text_chunk"),
        lane=_optional_text(metadata.get("lane")),
        retrieval_task_id=_optional_text(metadata.get("retrieval_task_id")),
        retrieval_unit_id=_optional_text(metadata.get("retrieval_unit_id")),
        rerankable=_bool_or_default(metadata.get("rerankable"), True),
        fusion_policy=_fusion_policy(metadata.get("fusion_policy")),
        structured_payload=_dict_value(metadata.get("structured_payload")),
    )
    return _with_provider_provenance(
        candidate,
        provider=provider,
        local_rank=item.rank or local_rank,
        local_evidence_id=item.evidence_id,
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _tuple_text(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    try:
        return tuple(str(item) for item in value if item)
    except TypeError:
        return (str(value),) if value else ()


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_or_default(value: Any, default: int) -> int:
    parsed = _int_or_none(value)
    return default if parsed is None else parsed


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_default(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled", "enable"}


def _fusion_policy(value: Any) -> FusionPolicy:
    text = str(value or "ranked").strip().lower()
    if text in {"pinned", "supporting"}:
        return text
    return "ranked"


def _dict_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}
