from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from atlas.retrieval.candidate_adapter import candidates_from_provider_result
from atlas.retrieval.contracts import ProviderResult
from atlas.retrieval.models.candidate import Candidate, FusionPolicy


DEFAULT_CROSS_PROVIDER_RRF_K = 60


class CandidateFusion:
    def __init__(self, *, rrf_k: int = DEFAULT_CROSS_PROVIDER_RRF_K) -> None:
        self.rrf_k = rrf_k

    def fuse(
        self,
        provider_results: Sequence[ProviderResult],
        *,
        limit: int,
    ) -> list[Candidate]:
        if limit <= 0:
            return []

        accumulators: dict[str, _CandidateAccumulator] = {}
        for provider_index, result in enumerate(provider_results):
            candidates = candidates_from_provider_result(result)
            for position, candidate in enumerate(candidates, start=1):
                key = _candidate_key(candidate, position)
                local_rank = _local_rank(candidate, position)
                contribution = _contribution(
                    result,
                    candidate,
                    provider_index=provider_index,
                    local_rank=local_rank,
                    rrf_k=self.rrf_k,
                )
                accumulator = accumulators.setdefault(key, _CandidateAccumulator())
                accumulator.add(candidate, contribution)

        fused = [item.to_candidate(rrf_k=self.rrf_k) for item in accumulators.values()]
        pinned = [candidate for candidate in fused if _candidate_fusion_policy(candidate) == "pinned"]
        ranked = [candidate for candidate in fused if _candidate_fusion_policy(candidate) == "ranked"]
        supporting = [
            candidate for candidate in fused if _candidate_fusion_policy(candidate) == "supporting"
        ]
        pinned.sort(key=_policy_sort_key)
        ranked.sort(key=_fusion_sort_key)
        supporting.sort(key=_policy_sort_key)
        selected = [*pinned, *ranked[:limit], *supporting]
        return [
            replace(
                candidate,
                fusion_rank=index,
                final_rank=index,
                metadata={
                    **dict(candidate.metadata or {}),
                    "rerankable": candidate.rerankable,
                    "fusion_policy": candidate.fusion_policy,
                    "cross_provider_fusion_rank": index,
                    "cross_provider_fusion_score": candidate.fusion_score,
                },
            )
            for index, candidate in enumerate(selected, start=1)
        ]


class _CandidateAccumulator:
    def __init__(self) -> None:
        self.candidates: list[Candidate] = []
        self.contributions: list[dict[str, Any]] = []

    def add(self, candidate: Candidate, contribution: dict[str, Any]) -> None:
        self.candidates.append(candidate)
        self.contributions.append(contribution)

    def to_candidate(self, *, rrf_k: int) -> Candidate:
        best_index, best = min(
            enumerate(self.candidates),
            key=lambda item: (
                _local_rank(item[1], item[0] + 1),
                -float(self.contributions[item[0]].get("score") or 0.0),
                item[1].chunk_id,
            ),
        )
        contribution_score = sum(float(item.get("score") or 0.0) for item in self.contributions)
        providers = _ordered_unique(item.get("provider") for item in self.contributions)
        provenance = []
        source_anchors = []
        for candidate, contribution in zip(self.candidates, self.contributions, strict=False):
            metadata = dict(candidate.metadata or {})
            source_anchor = metadata.get("source_anchor")
            if isinstance(source_anchor, dict):
                source_anchors.append(source_anchor)
            existing = metadata.get("provider_provenance")
            if isinstance(existing, list):
                for item in existing:
                    if not isinstance(item, dict):
                        continue
                    provenance.append(item)
                    item_source_anchor = item.get("source_anchor")
                    if isinstance(item_source_anchor, dict):
                        source_anchors.append(item_source_anchor)
            else:
                provider = contribution.get("provider") or metadata.get("provider") or candidate.provider
                candidate_provider = (
                    metadata.get("candidate_provider")
                    or metadata.get("source_provider")
                    or candidate.provider
                )
                provenance.append(
                    {
                        "provider": provider,
                        "provider_local_provider": metadata.get("provider_local_provider")
                        or provider,
                        "candidate_provider": candidate_provider,
                        "source_provider": metadata.get("source_provider") or candidate_provider,
                        "implementation_provider": metadata.get("implementation_provider")
                        or candidate_provider,
                        "provider_local_rank": metadata.get("provider_local_rank"),
                        "provider_local_evidence_id": metadata.get("provider_local_evidence_id"),
                        "chunk_id": candidate.chunk_id,
                        "parent_id": candidate.parent_id or metadata.get("parent_id"),
                        "source_anchor": source_anchor if isinstance(source_anchor, dict) else None,
                    }
                )
        metadata = _merge_metadata(item.metadata for item in self.candidates)
        fusion_policy = _merged_fusion_policy(self.candidates)
        rerankable = _merged_rerankable(self.candidates, fusion_policy=fusion_policy)
        structured_payload = _merged_structured_payload(self.candidates)
        lanes = _ordered_unique(
            lane
            for candidate in self.candidates
            for lane in _candidate_lanes(candidate)
        )
        metadata["provider_provenance"] = _dedupe_provenance(provenance)
        metadata["prompt_provider_provenance"] = metadata["provider_provenance"]
        metadata["prompt_providers"] = providers
        if source_anchors:
            metadata["source_anchors"] = _dedupe_dicts(source_anchors)
        if len(self.candidates) > 1:
            provider_local_evidence_ids = _ordered_unique(
                item.get("provider_local_evidence_id")
                for item in metadata["provider_provenance"]
                if item.get("provider_local_evidence_id")
            )
            candidate_ids = _deduped_candidate_ids(self.candidates)
            evidence_ids = provider_local_evidence_ids or candidate_ids
            metadata["prompt_deduped"] = True
            metadata["prompt_duplicate_count"] = len(self.candidates) - 1
            metadata["prompt_deduped_evidence_ids"] = evidence_ids
            metadata["prompt_deduped_candidate_ids"] = candidate_ids
            metadata["provider_local_deduped_evidence_ids"] = provider_local_evidence_ids
            metadata["cross_provider_deduped_candidate_ids"] = candidate_ids
        if lanes:
            metadata["lanes"] = lanes
            metadata["parent_lanes"] = _ordered_unique(
                [
                    *metadata.get("parent_lanes", []),
                    *lanes,
                ]
            )
        metadata["rerankable"] = rerankable
        metadata["fusion_policy"] = fusion_policy
        if structured_payload:
            metadata["structured_payload"] = structured_payload
        metadata["cross_provider_fusion"] = {
            "strategy": "weighted_rrf",
            "backend": "cross_provider_rrf",
            "version": "current",
            "rrf_k": rrf_k,
            "providers": providers,
            "contributions": list(self.contributions),
            "score": contribution_score,
            "winning_provider": self.contributions[best_index].get("provider"),
        }
        parent_id = _candidate_parent_id(best, metadata)
        if parent_id:
            metadata["parent_id"] = parent_id
            source_anchor = metadata.get("source_anchor")
            if isinstance(source_anchor, dict):
                metadata["source_anchor"] = {**source_anchor, "parent_id": parent_id}
        retrieved_by = _ordered_unique(
            source for candidate in self.candidates for source in candidate.retrieved_by
        )
        return replace(
            best,
            parent_id=parent_id,
            retrieved_by=tuple(retrieved_by),
            fusion_score=contribution_score,
            weighted_contribution=contribution_score,
            metadata=metadata,
            rerankable=rerankable,
            fusion_policy=fusion_policy,
            structured_payload=structured_payload,
        )


def _candidate_key(candidate: Candidate, position: int) -> str:
    if candidate.chunk_id:
        return f"chunk:{candidate.chunk_id}"
    parent_id = candidate.parent_id or candidate.metadata.get("parent_id")
    if parent_id:
        return f"parent:{parent_id}"
    return f"missing:{position}:{candidate.document_id}:{candidate.text[:64]}"


def _local_rank(candidate: Candidate, position: int) -> int:
    for value in (
        candidate.final_rank,
        candidate.rerank_rank,
        candidate.fusion_rank,
        candidate.lane_rank,
        candidate.dense_rank,
        candidate.lexical_rank,
    ):
        if value is not None and value > 0:
            return value
    metadata = dict(candidate.metadata or {})
    for key in ("provider_local_rank", "rank"):
        value = _int_or_none(metadata.get(key))
        if value is not None and value > 0:
            return value
    return position


def _contribution(
    result: ProviderResult,
    candidate: Candidate,
    *,
    provider_index: int,
    local_rank: int,
    rrf_k: int,
) -> dict[str, Any]:
    provider = str(result.provider)
    weight = _positive(candidate.unit_weight) * _positive(candidate.lane_weight)
    score = weight / (rrf_k + local_rank)
    metadata = dict(candidate.metadata or {})
    return {
        "provider": provider,
        "provider_index": provider_index,
        "rank": local_rank,
        "score": score,
        "weight": weight,
        "chunk_id": candidate.chunk_id,
        "parent_id": candidate.parent_id or metadata.get("parent_id"),
        "retrieval_task_id": candidate.retrieval_task_id,
        "retrieval_unit_id": candidate.retrieval_unit_id,
        "provider_local_evidence_id": metadata.get("provider_local_evidence_id"),
        "source_type": metadata.get("source_type") or candidate.source_type,
        "rerankable": _candidate_rerankable(candidate),
        "fusion_policy": _candidate_fusion_policy(candidate),
    }


def _fusion_sort_key(candidate: Candidate) -> tuple[float, int, str]:
    return (
        -(candidate.fusion_score or 0.0),
        candidate.final_rank or candidate.fusion_rank or 1_000_000_000,
        candidate.chunk_id,
    )


def _policy_sort_key(candidate: Candidate) -> tuple[int, float, str]:
    return (
        candidate.final_rank
        or candidate.rerank_rank
        or candidate.fusion_rank
        or candidate.lane_rank
        or candidate.dense_rank
        or candidate.lexical_rank
        or 1_000_000_000,
        -(candidate.fusion_score or candidate.lane_score or 0.0),
        candidate.chunk_id,
    )


def _merge_metadata(items) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for metadata in items:
        for key, value in dict(metadata or {}).items():
            if key not in merged or merged[key] in (None, "", [], {}):
                merged[key] = value
    return merged


def _candidate_parent_id(candidate: Candidate, metadata: dict[str, Any]) -> str | None:
    if candidate.parent_id:
        return candidate.parent_id
    if metadata.get("parent_id"):
        return str(metadata["parent_id"])
    candidate_metadata = dict(candidate.metadata or {})
    for item in candidate_metadata.get("provider_provenance", []):
        if isinstance(item, dict) and item.get("parent_id"):
            return str(item["parent_id"])
    for contribution in metadata.get("cross_provider_fusion", {}).get("contributions", []):
        if isinstance(contribution, dict) and contribution.get("parent_id"):
            return str(contribution["parent_id"])
    return None


def _deduped_candidate_ids(candidates: list[Candidate]) -> list[str]:
    return _ordered_unique(
        candidate.candidate_id
        or candidate.chunk_id
        or f"candidate:{index}"
        for index, candidate in enumerate(candidates, start=1)
    )


def _dedupe_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[Any] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = _key_value(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(item))
    return deduped


def _dedupe_provenance(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = (
            _key_value(item.get("provider")),
            _key_value(item.get("provider_local_provider")),
            _key_value(item.get("provider_local_evidence_id")),
            _key_value(item.get("chunk_id")),
            _key_value(item.get("parent_id")),
            _key_value(item.get("retrieval_task_id")),
            _key_value(item.get("retrieval_unit_id")),
        )
        if key in seen:
            source_anchor = item.get("source_anchor")
            if source_anchor and not by_key[key].get("source_anchor"):
                by_key[key]["source_anchor"] = source_anchor
            continue
        seen.add(key)
        copied = dict(item)
        by_key[key] = copied
        deduped.append(copied)
    return deduped


def _key_value(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return tuple(_key_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _key_value(item)) for key, item in value.items()))
    if isinstance(value, set):
        return tuple(sorted(_key_value(item) for item in value))
    return value


def _candidate_lanes(candidate: Candidate) -> list[str]:
    metadata = dict(candidate.metadata or {})
    values = []
    lanes = metadata.get("lanes")
    if isinstance(lanes, list | tuple):
        values.extend(_flatten_values(lanes))
    parent_lanes = metadata.get("parent_lanes")
    if isinstance(parent_lanes, list | tuple):
        values.extend(_flatten_values(parent_lanes))
    if metadata.get("lane"):
        values.append(metadata["lane"])
    if candidate.lane:
        values.append(candidate.lane)
    return _ordered_unique(values)


def _candidate_fusion_policy(candidate: Candidate) -> FusionPolicy:
    metadata = dict(candidate.metadata or {})
    text = str(
        metadata.get("fusion_policy")
        or getattr(candidate, "fusion_policy", None)
        or "ranked"
    ).strip().lower()
    if text in {"pinned", "supporting"}:
        return text
    return "ranked"


def _merged_fusion_policy(candidates: list[Candidate]) -> FusionPolicy:
    policies = [_candidate_fusion_policy(candidate) for candidate in candidates]
    if "pinned" in policies:
        return "pinned"
    if "ranked" in policies:
        return "ranked"
    return "supporting"


def _merged_rerankable(candidates: list[Candidate], *, fusion_policy: FusionPolicy) -> bool:
    if fusion_policy != "ranked":
        return False
    return all(_candidate_rerankable(candidate) for candidate in candidates)


def _candidate_rerankable(candidate: Candidate) -> bool:
    metadata = dict(candidate.metadata or {})
    value = metadata.get("rerankable", getattr(candidate, "rerankable", True))
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled", "enable"}


def _merged_structured_payload(candidates: list[Candidate]) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for candidate in candidates:
        payload = getattr(candidate, "structured_payload", None)
        if isinstance(payload, dict) and payload:
            payloads.append(dict(payload))
            continue
        metadata_payload = dict(candidate.metadata or {}).get("structured_payload")
        if isinstance(metadata_payload, dict) and metadata_payload:
            payloads.append(dict(metadata_payload))
    if not payloads:
        return {}
    merged: dict[str, Any] = {}
    for payload in payloads:
        for key, value in payload.items():
            if key not in merged:
                merged[key] = value
                continue
            if merged[key] == value:
                continue
            values = merged[key] if isinstance(merged[key], list) else [merged[key]]
            if not any(existing == value for existing in values):
                values.append(value)
            merged[key] = values
    return merged


def _flatten_values(values) -> list[Any]:
    flattened: list[Any] = []
    for value in values:
        if isinstance(value, list | tuple):
            flattened.extend(_flatten_values(value))
            continue
        flattened.append(value)
    return flattened


def _ordered_unique(values) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, list | tuple | set):
            for nested in _ordered_unique(value):
                if nested not in seen:
                    seen.add(nested)
                    ordered.append(nested)
            continue
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def _positive(value: float | None) -> float:
    if value is None:
        return 1.0
    parsed = float(value)
    return parsed if parsed > 0 else 0.0


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
