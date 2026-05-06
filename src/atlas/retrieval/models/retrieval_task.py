from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atlas.core.ids import new_id
from atlas.query_orchestrator.schema import QueryPlan, RetrievalUnit


@dataclass(frozen=True)
class RetrievalTask:
    task_id: str
    plan_id: str
    unit_id: str
    query_text: str
    provider: str = "hybrid"
    metadata_filter: dict[str, Any] = field(default_factory=dict)
    must_have_terms: tuple[str, ...] = ()
    should_terms: tuple[str, ...] = ()
    top_k: int = 10
    weight: float = 1.0
    lane_weights: dict[str, float] = field(default_factory=dict)
    provider_status: str = "ready"
    unsupported_reason: str | None = None
    internal_lanes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_unit(
        cls,
        plan: QueryPlan,
        unit: RetrievalUnit,
        *,
        task_id: str | None = None,
        executable_providers: tuple[str, ...] = ("hybrid",),
    ) -> RetrievalTask:
        metadata = dict(getattr(unit, "metadata", {}) or {})
        provider = _unit_provider(unit)
        internal_lanes = _internal_lanes(unit)
        provider_status = _provider_status(
            unit,
            metadata=metadata,
            provider=provider,
            executable_providers=executable_providers,
        )
        unsupported_reason = _unsupported_reason(
            unit,
            metadata=metadata,
            provider=provider,
            provider_status=provider_status,
            executable_providers=executable_providers,
        )
        return cls(
            task_id=task_id or new_id("rt"),
            plan_id=plan.plan_id,
            unit_id=unit.unit_id,
            provider=provider,
            query_text=unit.text,
            metadata_filter=_metadata_filter(plan, unit),
            must_have_terms=unit.must_have_terms,
            should_terms=unit.should_terms,
            top_k=unit.top_k,
            weight=unit.weight,
            lane_weights=dict(unit.lane_weights),
            provider_status=provider_status,
            unsupported_reason=unsupported_reason,
            internal_lanes=internal_lanes,
            metadata={
                **metadata,
                "legacy_retrievers": list(getattr(unit, "retrievers", ()) or ()),
            },
        )

    @property
    def lanes(self) -> tuple[str, ...]:
        return self.internal_lanes


def tasks_from_plan(
    plan: QueryPlan,
    *,
    executable_providers: tuple[str, ...] = ("hybrid",),
) -> list[RetrievalTask]:
    return [
        RetrievalTask.from_unit(
            plan,
            unit,
            executable_providers=executable_providers,
        )
        for unit in plan.retrieval_units
    ]


def serialize_retrieval_task(task: RetrievalTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "plan_id": task.plan_id,
        "unit_id": task.unit_id,
        "provider": task.provider,
        "query_text": task.query_text,
        "metadata_filter": dict(task.metadata_filter),
        "must_have_terms": list(task.must_have_terms),
        "should_terms": list(task.should_terms),
        "top_k": task.top_k,
        "weight": task.weight,
        "lane_weights": dict(task.lane_weights),
        "provider_status": task.provider_status,
        "unsupported_reason": task.unsupported_reason,
        "internal_lanes": list(task.internal_lanes),
        "metadata": dict(task.metadata),
    }


def _internal_lanes(unit: RetrievalUnit) -> tuple[str, ...]:
    if _unit_provider(unit) != "hybrid":
        return ()
    metadata = dict(getattr(unit, "metadata", {}) or {})
    explicit = (
        getattr(unit, "internal_lanes", None)
        or metadata.get("internal_lanes")
        or metadata.get("lanes")
    )
    if isinstance(explicit, str):
        values = (explicit,)
    elif isinstance(explicit, list | tuple):
        values = tuple(str(value) for value in explicit if value)
    else:
        purpose = unit.purpose.lower()
        if "table" in purpose:
            values = ("bm25", "table")
        elif "section" in purpose or _metadata_filter(None, unit).get("section_name"):
            values = ("dense", "bm25", "section")
        elif "metric" in purpose or unit.should_terms:
            values = ("dense", "bm25", "metric_alias")
        else:
            values = ("dense", "bm25")
    return tuple(lane for lane in values if lane)


def _unit_provider(unit: RetrievalUnit) -> str:
    provider = getattr(unit, "provider", None)
    if provider:
        return str(provider)
    return "hybrid"


def _metadata_filter(plan: QueryPlan | None, unit: RetrievalUnit) -> dict[str, Any]:
    metadata_filter: dict[str, Any] = {}
    plan_filter = getattr(plan, "metadata_filter", None) if plan is not None else None
    unit_filter = getattr(unit, "metadata_filter", None)
    if isinstance(plan_filter, dict):
        metadata_filter.update(plan_filter)
    if isinstance(unit_filter, dict):
        metadata_filter.update(unit_filter)
    return metadata_filter


def _provider_status(
    unit: RetrievalUnit,
    *,
    metadata: dict[str, Any],
    provider: str,
    executable_providers: tuple[str, ...],
) -> str:
    if provider not in executable_providers:
        return "skipped_non_executable"
    value = getattr(unit, "provider_status", None) or metadata.get("provider_status")
    if value:
        return str(value)
    return "ready"


def _unsupported_reason(
    unit: RetrievalUnit,
    *,
    metadata: dict[str, Any],
    provider: str,
    provider_status: str,
    executable_providers: tuple[str, ...],
) -> str | None:
    if provider not in executable_providers:
        return f"provider_not_executable_in_v1:{provider}"
    value = getattr(unit, "unsupported_reason", None) or metadata.get("unsupported_reason")
    if value:
        return str(value)
    if provider_status in {"unsupported", "skipped"}:
        return f"provider_status:{provider_status}"
    return None
