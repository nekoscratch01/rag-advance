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
    lanes: tuple[str, ...]
    filters: dict[str, Any] = field(default_factory=dict)
    must_have_terms: tuple[str, ...] = ()
    should_terms: tuple[str, ...] = ()
    top_k: int = 10
    weight: float = 1.0
    lane_weights: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_unit(
        cls,
        plan: QueryPlan,
        unit: RetrievalUnit,
        *,
        task_id: str | None = None,
    ) -> RetrievalTask:
        return cls(
            task_id=task_id or new_id("rt"),
            plan_id=plan.plan_id,
            unit_id=unit.unit_id,
            query_text=unit.text,
            lanes=tuple(str(retriever) for retriever in unit.retrievers),
            filters={**plan.filters, **unit.filters},
            must_have_terms=unit.must_have_terms,
            should_terms=unit.should_terms,
            top_k=unit.top_k,
            weight=unit.weight,
            lane_weights=dict(unit.lane_weights),
            metadata=dict(unit.metadata),
        )


def tasks_from_plan(plan: QueryPlan) -> list[RetrievalTask]:
    return [RetrievalTask.from_unit(plan, unit) for unit in plan.retrieval_units]


def serialize_retrieval_task(task: RetrievalTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "plan_id": task.plan_id,
        "unit_id": task.unit_id,
        "query_text": task.query_text,
        "lanes": list(task.lanes),
        "filters": dict(task.filters),
        "must_have_terms": list(task.must_have_terms),
        "should_terms": list(task.should_terms),
        "top_k": task.top_k,
        "weight": task.weight,
        "lane_weights": dict(task.lane_weights),
        "metadata": dict(task.metadata),
    }
