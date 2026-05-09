from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sqlalchemy.orm import Session

from atlas.query_orchestrator.schema import QueryPlan
from atlas.retrieval.contracts import ProviderResult
from atlas.retrieval.models.retrieval_task import RetrievalTask


@dataclass(frozen=True)
class RetrievalContext:
    db: Session
    query: str
    top_k: int
    filters: dict | None
    options: dict
    query_plan: QueryPlan
    retrieval_tasks: list[RetrievalTask]


class RetrievalProvider(ABC):
    provider_name: str

    @abstractmethod
    async def aretrieve_candidates(self, context: RetrievalContext) -> ProviderResult:
        """Return provider-local candidates for a retrieval plan.

        The provider contract is candidate-first. Synchronous providers should
        keep their DB work in ``retrieve_provider_result`` so the router can own
        sync session lifecycle in one worker thread.
        """

    def retrieve_provider_result(
        self,
        db: Session,
        *,
        query: str,
        top_k: int,
        filters: dict | None,
        options: dict,
        query_plan: QueryPlan,
        retrieval_tasks: list[RetrievalTask],
    ) -> ProviderResult:
        _ensure_no_running_loop()
        context = RetrievalContext(
            db=db,
            query=query,
            top_k=top_k,
            filters=filters,
            options=options,
            query_plan=query_plan,
            retrieval_tasks=retrieval_tasks,
        )
        return asyncio.run(self.aretrieve_candidates(context))


def _ensure_no_running_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        "retrieve_provider_result cannot run inside an active event loop; "
        "call aretrieve_candidates instead."
    )
