from __future__ import annotations

from typing import Protocol

from sqlalchemy.orm import Session

from atlas.query_orchestrator.schema import QueryPlan
from atlas.retrieval.contracts import ProviderResult
from atlas.retrieval.models.retrieval_task import RetrievalTask


class RetrievalProvider(Protocol):
    provider_name: str

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
        ...

