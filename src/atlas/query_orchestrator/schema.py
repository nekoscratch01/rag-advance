from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


QueryType = Literal[
    "fact_lookup",
    "financial_numeric_fact",
    "comparison",
    "calculation",
    "summarization",
    "explanation",
    "multi_hop",
    "ambiguous",
]

RetrieverName = Literal[
    "dense",
    "bm25",
    "table",
    "metric_alias",
    "section",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Entity(_StrictModel):
    value: str
    kind: str = "company"
    aliases: tuple[str, ...] = ()
    source_text: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Period(_StrictModel):
    value: str
    kind: str = "fiscal_year"
    normalized: str | None = None
    source_text: str | None = None


class Metric(_StrictModel):
    canonical_name: str
    aliases: tuple[str, ...] = ()
    value_type: str | None = None
    source_text: str | None = None


class RetrievalBudget(_StrictModel):
    max_units: int = Field(default=6, ge=1)
    dense_top_k: int = Field(default=50, ge=1)
    bm25_top_k: int = Field(default=50, ge=1)
    rrf_top_k: int = Field(default=40, ge=1)
    reranker_top_k: int = Field(default=30, ge=1)
    reranker_output_k: int = Field(default=10, ge=1)
    max_context_tokens: int | None = Field(default=None, ge=1)


class RetrievalUnit(_StrictModel):
    unit_id: str
    purpose: str
    text: str
    retrievers: tuple[RetrieverName, ...] = ("dense", "bm25")
    filters: dict[str, Any] = Field(default_factory=dict)
    must_have_terms: tuple[str, ...] = ()
    should_terms: tuple[str, ...] = ()
    top_k: int = Field(default=10, ge=1)
    weight: float = Field(default=1.0, gt=0.0)
    lane_weights: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        text = " ".join(value.split())
        if not text:
            raise ValueError("retrieval unit text must not be blank")
        return text

    @field_validator("retrievers")
    @classmethod
    def _retrievers_not_empty(cls, value: tuple[RetrieverName, ...]) -> tuple[RetrieverName, ...]:
        if not value:
            raise ValueError("retrieval unit requires at least one retriever")
        return value


class QueryPlan(_StrictModel):
    plan_id: str
    original_query: str
    standalone_query: str | None = None
    query_type: QueryType = "fact_lookup"
    entities: tuple[Entity, ...] = ()
    periods: tuple[Period, ...] = ()
    metrics: tuple[Metric, ...] = ()
    filters: dict[str, Any] = Field(default_factory=dict)
    retrieval_units: tuple[RetrievalUnit, ...] = Field(min_length=1)
    risk_flags: tuple[str, ...] = ()
    budget: RetrievalBudget = Field(default_factory=RetrievalBudget)
    planner: str = "unknown"
    planner_version: str = "v1_contract"
    validation_status: str = "unvalidated"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("original_query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        text = " ".join(value.split())
        if not text:
            raise ValueError("query must not be blank")
        return text

    @model_validator(mode="after")
    def _unit_count_within_budget(self) -> QueryPlan:
        if len(self.retrieval_units) > self.budget.max_units:
            raise ValueError("retrieval unit count exceeds budget.max_units")
        return self

    @property
    def retrieval_texts(self) -> tuple[str, ...]:
        return tuple(unit.text for unit in self.retrieval_units)


def serialize_query_plan(plan: QueryPlan) -> dict[str, Any]:
    return plan.model_dump(mode="json")
