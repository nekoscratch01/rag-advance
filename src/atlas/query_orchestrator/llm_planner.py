from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from atlas.core.config import Settings
from atlas.core.ids import new_id
from atlas.query_orchestrator.ontology import FinanceMetricOntology
from atlas.query_orchestrator.prompts import QUERY_PLANNER_INSTRUCTIONS, build_query_planner_input
from atlas.query_orchestrator.schema import (
    Entity,
    Metric,
    Period,
    QueryPlan,
    RetrievalBudget,
    RetrievalUnit,
)


class LLMQueryPlanner:
    def __init__(self, *, settings: Settings, ontology: FinanceMetricOntology) -> None:
        self.settings = settings
        self.ontology = ontology

    def available(self) -> bool:
        return self.settings.openai_api_key is not None

    def plan(self, query: str) -> QueryPlan:
        if self.settings.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY is required for LLM query planning.")

        client = OpenAI(
            api_key=self.settings.openai_api_key.get_secret_value(),
            timeout=self.settings.llm_timeout_seconds,
        )
        raw = self._request_plan(client, query)
        return self._plan_from_raw(query, raw)

    def _request_plan(self, client: OpenAI, query: str) -> dict[str, Any]:
        request = {
            "model": self.settings.query_planner_model,
            "instructions": QUERY_PLANNER_INSTRUCTIONS,
            "input": build_query_planner_input(
                query,
                _ontology_excerpt(self.ontology),
                self.settings.query_planner_max_units,
            ),
            "max_output_tokens": 2000,
            "reasoning": {"effort": self.settings.llm_reasoning_effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "atlas_query_plan",
                    "schema": _llm_plan_schema(),
                    "strict": True,
                }
            },
            "store": False,
        }
        try:
            response = client.responses.create(**request)
        except TypeError:
            request.pop("text", None)
            response = client.responses.create(**request)

        raw_output = _extract_output_text(response)
        return _parse_json_object(raw_output)

    def _plan_from_raw(self, query: str, raw: dict[str, Any]) -> QueryPlan:
        raw_metrics = [_metric_from_raw(item, self.ontology) for item in raw.get("metrics") or ()]
        unknown_metrics = [
            item["raw_value"]
            for item in raw_metrics
            if item["metric"] is None and item["raw_value"]
        ]
        if unknown_metrics:
            raise ValueError(f"LLM planner returned unknown metrics: {', '.join(unknown_metrics)}")
        metrics = tuple(item["metric"] for item in raw_metrics if item["metric"] is not None)
        return QueryPlan(
            plan_id=new_id("qp"),
            original_query=query,
            standalone_query=_optional_text(raw.get("standalone_query")) or query,
            query_type=str(raw.get("query_type") or "fact_lookup"),
            entities=tuple(_entity_from_raw(item) for item in raw.get("entities") or ()),
            periods=tuple(_period_from_raw(item) for item in raw.get("periods") or ()),
            metrics=metrics,
            filters=_dict_value(raw.get("filters")),
            retrieval_units=tuple(
                _retrieval_unit_from_raw(item, index)
                for index, item in enumerate(raw.get("retrieval_units") or ())
            ),
            budget=RetrievalBudget(max_units=self.settings.query_planner_max_units),
            planner="llm_structured",
            planner_version=self.settings.query_planner_version,
            validation_status="unvalidated",
            metadata={"model": self.settings.query_planner_model},
        )


def _entity_from_raw(item: Any) -> Entity:
    raw = _dict_value(item)
    return Entity(
        value=str(raw.get("value") or raw.get("name") or ""),
        kind=str(raw.get("kind") or "company"),
        aliases=tuple(str(alias) for alias in raw.get("aliases") or ()),
        source_text=_optional_text(raw.get("source_text")),
    )


def _period_from_raw(item: Any) -> Period:
    raw = _dict_value(item)
    value = str(raw.get("value") or raw.get("normalized") or "")
    return Period(
        value=value,
        kind=str(raw.get("kind") or "fiscal_year"),
        normalized=_optional_text(raw.get("normalized")),
        source_text=_optional_text(raw.get("source_text") or value),
    )


def _metric_from_raw(item: Any, ontology: FinanceMetricOntology) -> dict[str, Any]:
    raw = _dict_value(item)
    value = str(
        raw.get("canonical_name")
        or raw.get("value")
        or raw.get("name")
        or raw.get("source_text")
        or ""
    )
    definition = ontology.get(value) or ontology.canonicalize(value)
    if definition is None:
        return {"metric": None, "raw_value": value or "<blank_metric>"}
    return {
        "metric": definition.to_metric(source_text=_optional_text(raw.get("source_text") or value)),
        "raw_value": value,
    }


def _retrieval_unit_from_raw(item: Any, index: int) -> RetrievalUnit:
    raw = _dict_value(item)
    retrievers = tuple(str(value) for value in raw.get("retrievers") or ("dense", "bm25"))
    return RetrievalUnit(
        unit_id=str(raw.get("unit_id") or f"u{index}"),
        purpose=str(raw.get("purpose") or "llm_generated"),
        text=str(raw.get("text") or ""),
        retrievers=retrievers,  # type: ignore[arg-type]
        filters=_dict_value(raw.get("filters")),
        must_have_terms=tuple(str(value) for value in raw.get("must_have_terms") or ()),
        should_terms=tuple(str(value) for value in raw.get("should_terms") or ()),
        top_k=int(raw.get("top_k") or 10),
        weight=float(raw.get("weight") or 1.0),
        lane_weights={
            str(key): float(value)
            for key, value in _dict_value(raw.get("lane_weights")).items()
            if _is_number(value)
        },
        metadata=_dict_value(raw.get("metadata")),
    )


def _llm_plan_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "standalone_query": {"type": ["string", "null"]},
            "query_type": {
                "type": "string",
                "enum": [
                    "fact_lookup",
                    "financial_numeric_fact",
                    "comparison",
                    "calculation",
                    "summarization",
                    "explanation",
                    "multi_hop",
                    "ambiguous",
                ],
            },
            "entities": {"type": "array", "items": _entity_schema()},
            "periods": {
                "type": "array",
                "items": _period_schema(),
            },
            "metrics": {
                "type": "array",
                "items": _metric_schema(),
            },
            "retrieval_units": {
                "type": "array",
                "items": _retrieval_unit_schema(),
            },
        },
        "required": [
            "standalone_query",
            "query_type",
            "entities",
            "periods",
            "metrics",
            "retrieval_units",
        ],
    }


def _entity_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "value": {"type": "string"},
            "kind": {"type": "string"},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "source_text": {"type": ["string", "null"]},
        },
        "required": ["value", "kind", "aliases", "source_text"],
    }


def _period_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "value": {"type": "string"},
            "kind": {"type": "string"},
            "normalized": {"type": ["string", "null"]},
            "source_text": {"type": ["string", "null"]},
        },
        "required": ["value", "kind", "normalized", "source_text"],
    }


def _metric_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "canonical_name": {"type": "string"},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "value_type": {"type": ["string", "null"]},
            "source_text": {"type": ["string", "null"]},
        },
        "required": ["canonical_name", "aliases", "value_type", "source_text"],
    }


def _retrieval_unit_schema() -> dict[str, Any]:
    string_array = {"type": "array", "items": {"type": "string"}}
    retriever_array = {
        "type": "array",
        "items": {
            "type": "string",
            "enum": ["dense", "bm25", "table", "metric_alias", "section"],
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "unit_id": {"type": "string"},
            "purpose": {"type": "string"},
            "text": {"type": "string"},
            "retrievers": retriever_array,
            "must_have_terms": string_array,
            "should_terms": string_array,
            "top_k": {"type": "integer"},
            "weight": {"type": "number"},
        },
        "required": [
            "unit_id",
            "purpose",
            "text",
            "retrievers",
            "must_have_terms",
            "should_terms",
            "top_k",
            "weight",
        ],
    }


def _ontology_excerpt(ontology: FinanceMetricOntology) -> str:
    rows = []
    for metric in list(ontology.metrics.values())[:24]:
        rows.append(f"- {metric.canonical_name}: {', '.join(metric.aliases[:8])}")
    return "\n".join(rows)


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def _parse_json_object(raw_output: str) -> dict[str, Any]:
    try:
        value = json.loads(raw_output)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM planner did not return JSON")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LLM planner JSON must be an object")
    return value


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
