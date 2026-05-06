from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from atlas.core.config import Settings, enabled_query_providers
from atlas.core.ids import new_id
from atlas.query_orchestrator.ontology import FinanceMetricOntology
from atlas.query_orchestrator.prompts import (
    build_query_planner_input,
    build_query_planner_instructions,
)
from atlas.query_orchestrator.schema import (
    Entity,
    Metric,
    Period,
    QueryPlan,
    RetrievalBudget,
    RetrievalUnit,
)
from atlas.query_orchestrator.validator import QueryPlanValidator


PLANNER_PROVIDER_NAMES = ("hybrid", "sql", "graph")


class LLMQueryPlanner:
    def __init__(self, *, settings: Settings, ontology: FinanceMetricOntology) -> None:
        self.settings = settings
        self.ontology = ontology

    def available(self) -> bool:
        return self.settings.openai_api_key is not None

    def plan(self, query: str, *, validation_feedback: str | None = None) -> QueryPlan:
        if self.settings.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY is required for LLM query planning.")

        client = OpenAI(
            api_key=self.settings.openai_api_key.get_secret_value(),
            timeout=self.settings.llm_timeout_seconds,
        )
        feedback = validation_feedback
        last_error: Exception | None = None
        attempts = _planner_attempt_count(self.settings)
        validator = QueryPlanValidator(
            self.ontology,
            enabled_providers=_enabled_provider_names(self.settings),
        )
        for _attempt in range(attempts):
            raw = self._request_plan(client, query, validation_feedback=feedback)
            try:
                candidate = self._plan_from_raw(query, raw)
            except (ValueError, ValidationError) as exc:
                last_error = exc
                feedback = _exception_feedback(exc)
                continue
            validation = validator.validate(candidate)
            if validation.ok:
                return candidate
            feedback = _validation_feedback(validation.reasons, validation.warnings)
            last_error = ValueError(feedback)

        if last_error is not None:
            raise ValueError(
                f"LLM planner failed validation after {attempts} attempt(s): {last_error}"
            ) from last_error
        raise ValueError("LLM planner failed validation without an error detail")

    def _request_plan(
        self,
        client: OpenAI,
        query: str,
        *,
        validation_feedback: str | None = None,
    ) -> dict[str, Any]:
        enabled_providers = _enabled_provider_names(self.settings)
        request = {
            "model": self.settings.query_planner_model,
            "instructions": build_query_planner_instructions(enabled_providers),
            "input": build_query_planner_input(
                query,
                _ontology_excerpt(self.ontology),
                self.settings.query_planner_max_units,
                validation_feedback=validation_feedback,
            ),
            "max_output_tokens": 2000,
            "reasoning": {"effort": self.settings.llm_reasoning_effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "atlas_query_plan",
                    "schema": _llm_plan_schema(enabled_providers),
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
        _reject_legacy_filters(raw, "query_plan")
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
            metadata_filter=_dict_value(raw.get("metadata_filter")),
            retrieval_units=tuple(
                _retrieval_unit_from_raw(item, index)
                for index, item in enumerate(raw.get("retrieval_units") or ())
            ),
            budget=RetrievalBudget(max_units=self.settings.query_planner_max_units),
            planner="llm_structured",
            planner_version=self.settings.query_planner_version,
            validation_status="unvalidated",
            metadata={
                "model": self.settings.query_planner_model,
                "enabled_providers": list(_enabled_provider_names(self.settings)),
            },
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
    _reject_legacy_filters(raw, f"retrieval_units[{index}]")
    retrievers = tuple(str(value) for value in raw.get("retrievers") or ("hybrid",))
    return RetrievalUnit(
        unit_id=str(raw.get("unit_id") or f"u{index}"),
        purpose=str(raw.get("purpose") or "llm_generated"),
        text=str(raw.get("text") or ""),
        retrievers=retrievers,  # type: ignore[arg-type]
        metadata_filter=_dict_value(raw.get("metadata_filter")),
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


def _llm_plan_schema(enabled_providers: tuple[str, ...]) -> dict[str, Any]:
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
            "metadata_filter": {
                "type": "object",
                "additionalProperties": True,
            },
            "retrieval_units": {
                "type": "array",
                "items": _retrieval_unit_schema(enabled_providers),
            },
        },
        "required": [
            "standalone_query",
            "query_type",
            "entities",
            "periods",
            "metrics",
            "metadata_filter",
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


def _retrieval_unit_schema(enabled_providers: tuple[str, ...]) -> dict[str, Any]:
    enabled_providers = _normalize_provider_names(enabled_providers)
    string_array = {"type": "array", "items": {"type": "string"}}
    retriever_array = {
        "type": "array",
        "items": {
            "type": "string",
            "enum": list(enabled_providers),
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
            "metadata_filter": {
                "type": "object",
                "additionalProperties": True,
            },
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
            "metadata_filter",
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


def _enabled_provider_names(settings: Settings) -> tuple[str, ...]:
    return _normalize_provider_names(enabled_query_providers(settings))


def _normalize_provider_names(values: tuple[str, ...]) -> tuple[str, ...]:
    providers: list[str] = []
    seen: set[str] = set()
    for value in values:
        provider = str(value).strip().lower()
        if provider not in PLANNER_PROVIDER_NAMES or provider in seen:
            continue
        providers.append(provider)
        seen.add(provider)
    return tuple(providers) or ("hybrid",)


def _planner_attempt_count(settings: Settings) -> int:
    try:
        retry_count = int(getattr(settings, "query_planner_retry_count", 2))
    except (TypeError, ValueError):
        retry_count = 2
    return 1 + max(0, min(retry_count, 2))


def _exception_feedback(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    return f"structured_output_validation_failed: {message[:800]}"


def _validation_feedback(reasons: tuple[str, ...], warnings: tuple[str, ...]) -> str:
    parts: list[str] = []
    if reasons:
        parts.append("reasons=" + "; ".join(reasons))
    if warnings:
        parts.append("warnings=" + "; ".join(warnings))
    return "query_plan_validation_failed: " + " | ".join(parts)


def _reject_legacy_filters(raw: dict[str, Any], context: str) -> None:
    if "filters" in raw:
        raise ValueError(f"{context}.filters is not supported; use metadata_filter")


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
