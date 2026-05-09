from __future__ import annotations

from copy import deepcopy
import json
import re
import time
from typing import Any

from pydantic import ValidationError

from atlas.core.config import Settings, executable_query_providers, known_query_providers
from atlas.core.ids import new_id
from atlas.llm.clients import LLMClient, OpenAIClient
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
    RESERVED_INTERNAL_PROVIDER_NAMES,
)
from atlas.query_orchestrator.validator import QueryPlanValidator


class LLMQueryPlanner:
    def __init__(
        self,
        *,
        settings: Settings,
        ontology: FinanceMetricOntology,
        client: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.ontology = ontology
        self.client = client
        self.last_observability: dict[str, Any] = {}

    def available(self) -> bool:
        return self.client is not None or self.settings.openai_api_key is not None

    def plan(
        self,
        query: str,
        *,
        validation_feedback: str | None = None,
        executable_providers: tuple[str, ...] | None = None,
    ) -> QueryPlan:
        plan, _observability = self.plan_with_observability(
            query,
            validation_feedback=validation_feedback,
            executable_providers=executable_providers,
        )
        return plan

    def plan_with_observability(
        self,
        query: str,
        *,
        validation_feedback: str | None = None,
        executable_providers: tuple[str, ...] | None = None,
    ) -> tuple[QueryPlan, dict[str, Any]]:
        if self.client is None and self.settings.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY is required for LLM query planning.")

        client = self.client or OpenAIClient(self.settings)
        feedback = validation_feedback
        last_error: Exception | None = None
        attempts = _planner_attempt_count(self.settings)
        runtime_executable_providers = _runtime_executable_provider_names(
            self.settings,
            executable_providers,
        )
        planner_calls: list[dict[str, Any]] = []
        validator = QueryPlanValidator(
            self.ontology,
            known_providers=_known_provider_names(self.settings),
        )
        for attempt_index in range(1, attempts + 1):
            try:
                raw, planner_call, request_error = self._request_plan(
                    client,
                    query,
                    validation_feedback=feedback,
                    attempt_index=attempt_index,
                    executable_providers=runtime_executable_providers,
                )
            except _PlannerCallFailed as exc:
                planner_calls.append(exc.call)
                observability = _planner_observability_payload(planner_calls)
                raise _with_observability(exc.cause, observability) from exc
            planner_calls.append(planner_call)
            if request_error is not None:
                last_error = request_error
                feedback = _exception_feedback(request_error)
                continue
            try:
                candidate = self._plan_from_raw(
                    query,
                    raw,
                    executable_providers=runtime_executable_providers,
                )
            except (ValueError, ValidationError) as exc:
                last_error = exc
                _mark_planner_call_invalid(planner_call, error_message=_exception_feedback(exc))
                feedback = _exception_feedback(exc)
                continue
            validation = validator.validate(candidate)
            if validation.ok:
                _mark_planner_call_validated(planner_call, candidate)
                candidate = _with_planner_call_metadata(
                    candidate.model_copy(update={"validation_status": "validated"}),
                    planner_call,
                )
                observability = _planner_observability_payload(planner_calls)
                self.last_observability = observability
                return candidate, observability
            feedback = _validation_feedback(validation.reasons, validation.warnings)
            last_error = ValueError(feedback)
            _mark_planner_call_invalid(planner_call, error_message=feedback)

        if last_error is not None:
            observability = _planner_observability_payload(planner_calls)
            raise _with_observability(
                ValueError(
                    f"LLM planner failed validation after {attempts} attempt(s): {last_error}"
                ),
                observability,
            ) from last_error
        raise _with_observability(
            ValueError("LLM planner failed validation without an error detail"),
            _planner_observability_payload(planner_calls),
        )

    def _request_plan(
        self,
        client: LLMClient,
        query: str,
        *,
        validation_feedback: str | None = None,
        attempt_index: int,
        executable_providers: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], Exception | None]:
        known_providers = _known_provider_names(self.settings)
        runtime_executable_providers = _runtime_executable_provider_names(
            self.settings,
            executable_providers,
        )
        request = {
            "model": self.settings.query_planner_model,
            "instructions": build_query_planner_instructions(
                known_providers,
                runtime_executable_providers,
            ),
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
                    "schema": _llm_plan_schema(known_providers),
                    "strict": True,
                }
            },
            "store": False,
        }
        planner_call: dict[str, Any] = {
            "call_id": new_id("llmc"),
            "stage": "planner",
            "attempt_index": attempt_index,
            "sequence_index": attempt_index,
            "status": "started",
            "validation_status": "started",
            "model_name": self.settings.query_planner_model,
            "planner_version": self.settings.query_planner_version,
            "request": _json_safe(request),
            "response": None,
            "usage": {},
            "metadata": {
                "known_providers": list(known_providers),
                "executable_providers": list(runtime_executable_providers),
                "validation_feedback": validation_feedback,
            },
        }
        started = time.perf_counter()
        try:
            response = client.create_response(request)
        except Exception as exc:
            planner_call.update(
                {
                    "status": "failed",
                    "validation_status": "failed",
                    "error_message": _exception_feedback(exc),
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "response": None,
                }
            )
            raise _PlannerCallFailed(planner_call, exc) from exc
        planner_call["latency_ms"] = int((time.perf_counter() - started) * 1000)
        raw_output = response.output_text
        usage = _usage_payload(getattr(response, "usage", None))
        planner_call["usage"] = usage
        try:
            parsed = _parse_json_object(raw_output)
        except ValueError as exc:
            planner_call.update(
                {
                    "status": "invalid",
                    "validation_status": "parse_error",
                    "error_message": _exception_feedback(exc),
                    "raw_output": raw_output,
                    "response": {
                        "raw_output": raw_output,
                        "parsed_json": None,
                        "usage": usage,
                    },
                }
            )
            return {}, planner_call, exc
        planner_call.update(
            {
                "status": "completed",
                "validation_status": "parsed",
                "raw_output": raw_output,
                "response": {
                    "raw_output": raw_output,
                    "parsed_json": parsed,
                    "usage": usage,
                },
            }
        )
        return parsed, planner_call, None

    def _plan_from_raw(
        self,
        query: str,
        raw: dict[str, Any],
        *,
        executable_providers: tuple[str, ...] | None = None,
    ) -> QueryPlan:
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
            metadata_filter=_metadata_filter_value(raw.get("metadata_filter")),
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
                "known_providers": list(_known_provider_names(self.settings)),
                "executable_providers": list(
                    _runtime_executable_provider_names(
                        self.settings,
                        executable_providers,
                    )
                ),
            },
        )


class _PlannerCallFailed(Exception):
    def __init__(self, call: dict[str, Any], cause: Exception) -> None:
        super().__init__(str(cause))
        self.call = call
        self.cause = cause


def _with_observability(exc: Exception, observability: dict[str, Any]) -> Exception:
    setattr(exc, "planner_observability", observability)
    return exc


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
    provider = str(raw.get("provider") or "")
    retrievers = tuple(str(value) for value in raw.get("retrievers") or ())
    if len(retrievers) > 1:
        raise ValueError(
            "compound_unit_must_be_split: compound providers like [sql, hybrid] "
            "are forbidden. Split this into separate single-purpose unit_proposals."
        )
    if not provider and retrievers:
        provider = retrievers[0]
    if not provider:
        provider = "hybrid"
    return RetrievalUnit(
        unit_id=str(raw.get("unit_id") or f"u{index}"),
        purpose=str(raw.get("purpose") or "llm_generated"),
        text=str(raw.get("text") or ""),
        provider=provider,  # type: ignore[arg-type]
        metadata_filter=_metadata_filter_value(raw.get("metadata_filter")),
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


def _llm_plan_schema(known_providers: tuple[str, ...]) -> dict[str, Any]:
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
            "metadata_filter": _metadata_filter_schema(),
            "retrieval_units": {
                "type": "array",
                "items": _retrieval_unit_schema(known_providers),
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


def _retrieval_unit_schema(known_providers: tuple[str, ...]) -> dict[str, Any]:
    known_providers = _normalize_provider_names(known_providers)
    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "unit_id": {"type": "string"},
            "purpose": {"type": "string"},
            "text": {"type": "string"},
            "provider": {
                "type": "string",
                "enum": list(known_providers),
            },
            "metadata_filter": _metadata_filter_schema(),
            "must_have_terms": string_array,
            "should_terms": string_array,
            "top_k": {"type": "integer"},
            "weight": {"type": "number"},
        },
        "required": [
            "unit_id",
            "purpose",
            "text",
            "provider",
            "metadata_filter",
            "must_have_terms",
            "should_terms",
            "top_k",
            "weight",
        ],
    }


def _metadata_filter_schema() -> dict[str, Any]:
    string_or_null = {"type": ["string", "null"]}
    integer_or_null = {"type": ["integer", "null"]}
    string_array = {"type": "array", "items": {"type": "string"}}
    properties = {
        "document_ids": string_array,
        "document_type": string_or_null,
        "filing_type": string_or_null,
        "file_type": string_or_null,
        "section_name": string_or_null,
        "section_title": string_or_null,
        "company": string_or_null,
        "ticker": string_or_null,
        "year": string_or_null,
        "fiscal_year": string_or_null,
        "source_type": string_or_null,
        "language": string_or_null,
        "parent_id": string_or_null,
        "title": string_or_null,
        "source_uri": string_or_null,
        "page_start": integer_or_null,
        "page_end": integer_or_null,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _ontology_excerpt(ontology: FinanceMetricOntology) -> str:
    rows = []
    for metric in list(ontology.metrics.values())[:24]:
        rows.append(f"- {metric.canonical_name}: {', '.join(metric.aliases[:8])}")
    return "\n".join(rows)


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


def _mark_planner_call_validated(
    planner_call: dict[str, Any],
    candidate: QueryPlan,
) -> None:
    response = dict(planner_call.get("response") or {})
    response["parsed_plan_id"] = candidate.plan_id
    response["validation_status"] = "validated"
    planner_call.update(
        {
            "status": "completed",
            "validation_status": "validated",
            "error_message": None,
            "parsed_plan_id": candidate.plan_id,
            "response": response,
        }
    )


def _mark_planner_call_invalid(
    planner_call: dict[str, Any],
    *,
    error_message: str,
) -> None:
    response = dict(planner_call.get("response") or {})
    response["validation_status"] = "invalid"
    planner_call.update(
        {
            "status": "invalid",
            "validation_status": "invalid",
            "error_message": error_message,
            "response": response,
        }
    )


def _with_planner_call_metadata(
    plan: QueryPlan,
    planner_call: dict[str, Any],
) -> QueryPlan:
    metadata = {
        **plan.metadata,
        **_planner_call_pointer(planner_call),
    }
    return plan.model_copy(update={"metadata": metadata})


def _planner_call_pointer(planner_call: dict[str, Any]) -> dict[str, Any]:
    pointer: dict[str, Any] = {}
    call_id = _optional_text(planner_call.get("call_id"))
    if call_id:
        pointer["planner_llm_call_id"] = call_id
    status = _optional_text(planner_call.get("status"))
    if status:
        pointer["planner_llm_status"] = status
    validation_status = _optional_text(planner_call.get("validation_status"))
    if validation_status:
        pointer["planner_validation_status"] = validation_status
    return pointer


def _planner_observability_payload(
    planner_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    calls = [_json_safe(call) for call in planner_calls if call]
    if not calls:
        return {}
    return {
        "planner_llm_calls": calls,
        "planner_llm_call": deepcopy(calls[-1]),
    }


def _usage_payload(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    payload: dict[str, Any] = {}
    if hasattr(usage, "model_dump"):
        try:
            model_payload = usage.model_dump()
            if isinstance(model_payload, dict):
                payload.update(_json_safe(model_payload))
        except Exception:
            pass
    input_tokens = _first_attr(usage, "input_tokens", "prompt_tokens")
    output_tokens = _first_attr(usage, "output_tokens", "completion_tokens")
    total_tokens = _first_attr(usage, "total_tokens")
    if input_tokens is not None:
        payload["input_tokens"] = input_tokens
    if output_tokens is not None:
        payload["output_tokens"] = output_tokens
    if total_tokens is not None:
        payload["total_tokens"] = total_tokens
    return payload


def _first_attr(value: Any, *names: str) -> Any:
    for name in names:
        item = getattr(value, name, None)
        if item is not None:
            return item
    return None


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _known_provider_names(settings: Settings) -> tuple[str, ...]:
    return _normalize_provider_names(known_query_providers(settings))


def _runtime_executable_provider_names(
    settings: Settings,
    executable_providers: tuple[str, ...] | None,
) -> tuple[str, ...]:
    values = (
        executable_providers
        if executable_providers is not None
        else executable_query_providers(settings)
    )
    return _provider_names(values)


def _normalize_provider_names(values: tuple[str, ...]) -> tuple[str, ...]:
    return _provider_names(values) or ("hybrid",)


def _provider_names(values: tuple[str, ...]) -> tuple[str, ...]:
    providers: list[str] = []
    seen: set[str] = set()
    for value in values:
        provider = str(value).strip().lower()
        if (
            not provider
            or provider in seen
            or provider in RESERVED_INTERNAL_PROVIDER_NAMES
        ):
            continue
        providers.append(provider)
        seen.add(provider)
    return tuple(providers)


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


def _metadata_filter_value(value: Any) -> dict[str, Any]:
    raw = _dict_value(value)
    metadata_filter: dict[str, Any] = {}
    for key, item in raw.items():
        if item is None:
            continue
        if isinstance(item, list):
            values = [str(value) for value in item if value is not None and str(value)]
            if values:
                metadata_filter[str(key)] = values
            continue
        if isinstance(item, str):
            text = item.strip()
            if text:
                metadata_filter[str(key)] = text
            continue
        metadata_filter[str(key)] = item
    return metadata_filter


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
