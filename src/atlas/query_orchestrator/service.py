from __future__ import annotations

from copy import deepcopy
import inspect
from typing import Any

from atlas.core.config import Settings, executable_query_providers, known_query_providers
from atlas.query_orchestrator.fallback import build_fallback_plan
from atlas.query_orchestrator.llm_planner import LLMQueryPlanner
from atlas.query_orchestrator.ontology import FinanceMetricOntology
from atlas.query_orchestrator.schema import QueryPlan
from atlas.query_orchestrator.validator import QueryPlanValidator


class QueryOrchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        ontology: FinanceMetricOntology | None = None,
        llm_planner: LLMQueryPlanner | None = None,
    ) -> None:
        self.settings = settings
        self.ontology = ontology or FinanceMetricOntology.load(
            settings.finance_metric_ontology_path
        )
        self.validator = QueryPlanValidator(
            self.ontology,
            known_providers=known_query_providers(settings),
        )
        self.llm_planner = llm_planner or LLMQueryPlanner(settings=settings, ontology=self.ontology)
        self.last_observability: dict[str, Any] = {}

    def plan(
        self,
        query: str,
        *,
        use_llm: bool = True,
        executable_providers: tuple[str, ...] | None = None,
    ) -> QueryPlan:
        plan, observability = self.plan_with_observability(
            query,
            use_llm=use_llm,
            executable_providers=executable_providers,
        )
        self.last_observability = observability
        return plan

    def plan_with_observability(
        self,
        query: str,
        *,
        use_llm: bool = True,
        executable_providers: tuple[str, ...] | None = None,
    ) -> tuple[QueryPlan, dict[str, Any]]:
        fallback_reason = "llm_unavailable" if use_llm else "manual_fallback_requested"
        fallback_details: dict[str, Any] = {}
        planner_observability: dict[str, Any] = {}
        if use_llm and self.llm_planner.available():
            try:
                candidate, planner_observability = _call_llm_planner_with_observability(
                    self.llm_planner,
                    query,
                    executable_providers=executable_providers,
                )
                validation = self.validator.validate(candidate)
                if validation.ok:
                    planner_observability = _mark_last_planner_call(
                        planner_observability,
                        status="completed",
                        validation_status="validated",
                    )
                    plan = candidate.model_copy(
                        update={
                            "validation_status": "validated",
                            "metadata": {
                                **candidate.metadata,
                                "known_providers": list(known_query_providers(self.settings)),
                                "executable_providers": list(
                                    _runtime_executable_providers(
                                        self.settings,
                                        executable_providers,
                                    )
                                ),
                                "validation": {
                                    "warnings": list(validation.warnings),
                                    "reasons": list(validation.reasons),
                                },
                                **_planner_pointer_metadata(planner_observability),
                            },
                        }
                    )
                    self.last_observability = planner_observability
                    return plan, planner_observability
                fallback_reason = "llm_validation_failed"
                fallback_details = {
                    "warnings": list(validation.warnings),
                    "reasons": list(validation.reasons),
                }
                planner_observability = _mark_last_planner_call(
                    planner_observability,
                    status="invalid",
                    validation_status="invalid",
                    error_message="; ".join([*validation.reasons, *validation.warnings])[:500],
                )
            except ValueError as exc:
                fallback_reason = "llm_validation_failed"
                planner_observability = (
                    _planner_observability_from_exception(exc)
                    or _planner_observability_from(self.llm_planner)
                )
                planner_observability = _mark_last_planner_call(
                    planner_observability,
                    status="invalid",
                    validation_status="invalid",
                    error_message=str(exc)[:500],
                )
                fallback_details = {
                    "type": exc.__class__.__name__,
                    "message": str(exc)[:500],
                }
            except Exception as exc:
                fallback_reason = "llm_exception"
                planner_observability = (
                    _planner_observability_from_exception(exc)
                    or _planner_observability_from(self.llm_planner)
                )
                planner_observability = _mark_last_planner_call(
                    planner_observability,
                    status="failed",
                    validation_status="failed",
                    error_message=str(exc)[:500],
                )
                fallback_details = {
                    "type": exc.__class__.__name__,
                    "message": str(exc)[:500],
                }

        fallback = build_fallback_plan(
            query,
            ontology=self.ontology,
            max_units=self.settings.query_planner_max_units,
        )
        validation = self.validator.validate(fallback)
        plan = fallback.model_copy(
            update={
                "validation_status": "validated" if validation.ok else "fallback_unvalidated",
                "metadata": {
                    **fallback.metadata,
                    "known_providers": list(known_query_providers(self.settings)),
                    "executable_providers": list(
                        _runtime_executable_providers(
                            self.settings,
                            executable_providers,
                        )
                    ),
                    "fallback_reason": fallback_reason,
                    "quality_eligible": False,
                    "not_quality_reason": "planner_fallback_not_quality_run",
                    "llm_rejection": fallback_details,
                    "validation": {
                        "warnings": list(validation.warnings),
                        "reasons": list(validation.reasons),
                    },
                    **_planner_pointer_metadata(planner_observability),
                },
            }
        )
        self.last_observability = planner_observability
        return plan, planner_observability


def _call_llm_planner_with_observability(
    llm_planner: LLMQueryPlanner,
    query: str,
    *,
    executable_providers: tuple[str, ...] | None = None,
) -> tuple[QueryPlan, dict[str, Any]]:
    plan_with_observability = getattr(llm_planner, "plan_with_observability", None)
    if callable(plan_with_observability):
        if _call_accepts_keyword(plan_with_observability, "executable_providers"):
            plan, observability = plan_with_observability(
                query,
                executable_providers=executable_providers,
            )
        else:
            plan, observability = plan_with_observability(query)
        return plan, _planner_observability_from_payload(observability)
    plan_method = getattr(llm_planner, "plan")
    if _call_accepts_keyword(plan_method, "executable_providers"):
        plan = llm_planner.plan(query, executable_providers=executable_providers)
    else:
        plan = llm_planner.plan(query)
    return plan, _planner_observability_from(llm_planner)


def _runtime_executable_providers(
    settings: Settings,
    executable_providers: tuple[str, ...] | None,
) -> tuple[str, ...]:
    return (
        executable_providers
        if executable_providers is not None
        else executable_query_providers(settings)
    )


def _call_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD or name == keyword
        for name, parameter in signature.parameters.items()
    )


def _planner_observability_from(llm_planner: object) -> dict[str, Any]:
    return _planner_observability_from_payload(
        getattr(llm_planner, "last_observability", None)
    )


def _planner_observability_from_exception(exc: Exception) -> dict[str, Any]:
    return _planner_observability_from_payload(
        getattr(exc, "planner_observability", None)
    )


def _planner_observability_from_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    calls = _list_mapping(value.get("planner_llm_calls"))
    if not calls:
        call = _mapping(value.get("planner_llm_call"))
        if call:
            calls = [call]
    if not calls:
        return {}
    return {
        "planner_llm_calls": calls,
        "planner_llm_call": deepcopy(calls[-1]),
    }


def _mark_last_planner_call(
    observability: dict[str, Any],
    *,
    status: str,
    validation_status: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    calls = _list_mapping(observability.get("planner_llm_calls"))
    if not calls:
        return {}
    last = dict(calls[-1])
    last["status"] = status
    last["validation_status"] = validation_status
    if error_message:
        last["error_message"] = error_message
    elif status == "completed":
        last["error_message"] = None
    response = _mapping(last.get("response"))
    if response:
        response["validation_status"] = validation_status
        last["response"] = response
    calls[-1] = last
    return {
        "planner_llm_calls": calls,
        "planner_llm_call": deepcopy(last),
    }


def _planner_pointer_metadata(observability: dict[str, Any]) -> dict[str, Any]:
    call = _mapping(observability.get("planner_llm_call"))
    pointer: dict[str, Any] = {}
    call_id = _optional_str(call.get("call_id"))
    if call_id:
        pointer["planner_llm_call_id"] = call_id
    status = _optional_str(call.get("status"))
    if status:
        pointer["planner_llm_status"] = status
    validation_status = _optional_str(call.get("validation_status"))
    if validation_status:
        pointer["planner_validation_status"] = validation_status
    return pointer


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list_mapping(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
