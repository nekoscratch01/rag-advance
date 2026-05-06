from __future__ import annotations

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

    def plan(self, query: str, *, use_llm: bool = True) -> QueryPlan:
        fallback_reason = "llm_unavailable"
        if use_llm and self.llm_planner.available():
            try:
                candidate = self.llm_planner.plan(query)
                validation = self.validator.validate(candidate)
                if validation.ok:
                    return candidate.model_copy(
                        update={
                            "validation_status": "validated",
                            "metadata": {
                                **candidate.metadata,
                                "known_providers": list(known_query_providers(self.settings)),
                                "executable_providers": list(
                                    executable_query_providers(self.settings)
                                ),
                                "validation": {
                                    "warnings": list(validation.warnings),
                                    "reasons": list(validation.reasons),
                                },
                            },
                        }
                    )
                fallback_reason = "llm_validation_failed"
                fallback_details = {
                    "warnings": list(validation.warnings),
                    "reasons": list(validation.reasons),
                }
            except ValueError as exc:
                fallback_reason = "llm_validation_failed"
                fallback_details = {
                    "type": exc.__class__.__name__,
                    "message": str(exc)[:500],
                }
            except Exception as exc:
                fallback_reason = "llm_exception"
                fallback_details = {
                    "type": exc.__class__.__name__,
                    "message": str(exc)[:500],
                }
        else:
            fallback_details = {}

        fallback = build_fallback_plan(
            query,
            ontology=self.ontology,
            max_units=self.settings.query_planner_max_units,
        )
        validation = self.validator.validate(fallback)
        return fallback.model_copy(
            update={
                "validation_status": "validated" if validation.ok else "fallback_unvalidated",
                "metadata": {
                    **fallback.metadata,
                    "known_providers": list(known_query_providers(self.settings)),
                    "executable_providers": list(executable_query_providers(self.settings)),
                    "fallback_reason": fallback_reason,
                    "llm_rejection": fallback_details,
                    "validation": {
                        "warnings": list(validation.warnings),
                        "reasons": list(validation.reasons),
                    },
                },
            }
        )
