from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from atlas.query_orchestrator.company_aliases import KNOWN_COMPANY_ALIASES
from atlas.query_orchestrator.ontology import FinanceMetricOntology
from atlas.query_orchestrator.schema import KNOWN_PROVIDERS, QueryPlan


@dataclass(frozen=True)
class PlanValidation:
    ok: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


class QueryPlanValidator:
    def __init__(
        self,
        ontology: FinanceMetricOntology,
        *,
        known_providers: tuple[str, ...] = KNOWN_PROVIDERS,
    ) -> None:
        self.ontology = ontology
        self.known_providers = tuple(str(provider).strip().lower() for provider in known_providers)

    def validate(self, plan: QueryPlan) -> PlanValidation:
        reasons: list[str] = []
        warnings: list[str] = []
        query_text = _normalize(" ".join([plan.original_query, plan.standalone_query or ""]))

        for entity in plan.entities:
            aliases = (entity.value, entity.source_text or "", *entity.aliases)
            if not any(_grounded(alias, query_text) for alias in aliases):
                reasons.append(f"ungrounded_entity:{entity.value}")

        for period in plan.periods:
            aliases = (period.value, period.normalized or "", period.source_text or "")
            if not any(_grounded(alias, query_text) for alias in aliases):
                reasons.append(f"ungrounded_period:{period.value}")

        for metric in plan.metrics:
            definition = self.ontology.get(metric.canonical_name)
            if definition is None:
                reasons.append(f"unknown_metric:{metric.canonical_name}")
                continue
            aliases = (metric.canonical_name, *definition.aliases)
            if not any(_grounded(alias, query_text) for alias in aliases):
                reasons.append(f"ungrounded_metric:{metric.canonical_name}")

        if len(plan.retrieval_units) > plan.budget.max_units:
            reasons.append("too_many_retrieval_units")

        for unit in plan.retrieval_units:
            provider = unit.provider
            if provider not in self.known_providers:
                reasons.append(f"unknown_provider:{unit.unit_id}:{provider}")
            if provider == "hybrid" and unit.purpose in {"hyde", "query2doc"}:
                reasons.append(f"hyde_not_allowed_for_hybrid_sparse:{unit.unit_id}")
            if unit.must_have_terms and not any(
                _grounded(term, query_text) for term in unit.must_have_terms
            ):
                warnings.append(f"must_have_terms_not_grounded:{unit.unit_id}")
            allowed_terms = _allowed_unit_terms(plan, query_text, self.ontology)
            for bare_company in _known_company_terms(unit.text):
                if not _allowed_by_terms(bare_company, allowed_terms):
                    reasons.append(f"ungrounded_unit_entity:{unit.unit_id}:{bare_company}")
            for entity_like in _company_like_terms(unit.text):
                if not _allowed_by_terms(entity_like, allowed_terms):
                    reasons.append(f"ungrounded_unit_entity:{unit.unit_id}:{entity_like}")
            for year in _year_terms(unit.text):
                if not _allowed_by_terms(year, allowed_terms):
                    reasons.append(f"ungrounded_unit_period:{unit.unit_id}:{year}")
            for term in _metric_alias_terms(unit.text, self.ontology):
                if not _allowed_by_terms(term, allowed_terms):
                    reasons.append(f"ungrounded_metric_alias:{unit.unit_id}:{term}")
            for term in unit.should_terms:
                if not _allowed_by_terms(term, allowed_terms):
                    reasons.append(f"ungrounded_metric_alias:{unit.unit_id}:{term}")

        return PlanValidation(
            ok=not reasons,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
            details={"planner": plan.planner, "plan_id": plan.plan_id},
        )


def _grounded(value: str, query_text: str) -> bool:
    text = _normalize(value)
    return bool(text) and _phrase_in_text(text, query_text)


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").split())


def _allowed_unit_terms(
    plan: QueryPlan,
    query_text: str,
    ontology: FinanceMetricOntology,
) -> set[str]:
    terms = {_normalize(plan.original_query), _normalize(plan.standalone_query or "")}
    terms.update(_normalize(entity.value) for entity in plan.entities)
    terms.update(_normalize(alias) for entity in plan.entities for alias in entity.aliases)
    terms.update(_normalize(period.value) for period in plan.periods)
    terms.update(_normalize(period.normalized or "") for period in plan.periods)
    for metric in plan.metrics:
        definition = ontology.get(metric.canonical_name)
        aliases = (definition.aliases if definition else metric.aliases)
        terms.add(_normalize(metric.canonical_name))
        terms.update(_normalize(alias) for alias in aliases)
    terms.update(
        _normalize(alias)
        for metric, alias in ontology.find_mentions(query_text)
        for alias in (metric.canonical_name, alias, *metric.aliases)
    )
    return {term for term in terms if term}


def _allowed_by_terms(value: str, allowed_terms: set[str]) -> bool:
    text = _normalize(value)
    if not text:
        return True
    return any(text == term or text in term or term in text for term in allowed_terms)


def _company_like_terms(text: str) -> list[str]:
    pattern = re.compile(
        r"\b[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,5}\s+"
        r"(?:Inc\.?|Corp\.?|Corporation|Company|Co\.?|Ltd\.?|Limited|LLC|PLC|"
        r"Holdings|Group|Technologies|Technology)\b"
    )
    return [match.strip(" .,?;:") for match in pattern.findall(text or "")]


def _known_company_terms(text: str) -> list[str]:
    normalized = _normalize(text)
    aliases = {
        alias
        for values in KNOWN_COMPANY_ALIASES.values()
        for alias in values
    }
    return sorted(alias for alias in aliases if _phrase_in_text(alias, normalized))


def _year_terms(text: str) -> list[str]:
    return re.findall(r"\b(?:FY)?((?:19|20)\d{2})\b", text or "", flags=re.IGNORECASE)


def _metric_alias_terms(text: str, ontology: FinanceMetricOntology) -> list[str]:
    normalized = _normalize(text)
    found: list[str] = []
    for metric in ontology.metrics.values():
        for alias in (metric.canonical_name, *metric.aliases):
            normalized_alias = _normalize(alias)
            if normalized_alias and _phrase_in_text(normalized_alias, normalized):
                found.append(alias)
    return found


def _phrase_in_text(phrase: str, text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None
