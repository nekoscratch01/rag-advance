from __future__ import annotations

import re

from atlas.core.ids import new_id
from atlas.query_orchestrator.ontology import FinanceMetricOntology
from atlas.query_orchestrator.company_aliases import KNOWN_COMPANY_ALIASES
from atlas.query_orchestrator.schema import (
    Entity,
    Metric,
    Period,
    QueryPlan,
    RetrievalBudget,
    RetrievalUnit,
)


_YEAR_RE = re.compile(r"\b(?:FY)?((?:19|20)\d{2})\b", re.IGNORECASE)
_CORPORATE_SUFFIX_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,5}\s+"
    r"(?:Inc\.?|Corp\.?|Corporation|Company|Co\.?|Ltd\.?|Limited|LLC|PLC|"
    r"Holdings|Group|Technologies|Technology)\b"
)
def build_fallback_plan(
    query: str,
    *,
    ontology: FinanceMetricOntology,
    max_units: int = 6,
) -> QueryPlan:
    normalized_query = " ".join(query.split())
    entities = tuple(_extract_entities(normalized_query))
    periods = tuple(_extract_periods(normalized_query))
    metrics = tuple(_extract_metrics(normalized_query, ontology))
    units = _build_units(normalized_query, entities, periods, metrics, max_units=max_units)
    query_type = "financial_numeric_fact" if metrics and periods else "fact_lookup"

    return QueryPlan(
        plan_id=new_id("qp"),
        original_query=normalized_query,
        standalone_query=normalized_query,
        query_type=query_type,
        entities=entities,
        periods=periods,
        metrics=metrics,
        retrieval_units=tuple(units),
        budget=RetrievalBudget(max_units=max_units),
        planner="rule_based_fallback",
        validation_status="fallback",
        metadata={"fallback_reason": "deterministic_rule_based"},
    )


def _extract_entities(query: str) -> list[Entity]:
    normalized = query.lower()
    entities: list[Entity] = []
    for value, aliases in KNOWN_COMPANY_ALIASES.items():
        if any(_contains_phrase(normalized, alias) for alias in aliases):
            entities.append(Entity(value=value, aliases=aliases, source_text=value))
    for match in _CORPORATE_SUFFIX_RE.findall(query):
        value = match.strip(" .,?;:")
        if not any(item.value.lower() == value.lower() for item in entities):
            entities.append(Entity(value=value, source_text=value))
    return entities


def _extract_periods(query: str) -> list[Period]:
    periods = []
    for year in dict.fromkeys(match.group(1) for match in _YEAR_RE.finditer(query)):
        source = f"FY{year}" if f"fy{year}" in query.lower() else year
        periods.append(Period(value=source, normalized=year, source_text=source))
    return periods


def _extract_metrics(query: str, ontology: FinanceMetricOntology) -> list[Metric]:
    return [metric.to_metric(source_text=alias) for metric, alias in ontology.find_mentions(query)]


def _build_units(
    query: str,
    entities: tuple[Entity, ...],
    periods: tuple[Period, ...],
    metrics: tuple[Metric, ...],
    *,
    max_units: int,
) -> list[RetrievalUnit]:
    units = [
        RetrievalUnit(
            unit_id="u0",
            purpose="original",
            text=query,
            retrievers=("dense", "bm25"),
            top_k=10,
            weight=1.0,
        )
    ]
    anchor_terms = tuple(
        term
        for term in [
            *(period.normalized or period.value for period in periods),
            *(entity.value for entity in entities),
        ]
        if term
    )
    if metrics and len(units) < max_units:
        metric = metrics[0]
        entity_text = " ".join(entity.value for entity in entities)
        period_text = " ".join(period.normalized or period.value for period in periods)
        alias_text = " ".join(metric.aliases[:4])
        units.append(
            RetrievalUnit(
                unit_id="u1",
                purpose="metric_alias",
                text=" ".join(part for part in (entity_text, period_text, alias_text) if part),
                retrievers=("bm25", "metric_alias"),
                must_have_terms=anchor_terms,
                should_terms=metric.aliases[:4],
                top_k=10,
                weight=1.5,
                lane_weights={"bm25": 1.2, "metric_alias": 1.5},
            )
        )
    if periods and len(units) < max_units:
        period_terms = tuple(period.normalized or period.value for period in periods)
        units.append(
            RetrievalUnit(
                unit_id=f"u{len(units)}",
                purpose="period_anchor",
                text=" ".join([query, *period_terms]),
                retrievers=("bm25",),
                must_have_terms=period_terms,
                top_k=10,
                weight=1.2,
                lane_weights={"bm25": 1.3},
            )
        )
    return units[:max_units]


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase in text
