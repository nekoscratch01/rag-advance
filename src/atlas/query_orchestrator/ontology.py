from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml

from atlas.query_orchestrator.schema import Metric


@dataclass(frozen=True)
class FinanceMetricDefinition:
    canonical_name: str
    aliases: tuple[str, ...]
    statement_hints: tuple[str, ...] = ()
    value_type: str | None = None

    def to_metric(self, *, source_text: str | None = None) -> Metric:
        return Metric(
            canonical_name=self.canonical_name,
            aliases=self.aliases,
            value_type=self.value_type,
            source_text=source_text,
        )


class FinanceMetricOntology:
    def __init__(self, metrics: dict[str, FinanceMetricDefinition]) -> None:
        self.metrics = dict(metrics)
        self._alias_to_canonical = {
            _normalize(alias): canonical
            for canonical, metric in self.metrics.items()
            for alias in (metric.canonical_name, *metric.aliases)
        }

    @classmethod
    def load(cls, path: str | Path) -> FinanceMetricOntology:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("finance metric ontology must be a mapping")

        metrics: dict[str, FinanceMetricDefinition] = {}
        for key, value in raw.items():
            item = value if isinstance(value, dict) else {}
            canonical = str(item.get("canonical_name") or key)
            aliases = tuple(str(alias) for alias in item.get("aliases") or ())
            metrics[canonical] = FinanceMetricDefinition(
                canonical_name=canonical,
                aliases=aliases,
                statement_hints=tuple(str(hint) for hint in item.get("statement_hints") or ()),
                value_type=str(item["value_type"]) if item.get("value_type") else None,
            )
        return cls(metrics)

    def get(self, canonical_name: str) -> FinanceMetricDefinition | None:
        return self.metrics.get(canonical_name)

    def canonicalize(self, value: str) -> FinanceMetricDefinition | None:
        canonical = self._alias_to_canonical.get(_normalize(value))
        return self.metrics.get(canonical) if canonical else None

    def find_mentions(self, query: str) -> list[tuple[FinanceMetricDefinition, str]]:
        normalized_query = _normalize(query)
        matches: list[tuple[int, FinanceMetricDefinition, str]] = []
        for metric in self.metrics.values():
            aliases = (metric.canonical_name, *metric.aliases)
            for alias in aliases:
                normalized_alias = _normalize(alias)
                if normalized_alias and _phrase_in_text(normalized_alias, normalized_query):
                    matches.append((len(normalized_alias), metric, alias))
                    break
        matches.sort(key=lambda item: (-item[0], item[1].canonical_name))
        return [(metric, alias) for _, metric, alias in matches]


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").split())


def _phrase_in_text(phrase: str, text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None
