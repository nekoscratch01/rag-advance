from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from atlas.retrieval.providers.sql.models import SQLIntentDecision


TABLE_PURPOSE_HINTS = frozenset(
    {
        "structured",
        "table",
        "tabular",
        "sql",
        "numeric",
        "number",
        "aggregation",
        "calculation",
        "lookup",
        "ranking",
        "top",
        "filter",
        "statistics",
    }
)
NUMERIC_SIGNALS = frozenset(
    {
        "amount",
        "average",
        "avg",
        "count",
        "highest",
        "largest",
        "lowest",
        "max",
        "mean",
        "median",
        "min",
        "number",
        "rank",
        "sum",
        "total",
        "top",
        "value",
    }
)
TABLE_SIGNALS = frozenset(
    {
        "column",
        "columns",
        "csv",
        "dataset",
        "field",
        "fields",
        "filter",
        "group",
        "grouped",
        "row",
        "rows",
        "table",
        "where",
    }
)
TEXT_ONLY_SIGNALS = frozenset(
    {
        "describe",
        "discuss",
        "essay",
        "explain",
        "management",
        "narrative",
        "opinion",
        "qualitative",
        "reason",
        "risk",
        "strategy",
        "summarize",
        "summary",
        "why",
    }
)


class SQLIntentGate:
    """Allow only narrow single-table analytic questions into SQLProvider."""

    def evaluate(
        self,
        question: str,
        *,
        purpose: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SQLIntentDecision:
        text = _normalize(question)
        purpose_text = _normalize(purpose or "")
        metadata = dict(metadata or {})
        metadata_text = _normalize(" ".join(str(value) for value in metadata.values()))
        signals = _signals(text, purpose_text, metadata_text)

        force_requested = _truthy(metadata.get("force_sql")) or _truthy(
            metadata.get("sql_provider")
        )
        force_trusted = _truthy(metadata.get("force_sql_trusted")) or _truthy(
            metadata.get("_trusted_internal_sql_override")
        )
        if force_requested:
            signals.add("metadata_force")
        if force_requested and force_trusted:
            return SQLIntentDecision(
                allowed=True,
                status="success",
                reason="forced_by_task_metadata",
                intent_type="forced_table_query",
                signals=tuple(sorted(signals | {"metadata_force_trusted"})),
            )
        if force_requested:
            signals.add("metadata_force_untrusted")

        if signals & TEXT_ONLY_SIGNALS and not (signals & NUMERIC_SIGNALS):
            return SQLIntentDecision(
                allowed=False,
                status="skipped_not_table_query",
                reason="textual_or_explanatory_query",
                intent_type="text_query",
                signals=tuple(sorted(signals)),
            )

        has_table_purpose = bool(signals & TABLE_PURPOSE_HINTS)
        has_numeric = bool(signals & NUMERIC_SIGNALS) or bool(re.search(r"\b\d{2,4}\b", text))
        has_table_language = bool(signals & TABLE_SIGNALS)

        if has_table_purpose or (has_numeric and has_table_language) or _looks_like_top_k(text):
            return SQLIntentDecision(
                allowed=True,
                status="success",
                reason="supported_table_numeric_query",
                intent_type=_intent_type(signals, text),
                signals=tuple(sorted(signals)),
            )

        return SQLIntentDecision(
            allowed=False,
            status="skipped_not_table_query",
            reason="no_supported_table_numeric_intent",
            intent_type="unknown",
            signals=tuple(sorted(signals)),
        )


def _signals(*texts: str) -> set[str]:
    haystack = " ".join(texts)
    tokens = set(re.findall(r"[a-z0-9_]+", haystack))
    matched = {
        signal
        for signal in TABLE_PURPOSE_HINTS | NUMERIC_SIGNALS | TABLE_SIGNALS | TEXT_ONLY_SIGNALS
        if signal in tokens
    }
    if re.search(r"\btop\s+\d+\b", haystack):
        matched.add("top")
    if re.search(r"\b(how many|number of)\b", haystack):
        matched.add("count")
    if re.search(r"\b(group|grouped)\s+by\b", haystack):
        matched.add("group")
    return matched


def _intent_type(signals: set[str], text: str) -> str:
    if {"top", "rank"} & signals or _looks_like_top_k(text):
        return "ranking_top_k"
    if {"sum", "total", "avg", "average", "mean", "min", "max", "count"} & signals:
        return "aggregation"
    if {"group", "grouped"} & signals:
        return "grouped_statistics"
    if {"filter", "where"} & signals:
        return "filtering"
    return "numeric_table_lookup"


def _looks_like_top_k(text: str) -> bool:
    return bool(re.search(r"\b(top|highest|lowest|largest|smallest)\b", text))


def _normalize(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled", "enable"}
