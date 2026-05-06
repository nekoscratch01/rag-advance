from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


V1_COMPONENT_ORDER = (
    "query_orchestrator",
    "retrieval_plan",
    "text_hybrid_provider",
    "dense_lane",
    "bm25_lane",
    "table_lane",
    "provider_local_weighted_rrf",
    "reranker",
    "evidence_builder",
    "evidence_evaluator",
    "answer_generator",
    "citation_verifier",
    "trace_eval_cache",
)


def build_v1_record_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build component-level V1 eval metrics from a benchmark record and trace."""

    trace = _mapping(record.get("trace"))
    details = _mapping(_nested(trace, "result", "details"))
    retrieval_trace = _mapping(details.get("retrieval_trace"))
    critic = _mapping(details.get("critic"))
    candidates = _candidate_items(trace, retrieval_trace)
    evidence_pack = _evidence_pack(trace, details, candidates)
    query_plan = _query_plan(trace, details)
    retrieval_tasks = _retrieval_tasks(trace, details)
    evidence_evaluation = _verification_payload(trace, critic, "evidence_evaluation")
    citation_verification = _verification_payload(trace, critic, "citation_verification")
    lane_names = _lane_names(candidates, retrieval_tasks)
    lane_traces = _lane_trace_items(candidates)
    latency = build_latency_metrics(record, trace, details, lane_traces, candidates)
    evidence = build_evidence_metrics(evidence_pack, candidates, evidence_evaluation)
    components = {
        "query_orchestrator": {
            "present": bool(query_plan),
            "planner": _optional_str(query_plan.get("planner")),
            "query_type": _optional_str(query_plan.get("query_type")),
            "retrieval_unit_count": len(_list_mapping(query_plan.get("retrieval_units"))),
            "risk_flag_count": len(_list_value(query_plan.get("risk_flags"))),
            "latency_ms": _coerce_number(details.get("plan_latency_ms")),
        },
        "retrieval_plan": {
            "present": bool(retrieval_tasks),
            "task_count": len(retrieval_tasks),
            "unit_count": len(
                {
                    str(task.get("unit_id"))
                    for task in retrieval_tasks
                    if task.get("unit_id")
                }
            ),
            "lane_count": sum(len(_list_value(task.get("lanes"))) for task in retrieval_tasks),
        },
        "text_hybrid_provider": {
            "present": _provider_seen(candidates),
            "candidate_count": len(candidates),
            "lanes_seen": lane_names,
            "retrieval_latency_ms": latency.get("retrieval_latency_ms"),
        },
        "dense_lane": _lane_metrics("dense", candidates, lane_traces),
        "bm25_lane": _lane_metrics("bm25", candidates, lane_traces),
        "table_lane": _lane_metrics("table", candidates, lane_traces),
        "provider_local_weighted_rrf": {
            "present": _weighted_rrf_seen(candidates),
            "candidate_count": len(candidates),
            "candidate_with_fusion_score_count": sum(
                1 for item in candidates if item.get("fusion_score") is not None
            ),
            "lane_contribution_count": sum(
                len(_list_mapping(item.get("lane_contributions"))) for item in candidates
            ),
        },
        "reranker": _reranker_metrics(candidates),
        "evidence_builder": {
            "present": bool(evidence_pack) or bool(candidates),
            "selected_block_count": evidence["selected_block_count"],
            "dropped_block_count": evidence["dropped_block_count"],
            "token_count": evidence["token_count"],
            "covered_query_unit_count": evidence["covered_query_unit_count"],
        },
        "evidence_evaluator": {
            "present": bool(evidence_evaluation),
            "status": _optional_str(evidence_evaluation.get("status")),
            "reason_count": len(_list_value(evidence_evaluation.get("reasons"))),
            "warning_count": len(_list_value(evidence_evaluation.get("warnings"))),
        },
        "answer_generator": {
            "present": bool(record.get("answer")) or bool(_nested(trace, "result", "answer")),
            "confidence": _optional_str(record.get("confidence"))
            or _optional_str(_nested(trace, "result", "confidence")),
            "latency_ms": latency.get("generation_latency_ms"),
            "input_tokens": _generation_token_total(trace, "input_tokens"),
            "output_tokens": _generation_token_total(trace, "output_tokens"),
        },
        "citation_verifier": {
            "present": bool(citation_verification),
            "status": _optional_str(citation_verification.get("status")),
            "reason_count": len(_list_value(citation_verification.get("reasons"))),
            "warning_count": len(_list_value(citation_verification.get("warnings"))),
        },
        "trace_eval_cache": {
            "present": bool(trace),
            "cache_hit": _first_bool(
                record.get("cache_hit"),
                _nested(trace, "cache", "hit"),
            ),
            "v1_trace_table_count": _v1_trace_table_count(trace),
            "cache_latency_ms": latency.get("cache_latency_ms"),
        },
    }
    return {
        "components": components,
        "evidence": evidence,
        "latency": latency,
        "failure_buckets": build_v1_failure_buckets(record, components, evidence),
    }


def build_evidence_metrics(
    evidence_pack: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    evidence_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    blocks = _list_mapping(evidence_pack.get("blocks"))
    dropped_blocks = _list_mapping(evidence_pack.get("dropped_blocks"))
    selected_block_count = _coerce_int(evidence_pack.get("block_count"))
    if selected_block_count is None:
        selected_block_count = len(blocks) or len(candidates)
    dropped_block_count = _coerce_int(evidence_pack.get("dropped_block_count"))
    if dropped_block_count is None:
        dropped_block_count = len(dropped_blocks)
    coverage_items = [_mapping(item.get("coverage")) for item in candidates]
    coverage_items.extend(_mapping(item.get("coverage")) for item in blocks)
    covered_units = set()
    missing_entities = 0
    missing_periods = 0
    missing_metrics = 0
    for coverage in coverage_items:
        if not coverage:
            continue
        for key in ("retrieval_unit_ids", "query_units", "units"):
            covered_units.update(str(item) for item in _list_value(coverage.get(key)) if item)
        missing_entities += _missing_count(coverage, "entities")
        missing_periods += _missing_count(coverage, "periods")
        missing_metrics += _missing_count(coverage, "metrics")
    return {
        "selected_block_count": selected_block_count,
        "dropped_block_count": dropped_block_count,
        "prompt_included_count": sum(
            1
            for item in candidates
            if item.get("included_in_prompt") is True
            or _nested(item, "evidence_pack", "included_in_prompt") is True
        ),
        "token_count": _coerce_int(evidence_pack.get("token_count")),
        "max_context_tokens": _coerce_int(evidence_pack.get("max_context_tokens")),
        "covered_query_unit_count": len(covered_units),
        "missing_entities_count": missing_entities,
        "missing_periods_count": missing_periods,
        "missing_metrics_count": missing_metrics,
        "evaluation_status": _optional_str(evidence_evaluation.get("status")),
    }


def build_latency_metrics(
    record: Mapping[str, Any],
    trace: Mapping[str, Any],
    details: Mapping[str, Any],
    lane_traces: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lane_latency_by_name: dict[str, int] = {}
    for item in lane_traces:
        lane = _optional_str(item.get("lane"))
        latency = _coerce_int(item.get("latency_ms"))
        if lane and latency is not None:
            lane_latency_by_name[lane] = lane_latency_by_name.get(lane, 0) + latency
    reranker_latencies = [
        value
        for item in candidates
        if (value := _coerce_int(_nested(item, "reranker", "latency_ms"))) is not None
    ]
    return {
        "total_latency_ms": _first_number(
            _nested(trace, "latency", "total_latency_ms"),
            record.get("latency_ms"),
        ),
        "wall_latency_ms": _coerce_number(record.get("wall_latency_ms")),
        "plan_latency_ms": _coerce_number(details.get("plan_latency_ms")),
        "cache_latency_ms": _coerce_number(_nested(trace, "latency", "cache_latency_ms")),
        "retrieval_latency_ms": _first_number(
            _nested(trace, "latency", "retrieval_latency_ms"),
            _nested(_first_provider_payload(candidates), "retrieval_latency_ms"),
        ),
        "generation_latency_ms": _coerce_number(_nested(trace, "latency", "generation_latency_ms")),
        "lane_latency_ms": lane_latency_by_name,
        "reranker_latency_ms": max(reranker_latencies) if reranker_latencies else None,
    }


def build_v1_failure_buckets(
    record: Mapping[str, Any],
    components: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> list[str]:
    buckets: list[str] = []
    if _nested(record, "retrieval", "doc_hit_at", "10") is False:
        buckets.append("retrieval_doc_miss")
    if _nested(record, "retrieval", "page_hit_at", "10") is False:
        buckets.append("retrieval_page_miss")
    if evidence.get("evaluation_status") in {"insufficient", "contradicted"}:
        buckets.append(f"evidence_{evidence['evaluation_status']}")
    if _nested(record, "citation", "citation_doc_hit_at", "10") is False:
        buckets.append("citation_doc_miss")
    if _nested(record, "citation", "citation_page_hit_at", "10") is False:
        buckets.append("citation_page_miss")
    if _nested(record, "answer_metrics", "answer_numeric_match") is False:
        buckets.append("answer_numeric_miss")
    if record.get("error"):
        buckets.append("runtime_error")
    return _ordered_unique(buckets)


def summarize_v1_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    component_presence: dict[str, dict[str, Any]] = {}
    for component in V1_COMPONENT_ORDER:
        values = [
            _nested(record, "v1_metrics", "components", component, "present")
            for record in records
        ]
        evaluated = [value for value in values if value is not None]
        present = sum(1 for value in evaluated if value is True)
        component_presence[component] = {
            "hits": present,
            "present": present,
            "total": len(evaluated),
            "rate": present / len(evaluated) if evaluated else None,
        }

    evidence_metrics = {
        "selected_block_count": _average_nested(
            records, "v1_metrics", "evidence", "selected_block_count"
        ),
        "dropped_block_count": _average_nested(
            records, "v1_metrics", "evidence", "dropped_block_count"
        ),
        "missing_entities_count": _average_nested(
            records, "v1_metrics", "evidence", "missing_entities_count"
        ),
        "missing_periods_count": _average_nested(
            records, "v1_metrics", "evidence", "missing_periods_count"
        ),
        "missing_metrics_count": _average_nested(
            records, "v1_metrics", "evidence", "missing_metrics_count"
        ),
    }
    latency_metrics = {
        "plan_latency_ms": _average_nested(records, "v1_metrics", "latency", "plan_latency_ms"),
        "retrieval_latency_ms": _average_nested(
            records, "v1_metrics", "latency", "retrieval_latency_ms"
        ),
        "reranker_latency_ms": _average_nested(
            records, "v1_metrics", "latency", "reranker_latency_ms"
        ),
        "generation_latency_ms": _average_nested(
            records, "v1_metrics", "latency", "generation_latency_ms"
        ),
        "cache_latency_ms": _average_nested(records, "v1_metrics", "latency", "cache_latency_ms"),
    }
    failures = Counter(
        bucket
        for record in records
        for bucket in _list_value(_nested(record, "v1_metrics", "failure_buckets"))
    )
    return {
        "component_presence": component_presence,
        "evidence_metrics": evidence_metrics,
        "latency_metrics": latency_metrics,
        "failure_buckets": dict(sorted(failures.items())),
    }


def _query_plan(trace: Mapping[str, Any], details: Mapping[str, Any]) -> dict[str, Any]:
    return _first_mapping(
        details.get("query_plan"),
        _nested(trace, "metadata", "query_plan"),
        _first_v1_payload(trace, "query_plans"),
    )


def _retrieval_tasks(trace: Mapping[str, Any], details: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = _list_mapping(details.get("retrieval_tasks"))
    if tasks:
        return tasks
    tasks = _list_mapping(_nested(trace, "metadata", "retrieval_tasks"))
    if tasks:
        return tasks
    return _v1_payloads(trace, "retrieval_tasks")


def _candidate_items(
    trace: Mapping[str, Any],
    retrieval_trace: Mapping[str, Any],
) -> list[dict[str, Any]]:
    items = _list_mapping(retrieval_trace.get("top_k"))
    if items:
        return items
    items = _v1_payloads(trace, "candidates")
    if items:
        return items
    return _list_mapping(_nested(trace, "retrieval", "top_k"))


def _evidence_pack(
    trace: Mapping[str, Any],
    details: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pack = _mapping(details.get("evidence_pack"))
    if pack:
        return pack
    for item in candidates:
        pack = _mapping(item.get("evidence_pack"))
        if pack:
            return pack
    return _first_v1_payload(trace, "evidence_packs")


def _verification_payload(
    trace: Mapping[str, Any],
    critic: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    direct = _mapping(critic.get(key))
    if direct:
        return direct
    table = "evidence_evaluations" if key == "evidence_evaluation" else "citation_verifications"
    return _first_v1_payload(trace, table)


def _lane_metrics(
    lane: str,
    candidates: Sequence[Mapping[str, Any]],
    lane_traces: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_count = sum(1 for item in candidates if lane in _item_lanes(item))
    lane_trace_count = sum(1 for item in lane_traces if item.get("lane") == lane)
    latencies = [
        value
        for item in lane_traces
        if item.get("lane") == lane
        and (value := _coerce_int(item.get("latency_ms"))) is not None
    ]
    return {
        "present": candidate_count > 0 or lane_trace_count > 0,
        "candidate_count": candidate_count,
        "lane_trace_count": lane_trace_count,
        "latency_ms": sum(latencies) if latencies else None,
    }


def _reranker_metrics(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reranked = [
        _mapping(item.get("reranker"))
        for item in candidates
        if _mapping(item.get("reranker"))
    ]
    if not reranked:
        return {
            "present": False,
            "input_count": 0,
            "output_count": 0,
            "model": None,
            "latency_ms": None,
        }
    return {
        "present": True,
        "input_count": max((_coerce_int(item.get("top_n")) or 0 for item in reranked), default=0),
        "output_count": len(reranked),
        "model": _optional_str(reranked[0].get("model")),
        "latency_ms": max(
            (_coerce_int(item.get("latency_ms")) or 0 for item in reranked),
            default=None,
        ),
    }


def _lane_names(
    candidates: Sequence[Mapping[str, Any]],
    retrieval_tasks: Sequence[Mapping[str, Any]],
) -> list[str]:
    lanes: list[str] = []
    for item in candidates:
        lanes.extend(_item_lanes(item))
    for task in retrieval_tasks:
        lanes.extend(str(item) for item in _list_value(task.get("lanes")) if item)
    return _ordered_unique(lanes)


def _lane_trace_items(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in candidates:
        provider = _first_provider_payload([item])
        for trace in _list_mapping(provider.get("lanes")):
            key = (
                trace.get("provider"),
                trace.get("lane"),
                trace.get("lane_family"),
                trace.get("task_id"),
                trace.get("unit_id"),
                trace.get("query_text"),
                trace.get("requested_top_k"),
                trace.get("returned"),
                trace.get("latency_ms"),
            )
            if key in seen:
                continue
            seen.add(key)
            traces.append(trace)
    return traces


def _item_lanes(item: Mapping[str, Any]) -> list[str]:
    lanes = [str(value) for value in _list_value(item.get("lanes")) if value]
    lane = item.get("lane")
    if lane and lane != "multi_lane":
        lanes.append(str(lane))
    for payload in _list_mapping(item.get("lane_attributions")):
        lane_name = payload.get("lane")
        if lane_name:
            lanes.append(str(lane_name))
    return _ordered_unique(lanes)


def _provider_seen(candidates: Sequence[Mapping[str, Any]]) -> bool:
    for item in candidates:
        if item.get("provider") == "text_hybrid":
            return True
        if _mapping(item.get("text_hybrid_provider")):
            return True
    return bool(candidates)


def _weighted_rrf_seen(candidates: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        item.get("fusion_score") is not None
        or item.get("weighted_contribution") is not None
        or bool(_list_mapping(item.get("lane_contributions")))
        for item in candidates
    )


def _first_provider_payload(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for item in candidates:
        provider = _mapping(item.get("text_hybrid_provider"))
        if provider:
            return provider
        metadata = _mapping(item.get("metadata"))
        provider = _mapping(metadata.get("text_hybrid_provider"))
        if provider:
            return provider
    return {}


def _v1_trace_table_count(trace: Mapping[str, Any]) -> int:
    v1_trace = _mapping(trace.get("v1_trace"))
    return sum(1 for value in v1_trace.values() if _list_value(value))


def _v1_payloads(trace: Mapping[str, Any], table_name: str) -> list[dict[str, Any]]:
    rows = _list_mapping(_nested(trace, "v1_trace", table_name))
    return [_mapping(row.get("payload")) for row in rows if _mapping(row.get("payload"))]


def _first_v1_payload(trace: Mapping[str, Any], table_name: str) -> dict[str, Any]:
    payloads = _v1_payloads(trace, table_name)
    return payloads[0] if payloads else {}


def _generation_token_total(trace: Mapping[str, Any], key: str) -> int | None:
    values = [
        value
        for item in _list_mapping(trace.get("generation"))
        if (value := _coerce_int(item.get(key))) is not None
    ]
    return sum(values) if values else None


def _missing_count(coverage: Mapping[str, Any], key: str) -> int:
    payload = _mapping(coverage.get(key))
    if not payload:
        return 0
    return len(_list_value(payload.get("missing")))


def _average_nested(records: Sequence[Mapping[str, Any]], *path: str) -> dict[str, Any]:
    values = [
        float(value)
        for record in records
        if (value := _coerce_number(_nested(record, *path))) is not None
    ]
    return {
        "total": len(values),
        "average": sum(values) / len(values) if values else None,
    }


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        mapped = _mapping(value)
        if mapped:
            return mapped
    return {}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_mapping(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return list(value)
    return [value]


def _nested(value: Any, *path: str) -> Any:
    current = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def _coerce_number(value: Any) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _first_number(*values: Any) -> float | int | None:
    for value in values:
        number = _coerce_number(value)
        if number is not None:
            return number
    return None


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _first_bool(*values: Any) -> bool | None:
    for value in values:
        coerced = _bool_value(value)
        if coerced is not None:
            return coerced
    return None


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
