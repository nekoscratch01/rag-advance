from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from atlas.eval.metrics import answer_metric_details, dense_retrieval_metrics
from atlas.eval.service import EvalCase, load_cases


HIT_KS = (1, 3, 5, 10)
FAILURE_BUCKETS = (
    "dense_missed_bm25_hit",
    "hybrid_found_reranker_lost",
    "reranker_improved_page_rank",
    "retrieved_correct_page_answer_wrong",
    "citation_missing",
    "critic_false_reject",
    "cache_hit",
)


@dataclass(frozen=True)
class BenchmarkRun:
    run_id: str
    output_dir: Path
    summary: dict[str, Any]
    cases: list[dict[str, Any]]
    failure_buckets: dict[str, Any]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run multi-mode Atlas FinanceBench benchmark.")
    parser.add_argument("--cases", default="evals/financebench_cases.yaml")
    parser.add_argument(
        "--modes",
        default="dense_only,bm25_only,hybrid_rrf,hybrid_rrf_reranker",
        help=(
            "Comma-separated modes, for example "
            "dense_only,bm25_only,hybrid_rrf,hybrid_rrf_reranker."
        ),
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--out", default="benchmarks/financebench/runs")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--cache-policy",
        choices=("off", "bypass", "on"),
        default="off",
        help="Cache intent for the primary benchmark pass. Defaults to cache off.",
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Also run a separate warm-cache phase after a cache-enabled warmup pass.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args(argv)

    modes = parse_modes(args.modes)
    cases = load_cases(args.cases)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    run_id = args.run_id or datetime.now(UTC).strftime("financebench_%Y%m%dT%H%M%SZ")
    output_dir = Path(args.out)

    with httpx.Client(timeout=args.timeout) as client:
        run = run_benchmark(
            cases=cases,
            modes=modes,
            base_url=args.base_url,
            client=client,
            output_dir=output_dir,
            run_id=run_id,
            cases_path=args.cases,
            top_k=args.top_k,
            cache_policy=args.cache_policy,
            warm_cache=args.warm_cache,
            timeout_seconds=args.timeout,
        )

    print(f"FinanceBench benchmark run: {run.run_id}")
    print(f"Output: {run.output_dir}")
    print(f"- {run.output_dir / 'summary.json'}")
    print(f"- {run.output_dir / 'cases.jsonl'}")
    print(f"- {run.output_dir / 'failure_buckets.json'}")
    print(f"- {run.output_dir / 'report.md'}")
    return 0


def parse_modes(raw_modes: str | Sequence[str]) -> list[str]:
    if isinstance(raw_modes, str):
        values = raw_modes.split(",")
    else:
        values = list(raw_modes)
    modes = [str(mode).strip() for mode in values if str(mode).strip()]
    if not modes:
        raise ValueError("At least one benchmark mode is required.")
    return modes


def run_benchmark(
    *,
    cases: list[EvalCase],
    modes: list[str],
    base_url: str,
    client: Any,
    output_dir: str | Path,
    run_id: str,
    cases_path: str,
    top_k: int | None,
    cache_policy: str,
    warm_cache: bool,
    timeout_seconds: float,
) -> BenchmarkRun:
    started_at = datetime.now(UTC)
    records: list[dict[str, Any]] = []
    primary_phase = "cold" if cache_policy in {"off", "bypass"} else "primary"

    for mode in modes:
        records.extend(
            run_mode_phase(
                cases=cases,
                mode=mode,
                phase=primary_phase,
                cache_policy=cache_policy,
                base_url=base_url,
                client=client,
                top_k=top_k,
                record=True,
            )
        )
        if warm_cache:
            run_mode_phase(
                cases=cases,
                mode=mode,
                phase="warmup",
                cache_policy="on",
                base_url=base_url,
                client=client,
                top_k=top_k,
                record=False,
            )
            records.extend(
                run_mode_phase(
                    cases=cases,
                    mode=mode,
                    phase="warm_cache",
                    cache_policy="on",
                    base_url=base_url,
                    client=client,
                    top_k=top_k,
                    record=True,
                )
            )

    output_path = Path(output_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    failure_buckets = build_failure_buckets(records)
    summary = build_summary(
        records,
        run_id=run_id,
        cases_path=cases_path,
        base_url=base_url,
        modes=modes,
        top_k=top_k,
        cache_policy=cache_policy,
        warm_cache=warm_cache,
        timeout_seconds=timeout_seconds,
        started_at=started_at,
        finished_at=datetime.now(UTC),
    )
    write_artifacts(output_path, summary, records, failure_buckets)
    return BenchmarkRun(
        run_id=run_id,
        output_dir=output_path,
        summary=summary,
        cases=records,
        failure_buckets=failure_buckets,
    )


def run_mode_phase(
    *,
    cases: list[EvalCase],
    mode: str,
    phase: str,
    cache_policy: str,
    base_url: str,
    client: Any,
    top_k: int | None,
    record: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in cases:
        case_record = execute_case(
            case=case,
            mode=mode,
            phase=phase,
            cache_policy=cache_policy,
            base_url=base_url,
            client=client,
            top_k=top_k,
        )
        if record:
            records.append(case_record)
    return records


def execute_case(
    *,
    case: EvalCase,
    mode: str,
    phase: str,
    cache_policy: str,
    base_url: str,
    client: Any,
    top_k: int | None,
) -> dict[str, Any]:
    request_payload = build_request_payload(
        case=case,
        mode=mode,
        phase=phase,
        cache_policy=cache_policy,
        top_k=top_k,
    )
    request_params = build_request_params(mode=mode, cache_policy=cache_policy)
    started = time.perf_counter()
    latency_ms = 0
    try:
        response = client.post(
            f"{base_url.rstrip('/')}/v1/query",
            params=request_params,
            json=request_payload,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return evaluate_case_result(
            case=case,
            mode=mode,
            phase=phase,
            cache_policy=cache_policy,
            payload={},
            trace={},
            latency_ms=latency_ms,
            error=f"http_{exc.response.status_code}",
            request={"json": request_payload, "params": request_params},
        )
    except httpx.RequestError as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return evaluate_case_result(
            case=case,
            mode=mode,
            phase=phase,
            cache_policy=cache_policy,
            payload={},
            trace={},
            latency_ms=latency_ms,
            error=exc.__class__.__name__,
            request={"json": request_payload, "params": request_params},
        )

    trace = fetch_trace(client, base_url, payload.get("query_id"))
    return evaluate_case_result(
        case=case,
        mode=mode,
        phase=phase,
        cache_policy=cache_policy,
        payload=payload,
        trace=trace,
        latency_ms=latency_ms,
        error=None,
        request={"json": request_payload, "params": request_params},
    )


def build_request_payload(
    *,
    case: EvalCase,
    mode: str,
    phase: str,
    cache_policy: str,
    top_k: int | None,
) -> dict[str, Any]:
    options = mode_options(mode)
    effective_cache_policy = _effective_cache_policy(mode, phase, cache_policy)
    options.update(
        {
            "benchmark_mode": mode,
            "benchmark_phase": phase,
            "cache_policy": effective_cache_policy,
            "cache_enabled": effective_cache_policy not in {"off", "bypass"},
        }
    )
    payload: dict[str, Any] = {
        "query": case.question,
        "options": options,
    }
    if top_k is not None:
        payload["top_k"] = top_k
        options["top_k"] = top_k
    return payload


def build_request_params(*, mode: str, cache_policy: str) -> dict[str, Any]:
    options = mode_options(mode)
    effective_cache_policy = _effective_cache_policy(mode, "primary", cache_policy)
    return {
        "mode": mode,
        "retrieval_mode": options["retrieval_mode"],
        "cache_policy": effective_cache_policy,
        "cache": "on" if effective_cache_policy not in {"off", "bypass"} else "off",
    }


def mode_options(mode: str) -> dict[str, Any]:
    normalized = mode.strip().lower()
    retrieval_mode = normalized
    if normalized in {"dense_only"}:
        retrieval_mode = "dense"
    if normalized in {"bm25_only", "lexical"}:
        retrieval_mode = "bm25"
    if normalized in {"hybrid_rrf"}:
        retrieval_mode = "hybrid"
    if normalized in {
        "hybrid_rerank",
        "hybrid-rerank",
        "hybrid_reranker",
        "hybrid_rrf_reranker",
        "hybrid_rrf_reranker_cache_warm",
    }:
        retrieval_mode = "hybrid"

    reranker_enabled = normalized in {
        "hybrid_rerank",
        "hybrid-rerank",
        "hybrid_reranker",
        "hybrid_rrf_reranker",
        "hybrid_rrf_reranker_cache_warm",
    }

    options: dict[str, Any] = {
        "retrieval_mode": retrieval_mode,
        "bm25_enabled": retrieval_mode in {"bm25", "lexical", "hybrid"},
        "reranker_enabled": reranker_enabled,
    }
    if reranker_enabled:
        options["rerank"] = True
    if normalized == "hybrid_rrf_reranker_cache_warm":
        options["cache_enabled"] = True
        options["cache_policy"] = "on"
    return options


def _effective_cache_policy(mode: str, phase: str, cache_policy: str) -> str:
    normalized_mode = mode.strip().lower()
    normalized_phase = phase.strip().lower()
    if normalized_mode == "hybrid_rrf_reranker_cache_warm" or normalized_phase == "warm_cache":
        return "on"
    return cache_policy


def fetch_trace(client: Any, base_url: str, query_id: str | None) -> dict[str, Any]:
    if not query_id:
        return {}
    try:
        response = client.get(f"{base_url.rstrip('/')}/v1/query/{query_id}/trace")
    except httpx.RequestError:
        return {}
    if response.status_code != 200:
        return {}
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def evaluate_case_result(
    *,
    case: EvalCase,
    mode: str,
    phase: str,
    cache_policy: str,
    payload: Mapping[str, Any],
    trace: Mapping[str, Any],
    latency_ms: int,
    error: str | None = None,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    answer = _string_or_none(payload.get("answer"))
    confidence = _string_or_none(payload.get("confidence"))
    citations = _json_list(payload.get("citations"))
    retrieved_top_k = retrieved_top_k_from_trace(trace)
    retrieval_metrics = rank_metrics(
        retrieved_top_k,
        case.expected_evidence,
        case.expected_sources,
        prefix="",
    )
    citation_metrics = rank_metrics(
        citations_to_retrieved(citations),
        case.expected_evidence,
        case.expected_sources,
        prefix="citation_",
    )
    answer_metrics = answer_metric_details(answer, case.expected_answer)
    effective_latency_ms = _trace_latency_ms(trace) or latency_ms
    observed_retriever_type = observed_retriever_type_from_trace(trace)
    record = {
        "case_id": case.id,
        "question": case.question,
        "mode": mode,
        "phase": phase,
        "cache_policy": cache_policy,
        "query_id": payload.get("query_id"),
        "trace_id": payload.get("trace_id"),
        "confidence": confidence,
        "expected_confidence": case.expected_confidence,
        "answer": answer,
        "expected_answer": case.expected_answer,
        "citations": citations,
        "latency_ms": effective_latency_ms,
        "wall_latency_ms": latency_ms,
        "cache_hit": cache_hit_from_payload_or_trace(payload, trace),
        "observed_retriever_type": observed_retriever_type,
        "mode_applied": mode_applied(mode, observed_retriever_type),
        "retrieval": retrieval_metrics,
        "citation": citation_metrics,
        "answer_metrics": answer_metrics,
        "unsupported_answer": unsupported_answer(case, answer, confidence),
        "false_insufficient": false_insufficient(case, confidence),
        "retrieved_top_k": retrieved_top_k,
        "trace": dict(trace),
        "request": dict(request or {}),
        "metadata": case.metadata,
        "error": error,
    }
    record["failure_reasons"] = failure_reasons(record)
    return record


def rank_metrics(
    retrieved_items: list[dict[str, Any]],
    expected_evidence: list[dict[str, Any]],
    expected_sources: list[str],
    *,
    prefix: str,
) -> dict[str, Any]:
    details = dense_retrieval_metrics(retrieved_items, expected_evidence, expected_sources)
    doc_has_expectation = details["retrieval_doc_hit"] is not None
    page_has_expectation = details["retrieval_page_hit"] is not None
    doc_rank = details["first_doc_match_rank"]
    page_rank = details["first_page_match_rank"]
    return {
        f"{prefix}doc_hit_at": {
            str(k): hit_at(doc_rank, doc_has_expectation, k) for k in HIT_KS
        },
        f"{prefix}page_hit_at": {
            str(k): hit_at(page_rank, page_has_expectation, k) for k in HIT_KS
        },
        f"{prefix}mrr_doc": details["retrieval_doc_mrr"],
        f"{prefix}mrr_page": details["retrieval_page_mrr"],
        f"{prefix}first_doc_rank": doc_rank,
        f"{prefix}first_page_rank": page_rank,
        "normalized_expected_evidence": details["normalized_expected_evidence"],
    }


def hit_at(rank: int | None, has_expectation: bool, k: int) -> bool | None:
    if not has_expectation:
        return None
    return rank is not None and rank <= k


def build_summary(
    records: list[dict[str, Any]],
    *,
    run_id: str,
    cases_path: str,
    base_url: str,
    modes: list[str],
    top_k: int | None,
    cache_policy: str,
    warm_cache: bool,
    timeout_seconds: float,
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[mode_phase_key(record)].append(record)

    mode_results = {
        key: summarize_records(items)
        for key, items in sorted(grouped.items(), key=lambda item: item[0])
    }
    all_failures = [
        {
            "case_id": record["case_id"],
            "mode": record["mode"],
            "phase": record["phase"],
            "query_id": record["query_id"],
            "reasons": record["failure_reasons"],
            "error": record["error"],
        }
        for record in records
        if record["failure_reasons"]
    ]
    return {
        "run_id": run_id,
        "generated_at": finished_at.isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "cases_path": cases_path,
        "case_count": len({record["case_id"] for record in records}),
        "result_count": len(records),
        "base_url": base_url,
        "modes": modes,
        "top_k": top_k,
        "cache_policy": cache_policy,
        "warm_cache": warm_cache,
        "timeout_seconds": timeout_seconds,
        "mode_control": mode_control_instructions(modes, cache_policy),
        "mode_results": mode_results,
        "failures": all_failures,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [record["latency_ms"] for record in records if record.get("latency_ms") is not None]
    observed_types = Counter(
        str(record.get("observed_retriever_type") or "unknown") for record in records
    )
    summary: dict[str, Any] = {
        "mode": records[0]["mode"] if records else None,
        "phase": records[0]["phase"] if records else None,
        "total_cases": len(records),
        "completed_cases": sum(1 for record in records if not record.get("error")),
        "error_cases": sum(1 for record in records if record.get("error")),
        "observed_retriever_types": dict(observed_types),
        "mode_applied_rate": aggregate_bool(record.get("mode_applied") for record in records),
        "latency_p50": percentile(latencies, 0.50),
        "latency_p95": percentile(latencies, 0.95),
    }
    metrics: dict[str, Any] = {}
    for k in HIT_KS:
        metrics[f"doc_hit@{k}"] = aggregate_bool(
            nested(record, "retrieval", "doc_hit_at", str(k)) for record in records
        )
        metrics[f"page_hit@{k}"] = aggregate_bool(
            nested(record, "retrieval", "page_hit_at", str(k)) for record in records
        )
    metrics["MRR_doc"] = average_metric(
        nested(record, "retrieval", "mrr_doc") for record in records
    )
    metrics["MRR_page"] = average_metric(
        nested(record, "retrieval", "mrr_page") for record in records
    )
    metrics["citation_doc_hit"] = aggregate_bool(
        nested(record, "citation", "citation_doc_hit_at", "10") for record in records
    )
    metrics["citation_page_hit"] = aggregate_bool(
        nested(record, "citation", "citation_page_hit_at", "10") for record in records
    )
    metrics["answer_gold_contains"] = aggregate_bool(
        nested(record, "answer_metrics", "answer_gold_contains") for record in records
    )
    metrics["answer_numeric_match"] = aggregate_bool(
        nested(record, "answer_metrics", "answer_numeric_match") for record in records
    )
    metrics["unsupported_answer_rate"] = aggregate_bool(
        record.get("unsupported_answer") for record in records
    )
    metrics["false_insufficient_rate"] = aggregate_bool(
        record.get("false_insufficient") for record in records
    )
    metrics["cache_hit_rate"] = aggregate_bool(record.get("cache_hit") for record in records)
    summary["metrics"] = metrics
    return summary


def build_failure_buckets(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in FAILURE_BUCKETS}
    primary_records = [
        record for record in records if record.get("phase") not in {"warmup", "warm_cache"}
    ]
    by_case_phase: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in primary_records:
        by_case_phase[(record["case_id"], record["phase"])][record["mode"]] = record

    for (_case_id, phase), by_mode in by_case_phase.items():
        dense = by_mode.get("dense_only") or by_mode.get("dense")
        bm25 = by_mode.get("bm25_only") or by_mode.get("bm25") or by_mode.get("lexical")
        hybrid = by_mode.get("hybrid_rrf") or by_mode.get("hybrid")
        rerank = (
            by_mode.get("hybrid_rrf_reranker")
            or by_mode.get("hybrid_rrf_reranker_cache_warm")
            or by_mode.get("hybrid_rerank")
            or by_mode.get("hybrid-rerank")
        )

        if dense and bm25 and not correct_retrieval_hit(dense) and correct_retrieval_hit(bm25):
            buckets["dense_missed_bm25_hit"].append(
                cross_mode_bucket_entry(phase, dense, bm25)
            )
        if hybrid and rerank and correct_retrieval_hit(hybrid) and not correct_retrieval_hit(rerank):
            buckets["hybrid_found_reranker_lost"].append(
                cross_mode_bucket_entry(phase, hybrid, rerank)
            )
        if hybrid and rerank and reranker_improved_page_rank(hybrid, rerank):
            buckets["reranker_improved_page_rank"].append(
                cross_mode_bucket_entry(phase, hybrid, rerank)
            )

    for record in records:
        if retrieved_correct_but_answer_wrong(record):
            buckets["retrieved_correct_page_answer_wrong"].append(bucket_entry(record))
        if citation_missing(record):
            buckets["citation_missing"].append(bucket_entry(record))
        if record.get("false_insufficient") is True and correct_retrieval_hit(record):
            buckets["critic_false_reject"].append(bucket_entry(record))
        if record.get("cache_hit") is True:
            buckets["cache_hit"].append(bucket_entry(record))

    return {
        "bucket_counts": {name: len(items) for name, items in buckets.items()},
        "buckets": buckets,
    }


def write_artifacts(
    output_dir: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    failure_buckets: dict[str, Any],
) -> None:
    (output_dir / "summary.json").write_text(
        stable_json(summary) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(stable_json(record) + "\n")
    (output_dir / "failure_buckets.json").write_text(
        stable_json(failure_buckets) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        build_report(summary, failure_buckets),
        encoding="utf-8",
    )


def build_report(summary: dict[str, Any], failure_buckets: dict[str, Any]) -> str:
    lines = [
        "# FinanceBench Multi-mode Benchmark",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Generated: `{summary['generated_at']}`",
        f"- Cases: `{summary['cases_path']}`",
        f"- Base URL: `{summary['base_url']}`",
        f"- Modes: `{', '.join(summary['modes'])}`",
        f"- Primary cache policy: `{summary['cache_policy']}`",
        f"- Warm cache phase: `{summary['warm_cache']}`",
        "",
        "## Mode Control",
        "",
        (
            "The runner sends requested mode/cache settings through both the request `options` "
            "field and query parameters. If the service does not yet apply per-request options, "
            "restart it with the env shown below before running each mode."
        ),
        "",
        "| Mode | Restart env |",
        "| --- | --- |",
    ]
    for item in summary["mode_control"]["restart_env"]:
        env_text = " ".join(f"{key}={value}" for key, value in item["env"].items())
        lines.append(f"| `{item['mode']}` | `{env_text}` |")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            (
                "| Mode phase | n | doc@1 | doc@3 | doc@5 | doc@10 | page@1 | page@3 | "
                "page@5 | page@10 | MRR doc | MRR page | citation doc | citation page | "
                "answer contains | numeric | unsupported | false insufficient | p50 ms | "
                "p95 ms | cache hit |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, mode_summary in summary["mode_results"].items():
        metrics = mode_summary["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{key}`",
                    str(mode_summary["total_cases"]),
                    fmt_rate(metrics["doc_hit@1"]),
                    fmt_rate(metrics["doc_hit@3"]),
                    fmt_rate(metrics["doc_hit@5"]),
                    fmt_rate(metrics["doc_hit@10"]),
                    fmt_rate(metrics["page_hit@1"]),
                    fmt_rate(metrics["page_hit@3"]),
                    fmt_rate(metrics["page_hit@5"]),
                    fmt_rate(metrics["page_hit@10"]),
                    fmt_average(metrics["MRR_doc"]),
                    fmt_average(metrics["MRR_page"]),
                    fmt_rate(metrics["citation_doc_hit"]),
                    fmt_rate(metrics["citation_page_hit"]),
                    fmt_rate(metrics["answer_gold_contains"]),
                    fmt_rate(metrics["answer_numeric_match"]),
                    fmt_rate(metrics["unsupported_answer_rate"]),
                    fmt_rate(metrics["false_insufficient_rate"]),
                    fmt_number(mode_summary["latency_p50"]),
                    fmt_number(mode_summary["latency_p95"]),
                    fmt_rate(metrics["cache_hit_rate"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Failure Buckets", "", "| Bucket | Count |", "| --- | ---: |"])
    for bucket, count in failure_buckets["bucket_counts"].items():
        lines.append(f"| `{bucket}` | {count} |")

    failures = summary.get("failures", [])[:25]
    lines.extend(["", "## Top Failures", ""])
    if not failures:
        lines.append("No failures detected by the benchmark runner.")
    else:
        for failure in failures:
            reasons = ", ".join(failure["reasons"])
            lines.append(
                f"- `{failure['mode']}:{failure['phase']}:{failure['case_id']}` "
                f"{reasons}; query_id=`{failure.get('query_id')}`"
            )
    lines.append("")
    return "\n".join(lines)


def mode_control_instructions(modes: list[str], cache_policy: str) -> dict[str, Any]:
    return {
        "per_request_attempt": "options_and_query_params",
        "cache_off_env": {"ATLAS_CACHE_ENABLED": "false"},
        "primary_cache_policy": cache_policy,
        "restart_env": [
            {
                "mode": mode,
                "env": restart_env_for_mode(mode, cache_policy),
            }
            for mode in modes
        ],
    }


def restart_env_for_mode(mode: str, cache_policy: str) -> dict[str, str]:
    options = mode_options(mode)
    env = {
        "ATLAS_RETRIEVAL_MODE": str(options["retrieval_mode"]),
        "ATLAS_BM25_ENABLED": "true" if options["bm25_enabled"] else "false",
        "ATLAS_CACHE_ENABLED": "true" if cache_policy not in {"off", "bypass"} else "false",
    }
    if options.get("reranker_enabled"):
        env["ATLAS_RERANKER_ENABLED"] = "true"
    return env


def retrieved_top_k_from_trace(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    retrieval = trace.get("retrieval")
    items: Any = []
    if isinstance(retrieval, Mapping):
        for key in ("top_k", "reranked_top_k", "candidates", "items"):
            if isinstance(retrieval.get(key), list):
                items = retrieval[key]
                break
    if not items and isinstance(trace.get("evidence"), Mapping):
        evidence = trace["evidence"]
        if isinstance(evidence.get("items"), list):
            items = evidence["items"]
    if not isinstance(items, list):
        return []

    retrieved: list[dict[str, Any]] = []
    for fallback_rank, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        retrieved.append(
            {
                "rank": item.get("rank") or fallback_rank,
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id") or metadata.get("document_id"),
                "source_title": item.get("source_title")
                or item.get("doc_name")
                or metadata.get("source_title")
                or metadata.get("doc_name"),
                "source_uri": item.get("source_uri") or metadata.get("source_uri"),
                "page_start": item.get("page_start") or metadata.get("page_start"),
                "page_end": item.get("page_end") or metadata.get("page_end"),
                "score": item.get("score") or item.get("retrieval_score"),
                "retrieval_score": item.get("retrieval_score") or item.get("score"),
                "retriever_type": item.get("retriever_type") or metadata.get("retriever_type"),
                "metadata": dict(metadata),
            }
        )
    return retrieved


def citations_to_retrieved(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retrieved: list[dict[str, Any]] = []
    for rank, item in enumerate(citations, start=1):
        if not isinstance(item, Mapping):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        retrieved.append(
            {
                "rank": item.get("rank") or rank,
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id") or metadata.get("document_id"),
                "source_title": item.get("source_title")
                or item.get("doc_name")
                or metadata.get("source_title")
                or metadata.get("doc_name"),
                "source_uri": item.get("source_uri") or metadata.get("source_uri"),
                "page_start": item.get("page_start") or metadata.get("page_start"),
                "page_end": item.get("page_end") or metadata.get("page_end"),
                "retrieval_score": item.get("retrieval_score") or item.get("score"),
                "metadata": dict(metadata),
            }
        )
    return retrieved


def observed_retriever_type_from_trace(trace: Mapping[str, Any]) -> str | None:
    retrieval = trace.get("retrieval")
    if isinstance(retrieval, Mapping):
        retriever_type = retrieval.get("retriever_type")
        if retriever_type:
            return str(retriever_type)
    types = {
        str(item.get("retriever_type"))
        for item in retrieved_top_k_from_trace(trace)
        if item.get("retriever_type")
    }
    if not types:
        return None
    if len(types) == 1:
        return next(iter(types))
    if "hybrid" in types or {"dense", "bm25"} <= types or {"dense", "lexical"} <= types:
        return "hybrid"
    return "mixed"


def mode_applied(mode: str, observed: str | None) -> bool | None:
    if not observed or observed == "unknown":
        return None
    expected = mode_options(mode)["retrieval_mode"]
    observed_normalized = observed.strip().lower()
    if expected == "dense":
        return observed_normalized == "dense"
    if expected in {"bm25", "lexical"}:
        return observed_normalized in {"bm25", "lexical"}
    if expected == "hybrid":
        return observed_normalized in {"hybrid", "mixed"}
    return expected == observed_normalized


def cache_hit_from_payload_or_trace(
    payload: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> bool:
    explicit = first_bool(
        payload.get("cache_hit"),
        nested(payload, "cache", "hit"),
        nested(trace, "cache", "hit"),
        nested(trace, "query", "cache_hit"),
        nested(trace, "result", "cache_hit"),
    )
    if explicit is not None:
        return explicit

    retrieval_count = nested(trace, "retrieval", "event_count")
    generation = trace.get("generation")
    confidence = payload.get("confidence") or nested(trace, "result", "confidence")
    answer = payload.get("answer") or nested(trace, "result", "answer")
    no_generation = generation == [] or generation is None
    return (
        retrieval_count == 0
        and no_generation
        and bool(answer)
        and str(confidence or "").lower() not in {"", "unknown", "insufficient"}
    )


def unsupported_answer(
    case: EvalCase,
    answer: str | None,
    confidence: str | None,
) -> bool | None:
    if case.expected_confidence != "insufficient":
        return None
    if confidence == "insufficient":
        return False
    return bool(answer and answer.strip())


def false_insufficient(case: EvalCase, confidence: str | None) -> bool | None:
    if case.expected_confidence != "supported":
        return None
    return confidence == "insufficient"


def failure_reasons(record: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if record.get("error"):
        reasons.append("error")
    if nested(record, "retrieval", "doc_hit_at", "10") is False:
        reasons.append("doc_miss@10")
    if nested(record, "retrieval", "page_hit_at", "10") is False:
        reasons.append("page_miss@10")
    if nested(record, "citation", "citation_doc_hit_at", "10") is False:
        reasons.append("citation_doc_miss")
    if nested(record, "citation", "citation_page_hit_at", "10") is False:
        reasons.append("citation_page_miss")
    if nested(record, "answer_metrics", "answer_gold_contains") is False:
        reasons.append("answer_gold_missing")
    if nested(record, "answer_metrics", "answer_numeric_match") is False:
        reasons.append("answer_numeric_miss")
    if record.get("unsupported_answer") is True:
        reasons.append("unsupported_answer")
    if record.get("false_insufficient") is True:
        reasons.append("false_insufficient")
    if record.get("mode_applied") is False:
        reasons.append("mode_not_applied")
    return reasons


def correct_retrieval_hit(record: Mapping[str, Any]) -> bool:
    page_hit = nested(record, "retrieval", "page_hit_at", "10")
    if page_hit is not None:
        return page_hit is True
    return nested(record, "retrieval", "doc_hit_at", "10") is True


def retrieved_correct_but_answer_wrong(record: Mapping[str, Any]) -> bool:
    if not correct_retrieval_hit(record):
        return False
    answer_contains = nested(record, "answer_metrics", "answer_gold_contains")
    numeric_match = nested(record, "answer_metrics", "answer_numeric_match")
    return answer_contains is False or numeric_match is False


def citation_missing(record: Mapping[str, Any]) -> bool:
    if not correct_retrieval_hit(record):
        return False
    citation_page_hit = nested(record, "citation", "citation_page_hit_at", "10")
    if citation_page_hit is not None:
        return citation_page_hit is False
    citation_doc_hit = nested(record, "citation", "citation_doc_hit_at", "10")
    return citation_doc_hit is False


def reranker_improved_page_rank(
    hybrid: Mapping[str, Any],
    rerank: Mapping[str, Any],
) -> bool:
    rerank_rank = nested(rerank, "retrieval", "first_page_rank")
    if rerank_rank is None:
        return False
    hybrid_rank = nested(hybrid, "retrieval", "first_page_rank")
    return hybrid_rank is None or int(rerank_rank) < int(hybrid_rank)


def bucket_entry(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": record.get("case_id"),
        "question": record.get("question"),
        "mode": record.get("mode"),
        "phase": record.get("phase"),
        "query_id": record.get("query_id"),
        "trace_id": record.get("trace_id"),
        "first_doc_rank": nested(record, "retrieval", "first_doc_rank"),
        "first_page_rank": nested(record, "retrieval", "first_page_rank"),
        "confidence": record.get("confidence"),
        "failure_reasons": record.get("failure_reasons", []),
    }


def cross_mode_bucket_entry(
    phase: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": left.get("case_id"),
        "question": left.get("question"),
        "phase": phase,
        str(left.get("mode")): compact_mode_record(left),
        str(right.get("mode")): compact_mode_record(right),
    }


def compact_mode_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query_id": record.get("query_id"),
        "trace_id": record.get("trace_id"),
        "doc_hit@10": nested(record, "retrieval", "doc_hit_at", "10"),
        "page_hit@10": nested(record, "retrieval", "page_hit_at", "10"),
        "first_doc_rank": nested(record, "retrieval", "first_doc_rank"),
        "first_page_rank": nested(record, "retrieval", "first_page_rank"),
        "confidence": record.get("confidence"),
    }


def aggregate_bool(values: Any) -> dict[str, Any]:
    evaluated = [value for value in values if value is not None]
    hits = sum(1 for value in evaluated if value is True)
    total = len(evaluated)
    return {
        "hits": hits,
        "total": total,
        "rate": hits / total if total else None,
    }


def average_metric(values: Any) -> dict[str, Any]:
    evaluated = [float(value) for value in values if value is not None]
    total = len(evaluated)
    return {
        "total": total,
        "average": sum(evaluated) / total if total else None,
    }


def percentile(values: list[int | float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def first_bool(*values: Any) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
    return None


def _trace_latency_ms(trace: Mapping[str, Any]) -> int | None:
    value = nested(trace, "latency", "total_latency_ms")
    if isinstance(value, int | float):
        return int(value)
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _json_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def mode_phase_key(record: Mapping[str, Any]) -> str:
    mode = str(record.get("mode"))
    phase = str(record.get("phase"))
    if mode == "hybrid_rrf_reranker" and phase == "warm_cache":
        return "hybrid_rrf_reranker_cache_warm"
    return f"{record.get('mode')}:{record.get('phase')}"


def fmt_rate(metric: Mapping[str, Any]) -> str:
    total = metric.get("total") or 0
    if not total:
        return "n/a"
    rate = metric.get("rate")
    return f"{metric.get('hits', 0)}/{total} ({rate:.2f})"


def fmt_average(metric: Mapping[str, Any]) -> str:
    total = metric.get("total") or 0
    if not total:
        return "n/a"
    return f"{float(metric.get('average')):.3f}"


def fmt_number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.0f}"


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    raise SystemExit(main())
