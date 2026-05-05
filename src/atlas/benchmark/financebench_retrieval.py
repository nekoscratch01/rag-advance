from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atlas.core.config import Settings
from atlas.db.session import SessionLocal, init_db
from atlas.embeddings.bge_local import LocalBGEEmbedder
from atlas.embeddings.bm25_sparse import BM25SparseEncoder
from atlas.eval.metrics import _evidence_matches, dense_retrieval_metrics, normalize_expected_evidence
from atlas.eval.service import EvalCase, load_cases
from atlas.retrieval.bm25_retriever import BM25Retriever
from atlas.retrieval.dense_retriever import DenseRetriever
from atlas.retrieval.evidence import Evidence
from atlas.retrieval.hybrid_retriever import HybridRetriever
from atlas.retrieval.reranker import CrossEncoderReranker
from atlas.vector.qdrant_client import get_qdrant_client


HIT_KS = (1, 3, 5, 10)
DEFAULT_MODES = ("dense_only", "bm25_only", "hybrid_rrf", "hybrid_rrf_reranker")


@dataclass(frozen=True)
class RetrievalBenchmark:
    run_id: str
    output_dir: Path
    summary: dict[str, Any]
    records: list[dict[str, Any]]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run FinanceBench retrieval-only ablations without calling an LLM."
    )
    parser.add_argument("--cases", default="evals/financebench_cases.yaml")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--out", default="benchmarks/financebench/retrieval_runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rrf-top-k", type=int, default=40)
    parser.add_argument("--reranker-top-k", type=int, default=30)
    parser.add_argument("--reranker-output-k", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    run_id = args.run_id or datetime.now(UTC).strftime("retrieval_%Y%m%dT%H%M%SZ")
    benchmark = run_retrieval_benchmark(
        cases=cases,
        modes=parse_modes(args.modes),
        cases_path=args.cases,
        output_dir=Path(args.out),
        run_id=run_id,
        top_k=args.top_k,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        rrf_k=args.rrf_k,
        rrf_top_k=args.rrf_top_k,
        reranker_top_k=args.reranker_top_k,
        reranker_output_k=args.reranker_output_k,
    )

    print(f"FinanceBench retrieval-only benchmark run: {benchmark.run_id}")
    print(f"Output: {benchmark.output_dir}")
    print(f"- {benchmark.output_dir / 'summary.json'}")
    print(f"- {benchmark.output_dir / 'cases.jsonl'}")
    print(f"- {benchmark.output_dir / 'report.md'}")
    return 0


def parse_modes(raw_modes: str | Sequence[str]) -> list[str]:
    values = raw_modes.split(",") if isinstance(raw_modes, str) else list(raw_modes)
    modes = [str(mode).strip() for mode in values if str(mode).strip()]
    if not modes:
        raise ValueError("At least one retrieval mode is required.")
    return modes


def run_retrieval_benchmark(
    *,
    cases: list[EvalCase],
    modes: list[str],
    cases_path: str,
    output_dir: Path,
    run_id: str,
    top_k: int,
    dense_top_k: int,
    bm25_top_k: int,
    rrf_k: int,
    rrf_top_k: int,
    reranker_top_k: int,
    reranker_output_k: int | None,
) -> RetrievalBenchmark:
    init_db()
    settings = Settings()
    qdrant = get_qdrant_client()
    embedder = LocalBGEEmbedder(settings)
    sparse_encoder = BM25SparseEncoder(settings)
    dense = DenseRetriever(settings=settings, embedder=embedder, qdrant=qdrant)
    bm25 = BM25Retriever(settings=settings, sparse_encoder=sparse_encoder, qdrant=qdrant)
    reranker = CrossEncoderReranker(settings.reranker_model)
    retrievers = {
        "dense_only": dense,
        "bm25_only": bm25,
        "hybrid_rrf": HybridRetriever(
            dense,
            bm25,
            rrf_k=rrf_k,
            rrf_top_k=rrf_top_k,
            reranker=None,
            reranker_enabled=False,
            dense_top_k=dense_top_k,
            lexical_top_k=bm25_top_k,
            max_context_tokens=settings.max_context_tokens,
        ),
        "hybrid_rrf_reranker": HybridRetriever(
            dense,
            bm25,
            rrf_k=rrf_k,
            rrf_top_k=rrf_top_k,
            reranker=reranker,
            reranker_enabled=True,
            reranker_top_k=reranker_top_k,
            reranker_output_k=reranker_output_k or top_k,
            dense_top_k=dense_top_k,
            lexical_top_k=bm25_top_k,
            max_context_tokens=settings.max_context_tokens,
        ),
    }

    started_at = datetime.now(UTC)
    records: list[dict[str, Any]] = []
    with SessionLocal() as db:
        for mode in modes:
            retriever = retrievers.get(mode)
            if retriever is None:
                raise ValueError(f"Unknown retrieval mode: {mode}")
            for case in cases:
                started = time.perf_counter()
                error = None
                evidence: list[Evidence] = []
                try:
                    evidence = retriever.retrieve(db, query=case.question, top_k=top_k, filters={})
                except Exception as exc:
                    error = f"{exc.__class__.__name__}: {exc}"
                latency_ms = int((time.perf_counter() - started) * 1000)
                records.append(
                    evaluate_retrieval_case(
                        case=case,
                        mode=mode,
                        evidence=evidence,
                        latency_ms=latency_ms,
                        error=error,
                    )
                )

    summary = build_summary(
        records,
        run_id=run_id,
        cases_path=cases_path,
        modes=modes,
        settings=settings,
        top_k=top_k,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
        rrf_k=rrf_k,
        rrf_top_k=rrf_top_k,
        reranker_top_k=reranker_top_k,
        reranker_output_k=reranker_output_k or top_k,
        started_at=started_at,
        finished_at=datetime.now(UTC),
    )
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_artifacts(run_dir, summary, records)
    return RetrievalBenchmark(run_id=run_id, output_dir=run_dir, summary=summary, records=records)


def evaluate_retrieval_case(
    *,
    case: EvalCase,
    mode: str,
    evidence: list[Evidence],
    latency_ms: int,
    error: str | None,
) -> dict[str, Any]:
    retrieved = [evidence_to_retrieved(item, index) for index, item in enumerate(evidence, start=1)]
    metrics = rank_metrics(retrieved, case.expected_evidence, case.expected_sources)
    return {
        "case_id": case.id,
        "mode": mode,
        "question": case.question,
        "expected_sources": case.expected_sources,
        "expected_evidence": case.expected_evidence,
        "retrieved_top_k": retrieved,
        "metrics": metrics,
        "latency_ms": latency_ms,
        "error": error,
        "failure_reasons": failure_reasons(metrics, error),
    }


def evidence_to_retrieved(evidence: Evidence, fallback_rank: int) -> dict[str, Any]:
    metadata = evidence.metadata or {}
    return {
        "rank": evidence.rank or fallback_rank,
        "evidence_id": evidence.evidence_id,
        "chunk_id": evidence.chunk_id,
        "parent_id": evidence.parent_id or metadata.get("parent_id"),
        "child_ids": list(evidence.child_ids),
        "document_id": evidence.document_id,
        "source_title": evidence.source_title,
        "source_uri": evidence.source_uri,
        "page_start": evidence.page_start,
        "page_end": evidence.page_end,
        "retrieval_score": evidence.retrieval_score,
        "retriever_type": _retriever_type(evidence),
        "retrieved_by": list(evidence.retrieved_by),
        "metadata": metadata,
    }


def rank_metrics(
    retrieved: list[dict[str, Any]],
    expected_evidence: list[dict[str, Any]],
    expected_sources: list[str],
) -> dict[str, Any]:
    details = dense_retrieval_metrics(retrieved, expected_evidence, expected_sources)
    doc_has_expectation = details["retrieval_doc_hit"] is not None
    page_has_expectation = details["retrieval_page_hit"] is not None
    doc_rank = details["first_doc_match_rank"]
    page_rank = details["first_page_match_rank"]
    return {
        "doc_hit_at": {str(k): hit_at(doc_rank, doc_has_expectation, k) for k in HIT_KS},
        "page_hit_at": {str(k): hit_at(page_rank, page_has_expectation, k) for k in HIT_KS},
        "mrr_doc": details["retrieval_doc_mrr"],
        "mrr_page": details["retrieval_page_mrr"],
        "map_doc": average_precision(
            retrieved,
            details["normalized_expected_evidence"],
            match_page=False,
        ),
        "map_page": average_precision(
            retrieved,
            details["normalized_expected_evidence"],
            match_page=True,
        ),
        "first_doc_rank": doc_rank,
        "first_page_rank": page_rank,
        "normalized_expected_evidence": details["normalized_expected_evidence"],
    }


def average_precision(
    retrieved: list[dict[str, Any]],
    normalized_expected_evidence: list[dict[str, Any]],
    *,
    match_page: bool,
) -> float | None:
    if match_page and not any(item["page_candidates"] for item in normalized_expected_evidence):
        return None
    if not match_page and not any(
        item["document_ids"] or item["doc_hints"] for item in normalized_expected_evidence
    ):
        return None

    relevant_total = max(1, len(normalized_expected_evidence))
    hits = 0
    precision_sum = 0.0
    matched_keys: set[str] = set()
    for fallback_rank, item in enumerate(retrieved, start=1):
        rank = _int_value(item.get("rank")) or fallback_rank
        matched_key = first_matching_expected_key(
            item,
            normalized_expected_evidence,
            match_page=match_page,
        )
        if matched_key is None or matched_key in matched_keys:
            continue
        matched_keys.add(matched_key)
        hits += 1
        precision_sum += hits / rank
    return precision_sum / relevant_total


def first_matching_expected_key(
    retrieved: dict[str, Any],
    normalized_expected_evidence: list[dict[str, Any]],
    *,
    match_page: bool,
) -> str | None:
    for index, expected in enumerate(normalized_expected_evidence):
        if _evidence_matches(retrieved, expected, match_page=match_page):
            return str(index)
    return None


def hit_at(rank: int | None, has_expectation: bool, k: int) -> bool | None:
    if not has_expectation:
        return None
    return rank is not None and rank <= k


def build_summary(
    records: list[dict[str, Any]],
    *,
    run_id: str,
    cases_path: str,
    modes: list[str],
    settings: Settings,
    top_k: int,
    dense_top_k: int,
    bm25_top_k: int,
    rrf_k: int,
    rrf_top_k: int,
    reranker_top_k: int,
    reranker_output_k: int,
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["mode"]].append(record)
    mode_results = {mode: summarize_records(grouped[mode]) for mode in modes}
    return {
        "run_id": run_id,
        "generated_at": finished_at.isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "cases_path": cases_path,
        "case_count": len({record["case_id"] for record in records}),
        "result_count": len(records),
        "modes": modes,
        "config": {
            "collection": settings.qdrant_collection,
            "embedding_model": settings.embedding_model,
            "bm25_model": settings.bm25_model,
            "bm25_language": settings.bm25_language,
            "bm25_k": settings.bm25_k,
            "bm25_b": settings.bm25_b,
            "bm25_avg_len": settings.bm25_avg_len,
            "reranker_model": settings.reranker_model,
            "top_k": top_k,
            "dense_top_k": dense_top_k,
            "bm25_top_k": bm25_top_k,
            "rrf_k": rrf_k,
            "rrf_top_k": rrf_top_k,
            "reranker_top_k": reranker_top_k,
            "reranker_output_k": reranker_output_k,
        },
        "mode_results": mode_results,
        "failure_buckets": failure_buckets(records),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [record["latency_ms"] for record in records if record.get("latency_ms") is not None]
    metrics = {
        f"doc_hit@{k}": aggregate_bool(
            nested(record, "metrics", "doc_hit_at", str(k)) for record in records
        )
        for k in HIT_KS
    }
    metrics.update(
        {
            f"page_hit@{k}": aggregate_bool(
                nested(record, "metrics", "page_hit_at", str(k)) for record in records
            )
            for k in HIT_KS
        }
    )
    metrics["MRR_doc"] = average_metric(
        nested(record, "metrics", "mrr_doc") for record in records
    )
    metrics["MRR_page"] = average_metric(
        nested(record, "metrics", "mrr_page") for record in records
    )
    metrics["MAP_doc"] = average_metric(
        nested(record, "metrics", "map_doc") for record in records
    )
    metrics["MAP_page"] = average_metric(
        nested(record, "metrics", "map_page") for record in records
    )
    return {
        "total_cases": len(records),
        "completed_cases": sum(1 for record in records if not record.get("error")),
        "error_cases": sum(1 for record in records if record.get("error")),
        "latency_p50": percentile(latencies, 0.50),
        "latency_p95": percentile(latencies, 0.95),
        "metrics": metrics,
    }


def failure_buckets(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_case[record["case_id"]][record["mode"]] = record

    buckets: dict[str, list[dict[str, Any]]] = {
        "dense_missed_bm25_hit": [],
        "hybrid_found_reranker_lost": [],
        "reranker_improved_page_rank": [],
        "both_dense_and_hybrid_missed": [],
        "errors": [],
    }
    for _case_id, modes in by_case.items():
        dense = modes.get("dense_only")
        bm25 = modes.get("bm25_only")
        hybrid = modes.get("hybrid_rrf")
        reranker = modes.get("hybrid_rrf_reranker")
        if dense and bm25 and not correct_page_or_doc(dense) and correct_page_or_doc(bm25):
            buckets["dense_missed_bm25_hit"].append(bucket_pair(dense, bm25))
        if hybrid and reranker and correct_page_or_doc(hybrid) and not correct_page_or_doc(reranker):
            buckets["hybrid_found_reranker_lost"].append(bucket_pair(hybrid, reranker))
        if hybrid and reranker and improved_page_rank(hybrid, reranker):
            buckets["reranker_improved_page_rank"].append(bucket_pair(hybrid, reranker))
        if dense and hybrid and not correct_page_or_doc(dense) and not correct_page_or_doc(hybrid):
            buckets["both_dense_and_hybrid_missed"].append(bucket_entry(hybrid))
    for record in records:
        if record.get("error"):
            buckets["errors"].append(bucket_entry(record))
    return {
        "bucket_counts": {name: len(items) for name, items in buckets.items()},
        "buckets": buckets,
    }


def write_artifacts(output_dir: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    (output_dir / "summary.json").write_text(stable_json(summary) + "\n", encoding="utf-8")
    with (output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(stable_json(record) + "\n")
    (output_dir / "report.md").write_text(build_report(summary), encoding="utf-8")


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# FinanceBench Retrieval-only Ablation",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Generated: `{summary['generated_at']}`",
        f"- Cases: `{summary['cases_path']}`",
        f"- Collection: `{summary['config']['collection']}`",
        f"- Embedding: `{summary['config']['embedding_model']}`",
        f"- BM25: `{summary['config']['bm25_model']}`",
        f"- Reranker: `{summary['config']['reranker_model']}`",
        "",
        "## Metrics",
        "",
        (
            "| Mode | n | doc@1 | doc@3 | doc@5 | doc@10 | page@1 | page@3 | page@5 | "
            "page@10 | MRR doc | MRR page | MAP doc | MAP page | p50 ms | p95 ms | errors |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, result in summary["mode_results"].items():
        metrics = result["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{mode}`",
                    str(result["total_cases"]),
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
                    fmt_average(metrics["MAP_doc"]),
                    fmt_average(metrics["MAP_page"]),
                    fmt_number(result["latency_p50"]),
                    fmt_number(result["latency_p95"]),
                    str(result["error_cases"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Failure Buckets", "", "| Bucket | Count |", "| --- | ---: |"])
    for name, count in summary["failure_buckets"]["bucket_counts"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.append("")
    return "\n".join(lines)


def failure_reasons(metrics: dict[str, Any], error: str | None) -> list[str]:
    reasons = []
    if error:
        reasons.append("error")
    if metrics["doc_hit_at"]["10"] is False:
        reasons.append("doc_miss@10")
    if metrics["page_hit_at"]["10"] is False:
        reasons.append("page_miss@10")
    return reasons


def correct_page_or_doc(record: Mapping[str, Any]) -> bool:
    page_hit = nested(record, "metrics", "page_hit_at", "10")
    if page_hit is not None:
        return page_hit is True
    return nested(record, "metrics", "doc_hit_at", "10") is True


def improved_page_rank(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    right_rank = nested(right, "metrics", "first_page_rank")
    if right_rank is None:
        return False
    left_rank = nested(left, "metrics", "first_page_rank")
    return left_rank is None or int(right_rank) < int(left_rank)


def bucket_entry(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": record.get("case_id"),
        "mode": record.get("mode"),
        "question": record.get("question"),
        "first_doc_rank": nested(record, "metrics", "first_doc_rank"),
        "first_page_rank": nested(record, "metrics", "first_page_rank"),
        "failure_reasons": record.get("failure_reasons"),
        "error": record.get("error"),
    }


def bucket_pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": left.get("case_id"),
        "question": left.get("question"),
        str(left.get("mode")): bucket_entry(left),
        str(right.get("mode")): bucket_entry(right),
    }


def aggregate_bool(values) -> dict[str, Any]:
    present = [value for value in values if value is not None]
    total = len(present)
    hits = sum(1 for value in present if value is True)
    return {"hits": hits, "total": total, "rate": hits / total if total else None}


def average_metric(values) -> dict[str, Any]:
    present = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    total = len(present)
    return {"total": total, "average": sum(present) / total if total else None}


def percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))
    return ordered[index]


def nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fmt_rate(metric: Mapping[str, Any]) -> str:
    total = metric.get("total") or 0
    if not total:
        return "n/a"
    rate = metric.get("rate")
    return f"{metric.get('hits', 0)}/{total} ({rate:.3f})"


def fmt_average(metric: Mapping[str, Any]) -> str:
    total = metric.get("total") or 0
    if not total:
        return "n/a"
    return f"{float(metric.get('average')):.3f}"


def fmt_number(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _retriever_type(evidence: Evidence) -> str:
    sources = set(evidence.retrieved_by or ())
    if {"dense", "bm25"} <= sources or {"dense", "lexical"} <= sources:
        return "hybrid"
    if "bm25" in sources or "lexical" in sources:
        return "bm25"
    if "dense" in sources:
        return "dense"
    return "unknown"


def _int_value(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
