from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from atlas.query_orchestrator.ontology import FinanceMetricOntology
from atlas.retrieval.candidate import Candidate
from atlas.retrieval.fusion import DEFAULT_RRF_K, WeightedRRFInput, weighted_rrf_fuse


DEFAULT_REPORT_DIR = Path("benchmarks/rag_quality/v1_hybrid_provider_reset/smoke_runs")
DEFAULT_ONTOLOGY_PATH = Path("configs/finance_metric_ontology.yaml")
HIT_KS = (1, 3)


@dataclass(frozen=True)
class SyntheticChunk:
    chunk_id: str
    parent_id: str
    document_id: str
    source_title: str
    company: str
    page_start: int
    page_end: int
    text: str
    parent_text: str
    previous_page_text: str = ""
    next_page_text: str = ""
    section_title: str | None = None
    metrics: tuple[str, ...] = ()
    periods: tuple[str, ...] = ()
    source_type: str = "text_chunk"


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    question: str
    entity: str
    periods: tuple[str, ...]
    metric: str
    expected_document_id: str
    expected_page: int
    expected_parent_id: str
    expected_terms: tuple[str, ...]
    local_should_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class AblationConfig:
    name: str
    group: str
    description: str
    rewrite_policy: str = "ontology_aliases"
    filter_strategy: str = "metadata_filter_only"
    fusion_strategy: str = "python_weighted_rrf"
    candidate_shape: str = "parent_block"
    reranker_input: str = "local_terms_candidate"
    top_k: int = 3
    rrf_k: int = DEFAULT_RRF_K
    max_context_tokens: int | None = None
    planned_only: bool = False


@dataclass(frozen=True)
class BenchmarkRun:
    run_id: str
    output_dir: Path
    summary: dict[str, Any]
    records: list[dict[str, Any]]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an offline V1 hybrid provider reset smoke and ablation harness."
    )
    parser.add_argument("--out", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY_PATH))
    args = parser.parse_args(argv)

    run_id = args.run_id or datetime.now(UTC).strftime("provider_reset_%Y%m%dT%H%M%SZ")
    run = run_provider_reset_smoke(
        output_dir=Path(args.out),
        run_id=run_id,
        ontology_path=Path(args.ontology),
    )
    print(f"V1 hybrid provider reset smoke: {run.run_id}")
    print(f"Output: {run.output_dir}")
    print(f"- {run.output_dir / 'summary.json'}")
    print(f"- {run.output_dir / 'cases.jsonl'}")
    print(f"- {run.output_dir / 'report.md'}")
    return 0


def run_provider_reset_smoke(
    *,
    output_dir: Path = DEFAULT_REPORT_DIR,
    run_id: str | None = None,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
) -> BenchmarkRun:
    ontology = FinanceMetricOntology.load(ontology_path)
    cases = synthetic_cases()
    corpus = synthetic_corpus()
    configs = default_ablation_configs()
    run_id = run_id or datetime.now(UTC).strftime("provider_reset_%Y%m%dT%H%M%SZ")
    started_at = datetime.now(UTC)

    records: list[dict[str, Any]] = []
    for config in configs:
        for case in cases:
            records.append(
                evaluate_case(
                    case=case,
                    corpus=corpus,
                    config=config,
                    ontology=ontology,
                )
            )

    finished_at = datetime.now(UTC)
    summary = build_summary(
        records,
        configs=configs,
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        ontology_path=ontology_path,
    )
    run_dir = output_dir / run_id
    write_artifacts(run_dir, summary, records)
    return BenchmarkRun(run_id=run_id, output_dir=run_dir, summary=summary, records=records)


def default_ablation_configs() -> list[AblationConfig]:
    return [
        AblationConfig(
            name="rewrite_unit_text",
            group="query_rewrite",
            description="只使用 unit.text，不追加 should_terms 或 ontology aliases。",
            rewrite_policy="unit_text",
        ),
        AblationConfig(
            name="rewrite_should_terms",
            group="query_rewrite",
            description=(
                "unit.text + 局部 should_terms，"
                "主要模拟人工或 planner 给出的局部词法提示。"
            ),
            rewrite_policy="should_terms",
        ),
        AblationConfig(
            name="rewrite_ontology_aliases",
            group="query_rewrite",
            description="unit.text + should_terms + finance ontology aliases。",
            rewrite_policy="ontology_aliases",
        ),
        AblationConfig(
            name="filter_no_hard_filter",
            group="filter_strategy",
            description="不做 metadata 过滤，也不硬性要求 must_have_terms。",
            filter_strategy="no_hard_filter",
        ),
        AblationConfig(
            name="filter_metadata_only",
            group="filter_strategy",
            description=(
                "只使用 company/document 级 metadata filter，"
                "不用 must_have_terms 硬过滤。"
            ),
            filter_strategy="metadata_filter_only",
        ),
        AblationConfig(
            name="filter_must_have_hard",
            group="filter_strategy",
            description=(
                "metadata filter 后强制要求 must_have_terms 全部出现在候选文本。"
            ),
            filter_strategy="must_have_hard_filter",
        ),
        AblationConfig(
            name="filter_must_terms_sparse_boost",
            group="filter_strategy",
            description=(
                "metadata filter 后把 must_have_terms 作为 sparse boost，"
                "而不是召回硬门槛。"
            ),
            filter_strategy="must_terms_sparse_boost",
        ),
        AblationConfig(
            name="fusion_dense_only",
            group="fusion",
            description="dense-only baseline。",
            fusion_strategy="dense_only",
            reranker_input="",
        ),
        AblationConfig(
            name="fusion_sparse_only",
            group="fusion",
            description="sparse/BM25-only baseline。",
            fusion_strategy="sparse_only",
            reranker_input="",
        ),
        AblationConfig(
            name="fusion_python_weighted_rrf",
            group="fusion",
            description="当前可执行的 provider-local Python Weighted RRF。",
            fusion_strategy="python_weighted_rrf",
            reranker_input="",
        ),
        AblationConfig(
            name="fusion_qdrant_rrf_planned",
            group="fusion",
            description="Qdrant server-side RRF 计划项；当前 smoke 不触发外部 Qdrant。",
            fusion_strategy="qdrant_rrf",
            planned_only=True,
        ),
        AblationConfig(
            name="shape_child_chunk",
            group="candidate_shape",
            description="只把检索 child chunk 交给下游。",
            candidate_shape="child_chunk",
        ),
        AblationConfig(
            name="shape_parent_block",
            group="candidate_shape",
            description="child 命中后回填 parent block。",
            candidate_shape="parent_block",
        ),
        AblationConfig(
            name="shape_page_neighborhood",
            group="candidate_shape",
            description="parent block 外再拼接相邻页 neighborhood。",
            candidate_shape="page_neighborhood",
        ),
        AblationConfig(
            name="shape_token_budget_18",
            group="candidate_shape",
            description=(
                "parent block 形态下施加极小 token budget，观察证据被挤出。"
            ),
            candidate_shape="parent_block",
            max_context_tokens=18,
        ),
        AblationConfig(
            name="rerank_original_query_candidate",
            group="reranker_input",
            description="reranker 输入只含 original query + candidate。",
            reranker_input="original_query_candidate",
        ),
        AblationConfig(
            name="rerank_current_unit_candidate",
            group="reranker_input",
            description="reranker 输入使用当前 retrieval unit + candidate。",
            reranker_input="current_unit_candidate",
        ),
        AblationConfig(
            name="rerank_local_terms_candidate",
            group="reranker_input",
            description="reranker 输入使用当前 unit + must/should local terms + candidate。",
            reranker_input="local_terms_candidate",
        ),
        AblationConfig(
            name="rerank_full_plan_summary_candidate",
            group="reranker_input",
            description="reranker 输入使用 full plan summary + candidate。",
            reranker_input="full_plan_summary_candidate",
        ),
        AblationConfig(
            name="rerank_full_plan_all_units_candidate",
            group="reranker_input",
            description="reranker 输入使用 full plan/all units + candidate。",
            reranker_input="full_plan_all_units_candidate",
        ),
    ]


def evaluate_case(
    *,
    case: SyntheticCase,
    corpus: Sequence[SyntheticChunk],
    config: AblationConfig,
    ontology: FinanceMetricOntology,
) -> dict[str, Any]:
    if config.planned_only:
        return planned_record(case, config)

    query = build_retrieval_query(case, config, ontology)
    candidates_by_lane = retrieve_lanes(
        case=case,
        corpus=corpus,
        query=query,
        config=config,
        ontology=ontology,
    )
    candidates = fuse_candidates(candidates_by_lane, config)
    if config.reranker_input:
        candidates = rerank_candidates(
            case=case,
            query=query,
            candidates=candidates,
            config=config,
            ontology=ontology,
        )
    candidates = apply_candidate_shape(candidates, corpus, config)
    included, dropped = apply_token_budget(candidates, config)
    metrics = rank_metrics(included, case, config.top_k)

    return {
        "case_id": case.case_id,
        "variant": config.name,
        "group": config.group,
        "status": "completed",
        "question": case.question,
        "query": query,
        "config": config_payload(config),
        "metrics": metrics,
        "included_top_k": [
            candidate_payload(item, index)
            for index, item in enumerate(included, start=1)
        ],
        "dropped": [candidate_payload(item, index) for index, item in enumerate(dropped, start=1)],
        "failure_reasons": failure_reasons(metrics, dropped, case),
    }


def planned_record(case: SyntheticCase, config: AblationConfig) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "variant": config.name,
        "group": config.group,
        "status": "planned_not_run",
        "question": case.question,
        "query": case.question,
        "config": config_payload(config),
        "metrics": empty_metrics(),
        "included_top_k": [],
        "dropped": [],
        "failure_reasons": ["planned_not_run"],
    }


def build_retrieval_query(
    case: SyntheticCase,
    config: AblationConfig,
    ontology: FinanceMetricOntology,
) -> dict[str, Any]:
    aliases = metric_aliases(case.metric, ontology)
    should_terms: tuple[str, ...] = ()
    if config.rewrite_policy in {"should_terms", "ontology_aliases"}:
        should_terms = case.local_should_terms
    if config.rewrite_policy == "ontology_aliases":
        should_terms = dedupe((*should_terms, *aliases[:6]))

    must_have_terms = dedupe((case.entity, *case.periods))
    return {
        "unit_text": case.question,
        "sparse_text": " ".join([case.question, *should_terms]),
        "original_query": case.question,
        "must_have_terms": list(must_have_terms),
        "should_terms": list(should_terms),
        "entity": case.entity,
        "periods": list(case.periods),
        "metric": case.metric,
    }


def retrieve_lanes(
    *,
    case: SyntheticCase,
    corpus: Sequence[SyntheticChunk],
    query: dict[str, Any],
    config: AblationConfig,
    ontology: FinanceMetricOntology,
) -> dict[str, list[Candidate]]:
    lanes = ("dense", "bm25", "metric_alias", "table")
    candidates_by_lane: dict[str, list[Candidate]] = {}
    for lane in lanes:
        if config.fusion_strategy == "dense_only" and lane != "dense":
            continue
        if config.fusion_strategy == "sparse_only" and lane != "bm25":
            continue

        scored: list[tuple[float, SyntheticChunk, dict[str, Any]]] = []
        for chunk in corpus:
            if not passes_filter(chunk, case, query, config):
                continue
            score, details = lane_score(chunk, case, query, lane, config, ontology)
            if score <= 0:
                continue
            scored.append((score, chunk, details))
        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].document_id,
                item[1].page_start,
                item[1].chunk_id,
            )
        )
        candidates_by_lane[lane] = [
            chunk_to_candidate(
                chunk,
                lane=lane,
                rank=rank,
                score=score,
                details=details,
            )
            for rank, (score, chunk, details) in enumerate(scored[:8], start=1)
        ]
    return candidates_by_lane


def passes_filter(
    chunk: SyntheticChunk,
    case: SyntheticCase,
    query: dict[str, Any],
    config: AblationConfig,
) -> bool:
    if config.filter_strategy != "no_hard_filter" and chunk.company != case.entity:
        return False
    if config.filter_strategy != "must_have_hard_filter":
        return True
    searchable = normalize_term(
        " ".join([chunk.text, chunk.parent_text, chunk.section_title or ""])
    )
    return all(normalize_term(term) in searchable for term in query["must_have_terms"])


def lane_score(
    chunk: SyntheticChunk,
    case: SyntheticCase,
    query: dict[str, Any],
    lane: str,
    config: AblationConfig,
    ontology: FinanceMetricOntology,
) -> tuple[float, dict[str, Any]]:
    text_blob = normalized_blob(chunk)
    dense_tokens = token_set(query["unit_text"])
    sparse_tokens = token_set(query.get("sparse_text") or query["unit_text"])
    term_tokens = token_set(" ".join([*query["must_have_terms"], *query["should_terms"]]))
    aliases = metric_aliases(case.metric, ontology)
    alias_hits = [
        alias
        for alias in aliases
        if normalize_term(alias) and normalize_term(alias) in text_blob
    ]
    period_hits = [period for period in case.periods if period in text_blob]
    entity_hit = normalize_term(case.entity) in text_blob or chunk.company == case.entity
    expected_metric_hit = case.metric in chunk.metrics

    if lane == "dense":
        score = 0.15 * len(dense_tokens & token_set(chunk.text))
        if expected_metric_hit:
            score += 1.8
        if entity_hit:
            score += 1.0
        score += 0.45 * len(period_hits)
    elif lane == "bm25":
        score = 0.35 * len(sparse_tokens & token_set(chunk.text))
        score += 0.25 * len(period_hits)
        if entity_hit:
            score += 0.8
        if config.filter_strategy == "must_terms_sparse_boost":
            score += 0.5 * must_term_hits(chunk, query)
    elif lane == "metric_alias":
        score = 0.2 * len(term_tokens & token_set(chunk.text))
        score += 1.2 * len(alias_hits)
        if entity_hit:
            score += 0.7
        score += 0.35 * len(period_hits)
    elif lane == "table":
        score = 0.2 * len(term_tokens & token_set(chunk.text))
        if re.search(r"\b\d[\d,.]*\b", chunk.text):
            score += 0.9
        if "cash flow" in text_blob or "statement" in text_blob:
            score += 0.5
        if expected_metric_hit:
            score += 0.8
        score += 0.25 * len(period_hits)
    else:
        score = 0.0

    return score, {
        "alias_hits": alias_hits,
        "period_hits": period_hits,
        "entity_hit": entity_hit,
        "expected_metric_hit": expected_metric_hit,
    }


def fuse_candidates(
    candidates_by_lane: dict[str, list[Candidate]],
    config: AblationConfig,
) -> list[Candidate]:
    if config.fusion_strategy == "dense_only":
        return rank_final(candidates_by_lane.get("dense", ()))
    if config.fusion_strategy == "sparse_only":
        return rank_final(candidates_by_lane.get("bm25", ()))

    lane_weights = {
        "dense": 1.0,
        "bm25": 1.1,
        "metric_alias": 1.6,
        "table": 1.4,
    }
    return weighted_rrf_fuse(
        [
            WeightedRRFInput(
                lane=lane,
                candidates=candidates,
                weight=lane_weights.get(lane, 1.0),
            )
            for lane, candidates in candidates_by_lane.items()
        ],
        rrf_k=config.rrf_k,
        limit=8,
    )


def rerank_candidates(
    *,
    case: SyntheticCase,
    query: dict[str, Any],
    candidates: list[Candidate],
    config: AblationConfig,
    ontology: FinanceMetricOntology,
) -> list[Candidate]:
    prompt_terms = reranker_prompt_terms(case, query, config, ontology)
    scored = []
    for candidate in candidates:
        text_tokens = token_set(candidate.text)
        score = 0.25 * len(prompt_terms & text_tokens)
        if normalize_term(case.entity) in normalized_blob_text(candidate.text):
            score += 1.0
        score += 0.55 * sum(
            1 for period in case.periods if period in normalized_blob_text(candidate.text)
        )
        if case.metric in tuple(candidate.metadata.get("metrics") or ()):
            score += 1.4
        if config.reranker_input == "full_plan_all_units_candidate":
            # A small noise penalty captures the observed risk of broad plan prompts:
            # more terms can make generic financial pages look relevant.
            score -= max(0, len(prompt_terms) - 18) * 0.03
        scored.append((score, candidate))
    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].fusion_rank or item[1].final_rank or 999,
            item[1].chunk_id,
        )
    )
    ranked: list[Candidate] = []
    for rank, (score, candidate) in enumerate(scored, start=1):
        metadata = dict(candidate.metadata)
        metadata["reranker_input_variant"] = config.reranker_input
        metadata["reranker_prompt_terms"] = sorted(prompt_terms)
        ranked.append(
            replace(
                candidate,
                rerank_rank=rank,
                rerank_score=score,
                final_rank=rank,
                metadata=metadata,
            )
        )
    return ranked


def apply_candidate_shape(
    candidates: list[Candidate],
    corpus: Sequence[SyntheticChunk],
    config: AblationConfig,
) -> list[Candidate]:
    by_chunk = {chunk.chunk_id: chunk for chunk in corpus}
    shaped: list[Candidate] = []
    seen_parent: set[str] = set()
    for candidate in candidates:
        chunk = by_chunk.get(str(candidate.metadata.get("raw_chunk_id") or candidate.chunk_id))
        if chunk is None:
            shaped.append(candidate)
            continue
        if config.candidate_shape == "child_chunk":
            shaped.append(candidate)
            continue
        if chunk.parent_id in seen_parent:
            continue
        seen_parent.add(chunk.parent_id)
        text = chunk.parent_text
        source_type = "parent_block"
        if config.candidate_shape == "page_neighborhood":
            text = "\n".join(
                part
                for part in (chunk.previous_page_text, chunk.parent_text, chunk.next_page_text)
                if part
            )
            source_type = "page_neighborhood"
        shaped.append(
            replace(
                candidate,
                chunk_id=chunk.parent_id,
                parent_id=chunk.parent_id,
                text=text,
                token_count=token_count(text),
                page_start=chunk.page_start - (1 if chunk.previous_page_text else 0),
                page_end=chunk.page_end + (1 if chunk.next_page_text else 0),
                source_type=source_type,
                metadata={
                    **candidate.metadata,
                    "raw_chunk_id": chunk.chunk_id,
                    "candidate_shape": config.candidate_shape,
                },
            )
        )
    return rank_final(shaped)


def apply_token_budget(
    candidates: list[Candidate],
    config: AblationConfig,
) -> tuple[list[Candidate], list[Candidate]]:
    if config.max_context_tokens is None:
        return candidates[: config.top_k], []
    included: list[Candidate] = []
    dropped: list[Candidate] = []
    used = 0
    for candidate in candidates:
        if len(included) >= config.top_k:
            break
        if used + candidate.token_count <= config.max_context_tokens:
            included.append(candidate)
            used += candidate.token_count
        else:
            metadata = dict(candidate.metadata)
            metadata["drop_reason"] = "token_budget"
            dropped.append(replace(candidate, metadata=metadata))
    return rank_final(included), dropped


def rank_metrics(
    candidates: list[Candidate],
    case: SyntheticCase,
    top_k: int,
) -> dict[str, Any]:
    first_doc_rank: int | None = None
    first_page_rank: int | None = None
    first_parent_rank: int | None = None
    first_answer_terms_rank: int | None = None
    for fallback_rank, candidate in enumerate(candidates[:top_k], start=1):
        rank = (
            candidate.final_rank
            or candidate.rerank_rank
            or candidate.fusion_rank
            or fallback_rank
        )
        if candidate.document_id == case.expected_document_id and first_doc_rank is None:
            first_doc_rank = rank
        if (
            candidate.parent_id == case.expected_parent_id
            or candidate.chunk_id == case.expected_parent_id
        ) and first_parent_rank is None:
            first_parent_rank = rank
        if page_matches(candidate, case.expected_page) and first_page_rank is None:
            first_page_rank = rank
        if answer_terms_match(candidate, case) and first_answer_terms_rank is None:
            first_answer_terms_rank = rank
    return {
        "doc_hit_at": {str(k): hit_at(first_doc_rank, k) for k in HIT_KS},
        "page_hit_at": {str(k): hit_at(first_page_rank, k) for k in HIT_KS},
        "parent_hit_at": {str(k): hit_at(first_parent_rank, k) for k in HIT_KS},
        "answer_terms_hit_at": {
            str(k): hit_at(first_answer_terms_rank, k) for k in HIT_KS
        },
        "mrr_doc": reciprocal_rank(first_doc_rank),
        "mrr_page": reciprocal_rank(first_page_rank),
        "mrr_answer_terms": reciprocal_rank(first_answer_terms_rank),
        "first_doc_rank": first_doc_rank,
        "first_page_rank": first_page_rank,
        "first_parent_rank": first_parent_rank,
        "first_answer_terms_rank": first_answer_terms_rank,
    }


def build_summary(
    records: list[dict[str, Any]],
    *,
    configs: Sequence[AblationConfig],
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    ontology_path: Path,
) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[str]] = defaultdict(list)
    for config in configs:
        by_group[config.group].append(config.name)
    for record in records:
        by_variant[record["variant"]].append(record)

    variant_results = {
        variant: summarize_records(items)
        for variant, items in by_variant.items()
    }
    return {
        "run_id": run_id,
        "generated_at": finished_at.isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "ontology_path": str(ontology_path),
        "case_count": len({record["case_id"] for record in records}),
        "variant_count": len(variant_results),
        "groups": dict(by_group),
        "variant_results": variant_results,
        "failure_buckets": failure_buckets(records),
        "notes": [
            (
                "offline synthetic smoke; not a replacement for FinanceBench retrieval-only "
                "or generated-answer evaluation"
            ),
            "does not call Postgres, Qdrant, OpenAI, or local CrossEncoder models",
        ],
    }


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record["status"] == "completed"]
    planned = [record for record in records if record["status"] == "planned_not_run"]
    metrics = {
        f"doc_hit@{k}": aggregate_bool(
            record["metrics"]["doc_hit_at"][str(k)] for record in completed
        )
        for k in HIT_KS
    }
    metrics.update(
        {
            f"page_hit@{k}": aggregate_bool(
                record["metrics"]["page_hit_at"][str(k)] for record in completed
            )
            for k in HIT_KS
        }
    )
    metrics.update(
        {
            f"parent_hit@{k}": aggregate_bool(
                record["metrics"]["parent_hit_at"][str(k)] for record in completed
            )
            for k in HIT_KS
        }
    )
    metrics.update(
        {
            f"answer_terms_hit@{k}": aggregate_bool(
                record["metrics"]["answer_terms_hit_at"][str(k)] for record in completed
            )
            for k in HIT_KS
        }
    )
    metrics["MRR_doc"] = average_metric(record["metrics"]["mrr_doc"] for record in completed)
    metrics["MRR_page"] = average_metric(record["metrics"]["mrr_page"] for record in completed)
    metrics["MRR_answer_terms"] = average_metric(
        record["metrics"]["mrr_answer_terms"] for record in completed
    )
    return {
        "total_cases": len(records),
        "completed_cases": len(completed),
        "planned_cases": len(planned),
        "metrics": metrics,
        "failure_counts": count_failures(records),
    }


def failure_buckets(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for reason in record["failure_reasons"]:
            buckets[reason].append(
                {
                    "case_id": record["case_id"],
                    "variant": record["variant"],
                    "group": record["group"],
                    "first_doc_rank": record["metrics"]["first_doc_rank"],
                    "first_page_rank": record["metrics"]["first_page_rank"],
                }
            )
    return {
        "bucket_counts": {name: len(items) for name, items in sorted(buckets.items())},
        "buckets": dict(buckets),
    }


def write_artifacts(
    output_dir: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(stable_json(summary) + "\n", encoding="utf-8")
    with (output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(stable_json(record) + "\n")
    (output_dir / "report.md").write_text(build_smoke_report(summary), encoding="utf-8")


def build_smoke_report(summary: dict[str, Any]) -> str:
    lines = [
        "# V1 Hybrid Provider Reset 离线 Smoke",
        "",
        f"- 运行 ID：`{summary['run_id']}`",
        f"- 生成时间：`{summary['generated_at']}`",
        f"- 案例数：`{summary['case_count']}`",
        f"- 变体数：`{summary['variant_count']}`",
        "",
        "## 分组结果",
        "",
    ]
    for group, variants in summary["groups"].items():
        lines.extend(
            [
                f"### {group}",
                "",
                (
                    "| 变体 | 已完成 | 计划项 | page@1 | page@3 | "
                    "answer_terms@3 | MRR page | 失败桶 |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for variant in variants:
            result = summary["variant_results"][variant]
            metrics = result["metrics"]
            failures = ", ".join(
                f"{name}:{count}" for name, count in result["failure_counts"].items()
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{variant}`",
                        str(result["completed_cases"]),
                        str(result["planned_cases"]),
                        fmt_rate(metrics["page_hit@1"]),
                        fmt_rate(metrics["page_hit@3"]),
                        fmt_rate(metrics["answer_terms_hit@3"]),
                        fmt_number(metrics["MRR_page"]),
                        failures or "-",
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## 说明",
            "",
            (
                "这是离线 synthetic smoke，用来检查消融维度、trace 字段和"
                "排序/过滤"
                "取舍是否可解释；它不是 FinanceBench 全量质量结论。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def synthetic_cases() -> list[SyntheticCase]:
    return [
        SyntheticCase(
            case_id="smoke_3m_capex_2018",
            question="What is the FY2018 capital expenditure amount for 3M?",
            entity="3M",
            periods=("2018",),
            metric="capital_expenditure",
            expected_document_id="doc_3m_2018_10k",
            expected_page=60,
            expected_parent_id="par_3m_2018_p60",
            expected_terms=("1,577", "Purchases of property, plant and equipment"),
            local_should_terms=(
                "capital expenditures",
                "purchases of property, plant and equipment",
            ),
        ),
        SyntheticCase(
            case_id="smoke_3m_capex_compare",
            question="Did 3M spend more on capex in FY2018 than FY2017?",
            entity="3M",
            periods=("2018", "2017"),
            metric="capital_expenditure",
            expected_document_id="doc_3m_2018_10k",
            expected_page=60,
            expected_parent_id="par_3m_2018_p60",
            expected_terms=("1,577", "1,373"),
            local_should_terms=("capex", "purchases of property, plant and equipment"),
        ),
        SyntheticCase(
            case_id="smoke_apple_net_sales_2019",
            question="What were Apple net sales in 2019?",
            entity="Apple",
            periods=("2019",),
            metric="revenue",
            expected_document_id="doc_apple_2019_10k",
            expected_page=31,
            expected_parent_id="par_apple_2019_p31",
            expected_terms=("260,174", "Net sales"),
            local_should_terms=("net sales", "revenue"),
        ),
        SyntheticCase(
            case_id="smoke_msft_capex_2020",
            question="What was Microsoft FY2020 capex?",
            entity="Microsoft",
            periods=("2020",),
            metric="capital_expenditure",
            expected_document_id="doc_msft_2020_10k",
            expected_page=54,
            expected_parent_id="par_msft_2020_p54",
            expected_terms=("15,441", "Additions to property and equipment"),
            local_should_terms=("capex", "additions to property, plant and equipment"),
        ),
        SyntheticCase(
            case_id="smoke_ibm_dividends_2021",
            question="How much cash dividends did IBM pay in 2021?",
            entity="IBM",
            periods=("2021",),
            metric="dividends",
            expected_document_id="doc_ibm_2021_10k",
            expected_page=78,
            expected_parent_id="par_ibm_2021_p78",
            expected_terms=("5,869", "cash dividends paid"),
            local_should_terms=("cash dividends", "dividends paid"),
        ),
    ]


def synthetic_corpus() -> list[SyntheticChunk]:
    return [
        SyntheticChunk(
            chunk_id="chk_3m_2018_p60_capex",
            parent_id="par_3m_2018_p60",
            document_id="doc_3m_2018_10k",
            source_title="3M 2018 Form 10-K",
            company="3M",
            page_start=60,
            page_end=60,
            text=(
                "3M consolidated statement of cash flows. Purchases of property, plant and "
                "equipment were 1,577 in 2018 and 1,373 in 2017."
            ),
            parent_text=(
                "3M cash flow page. Investing activities include purchases of property, plant "
                "and equipment of 1,577 in 2018 and 1,373 in 2017, reported in millions."
            ),
            previous_page_text="3M cash flow operating activities and reconciliation.",
            next_page_text="3M notes discuss depreciation and capital investments.",
            section_title="Consolidated Statement of Cash Flows",
            metrics=("capital_expenditure",),
            periods=("2018", "2017"),
        ),
        SyntheticChunk(
            chunk_id="chk_3m_2018_p45_capital_plan",
            parent_id="par_3m_2018_p45",
            document_id="doc_3m_2018_10k",
            source_title="3M 2018 Form 10-K",
            company="3M",
            page_start=45,
            page_end=45,
            text="3M FY2018 capital expenditure plans discuss future manufacturing investments.",
            parent_text=(
                "3M management discussion mentions capital expenditure plans but gives no "
                "amount."
            ),
            section_title="MD&A",
            metrics=("capital_expenditure",),
            periods=("2018",),
        ),
        SyntheticChunk(
            chunk_id="chk_3m_2018_p40_revenue",
            parent_id="par_3m_2018_p40",
            document_id="doc_3m_2018_10k",
            source_title="3M 2018 Form 10-K",
            company="3M",
            page_start=40,
            page_end=40,
            text="3M 2018 net sales were 32,765 with operating income by business segment.",
            parent_text="3M income statement page with net sales and operating income.",
            section_title="Income Statement",
            metrics=("revenue", "operating_income"),
            periods=("2018",),
        ),
        SyntheticChunk(
            chunk_id="chk_adobe_2018_p72_capex",
            parent_id="par_adobe_2018_p72",
            document_id="doc_adobe_2018_10k",
            source_title="Adobe 2018 Form 10-K",
            company="Adobe",
            page_start=72,
            page_end=72,
            text=(
                "Adobe capital expenditures in 2018 were 264 and purchases of property "
                "and equipment increased."
            ),
            parent_text="Adobe cash flow table with capital expenditures.",
            section_title="Cash Flows",
            metrics=("capital_expenditure",),
            periods=("2018",),
        ),
        SyntheticChunk(
            chunk_id="chk_apple_2019_p31_net_sales",
            parent_id="par_apple_2019_p31",
            document_id="doc_apple_2019_10k",
            source_title="Apple 2019 Form 10-K",
            company="Apple",
            page_start=31,
            page_end=31,
            text="Apple consolidated statements of operations. Net sales were 260,174 in 2019.",
            parent_text=(
                "Apple income statement shows net sales of 260,174 in 2019, "
                "265,595 in 2018."
            ),
            section_title="Consolidated Statements of Operations",
            metrics=("revenue",),
            periods=("2019",),
        ),
        SyntheticChunk(
            chunk_id="chk_apple_2019_p45_cash",
            parent_id="par_apple_2019_p45",
            document_id="doc_apple_2019_10k",
            source_title="Apple 2019 Form 10-K",
            company="Apple",
            page_start=45,
            page_end=45,
            text="Apple cash, cash equivalents and marketable securities are discussed for 2019.",
            parent_text="Apple liquidity page with cash and marketable securities.",
            section_title="Liquidity",
            metrics=("cash_and_cash_equivalents",),
            periods=("2019",),
        ),
        SyntheticChunk(
            chunk_id="chk_msft_2020_p54_additions",
            parent_id="par_msft_2020_p54",
            document_id="doc_msft_2020_10k",
            source_title="Microsoft 2020 Form 10-K",
            company="Microsoft",
            page_start=54,
            page_end=54,
            text=(
                "Microsoft investing activities. Additions to property and equipment "
                "were 15,441 in 2020."
            ),
            parent_text=(
                "Microsoft statement of cash flows reports additions to property and equipment "
                "of 15,441 in 2020, in millions."
            ),
            previous_page_text="Microsoft cash flow operating activities.",
            next_page_text="Microsoft financing activities and dividends.",
            section_title="Statement of Cash Flows",
            metrics=("capital_expenditure",),
            periods=("2020",),
        ),
        SyntheticChunk(
            chunk_id="chk_msft_2020_p33_revenue",
            parent_id="par_msft_2020_p33",
            document_id="doc_msft_2020_10k",
            source_title="Microsoft 2020 Form 10-K",
            company="Microsoft",
            page_start=33,
            page_end=33,
            text="Microsoft revenue increased in fiscal year 2020 across cloud services.",
            parent_text="Microsoft revenue discussion for fiscal year 2020.",
            section_title="Revenue",
            metrics=("revenue",),
            periods=("2020",),
        ),
        SyntheticChunk(
            chunk_id="chk_ibm_2021_p78_dividends",
            parent_id="par_ibm_2021_p78",
            document_id="doc_ibm_2021_10k",
            source_title="IBM 2021 Form 10-K",
            company="IBM",
            page_start=78,
            page_end=78,
            text="International Business Machines cash dividends paid were 5,869 in 2021.",
            parent_text=(
                "The consolidated cash flow statement lists cash dividends paid of "
                "5,869 in 2021."
            ),
            section_title="Cash Flow",
            metrics=("dividends",),
            periods=("2021",),
        ),
        SyntheticChunk(
            chunk_id="chk_ibm_2021_p20_revenue",
            parent_id="par_ibm_2021_p20",
            document_id="doc_ibm_2021_10k",
            source_title="IBM 2021 Form 10-K",
            company="IBM",
            page_start=20,
            page_end=20,
            text="IBM revenue and hybrid cloud results are discussed for 2021.",
            parent_text="IBM management discussion of revenue.",
            section_title="Management Discussion",
            metrics=("revenue",),
            periods=("2021",),
        ),
    ]


def chunk_to_candidate(
    chunk: SyntheticChunk,
    *,
    lane: str,
    rank: int,
    score: float,
    details: dict[str, Any],
) -> Candidate:
    family = "dense" if lane == "dense" else "lexical"
    return Candidate(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        doc_name=chunk.source_title,
        source_title=chunk.source_title,
        company=chunk.company,
        text=chunk.text,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        chunk_index=rank,
        token_count=token_count(chunk.text),
        retrieved_by=(lane,),
        dense_rank=rank if family == "dense" else None,
        dense_score=score if family == "dense" else None,
        lexical_rank=rank if family == "lexical" else None,
        lexical_score=score if family == "lexical" else None,
        lexical_backend="synthetic_sparse" if family == "lexical" else None,
        final_rank=rank,
        metadata={
            "raw_chunk_id": chunk.chunk_id,
            "parent_id": chunk.parent_id,
            "metrics": list(chunk.metrics),
            "periods": list(chunk.periods),
            "lane_details": details,
        },
        section_title=chunk.section_title,
        parent_id=chunk.parent_id,
        source_type=chunk.source_type,
        lane=lane,
        retrieval_task_id=f"rt_{lane}",
        retrieval_unit_id="u0",
        lane_rank=rank,
        lane_score=score,
        lane_weight=1.0,
    )


def rank_final(candidates: Sequence[Candidate]) -> list[Candidate]:
    ranked: list[Candidate] = []
    for rank, candidate in enumerate(candidates, start=1):
        ranked.append(replace(candidate, final_rank=rank))
    return ranked


def reranker_prompt_terms(
    case: SyntheticCase,
    query: dict[str, Any],
    config: AblationConfig,
    ontology: FinanceMetricOntology,
) -> set[str]:
    if config.reranker_input == "original_query_candidate":
        return token_set(query["original_query"])
    if config.reranker_input == "current_unit_candidate":
        return token_set(query["unit_text"])
    if config.reranker_input == "local_terms_candidate":
        return token_set(
            " ".join([query["unit_text"], *query["must_have_terms"], *query["should_terms"]])
        )
    if config.reranker_input == "full_plan_summary_candidate":
        return token_set(
            " ".join(
                [
                    query["original_query"],
                    case.entity,
                    *case.periods,
                    case.metric.replace("_", " "),
                    *metric_aliases(case.metric, ontology)[:3],
                ]
            )
        )
    if config.reranker_input == "full_plan_all_units_candidate":
        other_metrics = ["revenue", "cash flow", "operating income", "assets"]
        return token_set(
            " ".join(
                [
                    query["original_query"],
                    query["unit_text"],
                    *query["must_have_terms"],
                    *query["should_terms"],
                    *metric_aliases(case.metric, ontology),
                    *other_metrics,
                ]
            )
        )
    return token_set(query["unit_text"])


def metric_aliases(metric: str, ontology: FinanceMetricOntology) -> tuple[str, ...]:
    definition = ontology.get(metric)
    if definition is None:
        return ()
    return dedupe(
        (
            definition.canonical_name.replace("_", " "),
            *definition.aliases,
            *definition.statement_hints,
        )
    )


def candidate_payload(candidate: Candidate, fallback_rank: int) -> dict[str, Any]:
    return {
        "rank": (
            candidate.final_rank
            or candidate.rerank_rank
            or candidate.fusion_rank
            or fallback_rank
        ),
        "chunk_id": candidate.chunk_id,
        "parent_id": candidate.parent_id or candidate.metadata.get("parent_id"),
        "document_id": candidate.document_id,
        "source_title": candidate.source_title,
        "company": candidate.company,
        "page_start": candidate.page_start,
        "page_end": candidate.page_end,
        "source_type": candidate.source_type,
        "retrieved_by": list(candidate.retrieved_by),
        "lane": candidate.lane,
        "fusion_rank": candidate.fusion_rank,
        "fusion_score": candidate.fusion_score,
        "rerank_rank": candidate.rerank_rank,
        "rerank_score": candidate.rerank_score,
        "token_count": candidate.token_count,
        "metadata": candidate.metadata,
    }


def config_payload(config: AblationConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "group": config.group,
        "description": config.description,
        "rewrite_policy": config.rewrite_policy,
        "filter_strategy": config.filter_strategy,
        "fusion_strategy": config.fusion_strategy,
        "candidate_shape": config.candidate_shape,
        "reranker_input": config.reranker_input,
        "top_k": config.top_k,
        "rrf_k": config.rrf_k,
        "max_context_tokens": config.max_context_tokens,
        "planned_only": config.planned_only,
    }


def empty_metrics() -> dict[str, Any]:
    return {
        "doc_hit_at": {str(k): None for k in HIT_KS},
        "page_hit_at": {str(k): None for k in HIT_KS},
        "parent_hit_at": {str(k): None for k in HIT_KS},
        "answer_terms_hit_at": {str(k): None for k in HIT_KS},
        "mrr_doc": None,
        "mrr_page": None,
        "mrr_answer_terms": None,
        "first_doc_rank": None,
        "first_page_rank": None,
        "first_parent_rank": None,
        "first_answer_terms_rank": None,
    }


def failure_reasons(
    metrics: dict[str, Any],
    dropped: Sequence[Candidate],
    case: SyntheticCase,
) -> list[str]:
    reasons: list[str] = []
    if metrics["doc_hit_at"]["3"] is False:
        reasons.append("doc_miss@3")
    if metrics["page_hit_at"]["3"] is False:
        reasons.append("page_miss@3")
    if metrics["answer_terms_hit_at"]["3"] is False:
        reasons.append("answer_terms_miss@3")
    if any(
        (
            candidate.parent_id == case.expected_parent_id
            or candidate.chunk_id == case.expected_parent_id
        )
        and candidate.metadata.get("drop_reason") == "token_budget"
        for candidate in dropped
    ):
        reasons.append("expected_evidence_dropped_by_token_budget")
    return reasons


def count_failures(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        for reason in record["failure_reasons"]:
            counts[reason] += 1
    return dict(sorted(counts.items()))


def page_matches(candidate: Candidate, expected_page: int) -> bool:
    start = candidate.page_start
    end = candidate.page_end or start
    if start is None or end is None:
        return False
    if start > end:
        start, end = end, start
    return start <= expected_page <= end


def answer_terms_match(candidate: Candidate, case: SyntheticCase) -> bool:
    text = normalize_term(
        " ".join(
            [
                candidate.text,
                str(candidate.metadata.get("parent_text", "")),
                str(candidate.metadata.get("shape", "")),
            ]
        )
    )
    return all(normalize_term(term) in text for term in case.expected_terms)


def hit_at(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def aggregate_bool(values) -> dict[str, Any]:
    items = [value for value in values if value is not None]
    if not items:
        return {"count": 0, "hits": 0, "rate": None}
    hits = sum(1 for value in items if value is True)
    return {"count": len(items), "hits": hits, "rate": hits / len(items)}


def average_metric(values) -> float | None:
    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return mean(items)


def fmt_rate(value: dict[str, Any]) -> str:
    rate = value.get("rate") if isinstance(value, dict) else None
    if rate is None:
        return "-"
    return f"{rate:.3f}"


def fmt_number(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def must_term_hits(chunk: SyntheticChunk, query: dict[str, Any]) -> int:
    searchable = normalized_blob(chunk)
    return sum(1 for term in query["must_have_terms"] if normalize_term(term) in searchable)


def normalized_blob(chunk: SyntheticChunk) -> str:
    return normalize_term(
        " ".join(
            [
                chunk.company,
                chunk.source_title,
                chunk.text,
                chunk.parent_text,
                chunk.section_title or "",
                *chunk.metrics,
                *chunk.periods,
            ]
        )
    )


def normalized_blob_text(text: str) -> str:
    return normalize_term(text)


def normalize_term(value: str) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").split())


def token_set(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_term(value))
        if token not in {"the", "and", "for", "what", "were", "was", "did", "than", "how", "much"}
    }


def token_count(value: str) -> int:
    return max(1, len(re.findall(r"\S+", value or "")))


def dedupe(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return tuple(result)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
