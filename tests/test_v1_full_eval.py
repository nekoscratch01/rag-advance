from __future__ import annotations

from datetime import UTC, datetime

from atlas.benchmark.financebench import build_failure_buckets, build_report, build_summary
from atlas.benchmark.financebench import evaluate_case_result
from atlas.eval.v1_full import build_v1_record_metrics, summarize_v1_records
from atlas.eval.service import EvalCase


def test_v1_record_metrics_cover_design_components_from_trace() -> None:
    record = {
        "answer": "3M FY2018 capital expenditures were 1,577 million [c1].",
        "confidence": "supported",
        "latency_ms": 120,
        "wall_latency_ms": 130,
        "cache_hit": False,
        "retrieval": {"doc_hit_at": {"10": True}, "page_hit_at": {"10": True}},
        "citation": {"citation_doc_hit_at": {"10": True}, "citation_page_hit_at": {"10": True}},
        "answer_metrics": {"answer_numeric_match": True},
        "trace": {
            "result": {
                "answer": "3M FY2018 capital expenditures were 1,577 million [c1].",
                "confidence": "supported",
                "details": {
                    "plan_latency_ms": 5,
                    "query_plan": {
                        "plan_id": "plan_1",
                        "planner": "fallback",
                        "query_type": "financial_numeric_fact",
                        "risk_flags": [],
                        "retrieval_units": [{"unit_id": "u0"}],
                    },
                    "retrieval_tasks": [
                        {
                            "task_id": "rt_1",
                            "unit_id": "u0",
                            "lanes": ["dense", "bm25", "table"],
                        }
                    ],
                    "retrieval_trace": {
                        "top_k": [
                            {
                                "evidence_id": "c1",
                                "chunk_id": "chk_1",
                                "provider": "text_hybrid",
                                "lane": "multi_lane",
                                "lanes": ["dense", "bm25", "table"],
                                "fusion_score": 0.2,
                                "lane_contributions": [
                                    {"lane": "dense", "weighted_contribution": 0.01},
                                    {"lane": "bm25", "weighted_contribution": 0.02},
                                    {"lane": "table", "weighted_contribution": 0.03},
                                ],
                                "reranker": {
                                    "model": "fake-reranker",
                                    "latency_ms": 7,
                                    "top_n": 3,
                                    "top_m": 1,
                                },
                                "text_hybrid_provider": {
                                    "provider": "text_hybrid",
                                    "retrieval_latency_ms": 10,
                                    "lanes": [
                                        {"lane": "dense", "latency_ms": 1},
                                        {"lane": "bm25", "latency_ms": 2},
                                        {"lane": "table", "latency_ms": 3},
                                    ],
                                },
                                "evidence_pack": {
                                    "pack_id": "ep_1",
                                    "block_count": 1,
                                    "dropped_block_count": 1,
                                    "token_count": 50,
                                    "max_context_tokens": 100,
                                    "dropped_blocks": [{"drop_reason": "token_budget"}],
                                },
                                "coverage": {
                                    "retrieval_unit_ids": ["u0"],
                                    "entities": {"missing": []},
                                    "periods": {"missing": []},
                                    "metrics": {"missing": []},
                                },
                                "included_in_prompt": True,
                            }
                        ]
                    },
                    "critic": {
                        "evidence_evaluation": {"status": "supported"},
                        "citation_verification": {"status": "supported"},
                    },
                },
            },
            "latency": {
                "total_latency_ms": 120,
                "cache_latency_ms": 1,
                "retrieval_latency_ms": 10,
                "generation_latency_ms": 20,
            },
            "generation": [{"input_tokens": 100, "output_tokens": 20}],
            "cache": {"hit": False},
            "v1_trace": {
                "query_plans": [{"payload": {"plan_id": "plan_1"}}],
                "retrieval_tasks": [{"payload": {"task_id": "rt_1"}}],
                "retrieval_results": [{"payload": {"status": "completed"}}],
            },
        },
    }

    metrics = build_v1_record_metrics(record)

    assert metrics["components"]["query_orchestrator"]["present"] is True
    assert metrics["components"]["retrieval_plan"]["task_count"] == 1
    assert metrics["components"]["text_hybrid_provider"]["lanes_seen"] == [
        "dense",
        "bm25",
        "table",
    ]
    assert metrics["components"]["table_lane"]["present"] is True
    assert metrics["components"]["provider_local_weighted_rrf"]["present"] is True
    assert metrics["components"]["reranker"]["model"] == "fake-reranker"
    assert metrics["evidence"]["selected_block_count"] == 1
    assert metrics["evidence"]["dropped_block_count"] == 1
    assert metrics["latency"]["reranker_latency_ms"] == 7
    assert metrics["failure_buckets"] == []


def test_v1_summary_aggregates_component_presence_and_failures() -> None:
    record = {
        "answer": "",
        "confidence": "insufficient",
        "retrieval": {"doc_hit_at": {"10": False}, "page_hit_at": {"10": False}},
        "citation": {"citation_doc_hit_at": {"10": False}, "citation_page_hit_at": {"10": False}},
        "answer_metrics": {"answer_numeric_match": False},
        "trace": {},
        "error": None,
    }
    record["v1_metrics"] = build_v1_record_metrics(record)

    summary = summarize_v1_records([record])

    assert summary["component_presence"]["query_orchestrator"]["rate"] == 0.0
    assert summary["failure_buckets"]["retrieval_doc_miss"] == 1
    assert summary["failure_buckets"]["citation_page_miss"] == 1
    assert summary["failure_buckets"]["answer_numeric_miss"] == 1
    assert "query_orchestrator_missing" not in summary["failure_buckets"]


def test_financebench_report_includes_v1_component_benchmark_smoke() -> None:
    case = EvalCase(
        id="case_1",
        question="What was 3M FY2018 capex?",
        expected_answer="1,577 million",
        expected_sources=["3M 2018 10-K"],
        expected_evidence=[
            {
                "source_title": "3M 2018 10-K",
                "page_start": 60,
                "page_end": 60,
            }
        ],
    )
    trace = {
        "retrieval": {
            "top_k": [
                {
                    "rank": 1,
                    "chunk_id": "chk_1",
                    "document_id": "doc_1",
                    "source_title": "3M 2018 10-K",
                    "page_start": 60,
                    "page_end": 60,
                    "retriever_type": "hybrid",
                }
            ]
        },
        "result": {
            "details": {
                "plan_latency_ms": 5,
                "query_plan": {
                    "plan_id": "plan_1",
                    "planner": "fallback",
                    "retrieval_units": [{"unit_id": "u0"}],
                },
                "retrieval_tasks": [
                    {
                        "task_id": "rt_1",
                        "unit_id": "u0",
                        "lanes": ["dense", "bm25", "table"],
                    }
                ],
                "retrieval_trace": {
                    "top_k": [
                        {
                            "evidence_id": "c1",
                            "chunk_id": "chk_1",
                            "provider": "text_hybrid",
                            "lanes": ["dense", "bm25", "table"],
                            "fusion_score": 0.2,
                            "lane_contributions": [{"lane": "dense"}],
                            "reranker": {
                                "model": "fake-reranker",
                                "latency_ms": 7,
                                "top_n": 3,
                                "top_m": 1,
                            },
                            "evidence_pack": {
                                "pack_id": "ep_1",
                                "block_count": 1,
                                "dropped_block_count": 0,
                            },
                            "coverage": {"retrieval_unit_ids": ["u0"]},
                            "included_in_prompt": True,
                        }
                    ]
                },
                "critic": {
                    "evidence_evaluation": {"status": "supported"},
                    "citation_verification": {"status": "supported"},
                },
            }
        },
        "latency": {
            "total_latency_ms": 120,
            "cache_latency_ms": 1,
            "retrieval_latency_ms": 10,
            "generation_latency_ms": 20,
        },
        "generation": [{"input_tokens": 100, "output_tokens": 20}],
        "cache": {"hit": False},
    }
    record = evaluate_case_result(
        case=case,
        mode="hybrid_rrf_reranker",
        phase="cold",
        cache_policy="off",
        payload={
            "query_id": "q_1",
            "trace_id": "tr_1",
            "answer": "3M FY2018 capital expenditures were 1,577 million [c1].",
            "confidence": "supported",
            "citations": [
                {
                    "citation_id": "c1",
                    "source_title": "3M 2018 10-K",
                    "page_start": 60,
                    "page_end": 60,
                }
            ],
        },
        trace=trace,
        latency_ms=130,
        request={},
    )
    now = datetime.now(UTC)
    summary = build_summary(
        [record],
        run_id="smoke",
        cases_path="evals/financebench_cases.yaml",
        base_url="http://localhost:8000",
        modes=["hybrid_rrf_reranker"],
        top_k=10,
        cache_policy="off",
        warm_cache=False,
        timeout_seconds=120,
        started_at=now,
        finished_at=now,
    )
    report = build_report(summary, build_failure_buckets([record]))

    assert "V1 Component Benchmarks" in report
    assert "`query_orchestrator`" in report
    assert summary["mode_results"]["hybrid_rrf_reranker:cold"]["v1_components"][
        "component_presence"
    ]["reranker"]["rate"] == 1.0
