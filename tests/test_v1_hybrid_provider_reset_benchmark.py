from __future__ import annotations

import json

from atlas.benchmark.v1_hybrid_provider_reset import run_provider_reset_smoke


def test_provider_reset_smoke_writes_artifacts_and_required_groups(tmp_path) -> None:
    run = run_provider_reset_smoke(output_dir=tmp_path, run_id="smoke_test")

    assert (run.output_dir / "summary.json").exists()
    assert (run.output_dir / "cases.jsonl").exists()
    assert (run.output_dir / "report.md").exists()

    summary = json.loads((run.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["groups"]) == {
        "query_rewrite",
        "filter_strategy",
        "fusion",
        "candidate_shape",
        "reranker_input",
    }
    assert "rewrite_ontology_aliases" in summary["variant_results"]
    assert "filter_must_terms_sparse_boost" in summary["variant_results"]
    assert "fusion_python_weighted_rrf" in summary["variant_results"]
    assert "shape_page_neighborhood" in summary["variant_results"]
    assert "rerank_full_plan_all_units_candidate" in summary["variant_results"]


def test_provider_reset_smoke_marks_qdrant_rrf_as_planned(tmp_path) -> None:
    run = run_provider_reset_smoke(output_dir=tmp_path, run_id="smoke_test")
    result = run.summary["variant_results"]["fusion_qdrant_rrf_planned"]

    assert result["completed_cases"] == 0
    assert result["planned_cases"] == 5
    assert result["failure_counts"] == {"planned_not_run": 5}


def test_ontology_alias_rewrite_does_not_regress_smoke_page_recall(tmp_path) -> None:
    run = run_provider_reset_smoke(output_dir=tmp_path, run_id="smoke_test")
    unit_text = run.summary["variant_results"]["rewrite_unit_text"]["metrics"]["page_hit@3"]
    ontology_aliases = run.summary["variant_results"]["rewrite_ontology_aliases"]["metrics"][
        "page_hit@3"
    ]

    assert ontology_aliases["rate"] >= unit_text["rate"]


def test_provider_reset_smoke_reports_answer_term_coverage(tmp_path) -> None:
    run = run_provider_reset_smoke(output_dir=tmp_path, run_id="smoke_test")
    metrics = run.summary["variant_results"]["filter_must_have_hard"]["metrics"]

    assert "answer_terms_hit@3" in metrics
    assert metrics["answer_terms_hit@3"]["rate"] == 0.8
    assert run.summary["variant_results"]["filter_must_have_hard"]["failure_counts"][
        "answer_terms_miss@3"
    ] == 1
