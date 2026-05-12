from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from atlas.core.config import Settings, known_query_providers
from atlas.ingestion.chunker import approx_token_count
from atlas.llm.clients import LLMResponse, OpenAIClient
from atlas.llm.openai_client import OpenAIAnswerGenerator
from atlas.query_orchestrator.llm_planner import LLMQueryPlanner
from atlas.query_orchestrator.ontology import FinanceMetricOntology
from atlas.query_orchestrator.service import QueryOrchestrator
from atlas.query_runtime.service import QueryRuntime
from atlas.retrieval.contracts import ProviderResult
from atlas.retrieval.models.candidate import Candidate
from atlas.retrieval.providers.base import RetrievalContext, RetrievalProvider
from atlas.retrieval.providers.sql.compiler import SQLCompiler
from atlas.retrieval.providers.sql.executor import DuckDBExecutor
from atlas.retrieval.providers.sql.provider import SQLProvider
from atlas.retrieval.router import ProviderRouter


LIVE_ENV = "ATLAS_RUN_LIVE_ACCEPTANCE"
API_KEY_ENV = "OPENAI_API_KEY"

ACCEPTANCE_QUERY = (
    "For Acme Robotics' FY2024 Vision Sensor launch, use three independent "
    "evidence sources: a hybrid text evidence branch for the exact launch "
    "constraint wording; a graph relationship context branch for the Acme "
    "Robotics -> PhotonWorks supplier dependency relationship; and a SQL revenue "
    "table branch for the FY2024 Vision Sensor total revenue. What total revenue "
    "does the table report, and what supplier dependency risk explains the launch "
    "constraint?"
)
KNOWN_PROVIDERS = ("hybrid", "graph", "sql")
RESERVED_NON_PROVIDERS = {"dense", "bm25", "table", "section", "metric_alias"}
FORBIDDEN_NON_SQL_SOURCE_TYPES = {
    "schema_card",
    "schema_routing_card",
    "table_card",
    "table_profile",
    "profile_object",
    "structured_table",
    "structured_table_fixture",
    "table_asset",
    "table",
    "card",
    "profile",
}
GOLD_TOTAL = "123456"

MANIFEST_ID = "tri_provider_acceptance_manifest_v1"
EXPECTED_SQL_TABLE_ID = "tbl_acme_fy2024_vision_sensor_revenue"
EXPECTED_SQL_COLUMN_IDS = {"col_fiscal_year", "col_product", "col_revenue"}
SAFE_REVENUE_IDENTIFIER = "revenue"
SQL_FIXTURE_ID = "tri_provider_acceptance_revenue_fixture_v1"
GRAPH_VERSION = "graph_fixture_v2026_05_tri_provider"
QDRANT_NAMESPACE_PLACEHOLDER = "synthetic-no-qdrant"
DOC_ID = "doc_acme_robotics_fy2024_launch"
CHUNK_ID = "chunk_acme_supplier_dependency"
PARENT_ID = "parent_acme_supplier_dependency"
RELATIONSHIP_ID = "rel_acme_photonworks_optical_lens_2024"

TEXT_FIXTURE_CHUNK = (
    "Acme Robotics' FY2024 Vision Sensor launch was constrained by a supplier "
    "dependency risk: the optical lens module depended on PhotonWorks as the sole "
    "qualified supplier. The launch constraint was that ramp volume could not "
    "increase until PhotonWorks completed second-source qualification and yield "
    "recovery. Relationship rel_acme_photonworks_optical_lens_2024 links Acme "
    "Robotics to PhotonWorks for the optical lens module."
)
GRAPH_SUMMARY_SENTINEL = "GRAPH_SUMMARY_SENTINEL_not_evidence_text"
GRAPH_PATH_TEXT_SENTINEL = "GRAPH_PATH_TEXT_SENTINEL_not_evidence_text"

COMBOS: tuple[tuple[str, ...], ...] = (
    ("hybrid",),
    ("graph",),
    ("sql",),
    ("hybrid", "graph"),
    ("hybrid", "sql"),
    ("graph", "sql"),
    ("hybrid", "graph", "sql"),
)
FULL_COMBO = ("hybrid", "graph", "sql")
EXPECTED_FAILURES = {
    "hybrid": ("missing_sql_result", "missing_graph_relationship_context"),
    "graph": ("missing_sql_result", "missing_hybrid_text_coverage"),
    "sql": ("missing_supplier_risk_text_evidence",),
    "hybrid+graph": ("missing_sql_result",),
    "hybrid+sql": ("missing_graph_relationship_provenance",),
    "graph+sql": ("missing_hybrid_text_coverage",),
    "hybrid+graph+sql": (),
}


class LiveAcceptanceSkipped(RuntimeError):
    pass


class AcceptanceFailure(AssertionError):
    pass


@dataclass
class _AcceptanceDB:
    added: list[Any] = field(default_factory=list)
    commits: int = 0
    rollbacks: int = 0
    cache_gets: list[tuple[Any, Any]] = field(default_factory=list)

    def add(self, value: Any) -> None:
        self.added.append(value)

    def flush(self) -> None:
        return None

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def get(self, model: Any, key: Any) -> Any:
        self.cache_gets.append((model, key))
        return None


class _SyntheticChunkProvider(RetrievalProvider):
    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name
        self.calls: list[dict[str, Any]] = []

    async def aretrieve_candidates(self, context: RetrievalContext) -> ProviderResult:
        return self.retrieve_provider_result(
            context.db,
            query=context.query,
            top_k=context.top_k,
            filters=context.filters,
            options=context.options,
            query_plan=context.query_plan,
            retrieval_tasks=context.retrieval_tasks,
        )

    def retrieve_provider_result(
        self,
        db: Any,
        *,
        query: str,
        top_k: int,
        filters: dict | None,
        options: dict,
        query_plan: Any,
        retrieval_tasks: list[Any],
    ) -> ProviderResult:
        task = retrieval_tasks[0]
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "task_ids": [item.task_id for item in retrieval_tasks],
                "unit_ids": [item.unit_id for item in retrieval_tasks],
            }
        )
        metadata = self._metadata(task)
        candidate = Candidate(
            candidate_id=f"cand_{self.provider_name}_{CHUNK_ID}",
            chunk_id=CHUNK_ID,
            document_id=DOC_ID,
            doc_name="Acme Robotics FY2024 Launch Notes",
            source_title="Acme Robotics FY2024 Vision Sensor Launch Notes",
            company="Acme Robotics",
            text=TEXT_FIXTURE_CHUNK,
            page_start=7,
            page_end=7,
            chunk_index=1,
            token_count=approx_token_count(TEXT_FIXTURE_CHUNK),
            retrieved_by=(self.provider_name,),
            dense_rank=1 if self.provider_name == "hybrid" else None,
            dense_score=0.91 if self.provider_name == "hybrid" else None,
            lexical_rank=1 if self.provider_name == "hybrid" else None,
            lexical_score=0.89 if self.provider_name == "hybrid" else None,
            fusion_rank=1,
            fusion_score=1.0,
            final_rank=1,
            metadata=metadata,
            source_uri="synthetic://acme/fy2024-launch-notes",
            section_title="Launch Constraints",
            parent_id=PARENT_ID,
            provider=self.provider_name,
            source_type="text_chunk",
            retrieval_task_id=task.task_id,
            retrieval_unit_id=task.unit_id,
        )
        return ProviderResult(
            provider=self.provider_name,
            task_id=task.task_id,
            unit_id=task.unit_id,
            status="executed",
            candidates=(candidate,),
            latency_ms=1,
            trace={
                "provider": self.provider_name,
                "status": "executed",
                "synthetic_fixture": True,
                "chunk_id": CHUNK_ID,
                **self._provider_trace(),
            },
        )

    def _metadata(self, task: Any) -> dict[str, Any]:
        source_anchor = {
            "document_id": DOC_ID,
            "chunk_id": CHUNK_ID,
            "parent_id": PARENT_ID,
            "page_start": 7,
            "page_end": 7,
            "text_span": "supplier dependency risk",
            "table_id": None,
            "cell_ids": [],
            "graph_ids": [RELATIONSHIP_ID] if self.provider_name == "graph" else [],
            "metadata": {
                "provider": self.provider_name,
                "source_type": "text_chunk",
                "manifest_id": MANIFEST_ID,
            },
        }
        metadata: dict[str, Any] = {
            "provider": self.provider_name,
            "source_type": "text_chunk",
            "source_anchor": source_anchor,
            "manifest_id": MANIFEST_ID,
            "retrieval_task_id": task.task_id,
            "retrieval_unit_id": task.unit_id,
            "retrieval_unit_ids": [task.unit_id],
            "provider_local_evidence_id": f"{self.provider_name}_local_{CHUNK_ID}",
            "rerankable": True,
            "fusion_policy": "ranked",
        }
        if self.provider_name == "graph":
            graph_metadata = {
                "graph_version": GRAPH_VERSION,
                "graph_mode": "local/path",
                "mode": "local/path",
                "relationship_id": RELATIONSHIP_ID,
                "grounded_source_chunk_ids": [CHUNK_ID],
                "graph_summary": GRAPH_SUMMARY_SENTINEL,
                "path_text": GRAPH_PATH_TEXT_SENTINEL,
            }
            metadata.update(graph_metadata)
            source_anchor["metadata"] = {**source_anchor["metadata"], **graph_metadata}
        return metadata

    def _provider_trace(self) -> dict[str, Any]:
        if self.provider_name != "graph":
            return {}
        return {
            "graph_version": GRAPH_VERSION,
            "mode": "local/path",
            "relationship_id": RELATIONSHIP_ID,
            "grounded_source_chunk_ids": [CHUNK_ID],
            "source_anchor": {
                "document_id": DOC_ID,
                "chunk_id": CHUNK_ID,
                "graph_ids": [RELATIONSHIP_ID],
                "metadata": {
                    "graph_version": GRAPH_VERSION,
                    "relationship_id": RELATIONSHIP_ID,
                    "grounded_source_chunk_ids": [CHUNK_ID],
                },
            },
        }


class _RecordingReranker:
    model_name = "recording-deterministic-reranker"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[Candidate],
        top_k: int,
        query_plan: Any = None,
        retrieval_tasks: Sequence[Any] | None = None,
        output_k: int | None = None,
    ) -> list[Candidate]:
        selected = list(candidates[:top_k])
        call = {
            "status": "executed",
            "query_plan_id": getattr(query_plan, "plan_id", None),
            "input_count": len(selected),
            "output_k": output_k,
            "input_chunk_ids": [candidate.chunk_id for candidate in selected],
            "input_source_types": [_candidate_source_type(candidate) for candidate in selected],
            "input_providers": [candidate.provider for candidate in selected],
            "input_has_sql_result": any(
                _candidate_source_type(candidate) == "sql_result" for candidate in selected
            ),
        }
        ranked: list[Candidate] = []
        for rank, candidate in enumerate(selected[: output_k or len(selected)], start=1):
            metadata = dict(candidate.metadata or {})
            metadata["reranker"] = {
                "enabled": True,
                "model": self.model_name,
                "rank": rank,
                "output_rank": rank,
                "score": float(100 - rank),
                "input_rank": candidate.final_rank or candidate.fusion_rank,
                "candidates_scored": len(selected),
                "top_n": len(selected),
                "top_m": output_k,
                "query_plan_id": getattr(query_plan, "plan_id", None),
                "retrieval_task_id": candidate.retrieval_task_id,
                "retrieval_unit_id": candidate.retrieval_unit_id,
            }
            metadata["reranker_input"] = {
                "query_plan_id": getattr(query_plan, "plan_id", None),
                "retrieval_task_id": candidate.retrieval_task_id,
                "retrieval_unit_id": candidate.retrieval_unit_id,
                "candidate_id": candidate.candidate_id,
                "chunk_id": candidate.chunk_id,
                "candidate_text_chars": len(candidate.text),
            }
            ranked.append(
                replace(
                    candidate,
                    final_rank=rank,
                    rerank_rank=rank,
                    rerank_score=float(100 - rank),
                    metadata=metadata,
                )
            )
        call["output_count"] = len(ranked)
        self.calls.append(call)
        return ranked


class _LiveRecordingClient:
    def __init__(self, settings: Settings, *, stage: str) -> None:
        self.stage = stage
        self.settings = settings
        self.calls: list[dict[str, Any]] = []
        self._client = OpenAIClient(settings)

    def create_response(self, request: dict[str, Any]) -> LLMResponse:
        self._assert_live_request(request)
        started = time.perf_counter()
        response = self._client.create_response(request)
        self.calls.append(
            {
                "stage": self.stage,
                "status": "completed",
                "model": request.get("model"),
                "reasoning_effort": _reasoning_effort(request),
                "has_json_schema": bool(
                    isinstance(request.get("text"), dict)
                    and isinstance(request["text"].get("format"), dict)
                    and request["text"]["format"].get("type") == "json_schema"
                ),
                "input_chars": len(str(request.get("input") or "")),
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "usage": _usage_payload(response.usage),
                "real_openai_response_api": True,
            }
        )
        return response

    def _assert_live_request(self, request: Mapping[str, Any]) -> None:
        model = request.get("model")
        if model != "gpt-5-nano":
            raise AcceptanceFailure(
                f"{self.stage} LLM must use gpt-5-nano; observed {model!r}"
            )
        effort = _reasoning_effort(request)
        if effort != "low":
            raise AcceptanceFailure(
                f"{self.stage} LLM must use low reasoning effort; observed {effort!r}"
            )
        if request.get("store") is not False:
            raise AcceptanceFailure(f"{self.stage} LLM request must set store=false")


class _OpenAISQLCompilerCallable:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.calls: list[dict[str, Any]] = []
        self._client = OpenAIClient(settings)

    def __call__(self, *, prompt: str, question: str, schema: Any) -> str:
        request = {
            "model": self.settings.llm_model,
            "instructions": (
                "You are the live Atlas SQL compiler acceptance caller. Return exactly one "
                "safe DuckDB SELECT statement. Use only the safe identifiers listed in the "
                "Atlas compiler prompt. Use single quotes for string literals."
            ),
            "input": (
                f"{prompt}\n\n"
                "Acceptance requirement: filter to fiscal_year = 'FY2024' and "
                "product = 'Vision Sensor' when those safe columns are available. "
                "Return SQL only."
            ),
            "max_output_tokens": 500,
            "reasoning": {"effort": self.settings.llm_reasoning_effort},
            "store": False,
        }
        started = time.perf_counter()
        response = self._client.create_response(request)
        self.calls.append(
            {
                "stage": "sql_compiler",
                "status": "completed",
                "model": request["model"],
                "reasoning_effort": _reasoning_effort(request),
                "schema_table": getattr(schema, "safe_table_name", None),
                "safe_columns": list(getattr(schema, "safe_column_names", ()) or ()),
                "input_chars": len(request["input"]),
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "usage": _usage_payload(response.usage),
                "real_openai_response_api": True,
            }
        )
        return response.output_text


@dataclass(frozen=True)
class _ComboSnapshot:
    name: str
    executable_providers: tuple[str, ...]
    settings: Settings
    result: Any | None
    details: dict[str, Any]
    db: _AcceptanceDB
    reranker_calls: list[dict[str, Any]]
    planner_calls: list[dict[str, Any]]
    answer_calls: list[dict[str, Any]]
    sql_compiler_calls: list[dict[str, Any]]
    error: str | None = None
    failure_reasons: tuple[str, ...] = ()
    assertions: tuple[str, ...] = ()


def run_acceptance(
    *,
    run_id: str | None = None,
    artifact_root: str | Path | None = None,
    combos: Sequence[tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    _ensure_live_enabled()
    _assert_static_fixture_no_leakage()

    run_id = run_id or _default_run_id()
    root = Path(artifact_root) if artifact_root is not None else (
        REPO_ROOT / "benchmarks" / "system_acceptance" / "tri_provider_full_stack"
    )
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshots: list[_ComboSnapshot] = []
    selected_combos = tuple(combos or COMBOS)
    for combo in selected_combos:
        snapshots.append(_run_combo(combo, run_dir=run_dir))

    summary = _build_summary(snapshots, run_id=run_id, run_dir=run_dir)
    redacted_trace = _build_redacted_trace(snapshots, summary)
    _write_json(run_dir / "redacted_trace.json", redacted_trace)
    _write_json(run_dir / "summary.json", summary)
    _write_report(run_dir / "report.md", summary)

    findings = _secret_scan(run_dir, api_key=os.environ.get(API_KEY_ENV))
    if findings:
        summary["success"] = False
        summary["secret_scan"] = {"status": "failed", "findings": findings}
        _write_json(run_dir / "summary.json", summary)
        _write_report(run_dir / "report.md", summary)
    else:
        summary["secret_scan"] = {"status": "passed", "findings": []}
        _write_json(run_dir / "summary.json", summary)
        _write_report(run_dir / "report.md", summary)

    return summary


def _run_combo(combo: tuple[str, ...], *, run_dir: Path) -> _ComboSnapshot:
    name = _combo_name(combo)
    settings = _settings_for_combo(combo, run_dir=run_dir)
    db = _AcceptanceDB()
    reranker = _RecordingReranker()
    planner_client: _LiveRecordingClient | None = None
    answer_client: _LiveRecordingClient | None = None
    sql_compiler: _OpenAISQLCompilerCallable | None = None
    assertions: list[str] = []
    details: dict[str, Any] = {}
    result = None
    error = None
    failure_reasons: tuple[str, ...] = ()

    try:
        planner_client = _LiveRecordingClient(settings, stage="planner")
        answer_client = _LiveRecordingClient(settings, stage="answer")
        sql_compiler = _OpenAISQLCompilerCallable(settings)
        router = _provider_router_for_combo(
            combo,
            settings=settings,
            reranker=reranker,
            sql_compiler=sql_compiler,
            run_dir=run_dir,
        )
        orchestrator = _orchestrator(settings, planner_client)
        runtime = QueryRuntime(
            settings=settings,
            provider_router=router,
            generator=OpenAIAnswerGenerator(settings, client=answer_client),
            orchestrator=orchestrator,
        )
        result = runtime.run(
            db,
            query=ACCEPTANCE_QUERY,
            top_k=5,
            filters={},
            options=_runtime_options(),
        )
        details = dict(result.details or {})
        snapshot = _ComboSnapshot(
            name=name,
            executable_providers=combo,
            settings=settings,
            result=result,
            details=details,
            db=db,
            reranker_calls=list(reranker.calls),
            planner_calls=list(planner_client.calls),
            answer_calls=list(answer_client.calls),
            sql_compiler_calls=list(sql_compiler.calls),
        )
        assertions.extend(_assert_common_contract(snapshot))
        failure_reasons = _acceptance_failure_reasons(snapshot)
        assertions.extend(_assert_expected_combo_outcome(snapshot, failure_reasons))
        if combo == FULL_COMBO:
            assertions.extend(_assert_full_stack_contract(snapshot))
    except Exception as exc:  # Keep artifact output even for live failures.
        error = _safe_error_message(exc)
    return _ComboSnapshot(
        name=name,
        executable_providers=combo,
        settings=settings,
        result=result,
        details=details,
        db=db,
        reranker_calls=list(reranker.calls),
        planner_calls=list(planner_client.calls) if planner_client else [],
        answer_calls=list(answer_client.calls) if answer_client else [],
        sql_compiler_calls=list(sql_compiler.calls) if sql_compiler else [],
        error=error,
        failure_reasons=failure_reasons,
        assertions=tuple(assertions),
    )


def _settings_for_combo(combo: tuple[str, ...], *, run_dir: Path) -> Settings:
    return Settings(
        llm_model="gpt-5-nano",
        query_planner_model="gpt-5-nano",
        llm_reasoning_effort="low",
        llm_max_output_tokens=700,
        query_planner_retry_count=2,
        query_planner_known_providers="hybrid,graph,sql",
        query_runtime_executable_providers=",".join(combo),
        sql_provider_enabled="sql" in combo,
        structured_sql_compiler_mode="llm",
        structured_sql_duckdb_dir=str(run_dir / "duckdb"),
        structured_sql_timeout_ms=5000,
        structured_sql_max_rows=20,
        structured_sql_max_result_bytes=65536,
        structured_sql_min_table_score=0.05,
        structured_sql_min_score_margin=0.0,
        structured_sql_max_candidate_tables=1,
        cache_enabled=False,
        reranker_enabled=True,
        reranker_top_k=8,
        reranker_output_k=5,
        default_top_k=5,
        max_top_k=8,
        max_context_tokens=3000,
        qdrant_collection=QDRANT_NAMESPACE_PLACEHOLDER,
    )


def _provider_router_for_combo(
    combo: tuple[str, ...],
    *,
    settings: Settings,
    reranker: _RecordingReranker,
    sql_compiler: _OpenAISQLCompilerCallable,
    run_dir: Path,
) -> ProviderRouter:
    providers: dict[str, RetrievalProvider] = {}
    if "hybrid" in combo:
        providers["hybrid"] = _SyntheticChunkProvider("hybrid")
    if "graph" in combo:
        providers["graph"] = _SyntheticChunkProvider("graph")
    if "sql" in combo:
        providers["sql"] = SQLProvider(
            settings=settings,
            compiler=SQLCompiler(llm_callable=sql_compiler, default_limit=5),
            executor=DuckDBExecutor(
                duckdb_dir=run_dir / "duckdb",
                timeout_ms=settings.structured_sql_timeout_ms,
                max_rows=settings.structured_sql_max_rows,
                max_result_bytes=settings.structured_sql_max_result_bytes,
                memory_limit=settings.structured_sql_memory_limit,
            ),
        )
    return ProviderRouter(
        providers,
        known_providers=KNOWN_PROVIDERS,
        non_executable_providers=() if "sql" in combo else ("sql",),
        reranker=reranker,
        reranker_enabled=True,
        reranker_top_k=settings.reranker_top_k,
        reranker_output_k=settings.reranker_output_k,
        max_context_tokens=settings.max_context_tokens,
    )


def _orchestrator(settings: Settings, planner_client: _LiveRecordingClient) -> QueryOrchestrator:
    ontology = FinanceMetricOntology.load(settings.finance_metric_ontology_path)
    return QueryOrchestrator(
        settings=settings,
        ontology=ontology,
        llm_planner=LLMQueryPlanner(
            settings=settings,
            ontology=ontology,
            client=planner_client,
        ),
    )


def _runtime_options() -> dict[str, Any]:
    return {
        "cache_policy": "disabled",
        "cross_provider_reranker_enabled": True,
        "structured_tables": [_structured_table_payload()],
        "acceptance_manifest_id": MANIFEST_ID,
        "return_trace": True,
    }


def _structured_table_payload() -> dict[str, Any]:
    source_uri = "synthetic://acme/fy2024-revenue-table"
    columns = [
        {
            "column_id": "col_fiscal_year",
            "name": "Fiscal Year",
            "data_type": "string",
            "semantic_role": "period",
            "source_locator": {"column_ref": "A", "locator_precision": "column"},
        },
        {
            "column_id": "col_product",
            "name": "Product",
            "data_type": "string",
            "semantic_role": "dimension",
            "source_locator": {"column_ref": "B", "locator_precision": "column"},
        },
        {
            "column_id": "col_category",
            "name": "Category",
            "data_type": "string",
            "semantic_role": "dimension",
            "source_locator": {"column_ref": "C", "locator_precision": "column"},
        },
        {
            "column_id": "col_revenue",
            "name": "Revenue ($)",
            "data_type": "number",
            "semantic_role": "measure",
            "unit": "USD",
            "source_locator": {"column_ref": "D", "locator_precision": "column"},
        },
    ]
    rows = [
        {
            "Fiscal Year": "FY2024",
            "Product": "Vision Sensor",
            "Category": "Hardware",
            "Revenue ($)": 50000,
        },
        {
            "Fiscal Year": "FY2024",
            "Product": "Vision Sensor",
            "Category": "Software",
            "Revenue ($)": 40000,
        },
        {
            "Fiscal Year": "FY2024",
            "Product": "Vision Sensor",
            "Category": "Services",
            "Revenue ($)": 33456,
        },
        {
            "Fiscal Year": "FY2024",
            "Product": "Other Product",
            "Category": "Hardware",
            "Revenue ($)": 99999,
        },
    ]
    storage_ref = {
        "mode": "synthetic_structured_table_fixture",
        "fixture_id": SQL_FIXTURE_ID,
        "artifact_id": f"artifact_{SQL_FIXTURE_ID}",
        "table_id": EXPECTED_SQL_TABLE_ID,
        "manifest_id": MANIFEST_ID,
        "backend": "in_memory_acceptance_fixture",
    }
    schema_hash = _hash_json(
        {
            "table_id": EXPECTED_SQL_TABLE_ID,
            "columns": columns,
        }
    )
    source_hash = _hash_json(
        {
            "document_id": DOC_ID,
            "source_uri": source_uri,
            "page": 12,
            "table_id": EXPECTED_SQL_TABLE_ID,
        }
    )
    artifact_hash = _hash_json(
        {
            "table_id": EXPECTED_SQL_TABLE_ID,
            "columns": columns,
            "rows": rows,
        }
    )
    manifest_snapshot = {
        "manifest_id": MANIFEST_ID,
        "status": "success",
        "orphaned": False,
        "row_count": len(rows),
        "mode": "synthetic_structured_table_fixture",
    }
    audit_metadata = {
        "manifest_id": MANIFEST_ID,
        "manifest_status": "success",
        "schema_status": "success",
        "source_status": "success",
        "artifact_status": "success",
        "orphaned": False,
        "row_count": len(rows),
        "source_hash": source_hash,
        "artifact_hash": artifact_hash,
        "schema_hash": schema_hash,
        "storage_ref": storage_ref,
        "manifest": manifest_snapshot,
        "mode": "synthetic_structured_table_fixture",
    }
    return {
        "artifact_type": "table",
        "table_id": EXPECTED_SQL_TABLE_ID,
        "table_title": "Acme Robotics FY2024 Vision Sensor Revenue Table",
        "document_id": DOC_ID,
        "source_uri": source_uri,
        "source_locator": {
            "page": 12,
            "artifact": "tri_provider_acceptance",
            "document_id": DOC_ID,
            "source_uri": source_uri,
            "table_id": EXPECTED_SQL_TABLE_ID,
            "manifest_id": MANIFEST_ID,
            **audit_metadata,
        },
        "routing_text": (
            "Acme Robotics FY2024 Vision Sensor revenue table with fiscal year, "
            "product, category, and revenue columns."
        ),
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "metadata": {
            **audit_metadata,
            "manifest_id": MANIFEST_ID,
            "source_type": "structured_table_fixture",
        },
    }


def _assert_common_contract(snapshot: _ComboSnapshot) -> list[str]:
    assertions: list[str] = []
    details = snapshot.details
    _require(not snapshot.error, f"{snapshot.name}: runtime error: {snapshot.error}")
    _require(details, f"{snapshot.name}: QueryRuntime did not return details")

    _assert_fixed_config(snapshot)
    assertions.append("fixed_config")
    _assert_plan_contract(snapshot)
    assertions.append("plan_contract")
    _assert_task_contract(snapshot)
    assertions.append("retrieval_task_contract")
    _assert_provider_status_contract(snapshot)
    assertions.append("provider_status_contract")
    _assert_cache_and_llm_calls(snapshot)
    assertions.append("cache_disabled_and_llm_calls")
    _assert_fixture_leakage(snapshot)
    assertions.append("fixture_leakage")
    return assertions


def _assert_fixed_config(snapshot: _ComboSnapshot) -> None:
    settings = snapshot.settings
    _require(settings.llm_model == "gpt-5-nano", "llm_model must be gpt-5-nano")
    _require(settings.query_planner_model == "gpt-5-nano", "planner model must be gpt-5-nano")
    _require(settings.query_planner_retry_count == 2, "planner validation retry count must be 2")
    _require(settings.llm_reasoning_effort == "low", "reasoning effort must be low")
    _require(settings.structured_sql_compiler_mode == "llm", "SQL compiler mode must be llm")
    _require(settings.cache_enabled is False, "cache must be disabled in settings")
    _require(
        tuple(known_query_providers(settings)) == KNOWN_PROVIDERS,
        f"known providers changed: {known_query_providers(settings)}",
    )
    if "sql" in snapshot.executable_providers:
        _require(settings.sql_provider_enabled is True, "SQL double opt-in flag must be true")
    else:
        _require(settings.sql_provider_enabled is False, "SQL flag must be false when sql not executable")


def _assert_plan_contract(snapshot: _ComboSnapshot) -> None:
    plan = _plan(snapshot)
    units = _plan_units(snapshot)
    providers = [unit.get("provider") for unit in units]
    _require(set(providers) == set(KNOWN_PROVIDERS), f"{snapshot.name}: plan providers {providers}")
    _require(
        not (set(providers) & RESERVED_NON_PROVIDERS),
        f"{snapshot.name}: plan leaked internal lanes as providers: {providers}",
    )
    metadata = _mapping(plan.get("metadata"))
    _require(
        tuple(metadata.get("known_providers") or ()) == KNOWN_PROVIDERS,
        f"{snapshot.name}: plan known_providers mismatch: {metadata.get('known_providers')}",
    )
    _require(
        tuple(metadata.get("executable_providers") or ()) == snapshot.executable_providers,
        (
            f"{snapshot.name}: plan executable providers mismatch: "
            f"{metadata.get('executable_providers')} != {snapshot.executable_providers}"
        ),
    )
    _require(plan.get("planner") == "llm_structured", f"{snapshot.name}: planner fallback used")
    _require("fallback_reason" not in metadata, f"{snapshot.name}: planner fallback metadata present")


def _assert_task_contract(snapshot: _ComboSnapshot) -> None:
    plan = _plan(snapshot)
    units = _plan_units(snapshot)
    tasks = _tasks(snapshot)
    _require(len(units) == len(tasks), f"{snapshot.name}: plan/task count mismatch")
    unit_by_id = {unit["unit_id"]: unit for unit in units}
    for task in tasks:
        unit = unit_by_id.get(task.get("unit_id"))
        _require(unit is not None, f"{snapshot.name}: task without matching unit {task}")
        _require(task.get("provider") == unit.get("provider"), f"{snapshot.name}: provider not preserved")
        _require(
            task.get("metadata", {}).get("purpose") == unit.get("purpose"),
            f"{snapshot.name}: purpose not preserved for {task.get('unit_id')}",
        )
        expected_metadata_filter = _expected_task_metadata_filter(plan, unit)
        observed_metadata_filter = _metadata_filter_dict(task.get("metadata_filter"))
        _require(
            observed_metadata_filter == expected_metadata_filter,
            (
                f"{snapshot.name}: metadata_filter mismatch for {task.get('unit_id')}: "
                f"expected={expected_metadata_filter!r}, observed={observed_metadata_filter!r}"
            ),
        )
    blob = json.dumps(snapshot.details, sort_keys=True, default=str)
    _require("hybrid_backfill" not in blob, f"{snapshot.name}: hybrid_backfill appeared in trace")


def _expected_task_metadata_filter(
    plan: Mapping[str, Any],
    unit: Mapping[str, Any],
) -> dict[str, Any]:
    metadata_filter: dict[str, Any] = {}
    metadata_filter.update(_metadata_filter_dict(plan.get("metadata_filter")))
    metadata_filter.update(_metadata_filter_dict(unit.get("metadata_filter")))
    return metadata_filter


def _metadata_filter_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _assert_provider_status_contract(snapshot: _ComboSnapshot) -> None:
    provider_results = _provider_results(snapshot)
    by_provider = {item.get("provider"): item for item in provider_results}
    for provider in KNOWN_PROVIDERS:
        _require(provider in by_provider, f"{snapshot.name}: missing provider result for {provider}")
    if snapshot.executable_providers == FULL_COMBO:
        for provider in KNOWN_PROVIDERS:
            status = by_provider[provider].get("status")
            _require(
                status in {"executed", "success"},
                f"full run provider {provider} must execute successfully, observed {status}",
            )
        blob = json.dumps(provider_results, sort_keys=True, default=str)
        _require("skipped_non_executable" not in blob, "full run must not skip providers")
        _require("provider_not_registered" not in blob, "full run provider missing from router")


def _assert_cache_and_llm_calls(snapshot: _ComboSnapshot) -> None:
    _require(
        snapshot.settings.cache_enabled is False,
        f"{snapshot.name}: settings cache enabled",
    )
    trace = _mapping(snapshot.details.get("trace"))
    trace_metadata = _mapping(trace.get("metadata")) if "metadata" in trace else {}
    if trace_metadata:
        cache_status = trace_metadata.get("cache_status")
        cache_policy = trace_metadata.get("cache_policy")
        _require(
            cache_policy in {"disabled", "bypassed"},
            f"{snapshot.name}: cache policy {cache_policy}",
        )
        _require(
            cache_status in {"disabled", "bypassed"},
            f"{snapshot.name}: cache status {cache_status}",
        )
    else:
        runtime_options = _runtime_options()
        _require(
            runtime_options.get("cache_policy") == "disabled",
            f"{snapshot.name}: runtime cache policy {runtime_options.get('cache_policy')}",
        )
    _require(not snapshot.db.cache_gets, f"{snapshot.name}: cache store was touched")
    _require(snapshot.planner_calls, f"{snapshot.name}: planner LLM did not call OpenAI")
    _require(snapshot.answer_calls, f"{snapshot.name}: answer LLM did not call OpenAI")
    if "sql" in snapshot.executable_providers:
        _require(snapshot.sql_compiler_calls, f"{snapshot.name}: SQL compiler LLM did not call OpenAI")
    for call in [*snapshot.planner_calls, *snapshot.answer_calls, *snapshot.sql_compiler_calls]:
        _require(call.get("real_openai_response_api") is True, f"{snapshot.name}: non-real LLM call")
        _require(call.get("model") == "gpt-5-nano", f"{snapshot.name}: wrong LLM model {call}")
        _require(call.get("reasoning_effort") == "low", f"{snapshot.name}: wrong reasoning effort {call}")


def _assert_fixture_leakage(snapshot: _ComboSnapshot) -> None:
    _assert_static_fixture_no_leakage()
    for label, payload in _non_sql_provider_and_evidence_payloads(snapshot):
        blob = json.dumps(payload, sort_keys=True, default=str)
        _require(GOLD_TOTAL not in blob, f"{snapshot.name}: gold total leaked into {label}")
        leaked_rows = [row for row in _raw_row_serializations() if row and row in blob]
        _require(not leaked_rows, f"{snapshot.name}: CSV raw SQL fixture row leaked into {label}: {leaked_rows}")
        source_types = _source_type_values(payload)
        forbidden = sorted(source_types & FORBIDDEN_NON_SQL_SOURCE_TYPES)
        _require(
            not forbidden,
            f"{snapshot.name}: structured schema/card/profile source_type leaked into {label}: {forbidden}",
        )
    schema_text = json.dumps(
        {
            "routing_text": _structured_table_payload().get("routing_text"),
            "columns": _structured_table_payload().get("columns"),
            "metadata": _structured_table_payload().get("metadata"),
            "source_locator": _structured_table_payload().get("source_locator"),
        },
        sort_keys=True,
        default=str,
    )
    _require(GOLD_TOTAL not in schema_text, "gold total leaked into schema/card text")


def _non_sql_provider_and_evidence_payloads(snapshot: _ComboSnapshot) -> list[tuple[str, Any]]:
    payloads: list[tuple[str, Any]] = []
    for result in _provider_results(snapshot):
        provider = result.get("provider")
        if provider in {"hybrid", "graph"}:
            payloads.append((f"{provider} provider trace", result))
    retrieval_trace = _mapping(snapshot.details.get("retrieval_trace"))
    for index, item in enumerate(_list_of_mappings(retrieval_trace.get("top_k")), start=1):
        if _is_non_sql_evidence_payload(item):
            payloads.append((f"non-sql retrieval evidence #{index}", item))
    evidence_pack = _mapping(snapshot.details.get("evidence_pack"))
    for index, item in enumerate(_list_of_mappings(evidence_pack.get("blocks")), start=1):
        if _is_non_sql_evidence_payload(item):
            payloads.append((f"non-sql evidence pack block #{index}", item))
    return payloads


def _is_non_sql_evidence_payload(item: Mapping[str, Any]) -> bool:
    if _source_anchor_source_type(item) == "sql_result":
        return False
    providers = set(item.get("retrieved_by") or [])
    providers.update(item.get("prompt_providers") or [])
    provider = item.get("provider")
    if provider:
        providers.add(str(provider))
    for provenance in _list_of_mappings(item.get("prompt_provider_provenance")):
        providers.add(str(provenance.get("provider_local_provider") or provenance.get("provider")))
    return bool(providers & {"hybrid", "graph"}) or item.get("chunk_id") == CHUNK_ID


def _raw_row_serializations() -> list[str]:
    table = _structured_table_payload()
    column_names = [str(column.get("name")) for column in table.get("columns") or [] if column.get("name")]
    serializations: list[str] = []
    for row in _list_of_mappings(table.get("rows")):
        values = [str(row.get(column, "")) for column in column_names]
        serializations.append(",".join(values))
        serializations.append(", ".join(values))
    return serializations


def _source_type_values(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        values = {
            str(item).strip().lower()
            for key, item in value.items()
            if str(key).lower() == "source_type" and item is not None
        }
        for child in value.values():
            values.update(_source_type_values(child))
        return values
    if isinstance(value, list | tuple):
        values: set[str] = set()
        for child in value:
            values.update(_source_type_values(child))
        return values
    return set()


def _assert_expected_combo_outcome(
    snapshot: _ComboSnapshot,
    failure_reasons: tuple[str, ...],
) -> list[str]:
    expected = EXPECTED_FAILURES[snapshot.name]
    _require(
        tuple(failure_reasons) == expected,
        f"{snapshot.name}: expected failures {expected}, observed {failure_reasons}",
    )
    if snapshot.executable_providers == FULL_COMBO:
        confidence = getattr(snapshot.result, "confidence", None)
        _require(confidence == "supported", f"full run confidence must be supported, observed {confidence}")
    return ["expected_failure_reasons"]


def _assert_full_stack_contract(snapshot: _ComboSnapshot) -> list[str]:
    assertions: list[str] = []
    _assert_sql_contract(snapshot)
    assertions.append("sql_contract")
    _assert_graph_contract(snapshot)
    assertions.append("graph_contract")
    _assert_dedupe_provenance_contract(snapshot)
    assertions.append("dedupe_provenance")
    _assert_reranker_contract(snapshot)
    assertions.append("reranker_contract")
    _assert_evidence_and_prompt_contract(snapshot)
    assertions.append("evidence_prompt_contract")
    _assert_final_answer_contract(snapshot)
    assertions.append("final_answer_contract")
    return assertions


def _assert_sql_contract(snapshot: _ComboSnapshot) -> None:
    sql_result = _provider_result(snapshot, "sql")
    _require(sql_result.get("status") == "success", f"SQL provider status {sql_result.get('status')}")
    candidate = _single_candidate(sql_result, "sql")
    payload = _mapping(candidate.get("structured_payload"))
    required = {
        "sql",
        "validated_sql",
        "dialect",
        "candidate_table_id",
        "used_column_ids",
        "safe_to_raw_identifier_map",
        "manifest_id",
        "storage_ref",
        "validation_status",
        "execution_status",
        "columns",
        "rows",
        "row_count",
        "answer_synthesis_verified",
        "table_id",
        "raw_table_name",
        "display_table_name",
        "safe_table_name",
        "used_safe_columns",
        "safe_to_raw",
        "source_anchor",
        "source_locator",
    }
    missing = sorted(required - set(payload))
    _require(not missing, f"SQL structured payload missing fields: {missing}")
    _require(payload.get("dialect") == "duckdb", f"SQL dialect mismatch: {payload.get('dialect')}")
    _require(
        payload.get("candidate_table_id") == EXPECTED_SQL_TABLE_ID
        and payload.get("table_id") == EXPECTED_SQL_TABLE_ID,
        f"SQL table id mismatch: {payload.get('candidate_table_id')}/{payload.get('table_id')}",
    )
    _require(payload.get("manifest_id") == MANIFEST_ID, f"SQL manifest mismatch: {payload.get('manifest_id')}")
    storage_ref = _mapping(payload.get("storage_ref"))
    _require(storage_ref, "SQL structured payload missing storage_ref")
    _require(
        storage_ref.get("mode") == "synthetic_structured_table_fixture"
        and storage_ref.get("fixture_id") == SQL_FIXTURE_ID
        and storage_ref.get("table_id") == EXPECTED_SQL_TABLE_ID,
        f"SQL storage_ref does not belong to this fixture: {storage_ref}",
    )
    _require(payload.get("validation_status") == "success", f"SQL validation status {payload.get('validation_status')}")
    _require(payload.get("execution_status") == "success", f"SQL execution status {payload.get('execution_status')}")
    _require(
        payload.get("answer_synthesis_verified") is False,
        f"SQL answer_synthesis_verified must be False: {payload.get('answer_synthesis_verified')}",
    )
    values = [value for row in payload.get("rows") or [] for value in _mapping(row).values()]
    _require(
        any(_numeric_equal(value, 123456) for value in values),
        f"SQL result rows do not contain {GOLD_TOTAL}: {payload.get('rows')}",
    )
    sql = str(payload.get("sql") or "")
    validated_sql = str(payload.get("validated_sql") or "")
    for label, sql_text in (("sql", sql), ("validated_sql", validated_sql)):
        _require("Revenue ($)" not in sql_text, f"{label} used raw header: {sql_text}")
        _require(
            _sql_references_identifier(sql_text, SAFE_REVENUE_IDENTIFIER),
            f"{label} did not use safe revenue identifier: {sql_text}",
        )
        _require(_sql_uses_expected_aggregation(sql_text), f"{label} missing SUM aggregation: {sql_text}")
        _require(_sql_filters_fixture_scope(sql_text), f"{label} missing FY2024/Vision Sensor filters: {sql_text}")
    used_column_ids = set(payload.get("used_column_ids") or [])
    _require(
        EXPECTED_SQL_COLUMN_IDS <= used_column_ids,
        f"SQL used_column_ids missing fixture columns: {sorted(used_column_ids)}",
    )
    safe_map = _mapping(payload.get("safe_to_raw_identifier_map"))
    revenue_mapping = _mapping(safe_map.get(SAFE_REVENUE_IDENTIFIER))
    _require(
        revenue_mapping.get("raw_source_name") == "Revenue ($)"
        and revenue_mapping.get("display_name") == "Revenue ($)",
        f"safe revenue identifier does not map to raw/display header: {revenue_mapping}",
    )
    source_anchor = _mapping(payload.get("source_anchor"))
    anchor_metadata = _mapping(source_anchor.get("metadata"))
    _require(source_anchor.get("table_id") == EXPECTED_SQL_TABLE_ID, f"SQL source_anchor table mismatch: {source_anchor}")
    _require(anchor_metadata.get("source_type") == "sql_result", f"SQL source_anchor type mismatch: {source_anchor}")
    source_locator = _mapping(payload.get("source_locator"))
    _require(source_locator.get("manifest_id") == MANIFEST_ID, f"SQL source_locator manifest mismatch: {source_locator}")
    _require(source_locator.get("artifact_hash"), f"SQL source_locator missing artifact hash: {source_locator}")
    _require(candidate.get("source_type") == "sql_result", "SQL candidate source_type must be sql_result")
    _require(candidate.get("rerankable") is False, "SQL candidate must be non-rerankable")
    _require(candidate.get("fusion_policy") == "pinned", "SQL candidate must be pinned")
    _assert_sql_provider_trace(snapshot, sql_result)


def _assert_sql_provider_trace(snapshot: _ComboSnapshot, sql_result: Mapping[str, Any]) -> None:
    trace = _mapping(sql_result.get("trace"))
    intent = _mapping(trace.get("intent"))
    _require(intent.get("allowed") is True, f"SQL intent gate not allowed: {intent}")
    _require(
        intent.get("intent_status") == "allowed" or intent.get("status") == "success",
        f"SQL intent status mismatch: {intent}",
    )
    signals = {str(signal).strip().lower() for signal in intent.get("signals") or []}
    intent_reason = str(intent.get("reason") or "").lower()
    intent_type = str(intent.get("intent_type") or "").lower()
    table_intent_proven = "table" in signals or "sql" in signals or "table" in intent_reason
    _require(table_intent_proven, f"SQL intent trace missing table signal: {intent}")
    numeric_intent_proven = bool(signals & {"total", "sum", "amount", "number", "value"}) or intent_type in {
        "aggregation",
        "filtering",
        "numeric_table_lookup",
    }
    _require(
        numeric_intent_proven,
        f"SQL intent trace missing numeric/aggregation proof: {intent}",
    )

    routing = _mapping(trace.get("schema_routing"))
    _require(
        routing.get("selected_table_id") == EXPECTED_SQL_TABLE_ID,
        f"SQL schema routing selected wrong table: {routing}",
    )
    top1_score = routing.get("top1_score")
    min_score = routing.get("min_table_score")
    _require(top1_score is not None, f"SQL schema routing missing top1_score: {routing}")
    _require(min_score is not None and float(top1_score) >= float(min_score), f"SQL routing score below threshold: {routing}")
    margin = routing.get("top1_top2_margin")
    min_margin = routing.get("min_score_margin")
    _require(
        margin is None or min_margin is None or float(margin) >= float(min_margin),
        f"SQL routing margin below threshold: {routing}",
    )
    _require(routing.get("max_candidate_tables") == 1, f"SQL routing max_candidate_tables mismatch: {routing}")

    compiler = _mapping(trace.get("compiler"))
    _require(compiler.get("compiler_mode") == "llm", f"SQL compiler did not use llm mode: {compiler}")
    compiler_model = compiler.get("model_name")
    compiler_call_model_ok = any(call.get("model") == "gpt-5-nano" for call in snapshot.sql_compiler_calls)
    _require(
        compiler_model == "gpt-5-nano" or compiler_call_model_ok,
        f"SQL compiler model not proven gpt-5-nano: trace={compiler}, calls={snapshot.sql_compiler_calls}",
    )
    _require(compiler.get("fallback_used") is False, f"SQL compiler fallback marker present: {compiler}")
    compiler_blob = json.dumps(compiler, sort_keys=True, default=str).lower()
    _require("heuristic" not in compiler_blob, f"SQL compiler trace contains heuristic marker: {compiler}")

    validator = _mapping(trace.get("validator"))
    _require(validator.get("valid") is True, f"SQL validator not valid: {validator}")
    _require(validator.get("status") == "success", f"SQL validator status mismatch: {validator}")
    checks = _mapping(validator.get("checks"))
    required_checks = {
        "single_statement",
        "select_only",
        "table_allowlist_passed",
        "column_allowlist_passed",
        "disallowed_nodes_absent",
        "external_access_absent",
        "select_star_absent",
        "join_absent",
        "subquery_absent",
    }
    failed_checks = sorted(key for key in required_checks if checks.get(key) is not True)
    _require(not failed_checks, f"SQL validator required checks failed: {failed_checks}; checks={checks}")

    execution = _mapping(trace.get("execution"))
    _require(execution.get("status") == "success", f"SQL execution trace not successful: {execution}")


def _sql_references_identifier(sql: str, identifier: str) -> bool:
    return bool(re.search(rf"\b{re.escape(identifier)}\b", sql, flags=re.IGNORECASE))


def _sql_uses_expected_aggregation(sql: str) -> bool:
    return bool(
        re.search(
            rf"\bsum\s*\(\s*(?:[a-z_][a-z0-9_]*\.)?{re.escape(SAFE_REVENUE_IDENTIFIER)}\s*\)",
            sql,
            flags=re.IGNORECASE,
        )
    )


def _sql_filters_fixture_scope(sql: str) -> bool:
    return bool(
        re.search(r"\bwhere\b", sql, flags=re.IGNORECASE)
        and _sql_has_literal_filter(sql, "fiscal_year", "FY2024")
        and _sql_has_literal_filter(sql, "product", "Vision Sensor")
    )


def _sql_has_literal_filter(sql: str, safe_column: str, literal: str) -> bool:
    return bool(
        re.search(
            rf"\b{re.escape(safe_column)}\b\s*=\s*(['\"]){re.escape(literal)}\1",
            sql,
            flags=re.IGNORECASE,
        )
    )


def _assert_graph_contract(snapshot: _ComboSnapshot) -> None:
    graph_result = _provider_result(snapshot, "graph")
    candidate = _single_candidate(graph_result, "graph")
    source_anchor = _mapping(candidate.get("source_anchor"))
    anchor_metadata = _mapping(source_anchor.get("metadata"))
    for key, expected in {
        "graph_version": GRAPH_VERSION,
        "relationship_id": RELATIONSHIP_ID,
        "mode": "local/path",
    }.items():
        _require(
            anchor_metadata.get(key) == expected,
            f"graph source_anchor {key} mismatch: {anchor_metadata}",
        )
    _require(
        CHUNK_ID in (anchor_metadata.get("grounded_source_chunk_ids") or []),
        f"graph source_anchor missing grounded chunk id: {anchor_metadata}",
    )
    _require(
        RELATIONSHIP_ID in (source_anchor.get("graph_ids") or []),
        f"graph source_anchor missing graph id: {source_anchor}",
    )
    evidence_text = _llm_text_snapshot(snapshot, CHUNK_ID)
    _require(
        _hash_text(evidence_text) == _hash_text(TEXT_FIXTURE_CHUNK),
        "graph-grounded evidence text hash must match source chunk text hash",
    )
    _require(
        GRAPH_SUMMARY_SENTINEL not in str(evidence_text)
        and GRAPH_PATH_TEXT_SENTINEL not in str(evidence_text),
        "graph summary/path_text leaked into evidence text",
    )


def _assert_dedupe_provenance_contract(snapshot: _ComboSnapshot) -> None:
    text_evidence = _text_evidence(snapshot)
    provenance = _list_of_mappings(text_evidence.get("prompt_provider_provenance"))
    providers = {item.get("provider_local_provider") or item.get("provider") for item in provenance}
    _require({"hybrid", "graph"} <= providers, f"evidence provenance lost providers: {provenance}")
    _require(text_evidence.get("prompt_deduped") is True, "hybrid/graph same chunk was not deduped")
    prompt_providers = set(text_evidence.get("prompt_providers") or [])
    _require({"hybrid", "graph"} <= prompt_providers, f"prompt providers missing: {prompt_providers}")
    pack_blocks = _list_of_mappings(_mapping(snapshot.details.get("evidence_pack")).get("blocks"))
    text_block = _find_by_chunk(pack_blocks, CHUNK_ID)
    _require(text_block, "EvidencePack missing deduped text chunk block")
    coverage = _mapping(text_block.get("coverage"))
    _require("covered_retrieval_unit_ids" in coverage, "EvidencePack block missing coverage contract")
    llm_evidence = _llm_call_evidence_records(snapshot)
    text_prompt = [item for item in llm_evidence if item.get("chunk_id") == CHUNK_ID]
    _require(text_prompt, "llm_call_evidence missing deduped text evidence")
    _require(
        "hybrid" in str(text_prompt[0].get("provider"))
        and "graph" in str(text_prompt[0].get("provider")),
        f"llm_call_evidence provider coverage lost dedupe provenance: {text_prompt}",
    )


def _assert_reranker_contract(snapshot: _ComboSnapshot) -> None:
    _require(snapshot.reranker_calls, "global reranker did not execute")
    call = snapshot.reranker_calls[0]
    _require(call.get("status") == "executed", f"reranker status not executed: {call}")
    _require(call.get("input_count", 0) > 0, f"reranker input_count must be >0: {call}")
    _require(call.get("output_count", 0) > 0, f"reranker output_count must be >0: {call}")
    _require(call.get("input_has_sql_result") is False, f"SQL result entered reranker: {call}")
    _require(
        set(call.get("input_source_types") or []) <= {"text_chunk", "parent_block"},
        f"reranker received non-text inputs: {call}",
    )
    pack_blocks = _list_of_mappings(_mapping(snapshot.details.get("evidence_pack")).get("blocks"))
    sql_block = next(
        (
            block
            for block in pack_blocks
            if _source_anchor_source_type(block) == "sql_result"
        ),
        None,
    )
    _require(sql_block is not None, "SQL pinned block missing from EvidencePack")
    _require(sql_block.get("drop_reason") in {None, ""}, f"SQL block dropped: {sql_block}")


def _assert_evidence_and_prompt_contract(snapshot: _ComboSnapshot) -> None:
    critic = _mapping(snapshot.details.get("critic"))
    pre = _mapping(critic.get("pre"))
    verification = _mapping(_mapping(pre.get("details")).get("verification"))
    _require(
        verification.get("status") == "supported",
        f"EvidenceEvaluator full run must be supported: {verification}",
    )
    _require(snapshot.details.get("llm_io", {}).get("status") == "completed", "answer LLM skipped")
    llm_evidence = _llm_call_evidence_records(snapshot)
    _require(len(llm_evidence) >= 2, f"answer prompt evidence missing: {llm_evidence}")
    for item in llm_evidence:
        _require(item.get("text_hash"), f"llm_call_evidence missing text_hash: {item}")
    redacted_keys = _nested_keys(snapshot.details.get("llm_io"))
    _require(
        {"request", "response", "raw_output", "parsed_answer"}.isdisjoint(redacted_keys),
        f"raw prompt/response leaked into runtime llm_io: {redacted_keys}",
    )


def _assert_final_answer_contract(snapshot: _ComboSnapshot) -> None:
    result = snapshot.result
    answer = str(getattr(result, "answer", "") or "")
    _require(GOLD_TOTAL in answer.replace(",", ""), f"answer missing {GOLD_TOTAL}: {answer}")
    normalized = answer.lower()
    _require(
        "supplier" in normalized
        and any(term in normalized for term in ("dependency", "sole", "single-source")),
        f"answer missing supplier dependency risk: {answer}",
    )
    _require("photonworks" in normalized or "sole qualified supplier" in normalized, f"answer vague on supplier risk: {answer}")
    citations = list(getattr(result, "citations", []) or [])
    _require(citations, f"answer has no citations: {answer}")
    citation_ids = {item.get("citation_id") for item in citations if isinstance(item, dict)}
    _require(citation_ids, f"citation ids missing: {citations}")

    evidence_by_id = {
        item.get("evidence_id"): item
        for item in _list_of_mappings(_mapping(snapshot.details.get("retrieval_trace")).get("top_k"))
    }
    cited_evidence = [evidence_by_id.get(citation_id) for citation_id in citation_ids]
    cited_evidence = [item for item in cited_evidence if item]
    _require(cited_evidence, f"citations do not map to retrieval evidence: {citations}")
    _require(
        any(_source_anchor_source_type(item) == "sql_result" for item in cited_evidence),
        f"revenue citation does not map to sql_result: {citations}",
    )
    text_citations = [
        item
        for item in cited_evidence
        if CHUNK_ID in (item.get("chunk_id"), *(item.get("chunk_ids") or []))
    ]
    _require(text_citations, f"risk citation does not map to source text: {citations}")
    provenance = _list_of_mappings(text_citations[0].get("prompt_provider_provenance"))
    _require(
        any(
            item.get("provider_local_provider") == "graph"
            and _relationship_id_in(item)
            for item in provenance
        ),
        f"risk citation lacks graph provenance: {provenance}",
    )


def _acceptance_failure_reasons(snapshot: _ComboSnapshot) -> tuple[str, ...]:
    providers = set(snapshot.executable_providers)
    reasons: list[str] = []
    has_sql = _has_sql_result(snapshot)
    has_hybrid = _has_provider_provenance(snapshot, "hybrid")
    has_graph = _has_provider_provenance(snapshot, "graph")
    has_text_risk = _has_supplier_risk_text(snapshot)

    if not has_sql:
        reasons.append("missing_sql_result")
    if providers == {"sql"} and not has_text_risk:
        reasons.append("missing_supplier_risk_text_evidence")
    elif "hybrid" not in providers and has_graph:
        reasons.append("missing_hybrid_text_coverage")
    elif "graph" not in providers and "hybrid" in providers and "sql" in providers:
        reasons.append("missing_graph_relationship_provenance")
    elif "graph" not in providers and "hybrid" in providers:
        reasons.append("missing_graph_relationship_context")
    if "hybrid" in providers and not has_hybrid:
        reasons.append("missing_hybrid_text_coverage")
    if "graph" in providers and not has_graph:
        reasons.append("missing_graph_relationship_context")
    return tuple(_dedupe(reasons))


def _has_sql_result(snapshot: _ComboSnapshot) -> bool:
    return any(
        result.get("provider") == "sql"
        and result.get("status") == "success"
        and any(
            _mapping(candidate).get("source_type") == "sql_result"
            for candidate in _list_of_mappings(result.get("candidates"))
        )
        for result in _provider_results(snapshot)
    )


def _has_provider_provenance(snapshot: _ComboSnapshot, provider: str) -> bool:
    for item in _list_of_mappings(_mapping(snapshot.details.get("retrieval_trace")).get("top_k")):
        providers = set(item.get("retrieved_by") or [])
        providers.update(item.get("prompt_providers") or [])
        for provenance in _list_of_mappings(item.get("prompt_provider_provenance")):
            providers.add(str(provenance.get("provider_local_provider") or provenance.get("provider")))
        if provider in providers:
            return True
    return False


def _has_supplier_risk_text(snapshot: _ComboSnapshot) -> bool:
    for item in _list_of_mappings(_mapping(snapshot.details.get("retrieval_trace")).get("top_k")):
        if item.get("chunk_id") == CHUNK_ID and _source_anchor_source_type(item) != "sql_result":
            return True
    return False


def _build_summary(
    snapshots: list[_ComboSnapshot],
    *,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    combo_summaries = [_combo_summary(snapshot) for snapshot in snapshots]
    success = all(item["passed"] for item in combo_summaries)
    full_snapshot = next(
        (snapshot for snapshot in snapshots if snapshot.executable_providers == FULL_COMBO),
        None,
    )
    return {
        "success": success,
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_dir": str(run_dir),
        "query": ACCEPTANCE_QUERY,
        "gold_total": GOLD_TOTAL,
        "combos": combo_summaries,
        "code_config_snapshot": _code_config_snapshot(snapshots, full_snapshot),
        "secret_scan": {"status": "pending", "findings": []},
    }


def _combo_summary(snapshot: _ComboSnapshot) -> dict[str, Any]:
    expected = EXPECTED_FAILURES.get(snapshot.name, ())
    passed = (
        snapshot.error is None
        and tuple(snapshot.failure_reasons) == expected
        and bool(snapshot.assertions)
    )
    if snapshot.executable_providers == FULL_COMBO:
        passed = passed and not snapshot.failure_reasons
    return {
        "name": snapshot.name,
        "executable_providers": list(snapshot.executable_providers),
        "expected_failure_reasons": list(expected),
        "observed_failure_reasons": list(snapshot.failure_reasons),
        "passed": passed,
        "error": snapshot.error,
        "assertions": list(snapshot.assertions),
        "details_present": bool(snapshot.details),
        "provider_results_present": bool(_provider_results_for_diagnostics(snapshot)),
        "provider_result_providers": [
            str(item.get("provider"))
            for item in _provider_results_for_diagnostics(snapshot)
            if item.get("provider") is not None
        ],
        "planner_calls": _redacted_call_summary(snapshot.planner_calls),
        "sql_compiler_calls": _redacted_call_summary(snapshot.sql_compiler_calls),
        "answer_calls": _redacted_call_summary(snapshot.answer_calls),
        "reranker_calls": snapshot.reranker_calls,
        "answer": getattr(snapshot.result, "answer", None),
        "confidence": getattr(snapshot.result, "confidence", None),
        "citations": getattr(snapshot.result, "citations", None),
    }


def _code_config_snapshot(
    snapshots: list[_ComboSnapshot],
    full_snapshot: _ComboSnapshot | None,
) -> dict[str, Any]:
    structured_fixture = _structured_fixture_snapshot()
    sql_snapshot = full_snapshot or next(
        (snapshot for snapshot in snapshots if "sql" in snapshot.executable_providers),
        None,
    )
    return {
        "git_sha": _git_sha(),
        "llm_model": "gpt-5-nano",
        "llm_reasoning_effort": "low",
        "known_providers": list(KNOWN_PROVIDERS),
        "executable_provider_matrix": {
            snapshot.name: list(snapshot.executable_providers) for snapshot in snapshots
        },
        "registry_snapshot": _registry_snapshot(),
        "graph_version": GRAPH_VERSION,
        "manifest_id": MANIFEST_ID,
        "manifest_snapshot": structured_fixture["manifest_snapshot"],
        "artifact_snapshot": structured_fixture["artifact_snapshot"],
        "structured_fixture": structured_fixture,
        "qdrant_namespace": QDRANT_NAMESPACE_PLACEHOLDER,
        "duckdb": _duckdb_snapshot(sql_snapshot),
        "cache": {"enabled": False, "policy": "disabled"},
        "sql_provider": {
            "enabled_in_full": bool(
                full_snapshot and "sql" in full_snapshot.executable_providers
            ),
            "compiler_mode": "llm",
            "double_opt_in": True,
        },
    }


def _structured_fixture_snapshot() -> dict[str, Any]:
    table_payload = _structured_table_payload()
    metadata = _mapping(table_payload.get("metadata"))
    source_locator = _mapping(table_payload.get("source_locator"))
    storage_ref = _mapping(metadata.get("storage_ref"))
    manifest_snapshot = {
        "mode": "synthetic_structured_table_fixture",
        "manifest_id": MANIFEST_ID,
        "status": metadata.get("manifest_status"),
        "orphaned": metadata.get("orphaned"),
        "row_count": len(table_payload.get("rows") or []),
        "schema_status": metadata.get("schema_status"),
        "source_status": metadata.get("source_status"),
        "artifact_status": metadata.get("artifact_status"),
        "schema_hash_present": bool(metadata.get("schema_hash")),
        "source_hash_present": bool(metadata.get("source_hash")),
        "artifact_hash_present": bool(metadata.get("artifact_hash")),
        "structured_artifact_writer_live": False,
        "postgres_live_ingestion_proof": False,
        "qdrant_live_ingestion_proof": False,
    }
    artifact_snapshot = {
        "mode": "synthetic_structured_table_fixture",
        "artifact_id": storage_ref.get("artifact_id"),
        "fixture_id": storage_ref.get("fixture_id"),
        "table_id": table_payload.get("table_id"),
        "storage_ref": storage_ref,
        "row_count": len(table_payload.get("rows") or []),
        "schema_hash": metadata.get("schema_hash"),
        "source_hash": metadata.get("source_hash"),
        "artifact_hash": metadata.get("artifact_hash"),
        "schema_hash_present": bool(metadata.get("schema_hash")),
        "source_hash_present": bool(metadata.get("source_hash")),
        "artifact_hash_present": bool(metadata.get("artifact_hash")),
        "source_locator_has_hashes": all(
            bool(source_locator.get(key))
            for key in ("schema_hash", "source_hash", "artifact_hash")
        ),
        "structured_artifact_writer_live": False,
        "postgres_live_ingestion_proof": False,
        "qdrant_live_ingestion_proof": False,
    }
    return {
        "mode": "synthetic_structured_table_fixture",
        "claim": "contract_wiring_fixture_not_live_ingestion_proof",
        "manifest_snapshot": manifest_snapshot,
        "artifact_snapshot": artifact_snapshot,
    }


def _duckdb_snapshot(snapshot: _ComboSnapshot | None) -> dict[str, Any]:
    table_payload = _structured_table_payload()
    base = {
        "schema_hash": _hash_json(
            {
                "table_id": table_payload["table_id"],
                "columns": table_payload["columns"],
                "rows": table_payload["rows"],
            }
        ),
    }
    if snapshot is None:
        return {
            **base,
            "status": "not_run",
            "path": None,
            "path_present": False,
            "timeout_isolation": None,
        }

    sql_result = _provider_result_for_diagnostics(snapshot, "sql")
    provider_results = _provider_results_for_diagnostics(snapshot)
    if not sql_result:
        return {
            **base,
            "status": "missing_sql_provider_result",
            "combo": snapshot.name,
            "path": None,
            "path_present": False,
            "timeout_isolation": None,
            "provider_results_present": bool(provider_results),
            "provider_result_providers": [
                str(item.get("provider"))
                for item in provider_results
                if item.get("provider") is not None
            ],
            "error": snapshot.error,
        }

    execution = _mapping(_mapping(sql_result.get("trace")).get("execution"))
    path = execution.get("duckdb_path")
    status = execution.get("status") or sql_result.get("status") or "missing_execution_trace"
    return {
        **base,
        "status": status,
        "path": path,
        "path_present": bool(path and Path(str(path)).exists()),
        "timeout_isolation": execution.get("timeout_isolation"),
    }


def _build_redacted_trace(
    snapshots: list[_ComboSnapshot],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": summary["run_id"],
        "generated_at": summary["generated_at"],
        "redaction": {
            "raw_prompts_saved": False,
            "raw_request_response_saved": False,
            "llm_call_evidence_text_snapshot": "redacted",
        },
        "runs": {
            snapshot.name: {
                "executable_providers": list(snapshot.executable_providers),
                "error": snapshot.error,
                "failure_reasons": list(snapshot.failure_reasons),
                "details": _redact(snapshot.details),
                "query_plan": _redact(snapshot.details.get("query_plan")),
                "retrieval_tasks": _redact(snapshot.details.get("retrieval_tasks")),
                "provider_router_trace": _redact(snapshot.details.get("provider_router_trace")),
                "provider_results": _redact(snapshot.details.get("provider_results")),
                "retrieval_trace": _redact(snapshot.details.get("retrieval_trace")),
                "evidence_pack": _redact(snapshot.details.get("evidence_pack")),
                "critic": _redact(snapshot.details.get("critic")),
                "llm_io": _redact(snapshot.details.get("llm_io")),
                "llm_calls": _redacted_db_llm_calls(snapshot),
                "llm_call_evidence": _redacted_db_llm_evidence(snapshot),
                "reranker_calls": snapshot.reranker_calls,
            }
            for snapshot in snapshots
        },
    }


def _redacted_db_llm_calls(snapshot: _ComboSnapshot) -> list[dict[str, Any]]:
    records = []
    for item in snapshot.db.added:
        if getattr(item, "__tablename__", None) != "llm_calls":
            continue
        records.append(
            {
                "call_id": getattr(item, "call_id", None),
                "stage": getattr(item, "stage", None),
                "status": getattr(item, "status", None),
                "model_name": getattr(item, "model_name", None),
                "validation_status": getattr(item, "validation_status", None),
                "parsed_plan_id": getattr(item, "parsed_plan_id", None),
                "parsed_confidence": getattr(item, "parsed_confidence", None),
                "latency_ms": getattr(item, "latency_ms", None),
                "input_tokens": getattr(item, "input_tokens", None),
                "output_tokens": getattr(item, "output_tokens", None),
                "raw_payload_hash": getattr(item, "raw_payload_hash", None),
                "request": "[redacted]",
                "response": "[redacted]",
            }
        )
    return records


def _redacted_db_llm_evidence(snapshot: _ComboSnapshot) -> list[dict[str, Any]]:
    return [
        {
            "record_id": item.get("record_id"),
            "call_id": item.get("call_id"),
            "evidence_id": item.get("evidence_id"),
            "rank": item.get("rank"),
            "provider": item.get("provider"),
            "chunk_id": item.get("chunk_id"),
            "document_id": item.get("document_id"),
            "text_hash": item.get("text_hash"),
            "text_snapshot": "[redacted]",
        }
        for item in _llm_call_evidence_records(snapshot)
    ]


def _redact(value: Any) -> Any:
    sensitive_keys = {
        "authorization",
        "api_key",
        "openai_api_key",
        "prompt",
        "request",
        "response",
        "messages",
        "completion",
        "raw_prompt",
        "raw_request",
        "raw_response",
        "request_json",
        "response_json",
        "instructions",
        "input",
        "input_text",
        "instructions_text",
        "output_text",
        "raw_output",
        "raw_output_text",
        "parsed_answer_text",
        "text_snapshot",
    }
    if isinstance(value, Mapping):
        redacted = {}
        for key, item in value.items():
            if str(key).lower() in sensitive_keys:
                redacted[str(key)] = "[redacted]"
            else:
                redacted[str(key)] = _redact(item)
        return redacted
    if isinstance(value, list | tuple):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _redact_secret_literals(value)
    return value


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Atlas Tri-Provider Full-Stack Acceptance",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Generated: `{summary['generated_at']}`",
        f"- Success: `{summary['success']}`",
        f"- Artifact dir: `{summary['artifact_dir']}`",
        f"- Acceptance query: `{summary['query']}`",
        "",
        "## Scope / Non-Claims",
        "",
        "- Scope: synthetic fixture acceptance for contract wiring, provider isolation, evidence coverage, and citation trace behavior.",
        "- Non-claims: this is not a FinanceBench benchmark, GraphRAG retrieval eval, Text-to-SQL benchmark, multi-table SQL proof, or general answer reliability proof.",
        "- Structured table mode: `synthetic_structured_table_fixture`; this report does not claim Postgres/Qdrant live ingestion proof.",
        f"- Canonical full provider order: `{'+'.join(FULL_COMBO)}`.",
        "",
        "## Combos",
        "",
        "| Combo | Executable providers | Expected failures | Observed failures | Passed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for combo in summary["combos"]:
        lines.append(
            "| `{name}` | `{providers}` | `{expected}` | `{observed}` | `{passed}` |".format(
                name=combo["name"],
                providers=",".join(combo["executable_providers"]),
                expected=",".join(combo["expected_failure_reasons"]) or "none",
                observed=",".join(combo["observed_failure_reasons"]) or "none",
                passed=combo["passed"],
            )
        )
    lines.extend(
        [
            "",
            "## Secret Scan",
            "",
            f"- Status: `{summary.get('secret_scan', {}).get('status')}`",
        ]
    )
    findings = summary.get("secret_scan", {}).get("findings") or []
    for finding in findings:
        lines.append(f"- `{finding}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _secret_scan(root: Path, *, api_key: str | None) -> list[str]:
    findings: list[str] = []
    literal_needles = ["Authorization", "Bearer", "sk-"]
    if api_key:
        literal_needles.append(api_key)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle in literal_needles:
            if needle and needle in text:
                findings.append(f"{path.relative_to(root)} contains forbidden literal {needle!r}")
        if re.search(r'"(?:request_json|response_json)"\s*:', text):
            findings.append(f"{path.relative_to(root)} contains raw request_json/response_json key")
    return findings


def _ensure_live_enabled() -> None:
    if os.environ.get(LIVE_ENV) != "1":
        raise LiveAcceptanceSkipped(f"set {LIVE_ENV}=1 to run live tri-provider acceptance")
    if not os.environ.get(API_KEY_ENV):
        raise LiveAcceptanceSkipped(f"set {API_KEY_ENV} in the environment for live acceptance")


def _assert_static_fixture_no_leakage() -> None:
    fixture_blob = "\n".join(
        [
            TEXT_FIXTURE_CHUNK,
            GRAPH_SUMMARY_SENTINEL,
            GRAPH_PATH_TEXT_SENTINEL,
        ]
    )
    _require(GOLD_TOTAL not in fixture_blob, "gold total leaked into text/graph fixtures")
    table = _structured_table_payload()
    for row in table["rows"]:
        _require(
            all(str(value) != GOLD_TOTAL for value in row.values()),
            "individual SQL fixture row leaked the gold total",
        )
    schema_blob = json.dumps(
        {
            "table_title": table["table_title"],
            "routing_text": table["routing_text"],
            "columns": table["columns"],
            "metadata": table["metadata"],
            "source_locator": table["source_locator"],
        },
        sort_keys=True,
        default=str,
    )
    _require(GOLD_TOTAL not in schema_blob, "schema/card text leaked the gold total")


def _plan(snapshot: _ComboSnapshot) -> dict[str, Any]:
    plan = _mapping(snapshot.details.get("query_plan"))
    _require(plan, f"{snapshot.name}: missing query_plan")
    return plan


def _plan_units(snapshot: _ComboSnapshot) -> list[dict[str, Any]]:
    return _list_of_mappings(_plan(snapshot).get("retrieval_units"))


def _tasks(snapshot: _ComboSnapshot) -> list[dict[str, Any]]:
    tasks = _list_of_mappings(snapshot.details.get("retrieval_tasks"))
    _require(tasks, f"{snapshot.name}: missing retrieval_tasks")
    return tasks


def _provider_results(snapshot: _ComboSnapshot) -> list[dict[str, Any]]:
    results = _list_of_mappings(snapshot.details.get("provider_results"))
    _require(results, f"{snapshot.name}: missing provider_results")
    return results


def _provider_results_for_diagnostics(snapshot: _ComboSnapshot) -> list[dict[str, Any]]:
    return _list_of_mappings(snapshot.details.get("provider_results"))


def _provider_result(snapshot: _ComboSnapshot, provider: str) -> dict[str, Any]:
    for result in _provider_results(snapshot):
        if result.get("provider") == provider:
            return result
    raise AcceptanceFailure(f"{snapshot.name}: missing provider result {provider}")


def _provider_result_for_diagnostics(
    snapshot: _ComboSnapshot,
    provider: str,
) -> dict[str, Any]:
    for result in _provider_results_for_diagnostics(snapshot):
        if result.get("provider") == provider:
            return result
    return {}


def _single_candidate(result: Mapping[str, Any], provider: str) -> dict[str, Any]:
    candidates = _list_of_mappings(result.get("candidates"))
    _require(candidates, f"{provider}: missing provider candidate")
    return candidates[0]


def _text_evidence(snapshot: _ComboSnapshot) -> dict[str, Any]:
    for item in _list_of_mappings(_mapping(snapshot.details.get("retrieval_trace")).get("top_k")):
        if item.get("chunk_id") == CHUNK_ID:
            return item
    raise AcceptanceFailure("missing text evidence in retrieval_trace")


def _find_by_chunk(items: Sequence[Mapping[str, Any]], chunk_id: str) -> dict[str, Any]:
    for item in items:
        chunk_ids = item.get("chunk_ids") or []
        if item.get("chunk_id") == chunk_id or chunk_id in chunk_ids:
            return dict(item)
    return {}


def _source_anchor_source_type(item: Mapping[str, Any]) -> str | None:
    source_anchor = _mapping(item.get("source_anchor"))
    return _mapping(source_anchor.get("metadata")).get("source_type")


def _relationship_id_in(item: Mapping[str, Any]) -> bool:
    if item.get("relationship_id") == RELATIONSHIP_ID:
        return True
    source_anchor = _mapping(item.get("source_anchor"))
    metadata = _mapping(source_anchor.get("metadata"))
    return metadata.get("relationship_id") == RELATIONSHIP_ID


def _llm_call_evidence_records(snapshot: _ComboSnapshot) -> list[dict[str, Any]]:
    records = []
    for item in snapshot.db.added:
        if getattr(item, "__tablename__", None) != "llm_call_evidence":
            continue
        records.append(
            {
                "record_id": getattr(item, "record_id", None),
                "call_id": getattr(item, "call_id", None),
                "evidence_id": getattr(item, "evidence_id", None),
                "rank": getattr(item, "rank", None),
                "provider": getattr(item, "provider", None),
                "chunk_id": getattr(item, "chunk_id", None),
                "document_id": getattr(item, "document_id", None),
                "text_hash": getattr(item, "text_hash", None),
                "text_snapshot": getattr(item, "text_snapshot", None),
            }
        )
    return records


def _llm_text_snapshot(snapshot: _ComboSnapshot, chunk_id: str) -> str | None:
    for item in _llm_call_evidence_records(snapshot):
        if item.get("chunk_id") == chunk_id and item.get("text_snapshot"):
            return str(item["text_snapshot"])
    return None


def _candidate_source_type(candidate: Candidate) -> str:
    return str(
        dict(candidate.metadata or {}).get("source_type")
        or getattr(candidate, "source_type", None)
        or ""
    )


def _numeric_equal(value: Any, expected: int) -> bool:
    try:
        return int(float(str(value).replace(",", ""))) == expected
    except (TypeError, ValueError):
        return False


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise AcceptanceFailure(message)


def _hash_text(text: Any) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _combo_name(combo: tuple[str, ...]) -> str:
    return "+".join(combo)


def _parse_combo_arg(value: str) -> tuple[str, ...]:
    normalized = re.sub(r"\s*\+\s*", "+", str(value).strip().replace(",", "+"))
    for combo in COMBOS:
        if normalized == _combo_name(combo):
            return combo
    valid = ", ".join(_combo_name(combo) for combo in COMBOS)
    raise argparse.ArgumentTypeError(f"unknown combo {value!r}; expected one of: {valid}")


def _default_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"tri_provider_{timestamp}_{_git_sha(short=True)}"


def _git_sha(*, short: bool = False) -> str:
    args = ["git", "rev-parse", "--short" if short else "HEAD"]
    try:
        return subprocess.check_output(
            args,
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _registry_snapshot() -> dict[str, Any]:
    try:
        from atlas.retrieval.providers.registry import provider_registry

        return {"providers": sorted(provider_registry.names)}
    except Exception as exc:
        return {"providers": [], "error": exc.__class__.__name__}


def _usage_payload(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        try:
            payload = usage.model_dump()
            if isinstance(payload, dict):
                return {
                    key: payload.get(key)
                    for key in ("input_tokens", "output_tokens", "total_tokens")
                    if payload.get(key) is not None
                }
        except Exception:
            pass
    payload = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            payload[key] = value
    return payload


def _reasoning_effort(request: Mapping[str, Any]) -> str | None:
    reasoning = request.get("reasoning")
    if isinstance(reasoning, Mapping):
        value = reasoning.get("effort")
        return str(value) if value is not None else None
    return None


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        keys = {str(key).lower() for key in value}
        for child in value.values():
            keys.update(_nested_keys(child))
        return keys
    if isinstance(value, list | tuple):
        keys: set[str] = set()
        for child in value:
            keys.update(_nested_keys(child))
        return keys
    return set()


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _redacted_call_summary(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {
        "stage",
        "status",
        "model",
        "reasoning_effort",
        "has_json_schema",
        "schema_table",
        "safe_columns",
        "input_chars",
        "latency_ms",
        "usage",
        "real_openai_response_api",
    }
    return [{key: value for key, value in call.items() if key in allowed} for call in calls]


def _safe_error_message(exc: BaseException) -> str:
    return _redact_secret_literals(f"{exc.__class__.__name__}: {exc}")


def _redact_secret_literals(text: str) -> str:
    api_key = os.environ.get(API_KEY_ENV)
    if api_key:
        text = text.replace(api_key, "[redacted-api-key]")
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted-openai-key]", text)
    text = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "[redacted-bearer-token]", text)
    return text


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, SimpleNamespace):
        return vars(value)
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Atlas tri-provider live acceptance harness")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument(
        "--combo",
        default=None,
        type=_parse_combo_arg,
        help="Run only one provider combo, for example hybrid+graph+sql.",
    )
    args = parser.parse_args(argv)

    try:
        summary = run_acceptance(
            run_id=args.run_id,
            artifact_root=args.artifact_root,
            combos=(args.combo,) if args.combo else None,
        )
    except LiveAcceptanceSkipped as exc:
        print(f"SKIPPED: {exc}")
        return 0

    print(json.dumps({"success": summary["success"], "artifact_dir": summary["artifact_dir"]}))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
