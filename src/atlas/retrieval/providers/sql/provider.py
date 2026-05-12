from __future__ import annotations

import time
from typing import Any

from sqlalchemy.orm import Session

from atlas.core.config import Settings
from atlas.query_orchestrator.schema import QueryPlan
from atlas.retrieval.contracts import ProviderResult
from atlas.retrieval.models.retrieval_task import RetrievalTask
from atlas.retrieval.providers.base import RetrievalContext, RetrievalProvider
from atlas.retrieval.providers.sql.compiler import OpenAIResponsesSQLCompiler, SQLCompiler
from atlas.retrieval.providers.sql.evidence import build_sql_result_evidence
from atlas.retrieval.providers.sql.executor import DuckDBExecutor, SQLExecutionTimeout
from atlas.retrieval.providers.sql.identifiers import IdentifierNormalizer
from atlas.retrieval.providers.sql.intent import SQLIntentGate
from atlas.retrieval.providers.sql.models import (
    SQL_PROVIDER,
    SQL_PROVIDER_VERSION,
    SQLDraft,
    SQLProviderStatus,
)
from atlas.retrieval.providers.sql.schema_router import AtlasSchemaRouter
from atlas.retrieval.providers.sql.validator import SQLValidator


READY_STATUSES = frozenset({"ready", "supported", "ok", "enabled"})


class SQLProvider(RetrievalProvider):
    provider_name = SQL_PROVIDER

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        intent_gate: SQLIntentGate | None = None,
        schema_router: AtlasSchemaRouter | None = None,
        normalizer: IdentifierNormalizer | None = None,
        compiler: SQLCompiler | None = None,
        validator: SQLValidator | None = None,
        executor: DuckDBExecutor | None = None,
    ) -> None:
        self.settings = settings
        self.intent_gate = intent_gate or SQLIntentGate()
        self.schema_router = schema_router or AtlasSchemaRouter(
            min_table_score=_setting(settings, "structured_sql_min_table_score", 0.15),
            min_score_margin=_setting(settings, "structured_sql_min_score_margin", 0.10),
            max_candidate_tables=_setting(settings, "structured_sql_max_candidate_tables", 1),
        )
        self.normalizer = normalizer or IdentifierNormalizer()
        max_rows = _setting(settings, "structured_sql_max_rows", 100)
        self.compiler = compiler or _build_compiler(settings, default_limit=min(max_rows, 5))
        self.validator = validator or SQLValidator(max_limit=max_rows)
        self.executor = executor or DuckDBExecutor(
            duckdb_dir=_setting(settings, "structured_sql_duckdb_dir", None),
            timeout_ms=_setting(settings, "structured_sql_timeout_ms", 1000),
            max_rows=max_rows,
            max_result_bytes=_setting(settings, "structured_sql_max_result_bytes", 65536),
            memory_limit=_setting(settings, "structured_sql_memory_limit", "128MB"),
        )

    def retrieve_provider_result(
        self,
        db: Session,
        *,
        query: str,
        top_k: int,
        filters: dict | None,
        options: dict,
        query_plan: QueryPlan,
        retrieval_tasks: list[RetrievalTask],
    ) -> ProviderResult:
        started = time.perf_counter()
        sql_tasks = _sql_tasks(retrieval_tasks)
        task = sql_tasks[0] if sql_tasks else None
        base_trace: dict[str, Any] = {
            "provider": SQL_PROVIDER,
            "provider_version": SQL_PROVIDER_VERSION,
            "query_plan_id": query_plan.plan_id,
            "task_ids": [task.task_id for task in retrieval_tasks],
            "unit_ids": [task.unit_id for task in retrieval_tasks],
            "sql_task_ids": [task.task_id for task in sql_tasks],
            "sql_unit_ids": [task.unit_id for task in sql_tasks],
        }
        if task is None:
            return self._diagnostic(
                "skipped_not_table_query",
                "no_sql_task",
                started=started,
                retrieval_tasks=retrieval_tasks,
                trace=base_trace,
            )
        if len(sql_tasks) > 1:
            return self._diagnostic(
                "unsupported_multi_table",
                "multiple_sql_tasks_unsupported",
                started=started,
                retrieval_tasks=sql_tasks,
                trace=base_trace,
            )
        if task.provider_status not in READY_STATUSES:
            return self._diagnostic(
                "skipped_not_table_query",
                task.unsupported_reason or f"provider_status:{task.provider_status}",
                started=started,
                retrieval_tasks=[task],
                trace={**base_trace, "provider_status": task.provider_status},
            )

        question = task.query_text or query
        intent = self.intent_gate.evaluate(
            question,
            purpose=task.metadata.get("purpose"),
            metadata=task.metadata,
        )
        base_trace["intent"] = {
            "allowed": intent.allowed,
            "status": intent.status,
            "intent_status": "allowed" if intent.allowed else intent.status,
            "reason": intent.reason,
            "intent_type": intent.intent_type,
            "signals": list(intent.signals),
        }
        base_trace["intent_allowed"] = intent.allowed
        base_trace["intent_status"] = "allowed" if intent.allowed else intent.status
        if not intent.allowed:
            return self._diagnostic(
                intent.status,
                intent.reason,
                started=started,
                retrieval_tasks=[task],
                trace=base_trace,
            )

        route = self.schema_router.route(
            question,
            db=db,
            filters=filters or task.metadata_filter,
            options=options or {},
        )
        base_trace["schema_routing"] = route.trace
        if route.status != "success" or route.table is None:
            return self._diagnostic(
                route.status,
                route.reason or route.status,
                started=started,
                retrieval_tasks=[task],
                trace=base_trace,
            )

        schema = self.normalizer.normalize(route.table)
        base_trace["schema_context"] = {
            "table_id": schema.table_id,
            "safe_table_name": schema.safe_table_name,
            "column_count": len(schema.columns),
            "row_count": schema.row_count,
            "safe_to_raw": schema.safe_to_raw,
            "safe_to_raw_identifier_map": schema.safe_to_raw,
        }

        draft = _coerce_draft(self.compiler.compile(question, schema))
        base_trace["compiler"] = {
            "status": draft.status,
            "reason": draft.reason,
            "compiler_version": draft.compiler_version,
            **dict(draft.trace),
        }
        if draft.status != "success" or not draft.sql:
            return self._diagnostic(
                "compiler_failed",
                draft.reason or "compiler_failed",
                started=started,
                retrieval_tasks=[task],
                trace=base_trace,
            )

        validation = self.validator.validate(draft.sql, schema)
        base_trace["validator"] = {
            "valid": validation.valid,
            "status": validation.status,
            "reason": validation.reason,
            "backend": validation.validator_backend,
            "used_column_ids": list(validation.used_column_ids),
            "warnings": list(validation.warnings),
            **dict(validation.trace),
        }
        if not validation.valid or not validation.sql:
            return self._diagnostic(
                "validation_failed",
                validation.reason or "validation_failed",
                started=started,
                retrieval_tasks=[task],
                trace=base_trace,
            )

        try:
            execution = self.executor.execute(schema, validation.sql)
        except SQLExecutionTimeout as exc:
            return self._diagnostic(
                "timeout",
                str(exc),
                started=started,
                retrieval_tasks=[task],
                trace={
                    **base_trace,
                    "execution": {
                        "status": "timeout",
                        "timeout_ms": getattr(exc, "timeout_ms", None),
                        "timeout_isolation": getattr(exc, "timeout_isolation", "thread_only"),
                    },
                },
            )
        except Exception as exc:
            return self._diagnostic(
                "execution_failed",
                _error_message(exc),
                started=started,
                retrieval_tasks=[task],
                trace={
                    **base_trace,
                    "execution": {
                        "error_type": exc.__class__.__name__,
                        "error_message": _error_message(exc),
                        "warnings": list(getattr(exc, "warnings", ()) or ()),
                        **dict(getattr(exc, "trace", {}) or {}),
                    },
                },
            )
        base_trace["execution"] = {
            "status": execution.status,
            "row_count": execution.row_count,
            "truncated": execution.truncated,
            "result_bytes": execution.result_bytes,
            "warnings": list(execution.warnings),
            **dict(execution.trace),
        }
        if execution.status != "success":
            return self._diagnostic(
                execution.status,
                execution.reason or execution.status,
                started=started,
                retrieval_tasks=[task],
                trace=base_trace,
            )

        evidence, candidate = build_sql_result_evidence(
            schema=schema,
            sql=validation.sql,
            validation=validation,
            execution=execution,
            task_id=task.task_id,
            unit_id=task.unit_id,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ProviderResult(
            provider=SQL_PROVIDER,
            task_id=task.task_id,
            unit_id=task.unit_id,
            status="success",
            candidates=(candidate,),
            evidence=(evidence,),
            latency_ms=latency_ms,
            reason=None,
            trace={**base_trace, "status": "success", "latency_ms": latency_ms},
        )

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

    def _diagnostic(
        self,
        status: SQLProviderStatus,
        reason: str,
        *,
        started: float,
        retrieval_tasks: list[RetrievalTask],
        trace: dict[str, Any],
    ) -> ProviderResult:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ProviderResult(
            provider=SQL_PROVIDER,
            task_id=None if len(retrieval_tasks) != 1 else retrieval_tasks[0].task_id,
            unit_id=None if len(retrieval_tasks) != 1 else retrieval_tasks[0].unit_id,
            status=status,
            candidates=(),
            evidence=(),
            latency_ms=latency_ms,
            reason=reason,
            trace={**trace, "status": status, "reason": reason, "latency_ms": latency_ms},
        )


def _sql_tasks(tasks: list[RetrievalTask]) -> list[RetrievalTask]:
    return [task for task in tasks if str(task.provider).strip().lower() == SQL_PROVIDER]


def _build_compiler(settings: Settings | None, *, default_limit: int) -> Any:
    mode = str(_setting(settings, "structured_sql_compiler_mode", "heuristic")).strip().lower()
    if mode == "llm" and settings is not None:
        return OpenAIResponsesSQLCompiler(settings, default_limit=default_limit)
    return SQLCompiler(default_limit=default_limit, compiler_mode="heuristic")


def _coerce_draft(value: Any) -> SQLDraft:
    if isinstance(value, SQLDraft):
        return value
    if isinstance(value, str):
        return SQLDraft(status="success", sql=value, raw_output=value)
    return SQLDraft(status="compiler_failed", reason="invalid_compiler_result")


def _setting(settings: Settings | None, name: str, default: Any) -> Any:
    if settings is None:
        return default
    return getattr(settings, name, default)


def _error_message(exc: Exception) -> str:
    return " ".join(str(exc).split()) or exc.__class__.__name__
