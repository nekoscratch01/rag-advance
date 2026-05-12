from __future__ import annotations

import concurrent.futures
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atlas.retrieval.providers.sql.duckdb_index import create_connection_for_schema
from atlas.retrieval.providers.sql.models import SQLExecutionResult, SQLSchemaContext


class SQLExecutionTimeout(TimeoutError):
    def __init__(self, message: str, *, timeout_ms: int) -> None:
        super().__init__(message)
        self.timeout_ms = timeout_ms
        self.timeout_isolation = "thread_only"


class SQLResultCapExceeded(RuntimeError):
    pass


class SQLSandboxConfigurationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        warnings: tuple[str, ...],
        trace: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.warnings = warnings
        self.trace = trace


@dataclass(frozen=True)
class SandboxSettingsResult:
    warnings: tuple[str, ...]
    trace: dict[str, Any]


class DuckDBExecutor:
    def __init__(
        self,
        *,
        duckdb_dir: str | Path | None = None,
        timeout_ms: int = 1000,
        max_rows: int = 100,
        max_result_bytes: int = 65536,
        memory_limit: str | None = "128MB",
    ) -> None:
        self.duckdb_dir = Path(duckdb_dir) if duckdb_dir else None
        self.timeout_ms = timeout_ms
        self.max_rows = max_rows
        self.max_result_bytes = max_result_bytes
        self.memory_limit = memory_limit

    def execute(self, schema: SQLSchemaContext, sql: str) -> SQLExecutionResult:
        started = time.perf_counter()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._execute_sync, schema, sql)
        try:
            timeout = self.timeout_ms / 1000 if self.timeout_ms and self.timeout_ms > 0 else None
            result = future.result(timeout=timeout)
            return SQLExecutionResult(
                **{
                    **result.__dict__,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                }
            )
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise SQLExecutionTimeout(
                f"sql_execution_timeout:{self.timeout_ms}ms",
                timeout_ms=self.timeout_ms,
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _execute_sync(self, schema: SQLSchemaContext, sql: str) -> SQLExecutionResult:
        connection = None
        warnings: list[str] = []
        try:
            connection, index = create_connection_for_schema(schema, index_dir=self.duckdb_dir)
            warnings.extend(index.warnings)
            sandbox = _apply_sandbox_settings(connection, memory_limit=self.memory_limit)
            warnings.extend(sandbox.warnings)
            cursor = connection.execute(sql)
            raw_rows = cursor.fetchmany(max(0, self.max_rows) + 1)
            columns = tuple(item[0] for item in (cursor.description or ()))
            truncated = len(raw_rows) > self.max_rows
            raw_rows = raw_rows[: self.max_rows]
            rows = tuple(
                {
                    columns[index]: _jsonable(value)
                    for index, value in enumerate(row)
                }
                for row in raw_rows
            )
            result_bytes = len(json.dumps(rows, ensure_ascii=False, default=str).encode("utf-8"))
            if result_bytes > self.max_result_bytes:
                raise SQLResultCapExceeded(
                    f"result_byte_cap_exceeded:{result_bytes}>{self.max_result_bytes}"
                )
            if truncated:
                warnings.append(f"row_cap_applied:{self.max_rows}")
            return SQLExecutionResult(
                status="success",
                columns=columns,
                rows=rows,
                row_count=len(rows),
                truncated=truncated,
                result_bytes=result_bytes,
                warnings=tuple(warnings),
                trace={
                    "duckdb_path": str(index.path) if index.path else None,
                    "sandbox": sandbox.trace,
                    "timeout_isolation": "thread_only",
                    "timeout_ms": self.timeout_ms,
                    "timeout_caveat": "thread_timeout_does_not_kill_native_duckdb_query",
                },
            )
        finally:
            if connection is not None:
                connection.close()


def _apply_sandbox_settings(connection: Any, *, memory_limit: str | None) -> SandboxSettingsResult:
    warnings: list[str] = []
    applied: list[str] = []
    failures: list[dict[str, Any]] = []
    critical_failures: list[dict[str, Any]] = []
    settings: list[tuple[str, Any, bool]] = [
        ("enable_external_access", False, True),
        ("autoload_known_extensions", False, True),
        ("autoinstall_known_extensions", False, True),
        ("allow_community_extensions", False, True),
    ]
    if memory_limit:
        settings.append(("memory_limit", memory_limit, False))
    settings.append(("lock_configuration", True, True))
    for name, value, critical in settings:
        try:
            connection.execute(_set_statement(name, value))
            applied.append(name)
        except Exception as exc:
            warning = f"sandbox_setting_failed:{name}:{exc.__class__.__name__}:{exc}"
            warnings.append(warning)
            failure = {
                "setting": name,
                "critical": critical,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            }
            failures.append(failure)
            if critical:
                critical_failures.append(failure)
    trace = {
        "applied_settings": tuple(applied),
        "failed_settings": tuple(failures),
        "critical_settings": (
            "enable_external_access",
            "autoload_known_extensions",
            "autoinstall_known_extensions",
            "allow_community_extensions",
            "lock_configuration",
        ),
        "noncritical_settings": ("memory_limit",),
    }
    if critical_failures:
        first = critical_failures[0]
        raise SQLSandboxConfigurationError(
            f"sandbox_configuration_failed:{first['setting']}:{first['error_type']}",
            warnings=tuple(warnings),
            trace=trace,
        )
    return SandboxSettingsResult(warnings=tuple(warnings), trace=trace)


def _set_statement(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"SET {name}={'true' if value else 'false'}"
    return f"SET {name}='{str(value).replace(chr(39), chr(39) + chr(39))}'"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)
