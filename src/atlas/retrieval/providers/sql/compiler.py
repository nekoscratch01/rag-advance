from __future__ import annotations

import inspect
import hashlib
import json
import re
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from atlas.core.config import Settings
from atlas.llm.clients import LLMClient, OpenAIClient
from atlas.retrieval.providers.sql.models import SQLDraft, SQLSchemaContext


VENDORED_LLAMAINDEX_SQL_COMPILER_NOTICE = (
    "Atlas SQLCompiler implements a small LlamaIndex-style text-to-SQL prompt and "
    "SQL extraction interface without vendoring LlamaIndex runtime code. Reference "
    "concept: llama-index SQL query engine prompt/context formatting; local "
    "modifications: Atlas single-table schema context, deterministic fallback, and "
    "validator-first execution boundary."
)


class SQLCompiler:
    def __init__(
        self,
        *,
        llm_callable: Callable[..., Any] | None = None,
        default_limit: int = 5,
        compiler_mode: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.llm_callable = llm_callable
        self.default_limit = default_limit
        self.compiler_mode = compiler_mode or ("llm" if llm_callable is not None else "heuristic")
        self.model_name = model_name

    def compile(self, question: str, schema: SQLSchemaContext) -> SQLDraft:
        prompt = format_sql_prompt(question, schema)
        trace = _compiler_trace(
            mode=self.compiler_mode,
            model_name=self.model_name,
            schema=schema,
            prompt=prompt,
        )
        if self.llm_callable is not None:
            try:
                raw_output = _call_llm(self.llm_callable, question=question, schema=schema, prompt=prompt)
            except Exception as exc:
                return SQLDraft(
                    status="compiler_failed",
                    reason=f"llm_compiler_failed:{exc.__class__.__name__}",
                    prompt=prompt,
                    trace={**trace, "error_type": exc.__class__.__name__, "error_message": str(exc)},
                )
            sql = extract_sql(str(_raw_text(raw_output) or ""))
            if not sql:
                return SQLDraft(
                    status="compiler_failed",
                    raw_output=str(raw_output),
                    reason="no_select_sql_extracted",
                    prompt=prompt,
                    trace=trace,
                )
            return SQLDraft(
                status="success",
                sql=sql,
                raw_output=str(raw_output),
                prompt=prompt,
                trace=trace,
            )

        sql = heuristic_sql(question, schema, default_limit=self.default_limit)
        if sql is None:
            return SQLDraft(
                status="compiler_failed",
                reason="no_llm_and_no_safe_heuristic_sql",
                prompt=prompt,
                trace=trace,
            )
        return SQLDraft(
            status="success",
            sql=sql,
            raw_output=sql,
            prompt=prompt,
            trace=trace,
        )


class OpenAIResponsesSQLCompiler:
    """OpenAI Responses API compiler used only when SQLProvider explicitly opts in."""

    compiler_version = "atlas_sql_compiler_v1_openai_responses"

    def __init__(
        self,
        settings: Settings,
        *,
        client: LLMClient | None = None,
        default_limit: int = 5,
    ) -> None:
        self.settings = settings
        self.client = client
        self.default_limit = default_limit

    def compile(self, question: str, schema: SQLSchemaContext) -> SQLDraft:
        prompt = format_sql_prompt(question, schema)
        request_call_id = _compiler_call_id()
        trace = _compiler_trace(
            mode="llm",
            model_name=self.settings.llm_model,
            schema=schema,
            prompt=prompt,
            call_id=request_call_id,
        )
        trace.update(
            {
                "request_call_id": request_call_id,
                "reasoning_effort": self.settings.llm_reasoning_effort,
                "store": False,
            }
        )
        request: dict[str, Any] = {
            "model": self.settings.llm_model,
            "input": prompt,
            "max_output_tokens": min(int(self.settings.llm_max_output_tokens), 1000),
            "reasoning": {"effort": self.settings.llm_reasoning_effort},
            "store": False,
        }
        try:
            client = self.client or OpenAIClient(self.settings)
            response = client.create_response(request)
        except Exception as exc:
            return SQLDraft(
                status="compiler_failed",
                reason=f"llm_compiler_failed:{exc.__class__.__name__}",
                compiler_version=self.compiler_version,
                trace={**trace, "error_type": exc.__class__.__name__, "error_message": str(exc)},
            )

        response_id = _response_id(response)
        if response_id:
            trace["compiler_call_id"] = response_id
            trace["response_id"] = response_id
        raw_text = str(_raw_text(response) or "")
        sql = extract_sql(raw_text)
        if not sql:
            return SQLDraft(
                status="compiler_failed",
                raw_output=raw_text,
                reason="no_select_sql_extracted",
                compiler_version=self.compiler_version,
                trace=trace,
            )
        return SQLDraft(
            status="success",
            sql=sql,
            raw_output=raw_text,
            compiler_version=self.compiler_version,
            trace=trace,
        )


def format_sql_prompt(question: str, schema: SQLSchemaContext) -> str:
    column_lines = "\n".join(
        f"- {column.safe_identifier}: raw={column.display_name!r}, type={column.data_type}, "
        f"role={column.semantic_role or 'unknown'}"
        for column in schema.columns
    )
    return (
        "You are an Atlas SQL compiler. Generate one safe DuckDB SELECT for the "
        "single table below. Use only listed safe identifiers. Do not use joins, "
        "subqueries, CTEs, DDL, DML, file reads, or SELECT *.\n\n"
        f"Table: {schema.safe_table_name} (raw: {schema.display_table_name})\n"
        f"Columns:\n{column_lines}\n\n"
        f"Question: {question}\n"
        "SQLQuery:"
    )


def extract_sql(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    sql_query = re.search(r"SQLQuery\s*:\s*(.*)", stripped, flags=re.IGNORECASE | re.DOTALL)
    if sql_query:
        stripped = sql_query.group(1).strip()
    select_match = re.search(r"\bselect\b", stripped, flags=re.IGNORECASE)
    if select_match is None:
        return None
    stripped = stripped[select_match.start() :].strip()
    if ";" in stripped:
        stripped = stripped.split(";", 1)[0].strip()
    return stripped or None


def heuristic_sql(question: str, schema: SQLSchemaContext, *, default_limit: int = 5) -> str | None:
    text = " ".join(question.lower().split())
    table = schema.safe_table_name
    columns = list(schema.columns)
    if not columns:
        return None

    if _is_count_question(text):
        return f"SELECT COUNT(*) AS row_count FROM {table}"

    group_col = _group_column(text, schema)
    numeric_col = _best_numeric_column(text, schema)
    if _looks_grouped(text) and group_col is not None and numeric_col is not None:
        alias = f"sum_{numeric_col.safe_identifier}"
        return (
            f"SELECT {group_col.safe_identifier}, SUM({numeric_col.safe_identifier}) AS {alias} "
            f"FROM {table} GROUP BY {group_col.safe_identifier} "
            f"ORDER BY {alias} DESC LIMIT {max(1, default_limit)}"
        )

    aggregate = _aggregate(text)
    if aggregate is not None:
        if aggregate == "COUNT":
            return f"SELECT COUNT(*) AS row_count FROM {table}"
        if numeric_col is None:
            return None
        alias = f"{aggregate.lower()}_{numeric_col.safe_identifier}"
        return f"SELECT {aggregate}({numeric_col.safe_identifier}) AS {alias} FROM {table}"

    if _looks_top_k(text):
        order_col = numeric_col or _first_column(columns)
        if order_col is None:
            return None
        select_cols = _select_columns_for_ranking(schema, order_col)
        direction = "ASC" if re.search(r"\b(lowest|smallest|least|min)\b", text) else "DESC"
        limit = _top_k(text) or default_limit
        return (
            f"SELECT {', '.join(select_cols)} FROM {table} "
            f"ORDER BY {order_col.safe_identifier} {direction} LIMIT {max(1, limit)}"
        )

    mentioned = _mentioned_columns(text, schema)
    if mentioned:
        return f"SELECT {', '.join(column.safe_identifier for column in mentioned)} FROM {table} LIMIT 1"

    if numeric_col is not None:
        return f"SELECT {numeric_col.safe_identifier} FROM {table} LIMIT 1"
    return None


def _call_llm(
    llm_callable: Callable[..., Any],
    *,
    question: str,
    schema: SQLSchemaContext,
    prompt: str,
) -> Any:
    try:
        signature = inspect.signature(llm_callable)
    except (TypeError, ValueError):
        return llm_callable(prompt)
    parameters = signature.parameters
    kwargs = {}
    if "prompt" in parameters:
        kwargs["prompt"] = prompt
    if "question" in parameters:
        kwargs["question"] = question
    if "schema" in parameters:
        kwargs["schema"] = schema
    if kwargs:
        return llm_callable(**kwargs)
    return llm_callable(prompt)


def _raw_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("sql", "text", "output_text", "raw_output"):
            if value.get(key):
                return str(value[key])
    for attr in ("sql", "text", "output_text", "raw_output"):
        if hasattr(value, attr):
            item = getattr(value, attr)
            if item:
                return str(item)
    return str(value) if value is not None else None


def _response_id(value: Any) -> str | None:
    raw = getattr(value, "raw", None)
    for item in (value, raw):
        if item is None:
            continue
        response_id = getattr(item, "id", None)
        if response_id:
            return str(response_id)
        if isinstance(item, Mapping) and item.get("id"):
            return str(item["id"])
    return None


def _compiler_trace(
    *,
    mode: str,
    model_name: str | None,
    schema: SQLSchemaContext,
    prompt: str,
    call_id: str | None = None,
) -> dict[str, Any]:
    return {
        "compiler_mode": mode,
        "model_name": model_name,
        "compiler_call_id": call_id or _compiler_call_id(),
        "fallback_used": False,
        "prompt_hash": _stable_hash(prompt),
        "schema_context_hash": _stable_hash(_schema_context_for_hash(schema)),
    }


def _compiler_call_id() -> str:
    return f"sql_compiler_{uuid.uuid4().hex}"


def _schema_context_for_hash(schema: SQLSchemaContext) -> dict[str, Any]:
    return {
        "table_id": schema.table_id,
        "safe_table_name": schema.safe_table_name,
        "columns": [
            {
                "column_id": column.column_id,
                "safe_identifier": column.safe_identifier,
                "data_type": column.data_type,
                "semantic_role": column.semantic_role,
            }
            for column in schema.columns
        ],
    }


def _stable_hash(value: Any) -> str:
    if isinstance(value, str):
        payload = value
    else:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_count_question(text: str) -> bool:
    return bool(re.search(r"\b(count|how many|number of rows|row count)\b", text))


def _aggregate(text: str) -> str | None:
    if re.search(r"\b(sum|total)\b", text):
        return "SUM"
    if re.search(r"\b(avg|average|mean)\b", text):
        return "AVG"
    if re.search(r"\b(max|maximum|highest|largest)\b", text):
        return "MAX"
    if re.search(r"\b(min|minimum|lowest|smallest)\b", text):
        return "MIN"
    if re.search(r"\bcount\b", text):
        return "COUNT"
    return None


def _looks_top_k(text: str) -> bool:
    return bool(re.search(r"\b(top\s+\d+|top|rank|highest|lowest|largest|smallest)\b", text))


def _looks_grouped(text: str) -> bool:
    return bool(re.search(r"\b(grouped?\s+by|by\s+each|per\s+)\b", text))


def _top_k(text: str) -> int | None:
    match = re.search(r"\btop\s+(\d+)\b", text)
    if match:
        return int(match.group(1))
    return None


def _group_column(text: str, schema: SQLSchemaContext):
    mentioned = _mentioned_columns(text, schema)
    for column in mentioned:
        if str(column.semantic_role or "").lower() in {"dimension", "period", "identifier"}:
            return column
    for column in schema.columns:
        if str(column.semantic_role or "").lower() in {"dimension", "period", "identifier"}:
            return column
    return None


def _best_numeric_column(text: str, schema: SQLSchemaContext):
    mentioned = _mentioned_columns(text, schema)
    numeric_mentioned = [column for column in mentioned if _is_numeric(column.data_type, column.semantic_role)]
    if numeric_mentioned:
        return numeric_mentioned[0]
    for column in schema.columns:
        if _is_numeric(column.data_type, column.semantic_role):
            return column
    return None


def _mentioned_columns(text: str, schema: SQLSchemaContext):
    tokens = set(re.findall(r"[a-z0-9_]+", text))
    mentioned = []
    for column in schema.columns:
        names = {
            column.safe_identifier,
            *_name_tokens(column.display_name),
            *_name_tokens(column.raw_source_name),
        }
        if tokens & names or column.safe_identifier in text:
            mentioned.append(column)
    return mentioned


def _select_columns_for_ranking(schema: SQLSchemaContext, order_col) -> list[str]:
    selected = []
    for column in schema.columns:
        if column.safe_identifier == order_col.safe_identifier:
            continue
        if str(column.semantic_role or "").lower() in {"dimension", "period", "identifier"}:
            selected.append(column.safe_identifier)
        if len(selected) >= 2:
            break
    selected.append(order_col.safe_identifier)
    return selected


def _first_column(columns):
    return columns[0] if columns else None


def _is_numeric(data_type: str, semantic_role: str | None) -> bool:
    return str(data_type or "").lower() in {
        "decimal",
        "double",
        "float",
        "integer",
        "number",
        "numeric",
    } or str(semantic_role or "").lower() == "measure"


def _name_tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", str(value).lower()) if len(token) > 1}
