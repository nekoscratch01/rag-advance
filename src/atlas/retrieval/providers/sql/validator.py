from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from atlas.retrieval.providers.sql.identifiers import is_safe_identifier
from atlas.retrieval.providers.sql.models import SQLSchemaContext, SQLValidationResult


AGGREGATES = frozenset({"COUNT", "SUM", "AVG", "MIN", "MAX"})
FORBIDDEN_KEYWORDS = (
    "alter",
    "attach",
    "call",
    "case",
    "cast",
    "coalesce",
    "copy",
    "create",
    "current_date",
    "current_time",
    "current_timestamp",
    "date_part",
    "date_trunc",
    "delete",
    "detach",
    "distinct",
    "drop",
    "else",
    "end",
    "except",
    "extract",
    "having",
    "ifnull",
    "insert",
    "install",
    "intersect",
    "join",
    "load",
    "nullif",
    "over",
    "pragma",
    "read_csv",
    "read_json",
    "read_parquet",
    "strptime",
    "then",
    "union",
    "update",
    "when",
    "window",
    "with",
)
SQLGLOT_ALLOWED_FUNCTION_CLASSES = frozenset({"Avg", "Count", "Max", "Min", "Sum"})
SQLGLOT_NON_FUNCTION_PREDICATE_CLASSES = frozenset(
    {
        "And",
        "Between",
        "EQ",
        "GT",
        "GTE",
        "ILike",
        "In",
        "Is",
        "LT",
        "LTE",
        "Like",
        "NEQ",
        "Not",
        "NullSafeEQ",
        "NullSafeNEQ",
        "Or",
        "Paren",
    }
)
SQLGLOT_FORBIDDEN_EXPRESSION_CLASSES = frozenset(
    {
        "Alter",
        "AnonymousAggFunc",
        "Attach",
        "Call",
        "Case",
        "Cast",
        "Coalesce",
        "Command",
        "Copy",
        "Create",
        "CTE",
        "Delete",
        "Distinct",
        "Drop",
        "Except",
        "Extract",
        "If",
        "IfNull",
        "Insert",
        "Intersect",
        "Join",
        "Nullif",
        "Qualify",
        "Subquery",
        "TryCast",
        "Union",
        "Update",
        "Window",
        "With",
    }
)
SQLGLOT_FORBIDDEN_CLASS_PREFIXES = ("Date", "Time", "Timestamp")
VALIDATION_CHECK_KEYS = (
    "single_statement",
    "select_only",
    "table_allowlist_passed",
    "column_allowlist_passed",
    "disallowed_nodes_absent",
    "external_access_absent",
    "select_star_absent",
    "join_absent",
    "subquery_absent",
)


class SQLValidator:
    def __init__(self, *, max_limit: int = 100, require_sqlglot: bool = False) -> None:
        self.max_limit = max_limit
        self.require_sqlglot = require_sqlglot

    def validate(self, sql: str, schema: SQLSchemaContext) -> SQLValidationResult:
        checks = _initial_checks(sql)
        early_reason = _early_validation_reason(sql)
        if early_reason is not None:
            return _invalid(early_reason, backend="precheck", checks=checks)
        try:
            import sqlglot  # type: ignore[import-not-found]
            from sqlglot import exp  # type: ignore[import-not-found]
        except ImportError:
            if self.require_sqlglot:
                return SQLValidationResult(
                    valid=False,
                    status="validation_failed",
                    reason=(
                        "optional_dependency_missing: sqlglot is required for SQLProvider "
                        "validation; install atlas-rag-kernel[structured-sql]"
                    ),
                    validator_backend="missing_sqlglot",
                    trace=_validation_trace(checks),
                )
            return self._validate_fallback(sql, schema, checks=checks)
        return self._validate_sqlglot(sql, schema, sqlglot=sqlglot, exp=exp, checks=checks)

    def _validate_sqlglot(
        self,
        sql: str,
        schema: SQLSchemaContext,
        *,
        sqlglot: Any,
        exp: Any,
        checks: dict[str, bool],
    ):
        try:
            parsed = sqlglot.parse(_strip_one_trailing_semicolon(sql), read="duckdb")
        except Exception as exc:
            return _invalid(f"parse_failed:{exc.__class__.__name__}", backend="sqlglot", checks=checks)
        if len(parsed) != 1:
            checks["single_statement"] = False
            return _invalid("multi_statement_forbidden", backend="sqlglot", checks=checks)
        checks["single_statement"] = True
        node = parsed[0]
        if not isinstance(node, exp.Select):
            checks["select_only"] = False
            return _invalid("only_select_allowed", backend="sqlglot", checks=checks)
        checks["select_only"] = True
        if _sqlglot_has_join(node, exp):
            checks["join_absent"] = False
            checks["disallowed_nodes_absent"] = False
            return _invalid("join_forbidden", backend="sqlglot", checks=checks)
        if node.args.get("distinct") is not None:
            checks["disallowed_nodes_absent"] = False
            return _invalid("distinct_forbidden", backend="sqlglot", checks=checks)
        if node.args.get("having") is not None:
            checks["disallowed_nodes_absent"] = False
            return _invalid("having_forbidden", backend="sqlglot", checks=checks)
        if node.args.get("qualify") is not None:
            checks["disallowed_nodes_absent"] = False
            return _invalid("qualify_forbidden", backend="sqlglot", checks=checks)
        if _sqlglot_has_projection_star(node, exp):
            checks["select_star_absent"] = False
            return _invalid("select_star_forbidden", backend="sqlglot", checks=checks)

        for projection in _sqlglot_select_expressions(node):
            projection_reason = _sqlglot_projection_forbidden_reason(projection, exp)
            if projection_reason is not None:
                return _invalid(projection_reason, backend="sqlglot", checks=checks)

        for expression in _sqlglot_walk(node):
            class_name = expression.__class__.__name__
            if class_name in SQLGLOT_FORBIDDEN_EXPRESSION_CLASSES or class_name.startswith(
                SQLGLOT_FORBIDDEN_CLASS_PREFIXES
            ):
                return _invalid(
                    f"forbidden_expression:{class_name}",
                    backend="sqlglot",
                    checks=checks,
                )
            function_name = _sqlglot_function_name(expression, exp)
            if function_name is None:
                continue
            if function_name not in AGGREGATES:
                return _invalid(
                    f"function_forbidden:{function_name or 'anonymous'}",
                    backend="sqlglot",
                    checks=checks,
                )
            star_reason = _sqlglot_function_star_reason(expression, function_name, exp)
            if star_reason is not None:
                return _invalid(star_reason, backend="sqlglot", checks=checks)

        tables = list(node.find_all(exp.Table))
        if len(tables) != 1:
            checks["table_allowlist_passed"] = False
            return _invalid("single_table_select_required", backend="sqlglot", checks=checks)
        table_name = tables[0].name
        if table_name != schema.safe_table_name:
            checks["table_allowlist_passed"] = False
            return _invalid(f"unknown_table:{table_name}", backend="sqlglot", checks=checks)
        checks["table_allowlist_passed"] = True

        used_safe_columns: set[str] = set()
        for column in node.find_all(exp.Column):
            column_name = column.name
            table_prefix = column.table
            if table_prefix and table_prefix != schema.safe_table_name:
                checks["column_allowlist_passed"] = False
                return _invalid(f"unknown_table_prefix:{table_prefix}", backend="sqlglot", checks=checks)
            if column_name not in schema.column_ids_by_safe_name:
                checks["column_allowlist_passed"] = False
                return _invalid(f"unknown_column:{column_name}", backend="sqlglot", checks=checks)
            used_safe_columns.add(column_name)
        checks["column_allowlist_passed"] = True

        for alias in node.find_all(exp.Alias):
            alias_name = str(alias.alias or "")
            if alias_name and not is_safe_identifier(alias_name):
                return _invalid(f"unsafe_alias:{alias_name}", backend="sqlglot", checks=checks)

        limit = _sqlglot_limit_value(node, exp)
        if limit is None or limit > self.max_limit:
            node.set("limit", exp.Limit(expression=exp.Literal.number(self.max_limit)))
            limit = self.max_limit

        return SQLValidationResult(
            valid=True,
            status="success",
            sql=node.sql(dialect="duckdb"),
            used_column_ids=_column_ids(schema, used_safe_columns),
            used_safe_columns=tuple(sorted(used_safe_columns)),
            limit=limit,
            validator_backend="sqlglot",
            trace=_validation_trace(_success_checks(checks), limit_inserted_or_capped=limit == self.max_limit),
        )

    def _validate_fallback(
        self,
        sql: str,
        schema: SQLSchemaContext,
        *,
        checks: dict[str, bool],
    ) -> SQLValidationResult:
        clean = _strip_one_trailing_semicolon(sql)
        lowered = clean.lower()
        forbidden = _first_forbidden(clean)
        if forbidden is not None:
            return _invalid(forbidden, backend="fallback", checks=checks)
        match = re.match(
            r"^\s*select\s+(?P<select>.+?)\s+from\s+(?P<table>[a-z_][a-z0-9_]*)\s*(?P<tail>.*)$",
            clean,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            checks["select_only"] = False
            return _invalid("only_single_select_from_table_allowed", backend="fallback", checks=checks)
        checks["single_statement"] = True
        checks["select_only"] = True
        table_name = match.group("table")
        if table_name != schema.safe_table_name:
            checks["table_allowlist_passed"] = False
            return _invalid(f"unknown_table:{table_name}", backend="fallback", checks=checks)
        checks["table_allowlist_passed"] = True
        projection = match.group("select").strip()
        tail = match.group("tail").strip()
        if projection == "*" or re.search(r"(^|,)\s*[a-z_][a-z0-9_]*\.\*\s*(,|$)", projection):
            checks["select_star_absent"] = False
            return _invalid("select_star_forbidden", backend="fallback", checks=checks)
        if "select" in lowered[lowered.find(" from ") + 6 :]:
            checks["subquery_absent"] = False
            return _invalid("subquery_forbidden", backend="fallback", checks=checks)

        used_columns: set[str] = set()
        aliases: set[str] = set()
        for expression in _split_csv(projection):
            ok, reason, expression_columns, alias = _validate_projection_expression(expression, schema)
            if not ok:
                return _invalid(reason or "invalid_projection", backend="fallback", checks=checks)
            used_columns.update(expression_columns)
            if alias:
                aliases.add(alias)

        clauses = _parse_tail_clauses(tail)
        if clauses.get("invalid"):
            return _invalid(str(clauses["invalid"]), backend="fallback", checks=checks)
        try:
            for column in _columns_from_where(clauses.get("where", ""), schema):
                used_columns.add(column)
        except ValueError as exc:
            return _invalid(str(exc), backend="fallback", checks=checks)
        group_columns = _columns_from_plain_list(clauses.get("group_by", ""), schema)
        if group_columns is None:
            return _invalid("invalid_group_by", backend="fallback", checks=checks)
        used_columns.update(group_columns)
        order_columns = _columns_from_order_by(clauses.get("order_by", ""), schema, aliases)
        if order_columns is None:
            return _invalid("invalid_order_by", backend="fallback", checks=checks)
        used_columns.update(order_columns)
        checks["column_allowlist_passed"] = True

        try:
            limit = _limit_value(clauses.get("limit"))
        except ValueError as exc:
            return _invalid(str(exc), backend="fallback", checks=checks)
        sanitized = clean
        if limit is None:
            sanitized = f"{sanitized} LIMIT {self.max_limit}"
            limit = self.max_limit
        elif limit > self.max_limit:
            sanitized = re.sub(
                r"\blimit\s+\d+\b",
                f"LIMIT {self.max_limit}",
                sanitized,
                flags=re.IGNORECASE,
            )
            limit = self.max_limit

        return SQLValidationResult(
            valid=True,
            status="success",
            sql=" ".join(sanitized.split()),
            used_column_ids=_column_ids(schema, used_columns),
            used_safe_columns=tuple(sorted(used_columns)),
            limit=limit,
            validator_backend="fallback",
            warnings=("sqlglot_missing_conservative_fallback_used",),
            trace=_validation_trace(_success_checks(checks)),
        )


def _validate_projection_expression(
    expression: str,
    schema: SQLSchemaContext,
) -> tuple[bool, str | None, set[str], str | None]:
    expression = expression.strip()
    if not expression:
        return False, "empty_projection", set(), None
    expr, alias = _split_alias(expression)
    if alias and not is_safe_identifier(alias):
        return False, f"unsafe_alias:{alias}", set(), None
    if expr == "*":
        return False, "select_star_forbidden", set(), None
    if expr in schema.column_ids_by_safe_name:
        return True, None, {expr}, alias
    aggregate = re.match(
        r"^(?P<fn>count|sum|avg|min|max)\s*\(\s*(?P<arg>\*|[a-z_][a-z0-9_]*)\s*\)$",
        expr,
        flags=re.IGNORECASE,
    )
    if aggregate:
        fn = aggregate.group("fn").upper()
        arg = aggregate.group("arg")
        if fn not in AGGREGATES:
            return False, f"function_forbidden:{fn}", set(), None
        if arg == "*":
            if fn == "COUNT":
                return True, None, set(), alias
            return False, "only_count_allows_star_argument", set(), None
        if arg not in schema.column_ids_by_safe_name:
            return False, f"unknown_column:{arg}", set(), None
        return True, None, {arg}, alias
    return False, f"unsupported_projection:{expr}", set(), None


def _split_alias(expression: str) -> tuple[str, str | None]:
    match = re.match(
        r"^(?P<expr>.+?)\s+as\s+(?P<alias>[a-z_][a-z0-9_]*)$",
        expression,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group("expr").strip(), match.group("alias").strip()
    return expression.strip(), None


def _parse_tail_clauses(tail: str) -> dict[str, str]:
    if not tail:
        return {}
    if re.search(r"\bhaving\b", tail, flags=re.IGNORECASE):
        return {"invalid": "having_forbidden"}
    pattern = re.compile(
        r"\b(where|group\s+by|order\s+by|limit)\b",
        flags=re.IGNORECASE,
    )
    matches = list(pattern.finditer(tail))
    clauses: dict[str, str] = {}
    if not matches and tail.strip():
        return {"invalid": f"unsupported_clause:{tail.strip()}"}
    for index, match in enumerate(matches):
        key = match.group(1).lower().replace(" ", "_")
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(tail)
        clauses[key] = tail[start:end].strip()
    prefix = tail[: matches[0].start()].strip() if matches else ""
    if prefix:
        return {"invalid": f"unsupported_clause:{prefix}"}
    return clauses


def _columns_from_where(where_clause: str, schema: SQLSchemaContext) -> set[str]:
    if not where_clause:
        return set()
    columns: set[str] = set()
    parts = re.split(r"\s+and\s+", where_clause, flags=re.IGNORECASE)
    for condition in parts:
        match = re.match(
            r"^(?P<col>[a-z_][a-z0-9_]*)\s*(=|!=|<>|<=|>=|<|>|like)\s*"
            r"(?P<literal>'[^']*'|\"[^\"]*\"|-?\d+(?:\.\d+)?|true|false|null)$",
            condition.strip(),
            flags=re.IGNORECASE,
        )
        if not match:
            raise ValueError(f"invalid_where_condition:{condition.strip()}")
        column = match.group("col")
        if column not in schema.column_ids_by_safe_name:
            raise ValueError(f"unknown_column:{column}")
        columns.add(column)
    return columns


def _columns_from_plain_list(value: str, schema: SQLSchemaContext) -> set[str] | None:
    if not value:
        return set()
    columns = set()
    for item in _split_csv(value):
        if item not in schema.column_ids_by_safe_name:
            return None
        columns.add(item)
    return columns


def _columns_from_order_by(
    value: str,
    schema: SQLSchemaContext,
    aliases: set[str],
) -> set[str] | None:
    if not value:
        return set()
    columns = set()
    for item in _split_csv(value):
        match = re.match(r"^(?P<name>[a-z_][a-z0-9_]*)(?:\s+(asc|desc))?$", item, flags=re.I)
        if not match:
            return None
        name = match.group("name")
        if name in schema.column_ids_by_safe_name:
            columns.add(name)
        elif name not in aliases:
            return None
    return columns


def _limit_value(value: str | None) -> int | None:
    if not value:
        return None
    match = re.match(r"^(\d+)$", value.strip())
    if not match:
        raise ValueError("invalid_limit")
    return int(match.group(1))


def _first_forbidden(sql: str) -> str | None:
    stripped = sql.strip()
    if ";" in stripped:
        return "multi_statement_forbidden"
    if "--" in stripped or "/*" in stripped or "*/" in stripped:
        return "sql_comments_forbidden"
    if re.search(r"\b(https?|s3|file)://", stripped, flags=re.IGNORECASE):
        return "external_access_string_forbidden"
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", stripped, flags=re.IGNORECASE):
            return f"keyword_forbidden:{keyword}"
    return None


def _strip_one_trailing_semicolon(sql: str) -> str:
    stripped = sql.strip()
    if stripped.endswith(";"):
        stripped = stripped[:-1].strip()
    return stripped


def _split_csv(value: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in value:
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def _column_ids(schema: SQLSchemaContext, used_safe_columns: Iterable[str]) -> tuple[str, ...]:
    mapping = schema.column_ids_by_safe_name
    return tuple(mapping[column] for column in sorted(set(used_safe_columns)) if column in mapping)


def _sqlglot_walk(node: Any) -> Iterable[Any]:
    for item in node.walk():
        yield item[0] if isinstance(item, tuple) else item


def _sqlglot_select_expressions(node: Any) -> tuple[Any, ...]:
    expressions = getattr(node, "expressions", None)
    if expressions is None:
        expressions = getattr(node, "args", {}).get("expressions") or ()
    return tuple(expressions or ())


def _sqlglot_has_projection_star(node: Any, exp: Any) -> bool:
    for projection in _sqlglot_select_expressions(node):
        expression = _sqlglot_unwrap_alias(projection, exp)
        if _sqlglot_is_star_expression(expression, exp):
            return True
        if _sqlglot_is_column_star(expression, exp):
            return True
    return False


def _sqlglot_has_join(node: Any, exp: Any) -> bool:
    joins = getattr(node, "args", {}).get("joins") or ()
    if joins:
        return True
    join_cls = getattr(exp, "Join", None)
    for expression in _sqlglot_walk(node):
        if join_cls is not None and isinstance(expression, join_cls):
            return True
        if expression.__class__.__name__ == "Join":
            return True
    return False


def _sqlglot_projection_forbidden_reason(projection: Any, exp: Any) -> str | None:
    expression = _sqlglot_unwrap_alias(projection, exp)
    if _sqlglot_is_star_expression(expression, exp) or _sqlglot_is_column_star(expression, exp):
        return "select_star_forbidden"
    if _is_sqlglot_instance(expression, exp, "Column"):
        return None
    function_name = _sqlglot_function_name(expression, exp)
    if function_name is None:
        return f"unsupported_projection:{expression.__class__.__name__}"
    if function_name not in AGGREGATES:
        return f"function_forbidden:{function_name or 'anonymous'}"
    return _sqlglot_aggregate_argument_reason(expression, function_name, exp)


def _sqlglot_unwrap_alias(expression: Any, exp: Any) -> Any:
    if _is_sqlglot_instance(expression, exp, "Alias"):
        return getattr(expression, "this", None) or getattr(expression, "args", {}).get("this")
    return expression


def _sqlglot_function_name(expression: Any, exp: Any) -> str | None:
    if _is_sqlglot_instance(expression, exp, "Anonymous"):
        return str(getattr(expression, "name", "") or "anonymous").upper()
    if not _sqlglot_function(expression, exp):
        return None
    sql_name = getattr(expression, "sql_name", None)
    if callable(sql_name):
        try:
            value = str(sql_name() or "").strip()
            if value:
                return value.upper()
        except Exception:
            pass
    class_name = expression.__class__.__name__
    class_map = {
        "Avg": "AVG",
        "Count": "COUNT",
        "Max": "MAX",
        "Min": "MIN",
        "Sum": "SUM",
    }
    return class_map.get(class_name, _camel_to_upper_sql_name(class_name))


def _sqlglot_function(expression: Any, exp: Any) -> bool:
    if _is_sqlglot_instance(expression, exp, "Anonymous"):
        return True
    if expression.__class__.__name__ in SQLGLOT_NON_FUNCTION_PREDICATE_CLASSES:
        return False
    if expression.__class__.__name__ in SQLGLOT_ALLOWED_FUNCTION_CLASSES:
        return True
    return _is_sqlglot_instance(expression, exp, "Func") or _is_sqlglot_instance(
        expression,
        exp,
        "AggFunc",
    )


def _sqlglot_aggregate_argument_reason(expression: Any, function_name: str, exp: Any) -> str | None:
    arguments = _sqlglot_function_argument_nodes(expression)
    if not arguments:
        return f"invalid_function_argument:{function_name}"
    for argument in arguments:
        if _sqlglot_is_star_expression(argument, exp):
            if function_name == "COUNT":
                continue
            return "only_count_allows_star_argument"
        if _sqlglot_is_column_star(argument, exp):
            return "qualified_star_argument_forbidden"
        if not _is_sqlglot_instance(argument, exp, "Column"):
            return f"invalid_function_argument:{function_name}:{argument.__class__.__name__}"
    return None


def _sqlglot_function_star_reason(expression: Any, function_name: str, exp: Any) -> str | None:
    if not any(True for _ in expression.find_all(exp.Star)):
        return None
    if any(_sqlglot_is_column_star(column, exp) for column in expression.find_all(exp.Column)):
        return "qualified_star_argument_forbidden"
    if function_name != "COUNT":
        return "only_count_allows_star_argument"
    return None


def _sqlglot_function_argument_nodes(expression: Any) -> tuple[Any, ...]:
    values: list[Any] = []
    for attr_name in ("this", "expression"):
        value = getattr(expression, attr_name, None)
        if value is not None:
            values.append(value)
    values.extend(tuple(getattr(expression, "expressions", None) or ()))
    args = getattr(expression, "args", {}) or {}
    for key in ("this", "expression"):
        value = args.get(key)
        if value is not None:
            values.append(value)
    args_expressions = args.get("expressions")
    if isinstance(args_expressions, list | tuple):
        values.extend(args_expressions)
    deduped: list[Any] = []
    seen: set[int] = set()
    for value in values:
        if id(value) in seen:
            continue
        seen.add(id(value))
        deduped.append(value)
    return tuple(deduped)


def _sqlglot_is_star_expression(expression: Any, exp: Any) -> bool:
    return _is_sqlglot_instance(expression, exp, "Star")


def _sqlglot_is_column_star(expression: Any, exp: Any) -> bool:
    if not _is_sqlglot_instance(expression, exp, "Column"):
        return False
    if bool(getattr(expression, "is_star", False)):
        return True
    if str(getattr(expression, "name", "") or "") == "*":
        return True
    this = getattr(expression, "this", None) or getattr(expression, "args", {}).get("this")
    return _sqlglot_is_star_expression(this, exp)


def _is_sqlglot_instance(expression: Any, exp: Any, class_name: str) -> bool:
    cls = getattr(exp, class_name, None)
    return cls is not None and isinstance(expression, cls)


def _camel_to_upper_sql_name(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).upper()


def _sqlglot_limit_value(node: Any, exp: Any) -> int | None:
    limit = node.args.get("limit")
    if limit is None:
        return None
    expression = getattr(limit, "expression", None)
    if isinstance(expression, exp.Literal) and expression.is_int:
        return int(expression.this)
    return None


def _initial_checks(sql: str) -> dict[str, bool]:
    clean = _strip_one_trailing_semicolon(sql)
    return {
        "single_statement": ";" not in clean,
        "select_only": bool(re.match(r"^\s*select\b", clean, flags=re.IGNORECASE)),
        "table_allowlist_passed": False,
        "column_allowlist_passed": False,
        "disallowed_nodes_absent": not _has_disallowed_shape(clean),
        "external_access_absent": not _has_external_access(clean),
        "select_star_absent": _projection_star_absent(clean),
        "join_absent": not re.search(r"\bjoin\b", clean, flags=re.IGNORECASE),
        "subquery_absent": not _has_subquery_shape(clean),
    }


def _success_checks(checks: dict[str, bool]) -> dict[str, bool]:
    updated = dict(checks)
    for key in (
        "single_statement",
        "select_only",
        "table_allowlist_passed",
        "column_allowlist_passed",
    ):
        updated[key] = True
    return updated


def _validation_trace(checks: dict[str, bool], **extra: Any) -> dict[str, Any]:
    return {
        **extra,
        "checks": {key: bool(checks.get(key)) for key in VALIDATION_CHECK_KEYS},
    }


def _has_disallowed_shape(sql: str) -> bool:
    if "--" in sql or "/*" in sql or "*/" in sql:
        return True
    return any(
        re.search(rf"\b{re.escape(keyword)}\b", sql, flags=re.IGNORECASE)
        for keyword in FORBIDDEN_KEYWORDS
    )


def _has_external_access(sql: str) -> bool:
    return bool(
        re.search(r"\b(https?|s3|file)://", sql, flags=re.IGNORECASE)
        or re.search(r"\bread_(csv|json|parquet)\b", sql, flags=re.IGNORECASE)
        or re.search(r"\b(attach|copy|load|install)\b", sql, flags=re.IGNORECASE)
    )


def _early_validation_reason(sql: str) -> str | None:
    clean = _strip_one_trailing_semicolon(sql)
    if ";" in clean:
        return "multi_statement_forbidden"
    if "--" in clean or "/*" in clean or "*/" in clean:
        return "sql_comments_forbidden"
    if re.search(r"\b(https?|s3|file)://", clean, flags=re.IGNORECASE):
        return "external_access_string_forbidden"
    for keyword in ("attach", "copy", "install", "load", "read_csv", "read_json", "read_parquet"):
        if re.search(rf"\b{re.escape(keyword)}\b", clean, flags=re.IGNORECASE):
            return f"keyword_forbidden:{keyword}"
    return None


def _projection_star_absent(sql: str) -> bool:
    match = re.match(
        r"^\s*select\s+(?P<select>.+?)\s+from\b",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return True
    projection = match.group("select").strip()
    return not (
        projection == "*"
        or re.search(r"(^|,)\s*(?:[a-z_][a-z0-9_]*\.)?\*\s*(,|$)", projection)
    )


def _has_subquery_shape(sql: str) -> bool:
    lowered = sql.lower()
    if re.search(r"\b(with|union|intersect|except)\b", lowered):
        return True
    from_index = lowered.find(" from ")
    return from_index >= 0 and "select" in lowered[from_index + 6 :]


def _mark_checks_for_reason(checks: dict[str, bool], reason: str) -> dict[str, bool]:
    updated = dict(checks)
    lowered = reason.lower()
    if "multi_statement" in lowered:
        updated["single_statement"] = False
    if "only_select" in lowered or "only_single_select" in lowered:
        updated["select_only"] = False
    if "unknown_table" in lowered or "single_table" in lowered or "table_prefix" in lowered:
        updated["table_allowlist_passed"] = False
    if (
        "unknown_column" in lowered
        or "invalid_group_by" in lowered
        or "invalid_order_by" in lowered
        or "invalid_where" in lowered
        or "unsupported_projection" in lowered
    ):
        updated["column_allowlist_passed"] = False
    if "select_star" in lowered or "star_argument" in lowered:
        updated["select_star_absent"] = False
    if "join" in lowered:
        updated["join_absent"] = False
        updated["disallowed_nodes_absent"] = False
    if any(token in lowered for token in ("subquery", "cte", "with", "union", "intersect", "except")):
        updated["subquery_absent"] = False
        updated["disallowed_nodes_absent"] = False
    if any(token in lowered for token in ("external", "read_csv", "read_json", "read_parquet", "attach")):
        updated["external_access_absent"] = False
        updated["disallowed_nodes_absent"] = False
    if (
        "keyword_forbidden" in lowered
        or "function_forbidden" in lowered
        or "forbidden_expression" in lowered
        or "sql_comments" in lowered
        or "distinct" in lowered
        or "having" in lowered
        or "qualify" in lowered
        or "unsafe_alias" in lowered
        or "parse_failed" in lowered
    ):
        updated["disallowed_nodes_absent"] = False
    return updated


def _invalid(reason: str, *, backend: str, checks: dict[str, bool] | None = None) -> SQLValidationResult:
    trace = _validation_trace(_mark_checks_for_reason(checks, reason)) if checks is not None else {}
    return SQLValidationResult(
        valid=False,
        status="validation_failed",
        reason=reason,
        validator_backend=backend,
        trace=trace,
    )
