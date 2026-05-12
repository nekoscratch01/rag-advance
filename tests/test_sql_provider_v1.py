from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from atlas.core.config import Settings, executable_query_providers
import atlas.retrieval.providers.sql.compiler as sql_compiler_module
from atlas.query_orchestrator.schema import QueryPlan, RetrievalUnit
from atlas.retrieval.models.retrieval_task import tasks_from_plan
from atlas.retrieval.providers.sql.compiler import SQLCompiler, extract_sql
from atlas.retrieval.providers.sql.evidence import build_sql_result_evidence, format_sql_result_text
from atlas.retrieval.providers.sql.executor import (
    DuckDBExecutor,
    SQLResultCapExceeded,
    SQLSandboxConfigurationError,
    _apply_sandbox_settings,
)
from atlas.retrieval.providers.sql.identifiers import IdentifierNormalizer
from atlas.retrieval.providers.sql.intent import SQLIntentGate
from atlas.retrieval.providers.sql.models import (
    SQLDraft,
    SQLExecutionResult,
    SQLTableContext,
)
from atlas.retrieval.providers.sql.provider import SQLProvider
from atlas.retrieval.providers.sql.validator import SQLValidator
from atlas.retrieval.router import ProviderRouter


def _table_context() -> SQLTableContext:
    return SQLTableContext(
        table_id="tbl_revenue",
        raw_source_name="Revenue Table",
        display_name="Revenue Table",
        document_id="doc_1",
        source_locator={
            "manifest_id": "manifest_revenue",
            "storage_ref": {"backend": "duckdb", "path": "tables/revenue.duckdb"},
        },
        rows=(
            {"Year": "2022", "Revenue ($)": "10", "Region": "US"},
            {"Year": "2023", "Revenue ($)": "15", "Region": "EU"},
        ),
        columns=(
            {
                "column_id": "col_year",
                "name": "Year",
                "data_type": "integer",
                "semantic_role": "period",
            },
            {
                "column_id": "col_revenue",
                "name": "Revenue ($)",
                "data_type": "number",
                "semantic_role": "measure",
            },
            {
                "column_id": "col_region",
                "name": "Region",
                "data_type": "string",
                "semantic_role": "dimension",
            },
        ),
        routing_text="CSV table schema for Revenue Table. Columns: Year, Revenue, Region.",
    )


def _schema():
    return IdentifierNormalizer().normalize(_table_context())


def _vision_revenue_schema():
    return IdentifierNormalizer().normalize(
        SQLTableContext(
            table_id="tbl_vision_revenue",
            raw_source_name="revenue_table",
            display_name="Revenue Table",
            rows=(),
            columns=(
                {"column_id": "col_fiscal_year", "name": "fiscal_year", "data_type": "string"},
                {"column_id": "col_product", "name": "product", "data_type": "string"},
                {"column_id": "col_revenue", "name": "revenue", "data_type": "number"},
            ),
            routing_text="Revenue table with fiscal year, product, and revenue columns.",
        )
    )


def _sql_plan(query: str) -> QueryPlan:
    return QueryPlan(
        plan_id="plan_sql_llm",
        original_query=query,
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_sql",
                purpose="structured_lookup",
                text=query,
                provider="sql",
            ),
        ),
    )


def _required_validator_checks() -> tuple[str, ...]:
    return (
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


def test_sql_intent_gate_allows_table_numeric_and_skips_text_query() -> None:
    gate = SQLIntentGate()

    allowed = gate.evaluate("What is the total revenue in the table?")
    skipped = gate.evaluate("Explain management risk factors in the filing.")

    assert allowed.allowed is True
    assert allowed.status == "success"
    assert skipped.allowed is False
    assert skipped.status == "skipped_not_table_query"


def test_sql_intent_force_sql_requires_trusted_internal_override() -> None:
    gate = SQLIntentGate()

    untrusted = gate.evaluate(
        "Explain management risk factors in the filing.",
        metadata={"force_sql": True},
    )
    trusted = gate.evaluate(
        "Explain management risk factors in the filing.",
        metadata={"force_sql": True, "force_sql_trusted": True},
    )

    assert untrusted.allowed is False
    assert untrusted.reason == "textual_or_explanatory_query"
    assert "metadata_force_untrusted" in untrusted.signals
    assert trusted.allowed is True
    assert trusted.reason == "forced_by_task_metadata"


def test_identifier_normalizer_sanitizes_collisions_reserved_unicode_and_headers() -> None:
    table = SQLTableContext(
        table_id="tbl_bad",
        raw_source_name="123 table",
        display_name="Bad Table",
        rows=(),
        columns=(
            {"column_id": "c1", "name": "Name"},
            {"column_id": "c2", "name": "Name!"},
            {"column_id": "c3", "name": "select"},
            {"column_id": "c4", "name": "收入"},
            {"column_id": "c5", "name": "value); DROP TABLE x; --"},
        ),
    )

    schema = IdentifierNormalizer().normalize(table)

    assert schema.safe_table_name == "c_123_table"
    assert [column.safe_identifier for column in schema.columns] == [
        "name",
        "name_2",
        "select_col",
        "u6536_u5165",
        "value_drop_table_x",
    ]
    assert schema.safe_to_raw["name_2"]["raw_source_name"] == "Name!"
    assert schema.safe_to_raw_identifier_map == schema.safe_to_raw


def test_identifier_normalizer_preserves_first_raw_mapping_and_marks_ambiguity() -> None:
    table = SQLTableContext(
        table_id="tbl_duplicate_raw",
        raw_source_name="Duplicate Raw Table",
        display_name="Duplicate Raw Table",
        rows=(),
        columns=(
            {"column_id": "c1", "name": "Amount"},
            {"column_id": "c2", "name": "Amount"},
        ),
    )

    schema = IdentifierNormalizer().normalize(table)

    assert [column.safe_identifier for column in schema.columns] == ["amount", "amount_2"]
    assert schema.raw_to_safe["Amount"] == "amount"
    assert schema.metadata["ambiguous_raw_to_safe"]["Amount"] == ["amount", "amount_2"]
    assert schema.safe_to_raw["amount"]["raw_name_ambiguous"] is True
    assert schema.safe_to_raw["amount_2"]["raw_name_ambiguous"] is True


def test_sql_compiler_extracts_llm_sql_and_has_safe_heuristic() -> None:
    schema = _schema()
    compiler = SQLCompiler(llm_callable=lambda prompt: "```sql\nSELECT COUNT(*) FROM revenue_table\n```")

    draft = compiler.compile("How many rows are in the revenue table?", schema)
    heuristic = SQLCompiler().compile("What is the total revenue?", schema)

    assert extract_sql("SQLQuery: SELECT SUM(revenue) FROM revenue_table;") == (
        "SELECT SUM(revenue) FROM revenue_table"
    )
    assert draft.sql == "SELECT COUNT(*) FROM revenue_table"
    assert heuristic.sql == "SELECT SUM(revenue) AS sum_revenue FROM revenue_table"
    assert draft.trace["compiler_mode"] == "llm"
    assert draft.trace["fallback_used"] is False
    assert draft.trace["compiler_call_id"]
    assert heuristic.trace["compiler_mode"] == "heuristic"


def test_sql_validator_allows_count_star_and_rejects_malicious_shapes() -> None:
    schema = _schema()
    validator = SQLValidator(max_limit=10)

    allowed = validator.validate("SELECT COUNT(*) AS row_count FROM revenue_table", schema)
    simple_alias = validator.validate(
        "SELECT SUM(revenue) AS total_revenue FROM revenue_table",
        schema,
    )
    select_star = validator.validate("SELECT * FROM revenue_table", schema)
    join = validator.validate("SELECT revenue FROM revenue_table JOIN other ON true", schema)
    external = validator.validate("SELECT read_csv('http://evil.test/x.csv') FROM revenue_table", schema)
    distinct = validator.validate("SELECT DISTINCT revenue FROM revenue_table", schema)

    assert allowed.valid is True
    assert all(allowed.trace["checks"][key] is True for key in _required_validator_checks())
    assert allowed.used_column_ids == ()
    assert simple_alias.valid is True
    assert simple_alias.used_column_ids == ("col_revenue",)
    assert select_star.reason == "select_star_forbidden"
    assert select_star.trace["checks"]["select_star_absent"] is False
    assert "join" in (join.reason or "")
    assert join.trace["checks"]["join_absent"] is False
    assert external.reason == "external_access_string_forbidden"
    assert external.trace["checks"]["external_access_absent"] is False
    assert "distinct" in (distinct.reason or "")
    assert distinct.trace["checks"]["disallowed_nodes_absent"] is False


def test_sql_validator_sqlglot_path_allows_count_star_and_blocks_specialized_functions(
    monkeypatch,
) -> None:
    _install_fake_sqlglot(monkeypatch)
    schema = _schema()
    validator = SQLValidator(max_limit=10, require_sqlglot=True)

    allowed = validator.validate("SELECT COUNT(*) AS row_count FROM revenue_table", schema)
    select_star = validator.validate("SELECT * FROM revenue_table", schema)
    coalesce = validator.validate(
        "SELECT COALESCE(revenue, 0) AS revenue FROM revenue_table",
        schema,
    )
    sum_star = validator.validate("SELECT SUM(*) AS sum_all FROM revenue_table", schema)

    assert allowed.valid is True
    assert allowed.validator_backend == "sqlglot"
    assert all(allowed.trace["checks"][key] is True for key in _required_validator_checks())
    assert allowed.used_column_ids == ()
    assert select_star.reason == "select_star_forbidden"
    assert select_star.trace["checks"]["select_star_absent"] is False
    assert coalesce.reason == "function_forbidden:COALESCE"
    assert sum_star.reason == "only_count_allows_star_argument"


def test_sql_validator_sqlglot_path_allows_boolean_where_predicates() -> None:
    pytest.importorskip("sqlglot")
    schema = _vision_revenue_schema()
    validator = SQLValidator(max_limit=10, require_sqlglot=True)

    result = validator.validate(
        "SELECT SUM(revenue) AS total_revenue FROM revenue_table "
        "WHERE fiscal_year = 'FY2024' AND product = 'Vision Sensor'",
        schema,
    )

    assert result.valid is True, result.reason
    assert result.validator_backend == "sqlglot"
    assert set(result.used_column_ids) == {
        "col_fiscal_year",
        "col_product",
        "col_revenue",
    }
    assert all(result.trace["checks"][key] is True for key in _required_validator_checks())


def test_sql_evidence_is_deterministic_and_pinned_candidate() -> None:
    schema = _schema()
    execution = SQLExecutionResult(
        status="success",
        columns=("sum_revenue",),
        rows=({"sum_revenue": 25.0},),
        row_count=1,
        result_bytes=23,
    )
    validation = SQLValidator(max_limit=10).validate(
        "SELECT SUM(revenue) AS sum_revenue FROM revenue_table",
        schema,
    )

    evidence, candidate = build_sql_result_evidence(
        schema=schema,
        sql=validation.sql or "",
        validation=validation,
        execution=execution,
        task_id="rt_sql",
        unit_id="u_sql",
    )

    assert format_sql_result_text(schema, execution) == (
        "SQL result for table Revenue Table (tbl_revenue):\n"
        "sum_revenue = 25.0"
    )
    assert evidence.text == candidate.text
    assert candidate.provider == "sql"
    assert candidate.source_type == "sql_result"
    assert candidate.rerankable is False
    assert candidate.fusion_policy == "pinned"
    payload = candidate.structured_payload
    assert payload["sql"] == validation.sql
    assert payload["validated_sql"] == validation.sql
    assert payload["dialect"] == "duckdb"
    assert payload["candidate_table_id"] == "tbl_revenue"
    assert payload["table_id"] == "tbl_revenue"
    assert payload["used_column_ids"] == ["col_revenue"]
    assert payload["safe_to_raw_identifier_map"] == payload["safe_to_raw"]
    assert payload["manifest_id"] == "manifest_revenue"
    assert payload["storage_ref"] == {"backend": "duckdb", "path": "tables/revenue.duckdb"}
    assert payload["validation_status"] == "success"
    assert payload["execution_status"] == "success"
    assert payload["columns"] == ["sum_revenue"]
    assert payload["rows"] == [{"sum_revenue": 25.0}]
    assert payload["row_count"] == 1
    assert payload["answer_synthesis_verified"] is False


def test_settings_structured_sql_compiler_mode_default_and_env_opt_in(monkeypatch) -> None:
    assert Settings(openai_api_key=None).structured_sql_compiler_mode == "heuristic"

    monkeypatch.setenv("ATLAS_STRUCTURED_SQL_COMPILER_MODE", "llm")

    assert Settings(openai_api_key=None).structured_sql_compiler_mode == "llm"


def test_sql_provider_default_config_skips_sql_and_opt_in_makes_task_ready() -> None:
    disabled = Settings(
        openai_api_key=None,
        query_runtime_executable_providers="hybrid,sql,graph",
    )
    enabled = Settings(
        openai_api_key=None,
        sql_provider_enabled=True,
        query_runtime_executable_providers="hybrid,sql,graph",
    )

    assert executable_query_providers(disabled) == ("hybrid", "graph")
    assert executable_query_providers(enabled) == ("hybrid", "sql", "graph")

    plan = QueryPlan(
        plan_id="plan_sql_opt",
        original_query="How many rows are in the revenue table?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_sql",
                purpose="structured_lookup",
                text="How many rows are in the revenue table?",
                provider="sql",
            ),
        ),
    )
    default_task = tasks_from_plan(plan)[0]
    opt_in_task = tasks_from_plan(plan, executable_providers=("hybrid", "sql", "graph"))[0]

    assert default_task.provider_status == "skipped_non_executable"
    assert opt_in_task.provider_status == "ready"


def test_sql_provider_llm_compiler_opt_in_uses_settings_request_and_trace(monkeypatch) -> None:
    class _FakeOpenAIClient:
        requests = []

        def __init__(self, settings) -> None:
            self.settings = settings

        def create_response(self, request):
            self.requests.append(request)
            return SimpleNamespace(
                output_text="SQLQuery: SELECT COUNT(*) AS row_count FROM tbl_revenue",
                raw=SimpleNamespace(id="resp_sql_1"),
                usage=None,
            )

    class _Executor:
        def execute(self, schema, sql):
            return SQLExecutionResult(
                status="success",
                columns=("row_count",),
                rows=({"row_count": 2},),
                row_count=1,
                result_bytes=18,
            )

    monkeypatch.setattr(sql_compiler_module, "OpenAIClient", _FakeOpenAIClient)
    settings = Settings(
        openai_api_key="sk-test",
        structured_sql_compiler_mode="llm",
        llm_model="gpt-5-nano",
        llm_reasoning_effort="medium",
    )
    plan = _sql_plan("How many rows are in the revenue table?")
    task = tasks_from_plan(plan, executable_providers=("sql",))[0]
    provider = SQLProvider(settings=settings, executor=_Executor())

    result = provider.retrieve_provider_result(
        db=object(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={"structured_tables": [_table_context().__dict__]},
        query_plan=plan,
        retrieval_tasks=[task],
    )

    request = _FakeOpenAIClient.requests[0]
    compiler_trace = result.trace["compiler"]
    assert result.status == "success"
    assert request["model"] == "gpt-5-nano"
    assert request["reasoning"] == {"effort": "medium"}
    assert request["store"] is False
    assert compiler_trace["compiler_mode"] == "llm"
    assert compiler_trace["model_name"] == "gpt-5-nano"
    assert compiler_trace["compiler_call_id"] == "resp_sql_1"
    assert compiler_trace["fallback_used"] is False
    assert compiler_trace["prompt_hash"]
    assert compiler_trace["schema_context_hash"]
    assert result.trace["intent"]["intent_status"] == "allowed"
    assert result.trace["intent"]["allowed"] is True
    assert result.trace["schema_routing"]["selected_table_id"] == "tbl_revenue"
    assert result.trace["schema_routing"]["top1_score"] is not None
    assert "top1_top2_margin" in result.trace["schema_routing"]
    assert result.trace["schema_context"]["safe_to_raw_identifier_map"]


def test_sql_provider_llm_compiler_failure_does_not_fallback_to_heuristic(monkeypatch) -> None:
    class _FailingOpenAIClient:
        def __init__(self, settings) -> None:
            self.settings = settings

        def create_response(self, request):
            raise RuntimeError("upstream unavailable")

    class _Executor:
        def execute(self, schema, sql):
            raise AssertionError("executor should not run after compiler failure")

    monkeypatch.setattr(sql_compiler_module, "OpenAIClient", _FailingOpenAIClient)
    settings = Settings(
        openai_api_key="sk-test",
        structured_sql_compiler_mode="llm",
        llm_model="gpt-5-nano",
    )
    plan = _sql_plan("What is the total revenue in the table?")
    task = tasks_from_plan(plan, executable_providers=("sql",))[0]
    provider = SQLProvider(settings=settings, executor=_Executor())

    result = provider.retrieve_provider_result(
        db=object(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={"structured_tables": [_table_context().__dict__]},
        query_plan=plan,
        retrieval_tasks=[task],
    )

    compiler_trace = result.trace["compiler"]
    assert result.status == "compiler_failed"
    assert result.reason == "llm_compiler_failed:RuntimeError"
    assert compiler_trace["compiler_mode"] == "llm"
    assert compiler_trace["model_name"] == "gpt-5-nano"
    assert compiler_trace["compiler_call_id"]
    assert compiler_trace["fallback_used"] is False
    assert "execution" not in result.trace


def test_sql_provider_executes_through_router_with_fake_executor() -> None:
    class _Compiler:
        def compile(self, question, schema):
            return SQLDraft(
                status="success",
                sql=f"SELECT COUNT(*) AS row_count FROM {schema.safe_table_name}",
            )

    class _Executor:
        def execute(self, schema, sql):
            return SQLExecutionResult(
                status="success",
                columns=("row_count",),
                rows=({"row_count": 2},),
                row_count=1,
                result_bytes=18,
            )

    plan = QueryPlan(
        plan_id="plan_sql_router",
        original_query="How many rows are in the revenue table?",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_sql",
                purpose="structured_lookup",
                text="How many rows are in the revenue table?",
                provider="sql",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("sql",))
    provider = SQLProvider(compiler=_Compiler(), executor=_Executor())
    router = ProviderRouter(
        {"sql": provider},
        known_providers=("hybrid", "sql", "graph"),
        non_executable_providers=(),
        reranker_enabled=False,
    )

    result = router.retrieve(
        db=object(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={"structured_tables": [_table_context().__dict__]},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    assert result.provider_results[0].status == "success"
    assert result.provider_results[0].candidates[0].source_type == "sql_result"
    assert result.evidence[0].metadata["provider"] == "sql"
    assert result.evidence[0].metadata["fusion_policy"] == "pinned"


def test_sql_provider_rejects_multiple_sql_tasks_without_dropping_them() -> None:
    plan = QueryPlan(
        plan_id="plan_multiple_sql",
        original_query="Compare revenue and row count.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_sql_1",
                purpose="structured_lookup",
                text="How many rows are in the revenue table?",
                provider="sql",
            ),
            RetrievalUnit(
                unit_id="u_sql_2",
                purpose="structured_lookup",
                text="What is total revenue?",
                provider="sql",
            ),
        ),
    )
    tasks = tasks_from_plan(plan, executable_providers=("sql",))
    provider = SQLProvider()
    router = ProviderRouter(
        {"sql": provider},
        known_providers=("hybrid", "sql", "graph"),
        non_executable_providers=(),
        reranker_enabled=False,
    )

    result = router.retrieve(
        db=object(),
        query=plan.original_query,
        top_k=3,
        filters={},
        options={"structured_tables": [_table_context().__dict__]},
        query_plan=plan,
        retrieval_tasks=tasks,
    )

    provider_result = result.provider_results[0]
    assert provider_result.status == "unsupported_multi_table"
    assert provider_result.reason == "multiple_sql_tasks_unsupported"
    assert provider_result.candidates == ()
    assert provider_result.evidence == ()
    assert provider_result.trace["sql_task_ids"] == [task.task_id for task in tasks]


def test_duckdb_sandbox_critical_setting_failure_fails_closed() -> None:
    class _Connection:
        def execute(self, statement):
            if "enable_external_access" in statement:
                raise RuntimeError("setting unavailable")

    with pytest.raises(SQLSandboxConfigurationError) as exc_info:
        _apply_sandbox_settings(_Connection(), memory_limit=None)

    assert "sandbox_configuration_failed:enable_external_access" in str(exc_info.value)
    assert exc_info.value.warnings
    assert exc_info.value.trace["failed_settings"][0]["critical"] is True


def test_duckdb_sandbox_memory_limit_failure_is_noncritical_warning() -> None:
    class _Connection:
        def execute(self, statement):
            if "memory_limit" in statement:
                raise RuntimeError("setting unavailable")

    result = _apply_sandbox_settings(_Connection(), memory_limit="128MB")

    assert result.warnings == (
        "sandbox_setting_failed:memory_limit:RuntimeError:setting unavailable",
    )
    assert result.trace["failed_settings"][0]["critical"] is False


def test_duckdb_executor_enforces_result_byte_cap_when_available(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    schema = IdentifierNormalizer().normalize(
        SQLTableContext(
            table_id="tbl_big",
            raw_source_name="Big Table",
            display_name="Big Table",
            rows=({"Value": "x" * 1000},),
            columns=({"column_id": "col_value", "name": "Value", "data_type": "string"},),
            routing_text="Big Table Value",
        )
    )
    executor = DuckDBExecutor(duckdb_dir=tmp_path, max_rows=10, max_result_bytes=20)

    with pytest.raises(SQLResultCapExceeded):
        executor.execute(schema, "SELECT value FROM big_table LIMIT 10")


def _install_fake_sqlglot(monkeypatch) -> None:
    class Expression:
        def __init__(self, *children, **args) -> None:
            self._children = tuple(child for child in children if child is not None)
            self.args = dict(args)

        def walk(self):
            yield self
            for child in self._children:
                yield from child.walk()

        def find_all(self, cls):
            for item in self.walk():
                if isinstance(item, cls):
                    yield item

        def set(self, key, value) -> None:
            self.args[key] = value

        def sql(self, dialect=None) -> str:
            return self.args.get("sql", "SELECT fake FROM revenue_table")

    class Select(Expression):
        def __init__(self, expressions, table, sql) -> None:
            self.expressions = tuple(expressions)
            super().__init__(*self.expressions, table, expressions=self.expressions, sql=sql)

    class Star(Expression):
        pass

    class Table(Expression):
        def __init__(self, name) -> None:
            self.name = name
            super().__init__()

    class Column(Expression):
        def __init__(self, name, table="") -> None:
            self.name = name
            self.table = table
            self.this = Star() if name == "*" else name
            children = (self.this,) if isinstance(self.this, Expression) else ()
            super().__init__(*children, this=self.this)

    class Alias(Expression):
        def __init__(self, this, alias) -> None:
            self.this = this
            self.alias = alias
            super().__init__(this, this=this)

    class Func(Expression):
        function_name = None

        def sql_name(self) -> str:
            return self.function_name or self.__class__.__name__.upper()

    class AggFunc(Func):
        pass

    class Count(AggFunc):
        function_name = "COUNT"

        def __init__(self, this) -> None:
            self.this = this
            super().__init__(this, this=this)

    class Sum(AggFunc):
        function_name = "SUM"

        def __init__(self, this) -> None:
            self.this = this
            super().__init__(this, this=this)

    class Coalesce(Func):
        function_name = "COALESCE"

        def __init__(self, this) -> None:
            self.this = this
            super().__init__(this, this=this)

    class Anonymous(Func):
        def __init__(self, name) -> None:
            self.name = name
            super().__init__()

    class Literal(Expression):
        def __init__(self, this, *, is_int=True) -> None:
            self.this = this
            self.is_int = is_int
            super().__init__(this=this)

        @classmethod
        def number(cls, value):
            return cls(str(value), is_int=True)

    class Limit(Expression):
        def __init__(self, expression) -> None:
            self.expression = expression
            super().__init__(expression, expression=expression)

    exp = types.SimpleNamespace(
        AggFunc=AggFunc,
        Alias=Alias,
        Anonymous=Anonymous,
        Column=Column,
        Count=Count,
        Func=Func,
        Limit=Limit,
        Literal=Literal,
        Select=Select,
        Star=Star,
        Sum=Sum,
        Table=Table,
    )
    sqlglot = types.ModuleType("sqlglot")
    sqlglot.exp = exp

    def parse(sql, read=None):
        lowered = " ".join(sql.lower().split())
        table = Table("revenue_table")
        if lowered.startswith("select *"):
            return [Select((Star(),), table, "SELECT * FROM revenue_table")]
        if "coalesce" in lowered:
            return [
                Select(
                    (Alias(Coalesce(Column("revenue")), "revenue"),),
                    table,
                    "SELECT COALESCE(revenue, 0) AS revenue FROM revenue_table",
                )
            ]
        if "sum(*)" in lowered:
            return [
                Select(
                    (Alias(Sum(Star()), "sum_all"),),
                    table,
                    "SELECT SUM(*) AS sum_all FROM revenue_table",
                )
            ]
        return [
            Select(
                (Alias(Count(Star()), "row_count"),),
                table,
                "SELECT COUNT(*) AS row_count FROM revenue_table",
            )
        ]

    sqlglot.parse = parse
    monkeypatch.setitem(sys.modules, "sqlglot", sqlglot)
