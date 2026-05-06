import json
from pathlib import Path

from pydantic import SecretStr

from atlas.core.config import Settings
from atlas.query_orchestrator.ontology import FinanceMetricOntology
from atlas.query_orchestrator.llm_planner import LLMQueryPlanner
from atlas.query_orchestrator.schema import Entity, Metric, Period, QueryPlan, RetrievalUnit
from atlas.query_orchestrator.service import QueryOrchestrator
from atlas.query_orchestrator.validator import QueryPlanValidator


ONTOLOGY_PATH = Path("configs/finance_metric_ontology.yaml")


class _FakeLLMPlanner:
    def __init__(self, plan: QueryPlan, *, available: bool = True) -> None:
        self._plan = plan
        self._available = available

    def available(self) -> bool:
        return self._available

    def plan(self, _query: str, *, validation_feedback=None) -> QueryPlan:
        return self._plan


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.output_text = json.dumps(payload)


class _FakeResponses:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.calls = []

    def create(self, **request):
        self.calls.append(request)
        index = min(len(self.calls) - 1, len(self.payloads) - 1)
        return _FakeResponse(self.payloads[index])


def test_fallback_plan_builds_grounded_units_without_openai_key() -> None:
    settings = Settings(
        openai_api_key=None,
        finance_metric_ontology_path=str(ONTOLOGY_PATH),
        query_planner_max_units=6,
    )
    orchestrator = QueryOrchestrator(settings=settings)

    plan = orchestrator.plan("What is the FY2018 capital expenditure amount for 3M?")

    assert plan.planner == "rule_based_fallback"
    assert plan.validation_status == "validated"
    assert plan.metrics[0].canonical_name == "capital_expenditure"
    assert any(unit.purpose == "metric_alias" for unit in plan.retrieval_units)
    assert all(unit.retrievers == ("hybrid",) for unit in plan.retrieval_units)
    assert all(unit.text for unit in plan.retrieval_units)


def test_ontology_canonicalizes_metric_aliases() -> None:
    ontology = FinanceMetricOntology.load(ONTOLOGY_PATH)

    metric = ontology.canonicalize("capex")

    assert metric is not None
    assert metric.canonical_name == "capital_expenditure"


def test_validator_rejects_ungrounded_entity_and_unknown_metric() -> None:
    ontology = FinanceMetricOntology.load(ONTOLOGY_PATH)
    validator = QueryPlanValidator(ontology)
    plan = QueryPlan(
        plan_id="plan_bad",
        original_query="What is the FY2018 capital expenditure amount for 3M?",
        entities=(),
        metrics=(),
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="What is the FY2018 capital expenditure amount for 3M?",
            ),
        ),
    ).model_copy(
        update={
            "entities": (Entity(value="Imaginary Corp", kind="company"),),
            "metrics": (Metric(canonical_name="made_up_metric"),),
        }
    )

    validation = validator.validate(plan)

    assert validation.ok is False
    assert "ungrounded_entity:Imaginary Corp" in validation.reasons
    assert "unknown_metric:made_up_metric" in validation.reasons


def test_validator_rejects_known_metric_not_grounded_in_query() -> None:
    ontology = FinanceMetricOntology.load(ONTOLOGY_PATH)
    validator = QueryPlanValidator(ontology)
    plan = QueryPlan(
        plan_id="plan_bad_metric",
        original_query="What was 3M revenue in FY2018?",
        metrics=(Metric(canonical_name="capital_expenditure"),),
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="What was 3M revenue in FY2018?",
            ),
        ),
    )

    validation = validator.validate(plan)

    assert validation.ok is False
    assert "ungrounded_metric:capital_expenditure" in validation.reasons


def test_validator_rejects_metric_when_llm_source_text_is_misleading() -> None:
    ontology = FinanceMetricOntology.load(ONTOLOGY_PATH)
    validator = QueryPlanValidator(ontology)
    plan = QueryPlan(
        plan_id="plan_bad_source_text",
        original_query="What was 3M revenue in FY2018?",
        metrics=(Metric(canonical_name="capital_expenditure", source_text="revenue"),),
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="What was 3M revenue in FY2018?",
            ),
        ),
    )

    validation = validator.validate(plan)

    assert validation.ok is False
    assert "ungrounded_metric:capital_expenditure" in validation.reasons


def test_validator_rejects_retrieval_unit_with_ungrounded_entity_or_period() -> None:
    ontology = FinanceMetricOntology.load(ONTOLOGY_PATH)
    validator = QueryPlanValidator(ontology)
    plan = QueryPlan(
        plan_id="plan_bad_unit",
        original_query="What is the FY2018 capital expenditure amount for 3M?",
        entities=(Entity(value="3M"),),
        periods=(Period(value="FY2018", normalized="2018"),),
        metrics=(Metric(canonical_name="capital_expenditure", aliases=("capex",)),),
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="Apple Inc FY2024 revenue",
            ),
        ),
    )

    validation = validator.validate(plan)

    assert validation.ok is False
    assert "ungrounded_unit_entity:u0:Apple Inc" in validation.reasons
    assert "ungrounded_unit_period:u0:2024" in validation.reasons


def test_validator_rejects_retrieval_unit_with_ungrounded_bare_company() -> None:
    ontology = FinanceMetricOntology.load(ONTOLOGY_PATH)
    validator = QueryPlanValidator(ontology)
    plan = QueryPlan(
        plan_id="plan_bad_bare_company",
        original_query="What was 3M revenue in FY2018?",
        entities=(Entity(value="3M"),),
        periods=(Period(value="FY2018", normalized="2018"),),
        metrics=(Metric(canonical_name="revenue"),),
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="Tesla 2018 revenue",
            ),
        ),
    )

    validation = validator.validate(plan)

    assert validation.ok is False
    assert "ungrounded_unit_entity:u0:tesla" in validation.reasons


def test_validator_rejects_retrieval_unit_with_untriggered_metric_alias() -> None:
    ontology = FinanceMetricOntology.load(ONTOLOGY_PATH)
    validator = QueryPlanValidator(ontology)
    plan = QueryPlan(
        plan_id="plan_bad_alias",
        original_query="What was 3M revenue in FY2018?",
        entities=(Entity(value="3M"),),
        periods=(Period(value="FY2018", normalized="2018"),),
        metrics=(Metric(canonical_name="revenue"),),
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="3M 2018 cash and cash equivalents",
                should_terms=("cash and cash equivalents",),
            ),
        ),
    )

    validation = validator.validate(plan)

    assert validation.ok is False
    assert "ungrounded_metric_alias:u0:cash and cash equivalents" in validation.reasons


def test_validator_rejects_disabled_sql_or_graph_provider() -> None:
    ontology = FinanceMetricOntology.load(ONTOLOGY_PATH)
    validator = QueryPlanValidator(ontology, enabled_providers=("hybrid",))
    plan = QueryPlan(
        plan_id="plan_sql",
        original_query="Compare Apple and Microsoft 2023 R&D.",
        retrieval_units=(
            RetrievalUnit(
                unit_id="u_sql",
                purpose="numerical_aggregation",
                text="Apple Microsoft R&D 2023",
                retrievers=("sql",),
            ),
        ),
    )

    validation = validator.validate(plan)

    assert validation.ok is False
    assert "provider_not_enabled:u_sql:sql" in validation.reasons


def test_llm_planner_rejects_unknown_raw_metric_before_validation() -> None:
    settings = Settings(
        openai_api_key=SecretStr("test-key"),
        finance_metric_ontology_path=str(ONTOLOGY_PATH),
    )
    ontology = FinanceMetricOntology.load(ONTOLOGY_PATH)
    planner = LLMQueryPlanner(settings=settings, ontology=ontology)

    raw = {
        "standalone_query": "What is the FY2018 capital expenditure amount for 3M?",
        "query_type": "financial_numeric_fact",
        "entities": [],
        "periods": [],
        "metrics": [{"canonical_name": "made_up_metric"}],
        "metadata_filter": {},
        "retrieval_units": [
            {
                "unit_id": "u0",
                "purpose": "original",
                "text": "What is the FY2018 capital expenditure amount for 3M?",
                "retrievers": ["hybrid"],
                "metadata_filter": {},
                "must_have_terms": [],
                "should_terms": [],
                "top_k": 10,
                "weight": 1.0,
                "lane_weights": {},
                "metadata": {},
            }
        ],
    }

    try:
        planner._plan_from_raw(raw["standalone_query"], raw)
    except ValueError as exc:
        assert "made_up_metric" in str(exc)
    else:
        raise AssertionError("expected unknown metric to be rejected")


def test_llm_planner_rejects_blank_metric_with_unknown_source_text() -> None:
    settings = Settings(
        openai_api_key=SecretStr("test-key"),
        finance_metric_ontology_path=str(ONTOLOGY_PATH),
    )
    ontology = FinanceMetricOntology.load(ONTOLOGY_PATH)
    planner = LLMQueryPlanner(settings=settings, ontology=ontology)

    raw = {
        "standalone_query": "What was 3M revenue in FY2018?",
        "query_type": "financial_numeric_fact",
        "entities": [],
        "periods": [],
        "metrics": [{"canonical_name": "", "source_text": "EBITDA"}],
        "metadata_filter": {},
        "retrieval_units": [
            {
                "unit_id": "u0",
                "purpose": "original",
                "text": "What was 3M revenue in FY2018?",
                "retrievers": ["hybrid"],
                "metadata_filter": {},
                "must_have_terms": [],
                "should_terms": [],
                "top_k": 10,
                "weight": 1.0,
            }
        ],
    }

    try:
        planner._plan_from_raw(raw["standalone_query"], raw)
    except ValueError as exc:
        assert "EBITDA" in str(exc)
    else:
        raise AssertionError("expected blank unknown metric to be rejected")


def test_llm_planner_retries_with_validation_feedback(monkeypatch) -> None:
    invalid_payload = {
        "standalone_query": "What was 3M revenue in FY2018?",
        "query_type": "fact_lookup",
        "entities": [],
        "periods": [],
        "metrics": [],
        "filters": {},
        "retrieval_units": [],
    }
    valid_payload = {
        "standalone_query": "What was 3M revenue in FY2018?",
        "query_type": "fact_lookup",
        "entities": [],
        "periods": [],
        "metrics": [],
        "metadata_filter": {},
        "retrieval_units": [
            {
                "unit_id": "u0",
                "purpose": "original",
                "text": "What was 3M revenue in FY2018?",
                "retrievers": ["hybrid"],
                "metadata_filter": {},
                "must_have_terms": [],
                "should_terms": [],
                "top_k": 10,
                "weight": 1.0,
            }
        ],
    }
    created_clients = []

    class _FakeOpenAI:
        def __init__(self, **_kwargs) -> None:
            self.responses = _FakeResponses([invalid_payload, valid_payload])
            created_clients.append(self)

    monkeypatch.setattr("atlas.query_orchestrator.llm_planner.OpenAI", _FakeOpenAI)
    settings = Settings(
        openai_api_key=SecretStr("test-key"),
        finance_metric_ontology_path=str(ONTOLOGY_PATH),
    )
    ontology = FinanceMetricOntology.load(ONTOLOGY_PATH)
    planner = LLMQueryPlanner(settings=settings, ontology=ontology)

    plan = planner.plan("What was 3M revenue in FY2018?")

    calls = created_clients[0].responses.calls
    retriever_enum = calls[0]["text"]["format"]["schema"]["properties"]["retrieval_units"][
        "items"
    ]["properties"]["retrievers"]["items"]["enum"]
    assert plan.retrieval_units[0].retrievers == ("hybrid",)
    assert retriever_enum == ["hybrid"]
    assert len(calls) == 2
    assert "filters is not supported" in calls[1]["input"]


def test_ontology_does_not_match_cash_inside_operating_cash_flow() -> None:
    ontology = FinanceMetricOntology.load(ONTOLOGY_PATH)

    mentions = [
        metric.canonical_name
        for metric, _ in ontology.find_mentions("operating cash flow")
    ]

    assert mentions == ["operating_cash_flow"]


def test_orchestrator_uses_valid_llm_plan() -> None:
    settings = Settings(
        openai_api_key=SecretStr("test-key"),
        finance_metric_ontology_path=str(ONTOLOGY_PATH),
        query_planner_max_units=6,
    )
    llm_plan = QueryPlan(
        plan_id="plan_llm",
        original_query="What is the FY2018 capital expenditure amount for 3M?",
        standalone_query="What is the FY2018 capital expenditure amount for 3M?",
        metrics=(),
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="What is the FY2018 capital expenditure amount for 3M?",
            ),
        ),
        planner="llm_structured",
    )
    orchestrator = QueryOrchestrator(
        settings=settings,
        llm_planner=_FakeLLMPlanner(llm_plan),
    )

    plan = orchestrator.plan(llm_plan.original_query)

    assert plan.plan_id == "plan_llm"
    assert plan.validation_status == "validated"


def test_orchestrator_falls_back_when_llm_plan_fails_validation() -> None:
    settings = Settings(
        openai_api_key=SecretStr("test-key"),
        finance_metric_ontology_path=str(ONTOLOGY_PATH),
        query_planner_max_units=6,
    )
    bad_llm_plan = QueryPlan(
        plan_id="plan_bad_llm",
        original_query="What is the FY2018 capital expenditure amount for 3M?",
        entities=(),
        retrieval_units=(
            RetrievalUnit(
                unit_id="u0",
                purpose="original",
                text="What is the FY2018 capital expenditure amount for 3M?",
            ),
        ),
        planner="llm_structured",
    ).model_copy(update={"entities": (Entity(value="Imaginary Corp"),)})
    orchestrator = QueryOrchestrator(
        settings=settings,
        llm_planner=_FakeLLMPlanner(bad_llm_plan),
    )

    plan = orchestrator.plan("What is the FY2018 capital expenditure amount for 3M?")

    assert plan.planner == "rule_based_fallback"
    assert plan.validation_status == "validated"
    assert plan.metadata["fallback_reason"] == "llm_validation_failed"
    assert "ungrounded_entity:Imaginary Corp" in plan.metadata["llm_rejection"]["reasons"]
