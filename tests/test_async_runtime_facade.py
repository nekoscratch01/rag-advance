from __future__ import annotations

import asyncio
from types import MethodType

from fastapi.testclient import TestClient
import pytest

from atlas.api.dependencies import get_query_runtime
from atlas.main import create_app
from atlas.query_runtime.service import QueryResult, QueryRuntime


class _FakeOwnedSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_query_runtime_arun_uses_owned_sync_session(monkeypatch) -> None:
    session = _FakeOwnedSession()

    def session_factory():
        return session

    monkeypatch.setattr("atlas.db.session.SessionLocal", session_factory)
    runtime = object.__new__(QueryRuntime)
    seen = {}

    def run(self, db, *, query, top_k, filters, options=None):
        seen["db"] = db
        seen["query"] = query
        seen["top_k"] = top_k
        seen["filters"] = filters
        seen["options"] = options
        return QueryResult(
            query_id="q_async",
            trace_id="tr_async",
            answer="answer",
            confidence="supported",
            citations=[],
            details={},
        )

    runtime.run = MethodType(run, runtime)

    result = asyncio.run(
        runtime.arun(
            query="hello",
            top_k=3,
            filters={"company": "3M"},
            options={"return_trace": True},
        )
    )

    assert result.query_id == "q_async"
    assert seen == {
        "db": session,
        "query": "hello",
        "top_k": 3,
        "filters": {"company": "3M"},
        "options": {"return_trace": True},
    }
    assert session.closed is True


def test_query_runtime_arun_rejects_caller_owned_sync_session() -> None:
    runtime = object.__new__(QueryRuntime)

    with pytest.raises(RuntimeError, match="caller-owned sync Session"):
        asyncio.run(
            runtime.arun(
                _FakeOwnedSession(),
                query="hello",
                top_k=3,
                filters={},
                options={},
            )
        )


def test_query_endpoint_prefers_runtime_arun() -> None:
    class RuntimeWithAsyncFacade:
        def __init__(self) -> None:
            self.calls = []

        async def arun(self, *, query, top_k, filters, options=None):
            self.calls.append(
                {
                    "query": query,
                    "top_k": top_k,
                    "filters": filters,
                    "options": options,
                }
            )
            return QueryResult(
                query_id="q_api_async",
                trace_id="tr_api_async",
                answer="async answer",
                confidence="supported",
                citations=[],
                details={"kept": True},
            )

    runtime = RuntimeWithAsyncFacade()
    app = create_app()
    app.dependency_overrides[get_query_runtime] = lambda: runtime
    client = TestClient(app)

    response = client.post(
        "/v1/query",
        json={
            "query": "What changed?",
            "top_k": 2,
            "filters": {"company": "3M"},
            "options": {"return_trace": True},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_id"] == "q_api_async"
    assert payload["details"] == {"kept": True}
    assert runtime.calls == [
        {
            "query": "What changed?",
            "top_k": 2,
            "filters": {"company": "3M"},
            "options": {"return_trace": True},
        }
    ]


def test_query_endpoint_fallback_closes_owned_sync_session(monkeypatch) -> None:
    class RuntimeWithoutAsyncFacade:
        def __init__(self) -> None:
            self.calls = []

        def run(self, db, *, query, top_k, filters, options=None):
            self.calls.append(
                {
                    "db": db,
                    "query": query,
                    "top_k": top_k,
                    "filters": filters,
                    "options": options,
                }
            )
            return QueryResult(
                query_id="q_api_sync",
                trace_id="tr_api_sync",
                answer="sync answer",
                confidence="supported",
                citations=[],
                details={"kept": True},
            )

    session = _FakeOwnedSession()
    monkeypatch.setattr("atlas.db.session.SessionLocal", lambda: session)
    runtime = RuntimeWithoutAsyncFacade()
    app = create_app()
    app.dependency_overrides[get_query_runtime] = lambda: runtime
    client = TestClient(app)

    response = client.post(
        "/v1/query",
        json={
            "query": "What changed?",
            "top_k": 2,
            "filters": {"company": "3M"},
            "options": {"return_trace": True},
        },
    )

    assert response.status_code == 200
    assert response.json()["query_id"] == "q_api_sync"
    assert runtime.calls[0]["db"] is session
    assert session.closed is True
