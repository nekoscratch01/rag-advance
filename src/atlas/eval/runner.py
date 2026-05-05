import argparse
import time
from typing import Any

import httpx

from atlas.eval.metrics import (
    answer_metric_details,
    dense_retrieval_metrics,
    expected_confidence_hit,
    keyword_hit,
    source_hit,
)
from atlas.eval.service import EvalCase, load_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Atlas eval cases.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--cases", default="evals/smoke_cases.yaml")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    results = []

    with httpx.Client(timeout=120) as client:
        for case in cases:
            started = time.perf_counter()
            latency_ms = int((time.perf_counter() - started) * 1000)
            try:
                request_payload: dict[str, Any] = {"query": case.question}
                if args.top_k is not None:
                    request_payload["top_k"] = args.top_k
                response = client.post(
                    f"{args.base_url}/v1/query",
                    json=request_payload,
                )
                latency_ms = int((time.perf_counter() - started) * 1000)
                response.raise_for_status()
                payload: dict[str, Any] = response.json()
            except httpx.HTTPStatusError as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                results.append(_failed_result(case, latency_ms, f"http_{exc.response.status_code}"))
                continue
            except httpx.RequestError as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                results.append(_failed_result(case, latency_ms, exc.__class__.__name__))
                continue

            trace = _fetch_trace(client, args.base_url, payload.get("query_id"))
            generation = trace.get("generation", [{}])
            generation_event = generation[0] if generation else {}
            retrieved_top_k = _retrieved_top_k_from_trace(trace)
            retrieval_details = dense_retrieval_metrics(
                retrieved_top_k,
                case.expected_evidence,
                case.expected_sources,
            )
            answer_details = answer_metric_details(payload.get("answer"), case.expected_answer)
            results.append(
                {
                    "id": case.id,
                    "question": case.question,
                    "source_hit": source_hit(
                        payload.get("citations", []),
                        case.expected_sources,
                    ),
                    "confidence_hit": expected_confidence_hit(
                        payload.get("confidence"),
                        case.expected_confidence,
                    ),
                    "keyword_score": keyword_hit(
                        payload.get("answer", ""),
                        case.expected_keywords,
                    ),
                    "latency_ms": latency_ms,
                    "confidence": payload.get("confidence"),
                    "expected_confidence": case.expected_confidence,
                    "query_id": payload.get("query_id"),
                    "trace_id": payload.get("trace_id"),
                    "input_tokens": generation_event.get("input_tokens"),
                    "output_tokens": generation_event.get("output_tokens"),
                    "generation_latency_ms": generation_event.get("latency_ms"),
                    "retrieved_top_k": retrieved_top_k,
                    **retrieval_details,
                    **answer_details,
                }
            )

    total = len(results)
    source_hits = sum(1 for item in results if item["source_hit"])
    confidence_hits = sum(1 for item in results if item["confidence_hit"])
    avg_keyword = sum(item["keyword_score"] for item in results) / total if total else 0
    avg_latency = sum(item["latency_ms"] for item in results) / total if total else 0
    total_input_tokens = sum(item["input_tokens"] or 0 for item in results)
    total_output_tokens = sum(item["output_tokens"] or 0 for item in results)
    retrieval_doc_hit = _aggregate_bool(results, "retrieval_doc_hit")
    retrieval_page_hit = _aggregate_bool(results, "retrieval_page_hit")
    retrieval_doc_mrr = _average_metric(results, "retrieval_doc_mrr")
    retrieval_page_mrr = _average_metric(results, "retrieval_page_mrr")
    answer_contains = _aggregate_bool(results, "answer_gold_contains")
    answer_numeric = _aggregate_bool(results, "answer_numeric_match")
    failures = [
        item
        for item in results
        if item.get("error")
        or not item["source_hit"]
        or not item["confidence_hit"]
        or item["keyword_score"] < 0.5
        or item.get("retrieval_doc_hit") is False
        or item.get("retrieval_page_hit") is False
        or item.get("answer_gold_contains") is False
        or item.get("answer_numeric_match") is False
    ]

    print("Atlas Eval")
    print(f"Total cases: {total}")
    top_k_label = args.top_k if args.top_k is not None else "default"
    print(f"Dense retrieval doc hit@{top_k_label}: {_format_hits(retrieval_doc_hit)}")
    print(f"Dense retrieval page hit@{top_k_label}: {_format_hits(retrieval_page_hit)}")
    print(f"Dense retrieval doc MRR: {_format_average(retrieval_doc_mrr)}")
    print(f"Dense retrieval page MRR: {_format_average(retrieval_page_mrr)}")
    print(f"Answer gold contains: {_format_hits(answer_contains)}")
    print(f"Answer numeric match: {_format_hits(answer_numeric)}")
    print(f"Citation source hit: {source_hits}/{total}")
    print(f"Confidence hit: {confidence_hits}/{total}")
    print(f"Average keyword score: {avg_keyword:.2f}")
    print(f"Average latency: {avg_latency:.0f} ms")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print("")
    print("Cases:")
    for item in results:
        print(
            f"- {item['id']}: source_hit={item['source_hit']} "
            f"retrieval_doc_hit={_format_case_metric(item.get('retrieval_doc_hit'))} "
            f"retrieval_page_hit={_format_case_metric(item.get('retrieval_page_hit'))} "
            f"doc_mrr={_format_case_float(item.get('retrieval_doc_mrr'))} "
            f"page_mrr={_format_case_float(item.get('retrieval_page_mrr'))} "
            f"answer_gold_contains={_format_case_metric(item.get('answer_gold_contains'))} "
            f"answer_numeric_match={_format_case_metric(item.get('answer_numeric_match'))} "
            f"confidence={item['confidence']} expected={item['expected_confidence']} "
            f"confidence_hit={item['confidence_hit']} "
            f"keyword_score={item['keyword_score']:.2f} "
            f"latency_ms={item['latency_ms']} "
            f"tokens={item['input_tokens']}/{item['output_tokens']} "
            f"query_id={item['query_id']}"
        )

    if failures:
        print("")
        print("Top failures:")
        for item in failures:
            reasons = []
            if not item["source_hit"]:
                reasons.append("source_miss")
            if not item["confidence_hit"]:
                reasons.append("confidence_miss")
            if item["keyword_score"] < 0.5:
                reasons.append("keyword_low")
            if item.get("retrieval_doc_hit") is False:
                reasons.append("retrieval_doc_miss")
            if item.get("retrieval_page_hit") is False:
                reasons.append("retrieval_page_miss")
            if item.get("answer_gold_contains") is False:
                reasons.append("answer_gold_missing")
            if item.get("answer_numeric_match") is False:
                reasons.append("answer_numeric_miss")
            if item.get("error"):
                reasons.append(str(item["error"]))
            print(
                f"- {item['id']} ({', '.join(reasons)}): "
                f"question={item['question']} query_id={item['query_id']}"
            )


def _fetch_trace(client: httpx.Client, base_url: str, query_id: str | None) -> dict[str, Any]:
    if not query_id:
        return {}
    response = client.get(f"{base_url}/v1/query/{query_id}/trace")
    if response.status_code != 200:
        return {}
    return response.json()


def _failed_result(case: EvalCase, latency_ms: int, error: str) -> dict[str, Any]:
    return {
        "id": case.id,
        "question": case.question,
        "source_hit": False,
        "confidence_hit": False,
        "keyword_score": 0.0,
        "latency_ms": latency_ms,
        "confidence": None,
        "expected_confidence": case.expected_confidence,
        "query_id": None,
        "trace_id": None,
        "input_tokens": None,
        "output_tokens": None,
        "generation_latency_ms": None,
        "retrieved_top_k": [],
        "retrieval_doc_hit": False if case.expected_evidence or case.expected_sources else None,
        "retrieval_page_hit": False if case.expected_evidence else None,
        "retrieval_doc_mrr": 0.0 if case.expected_evidence or case.expected_sources else None,
        "retrieval_page_mrr": 0.0 if case.expected_evidence else None,
        "first_doc_match_rank": None,
        "first_page_match_rank": None,
        **answer_metric_details(None, case.expected_answer),
        "error": error,
    }


def _retrieved_top_k_from_trace(trace: dict[str, Any]) -> list[dict[str, Any]]:
    items = trace.get("retrieval", {}).get("top_k", [])
    retrieved: list[dict[str, Any]] = []
    for item in items:
        retrieved.append(
            {
                "rank": item.get("rank"),
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "source_title": item.get("source_title"),
                "source_uri": item.get("source_uri"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "score": item.get("retrieval_score"),
                "retrieval_score": item.get("retrieval_score"),
                "retriever_type": item.get("retriever_type"),
            }
        )
    return retrieved


def _aggregate_bool(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [item.get(key) for item in results if item.get(key) is not None]
    hits = sum(1 for value in values if value is True)
    total = len(values)
    return {"hits": hits, "total": total, "rate": hits / total if total else None}


def _average_metric(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(item[key]) for item in results if item.get(key) is not None]
    return {"total": len(values), "average": sum(values) / len(values) if values else None}


def _format_hits(metric: dict[str, Any]) -> str:
    if metric["total"] == 0:
        return "n/a"
    return f"{metric['hits']}/{metric['total']} ({metric['rate']:.2f})"


def _format_average(metric: dict[str, Any]) -> str:
    if metric["total"] == 0:
        return "n/a"
    return f"{metric['average']:.3f}"


def _format_case_metric(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _format_case_float(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    main()
