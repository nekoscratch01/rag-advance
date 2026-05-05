from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from atlas.db.models import (
    Chunk,
    Document,
    EvalRun,
    GenerationEvent,
    IngestionRun,
    QueryRun,
    RetrievalEvent,
)


def build_observability_summary(db: Session) -> dict[str, Any]:
    query_count = _count(db, QueryRun)
    generation_count = _count(db, GenerationEvent)
    completed_generations = db.scalar(
        select(func.count()).select_from(GenerationEvent).where(GenerationEvent.status == "completed")
    )
    failed_generations = db.scalar(
        select(func.count()).select_from(GenerationEvent).where(GenerationEvent.status == "failed")
    )

    latency = db.execute(
        select(
            func.avg(QueryRun.latency_ms),
            func.min(QueryRun.latency_ms),
            func.max(QueryRun.latency_ms),
        ).where(QueryRun.latency_ms.is_not(None))
    ).one()
    generation_latency = db.execute(
        select(
            func.avg(GenerationEvent.latency_ms),
            func.min(GenerationEvent.latency_ms),
            func.max(GenerationEvent.latency_ms),
        ).where(GenerationEvent.latency_ms.is_not(None))
    ).one()
    tokens = db.execute(
        select(
            func.coalesce(func.sum(GenerationEvent.input_tokens), 0),
            func.coalesce(func.sum(GenerationEvent.output_tokens), 0),
        )
    ).one()

    recent_queries = db.scalars(
        select(QueryRun).order_by(QueryRun.created_at.desc()).limit(10)
    ).all()

    return {
        "storage": {
            "documents": _count(db, Document),
            "chunks": _count(db, Chunk),
            "ingestion_runs": _count(db, IngestionRun),
        },
        "queries": {
            "query_runs": query_count,
            "retrieval_events": _count(db, RetrievalEvent),
            "by_confidence": _group_count(db, QueryRun.confidence),
            "recent": [
                {
                    "query_id": item.query_id,
                    "trace_id": item.trace_id,
                    "confidence": item.confidence,
                    "latency_ms": item.latency_ms,
                    "created_at": item.created_at.isoformat(),
                    "question": item.user_query,
                }
                for item in recent_queries
            ],
        },
        "generation": {
            "generation_events": generation_count,
            "completed": completed_generations or 0,
            "failed": failed_generations or 0,
            "by_model": _group_count(db, GenerationEvent.model_name),
            "input_tokens": int(tokens[0] or 0),
            "output_tokens": int(tokens[1] or 0),
        },
        "eval": {
            "eval_runs": _count(db, EvalRun),
            "by_status": _group_count(db, EvalRun.status),
        },
        "latency": {
            "query_ms": _latency_payload(latency),
            "generation_ms": _latency_payload(generation_latency),
        },
    }


def _count(db: Session, model) -> int:
    return int(db.scalar(select(func.count()).select_from(model)) or 0)


def _group_count(db: Session, column) -> dict[str, int]:
    rows = db.execute(select(column, func.count()).group_by(column)).all()
    return {str(key or "unknown"): int(count) for key, count in rows}


def _latency_payload(row) -> dict[str, int | None]:
    avg_value, min_value, max_value = row
    return {
        "avg": int(avg_value) if avg_value is not None else None,
        "min": int(min_value) if min_value is not None else None,
        "max": int(max_value) if max_value is not None else None,
    }
