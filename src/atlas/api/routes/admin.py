from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sqlalchemy import delete
from sqlalchemy.orm import Session

from atlas.api.dependencies import settings_dependency
from atlas.core.config import Settings
from atlas.db.models import (
    Chunk,
    Document,
    EvalResult,
    EvalRun,
    GenerationEvent,
    IngestionRun,
    QueryRun,
    RetrievalEvent,
)
from atlas.db.session import get_db
from atlas.vector.collections import ensure_chunk_collection
from atlas.vector.qdrant_client import get_qdrant_client

router = APIRouter(prefix="/admin", tags=["admin"])


class ResetDevDataRequest(BaseModel):
    scope: Literal["traces", "all"] = "traces"
    confirm: str


@router.post("/reset-dev-data")
def reset_dev_data(
    request: ResetDevDataRequest,
    db: Session = Depends(get_db),
    qdrant: QdrantClient = Depends(get_qdrant_client),
    settings: Settings = Depends(settings_dependency),
) -> dict[str, object]:
    if request.confirm != "RESET_DEV_DATA":
        return {
            "status": "rejected",
            "reason": "confirm must be RESET_DEV_DATA",
        }

    deleted = _delete_trace_and_eval(db)

    if request.scope == "all":
        deleted.update(_delete_documents(db))
        db.commit()
        if qdrant.collection_exists(settings.qdrant_collection):
            qdrant.delete_collection(settings.qdrant_collection)
        ensure_chunk_collection(qdrant, settings)
    else:
        db.commit()

    return {
        "status": "ok",
        "scope": request.scope,
        "deleted": deleted,
    }


def _delete_trace_and_eval(db: Session) -> dict[str, int]:
    deleted: dict[str, int] = {}
    for model in [EvalResult, EvalRun, GenerationEvent, RetrievalEvent, QueryRun]:
        result = db.execute(delete(model))
        deleted[model.__tablename__] = result.rowcount or 0
    return deleted


def _delete_documents(db: Session) -> dict[str, int]:
    deleted: dict[str, int] = {}
    for model in [Chunk, Document, IngestionRun]:
        result = db.execute(delete(model))
        deleted[model.__tablename__] = result.rowcount or 0
    return deleted
