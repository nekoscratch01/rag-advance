from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from atlas.api.dependencies import get_ingestion_service
from atlas.db.session import get_db
from atlas.ingestion.service import IngestionService

router = APIRouter(prefix="/documents", tags=["documents"])


class IngestRequest(BaseModel):
    paths: list[str] = Field(min_length=1)
    source_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.post("/ingest")
def ingest_documents(
    request: IngestRequest,
    db: Session = Depends(get_db),
    service: IngestionService = Depends(get_ingestion_service),
) -> dict[str, Any]:
    result = service.ingest_paths(
        db,
        paths=request.paths,
        source_uri=request.source_uri,
        metadata=request.metadata,
    )
    return {
        "ingestion_run_id": result.ingestion_run_id,
        "documents": [
            {
                "document_id": item.document_id,
                "title": item.title,
                "status": item.status,
                "chunk_count": item.chunk_count,
                "error_message": item.error_message,
            }
            for item in result.documents
        ],
    }
