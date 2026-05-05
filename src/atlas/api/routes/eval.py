from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from atlas.api.dependencies import get_eval_service
from atlas.db.session import get_db
from atlas.eval.service import EvalService, list_eval_runs, serialize_eval_run

router = APIRouter(prefix="/eval", tags=["eval"])


class EvalRunRequest(BaseModel):
    cases_path: str = "evals/smoke_cases.yaml"
    top_k: int = Field(default=8, ge=1, le=20)


@router.post("/run")
def run_eval(
    request: EvalRunRequest,
    db: Session = Depends(get_db),
    service: EvalService = Depends(get_eval_service),
) -> dict[str, Any]:
    eval_run = service.run_cases_file(db, cases_path=request.cases_path, top_k=request.top_k)
    return serialize_eval_run(eval_run)


@router.get("/{eval_run_id}")
def get_eval_run(
    eval_run_id: str,
    db: Session = Depends(get_db),
    service: EvalService = Depends(get_eval_service),
) -> dict[str, Any]:
    eval_run = service.get_eval_run(db, eval_run_id)
    if eval_run is None:
        raise HTTPException(status_code=404, detail="Eval run not found")
    return serialize_eval_run(eval_run)


@router.get("")
def list_recent_eval_runs(db: Session = Depends(get_db)) -> dict[str, Any]:
    return {
        "eval_runs": [
            {
                "eval_run_id": item.eval_run_id,
                "status": item.status,
                "total_cases": item.total_cases,
                "source_hits": item.source_hits,
                "confidence_hits": item.confidence_hits,
                "average_keyword_score": item.average_keyword_score,
                "average_latency_ms": item.average_latency_ms,
                "created_at": item.created_at.isoformat(),
                "finished_at": item.finished_at.isoformat() if item.finished_at else None,
            }
            for item in list_eval_runs(db)
        ]
    }
