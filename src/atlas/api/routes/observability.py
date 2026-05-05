from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from atlas.db.session import get_db
from atlas.observability.summary import build_observability_summary

router = APIRouter(prefix="/observability", tags=["observability"])


@router.get("/summary")
def observability_summary(db: Session = Depends(get_db)) -> dict[str, Any]:
    return build_observability_summary(db)
