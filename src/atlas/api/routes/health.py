from fastapi import APIRouter

from atlas.db.session import check_db
from atlas.vector.qdrant_client import check_qdrant

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    postgres_status = _status(check_db)
    qdrant_status = _status(check_qdrant)
    status = "ok" if postgres_status == "ok" and qdrant_status == "ok" else "degraded"
    return {
        "status": status,
        "postgres": postgres_status,
        "qdrant": qdrant_status,
    }


def _status(check) -> str:
    try:
        check()
        return "ok"
    except Exception:
        return "unavailable"
