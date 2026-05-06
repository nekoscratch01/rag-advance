import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response

from atlas.api.routes import admin, documents, eval, health, observability, query
from atlas.core.config import get_settings
from atlas.core.errors import AtlasError, atlas_error_handler, unhandled_error_handler
from atlas.core.logging import configure_logging
from atlas.db.session import init_db
from atlas.vector.collections import ensure_chunk_collection
from atlas.vector.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    try:
        init_db()
        ensure_chunk_collection(get_qdrant_client(), settings)
        logger.info("Atlas storage initialized")
    except Exception as exc:
        logger.warning("Atlas storage initialization skipped: %s", exc.__class__.__name__)
    yield


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

    @app.get("/")
    def root() -> dict[str, object]:
        return {
            "name": settings.app_name,
            "version": "1.0.0",
            "status_endpoint": f"{settings.api_prefix}/health",
            "docs": "/docs",
            "endpoints": [
                f"GET {settings.api_prefix}/health",
                f"POST {settings.api_prefix}/documents/ingest",
                f"POST {settings.api_prefix}/query",
                f"POST {settings.api_prefix}/retrieve",
                f"GET {settings.api_prefix}/query/{{query_id}}",
                f"GET {settings.api_prefix}/query/{{query_id}}/trace",
                f"POST {settings.api_prefix}/eval/run",
                f"GET {settings.api_prefix}/eval/{{eval_run_id}}",
                f"GET {settings.api_prefix}/eval",
                f"GET {settings.api_prefix}/observability/summary",
                f"POST {settings.api_prefix}/admin/reset-dev-data",
            ],
        }

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    app.add_exception_handler(AtlasError, atlas_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(documents.router, prefix=settings.api_prefix)
    app.include_router(query.router, prefix=settings.api_prefix)
    app.include_router(query.retrieve_router, prefix=settings.api_prefix)
    app.include_router(eval.router, prefix=settings.api_prefix)
    app.include_router(observability.router, prefix=settings.api_prefix)
    app.include_router(admin.router, prefix=settings.api_prefix)
    return app


app = create_app()
