from enum import StrEnum
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


class ErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    UPSTREAM_VECTOR_STORE_UNAVAILABLE = "UPSTREAM_VECTOR_STORE_UNAVAILABLE"
    UPSTREAM_LLM_UNAVAILABLE = "UPSTREAM_LLM_UNAVAILABLE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AtlasError(Exception):
    def __init__(
        self,
        error_code: ErrorCode,
        error_message: str,
        *,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> None:
        super().__init__(error_message)
        self.error_code = error_code
        self.error_message = error_message
        self.status_code = status_code
        self.details = details or {}
        self.trace_id = trace_id


async def atlas_error_handler(_request: Request, exc: AtlasError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": exc.error_code,
            "error_message": exc.error_message,
            "trace_id": exc.trace_id,
            "details": exc.details,
        },
    )


async def unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error_code": ErrorCode.INTERNAL_ERROR,
            "error_message": "Internal server error",
            "trace_id": None,
            "details": {"type": exc.__class__.__name__},
        },
    )
