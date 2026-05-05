from pathlib import Path

from atlas.core.config import Settings
from atlas.core.errors import AtlasError, ErrorCode


def allowed_document_roots(settings: Settings) -> list[Path]:
    roots = [
        Path(item.strip()).expanduser().resolve()
        for item in settings.document_roots.split(",")
        if item.strip()
    ]
    if not roots:
        raise AtlasError(
            ErrorCode.CONFIGURATION_ERROR,
            "ATLAS_DOCUMENT_ROOTS must contain at least one allowed document directory.",
            status_code=500,
        )
    return roots


def resolve_allowed_document_path(path_value: str, roots: list[Path]) -> Path:
    path = Path(path_value).expanduser().resolve()
    if any(_is_relative_to(path, root) for root in roots):
        return path

    raise AtlasError(
        ErrorCode.INVALID_REQUEST,
        "Document path is outside the configured ingestion roots.",
        status_code=400,
        details={
            "path": str(path),
            "allowed_roots": [str(root) for root in roots],
        },
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
