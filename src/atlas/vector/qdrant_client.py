from functools import lru_cache

from qdrant_client import QdrantClient

from atlas.core.config import get_settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url)


def check_qdrant() -> bool:
    client = get_qdrant_client()
    client.get_collections()
    return True
