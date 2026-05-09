from functools import lru_cache

from qdrant_client import QdrantClient

from atlas.backends import BackendBuildContext, build_vector_store
from atlas.core.config import get_settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return build_vector_store(settings.vector_store_backend, BackendBuildContext(settings=settings))


def check_qdrant() -> bool:
    client = get_qdrant_client()
    client.get_collections()
    return True
