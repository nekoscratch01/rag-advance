from qdrant_client import QdrantClient, models

from atlas.core.config import Settings, bm25_sparse_enabled
from atlas.core.errors import AtlasError, ErrorCode


def ensure_chunk_collection(client: QdrantClient, settings: Settings) -> None:
    if client.collection_exists(settings.qdrant_collection):
        _validate_collection(client, settings)
        return

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=_vectors_config(settings),
        sparse_vectors_config=_sparse_vectors_config(settings),
    )


def _validate_collection(client: QdrantClient, settings: Settings) -> None:
    info = client.get_collection(settings.qdrant_collection)
    if bm25_sparse_enabled(settings):
        _validate_v1_hybrid_collection(info, settings)
        return

    actual_size = _legacy_collection_vector_size(info)
    if actual_size != settings.embedding_dim:
        raise AtlasError(
            ErrorCode.CONFIGURATION_ERROR,
            "Existing Qdrant collection vector size does not match the configured embedding model.",
            status_code=500,
            details={
                "collection": settings.qdrant_collection,
                "expected_dim": settings.embedding_dim,
                "actual_dim": actual_size,
                "embedding_model": settings.embedding_model,
            },
        )


def _validate_v1_hybrid_collection(info, settings: Settings) -> None:
    actual_size = _named_collection_vector_size(info, settings.qdrant_dense_vector_name)
    if actual_size != settings.embedding_dim:
        raise AtlasError(
            ErrorCode.CONFIGURATION_ERROR,
            "Existing Qdrant collection dense vector size does not match the configured embedding model.",
            status_code=500,
            details={
                "collection": settings.qdrant_collection,
                "dense_vector_name": settings.qdrant_dense_vector_name,
                "expected_dim": settings.embedding_dim,
                "actual_dim": actual_size,
                "embedding_model": settings.embedding_model,
            },
        )

    sparse_params = _named_sparse_vector_params(info, settings.qdrant_sparse_vector_name)
    if sparse_params is None:
        raise AtlasError(
            ErrorCode.CONFIGURATION_ERROR,
            "Existing Qdrant collection is missing the configured BM25 sparse vector.",
            status_code=500,
            details={
                "collection": settings.qdrant_collection,
                "sparse_vector_name": settings.qdrant_sparse_vector_name,
            },
        )

    modifier = getattr(sparse_params, "modifier", None)
    if modifier != models.Modifier.IDF:
        raise AtlasError(
            ErrorCode.CONFIGURATION_ERROR,
            "Existing Qdrant collection BM25 sparse vector must use the IDF modifier.",
            status_code=500,
            details={
                "collection": settings.qdrant_collection,
                "sparse_vector_name": settings.qdrant_sparse_vector_name,
                "expected_modifier": models.Modifier.IDF.value,
                "actual_modifier": getattr(modifier, "value", modifier),
            },
        )


def _vectors_config(settings: Settings):
    dense_params = models.VectorParams(
        size=settings.embedding_dim,
        distance=models.Distance.COSINE,
    )
    if bm25_sparse_enabled(settings):
        return {settings.qdrant_dense_vector_name: dense_params}
    return dense_params


def _sparse_vectors_config(settings: Settings):
    if not bm25_sparse_enabled(settings):
        return None
    return {
        settings.qdrant_sparse_vector_name: models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        )
    }


def _legacy_collection_vector_size(info) -> int | None:
    params = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    if isinstance(vectors, dict):
        vectors = next(iter(vectors.values()), None)
    return getattr(vectors, "size", None)


def _named_collection_vector_size(info, vector_name: str) -> int | None:
    params = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    if not isinstance(vectors, dict):
        return None
    return getattr(vectors.get(vector_name), "size", None)


def _named_sparse_vector_params(info, vector_name: str):
    params = getattr(getattr(info, "config", None), "params", None)
    sparse_vectors = getattr(params, "sparse_vectors", None)
    if not isinstance(sparse_vectors, dict):
        return None
    return sparse_vectors.get(vector_name)
