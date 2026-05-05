from qdrant_client import models

from atlas.core.config import Settings


class BM25SparseEncoder:
    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.bm25_model
        self.language = settings.bm25_language
        self.k = settings.bm25_k
        self.b = settings.bm25_b
        self.avg_len = settings.bm25_avg_len
        self._model = None

    def embed_texts(self, texts: list[str]) -> list[models.SparseVector]:
        embeddings = self._fastembed_model().embed(texts)
        return [_to_qdrant_sparse_vector(embedding) for embedding in embeddings]

    def embed_query(self, query: str) -> models.SparseVector:
        model = self._fastembed_model()
        if hasattr(model, "query_embed"):
            embedding = next(iter(model.query_embed(query)))
        else:
            embedding = next(iter(model.embed([query])))
        return _to_qdrant_sparse_vector(embedding)

    def _fastembed_model(self):
        if self._model is None:
            try:
                from fastembed import SparseTextEmbedding
            except ImportError as exc:
                raise RuntimeError(
                    "fastembed is required for BM25 sparse retrieval. "
                    "Install project dependencies or disable BM25/hybrid retrieval."
                ) from exc

            self._model = SparseTextEmbedding(
                model_name=self.model_name,
                language=self.language,
                k=self.k,
                b=self.b,
                avg_len=self.avg_len,
            )
        return self._model


def _to_qdrant_sparse_vector(embedding) -> models.SparseVector:
    indices = _to_list(getattr(embedding, "indices", []), item_type=int)
    values = _to_list(getattr(embedding, "values", []), item_type=float)
    return models.SparseVector(indices=indices, values=values)


def _to_list(values, *, item_type):
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [item_type(value) for value in values]
