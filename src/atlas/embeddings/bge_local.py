from functools import cached_property

from atlas.core.config import Settings


class LocalBGEEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.embedding_model
        self.dimension = settings.embedding_dim

    @cached_property
    def model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, query: str) -> list[float]:
        query_text = f"为这个句子生成表示以用于检索相关文章：{query}"
        return self.embed_texts([query_text])[0]
