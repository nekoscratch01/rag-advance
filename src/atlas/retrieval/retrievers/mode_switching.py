from __future__ import annotations

from sqlalchemy.orm import Session

from atlas.retrieval.models.evidence import Evidence


class ModeSwitchingRetriever:
    def __init__(
        self,
        *,
        dense_retriever,
        bm25_retriever,
        hybrid_rrf_retriever,
        hybrid_rerank_retriever,
        default_mode: str,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.hybrid_rrf_retriever = hybrid_rrf_retriever
        self.hybrid_rerank_retriever = hybrid_rerank_retriever
        self.default_mode = default_mode

    def retrieve(
        self,
        db: Session,
        *,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[Evidence]:
        return self.retrieve_with_options(
            db,
            query=query,
            top_k=top_k,
            filters=filters,
            options={},
        )

    def retrieve_with_options(
        self,
        db: Session,
        *,
        query: str,
        top_k: int,
        filters: dict | None = None,
        options: dict | None = None,
    ) -> list[Evidence]:
        retriever = self._select_retriever(options or {})
        return retriever.retrieve(db, query=query, top_k=top_k, filters=filters)

    def _select_retriever(self, options: dict):
        mode = str(
            options.get("retrieval_mode")
            or options.get("mode")
            or options.get("benchmark_mode")
            or self.default_mode
        ).strip().lower()
        reranker_enabled = _optional_bool(options, "reranker_enabled")
        if "rerank" in options:
            reranker_enabled = _truthy(options.get("rerank"))

        if mode in {"dense", "dense_only"}:
            return self.dense_retriever
        if mode in {"bm25", "lexical", "bm25_only"}:
            return self.bm25_retriever
        if mode in {"hybrid_rerank", "hybrid-rerank", "hybrid_reranker"}:
            return self.hybrid_rerank_retriever
        if mode in {"hybrid_rrf", "hybrid-rrf", "hybrid_no_rerank"}:
            return self.hybrid_rrf_retriever
        if mode == "hybrid":
            if reranker_enabled is False:
                return self.hybrid_rrf_retriever
            return self.hybrid_rerank_retriever if reranker_enabled is True else self.hybrid_rerank_retriever
        return self.dense_retriever


def _optional_bool(options: dict, key: str) -> bool | None:
    if key not in options:
        return None
    return _truthy(options.get(key))


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled", "enable"}
