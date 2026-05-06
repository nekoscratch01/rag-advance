"""Concrete first-stage retrievers."""

from atlas.retrieval.retrievers.bm25 import BM25Retriever
from atlas.retrieval.retrievers.dense import DenseRetriever
from atlas.retrieval.retrievers.hybrid import CandidateRetriever, HybridRetriever
from atlas.retrieval.retrievers.mode_switching import ModeSwitchingRetriever

__all__ = [
    "BM25Retriever",
    "CandidateRetriever",
    "DenseRetriever",
    "HybridRetriever",
    "ModeSwitchingRetriever",
]
