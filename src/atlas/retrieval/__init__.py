"""Retrieval runtime."""

from atlas.retrieval.candidate import Candidate
from atlas.retrieval.evidence import Evidence
from atlas.retrieval.fusion import (
    DEFAULT_RRF_K,
    WeightedRRFInput,
    fusion_trace_payload,
    rrf_fuse,
    weighted_rrf_fuse,
)
from atlas.retrieval.hybrid_retriever import CandidateRetriever, HybridRetriever
from atlas.retrieval.mode_switching import ModeSwitchingRetriever
from atlas.retrieval.providers.text_hybrid import TextHybridProvider
from atlas.retrieval.reranker import CrossEncoderReranker, Reranker

__all__ = [
    "Candidate",
    "CandidateRetriever",
    "CrossEncoderReranker",
    "DEFAULT_RRF_K",
    "Evidence",
    "HybridRetriever",
    "ModeSwitchingRetriever",
    "Reranker",
    "TextHybridProvider",
    "WeightedRRFInput",
    "fusion_trace_payload",
    "rrf_fuse",
    "weighted_rrf_fuse",
]
