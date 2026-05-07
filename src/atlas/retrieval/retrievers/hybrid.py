"""Legacy import path only. Not a runtime provider contract."""

from atlas.retrieval.providers.text_hybrid.adapters.hybrid import (
    CandidateRetriever,
    HybridRetriever,
)

__all__ = ["CandidateRetriever", "HybridRetriever"]
