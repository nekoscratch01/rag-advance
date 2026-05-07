from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol
from urllib.parse import quote


GRAPH_CACHE_KEY_PREFIX = "graph"
GRAPH_CACHE_KEY_SEPARATOR = ":"


def _cache_key_part(value: object) -> str:
    text = str(value)
    if not text:
        raise ValueError("graph cache key parts must be non-empty")
    return quote(text, safe="")


def graph_cache_prefix(graph_version: str, namespace: str | None = None) -> str:
    parts: list[object] = [GRAPH_CACHE_KEY_PREFIX, graph_version]
    if namespace is not None:
        parts.append(namespace)
    prefix = GRAPH_CACHE_KEY_SEPARATOR.join(_cache_key_part(part) for part in parts)
    return f"{prefix}{GRAPH_CACHE_KEY_SEPARATOR}"


def graph_cache_key(graph_version: str, namespace: str, *parts: object) -> str:
    if not parts:
        raise ValueError("graph cache keys require at least one namespace-local part")
    encoded_parts = GRAPH_CACHE_KEY_SEPARATOR.join(_cache_key_part(part) for part in parts)
    return graph_cache_prefix(graph_version, namespace) + encoded_parts


def graph_cache_sequence_part(values: Iterable[object] | None) -> str:
    normalized: set[str] = set()
    for value in values or ():
        text = str(value)
        if text:
            normalized.add(text)
    return ",".join(sorted(normalized)) if normalized else "_"


def graph_cache_relation_types_part(relation_types: Iterable[str] | None) -> str:
    return graph_cache_sequence_part(relation_types)


class GraphCache(Protocol):
    def get(self, key: str) -> Any | None:
        ...

    def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        ...

    def delete_prefix(self, prefix: str) -> None:
        ...


class NoOpGraphCache:
    def get(self, key: str) -> Any | None:
        return None

    def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        return None

    def delete_prefix(self, prefix: str) -> None:
        return None


__all__ = [
    "GRAPH_CACHE_KEY_PREFIX",
    "GRAPH_CACHE_KEY_SEPARATOR",
    "GraphCache",
    "NoOpGraphCache",
    "graph_cache_key",
    "graph_cache_prefix",
    "graph_cache_relation_types_part",
    "graph_cache_sequence_part",
]
