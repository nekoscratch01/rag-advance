from __future__ import annotations

from abc import ABC

from atlas.backends.base import Backend, BackendBuildContext, BackendRegistry
from atlas.retrieval.providers.graph import GraphStore, PostgresGraphStore


class GraphStoreBackend(Backend[GraphStore], ABC):
    pass


class PostgresGraphStoreBackend(GraphStoreBackend):
    def build(self, context: BackendBuildContext) -> GraphStore:
        return PostgresGraphStore()


graph_store_backends: BackendRegistry[GraphStore] = BackendRegistry(
    namespace="graph_store",
    backend_type=GraphStoreBackend,
)


def build_graph_store(name: str, context: BackendBuildContext) -> GraphStore:
    return graph_store_backends.build(name, context)


def _register_builtin_backends() -> None:
    if "postgres_graph" not in graph_store_backends.names:
        graph_store_backends.register("postgres_graph", PostgresGraphStoreBackend())


_register_builtin_backends()
