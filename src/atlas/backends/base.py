from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from atlas.core.config import Settings
from atlas.core.errors import AtlasError, ErrorCode


T = TypeVar("T")


@dataclass(frozen=True)
class BackendBuildContext:
    settings: Settings


class Backend(ABC, Generic[T]):
    @abstractmethod
    def build(self, context: BackendBuildContext) -> T:
        ...


class BackendRegistry(Generic[T]):
    def __init__(self, *, namespace: str, backend_type: type[Backend[T]]) -> None:
        self.namespace = namespace
        self.backend_type = backend_type
        self._items: dict[str, Backend[T]] = {}

    def register(self, name: str, backend: Backend[T]) -> None:
        key = _backend_key(name)
        if not key:
            raise _configuration_error(
                namespace=self.namespace,
                backend_name=key,
                available=self.names,
                reason="backend_name_required",
            )
        if not isinstance(backend, self.backend_type):
            raise TypeError(f"{self.namespace}_backend_must_extend_{self.backend_type.__name__}")
        if key in self._items:
            raise _configuration_error(
                namespace=self.namespace,
                backend_name=key,
                available=self.names,
                reason="backend_already_registered",
            )
        self._items[key] = backend

    def get(self, name: str) -> Backend[T]:
        key = _backend_key(name)
        try:
            return self._items[key]
        except KeyError as exc:
            raise _configuration_error(
                namespace=self.namespace,
                backend_name=key,
                available=self.names,
                reason="backend_not_registered",
            ) from exc

    def build(self, name: str, context: BackendBuildContext) -> T:
        return self.get(name).build(context)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._items)


def _configuration_error(
    *,
    namespace: str,
    backend_name: str,
    available: tuple[str, ...],
    reason: str,
) -> AtlasError:
    available_text = ", ".join(available) if available else "none"
    if reason == "backend_name_required":
        message = f"{namespace} backend name is required. Available {namespace} backends: "
    elif reason == "backend_already_registered":
        message = f"{namespace} backend '{backend_name}' is already registered. "
        message += f"Available {namespace} backends: "
    else:
        message = f"Unknown {namespace} backend '{backend_name}'. Available {namespace} backends: "
    return AtlasError(
        ErrorCode.CONFIGURATION_ERROR,
        f"{message}{available_text}.",
        status_code=500,
        details={
            "backend_type": namespace,
            "backend": backend_name,
            "available": list(available),
            "reason": reason,
        },
    )


def _backend_key(name: str) -> str:
    return str(name).strip().lower()
