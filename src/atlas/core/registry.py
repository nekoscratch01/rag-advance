from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar


T = TypeVar("T")


class ComponentRegistry(Generic[T]):
    def __init__(self, *, namespace: str) -> None:
        self.namespace = namespace
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> None:
        key = _key(name)
        if not key:
            raise ValueError(f"{self.namespace}_component_name_required")
        if key in self._items:
            raise ValueError(f"{self.namespace}_component_already_registered:{key}")
        self._items[key] = item

    def get(self, name: str) -> T:
        key = _key(name)
        try:
            return self._items[key]
        except KeyError as exc:
            raise ValueError(f"{self.namespace}_component_not_registered:{key}") from exc

    def build(self, name: str, *args, **kwargs):
        item = self.get(name)
        if not isinstance(item, Callable):
            raise TypeError(f"{self.namespace}_component_not_callable:{_key(name)}")
        return item(*args, **kwargs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._items)


def _key(name: str) -> str:
    return str(name).strip().lower()
