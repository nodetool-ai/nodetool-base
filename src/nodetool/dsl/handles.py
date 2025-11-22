from __future__ import annotations

from typing import Any, Dict, Generic, TypeVar

from pydantic import Field

T = TypeVar("T")


class OutputHandle(Generic[T]):
    """Reference to the output of another graph node."""

    def __init__(self, node: Any, output_name: str = "output"):
        self.node = node
        self.output_name = output_name

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"OutputHandle(node={self.node!r}, output={self.output_name})"


class OutputsProxy(Dict[str, OutputHandle[Any]]):
    """Maps output names to handles for a specific node."""

    def __init__(self, node: Any):
        super().__init__()
        self._node = node

    def __getitem__(self, item: str) -> OutputHandle[Any]:
        return OutputHandle(self._node, item)


def connect_field(*, default: Any, description: str | None = None):
    """Small helper used by generated DSL code to declare a field."""

    return Field(default=default, description=description)

