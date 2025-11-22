from __future__ import annotations

import uuid
from typing import Any, ClassVar, Dict, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

OutputType = TypeVar("OutputType")


class BaseNode(BaseModel, Generic[OutputType]):
    """Lightweight stand-in for the real BaseNode.

    It behaves like a Pydantic model and exposes a couple of helper methods
    that the generated DSL code expects.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    _dynamic_properties: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _expose_as_tool: ClassVar[bool] = False
    _layout: ClassVar[str | None] = None

    def __init__(self, **data: Any):
        dynamic = data.pop("_dynamic_properties", None)
        super().__init__(**data)
        if dynamic:
            self._dynamic_properties.update(dynamic)

    def required_inputs(self) -> list[str]:
        return []

    async def process(self, context: Any) -> OutputType:  # pragma: no cover - overridden
        raise NotImplementedError

    def result_for_all_outputs(self, result: Any) -> Dict[str, Any]:
        return {"output": result}

    def result_for_client(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return result

    @classmethod
    def get_node_type(cls):
        return cls.__name__


class OutputNode(BaseNode[OutputType], Generic[OutputType]):
    """Nodes that simply return their value."""

    OutputType: ClassVar[Any] = Any

    def result_for_all_outputs(self, result: OutputType) -> Dict[str, OutputType]:
        return {"output": result}
