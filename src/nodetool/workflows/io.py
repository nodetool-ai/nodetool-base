from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, Iterable


class NodeInputs:
    def __init__(self, streams: Dict[str, Iterable[Any]] | None = None):
        self._streams = streams or {}

    async def stream(self, name: str) -> AsyncGenerator[Any, None]:
        values = self._streams.get(name, [])
        for value in values:
            yield value
            await asyncio.sleep(0)


class NodeOutputs:
    def __init__(self):
        self.emitted: Dict[str, list[Any]] = {}

    async def emit(self, name: str, value: Any):
        self.emitted.setdefault(name, []).append(value)
