from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass
class SaveUpdate:
    node_id: str
    name: str
    value: Any
    output_type: str


@dataclass
class Chunk:
    text: str = ""
    start: Optional[float] = None
    end: Optional[float] = None


NodeInputStream = Iterable[Any]

