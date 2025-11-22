from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TextToImageParams:
    prompt: str = ""
    negative_prompt: Optional[str] = None
    width: int = 0
    height: int = 0
    extra: dict[str, Any] = None  # type: ignore[assignment]


@dataclass
class ImageToImageParams:
    prompt: str = ""
    negative_prompt: Optional[str] = None
    image: Any = None
    strength: float = 1.0
    extra: dict[str, Any] = None  # type: ignore[assignment]

