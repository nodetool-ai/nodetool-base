"""Kie.ai audio/music generation nodes.

This module provides nodes for generating audio using Kie.ai's APIs:
- Suno Music API (AI music generation with vocals and instrumentals)
"""

from enum import Enum
from typing import Any, ClassVar

from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import AudioRef
from nodetool.workflows.processing_context import ProcessingContext

from .image import KieBaseNode

log = get_logger(__name__)


class SunoMusicGenerate(KieBaseNode):
    """Generate music using Suno AI via Kie.ai.

    kie, suno, music, audio, ai, generation, vocals, instrumental

    Creates full tracks with vocals and instrumentals up to around 8 minutes long.
    Supports the latest Suno V4.5+ model with improved vocal quality and composition.

    Use cases:
    - Generate background music for projects
    - Create AI-composed songs with vocals
    - Produce instrumentals for content
    - Generate music in various genres and styles
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="",
        description="Description of the music to generate (genre, mood, instruments, etc.).",
    )

    lyrics: str = Field(
        default="",
        description="Optional lyrics for the song. Leave empty for instrumental.",
    )

    class Style(str, Enum):
        POP = "pop"
        ROCK = "rock"
        JAZZ = "jazz"
        CLASSICAL = "classical"
        ELECTRONIC = "electronic"
        HIPHOP = "hip-hop"
        RNB = "r&b"
        COUNTRY = "country"
        FOLK = "folk"
        AMBIENT = "ambient"
        CUSTOM = "custom"

    style: Style = Field(
        default=Style.CUSTOM,
        description="Music style/genre. Use 'custom' for prompt-based generation.",
    )

    instrumental: bool = Field(
        default=False,
        description="Generate instrumental-only (no vocals).",
    )

    duration: int = Field(
        default=60,
        description="Approximate duration in seconds.",
        ge=30,
        le=480,
    )

    class Model(str, Enum):
        V4 = "v4"
        V4_5 = "v4.5"
        V4_5_PLUS = "v4.5+"

    model: Model = Field(
        default=Model.V4_5_PLUS,
        description="Suno model version to use.",
    )

    def _get_base_endpoint(self) -> str:
        return "/v1/market/suno"

    def _get_submit_payload(self) -> dict[str, Any]:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "instrumental": self.instrumental,
            "duration": self.duration,
            "model": self.model.value,
        }
        if self.lyrics:
            payload["lyrics"] = self.lyrics
        if self.style != self.Style.CUSTOM:
            payload["style"] = self.style.value
        return payload

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio_bytes = await self._execute_task(context)
        return await context.audio_from_bytes(audio_bytes)
