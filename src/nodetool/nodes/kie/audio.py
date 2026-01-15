"""Kie.ai audio/music generation nodes.

This module provides nodes for generating audio using Kie.ai's APIs:
- Suno Music API (AI music generation with vocals and instrumentals)
- ElevenLabs Text-to-Speech API (AI voice generation)
"""

from enum import Enum
from typing import Any, ClassVar

from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import AudioRef
from nodetool.workflows.processing_context import ProcessingContext

from .image import KieBaseNode

log = get_logger(__name__)


class Suno(KieBaseNode):
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
    _poll_interval: float = 4.0
    _max_poll_attempts: int = 120

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

    def _get_model(self) -> str:
        return "suno"

    @classmethod
    def get_title(cls) -> str:
        return "Suno Music Generator"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
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
        audio_bytes, _ = await self._execute_task(context)
        return await context.audio_from_bytes(audio_bytes)


class ElevenLabsTextToSpeech(KieBaseNode):
    """Generate speech using ElevenLabs AI via Kie.ai.

    kie, elevenlabs, tts, text-to-speech, voice, audio, ai, speech synthesis

    Creates natural-sounding speech from text using ElevenLabs' voice models.
    Supports multiple voices, stability controls, and multilingual output.

    Use cases:
    - Generate voiceovers for videos and podcasts
    - Create audiobooks and narrated content
    - Produce natural-sounding speech for applications
    - Generate speech in multiple languages and voices
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 2.0
    _max_poll_attempts: int = 60

    text: str = Field(
        default="",
        description="The text to convert to speech.",
    )

    voice: str = Field(
        default="Rachel",
        description="The voice ID to use for synthesis. Common voices: Rachel, Adam, Bella, Antoni.",
    )

    stability: float = Field(
        default=0.5,
        description="Stability of the voice output. Lower values are more expressive, higher values are more consistent.",
        ge=0.0,
        le=1.0,
    )

    similarity_boost: float = Field(
        default=0.75,
        description="How closely to clone the voice characteristics. Higher values match the voice more closely.",
        ge=0.0,
        le=1.0,
    )

    style: float = Field(
        default=0.0,
        description="Style parameter for voice expression. Range 0.0 to 1.0.",
        ge=0.0,
        le=1.0,
    )

    speed: float = Field(
        default=1.0,
        description="Speed of the speech. Range 0.5 to 1.5.",
        ge=0.5,
        le=1.5,
    )

    language_code: str = Field(
        default="",
        description="Language code for multilingual TTS (e.g., 'en', 'es', 'fr', 'de'). Leave empty for auto-detection.",
    )

    class Model(str, Enum):
        TURBO_2_5 = "text-to-speech-turbo-2-5"
        MULTILINGUAL_V2 = "text-to-speech-multilingual-v2"

    model: Model = Field(
        default=Model.TURBO_2_5,
        description="ElevenLabs model version to use.",
    )

    def _get_model(self) -> str:
        return f"elevenlabs/{self.model.value}"

    @classmethod
    def get_title(cls) -> str:
        return "ElevenLabs Text To Speech"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.text:
            raise ValueError("Text cannot be empty")
        payload: dict[str, Any] = {
            "text": self.text,
            "voice": self.voice,
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "speed": self.speed,
        }
        if self.language_code:
            payload["language_code"] = self.language_code
        return payload

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio_bytes, _ = await self._execute_task(context)
        return await context.audio_from_bytes(audio_bytes)
