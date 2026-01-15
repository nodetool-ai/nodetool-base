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


class ElevenLabsTextToSpeechMultilingualV2(KieBaseNode):
    """Generate high-quality speech using ElevenLabs Multilingual V2 model via Kie.ai.

    kie, elevenlabs, tts, text-to-speech, audio, ai, multilingual, voice

    Produces natural-sounding speech in multiple languages with support for
    voice cloning, stability controls, and style customization.

    Use cases:
    - Create voiceovers for videos and presentations
    - Generate audiobook narration
    - Produce multilingual content
    - Accessibility applications
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 4.0
    _max_poll_attempts: int = 60

    text: str = Field(
        default="",
        description="The text to convert to speech.",
        max_length=5000,
    )

    class Voice(str, Enum):
        RACHEL = "Rachel"
        ARIA = "Aria"
        ROGER = "Roger"
        SARAH = "Sarah"
        LAURA = "Laura"
        CHARLIE = "Charlie"
        GEORGE = "George"
        CALLUM = "Callum"
        RIVER = "River"
        LIAM = "Liam"
        CHARLOTTE = "Charlotte"
        ALICE = "Alice"
        MATILDA = "Matilda"
        WILL = "Will"
        JESSICA = "Jessica"
        ERIC = "Eric"
        CHRIS = "Chris"
        BRIAN = "Brian"
        DANIEL = "Daniel"
        LILY = "Lily"
        BILL = "Bill"

    voice: Voice = Field(
        default=Voice.RACHEL,
        description="The voice to use for speech generation.",
    )

    stability: float = Field(
        default=0.5,
        description="Voice stability (0-1). Higher values produce more consistent output.",
        ge=0.0,
        le=1.0,
    )

    similarity_boost: float = Field(
        default=0.75,
        description="Similarity boost (0-1). Higher values make speech more similar to the original voice.",
        ge=0.0,
        le=1.0,
    )

    style: float = Field(
        default=0.0,
        description="Style exaggeration (0-1). Higher values apply more style.",
        ge=0.0,
        le=1.0,
    )

    speed: float = Field(
        default=1.0,
        description="Speech speed (0.7-1.2). Values below 1.0 slow down, above 1.0 speed up.",
        ge=0.7,
        le=1.2,
    )

    timestamps: bool = Field(
        default=False,
        description="Whether to return timestamps for each word in the generated speech.",
    )

    language_code: str = Field(
        default="",
        description="Language code (ISO 639-1) to enforce a specific language.",
        max_length=10,
    )

    def _get_model(self) -> str:
        return "elevenlabs/text-to-speech-multilingual-v2"

    @classmethod
    def get_title(cls) -> str:
        return "ElevenLabs Multilingual TTS"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.text:
            raise ValueError("Text cannot be empty")
        payload: dict[str, Any] = {
            "text": self.text,
            "voice": self.voice.value,
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "speed": self.speed,
            "timestamps": self.timestamps,
        }
        if self.language_code:
            payload["language_code"] = self.language_code
        return payload

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio_bytes, _ = await self._execute_task(context)
        return await context.audio_from_bytes(audio_bytes)


class ElevenLabsTextToSpeechTurbo25(KieBaseNode):
    """Generate speech using ElevenLabs Turbo 2.5 model via Kie.ai.

    kie, elevenlabs, tts, text-to-speech, audio, ai, turbo, fast

    Fast text-to-speech model optimized for low latency while maintaining
    high quality natural-sounding output.

    Use cases:
    - Real-time voice applications
    - Interactive voice responses
    - Gaming audio
    - Quick voiceover generation
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 3.0
    _max_poll_attempts: int = 60

    text: str = Field(
        default="",
        description="The text to convert to speech.",
        max_length=5000,
    )

    class Voice(str, Enum):
        RACHEL = "Rachel"
        ARIA = "Aria"
        ROGER = "Roger"
        SARAH = "Sarah"
        LAURA = "Laura"
        CHARLIE = "Charlie"
        GEORGE = "George"
        CALLUM = "Callum"
        RIVER = "River"
        LIAM = "Liam"
        CHARLOTTE = "Charlotte"
        ALICE = "Alice"
        MATILDA = "Matilda"
        WILL = "Will"
        JESSICA = "Jessica"
        ERIC = "Eric"
        CHRIS = "Chris"
        BRIAN = "Brian"
        DANIEL = "Daniel"
        LILY = "Lily"
        BILL = "Bill"

    voice: Voice = Field(
        default=Voice.RACHEL,
        description="The voice to use for speech generation.",
    )

    stability: float = Field(
        default=0.5,
        description="Voice stability (0-1). Higher values produce more consistent output.",
        ge=0.0,
        le=1.0,
    )

    similarity_boost: float = Field(
        default=0.75,
        description="Similarity boost (0-1). Higher values make speech more similar to the original voice.",
        ge=0.0,
        le=1.0,
    )

    style: float = Field(
        default=0.0,
        description="Style exaggeration (0-1). Higher values apply more style.",
        ge=0.0,
        le=1.0,
    )

    speed: float = Field(
        default=1.0,
        description="Speech speed (0.7-1.2). Values below 1.0 slow down, above 1.0 speed up.",
        ge=0.7,
        le=1.2,
    )

    timestamps: bool = Field(
        default=False,
        description="Whether to return timestamps for each word in the generated speech.",
    )

    language_code: str = Field(
        default="",
        description="Language code (ISO 639-1) to enforce a specific language.",
        max_length=10,
    )

    def _get_model(self) -> str:
        return "elevenlabs/text-to-speech-turbo-2-5"

    @classmethod
    def get_title(cls) -> str:
        return "ElevenLabs Turbo TTS"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.text:
            raise ValueError("Text cannot be empty")
        payload: dict[str, Any] = {
            "text": self.text,
            "voice": self.voice.value,
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "speed": self.speed,
            "timestamps": self.timestamps,
        }
        if self.language_code:
            payload["language_code"] = self.language_code
        return payload

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio_bytes, _ = await self._execute_task(context)
        return await context.audio_from_bytes(audio_bytes)


class ElevenLabsSoundEffectV2(KieBaseNode):
    """Generate sound effects using ElevenLabs Sound Effect V2 model via Kie.ai.

    kie, elevenlabs, sfx, sound-effect, audio, ai, generation

    Creates AI-generated sound effects for games, videos, applications,
    and creative projects from text descriptions.

    Use cases:
    - Game audio development
    - Video production sound design
    - App sound effects
    - Creative audio projects
    """

    _expose_as_tool: ClassVar[bool] = True
    _poll_interval: float = 4.0
    _max_poll_attempts: int = 60

    text: str = Field(
        default="",
        description="The text describing the sound effect to generate.",
        max_length=5000,
    )

    loop: bool = Field(
        default=False,
        description="Whether to create a sound effect that loops smoothly.",
    )

    duration_seconds: float = Field(
        default=5.0,
        description="Duration in seconds (0.5-22). If None, optimal duration is determined from prompt.",
        ge=0.5,
        le=22.0,
    )

    prompt_influence: float = Field(
        default=0.3,
        description="How closely to follow the prompt (0-1). Higher values mean less variation.",
        ge=0.0,
        le=1.0,
    )

    class OutputFormat(str, Enum):
        MP3_22050_32 = "mp3_22050_32"
        MP3_44100_32 = "mp3_44100_32"
        MP3_44100_64 = "mp3_44100_64"
        MP3_44100_96 = "mp3_44100_96"
        MP3_44100_128 = "mp3_44100_128"
        MP3_44100_192 = "mp3_44100_192"
        PCM_22050 = "pcm_22050"
        PCM_44100 = "pcm_44100"
        PCM_48000 = "pcm_48000"
        OPUS_48000_128 = "opus_48000_128"

    output_format: OutputFormat = Field(
        default=OutputFormat.MP3_44100_128,
        description="Output format of the generated audio.",
    )

    def _get_model(self) -> str:
        return "elevenlabs/sound-effect-v2"

    @classmethod
    def get_title(cls) -> str:
        return "ElevenLabs Sound Effects"

    async def _get_input_params(
        self, context: ProcessingContext | None = None
    ) -> dict[str, Any]:
        if not self.text:
            raise ValueError("Text cannot be empty")
        return {
            "text": self.text,
            "loop": self.loop,
            "duration_seconds": self.duration_seconds,
            "prompt_influence": self.prompt_influence,
            "output_format": self.output_format.value,
        }

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio_bytes, _ = await self._execute_task(context)
        return await context.audio_from_bytes(audio_bytes)
