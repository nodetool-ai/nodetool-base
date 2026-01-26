from enum import Enum
from typing import ClassVar

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import AudioRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field

logger = get_logger(__name__)


class WhisperModel(str, Enum):
    """Available Whisper models on Groq."""

    WHISPER_LARGE_V3 = "whisper-large-v3"
    WHISPER_LARGE_V3_TURBO = "whisper-large-v3-turbo"
    DISTIL_WHISPER = "distil-whisper-large-v3-en"


class AudioTranscription(BaseNode):
    """
    Transcribe audio to text using Groq's ultra-fast Whisper models.
    groq, whisper, audio, transcription, speech-to-text, stt, voice

    Uses Groq's LPU inference for extremely fast audio transcription
    with OpenAI's Whisper models. Supports multiple languages.
    Requires a Groq API key.

    Use cases:
    - Real-time speech transcription
    - Podcast and video transcription
    - Meeting notes and recordings
    - Voice command processing
    - Multilingual audio content conversion
    """

    _expose_as_tool: ClassVar[bool] = True

    audio: AudioRef = Field(
        default=AudioRef(),
        description="The audio file to transcribe",
    )

    model: WhisperModel = Field(
        default=WhisperModel.WHISPER_LARGE_V3_TURBO,
        description="The Whisper model to use for transcription",
    )

    language: str = Field(
        default="",
        description="Optional language code (e.g., 'en', 'es', 'fr'). Auto-detected if empty.",
    )

    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for transcription. Lower is more deterministic.",
    )

    async def process(self, context: ProcessingContext) -> str:
        """
        Transcribe audio to text using Groq's Whisper models.

        Args:
            context: The processing context.

        Returns:
            str: The transcribed text.
        """
        if not self.audio.is_set():
            raise ValueError("Audio file is required")

        api_key = await context.get_secret("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not configured")

        from groq import AsyncGroq

        client = AsyncGroq(api_key=api_key)

        # Get the audio bytes
        audio_bytes = await context.asset_to_bytes(self.audio)

        # Prepare optional parameters
        kwargs = {
            "model": self.model.value,
            "temperature": self.temperature,
        }
        if self.language:
            kwargs["language"] = self.language

        # Create transcription
        response = await client.audio.transcriptions.create(
            file=("audio.mp3", audio_bytes),
            **kwargs,
        )

        if not response or not response.text:
            raise ValueError("No transcription received from Groq API")

        return response.text

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["audio", "model"]


class AudioTranslation(BaseNode):
    """
    Translate audio to English text using Groq's ultra-fast Whisper models.
    groq, whisper, audio, translation, speech-to-text, voice, language

    Uses Groq's LPU inference for extremely fast audio translation
    to English text from any supported language.
    Requires a Groq API key.

    Use cases:
    - Translate foreign language audio to English
    - Multilingual meeting transcription
    - International content localization
    - Cross-language audio processing
    """

    _expose_as_tool: ClassVar[bool] = True

    audio: AudioRef = Field(
        default=AudioRef(),
        description="The audio file to translate to English",
    )

    model: WhisperModel = Field(
        default=WhisperModel.WHISPER_LARGE_V3_TURBO,
        description="The Whisper model to use for translation",
    )

    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for translation. Lower is more deterministic.",
    )

    async def process(self, context: ProcessingContext) -> str:
        """
        Translate audio to English text using Groq's Whisper models.

        Args:
            context: The processing context.

        Returns:
            str: The translated English text.
        """
        if not self.audio.is_set():
            raise ValueError("Audio file is required")

        api_key = await context.get_secret("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not configured")

        from groq import AsyncGroq

        client = AsyncGroq(api_key=api_key)

        # Get the audio bytes
        audio_bytes = await context.asset_to_bytes(self.audio)

        # Create translation
        response = await client.audio.translations.create(
            file=("audio.mp3", audio_bytes),
            model=self.model.value,
            temperature=self.temperature,
        )

        if not response or not response.text:
            raise ValueError("No translation received from Groq API")

        return response.text

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["audio", "model"]
