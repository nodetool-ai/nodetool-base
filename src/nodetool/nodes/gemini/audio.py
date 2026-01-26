from enum import Enum
from typing import ClassVar

from nodetool.metadata.types import AudioRef, Provider
from nodetool.providers.gemini_provider import GeminiProvider
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field, field_validator


class TranscriptionModel(str, Enum):
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"


class TTSModel(str, Enum):
    GEMINI_2_5_FLASH_PREVIEW_TTS = "gemini-2.5-flash-preview-tts"
    GEMINI_2_5_PRO_PREVIEW_TTS = "gemini-2.5-pro-preview-tts"


class VoiceName(str, Enum):
    ACHERNAR = "achernar"
    ACHIRD = "achird"
    ALGENIB = "algenib"
    ALGIEBA = "algieba"
    ALNILAM = "alnilam"
    AOEDE = "aoede"
    AUTONOE = "autonoe"
    CALLIRRHOE = "callirrhoe"
    CHARON = "charon"
    DESPINA = "despina"
    ENCELADUS = "enceladus"
    ERINOME = "erinome"
    FENRIR = "fenrir"
    GACRUX = "gacrux"
    IAPETUS = "iapetus"
    KORE = "kore"
    LAOMEDEIA = "laomedeia"
    LEDA = "leda"
    ORUS = "orus"
    PUCK = "puck"
    PULCHERRIMA = "pulcherrima"
    RASALGETHI = "rasalgethi"
    SADACHBIA = "sadachbia"
    SADALTAGER = "sadaltager"
    SCHEDAR = "schedar"
    SULAFAT = "sulafat"
    UMBRIEL = "umbriel"
    VINDEMIATRIX = "vindemiatrix"
    ZEPHYR = "zephyr"
    ZUBENELGENUBI = "zubenelgenubi"


class TextToSpeech(BaseNode):
    """
    Generate speech audio from text using Google's Gemini text-to-speech models.
    google, text-to-speech, tts, audio, speech, voice, ai

    This node converts text input into natural-sounding speech audio using Google's
    advanced text-to-speech models with support for multiple voices and speech styles.

    Supported voices:
    - achernar, achird, algenib, algieba, alnilam
    - aoede, autonoe, callirrhoe, charon, despina
    - enceladus, erinome, fenrir, gacrux, iapetus
    - kore, laomedeia, leda, orus, puck
    - pulcherrima, rasalgethi, sadachbia, sadaltager, schedar
    - sulafat, umbriel, vindemiatrix, zephyr, zubenelgenubi

    Use cases:
    - Create voiceovers for videos and presentations
    - Generate audio content for podcasts and audiobooks
    - Add voice narration to applications
    - Create accessibility features with speech output
    - Generate multilingual audio content
    """

    _auto_save_asset: ClassVar[bool] = True
    _expose_as_tool: ClassVar[bool] = True

    text: str = Field(default="", description="The text to convert to speech.")

    model: TTSModel = Field(
        default=TTSModel.GEMINI_2_5_FLASH_PREVIEW_TTS,
        description="The text-to-speech model to use",
    )

    voice_name: VoiceName = Field(
        default=VoiceName.KORE, description="The voice to use for speech generation"
    )

    @field_validator("voice_name", mode="before")
    @classmethod
    def validate_voice_name(cls, v):
        if isinstance(v, VoiceName):
            return v
        if isinstance(v, str):
            try:
                # Try to map to a valid enum value
                return VoiceName(v.lower())
            except ValueError:
                # Default to Kore if invalid
                return VoiceName.KORE
        return v

    style_prompt: str = Field(
        default="",
        description="Optional style prompt to control speech characteristics (e.g., 'Say cheerfully', 'Speak with excitement')",
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        from google.genai import types
        from pydub import AudioSegment

        if not self.text:
            raise ValueError("The input text cannot be empty.")

        # Prepare the content with optional style prompt
        content = self.text
        if self.style_prompt:
            content = f"{self.style_prompt}: {self.text}"

        provider = await context.get_provider(Provider.Gemini)
        assert isinstance(provider, GeminiProvider)
        client = provider.get_client()  # pyright: ignore[reportAttributeAccessIssue]

        response = await client.models.generate_content(
            model=self.model.value,
            contents=content,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.voice_name.value,
                        )
                    )
                ),
            ),
        )

        if (
            not response.candidates
            or not response.candidates[0].content
            or not response.candidates[0].content.parts
        ):
            raise ValueError("No audio generated from the text-to-speech request")

        # Extract audio bytes from the response
        audio_part = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                audio_part = part
                break

        assert audio_part, "No audio part found in the response"
        assert audio_part.inline_data, "No audio data found in the response"
        assert audio_part.inline_data.data, "No audio data found in the response"

        wav_data = audio_part.inline_data.data

        audio_segment = AudioSegment(
            data=wav_data,
            sample_width=2,  # 16-bit PCM
            frame_rate=24000,  # 24 kHz
            channels=1,
        )

        return await context.audio_from_segment(audio_segment)


class Transcribe(BaseNode):
    """
    Transcribe audio to text using Google's Gemini models.
    google, transcription, speech-to-text, audio, whisper, ai

    This node converts audio input into text using Google's multimodal Gemini models.
    Supports various audio formats and provides accurate speech-to-text transcription.

    Use cases:
    - Convert recorded audio to text
    - Transcribe podcasts and interviews
    - Generate subtitles from audio tracks
    - Create meeting notes from audio recordings
    - Analyze speech content in audio files
    """

    _expose_as_tool: ClassVar[bool] = True

    audio: AudioRef = Field(
        default=AudioRef(), description="The audio file to transcribe."
    )

    model: TranscriptionModel = Field(
        default=TranscriptionModel.GEMINI_2_5_FLASH,
        description="The Gemini model to use for transcription",
    )

    prompt: str = Field(
        default="Transcribe the following audio accurately. Return only the transcription text without any additional commentary.",
        description="Instructions for the transcription. You can customize this to request specific formatting or focus.",
    )

    async def process(self, context: ProcessingContext) -> str:
        """
        Transcribe audio using the Gemini model.

        Returns:
            str: The transcribed text from the audio
        """
        from google.genai import types

        if not self.audio.is_set():
            raise ValueError("Audio file is required for transcription")

        provider = await context.get_provider(Provider.Gemini)
        assert isinstance(provider, GeminiProvider)
        client = provider.get_client()  # pyright: ignore[reportAttributeAccessIssue]

        # Get audio bytes and create the inline data
        audio_bytes = await context.asset_to_bytes(self.audio)

        # Create the audio part
        audio_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/mp3",
        )

        # Generate transcription using Gemini's multimodal capabilities
        response = await client.models.generate_content(
            model=self.model.value,
            contents=[self.prompt, audio_part],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT"],
            ),
        )

        if (
            not response.candidates
            or not response.candidates[0].content
            or not response.candidates[0].content.parts
        ):
            raise ValueError("No transcription generated from the audio")

        # Extract text from the response
        transcription_parts = []
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                transcription_parts.append(part.text)

        return "".join(transcription_parts)
