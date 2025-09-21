from io import BytesIO
from typing import ClassVar
from base64 import b64decode
from pydantic import Field
from enum import Enum
from nodetool.metadata.types import AudioRef
from nodetool.workflows.base_node import ApiKeyMissingError, BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from google.genai.client import AsyncClient
from google.genai import types
from nodetool.config.environment import Environment
from google.genai import Client
from pydub import AudioSegment


def get_genai_client() -> AsyncClient:
    env = Environment.get_environment()
    api_key = env.get("GEMINI_API_KEY")
    if not api_key:
        raise ApiKeyMissingError(
            "GEMINI_API_KEY is not configured in the nodetool settings"
        )
    return Client(api_key=api_key).aio


class TTSModel(str, Enum):
    GEMINI_2_5_FLASH_PREVIEW_TTS = "gemini-2.5-flash-preview-tts"
    GEMINI_2_5_PRO_PREVIEW_TTS = "gemini-2.5-pro-preview-tts"


class VoiceName(str, Enum):
    ZEPHYR = "Zephyr"
    PUCK = "Puck"
    NOVA = "Nova"
    QUEST = "Quest"
    ECHO = "Echo"
    FABLE = "Fable"
    ORBIT = "Orbit"
    CHIME = "Chime"
    KORE = "Kore"
    ZENITH = "Zenith"
    COSMOS = "Cosmos"
    SAGE = "Sage"
    BREEZE = "Breeze"
    GLIMMER = "Glimmer"
    DRIFT = "Drift"
    PEARL = "Pearl"
    FLUX = "Flux"
    PRISM = "Prism"
    VEGA = "Vega"
    LYRA = "Lyra"
    RIPPLE = "Ripple"
    AZURE = "Azure"
    JUNO = "Juno"
    RIVER = "River"
    STERLING = "Sterling"
    ATLAS = "Atlas"
    BEACON = "Beacon"
    EMBER = "Ember"
    HARMONY = "Harmony"
    SPIRIT = "Spirit"


class TextToSpeech(BaseNode):
    """
    Generate speech audio from text using Google's Gemini text-to-speech models.
    google, text-to-speech, tts, audio, speech, voice, ai

    This node converts text input into natural-sounding speech audio using Google's
    advanced text-to-speech models with support for multiple voices and speech styles.

    Use cases:
    - Create voiceovers for videos and presentations
    - Generate audio content for podcasts and audiobooks
    - Add voice narration to applications
    - Create accessibility features with speech output
    - Generate multilingual audio content
    """

    _expose_as_tool: ClassVar[bool] = True

    text: str = Field(default="", description="The text to convert to speech.")

    model: TTSModel = Field(
        default=TTSModel.GEMINI_2_5_FLASH_PREVIEW_TTS,
        description="The text-to-speech model to use",
    )

    voice_name: VoiceName = Field(
        default=VoiceName.KORE, description="The voice to use for speech generation"
    )

    style_prompt: str = Field(
        default="",
        description="Optional style prompt to control speech characteristics (e.g., 'Say cheerfully', 'Speak with excitement')",
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        if not self.text:
            raise ValueError("The input text cannot be empty.")

        # Prepare the content with optional style prompt
        content = self.text
        if self.style_prompt:
            content = f"{self.style_prompt}: {self.text}"

        client = get_genai_client()

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
