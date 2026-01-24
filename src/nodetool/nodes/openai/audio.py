import base64
from enum import Enum
from io import BytesIO
from typing import ClassVar, TypedDict

from nodetool.metadata.types import AudioChunk, AudioRef, Provider
from nodetool.providers.openai_prediction import run_openai
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field


class TextToSpeech(BaseNode):
    """
    Converts text to speech using OpenAI TTS models.
    audio, tts, text-to-speech, voice, synthesis
    """

    class TtsModel(str, Enum):
        tts_1 = "tts-1"
        tts_1_hd = "tts-1-hd"
        gpt_4o_mini_tts = "gpt-4o-mini-tts"

    class Voice(str, Enum):
        ALLOY = "alloy"
        ASH = "ash"
        BALLAD = "ballad"
        CORAL = "coral"
        ECHO = "echo"
        FABLE = "fable"
        ONYX = "onyx"
        NOVA = "nova"
        SAGE = "sage"
        SHIMMER = "shimmer"
        VERSE = "verse"

    model: TtsModel = Field(title="Model", default=TtsModel.tts_1)
    voice: Voice = Field(title="Voice", default=Voice.ALLOY)
    input: str = Field(title="Input", default="")
    speed: float = Field(title="Speed", default=1.0, ge=0.25, le=4.0)

    _auto_save_asset: ClassVar[bool] = True
    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> AudioRef:
        from pydub import AudioSegment

        res = await context.run_prediction(
            node_id=self._id,
            provider=Provider.OpenAI,
            model=self.model.value,
            run_prediction_function=run_openai,
            params={
                "input": self.input,
                "voice": self.voice,
                "speed": self.speed,
            },
        )

        segment = AudioSegment.from_mp3(BytesIO(res))
        audio = await context.audio_from_segment(segment)  # type: ignore
        return audio

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["input", "model", "voice"]


class Translate(BaseNode):
    """
    Translates speech in audio to English text.
    audio, translation, speech-to-text, localization
    """

    audio: AudioRef = Field(
        default=AudioRef(), description="The audio file to translate."
    )
    temperature: float = Field(
        default=0.0, description="The temperature to use for the translation."
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> str:
        from openai.types.audio.translation import Translation

        audio_bytes = await context.asset_to_io(self.audio)
        response = await context.run_prediction(
            node_id=self._id,
            provider=Provider.OpenAI,
            model="whisper-1",
            run_prediction_function=run_openai,
            params={
                "file": base64.b64encode(audio_bytes.read()).decode(),
                "temperature": self.temperature,
                "translate": True,
            },
        )
        res = Translation(**response)

        return res.text


class Transcribe(BaseNode):
    """
    Converts speech to text using OpenAI's speech-to-text API.
    audio, transcription, speech-to-text, stt, whisper
    """

    class TranscriptionModel(str, Enum):
        WHISPER_1 = "whisper-1"
        GPT_4O_TRANSCRIBE = "gpt-4o-transcribe"
        GPT_4O_MINI_TRANSCRIBE = "gpt-4o-mini-transcribe"

    class Language(str, Enum):
        NONE = "auto_detect"
        SPANISH = "spanish"
        ITALIAN = "italian"
        KOREAN = "korean"
        PORTUGUESE = "portuguese"
        ENGLISH = "english"
        JAPANESE = "japanese"
        GERMAN = "german"
        RUSSIAN = "russian"
        DUTCH = "dutch"
        POLISH = "polish"
        CATALAN = "catalan"
        FRENCH = "french"
        INDONESIAN = "indonesian"
        UKRAINIAN = "ukrainian"
        TURKISH = "turkish"
        MALAY = "malay"
        SWEDISH = "swedish"
        MANDARIN = "mandarin"
        FINNISH = "finnish"
        NORWEGIAN = "norwegian"
        ROMANIAN = "romanian"
        THAI = "thai"
        VIETNAMESE = "vietnamese"
        SLOVAK = "slovak"
        ARABIC = "arabic"
        CZECH = "czech"
        CROATIAN = "croatian"
        GREEK = "greek"
        SERBIAN = "serbian"
        DANISH = "danish"
        BULGARIAN = "bulgarian"
        HUNGARIAN = "hungarian"
        FILIPINO = "filipino"
        BOSNIAN = "bosnian"
        GALICIAN = "galician"
        MACEDONIAN = "macedonian"
        HINDI = "hindi"
        ESTONIAN = "estonian"
        SLOVENIAN = "slovenian"
        TAMIL = "tamil"
        LATVIAN = "latvian"
        AZERBAIJANI = "azerbaijani"
        URDU = "urdu"
        LITHUANIAN = "lithuanian"
        HEBREW = "hebrew"
        WELSH = "welsh"
        PERSIAN = "persian"
        ICELANDIC = "icelandic"
        KAZAKH = "kazakh"
        AFRIKAANS = "afrikaans"
        KANNADA = "kannada"
        MARATHI = "marathi"
        SWAHILI = "swahili"
        TELUGU = "telugu"
        MAORI = "maori"
        NEPALI = "nepali"
        ARMENIAN = "armenian"
        BELARUSIAN = "belarusian"
        GUJARATI = "gujarati"
        PUNJABI = "punjabi"
        BENGALI = "bengali"

    model: TranscriptionModel = Field(
        default=TranscriptionModel.WHISPER_1,
        description="The model to use for transcription.",
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="The audio file to transcribe (max 25 MB)."
    )
    language: Language = Field(
        default=Language.NONE,
        description="The language of the input audio",
    )
    timestamps: bool = Field(
        default=False,
        description="Whether to return timestamps for the generated text.",
    )
    prompt: str = Field(
        default="",
        description="Optional text to guide the model's style or continue a previous audio segment.",
    )
    temperature: float = Field(
        default=0,
        ge=0,
        le=1,
        description="The sampling temperature between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.",
    )

    _expose_as_tool: ClassVar[bool] = True

    class OutputType(TypedDict):
        text: str
        words: list[AudioChunk]
        segments: list[AudioChunk]

    async def process(self, context: ProcessingContext) -> OutputType:
        from openai.types.audio.transcription_verbose import TranscriptionVerbose

        audio_bytes = await context.audio_to_base64(self.audio)

        params = {
            "file": audio_bytes,
            "temperature": self.temperature,
        }

        is_new_model = self.model in [
            Transcribe.TranscriptionModel.GPT_4O_TRANSCRIBE,
            Transcribe.TranscriptionModel.GPT_4O_MINI_TRANSCRIBE,
        ]

        if self.timestamps:
            if not is_new_model:
                params["response_format"] = "verbose_json"
                # Request word and segment granularities for detailed timestamps with whisper-1
                params["timestamp_granularities"] = ["segment", "word"]
            else:
                raise ValueError("New models do not support verbose_json")
        else:
            params["response_format"] = "json"

        if self.language.value != "auto_detect":  # Language.NONE.value is "auto_detect"
            params["language"] = self.language.value
        if self.prompt:
            params["prompt"] = self.prompt

        response = await context.run_prediction(
            node_id=self._id,
            provider=Provider.OpenAI,
            model=self.model.value,
            run_prediction_function=run_openai,
            params=params,
        )

        final_text = ""
        final_words = []
        final_segments = []

        current_response_format = params["response_format"]

        if current_response_format == "verbose_json":
            # Expected for whisper-1 with timestamps=True
            # The response from run_openai is expected to be a dict
            if isinstance(response, dict):
                transcription = TranscriptionVerbose(**response)
                final_text = transcription.text
                if transcription.segments:
                    for segment_data in transcription.segments:
                        final_segments.append(
                            AudioChunk(
                                timestamp=(segment_data.start, segment_data.end),
                                text=segment_data.text,
                            )
                        )
                if (
                    transcription.words
                ):  # Relies on timestamp_granularities including "word"
                    for word_data in transcription.words:
                        final_words.append(  # Corrected: was appending to final_segments
                            AudioChunk(
                                timestamp=(word_data.start, word_data.end),
                                text=word_data.word,
                            )
                        )
            else:
                # Handle unexpected response type for verbose_json
                if isinstance(response, str):  # Failsafe if API returned raw text
                    final_text = response
                # else: final_text remains "", or log error
        else:
            if isinstance(response, dict):
                final_text = response.get("text", "")  # Use .get for safety
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")

        return {
            "text": final_text,
            "words": final_words,
            "segments": final_segments,
        }

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["audio", "language", "timestamps"]
