import base64
from enum import Enum
from nodetool.workflows.base_node import BaseNode
from pydantic import Field
from io import BytesIO
from pydub import AudioSegment
from nodetool.metadata.types import AudioChunk, AudioRef, Provider
from nodetool.workflows.processing_context import ProcessingContext
from openai.types.audio.transcription import Transcription
from openai.types.audio.transcription_verbose import TranscriptionVerbose
from openai.types.audio.translation import Translation

from nodetool.chat.providers.openai_prediction import run_openai


class TextToSpeech(BaseNode):
    """
    Converts text to speech using OpenAI TTS models.
    audio, tts, text-to-speech, voice, synthesis

    Use cases:
    - Generate spoken content for videos or podcasts
    - Create voice-overs for presentations
    - Assist visually impaired users with text reading
    - Produce audio versions of written content
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

    async def process(self, context: ProcessingContext) -> AudioRef:
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

    Use cases:
    - Translate foreign language audio content to English
    - Create English transcripts of multilingual recordings
    - Assist non-English speakers in understanding audio content
    - Enable cross-language communication in audio formats
    """

    audio: AudioRef = Field(
        default=AudioRef(), description="The audio file to translate."
    )
    temperature: float = Field(
        default=0.0, description="The temperature to use for the translation."
    )

    async def process(self, context: ProcessingContext) -> str:
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

    Use cases:
    - Generate accurate transcriptions of audio content
    - Create searchable text from audio recordings
    - Support multiple languages for transcription
    - Enable automated subtitling and captioning
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

    @classmethod
    def return_type(cls):
        return {
            "text": str,
            "words": list[AudioChunk],
            "segments": list[AudioChunk],
        }

    async def process(self, context: ProcessingContext) -> dict:
        audio_bytes = await context.audio_to_base64(self.audio)

        response_format = "verbose_json" if self.timestamps else "text"

        params = {
            "file": audio_bytes,
            "response_format": response_format,
            "temperature": self.temperature,
        }

        if self.language:
            params["language"] = self.language
        if self.prompt:
            params["prompt"] = self.prompt

        response = await context.run_prediction(
            node_id=self._id,
            provider=Provider.OpenAI,
            model="whisper-1",
            run_prediction_function=run_openai,
            params=params,
        )

        # Handle different response formats
        if response_format == "verbose_json":
            transcription = TranscriptionVerbose(**response)
            segments = []
            words = []

            if transcription.segments:
                for segment in transcription.segments:
                    segments.append(
                        AudioChunk(
                            timestamp=(segment.start, segment.end),
                            text=segment.text,
                        )
                    )
            if transcription.words:
                for word in transcription.words:
                    words.append(
                        AudioChunk(timestamp=(word.start, word.end), text=word.word)
                    )

            return {
                "text": transcription.text,
                "words": words,
                "segments": segments,
            }
        else:
            return {
                "text": response if isinstance(response, str) else response.text,
                "words": [],
                "segments": [],
            }

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["audio", "language", "timestamps"]
