from pydantic import Field
import typing
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.openai.audio


class TextToSpeech(GraphNode):
    """
    Converts text to speech using OpenAI TTS models.
    audio, tts, text-to-speech, voice, synthesis

    Use cases:
    - Generate spoken content for videos or podcasts
    - Create voice-overs for presentations
    - Assist visually impaired users with text reading
    - Produce audio versions of written content
    """

    TtsModel: typing.ClassVar[type] = nodetool.nodes.openai.audio.TextToSpeech.TtsModel
    Voice: typing.ClassVar[type] = nodetool.nodes.openai.audio.TextToSpeech.Voice
    model: nodetool.nodes.openai.audio.TextToSpeech.TtsModel = Field(
        default=nodetool.nodes.openai.audio.TextToSpeech.TtsModel.tts_1,
        description=None,
    )
    voice: nodetool.nodes.openai.audio.TextToSpeech.Voice = Field(
        default=nodetool.nodes.openai.audio.TextToSpeech.Voice.ALLOY, description=None
    )
    input: str | GraphNode | tuple[GraphNode, str] = Field(default="", description=None)
    speed: float | GraphNode | tuple[GraphNode, str] = Field(
        default=1.0, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "openai.audio.TextToSpeech"


class Transcribe(GraphNode):
    """
    Converts speech to text using OpenAI's speech-to-text API.
    audio, transcription, speech-to-text, stt, whisper

    Use cases:
    - Generate accurate transcriptions of audio content
    - Create searchable text from audio recordings
    - Support multiple languages for transcription
    - Enable automated subtitling and captioning
    """

    TranscriptionModel: typing.ClassVar[type] = (
        nodetool.nodes.openai.audio.Transcribe.TranscriptionModel
    )
    Language: typing.ClassVar[type] = nodetool.nodes.openai.audio.Transcribe.Language
    model: nodetool.nodes.openai.audio.Transcribe.TranscriptionModel = Field(
        default=nodetool.nodes.openai.audio.Transcribe.TranscriptionModel.WHISPER_1,
        description="The model to use for transcription.",
    )
    audio: types.AudioRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.AudioRef(type="audio", uri="", asset_id=None, data=None),
        description="The audio file to transcribe (max 25 MB).",
    )
    language: nodetool.nodes.openai.audio.Transcribe.Language = Field(
        default=nodetool.nodes.openai.audio.Transcribe.Language.NONE,
        description="The language of the input audio",
    )
    timestamps: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Whether to return timestamps for the generated text.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Optional text to guide the model's style or continue a previous audio segment.",
    )
    temperature: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0,
        description="The sampling temperature between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.",
    )

    @classmethod
    def get_node_type(cls):
        return "openai.audio.Transcribe"


class Translate(GraphNode):
    """
    Translates speech in audio to English text.
    audio, translation, speech-to-text, localization

    Use cases:
    - Translate foreign language audio content to English
    - Create English transcripts of multilingual recordings
    - Assist non-English speakers in understanding audio content
    - Enable cross-language communication in audio formats
    """

    audio: types.AudioRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.AudioRef(type="audio", uri="", asset_id=None, data=None),
        description="The audio file to translate.",
    )
    temperature: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description="The temperature to use for the translation."
    )

    @classmethod
    def get_node_type(cls):
        return "openai.audio.Translate"
