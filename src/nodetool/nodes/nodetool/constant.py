from datetime import date
import os
from typing import Any

from pydantic import Field

from nodetool.metadata.types import (
    ASRModel,
    AudioRef,
    Datetime,
    DocumentRef,
    EmbeddingModel,
    ImageModel,
    ImageRef,
    ImageSize as ImageSizeType,
    JSONRef,
    LanguageModel,
    Model3DRef,
    TTSModel,
    VideoModel,
    VideoRef,
)
from nodetool.metadata.types import DataframeRef as DataFrameRef
from nodetool.metadata.types import Date as DateType
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class Constant(BaseNode):
    """Base class for fixed-value nodes.

    constant, parameter, default

    Use cases:
    - Provide static inputs to a workflow
    - Hold configuration values
    - Simplify testing with deterministic outputs
    """

    def result_for_client(self, result: dict[str, Any]) -> dict[str, Any]:
        return {}


class Audio(Constant):
    """Represents an audio file constant in the workflow.
    audio, file, mp3, wav

    Use cases:
    - Provide a fixed audio input for audio processing nodes
    - Reference a specific audio file in the workflow
    - Set default audio for testing or demonstration purposes
    """

    _expose_as_tool = True

    value: AudioRef = AudioRef()

    async def process(self, context: ProcessingContext) -> AudioRef:
        await context.refresh_uri(self.value)
        return self.value


class Bool(Constant):
    """Represents a boolean constant in the workflow.
    boolean, logic, flag

    Use cases:
    - Control flow decisions in conditional nodes
    - Toggle features or behaviors in the workflow
    - Set default boolean values for configuration
    """

    _expose_as_tool = True

    value: bool = False

    async def process(self, context: ProcessingContext) -> bool:
        return self.value


class DataFrame(Constant):
    """Represents a fixed DataFrame constant in the workflow.
    table, data, dataframe, pandas

    Use cases:
    - Provide static data for analysis or processing
    - Define lookup tables or reference data
    - Set sample data for testing or demonstration
    """

    _expose_as_tool = True

    value: DataFrameRef = Field(title="DataFrame", default=DataFrameRef())

    async def process(self, context: ProcessingContext) -> DataFrameRef:
        return self.value


class Document(Constant):
    """Represents a document constant in the workflow.
    document, pdf, word, docx
    """

    _expose_as_tool = True

    value: DocumentRef = Field(title="Document", default=DocumentRef())

    async def process(self, context: ProcessingContext) -> DocumentRef:
        return self.value


class Dict(Constant):
    """Represents a dictionary constant in the workflow.
    dictionary, key-value, mapping

    Use cases:
    - Store configuration settings
    - Provide structured data inputs
    - Define parameter sets for other nodes
    """

    _expose_as_tool = True

    value: dict[(str, Any)] = {}

    async def process(self, context: ProcessingContext) -> dict[(str, Any)]:
        return self.value


class Image(Constant):
    """Represents an image file constant in the workflow.
    picture, photo, image

    Use cases:
    - Provide a fixed image input for image processing nodes
    - Reference a specific image file in the workflow
    - Set default image for testing or demonstration purposes
    """

    _expose_as_tool = True

    value: ImageRef = ImageRef()

    async def process(self, context: ProcessingContext) -> ImageRef:
        await context.refresh_uri(self.value)
        return self.value


class ImageSize(Constant):
    """Represents an image dimensions constant in the workflow.
    resolution, width, height, size, preset
    
    Use cases:
    - Set target resolution for image generation
    - Define standard sizes for resizing
    """

    _expose_as_tool = True

    value: ImageSizeType = Field(default_factory=ImageSizeType)

    async def process(self, context: ProcessingContext) -> tuple[ImageSizeType, int, int]:
        return self.value, self.value.width, self.value.height


class ImageList(Constant):
    """Represents a list of image file constants in the workflow.
    pictures, photos, images, collection

    Use cases:
    - Provide a fixed list of images for batch processing
    - Reference multiple image files in the workflow
    - Set default image list for testing or demonstration purposes
    """

    _expose_as_tool = True

    value: list[ImageRef] = Field(
        default_factory=list,
        description="List of image references",
    )

    async def process(self, context: ProcessingContext) -> list[ImageRef]:
        for img in self.value:
            await context.refresh_uri(img)
        return self.value


class VideoList(Constant):
    """Represents a list of video file constants in the workflow.
    videos, movies, clips, collection

    Use cases:
    - Provide a fixed list of videos for batch processing
    - Reference multiple video files in the workflow
    - Set default video list for testing or demonstration purposes
    """

    _expose_as_tool = True

    value: list[VideoRef] = Field(
        default_factory=list,
        description="List of video references",
    )

    async def process(self, context: ProcessingContext) -> list[VideoRef]:
        for video in self.value:
            await context.refresh_uri(video)
        return self.value


class AudioList(Constant):
    """Represents a list of audio file constants in the workflow.
    audios, sounds, audio files, collection

    Use cases:
    - Provide a fixed list of audio files for batch processing
    - Reference multiple audio files in the workflow
    - Set default audio list for testing or demonstration purposes
    """

    _expose_as_tool = True

    value: list[AudioRef] = Field(
        default_factory=list,
        description="List of audio references",
    )

    async def process(self, context: ProcessingContext) -> list[AudioRef]:
        for audio in self.value:
            await context.refresh_uri(audio)
        return self.value


class TextList(Constant):
    """Represents a list of text strings in the workflow.
    texts, strings, text collection

    Use cases:
    - Provide a fixed list of text strings for batch processing
    - Reference multiple text values in the workflow
    - Set default text list for testing or demonstration purposes
    """

    _expose_as_tool = True

    value: list[str] = Field(
        default_factory=list,
        description="List of text strings",
    )

    async def process(self, context: ProcessingContext) -> list[str]:
        return self.value


class Integer(Constant):
    """Represents an integer constant in the workflow.
    number, integer, whole

    Use cases:
    - Set numerical parameters for calculations
    - Define counts, indices, or sizes
    - Provide fixed numerical inputs for processing
    """

    value: int = 0

    async def process(self, context: ProcessingContext) -> int:
        return self.value


class List(Constant):
    """Represents a list constant in the workflow.
    array, sequence, collection

    Use cases:
    - Store multiple values of the same type
    - Provide ordered data inputs
    - Define sequences for iteration in other nodes
    """

    value: list[Any] = []

    async def process(self, context: ProcessingContext) -> list[Any]:
        return self.value


class Float(Constant):
    """Represents a floating-point number constant in the workflow.
    number, decimal, float

    Use cases:
    - Set numerical parameters for calculations
    - Define thresholds or limits
    - Provide fixed numerical inputs for processing
    """

    value: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.value


class String(Constant):
    """Represents a string constant in the workflow.
    text, string, characters

    Use cases:
    - Provide fixed text inputs for processing
    - Define labels, identifiers, or names
    - Set default text values for configuration
    """

    value: str = ""

    async def process(self, context: ProcessingContext) -> str:
        return self.value


class Select(Constant):
    """Represents a selection from a predefined set of options in the workflow.
    select, enum, dropdown, choice, options

    Use cases:
    - Choose from a fixed set of values
    - Configure options for downstream nodes
    - Provide enum-compatible inputs for nodes that expect specific values

    The output is a string that can be connected to enum-typed inputs.
    """

    value: str = Field(
        "",
        description="The currently selected value.",
        json_schema_extra={"type": "select"},
    )
    options: list[str] = Field(
        default=[],
        description="The list of available options to choose from.",
    )
    enum_type_name: str = Field(
        default="",
        description="The enum type name this select corresponds to (for type matching).",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        # Only show 'value' in the UI; hide options and enum_type_name as plumbing
        return ["value"]

    async def process(self, context: ProcessingContext) -> str:
        return self.value


class Video(Constant):
    """Represents a video file constant in the workflow.
    video, movie, mp4, file

    Use cases:
    - Provide a fixed video input for video processing nodes
    - Reference a specific video file in the workflow
    - Set default video for testing or demonstration purposes
    """

    _expose_as_tool = True

    value: VideoRef = VideoRef()

    async def process(self, context: ProcessingContext) -> VideoRef:
        await context.refresh_uri(self.value)
        return self.value


class Model3D(Constant):
    """Represents a 3D model constant in the workflow.
    3d, model, mesh, glb, obj, stl

    Use cases:
    - Provide a fixed 3D model input for processing nodes
    - Reference a specific 3D model file in the workflow
    - Set default 3D model for testing or demonstration purposes
    """

    _expose_as_tool = True

    value: Model3DRef = Model3DRef()

    async def process(self, context: ProcessingContext) -> Model3DRef:
        await context.refresh_uri(self.value)

        if not self.value.format:
            # If still no format and we have an asset ID, check the asset name
            if self.value.asset_id:
                asset = await context.find_asset(self.value.asset_id)
                if asset and asset.name:
                    _, ext = os.path.splitext(asset.name)
                    if ext:
                        self.value.format = ext.lower().lstrip(".")

            # First try to infer from URI (strip query params first)
            if not self.value.format and self.value.uri:
                path = self.value.uri.split("?")[0]
                _, ext = os.path.splitext(path)
                if ext:
                    self.value.format = ext.lower().lstrip(".")

        return self.value


class Date(BaseNode):
    """
    Make a date object from year, month, day.
    date, make, create
    """

    year: int = Field(default=1900, description="Year of the date", ge=1, le=9999)
    month: int = Field(default=1, description="Month of the date", ge=1, le=12)
    day: int = Field(default=1, description="Day of the date", ge=1, le=31)

    async def process(self, context: ProcessingContext) -> DateType:
        return DateType.from_date(date(self.year, self.month, self.day))  # type: ignore


class DateTime(Constant):
    """
    Make a datetime object from year, month, day, hour, minute, second.
    datetime, make, create
    """

    year: int = Field(default=1900, description="Year of the datetime", ge=1, le=9999)
    month: int = Field(default=1, description="Month of the datetime", ge=1, le=12)
    day: int = Field(default=1, description="Day of the datetime", ge=1, le=31)
    hour: int = Field(default=0, description="Hour of the datetime", ge=0, le=23)
    minute: int = Field(default=0, description="Minute of the datetime", ge=0, le=59)
    second: int = Field(default=0, description="Second of the datetime", ge=0, le=59)
    millisecond: int = Field(
        default=0, description="Millisecond of the datetime", ge=0, le=999
    )
    tzinfo: str = Field(default="UTC", description="Timezone of the datetime")
    utc_offset: int = Field(
        default=0,
        description="UTC offset of the datetime in minutes",
        ge=-720,
        le=840,
    )

    _expose_as_tool = True

    async def process(self, context: ProcessingContext) -> Datetime:
        utc_offset_seconds = self.utc_offset * 60
        microseconds = self.millisecond * 1000
        return Datetime(
            year=self.year,
            month=self.month,
            day=self.day,
            hour=self.hour,
            minute=self.minute,
            second=self.second,
            microsecond=microseconds,
            tzinfo=self.tzinfo,
            utc_offset=utc_offset_seconds,
        )


class JSON(Constant):
    """Represents a JSON constant in the workflow.
    json, object, dictionary
    """

    _expose_as_tool = True

    value: JSONRef = JSONRef()

    async def process(self, context: ProcessingContext) -> JSONRef:
        return self.value


class LanguageModelConstant(Constant):
    """Represents a language model constant in the workflow.
    llm, language, model, ai, chat, gpt

    Use cases:
    - Provide a fixed language model for chat or text generation
    - Set default language model for the workflow
    - Configure model selection without user input
    """

    _expose_as_tool = True

    value: LanguageModel = Field(default_factory=lambda: LanguageModel())

    async def process(self, context: ProcessingContext) -> LanguageModel:
        return self.value


class ImageModelConstant(Constant):
    """Represents an image generation model constant in the workflow.
    image, model, ai, generation, diffusion

    Use cases:
    - Provide a fixed image model for generation
    - Set default image model for the workflow
    - Configure model selection without user input
    """

    _expose_as_tool = True

    value: ImageModel = Field(default_factory=lambda: ImageModel())

    async def process(self, context: ProcessingContext) -> ImageModel:
        return self.value


class VideoModelConstant(Constant):
    """Represents a video generation model constant in the workflow.
    video, model, ai, generation

    Use cases:
    - Provide a fixed video model for generation
    - Set default video model for the workflow
    - Configure model selection without user input
    """

    _expose_as_tool = True

    value: VideoModel = Field(default_factory=lambda: VideoModel())

    async def process(self, context: ProcessingContext) -> VideoModel:
        return self.value


class TTSModelConstant(Constant):
    """Represents a text-to-speech model constant in the workflow.
    tts, speech, voice, model, audio

    Use cases:
    - Provide a fixed TTS model for speech synthesis
    - Set default TTS model for the workflow
    - Configure model selection without user input
    """

    _expose_as_tool = True

    value: TTSModel = Field(default_factory=lambda: TTSModel())

    async def process(self, context: ProcessingContext) -> TTSModel:
        return self.value


class ASRModelConstant(Constant):
    """Represents an automatic speech recognition model constant in the workflow.
    asr, speech, recognition, transcription, model

    Use cases:
    - Provide a fixed ASR model for transcription
    - Set default ASR model for the workflow
    - Configure model selection without user input
    """

    _expose_as_tool = True

    value: ASRModel = Field(default_factory=lambda: ASRModel())

    async def process(self, context: ProcessingContext) -> ASRModel:
        return self.value


class EmbeddingModelConstant(Constant):
    """Represents an embedding model constant in the workflow.
    embedding, model, vector, semantic

    Use cases:
    - Provide a fixed embedding model for vectorization
    - Set default embedding model for the workflow
    - Configure model selection without user input
    """

    _expose_as_tool = True

    value: EmbeddingModel = Field(default_factory=lambda: EmbeddingModel())

    async def process(self, context: ProcessingContext) -> EmbeddingModel:
        return self.value
