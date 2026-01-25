from typing import TypedDict

from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    ASRModel,
    AudioRef,
    ColorRef,
    DataframeRef,
    DocumentRef,
    EmbeddingModel,
    FolderRef,
    HuggingFaceModel,
    ImageModel,
    ImageRef,
    LanguageModel,
    Message,
    MessageAudioContent,
    MessageImageContent,
    MessageTextContent,
    Model3DRef,
    Provider,
    TTSModel,
    VideoModel,
    VideoRef,
)
from nodetool.workflows.base_node import BaseNode, InputNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

log = get_logger(__name__)


class FloatInput(InputNode):
    """
    Accepts a floating-point number as a parameter for workflows, typically constrained by a minimum and maximum value.  This input allows for precise numeric settings, such as adjustments, scores, or any value requiring decimal precision.
    input, parameter, float, number, decimal, range
    """

    value: float = 0.0
    min: float = 0
    max: float = 100

    @classmethod
    def return_type(cls):
        return float

    async def process(self, context: ProcessingContext) -> float:
        return self.value


class BooleanInput(InputNode):
    """
    Accepts a boolean (true/false) value as a parameter for workflows.  This input is used for binary choices, enabling or disabling features, or controlling conditional logic paths.
    input, parameter, boolean, bool, toggle, switch, flag
    """

    value: bool = False

    @classmethod
    def return_type(cls):
        return bool

    async def process(self, context: ProcessingContext) -> bool:
        return self.value


class IntegerInput(InputNode):
    """
    Accepts an integer (whole number) as a parameter for workflows, typically constrained by a minimum and maximum value.  This input is used for discrete numeric values like counts, indices, or iteration limits.
    input, parameter, integer, number, count, index, whole_number
    """

    value: int = 0
    min: int = 0
    max: int = 100

    @classmethod
    def return_type(cls):
        return int

    async def process(self, context: ProcessingContext) -> int:
        return self.value


class StringInput(InputNode):
    """
    Accepts a string value as a parameter for workflows.
    input, parameter, string, text, label, name, value

    Use cases:
    - Define a name for an entity or process.
    - Specify a label for a component or output.
    - Enter a short keyword or search term.
    - Provide a simple configuration value (e.g., an API key, a model name).
    - If you need to input multi-line text or the content of a file, use 'DocumentFileInput'.
    """

    value: str = ""
    max_length: int = Field(
        0,
        ge=0,
        le=100000,
        title="Max length",
        description="Maximum number of characters allowed. Use 0 for unlimited.",
    )
    line_mode: str = Field(
        "single_line",
        title="Line mode",
        description="Controls whether the UI should render the input as single-line or multiline.",
        json_schema_extra={"type": "enum", "values": ["single_line", "multiline"]},
    )

    @classmethod
    def return_type(cls):
        return str

    async def process(self, context: ProcessingContext) -> str:
        if self.max_length and len(self.value) > self.max_length:
            return self.value[: self.max_length]
        return self.value


class SelectInput(InputNode):
    """
    Accepts a selection from a predefined set of options as a parameter for workflows.
    input, parameter, select, enum, dropdown, choice, options

    Use cases:
    - Let users choose from a fixed set of values in app mode
    - Configure enum-like options for downstream nodes
    - Provide dropdown selection for workflow parameters

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
        basic_fields = super().get_basic_fields()
        return basic_fields + ["value"]

    @classmethod
    def return_type(cls):
        return str

    async def process(self, context: ProcessingContext) -> str:
        return self.value


class StringListInput(InputNode):
    """
    Accepts a list of strings as a parameter for workflows.
    input, parameter, string, text, label, name, value
    """

    value: list[str] = Field(
        default=[], description="The list of strings to use as input."
    )

    @classmethod
    def return_type(cls):
        return list[str]


class FolderPathInput(InputNode):
    """
    Accepts a folder path as a parameter for workflows.
    input, parameter, folder, path, folderpath, local_folder, filesystem
    """

    value: str = Field(
        "",
        description="The folder path to use as input.",
        json_schema_extra={"type": "folder_path"},
    )

    @classmethod
    def return_type(cls):
        return str


class HuggingFaceModelInput(InputNode):
    """
    Accepts a Hugging Face model as a parameter for workflows.
    input, parameter, model, huggingface, hugging_face, model_name
    """

    value: HuggingFaceModel = Field(
        HuggingFaceModel(), description="The Hugging Face model to use as input."
    )

    @classmethod
    def return_type(cls):
        return HuggingFaceModel


class ColorInput(InputNode):
    """
    Accepts a color value as a parameter for workflows.
    input, parameter, color, color_picker, color_input
    """

    value: ColorRef = Field(ColorRef(), description="The color to use as input.")

    @classmethod
    def return_type(cls):
        return ColorRef


class LanguageModelInput(InputNode):
    """
    Accepts a language model as a parameter for workflows.
    input, parameter, model, language, model_name
    """

    value: LanguageModel = Field(
        LanguageModel(), description="The language model to use as input."
    )

    @classmethod
    def return_type(cls):
        return LanguageModel


class ImageModelInput(InputNode):
    """
    Accepts an image generation model as a parameter for workflows.
    input, parameter, model, image, generation
    """

    value: ImageModel = Field(
        ImageModel(), description="The image generation model to use as input."
    )

    @classmethod
    def return_type(cls):
        return ImageModel


class VideoModelInput(InputNode):
    """
    Accepts a video generation model as a parameter for workflows.
    input, parameter, model, video, generation
    """

    value: VideoModel = Field(
        VideoModel(), description="The video generation model to use as input."
    )

    @classmethod
    def return_type(cls):
        return VideoModel


class TTSModelInput(InputNode):
    """
    Accepts a text-to-speech model as a parameter for workflows.
    input, parameter, model, tts, speech, voice
    """

    value: TTSModel = Field(
        TTSModel(), description="The text-to-speech model to use as input."
    )

    @classmethod
    def return_type(cls):
        return TTSModel


class ASRModelInput(InputNode):
    """
    Accepts an automatic speech recognition model as a parameter for workflows.
    input, parameter, model, asr, transcription, speech
    """

    value: ASRModel = Field(
        ASRModel(), description="The speech recognition model to use as input."
    )

    @classmethod
    def return_type(cls):
        return ASRModel


class EmbeddingModelInput(InputNode):
    """
    Accepts an embedding model as a parameter for workflows.
    input, parameter, model, embedding, vector
    """

    value: EmbeddingModel = Field(
        EmbeddingModel(), description="The embedding model to use as input."
    )

    @classmethod
    def return_type(cls):
        return EmbeddingModel


class DataframeInput(InputNode):
    """
    Accepts a reference to a dataframe asset for workflows.
    input, parameter, dataframe, table, data
    """

    value: DataframeRef = Field(
        DataframeRef(), description="The dataframe to use as input."
    )

    @classmethod
    def return_type(cls):
        return DataframeRef


class DocumentInput(InputNode):
    """
    Accepts a reference to a document asset for workflows, specified by a 'DocumentRef'.  A 'DocumentRef' points to a structured document (e.g., PDF, DOCX, TXT) which can be processed or analyzed. This node is used when the workflow needs to operate on a document as a whole entity, potentially including its structure and metadata, rather than just raw text.
    input, parameter, document, file, asset, reference

    Use cases:
    - Load a specific document (e.g., PDF, Word, text file) for content extraction or analysis.
    - Pass a document to models that are designed to process specific document formats.
    - Manage documents as distinct assets within a workflow.
    - If you have a local file path and need to convert it to a 'DocumentRef', consider using 'DocumentFileInput'.
    """

    value: DocumentRef = Field(
        DocumentRef(), description="The document to use as input."
    )

    @classmethod
    def return_type(cls):
        return DocumentRef


class ImageInput(InputNode):
    """
    Accepts a reference to an image asset for workflows, specified by an 'ImageRef'.  An 'ImageRef' points to image data that can be used for display, analysis, or processing by vision models.
    input, parameter, image, picture, graphic, visual, asset
    """

    value: ImageRef = Field(ImageRef(), description="The image to use as input.")

    @classmethod
    def return_type(cls):
        return ImageRef

    async def process(self, context: ProcessingContext) -> ImageRef:
        return self.value


class VideoInput(InputNode):
    """
    Accepts a reference to a video asset for workflows, specified by a 'VideoRef'.  A 'VideoRef' points to video data that can be used for playback, analysis, frame extraction, or processing by video-capable models.
    input, parameter, video, movie, clip, visual, asset
    """

    value: VideoRef = Field(VideoRef(), description="The video to use as input.")

    @classmethod
    def return_type(cls):
        return VideoRef

    async def process(self, context: ProcessingContext) -> VideoRef:
        return self.value


class AudioInput(InputNode):
    """
    Accepts a reference to an audio asset for workflows, specified by an 'AudioRef'.  An 'AudioRef' points to audio data that can be used for playback, transcription, analysis, or processing by audio-capable models.
    input, parameter, audio, sound, voice, speech, asset
    """

    value: AudioRef = Field(AudioRef(), description="The audio to use as input.")

    @classmethod
    def return_type(cls):
        return AudioRef

    async def process(self, context: ProcessingContext) -> AudioRef:
        return self.value


class Model3DInput(InputNode):
    """
    Accepts a reference to a 3D model asset for workflows, specified by a 'Model3DRef'.
    A 'Model3DRef' points to 3D model data that can be used for visualization, processing,
    or conversion by 3D-capable nodes.
    input, parameter, 3d, model, mesh, obj, glb, stl, ply, asset
    """

    value: Model3DRef = Field(Model3DRef(), description="The 3D model to use as input.")

    @classmethod
    def return_type(cls):
        return Model3DRef

    async def process(self, context: ProcessingContext) -> Model3DRef:
        return self.value


class RealtimeAudioInput(InputNode):
    """
    Accepts streaming audio data for workflows.
    input, parameter, audio, sound, voice, speech, asset
    """

    value: AudioRef = Field(AudioRef(), description="The audio to use as input.")

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        basic_fields = super().get_basic_fields()
        return basic_fields + ["value"]

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    class OutputType(TypedDict):
        chunk: Chunk

    @classmethod
    def return_type(cls):
        return cls.OutputType


class AssetFolderInput(InputNode):
    """
    Accepts an asset folder as a parameter for workflows.
    input, parameter, folder, path, folderpath, local_folder, filesystem
    """

    value: FolderRef = Field(FolderRef(), description="The folder to use as input.")

    @classmethod
    def return_type(cls):
        return FolderRef


class FilePathInput(InputNode):
    """
    Accepts a local filesystem path (to a file or directory) as input for workflows.
    input, parameter, path, filepath, directory, local_file, filesystem
    """

    value: str = Field(
        "",
        description="The path to use as input.",
        json_schema_extra={"type": "file_path"},
    )

    @classmethod
    def return_type(cls):
        return str


class DocumentFileInput(InputNode):
    """
    Accepts a local file path pointing to a document and converts it into a 'DocumentRef'.
    input, parameter, document, file, path, local_file, load
    """

    value: str = Field(
        "",
        description="The path to the document file.",
        json_schema_extra={"type": "file_path"},
    )

    class OutputType(TypedDict):
        document: DocumentRef
        path: str

    @classmethod
    def return_type(cls):
        return cls.OutputType


class MessageInput(InputNode):
    """
    Accepts a chat message object for workflows.
    input, parameter, message, chat, conversation
    """

    value: Message = Field(
        default=Message(),
        description="The message object containing role, content, and metadata.",
    )

    @classmethod
    def return_type(cls):
        return Message

    async def process(self, context: ProcessingContext) -> Message:
        return self.value


class MessageListInput(InputNode):
    """
    Accepts a list of chat message objects for workflows.
    input, parameter, messages, chat, conversation, history
    """

    value: list[Message] = Field(
        default=[],
        description="The list of message objects representing chat history.",
    )

    @classmethod
    def return_type(cls):
        return list[Message]

    async def process(self, context: ProcessingContext) -> list[Message]:
        return self.value


class MessageDeconstructor(BaseNode):
    """
    Deconstructs a chat message object into its individual fields.
    extract, decompose, message, fields, chat

    Use cases:
    - Extract specific fields from a message (e.g., role, content, thread_id).
    - Access message metadata for workflow logic.
    - Process different parts of a message separately.
    """

    value: Message = Field(
        default=Message(),
        description="The message object to deconstruct.",
    )

    class OutputType(TypedDict):
        id: str | None
        thread_id: str | None
        role: str
        text: str
        image: ImageRef | None
        audio: AudioRef | None
        model: LanguageModel | None

    async def process(self, context: ProcessingContext) -> OutputType:
        msg = self.value
        model = None
        image = None
        audio = None
        text = ""
        if msg.content:
            if isinstance(msg.content, str):
                image = None
                audio = None
                text = msg.content
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, MessageTextContent):
                        text = item.text
                    elif isinstance(item, MessageAudioContent):
                        audio = item.audio
                    elif isinstance(item, MessageImageContent):
                        image = item.image

        if msg.provider and msg.model:
            model = LanguageModel(provider=Provider(msg.provider), id=msg.model)

        return {
            "id": msg.id,
            "thread_id": msg.thread_id,
            "role": msg.role,
            "text": text,
            "image": image,
            "audio": audio,
            "model": model,
        }
