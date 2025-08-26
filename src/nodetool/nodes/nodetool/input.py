from typing import Any
import inspect
from pydantic import Field
from nodetool.metadata.types import (
    DataframeRef,
    DocumentRef,
    FilePath,
    Message,
    MessageAudioContent,
    MessageDocumentContent,
    MessageVideoContent,
)
from nodetool.metadata.types import (
    MessageImageContent,
    MessageTextContent,
    TextRef,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import AudioRef
from nodetool.metadata.types import ImageRef
from nodetool.workflows.base_node import InputNode
from nodetool.metadata.types import VideoRef
from nodetool.metadata.types import Collection, ToolName
from nodetool.common.environment import Environment


class FloatInput(InputNode):
    """
    Accepts a floating-point number as a parameter for workflows, typically constrained by a minimum and maximum value.  This input allows for precise numeric settings, such as adjustments, scores, or any value requiring decimal precision.
    input, parameter, float, number, decimal, range

    Use cases:
    - Specify a numeric value within a defined range (e.g., 0.0 to 1.0).
    - Set thresholds, confidence scores, or scaling factors.
    - Configure continuous parameters like opacity, volume, or temperature.
    """

    value: float = 0.0
    min: float = 0
    max: float = 100

    async def process(self, context: ProcessingContext) -> float:
        return min(max(self.value, self.min), self.max)


class BooleanInput(InputNode):
    """
    Accepts a boolean (true/false) value as a parameter for workflows.  This input is used for binary choices, enabling or disabling features, or controlling conditional logic paths.
    input, parameter, boolean, bool, toggle, switch, flag

    Use cases:
    - Toggle features or settings on or off.
    - Set binary flags to control workflow behavior.
    - Make conditional choices within a workflow (e.g., proceed if true).
    """

    value: bool = False

    async def process(self, context: ProcessingContext) -> bool:
        return self.value


class IntegerInput(InputNode):
    """
    Accepts an integer (whole number) as a parameter for workflows, typically constrained by a minimum and maximum value.  This input is used for discrete numeric values like counts, indices, or iteration limits.
    input, parameter, integer, number, count, index, whole_number

    Use cases:
    - Specify counts or quantities (e.g., number of items, iterations).
    - Set index values for accessing elements in a list or array.
    - Configure discrete numeric parameters like age, steps, or quantity.
    """

    value: int = 0
    min: int = 0
    max: int = 100

    async def process(self, context: ProcessingContext) -> int:
        return min(max(self.value, self.min), self.max)


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

    async def process(self, context: ProcessingContext) -> str:
        return self.value


class ChatInput(InputNode):
    """
    Accepts a list of chat messages as input for workflows, typically representing a conversation history.  The input is structured as a sequence of 'Message' objects. The node processes this list to extract elements like the latest message content (text, image, audio, video, document), the history, and any associated tool calls.
    input, parameter, chat, message, conversation, prompt, history

    Use cases:
    - Provide user prompts or queries to a language model.
    - Supply conversational context (history) for multi-turn interactions.
    - Capture complex inputs that include text alongside other media types or tool requests.
    - Initiate or continue a chat-based workflow.
    """

    value: list[Message] = Field([], description="The chat message to use as input.")

    @classmethod
    def return_type(cls):
        return {
            "history": list[Message],
            "text": str,
            "image": ImageRef,
            "audio": AudioRef,
            "video": VideoRef,
            "document": DocumentRef,
            "tools": list[ToolName],
        }

    async def process(self, context: ProcessingContext):
        if not self.value:
            raise ValueError("Chat input is empty, use the workflow chat bottom right")

        history = self.value[:-1]

        last_message = self.value[-1] if self.value else None
        text = ""
        image = ImageRef()
        audio = AudioRef()
        video = VideoRef()
        document = DocumentRef()

        if last_message and last_message.content:
            # Check all content items, taking the first instance of each type
            for content in last_message.content:
                if isinstance(content, MessageTextContent):
                    text = content.text
                elif isinstance(content, MessageImageContent):
                    image = content.image
                elif isinstance(content, MessageAudioContent):
                    audio = content.audio
                elif isinstance(content, MessageVideoContent):
                    video = content.video
                elif isinstance(content, MessageDocumentContent):
                    document = content.document

        def tool_name(name: str) -> ToolName:
            return ToolName(name=name)

        return {
            "history": history,
            "text": text,
            "image": image,
            "audio": audio,
            "video": video,
            "document": document,
            "tools": (
                [tool_name(tool) for tool in last_message.tools]
                if last_message and last_message.tools
                else []
            ),
        }


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

    async def process(self, context: ProcessingContext) -> DocumentRef:
        if self.value.is_empty():
            raise ValueError("Document input is empty, please provide a document")
        return self.value


class ImageInput(InputNode):
    """
    Accepts a reference to an image asset for workflows, specified by an 'ImageRef'.  An 'ImageRef' points to image data that can be used for display, analysis, or processing by vision models.
    input, parameter, image, picture, graphic, visual, asset

    Use cases:
    - Load an image for visual processing or analysis.
    - Provide an image as input to computer vision models (e.g., object detection, image classification).
    - Select an image for manipulation, enhancement, or inclusion in a document.
    - Display an image within a workflow interface.
    """

    value: ImageRef = Field(ImageRef(), description="The image to use as input.")

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.value.is_empty():
            raise ValueError("Image input is empty, please upload an image")
        return self.value


class VideoInput(InputNode):
    """
    Accepts a reference to a video asset for workflows, specified by a 'VideoRef'.  A 'VideoRef' points to video data that can be used for playback, analysis, frame extraction, or processing by video-capable models.
    input, parameter, video, movie, clip, visual, asset

    Use cases:
    - Load a video file for processing or content analysis.
    - Analyze video content for events, objects, or speech.
    - Extract frames or audio tracks from a video.
    - Provide video input to models that understand video data.
    """

    value: VideoRef = Field(VideoRef(), description="The video to use as input.")

    async def process(self, context: ProcessingContext) -> VideoRef:
        if self.value.is_empty():
            raise ValueError("Video input is empty, please upload a video")
        return self.value


class AudioInput(InputNode):
    """
    Accepts a reference to an audio asset for workflows, specified by an 'AudioRef'.  An 'AudioRef' points to audio data that can be used for playback, transcription, analysis, or processing by audio-capable models.
    input, parameter, audio, sound, voice, speech, asset

    Use cases:
    - Load an audio file for speech-to-text transcription.
    - Analyze sound for specific events or characteristics.
    - Provide audio input to models for tasks like voice recognition or music generation.
    - Process audio for enhancement or feature extraction.
    """

    value: AudioRef = Field(AudioRef(), description="The audio to use as input.")

    async def process(self, context: ProcessingContext) -> AudioRef:
        if self.value.is_empty():
            raise ValueError("Audio input is empty, please upload an audio file")
        return self.value


class PathInput(InputNode):
    """
    Accepts a local filesystem path (to a file or directory) as input for workflows.  This input provides a 'FilePath' object. Its usage is typically restricted to non-production environments due to security considerations around direct filesystem access.
    input, parameter, path, filepath, directory, local_file, filesystem

    Use cases:
    - Provide a local path to a specific file or directory for processing.
    - Specify an input or output location on the local filesystem for a development task.
    - Load local datasets or configuration files not managed as assets.
    - Not available in production: raises an error if used in a production environment.
    """

    value: FilePath = Field(FilePath(), description="The path to use as input.")

    async def process(self, context: ProcessingContext) -> FilePath:
        if Environment.is_production():
            raise ValueError("Path input is not available in production")
        if self.value.path == "":
            raise ValueError("Path input is empty, please provide a path")
        return self.value


class DocumentFileInput(InputNode):
    """
    Accepts a local file path pointing to a document and converts it into a 'DocumentRef'.  This node is a utility for loading a document directly from the local filesystem for use in workflows. It outputs both the 'DocumentRef' for the loaded document and the original 'FilePath'.  Note: This input type is generally not available in production environments due to filesystem access restrictions.
    input, parameter, document, file, path, local_file, load

    Use cases:
    - Directly load a document (e.g., PDF, TXT, DOCX) from a specified local file path.
    - Convert a local file path into a 'DocumentRef' that can be consumed by other document-processing nodes.
    - Useful for development or workflows that have legitimate access to the local filesystem.
    - To provide an existing 'DocumentRef', use 'DocumentInput'.
    """

    value: FilePath = Field(FilePath(), description="The path to the document file.")

    @classmethod
    def return_type(cls):
        return {
            "document": DocumentRef,
            "path": FilePath,
        }

    async def process(self, context: ProcessingContext):
        if Environment.is_production():
            raise ValueError("Document file input is not available in production")
        if self.value.path == "":
            raise ValueError("Document file input is empty, please provide a document")
        return {
            "document": DocumentRef(uri=f"file://{self.value.path}"),
            "path": self.value,
        }


class DataframeInput(InputNode):
    """
    Accepts a pandas DataFrame as input for workflows.
    input, parameter, dataframe, table, structured, csv, tabular_data, rows, columns

    Use cases:
    - Provide a pandas DataFrame as input to a workflow.
    """

    value: DataframeRef = Field(
        DataframeRef(), description="The dataframe to use as input."
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        return self.value


class ListInput(InputNode):
    """
    Accepts a list of items as input for workflows.
    input, parameter, list, array, sequence, collection

    Use cases:
    - Provide a list of items to a workflow.
    """

    value: list[Any] = Field([], description="The list of items to use as input.")

    async def process(self, context: ProcessingContext) -> list[Any]:
        return self.value


class CollectionInput(InputNode):
    """
    Accepts a reference to a specific data collection, typically within a vector database or similar storage system.
    The input is a 'Collection' object, which identifies the target collection for operations like data insertion, querying, or similarity search.
    Keywords: input, parameter, collection, database, vector_store, chroma, index

    Use cases:
    - Select a target vector database collection for indexing new documents.
    - Specify a collection to perform a similarity search or query against.
    - Choose a data source or destination that is represented as a named collection.
    """

    value: Collection = Field(
        Collection(), description="The collection to use as input."
    )

    async def process(self, context: ProcessingContext) -> Collection:
        if not self.value:
            raise ValueError("Collection input is empty, please select a collection")
        return self.value


class TextInput(InputNode):
    """Accepts a single line of text (``TextRef``) as a parameter for workflows.
    input, parameter, text, string, line, reference

    This node is a convenience wrapper around ``StringInput`` when the text value
    should be treated as a standalone asset reference (``TextRef``) rather than
    a raw ``str``.
    """

    value: TextRef = Field(TextRef(), description="The text asset to use as input.")

    async def process(self, context: ProcessingContext) -> TextRef:  # type: ignore[override]
        if self.value.is_empty():
            raise ValueError("Text input is empty, please provide a text asset")
        return self.value


class GroupInput(InputNode):
    """
    A minimal group input placeholder to satisfy imports in tests.
    """

    _value: Any | None = None

    async def process(self, context: ProcessingContext) -> Any:
        return self._value
