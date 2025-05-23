from typing import Any
from pydantic import Field
from nodetool.metadata.types import (
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
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import AudioRef
from nodetool.metadata.types import ImageRef
from nodetool.metadata.types import TextRef
from nodetool.workflows.base_node import BaseNode, InputNode
from nodetool.metadata.types import VideoRef
from nodetool.metadata.types import Collection, ToolName
from nodetool.common.environment import Environment


class FloatInput(InputNode):
    """
    Float parameter input for workflows.
    input, parameter, float, number

    Use cases:
    - Specify a numeric value within a defined range
    - Set thresholds or scaling factors
    - Configure continuous parameters like opacity or volume
    """

    value: float = 0.0
    min: float = 0
    max: float = 100

    async def process(self, context: ProcessingContext) -> float:
        return min(max(self.value, self.min), self.max)


class BooleanInput(InputNode):
    """
    Boolean parameter input for workflows.
    input, parameter, boolean, bool

    Use cases:
    - Toggle features on/off
    - Set binary flags
    - Control conditional logic
    """

    value: bool = False

    async def process(self, context: ProcessingContext) -> bool:
        return self.value


class IntegerInput(InputNode):
    """
    Integer parameter input for workflows.
    input, parameter, integer, number

    Use cases:
    - Specify counts or quantities
    - Set index values
    - Configure discrete numeric parameters
    """

    value: int = 0
    min: int = 0
    max: int = 100

    async def process(self, context: ProcessingContext) -> int:
        return min(max(self.value, self.min), self.max)


class StringInput(InputNode):
    """
    String parameter input for workflows.
    input, parameter, string, text

    Use cases:
    - Provide text labels or names
    - Enter search queries
    - Specify file paths or URLs
    """

    value: str = ""

    async def process(self, context: ProcessingContext) -> str:
        return self.value


class ChatInput(InputNode):
    """
    Chat message input for workflows.
    input, parameter, chat, message

    Use cases:
    - Accept user prompts or queries
    - Capture conversational input
    - Provide instructions to language models
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


class TextInput(InputNode):
    """
    Text content input for workflows.
    input, parameter, text

    Use cases:
    - Load text documents or articles
    - Process multi-line text content
    - Analyze large text bodies
    """

    value: TextRef = Field(TextRef(), description="The text to use as input.")

    async def process(self, context: ProcessingContext) -> TextRef:
        if self.value.is_empty():
            raise ValueError("Text input is empty, please provide a text")
        return self.value


class DocumentInput(InputNode):
    """
    Document asset input for workflows.
    input, parameter, document

    Use cases:
    - Load documents for processing
    - Analyze document content
    - Provide document input to models
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
    Image asset input for workflows.
    input, parameter, image

    Use cases:
    - Load images for processing or analysis
    - Provide visual input to models
    - Select images for manipulation
    """

    value: ImageRef = Field(ImageRef(), description="The image to use as input.")

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.value.is_empty():
            raise ValueError("Image input is empty, please upload an image")
        return self.value


class VideoInput(InputNode):
    """
    Video asset input for workflows.
    input, parameter, video

    Use cases:
    - Load video files for processing
    - Analyze video content
    - Extract frames or audio from videos
    """

    value: VideoRef = Field(VideoRef(), description="The video to use as input.")

    async def process(self, context: ProcessingContext) -> VideoRef:
        if self.value.is_empty():
            raise ValueError("Video input is empty, please upload a video")
        return self.value


class AudioInput(InputNode):
    """
    Audio asset input for workflows.
    input, parameter, audio

    Use cases:
    - Load audio files for processing
    - Analyze sound or speech content
    - Provide audio input to models
    """

    value: AudioRef = Field(AudioRef(), description="The audio to use as input.")

    async def process(self, context: ProcessingContext) -> AudioRef:
        if self.value.is_empty():
            raise ValueError("Audio input is empty, please upload an audio file")
        return self.value


class PathInput(InputNode):
    """
    Local path input for workflows.
    input, parameter, path

    Use cases:
    - Provide a local path to a file or directory
    - Specify a file or directory for processing
    - Load local data for analysis
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
    Document file input for workflows.
    input, parameter, document, text

    Use cases:
    - Load text documents for processing
    - Analyze document content
    - Extract text for NLP tasks
    - Index documents for search
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


class GroupInput(BaseNode):
    """
    Generic group input for loops.
    input, group, collection, loop

    Use cases:
    - provides input for a loop
    - iterates over a group of items
    """

    _value: Any = None

    async def process(self, context: Any) -> Any:
        return self._value

    @classmethod
    def is_cacheable(cls):
        return False


class EnumInput(InputNode):
    """
    Enumeration parameter input for workflows.
    input, parameter, enum, options, select

    Use cases:
    - Select from predefined options
    - Enforce choice from valid values
    - Configure categorical parameters
    """

    value: str = ""
    options: str = Field("", description="Comma-separated list of valid options")

    async def process(self, context: ProcessingContext) -> str:
        valid_options = [opt.strip() for opt in self.options.split(",") if opt.strip()]
        if not valid_options:
            return self.value
        if self.value not in valid_options:
            raise ValueError(
                f"Invalid option: {self.value}, please select from {valid_options}"
            )
        return self.value


class CollectionInput(InputNode):
    """
    Collection input for workflows.
    input, parameter, collection, chroma

    Use cases:
    - Select a vector database collection
    - Specify target collection for indexing
    - Choose collection for similarity search
    """

    value: Collection = Field(
        Collection(), description="The collection to use as input."
    )

    async def process(self, context: ProcessingContext) -> Collection:
        if not self.value:
            raise ValueError("Collection input is empty, please select a collection")
        return self.value
