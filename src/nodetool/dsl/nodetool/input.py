from pydantic import Field
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class AudioInput(GraphNode):
    """
    Audio asset input for workflows.
    input, parameter, audio

    Use cases:
    - Load audio files for processing
    - Analyze sound or speech content
    - Provide audio input to models
    """

    value: types.AudioRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.AudioRef(type="audio", uri="", asset_id=None, data=None),
        description="The audio to use as input.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.AudioInput"


class BooleanInput(GraphNode):
    """
    Boolean parameter input for workflows.
    input, parameter, boolean, bool

    Use cases:
    - Toggle features on/off
    - Set binary flags
    - Control conditional logic
    """

    value: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description=None
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.BooleanInput"


class ChatInput(GraphNode):
    """
    Chat message input for workflows.
    input, parameter, chat, message

    Use cases:
    - Accept user prompts or queries
    - Capture conversational input
    - Provide instructions to language models
    """

    value: list[types.Message] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="The chat message to use as input."
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.ChatInput"


class CollectionInput(GraphNode):
    """
    Collection input for workflows.
    input, parameter, collection, chroma

    Use cases:
    - Select a vector database collection
    - Specify target collection for indexing
    - Choose collection for similarity search
    """

    value: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to use as input.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.CollectionInput"


class DocumentFileInput(GraphNode):
    """
    Document file input for workflows.
    input, parameter, document, text

    Use cases:
    - Load text documents for processing
    - Analyze document content
    - Extract text for NLP tasks
    - Index documents for search
    """

    value: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="The path to the document file.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.DocumentFileInput"


class DocumentInput(GraphNode):
    """
    Document asset input for workflows.
    input, parameter, document

    Use cases:
    - Load documents for processing
    - Analyze document content
    - Provide document input to models
    """

    value: types.DocumentRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DocumentRef(type="document", uri="", asset_id=None, data=None),
        description="The document to use as input.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.DocumentInput"


class EnumInput(GraphNode):
    """
    Enumeration parameter input for workflows.
    input, parameter, enum, options, select

    Use cases:
    - Select from predefined options
    - Enforce choice from valid values
    - Configure categorical parameters
    """

    value: str | GraphNode | tuple[GraphNode, str] = Field(default="", description=None)
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    options: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Comma-separated list of valid options"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.EnumInput"


class FloatInput(GraphNode):
    """
    Float parameter input for workflows.
    input, parameter, float, number

    Use cases:
    - Specify a numeric value within a defined range
    - Set thresholds or scaling factors
    - Configure continuous parameters like opacity or volume
    """

    value: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    min: float | GraphNode | tuple[GraphNode, str] = Field(default=0, description=None)
    max: float | GraphNode | tuple[GraphNode, str] = Field(
        default=100, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.FloatInput"


class GroupInput(GraphNode):
    """
    Generic group input for loops.
    input, group, collection, loop

    Use cases:
    - provides input for a loop
    - iterates over a group of items
    """

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.GroupInput"


class ImageInput(GraphNode):
    """
    Image asset input for workflows.
    input, parameter, image

    Use cases:
    - Load images for processing or analysis
    - Provide visual input to models
    - Select images for manipulation
    """

    value: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The image to use as input.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.ImageInput"


class IntegerInput(GraphNode):
    """
    Integer parameter input for workflows.
    input, parameter, integer, number

    Use cases:
    - Specify counts or quantities
    - Set index values
    - Configure discrete numeric parameters
    """

    value: int | GraphNode | tuple[GraphNode, str] = Field(default=0, description=None)
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    min: int | GraphNode | tuple[GraphNode, str] = Field(default=0, description=None)
    max: int | GraphNode | tuple[GraphNode, str] = Field(default=100, description=None)

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.IntegerInput"


class PathInput(GraphNode):
    """
    Local path input for workflows.
    input, parameter, path

    Use cases:
    - Provide a local path to a file or directory
    - Specify a file or directory for processing
    - Load local data for analysis
    """

    value: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="The path to use as input.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.PathInput"


class StringInput(GraphNode):
    """
    String parameter input for workflows.
    input, parameter, string, text

    Use cases:
    - Provide text labels or names
    - Enter search queries
    - Specify file paths or URLs
    """

    value: str | GraphNode | tuple[GraphNode, str] = Field(default="", description=None)
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.StringInput"


class TextInput(GraphNode):
    """
    Text content input for workflows.
    input, parameter, text

    Use cases:
    - Load text documents or articles
    - Process multi-line text content
    - Analyze large text bodies
    """

    value: types.TextRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.TextRef(type="text", uri="", asset_id=None, data=None),
        description="The text to use as input.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.TextInput"


class VideoInput(GraphNode):
    """
    Video asset input for workflows.
    input, parameter, video

    Use cases:
    - Load video files for processing
    - Analyze video content
    - Extract frames or audio from videos
    """

    value: types.VideoRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.VideoRef(
            type="video", uri="", asset_id=None, data=None, duration=None, format=None
        ),
        description="The video to use as input.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.VideoInput"
