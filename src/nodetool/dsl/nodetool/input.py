from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class AudioInput(GraphNode):
    """
    Accepts a reference to an audio asset for workflows, specified by an 'AudioRef'.  An 'AudioRef' points to audio data that can be used for playback, transcription, analysis, or processing by audio-capable models.
    input, parameter, audio, sound, voice, speech, asset

    Use cases:
    - Load an audio file for speech-to-text transcription.
    - Analyze sound for specific events or characteristics.
    - Provide audio input to models for tasks like voice recognition or music generation.
    - Process audio for enhancement or feature extraction.
    """

    value: types.AudioRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.AudioRef(type="audio", uri="", asset_id=None, data=None),
        description="The audio to use as input.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the input for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.AudioInput"


class BooleanInput(GraphNode):
    """
    Accepts a boolean (true/false) value as a parameter for workflows.  This input is used for binary choices, enabling or disabling features, or controlling conditional logic paths.
    input, parameter, boolean, bool, toggle, switch, flag

    Use cases:
    - Toggle features or settings on or off.
    - Set binary flags to control workflow behavior.
    - Make conditional choices within a workflow (e.g., proceed if true).
    """

    value: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description=None
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the input for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.BooleanInput"


class ChatInput(GraphNode):
    """
    Accepts a list of chat messages as input for workflows, typically representing a conversation history.  The input is structured as a sequence of 'Message' objects. The node processes this list to extract elements like the latest message content (text, image, audio, video, document), the history, and any associated tool calls.
    input, parameter, chat, message, conversation, prompt, history

    Use cases:
    - Provide user prompts or queries to a language model.
    - Supply conversational context (history) for multi-turn interactions.
    - Capture complex inputs that include text alongside other media types or tool requests.
    - Initiate or continue a chat-based workflow.
    """

    value: list[types.Message] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="The chat message to use as input."
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the input for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.ChatInput"


class CollectionInput(GraphNode):
    """
    Accepts a reference to a specific data collection, typically within a vector database or similar storage system.
    The input is a 'Collection' object, which identifies the target collection for operations like data insertion, querying, or similarity search.
    Keywords: input, parameter, collection, database, vector_store, chroma, index

    Use cases:
    - Select a target vector database collection for indexing new documents.
    - Specify a collection to perform a similarity search or query against.
    - Choose a data source or destination that is represented as a named collection.
    """

    value: types.Collection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Collection(type="collection", name=""),
        description="The collection to use as input.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the input for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.CollectionInput"


class DataframeInput(GraphNode):
    """
    Accepts a pandas DataFrame as input for workflows.
    input, parameter, dataframe, table, structured, csv, tabular_data, rows, columns

    Use cases:
    - Provide a pandas DataFrame as input to a workflow.
    """

    value: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="The dataframe to use as input.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the input for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.DataframeInput"


class DocumentFileInput(GraphNode):
    """
    Accepts a local file path pointing to a document and converts it into a 'DocumentRef'.  This node is a utility for loading a document directly from the local filesystem for use in workflows. It outputs both the 'DocumentRef' for the loaded document and the original 'FilePath'.  Note: This input type is generally not available in production environments due to filesystem access restrictions.
    input, parameter, document, file, path, local_file, load

    Use cases:
    - Directly load a document (e.g., PDF, TXT, DOCX) from a specified local file path.
    - Convert a local file path into a 'DocumentRef' that can be consumed by other document-processing nodes.
    - Useful for development or workflows that have legitimate access to the local filesystem.
    - To provide an existing 'DocumentRef', use 'DocumentInput'.
    """

    value: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="The path to the document file.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the input for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.DocumentFileInput"


class DocumentInput(GraphNode):
    """
    Accepts a reference to a document asset for workflows, specified by a 'DocumentRef'.  A 'DocumentRef' points to a structured document (e.g., PDF, DOCX, TXT) which can be processed or analyzed. This node is used when the workflow needs to operate on a document as a whole entity, potentially including its structure and metadata, rather than just raw text.
    input, parameter, document, file, asset, reference

    Use cases:
    - Load a specific document (e.g., PDF, Word, text file) for content extraction or analysis.
    - Pass a document to models that are designed to process specific document formats.
    - Manage documents as distinct assets within a workflow.
    - If you have a local file path and need to convert it to a 'DocumentRef', consider using 'DocumentFileInput'.
    """

    value: types.DocumentRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DocumentRef(type="document", uri="", asset_id=None, data=None),
        description="The document to use as input.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the input for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.DocumentInput"


class FloatInput(GraphNode):
    """
    Accepts a floating-point number as a parameter for workflows, typically constrained by a minimum and maximum value.  This input allows for precise numeric settings, such as adjustments, scores, or any value requiring decimal precision.
    input, parameter, float, number, decimal, range

    Use cases:
    - Specify a numeric value within a defined range (e.g., 0.0 to 1.0).
    - Set thresholds, confidence scores, or scaling factors.
    - Configure continuous parameters like opacity, volume, or temperature.
    """

    value: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description=None
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the input for the workflow."
    )
    min: float | GraphNode | tuple[GraphNode, str] = Field(default=0, description=None)
    max: float | GraphNode | tuple[GraphNode, str] = Field(
        default=100, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.FloatInput"


class GroupInput(GraphNode):
    """A flexible input that can forward any value provided at runtime.

    This node exists mainly for compatibility with older workflows that expect a
    "group" input placeholder whose value is supplied programmatically (e.g. in
    tests).  Internally it just returns the value assigned to the private
    ``_value`` attribute that tests manipulate directly.
    """

    value: Any | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="The value of the input."
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the input for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.GroupInput"


class ImageInput(GraphNode):
    """
    Accepts a reference to an image asset for workflows, specified by an 'ImageRef'.  An 'ImageRef' points to image data that can be used for display, analysis, or processing by vision models.
    input, parameter, image, picture, graphic, visual, asset

    Use cases:
    - Load an image for visual processing or analysis.
    - Provide an image as input to computer vision models (e.g., object detection, image classification).
    - Select an image for manipulation, enhancement, or inclusion in a document.
    - Display an image within a workflow interface.
    """

    value: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The image to use as input.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the input for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.ImageInput"


class IntegerInput(GraphNode):
    """
    Accepts an integer (whole number) as a parameter for workflows, typically constrained by a minimum and maximum value.  This input is used for discrete numeric values like counts, indices, or iteration limits.
    input, parameter, integer, number, count, index, whole_number

    Use cases:
    - Specify counts or quantities (e.g., number of items, iterations).
    - Set index values for accessing elements in a list or array.
    - Configure discrete numeric parameters like age, steps, or quantity.
    """

    value: int | GraphNode | tuple[GraphNode, str] = Field(default=0, description=None)
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the input for the workflow."
    )
    min: int | GraphNode | tuple[GraphNode, str] = Field(default=0, description=None)
    max: int | GraphNode | tuple[GraphNode, str] = Field(default=100, description=None)

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.IntegerInput"


class ListInput(GraphNode):
    """
    Accepts a list of items as input for workflows.
    input, parameter, list, array, sequence, collection

    Use cases:
    - Provide a list of items to a workflow.
    """

    value: list[Any] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="The list of items to use as input."
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the input for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.ListInput"


class PathInput(GraphNode):
    """
    Accepts a local filesystem path (to a file or directory) as input for workflows.  This input provides a 'FilePath' object. Its usage is typically restricted to non-production environments due to security considerations around direct filesystem access.
    input, parameter, path, filepath, directory, local_file, filesystem

    Use cases:
    - Provide a local path to a specific file or directory for processing.
    - Specify an input or output location on the local filesystem for a development task.
    - Load local datasets or configuration files not managed as assets.
    - Not available in production: raises an error if used in a production environment.
    """

    value: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="The path to use as input.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the input for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.PathInput"


class StringInput(GraphNode):
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

    value: str | GraphNode | tuple[GraphNode, str] = Field(default="", description=None)
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the input for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.StringInput"


class TextInput(GraphNode):
    """Accepts a single line of text (``TextRef``) as a parameter for workflows.
    input, parameter, text, string, line, reference

    This node is a convenience wrapper around ``StringInput`` when the text value
    should be treated as a standalone asset reference (``TextRef``) rather than
    a raw ``str``.
    """

    value: types.TextRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.TextRef(type="text", uri="", asset_id=None, data=None),
        description="The text asset to use as input.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the input for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.TextInput"


class VideoInput(GraphNode):
    """
    Accepts a reference to a video asset for workflows, specified by a 'VideoRef'.  A 'VideoRef' points to video data that can be used for playback, analysis, frame extraction, or processing by video-capable models.
    input, parameter, video, movie, clip, visual, asset

    Use cases:
    - Load a video file for processing or content analysis.
    - Analyze video content for events, objects, or speech.
    - Extract frames or audio tracks from a video.
    - Provide video input to models that understand video data.
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
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the input for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.input.VideoInput"
