from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class ArrayOutput(GraphNode):
    """
    Output node for generic array data, typically numerical ('NPArray').
    array, numerical, list, tensor, vector, matrix

    Use cases:
    - Outputting results from machine learning models (e.g., embeddings, predictions).
    - Representing complex numerical data structures.
    - Passing arrays of numbers between processing steps.
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    value: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description=None,
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the output for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.output.ArrayOutput"


class AudioOutput(GraphNode):
    """
    Output node for audio content references ('AudioRef').
    audio, sound, media, voice, speech, asset, reference

    Use cases:
    - Displaying or returning processed or generated audio.
    - Passing audio data (as an 'AudioRef') between workflow nodes.
    - Returning results of audio analysis (e.g., transcription reference, audio features).
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    value: types.AudioRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.AudioRef(type="audio", uri="", asset_id=None, data=None),
        description=None,
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the output for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.output.AudioOutput"


class BooleanOutput(GraphNode):
    """
    Output node for a single boolean value.
    boolean, true, false, flag, condition, flow-control, branch, else, switch, toggle

    Use cases:
    - Returning binary results (yes/no, true/false)
    - Controlling conditional logic in workflows
    - Indicating success/failure of operations
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    value: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description=None
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the output for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.output.BooleanOutput"


class DataframeOutput(GraphNode):
    """
    Output node for structured data references, typically tabular ('DataframeRef').
    dataframe, table, structured, csv, tabular_data, rows, columns

    Use cases:
    - Outputting tabular data results from analysis or queries.
    - Passing structured datasets between processing or analysis steps.
    - Displaying data in a table format or making it available for download.
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    value: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description=None,
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the output for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.output.DataframeOutput"


class DictionaryOutput(GraphNode):
    """
    Output node for key-value pair data (dictionary).
    dictionary, key-value, mapping, object, json_object, struct

    Use cases:
    - Returning multiple named values as a single structured output.
    - Passing complex data structures or configurations between nodes.
    - Organizing heterogeneous output data into a named map.
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    value: dict[str, Any] | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description=None
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the output for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.output.DictionaryOutput"


class DocumentOutput(GraphNode):
    """
    Output node for document content references ('DocumentRef').
    document, file, pdf, text_file, asset, reference

    Use cases:
    - Displaying or returning processed or generated documents.
    - Passing document data (as a 'DocumentRef') between workflow nodes.
    - Returning results of document analysis or manipulation.
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    value: types.DocumentRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DocumentRef(type="document", uri="", asset_id=None, data=None),
        description=None,
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the output for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.output.DocumentOutput"


class FloatOutput(GraphNode):
    """
    Output node for a single float value.
    float, decimal, number

    Use cases:
    - Returning decimal results (e.g. percentages, ratios)
    - Passing floating-point parameters between nodes
    - Displaying numeric metrics with decimal precision
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    value: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description=None
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the output for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.output.FloatOutput"


class ImageOutput(GraphNode):
    """
    Output node for a single image reference ('ImageRef').
    image, picture, visual, asset, reference

    Use cases:
    - Displaying a single processed or generated image.
    - Passing image data (as an 'ImageRef') between workflow nodes.
    - Returning image analysis results encapsulated in an 'ImageRef'.
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    value: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description=None,
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the output for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.output.ImageOutput"


class IntegerOutput(GraphNode):
    """
    Output node for a single integer value.
    integer, number, count

    Use cases:
    - Returning numeric results (e.g. counts, indices)
    - Passing integer parameters between nodes
    - Displaying numeric metrics
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    value: int | GraphNode | tuple[GraphNode, str] = Field(default=0, description=None)
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the output for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.output.IntegerOutput"


class ListOutput(GraphNode):
    """
    Output node for a list of arbitrary values.
    list, output, any

    Use cases:
    - Returning multiple results from a workflow
    - Aggregating outputs from multiple nodes
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    value: list[Any] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description=None
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the output for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.output.ListOutput"


class StringOutput(GraphNode):
    """
    Output node for a string value.
    string, text, output, label, name

    Use cases:
    - Returning short text results or messages.
    - Passing concise string parameters or identifiers between nodes.
    - Displaying brief textual outputs.
    - For multi-line text or structured document content, use appropriate output nodes if available or consider how data is structured.
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    value: str | GraphNode | tuple[GraphNode, str] = Field(default="", description=None)
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the output for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.output.StringOutput"


class VideoOutput(GraphNode):
    """
    Output node for video content references ('VideoRef').
    video, media, clip, asset, reference

    Use cases:
    - Displaying processed or generated video content.
    - Passing video data (as a 'VideoRef') between workflow steps.
    - Returning results of video analysis encapsulated in a 'VideoRef'.
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The parameter name for the workflow."
    )
    value: types.VideoRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.VideoRef(
            type="video", uri="", asset_id=None, data=None, duration=None, format=None
        ),
        description=None,
    )
    description: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The description of the output for the workflow."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.output.VideoOutput"
