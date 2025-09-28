from typing import Any
import inspect

from nodetool.metadata.types import DocumentRef
from nodetool.metadata.types import NPArray
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import AudioRef
from nodetool.metadata.types import DataframeRef
from nodetool.metadata.types import ModelRef
from nodetool.metadata.types import ImageRef
from nodetool.workflows.base_node import (
    BaseNode,
    OutputNode,
)
from nodetool.metadata.types import FilePath, FolderPath
from nodetool.metadata.types import TextRef
from nodetool.metadata.types import VideoRef


class ListOutput(OutputNode):
    """
    Output node for a list of arbitrary values.
    list, output, any

    Use cases:
    - Returning multiple results from a workflow
    - Aggregating outputs from multiple nodes
    """

    value: list[Any] = []

    async def process(self, context: ProcessingContext) -> list[Any]:
        return self.value


class IntegerOutput(OutputNode):
    """
    Output node for a single integer value.
    integer, number, count

    Use cases:
    - Returning numeric results (e.g. counts, indices)
    - Passing integer parameters between nodes
    - Displaying numeric metrics
    """

    value: int = 0

    async def process(self, context: ProcessingContext) -> int:
        return self.value


class FloatOutput(OutputNode):
    """
    Output node for a single float value.
    float, decimal, number

    Use cases:
    - Returning decimal results (e.g. percentages, ratios)
    - Passing floating-point parameters between nodes
    - Displaying numeric metrics with decimal precision
    """

    value: float = 0

    async def process(self, context: ProcessingContext) -> float:
        return self.value


class BooleanOutput(OutputNode):
    """
    Output node for a single boolean value.
    boolean, true, false, flag, condition, flow-control, branch, else, switch, toggle

    Use cases:
    - Returning binary results (yes/no, true/false)
    - Controlling conditional logic in workflows
    - Indicating success/failure of operations
    """

    value: bool = False

    async def process(self, context: ProcessingContext) -> bool:
        return self.value


class StringOutput(OutputNode):
    """
    Output node for a string value.
    string, text, output, label, name

    Use cases:
    - Returning short text results or messages.
    - Passing concise string parameters or identifiers between nodes.
    - Displaying brief textual outputs.
    - For multi-line text or structured document content, use appropriate output nodes if available or consider how data is structured.
    """

    value: str = ""

    async def process(self, context: ProcessingContext) -> str:
        return self.value


class ImageOutput(OutputNode):
    """
    Output node for a single image reference ('ImageRef').
    image, picture, visual, asset, reference

    Use cases:
    - Displaying a single processed or generated image.
    - Passing image data (as an 'ImageRef') between workflow nodes.
    - Returning image analysis results encapsulated in an 'ImageRef'.
    """

    value: ImageRef = ImageRef()

    async def process(self, context: ProcessingContext) -> ImageRef:
        return self.value


class VideoOutput(OutputNode):
    """
    Output node for video content references ('VideoRef').
    video, media, clip, asset, reference

    Use cases:
    - Displaying processed or generated video content.
    - Passing video data (as a 'VideoRef') between workflow steps.
    - Returning results of video analysis encapsulated in a 'VideoRef'.
    """

    value: VideoRef = VideoRef()

    async def process(self, context: ProcessingContext) -> VideoRef:
        return self.value


class ArrayOutput(OutputNode):
    """
    Output node for generic array data, typically numerical ('NPArray').
    array, numerical, list, tensor, vector, matrix

    Use cases:
    - Outputting results from machine learning models (e.g., embeddings, predictions).
    - Representing complex numerical data structures.
    - Passing arrays of numbers between processing steps.
    """

    value: NPArray = NPArray()

    async def process(self, context: ProcessingContext) -> NPArray:
        return self.value


class AudioOutput(OutputNode):
    """
    Output node for audio content references ('AudioRef').
    audio, sound, media, voice, speech, asset, reference

    Use cases:
    - Displaying or returning processed or generated audio.
    - Passing audio data (as an 'AudioRef') between workflow nodes.
    - Returning results of audio analysis (e.g., transcription reference, audio features).
    """

    value: AudioRef = AudioRef()

    async def process(self, context: ProcessingContext) -> AudioRef:
        return self.value


class DataframeOutput(OutputNode):
    """
    Output node for structured data references, typically tabular ('DataframeRef').
    dataframe, table, structured, csv, tabular_data, rows, columns

    Use cases:
    - Outputting tabular data results from analysis or queries.
    - Passing structured datasets between processing or analysis steps.
    - Displaying data in a table format or making it available for download.
    """

    value: DataframeRef = DataframeRef()

    async def process(self, context: ProcessingContext) -> DataframeRef:
        return self.value


class DictionaryOutput(OutputNode):
    """
    Output node for key-value pair data (dictionary).
    dictionary, key-value, mapping, object, json_object, struct

    Use cases:
    - Returning multiple named values as a single structured output.
    - Passing complex data structures or configurations between nodes.
    - Organizing heterogeneous output data into a named map.
    """

    value: dict[str, Any] = {}

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        return self.value


class DocumentOutput(OutputNode):
    """
    Output node for document content references ('DocumentRef').
    document, file, pdf, text_file, asset, reference

    Use cases:
    - Displaying or returning processed or generated documents.
    - Passing document data (as a 'DocumentRef') between workflow nodes.
    - Returning results of document analysis or manipulation.
    """

    value: DocumentRef = DocumentRef()

    async def process(self, context: ProcessingContext) -> DocumentRef:
        return self.value


class FilePathOutput(OutputNode):
    """
    Output node for a file path.
    file, path, file_path
    """

    value: FilePath = FilePath()

    async def process(self, context: ProcessingContext) -> FilePath:
        return self.value


class FolderPathOutput(OutputNode):
    """
    Output node for a folder path.
    folder, path, folder_path
    """

    value: FolderPath = FolderPath()

    async def process(self, context: ProcessingContext) -> FolderPath:
        return self.value
