import numpy as np
import PIL.Image
from typing import Any
from pydantic import Field
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from nodetool.metadata.types import to_numpy
from nodetool.metadata.types import NPArray, AudioRef, ImageRef
from nodetool.media.audio.audio_helpers import numpy_to_audio_segment


class ConvertToArray(BaseNode):
    """
    Convert PIL Image to normalized tensor representation.
    image, tensor, conversion, normalization

    Use cases:
    - Prepare images for machine learning models
    - Convert between image formats for processing
    - Normalize image data for consistent calculations
    """

    image: ImageRef = Field(
        default=ImageRef(),
        description="The input image to convert to a tensor. The image should have either 1 (grayscale), 3 (RGB), or 4 (RGBA) channels.",
    )

    async def process(self, context: ProcessingContext) -> NPArray:
        if self.image.is_empty():
            raise ValueError("The input image is not connected.")

        image = await context.image_to_pil(self.image)
        image_data = np.array(image)
        tensor_data = image_data / 255.0
        if len(tensor_data.shape) == 2:
            tensor_data = tensor_data[:, :, np.newaxis]
        return NPArray.from_numpy(tensor_data)


class ConvertToImage(BaseNode):
    """
    Convert array data to PIL Image format.
    array, image, conversion, denormalization

    Use cases:
    - Visualize array data as images
    - Save processed array results as images
    - Convert model outputs back to viewable format
    """

    values: NPArray = Field(
        default=NPArray(),
        description="The input array to convert to an image. Should have either 1, 3, or 4 channels.",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.values.is_empty():
            raise ValueError("The input array is not connected.")

        array_data = to_numpy(self.values)
        if array_data.ndim not in [2, 3]:
            raise ValueError("The array should have 2 or 3 dimensions (HxW or HxWxC).")
        if (array_data.ndim == 3) and (array_data.shape[2] not in [1, 3, 4]):
            raise ValueError("The array channels should be either 1, 3, or 4.")
        if (array_data.ndim == 3) and (array_data.shape[2] == 1):
            array_data = array_data.reshape(array_data.shape[0], array_data.shape[1])
        array_data = (array_data * 255).astype(np.uint8)
        output_image = PIL.Image.fromarray(array_data)
        return await context.image_from_pil(output_image)


class ConvertToAudio(BaseNode):
    """
    Converts a array object back to an audio file.
    audio, conversion, array

    Use cases:
    - Save processed audio data as a playable file
    - Convert generated or modified audio arrays to audio format
    - Output results of audio processing pipelinesr
    """

    values: NPArray = Field(
        default=NPArray(), description="The array to convert to an audio file."
    )
    sample_rate: int = Field(
        default=44100, ge=0, le=44100, description="The sample rate of the audio file."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        audio = numpy_to_audio_segment(to_numpy(self.values), self.sample_rate)
        return await context.audio_from_segment(audio)


class ArrayToScalar(BaseNode):
    """
    Convert a single-element array to a scalar value.
    array, scalar, conversion, type

    Use cases:
    - Extract final results from array computations
    - Prepare values for non-array operations
    - Simplify output for human-readable results
    """

    values: NPArray = Field(default=NPArray(), description="Array to convert to scalar")

    async def process(self, context: ProcessingContext) -> float | int:
        return to_numpy(self.values).item()


class ScalarToArray(BaseNode):
    """
    Convert a scalar value to a single-element array.
    scalar, array, conversion, type

    Use cases:
    - Prepare scalar inputs for array operations
    - Create constant arrays for computations
    - Initialize array values in workflows
    """

    value: float | int = Field(
        default=0, description="Scalar value to convert to array"
    )

    async def process(self, context: ProcessingContext) -> NPArray:
        return NPArray.from_numpy(np.array([self.value]))


class ListToArray(BaseNode):
    """
    Convert a list of values to a array.
    list, array, conversion, type

    Use cases:
    - Prepare list data for array operations
    - Create arrays from Python data structures
    - Convert sequence data to array format
    """

    values: list[Any] = Field(
        default=[], description="List of values to convert to array"
    )

    async def process(self, context: ProcessingContext) -> NPArray:
        return NPArray.from_numpy(np.array(self.values))


class ArrayToList(BaseNode):
    """
    Convert a array to a nested list structure.
    array, list, conversion, type

    Use cases:
    - Prepare array data for JSON serialization
    - Convert array outputs to Python data structures
    - Interface array data with non-array operations
    """

    values: NPArray = Field(default=NPArray(), description="Array to convert to list")

    async def process(self, context: ProcessingContext) -> list[Any]:
        return to_numpy(self.values).tolist()
