from typing import ClassVar, Tuple
import numpy as np
from pydantic import Field
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from nodetool.metadata.types import to_numpy
from nodetool.metadata.types import NPArray
from pydub import AudioSegment


# Import numpy_to_audio_segment from the centralized location
from nodetool.media.audio.audio_helpers import numpy_to_audio_segment


def pad_arrays(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    If one of the arguments is a scalar, both arguments are returned as is.
    Pads the smaller array with zeros so that both arrays are the same size.
    This is useful for operations like addition and subtraction.
    """
    if a.size == 1 or b.size == 1:
        return (a, b)
    if len(a) != len(b):
        if len(a) > len(b):
            b = np.pad(b, (0, (len(a) - len(b))), "constant")
        else:
            a = np.pad(a, (0, (len(b) - len(a))), "constant")
    return (a, b)


async def convert_output(
    context: ProcessingContext, output: np.ndarray
) -> float | int | NPArray:
    if output.size == 1:
        return output.item()
    else:
        return NPArray.from_numpy(output)


class BinaryOperation(BaseNode):
    _layout: ClassVar[str] = "small"
    a: int | float | NPArray = Field(title="A", default=0.0)
    b: int | float | NPArray = Field(title="B", default=0.0)

    async def process(self, context: ProcessingContext) -> int | float | NPArray:
        a = to_numpy(self.a)
        b = to_numpy(self.b)
        if a.size > 1 and b.size > 1:
            (a, b) = pad_arrays(a, b)
        res = self.operation(a, b)
        return await convert_output(context, res)

    def operation(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
