from pydantic import Field
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from nodetool.metadata.types import to_numpy
from nodetool.metadata.types import NPArray


class Reshape1D(BaseNode):
    """
    Reshape an array to a 1D shape without changing its data.
    array, reshape, vector, flatten

    Use cases:
    - Flatten multi-dimensional data for certain algorithms
    - Convert images to vector form for machine learning
    - Prepare data for 1D operations
    """

    values: NPArray = Field(default=NPArray(), description="The input array to reshape")
    num_elements: int = Field(default=0, description="The number of elements")

    async def process(self, context: ProcessingContext) -> NPArray:
        arr = to_numpy(self.values)
        return NPArray.from_numpy(arr.reshape(self.num_elements))


class Reshape2D(BaseNode):
    """
    Reshape an array to a new shape without changing its data.
    array, reshape, dimensions, structure

    Use cases:
    - Convert between different dimensional representations
    - Prepare data for specific model architectures
    - Flatten or unflatten arrays
    """

    values: NPArray = Field(default=NPArray(), description="The input array to reshape")
    num_rows: int = Field(default=0, description="The number of rows")
    num_cols: int = Field(default=0, description="The number of columns")

    async def process(self, context: ProcessingContext) -> NPArray:
        arr = to_numpy(self.values)
        return NPArray.from_numpy(arr.reshape(self.num_rows, self.num_cols))


class Reshape3D(BaseNode):
    """
    Reshape an array to a 3D shape without changing its data.
    array, reshape, dimensions, volume

    Use cases:
    - Convert data for 3D visualization
    - Prepare image data with channels
    - Structure data for 3D convolutions
    """

    values: NPArray = Field(default=NPArray(), description="The input array to reshape")
    num_rows: int = Field(default=0, description="The number of rows")
    num_cols: int = Field(default=0, description="The number of columns")
    num_depths: int = Field(default=0, description="The number of depths")

    async def process(self, context: ProcessingContext) -> NPArray:
        arr = to_numpy(self.values)
        return NPArray.from_numpy(
            arr.reshape(self.num_rows, self.num_cols, self.num_depths)
        )


class Reshape4D(BaseNode):
    """
    Reshape an array to a 4D shape without changing its data.
    array, reshape, dimensions, batch

    Use cases:
    - Prepare batch data for neural networks
    - Structure spatiotemporal data
    - Format data for 3D image processing with channels
    """

    values: NPArray = Field(default=NPArray(), description="The input array to reshape")
    num_rows: int = Field(default=0, description="The number of rows")
    num_cols: int = Field(default=0, description="The number of columns")
    num_depths: int = Field(default=0, description="The number of depths")
    num_channels: int = Field(default=0, description="The number of channels")

    async def process(self, context: ProcessingContext) -> NPArray:
        arr = to_numpy(self.values)
        return NPArray.from_numpy(
            arr.reshape(
                self.num_rows, self.num_cols, self.num_depths, self.num_channels
            )
        )
