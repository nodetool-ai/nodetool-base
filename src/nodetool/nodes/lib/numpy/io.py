import numpy as np
from datetime import datetime
from io import BytesIO
from pydantic import Field
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from nodetool.metadata.types import to_numpy
from nodetool.metadata.types import NPArray, FolderRef


class SaveArray(BaseNode):
    """
    Save a numpy array to a file in the specified folder.
    array, save, file, storage

    Use cases:
    - Store processed arrays for later use
    - Save analysis results
    - Create checkpoints in processing pipelines
    """

    values: NPArray = Field(
        NPArray(),
        description="The array to save.",
    )
    folder: FolderRef = Field(
        FolderRef(),
        description="The folder to save the array in.",
    )
    name: str = Field(
        default="%Y-%m-%d_%H-%M-%S.npy",
        description="""
        The name of the asset to save.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
        """,
    )

    def required_inputs(self):
        return ["array"]

    async def process(self, context: ProcessingContext) -> NPArray:
        filename = datetime.now().strftime(self.name)
        array = to_numpy(self.values)
        buffer = BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        asset = await context.create_asset(
            name=filename,
            content_type="application/array",
            content=buffer,
            parent_id=self.folder.asset_id if self.folder.is_set() else None,
        )
        url = await context.get_asset_url(asset.id)
        return NPArray.from_numpy(
            array,
            uri=url,
            asset_id=asset.id,
        )
